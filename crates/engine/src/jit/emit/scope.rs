use super::{Block, BlockArg, GraphBuilder as FunctionBuilder, Inst, Variable};
use super::{FunctionCompiler, LoweredValue};
use crate::jit::helpers::Helper;
use crate::jit::semantic::Repr;
use crate::jit::semantic::low_level::{Reference, Scratch};
use crate::value::{TAG_OBJECT, TAG_SHIFT, TAG_STRING, Value};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{MemFlagsData, types};
use oxc_ast::ast::{
    BindingPattern, ForStatementInit, ForStatementLeft, Statement, VariableDeclaration,
    VariableDeclarationKind,
};
use oxc_syntax::symbol::SymbolId;

const _: () = assert!(TAG_STRING - TAG_OBJECT == 1 << TAG_SHIFT);

pub(super) const ADDRESS_MASK: i64 = (1 << TAG_SHIFT) - 1;

pub(super) fn is_heap(b: &mut FunctionBuilder<'_>, value: LoweredValue) -> LoweredValue {
    let value = b.as_word(value);
    let tag = b.ins().ushr_imm_s(value, TAG_SHIFT as i64);
    let first = (TAG_OBJECT >> TAG_SHIFT) as i64;
    let offset = b.ins().iadd_imm_s(tag, -first);
    b.ins().icmp_imm_s(IntCC::UnsignedLessThan, offset, 2)
}

impl FunctionCompiler<'_, '_> {
    // Binding slots are not reused by SSA temporaries, so the variable tracks their contents.
    pub(super) fn initialize_binding(&mut self, variable: Variable, value: LoweredValue) {
        let old = self.b.use_var(variable);
        self.initialize_local_with_old(variable, value, Some(old));
    }

    fn initialize_local_with_old(
        &mut self,
        variable: Variable,
        value: LoweredValue,
        known_old: Option<LoweredValue>,
    ) {
        let (index, stored) = self.store_local(variable, value);
        let old = known_old.unwrap_or(stored);
        let new_heap = self.is_heap(value);
        let old_heap = self.is_heap(old);
        let heap = self.b.ins().bor(new_heap, old_heap);
        let split = self.b.create_block();
        let transfer = self.b.create_block();
        let release = self.b.create_block();
        let done = self.b.create_block();
        self.brif(heap, split, &[], done, &[]);
        self.switch(split);
        self.brif(new_heap, transfer, &[], release, &[]);
        self.switch(transfer);
        let slot = self.b.ins().iconst(types::I64, index as i64);
        self.call_helper(Helper::LocalReplace, &[slot, value, old]);
        self.jump(done, &[]);
        self.switch(release);
        self.release_inline(old, done, false);
        self.switch(done);
        self.b.def_var(variable, value);
    }

    pub(super) fn increment(&mut self, value: LoweredValue) {
        self.b.reference(Reference::Retain, value);
    }

    pub(super) fn checked_temporary(&mut self, value: LoweredValue, retain: bool) -> LoweredValue {
        let word = self.b.as_word(value);
        let missing =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, word, Value::UNINITIALIZED.bits() as i64);
        let error = self.b.create_block();
        let valid = self.b.create_block();
        self.b.set_cold_block(error);
        self.brif(missing, error, &[], valid, &[]);
        self.switch(error);
        self.call_helper(Helper::Retain, &[value]);
        self.check_error();
        self.jump(valid, &[]);
        self.switch(valid);
        if !retain {
            return value;
        }
        let heap = self.is_heap(value);
        let push = self.b.create_block();
        let done = self.b.create_block();
        self.brif(heap, push, &[], done, &[]);
        self.switch(push);
        self.retain_temporary(value);
        self.jump(done, &[]);
        self.switch(done);
        value
    }

    /// Takes a reference to a value the block keeps until its end.
    pub(super) fn retain_temporary(&mut self, value: LoweredValue) -> LoweredValue {
        self.increment(value);
        self.track(value)
    }

    /// Keeps a reference the generated code already holds until the end of the block.
    pub(super) fn own_temporary(&mut self, value: LoweredValue) -> LoweredValue {
        self.track(value)
    }

    pub(super) fn track(&mut self, value: LoweredValue) -> LoweredValue {
        self.b.scratch(Scratch::Own, &[value]);
        value
    }

    pub(super) fn forward_references(&mut self, block: Block, args: &[BlockArg]) {
        if args.is_empty() {
            return;
        }
        let values: Vec<_> = args.iter().map(|&BlockArg::Value(value)| value).collect();
        self.b
            .scratch_owned(Scratch::Forward { target: block }, values);
    }

    pub(super) fn move_result_reference(&mut self, value: LoweredValue) {
        self.b.scratch(Scratch::MoveValue, &[value]);
    }

    pub(super) fn release_owned(&mut self) {
        self.b.scratch(Scratch::End, &[]);
        self.unchecked = None;
    }

    pub(super) fn release_tracked(&mut self, tokens: &[Inst]) {
        for &token in tokens {
            self.b.scratch(Scratch::ReleaseToken { token }, &[]);
        }
        self.unchecked = self.unchecked.filter(|token| !tokens.contains(token));
    }

    pub(super) fn unwind(&mut self, target: Block, args: &[BlockArg]) {
        self.b.scratch(
            Scratch::Unwind {
                exclude: self.unchecked,
            },
            &[],
        );
        self.b.ins().jump(target, args);
        self.terminated = true;
    }

    /// Loads a string literal from its cache in the function's data, creating it on the first
    /// execution. The cache owns the reference, so the value is never tracked.
    pub(super) fn string_constant(&mut self, value: &str) -> LoweredValue {
        let offset = self.data.push(0u64);
        let cache = self.data_address(offset);
        let cached = self
            .b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), cache, 0);
        let create = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param_repr(done, Repr::Boxed);
        self.b.set_cold_block(create);
        let hit = self.b.as_boxed(cached);
        self.brif(cached, done, &[BlockArg::Value(hit)], create, &[]);
        self.switch(create);
        let text: Box<str> = value.into();
        let ptr = self.b.ins().iconst(self.pointer, text.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, text.len() as i64);
        self.names.push(text);
        let created = self.call_helper(Helper::StringConstant, &[cache, ptr, len]);
        self.check_error();
        self.jump(done, &[BlockArg::Value(created)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn load_ctx(&mut self, offset: i32) -> LoweredValue {
        self.b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), self.ctx, offset)
    }

    /// Stores a value whose reference the slot already owns, releasing the value it replaces
    /// when the slot may already hold one.
    pub(super) fn replace_local(
        &mut self,
        variable: Variable,
        value: LoweredValue,
        repeated: bool,
    ) {
        let (_, old) = self.store_local(variable, value);
        if !repeated {
            self.b.def_var(variable, value);
            return;
        }
        let heap = self.is_heap(old);
        let release = self.b.create_block();
        let done = self.b.create_block();
        self.brif(heap, release, &[], done, &[]);
        self.switch(release);
        self.release_inline(old, done, true);
        self.switch(done);
        self.b.def_var(variable, value);
    }

    /// Empties a slot whose reference has moved elsewhere.
    pub(super) fn clear_local(&mut self, variable: Variable) {
        let undefined = self.constant(Value::UNDEFINED);
        self.store_local(variable, undefined);
    }

    fn store_local(&mut self, variable: Variable, value: LoweredValue) -> (u32, LoweredValue) {
        // Even scalar stores conservatively require the normal frame cleanup.
        if let Some(roots) = self.frame_roots {
            let dirty = self.b.ins().iconst(types::I8, 1);
            self.b.def_var(roots, dirty);
        }
        let index = *self.slots.entry(variable.as_u32()).or_insert_with(|| {
            self.slot_count += 1;
            self.slot_count - 1
        });
        let old = self.b.store_frame(self.ctx, value, index);
        (index, old)
    }

    /// Releases a heap-tagged value, calling the runtime only when its count would reach zero.
    pub(super) fn release_inline(&mut self, value: LoweredValue, done: Block, cold_drop: bool) {
        self.b.reference(
            Reference::Release {
                heap: true,
                cold: cold_drop,
            },
            value,
        );
        self.jump(done, &[]);
    }

    pub(super) fn is_heap(&mut self, value: LoweredValue) -> LoweredValue {
        is_heap(self.b, value)
    }
}

pub(in crate::jit) fn collect_var_symbols(
    statements: &[Statement<'_>],
    symbols: &mut Vec<SymbolId>,
) {
    for statement in statements {
        match statement {
            Statement::VariableDeclaration(declaration)
                if declaration.kind == VariableDeclarationKind::Var =>
            {
                collect_declaration_symbols(declaration, symbols);
            }
            Statement::BlockStatement(block) => collect_var_symbols(&block.body, symbols),
            Statement::SwitchStatement(statement) => {
                for case in &statement.cases {
                    collect_var_symbols(&case.consequent, symbols);
                }
            }
            Statement::IfStatement(statement) => {
                collect_var_symbols(std::slice::from_ref(&statement.consequent), symbols);
                if let Some(alternate) = &statement.alternate {
                    collect_var_symbols(std::slice::from_ref(alternate), symbols);
                }
            }
            Statement::DoWhileStatement(statement) => {
                collect_var_symbols(std::slice::from_ref(&statement.body), symbols);
            }
            Statement::WhileStatement(statement) => {
                collect_var_symbols(std::slice::from_ref(&statement.body), symbols);
            }
            Statement::TryStatement(statement) => {
                collect_var_symbols(&statement.block.body, symbols);
                if let Some(handler) = &statement.handler {
                    collect_var_symbols(&handler.body.body, symbols);
                }
            }
            Statement::ForInStatement(statement) => {
                if let ForStatementLeft::VariableDeclaration(declaration) = &statement.left
                    && declaration.kind == VariableDeclarationKind::Var
                {
                    collect_declaration_symbols(declaration, symbols);
                }
                collect_var_symbols(std::slice::from_ref(&statement.body), symbols);
            }
            Statement::ForOfStatement(statement) => {
                if let ForStatementLeft::VariableDeclaration(declaration) = &statement.left
                    && declaration.kind == VariableDeclarationKind::Var
                {
                    collect_declaration_symbols(declaration, symbols);
                }
                collect_var_symbols(std::slice::from_ref(&statement.body), symbols);
            }
            Statement::ForStatement(statement) => {
                if let Some(ForStatementInit::VariableDeclaration(declaration)) = &statement.init
                    && declaration.kind == VariableDeclarationKind::Var
                {
                    collect_declaration_symbols(declaration, symbols);
                }
                collect_var_symbols(std::slice::from_ref(&statement.body), symbols);
            }
            _ => {}
        }
    }
}

fn collect_declaration_symbols(declaration: &VariableDeclaration<'_>, symbols: &mut Vec<SymbolId>) {
    for declarator in &declaration.declarations {
        match &declarator.id {
            BindingPattern::BindingIdentifier(identifier) => {
                symbols.push(identifier.symbol_id.get().unwrap())
            }
            BindingPattern::ArrayPattern(pattern) => {
                for element in pattern.elements.iter().flatten() {
                    if let BindingPattern::BindingIdentifier(identifier) = element {
                        symbols.push(identifier.symbol_id.get().unwrap());
                    }
                }
            }
            BindingPattern::ObjectPattern(pattern) => {
                for property in &pattern.properties {
                    if let BindingPattern::BindingIdentifier(identifier) = &property.value {
                        symbols.push(identifier.symbol_id.get().unwrap());
                    }
                }
            }
            _ => {}
        }
    }
}
