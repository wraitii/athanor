use super::{ClifValue, FunctionCompiler};
use crate::heap::REFERENCES_OFFSET;
use crate::jit::helpers::Helper;
use crate::value::{TAG_OBJECT, TAG_SHIFT, TAG_STRING, Value};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{Block, BlockArg, Inst, InstBuilder, MemFlagsData, types};
use cranelift_frontend::{FunctionBuilder, Variable};
use oxc_ast::ast::{
    BindingPattern, ForStatementInit, ForStatementLeft, Statement, VariableDeclaration,
    VariableDeclarationKind,
};
use oxc_syntax::symbol::SymbolId;

const _: () = assert!(TAG_STRING - TAG_OBJECT == 1 << TAG_SHIFT);

pub(super) const ADDRESS_MASK: i64 = (1 << TAG_SHIFT) - 1;

pub(super) fn is_heap(b: &mut FunctionBuilder<'_>, value: ClifValue) -> ClifValue {
    let tag = b.ins().ushr_imm_s(value, TAG_SHIFT as i64);
    let first = (TAG_OBJECT >> TAG_SHIFT) as i64;
    let offset = b.ins().iadd_imm_s(tag, -first);
    b.ins().icmp_imm_s(IntCC::UnsignedLessThan, offset, 2)
}

pub(super) fn heap_address(b: &mut FunctionBuilder<'_>, value: ClifValue) -> ClifValue {
    let heap = is_heap(b, value);
    let address = b.ins().band_imm_s(value, ADDRESS_MASK);
    let zero = b.ins().iconst(types::I64, 0);
    b.ins().select(heap, address, zero)
}

pub(super) fn increment_references(b: &mut FunctionBuilder<'_>, address: ClifValue) -> Inst {
    let flags = MemFlagsData::trusted();
    let references = b.ins().load(types::I32, flags, address, REFERENCES_OFFSET);
    let references = b.ins().iadd_imm_s(references, 1);
    b.ins().store(flags, references, address, REFERENCES_OFFSET)
}

impl FunctionCompiler<'_, '_> {
    // Binding slots are not reused by SSA temporaries, so the variable tracks their contents.
    pub(super) fn initialize_binding(&mut self, variable: Variable, value: ClifValue) {
        let old = self.b.use_var(variable);
        self.initialize_local_with_old(variable, value, Some(old));
    }

    pub(super) fn initialize_local(&mut self, variable: Variable, value: ClifValue) {
        self.initialize_local_with_old(variable, value, None);
    }

    fn initialize_local_with_old(
        &mut self,
        variable: Variable,
        value: ClifValue,
        known_old: Option<ClifValue>,
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

    /// Stores a value in its slot under a reference of the slot's own; a temporary keeps any
    /// reference it holds until its block's clear. Only a slot written more than once per call
    /// can hold an old value to release.
    pub(super) fn own_local(&mut self, variable: Variable, value: ClifValue, repeated: bool) {
        self.increment(value);
        self.replace_local(variable, value, repeated);
    }

    pub(super) fn increment(&mut self, value: ClifValue) {
        let address = heap_address(self.b, value);
        let increment = self.b.create_block();
        let done = self.b.create_block();
        self.brif(address, increment, &[], done, &[]);
        self.switch(increment);
        let store = increment_references(self.b, address);
        self.mark_frame_store(store);
        self.jump(done, &[]);
        self.switch(done);
    }

    pub(super) fn checked_temporary(&mut self, value: ClifValue, retain: bool) -> ClifValue {
        let missing =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, value, Value::UNINITIALIZED.bits() as i64);
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
    pub(super) fn retain_temporary(&mut self, value: ClifValue) -> ClifValue {
        self.increment(value);
        self.track(value)
    }

    /// Keeps a reference the generated code already holds until the end of the block.
    pub(super) fn own_temporary(&mut self, value: ClifValue) -> ClifValue {
        self.track(value)
    }

    pub(super) fn track(&mut self, value: ClifValue) -> ClifValue {
        let variable = match self.free_owned.pop() {
            Some(variable) => variable,
            None => self.b.declare_var(types::I64),
        };
        self.b.def_var(variable, value);
        self.owned.push(variable);
        value
    }

    pub(super) fn release_owned(&mut self) {
        let owned = std::mem::take(&mut self.owned);
        self.release_variables(&owned);
        self.free_owned.extend(owned);
        self.unchecked = None;
    }

    pub(super) fn release_tracked(&mut self, variables: &[Variable]) {
        let unchecked = self
            .unchecked
            .filter(|unchecked| !variables.contains(unchecked));
        self.release_variables(variables);
        self.owned.retain(|owned| !variables.contains(owned));
        self.free_owned.extend_from_slice(variables);
        self.unchecked = unchecked;
    }

    /// References the current block holds in variables at this point, leaving out the result of
    /// a helper that has just failed, which owns nothing.
    pub(super) fn live_temporaries(&self) -> Vec<Variable> {
        self.owned
            .iter()
            .copied()
            .filter(|&variable| Some(variable) != self.unchecked)
            .collect()
    }

    /// Whether leaving the block here has anything to release.
    pub(super) fn unwind_releases(&self) -> bool {
        !self.live_temporaries().is_empty()
    }

    /// Leaves the block early through `target`, releasing the references live here on the way.
    /// Each target has a ladder of cold exits: the exit taking `depth` references releases the
    /// last one and continues to the exit below, so cleanup code grows with the deepest live set
    /// rather than with the number of sites.
    pub(super) fn unwind(&mut self, target: Block, args: &[BlockArg]) {
        let live = self.live_temporaries();
        let mut exit_args = args.to_vec();
        for &variable in &live {
            exit_args.push(BlockArg::Value(self.b.use_var(variable)));
        }
        if !self.unwinds.contains_key(&target) {
            let cleanup = target;
            self.unwinds.insert(target, (Vec::new(), cleanup));
        }
        let (ladder, cleanup) = self.unwinds.get_mut(&target).unwrap();
        let cleanup = *cleanup;
        let built = ladder.len();
        while ladder.len() < live.len() {
            ladder.push(self.b.create_block());
        }
        let ladder = ladder.clone();
        let exit = match live.len() {
            0 => cleanup,
            depth => ladder[depth - 1],
        };
        self.jump(exit, &exit_args);
        for depth in built + 1..=live.len() {
            let next = match depth - 1 {
                0 => cleanup,
                below => ladder[below - 1],
            };
            self.fill_exit(ladder[depth - 1], next, args.len() + depth);
        }
    }

    fn fill_exit(&mut self, exit: Block, next: Block, params: usize) {
        for _ in 0..params {
            self.b.append_block_param(exit, types::I64);
        }
        let created = self.b.func.dfg.num_blocks();
        self.switch(exit);
        let params = self.b.block_params(exit).to_vec();
        let (&released, kept) = params.split_last().unwrap();
        let unchecked = self.unchecked;
        self.release_heap(released, true);
        self.unchecked = unchecked;
        let kept: Vec<BlockArg> = kept.iter().map(|&value| BlockArg::Value(value)).collect();
        self.jump(next, &kept);
        self.b.set_cold_block(exit);
        for index in created..self.b.func.dfg.num_blocks() {
            self.b.set_cold_block(Block::from_u32(index as u32));
        }
    }

    fn release_variables(&mut self, variables: &[Variable]) {
        if variables.is_empty() {
            return;
        }
        let undefined = self.constant(Value::UNDEFINED);
        for &variable in variables.iter().rev() {
            let value = self.b.use_var(variable);
            self.b.def_var(variable, undefined);
            self.release_heap(value, true);
        }
    }

    /// Releases a value that may or may not be heap-tagged.
    fn release_heap(&mut self, value: ClifValue, cold_drop: bool) {
        let heap = self.is_heap(value);
        let release = self.b.create_block();
        let done = self.b.create_block();
        self.brif(heap, release, &[], done, &[]);
        self.switch(release);
        self.release_inline(value, done, cold_drop);
        self.switch(done);
    }

    /// Loads a string literal from its cache in the function's data, creating it on the first
    /// execution. The cache owns the reference, so the value is never tracked.
    pub(super) fn string_constant(&mut self, value: &str) -> ClifValue {
        let offset = self.data.push(0u64);
        let cache = self.data_address(offset);
        let cached = self
            .b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), cache, 0);
        let create = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.b.set_cold_block(create);
        self.brif(cached, done, &[BlockArg::Value(cached)], create, &[]);
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

    pub(super) fn load_ctx(&mut self, offset: i32) -> ClifValue {
        self.b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), self.ctx, offset)
    }

    /// Stores a value whose reference the slot already owns, releasing the value it replaces
    /// when the slot may already hold one.
    pub(super) fn replace_local(&mut self, variable: Variable, value: ClifValue, repeated: bool) {
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

    fn store_local(&mut self, variable: Variable, value: ClifValue) -> (u32, ClifValue) {
        // Even scalar stores conservatively require the normal frame cleanup.
        if let Some(roots) = self.frame_roots {
            let dirty = self.b.ins().iconst(types::I8, 1);
            self.b.def_var(roots, dirty);
        }
        let index = *self.slots.entry(variable.as_u32()).or_insert_with(|| {
            self.slot_count += 1;
            self.slot_count - 1
        });
        let flags = MemFlagsData::trusted();
        let base = self.b.ins().load(
            types::I64,
            flags,
            self.ctx,
            crate::jit::session::LOCALS_OFFSET,
        );
        let offset = (index as i32) * 8;
        let old = self.b.ins().load(types::I64, flags, base, offset);
        self.b.ins().store(flags, value, base, offset);
        (index, old)
    }

    /// Releases a heap-tagged value, calling the runtime only when its count would reach zero.
    pub(super) fn release_inline(&mut self, value: ClifValue, done: Block, cold_drop: bool) {
        let address = self.b.ins().band_imm_s(value, ADDRESS_MASK);
        let load = self.b.create_block();
        let decrement = self.b.create_block();
        let last = self.b.create_block();
        self.brif(address, load, &[], done, &[]);
        self.switch(load);
        let flags = MemFlagsData::trusted();
        let references = self
            .b
            .ins()
            .load(types::I32, flags, address, REFERENCES_OFFSET);
        let shared = self
            .b
            .ins()
            .icmp_imm_s(IntCC::UnsignedGreaterThan, references, 1);
        self.brif(shared, decrement, &[], last, &[]);
        self.switch(decrement);
        let references = self.b.ins().iadd_imm_s(references, -1);
        let store = self
            .b
            .ins()
            .store(flags, references, address, REFERENCES_OFFSET);
        self.mark_frame_store(store);
        self.jump(done, &[]);
        self.switch(last);
        if cold_drop {
            self.b.set_cold_block(last);
        }
        self.call_helper(Helper::Release, &[value]);
        self.jump(done, &[]);
    }

    pub(super) fn is_heap(&mut self, value: ClifValue) -> ClifValue {
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
