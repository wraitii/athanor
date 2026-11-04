use super::{ClifValue, FunctionCompiler};
use crate::jit::helpers::Helper;
use crate::value::{TAG_OBJECT, TAG_SHIFT, TAG_STRING};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{InstBuilder, MemFlags, types};
use cranelift_frontend::Variable;
use oxc_ast::ast::{
    BindingPattern, ForStatementInit, ForStatementLeft, Statement, VariableDeclaration,
    VariableDeclarationKind,
};
use oxc_syntax::symbol::SymbolId;

const _: () = assert!(TAG_STRING - TAG_OBJECT == 1 << TAG_SHIFT);

impl FunctionCompiler<'_, '_> {
    pub(super) fn initialize_local(&mut self, variable: Variable, value: ClifValue) {
        let index = variable.as_u32();
        self.slots = self.slots.max(index + 1);
        if self.frame_size.is_none() {
            let slot = self.b.ins().iconst(types::I64, index as i64);
            self.call_helper(Helper::LocalSet, &[slot, value]);
            self.b.def_var(variable, value);
            return;
        }
        let flags = MemFlags::trusted();
        let base = self.b.ins().load(
            types::I64,
            flags,
            self.ctx,
            crate::jit::session::LOCALS_OFFSET,
        );
        let offset = (index as i32) * 8;
        let old = self.b.ins().load(types::I64, flags, base, offset);
        self.b.ins().store(flags, value, base, offset);
        let new_heap = self.is_heap(value);
        let old_heap = self.is_heap(old);
        let heap = self.b.ins().bor(new_heap, old_heap);
        let slow = self.b.create_block();
        let done = self.b.create_block();
        self.brif(heap, slow, &[], done, &[]);
        self.switch(slow);
        let slot = self.b.ins().iconst(types::I64, index as i64);
        self.call_helper(Helper::LocalReplace, &[slot, value, old]);
        self.jump(done, &[]);
        self.switch(done);
        self.b.def_var(variable, value);
    }

    fn is_heap(&mut self, value: ClifValue) -> ClifValue {
        let tag = self.b.ins().ushr_imm(value, TAG_SHIFT as i64);
        let first = (TAG_OBJECT >> TAG_SHIFT) as i64;
        let offset = self.b.ins().iadd_imm(tag, -first);
        self.b.ins().icmp_imm(IntCC::UnsignedLessThan, offset, 2)
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
                if let ForStatementLeft::VariableDeclaration(declaration) = &statement.left {
                    if declaration.kind == VariableDeclarationKind::Var {
                        collect_declaration_symbols(declaration, symbols);
                    }
                }
                collect_var_symbols(std::slice::from_ref(&statement.body), symbols);
            }
            Statement::ForOfStatement(statement) => {
                if let ForStatementLeft::VariableDeclaration(declaration) = &statement.left {
                    if declaration.kind == VariableDeclarationKind::Var {
                        collect_declaration_symbols(declaration, symbols);
                    }
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
