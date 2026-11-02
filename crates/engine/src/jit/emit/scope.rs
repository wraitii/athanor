use std::collections::HashMap;

use cranelift_codegen::ir::{InstBuilder, Value as ClifValue, types};
use cranelift_frontend::Variable;
use oxc_ast::ast::{
    BindingPattern, ForStatementInit, Function, Statement, VariableDeclaration,
    VariableDeclarationKind,
};
use oxc_syntax::symbol::SymbolId;

use crate::error::Error;
use crate::jit::helpers::Helper;
use crate::value::Value;

use super::FunctionCompiler;

#[derive(Default)]
struct Scope {
    bindings: HashMap<SymbolId, Variable>,
}

pub(super) struct ScopeStack {
    scopes: Vec<Scope>,
}

impl ScopeStack {
    pub(super) fn new() -> Self {
        Self { scopes: Vec::new() }
    }

    pub(super) fn push(&mut self) {
        self.scopes.push(Scope::default());
    }

    pub(super) fn pop(&mut self) -> Vec<Variable> {
        self.scopes.pop().unwrap().bindings.into_values().collect()
    }

    pub(super) fn resolve(&self, symbol: SymbolId) -> Option<Variable> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.bindings.get(&symbol).copied())
    }

    pub(super) fn resolve_current(&self, symbol: SymbolId) -> Option<Variable> {
        self.scopes
            .last()
            .and_then(|scope| scope.bindings.get(&symbol).copied())
    }

    pub(super) fn resolve_function(&self, symbol: SymbolId) -> Option<Variable> {
        self.scopes
            .first()
            .and_then(|scope| scope.bindings.get(&symbol).copied())
    }

    pub(super) fn insert_current(&mut self, symbol: SymbolId, variable: Variable) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.bindings.insert(symbol, variable);
        }
    }

    pub(super) fn insert_function(&mut self, symbol: SymbolId, variable: Variable) {
        if let Some(scope) = self.scopes.first_mut() {
            scope.bindings.insert(symbol, variable);
        }
    }
}

impl FunctionCompiler<'_, '_> {
    pub(super) fn initialize_function(&mut self, function: &Function<'_>) -> Result<(), Error> {
        let body = function.body.as_ref().unwrap();
        self.scopes.push();

        let mut var_symbols = Vec::new();
        collect_var_symbols(&body.statements, &mut var_symbols);
        for symbol in var_symbols {
            self.declare_function_binding(symbol);
        }

        for (index, parameter) in function.params.items.iter().enumerate() {
            let symbol = param_symbol(&parameter.pattern);
            let variable = self.declare_function_binding(symbol);
            let value = self.load_arg(index);
            self.store_local(variable, value);
        }
        if let Some(rest) = &function.params.rest {
            let symbol = param_symbol(&rest.rest.argument);
            let variable = self.declare_function_binding(symbol);
            let start = self
                .b
                .ins()
                .iconst(types::I64, function.params.items.len() as i64);
            let value = self.call_helper(Helper::Rest, &[self.argv, self.argc, start]);
            self.check_error();
            self.store_local(variable, value);
        }
        Ok(())
    }

    pub(super) fn enter_block(&mut self, statements: &[Statement<'_>]) {
        self.scopes.push();
        let mut symbols = Vec::new();
        collect_lexical_symbols(statements, &mut symbols);
        for symbol in symbols {
            self.declare_lexical_binding(symbol);
        }
    }

    pub(super) fn leave_block(&mut self) {
        let variables = self.scopes.pop();
        if self.terminated {
            return;
        }
        for variable in variables {
            let undefined = self.constant(Value::UNDEFINED);
            self.store_local(variable, undefined);
        }
    }

    pub(super) fn read_local(&mut self, variable: Variable) -> ClifValue {
        let value = self.b.use_var(variable);
        self.call_helper(Helper::TempRetain, &[value])
    }

    pub(super) fn store_local(&mut self, variable: Variable, value: ClifValue) {
        let slot = self.b.ins().iconst(types::I64, variable.as_u32() as i64);
        self.call_helper(Helper::LocalSet, &[slot, value]);
        self.b.def_var(variable, value);
    }

    pub(super) fn enter_for_declaration(&mut self, declaration: &VariableDeclaration<'_>) {
        self.scopes.push();
        let mut symbols = Vec::new();
        collect_declaration_symbols(declaration, &mut symbols);
        for symbol in symbols {
            self.declare_lexical_binding(symbol);
        }
    }

    pub(super) fn local(&self, symbol: SymbolId) -> Option<Variable> {
        self.scopes.resolve(symbol)
    }

    pub(super) fn resolve_function_binding(&self, symbol: SymbolId) -> Option<Variable> {
        self.scopes.resolve_function(symbol)
    }

    pub(super) fn resolve_lexical_binding(&self, symbol: SymbolId) -> Option<Variable> {
        self.scopes.resolve_current(symbol)
    }

    fn declare_function_binding(&mut self, symbol: SymbolId) -> Variable {
        if let Some(variable) = self.scopes.resolve_function(symbol) {
            return variable;
        }
        let variable = self.b.declare_var(types::I64);
        self.scopes.insert_function(symbol, variable);
        let undefined = self.constant(Value::UNDEFINED);
        self.b.def_var(variable, undefined);
        variable
    }

    fn declare_lexical_binding(&mut self, symbol: SymbolId) -> Variable {
        if let Some(variable) = self.scopes.resolve_current(symbol) {
            return variable;
        }
        let variable = self.b.declare_var(types::I64);
        self.scopes.insert_current(symbol, variable);
        let uninitialized = self.constant(Value::UNINITIALIZED);
        self.b.def_var(variable, uninitialized);
        variable
    }
}

pub(super) fn param_symbol(pattern: &BindingPattern<'_>) -> SymbolId {
    match pattern {
        BindingPattern::BindingIdentifier(identifier) => identifier.symbol_id.get().unwrap(),
        _ => unreachable!("validated parameter pattern"),
    }
}

pub(super) fn collect_var_symbols(statements: &[Statement<'_>], symbols: &mut Vec<SymbolId>) {
    for statement in statements {
        match statement {
            Statement::VariableDeclaration(declaration)
                if declaration.kind == VariableDeclarationKind::Var =>
            {
                collect_declaration_symbols(declaration, symbols);
            }
            Statement::BlockStatement(block) => collect_var_symbols(&block.body, symbols),
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

fn collect_lexical_symbols(statements: &[Statement<'_>], symbols: &mut Vec<SymbolId>) {
    for statement in statements {
        if let Statement::VariableDeclaration(declaration) = statement
            && declaration.kind != VariableDeclarationKind::Var
        {
            collect_declaration_symbols(declaration, symbols);
        }
    }
}

fn collect_declaration_symbols(declaration: &VariableDeclaration<'_>, symbols: &mut Vec<SymbolId>) {
    for declarator in &declaration.declarations {
        if let BindingPattern::BindingIdentifier(identifier) = &declarator.id {
            symbols.push(identifier.symbol_id.get().unwrap());
        }
    }
}
