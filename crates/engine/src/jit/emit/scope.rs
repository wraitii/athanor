use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{BlockArg, InstBuilder, Value as ClifValue, types};
use cranelift_frontend::Variable;
use oxc_ast::ast::{
    BindingPattern, ForStatementInit, ForStatementLeft, FormalParameters, Statement,
    VariableDeclaration, VariableDeclarationKind,
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
    cells: HashSet<u32>,
}

impl ScopeStack {
    pub(super) fn new() -> Self {
        Self {
            scopes: Vec::new(),
            cells: HashSet::new(),
        }
    }

    pub(super) fn variables(&self) -> Vec<Variable> {
        self.scopes
            .iter()
            .flat_map(|scope| scope.bindings.values().copied())
            .collect()
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
    pub(super) fn initialize_function(
        &mut self,
        params: &FormalParameters<'_>,
        statements: &[Statement<'_>],
    ) -> Result<(), Error> {
        let has_defaults = params
            .items
            .iter()
            .any(|parameter| parameter.initializer.is_some());
        if has_defaults {
            for parameter in &params.items {
                self.declare_lexical_binding(param_symbol(&parameter.pattern));
            }
            if let Some(rest) = &params.rest {
                self.declare_lexical_binding(param_symbol(&rest.rest.argument));
            }
        }

        let mut var_symbols = Vec::new();
        collect_var_symbols(statements, &mut var_symbols);
        for symbol in var_symbols {
            self.declare_function_binding(symbol);
        }

        for statement in statements {
            if let Statement::FunctionDeclaration(function) = statement {
                self.declare_function_binding(
                    function.id.as_ref().unwrap().symbol_id.get().unwrap(),
                );
            }
        }

        for (index, parameter) in params.items.iter().enumerate() {
            let symbol = param_symbol(&parameter.pattern);
            let variable = self.declare_function_binding(symbol);
            let value = self.load_arg(index);
            let value = if let Some(initializer) = &parameter.initializer {
                let missing =
                    self.b
                        .ins()
                        .icmp_imm(IntCC::Equal, value, Value::UNDEFINED.bits() as i64);
                let default = self.b.create_block();
                let provided = self.b.create_block();
                let done = self.b.create_block();
                self.b.append_block_param(done, types::I64);
                self.brif(missing, default, &[], provided, &[]);
                self.switch(default);
                let result = self.emit_expression(initializer)?;
                self.jump(done, &[BlockArg::Value(result)]);
                self.switch(provided);
                self.jump(done, &[BlockArg::Value(value)]);
                self.switch(done);
                self.b.block_params(done)[0]
            } else {
                value
            };
            self.initialize_local(variable, value);
        }
        if let Some(rest) = &params.rest {
            let symbol = param_symbol(&rest.rest.argument);
            let variable = self.declare_function_binding(symbol);
            let start = self.b.ins().iconst(types::I64, params.items.len() as i64);
            let value = self.call_helper(Helper::Rest, &[self.argv, self.argc, start]);
            self.check_error();
            self.initialize_local(variable, value);
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

    pub(super) fn declare_block_functions(&mut self, statements: &[Statement<'_>]) {
        for statement in statements {
            if let Statement::FunctionDeclaration(function) = statement {
                self.declare_lexical_binding(
                    function.id.as_ref().unwrap().symbol_id.get().unwrap(),
                );
            }
        }
    }

    pub(super) fn leave_block(&mut self) {
        let variables = self.scopes.pop();
        if self.terminated {
            return;
        }
        for variable in variables {
            let undefined = self.constant(Value::UNDEFINED);
            if self.scopes.cells.remove(&variable.as_u32()) {
                let slot = self.b.ins().iconst(types::I64, variable.as_u32() as i64);
                self.call_helper(Helper::LocalSet, &[slot, undefined]);
            } else {
                self.store_local(variable, undefined);
            }
        }
    }

    pub(super) fn read_local(&mut self, variable: Variable) -> ClifValue {
        let value = self.b.use_var(variable);
        if self.scopes.cells.contains(&variable.as_u32()) {
            let result = self.call_helper(Helper::CellGet, &[value]);
            self.check_error();
            return result;
        }
        let value = self.call_helper(Helper::TempRetain, &[value]);
        self.check_error();
        value
    }

    pub(super) fn store_local(&mut self, variable: Variable, value: ClifValue) {
        if self.scopes.cells.contains(&variable.as_u32()) {
            let cell = self.b.use_var(variable);
            self.call_helper(Helper::CellGet, &[cell]);
            self.check_error();
        }
        self.initialize_local(variable, value);
    }

    pub(super) fn initialize_local(&mut self, variable: Variable, value: ClifValue) {
        if self.scopes.cells.contains(&variable.as_u32()) {
            let cell = self.b.use_var(variable);
            self.call_helper(Helper::CellSet, &[cell, value]);
            self.check_error();
            return;
        }
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

    pub(super) fn declare_function_binding(&mut self, symbol: SymbolId) -> Variable {
        if let Some(variable) = self.scopes.resolve_function(symbol) {
            return variable;
        }
        let variable = self.b.declare_var(types::I64);
        self.scopes.insert_function(symbol, variable);
        let undefined = self.constant(Value::UNDEFINED);
        self.b.def_var(variable, undefined);
        if self.generator.is_some() {
            let slot = self.b.ins().iconst(types::I64, variable.as_u32() as i64);
            self.call_helper(Helper::LocalSet, &[slot, undefined]);
        }
        self.initialize_cell(symbol, variable, undefined);
        variable
    }

    pub(super) fn declare_lexical_binding(&mut self, symbol: SymbolId) -> Variable {
        if let Some(variable) = self.scopes.resolve_current(symbol) {
            return variable;
        }
        let variable = self.b.declare_var(types::I64);
        self.scopes.insert_current(symbol, variable);
        let uninitialized = self.constant(Value::UNINITIALIZED);
        self.b.def_var(variable, uninitialized);
        if self.generator.is_some() {
            let slot = self.b.ins().iconst(types::I64, variable.as_u32() as i64);
            self.call_helper(Helper::LocalSet, &[slot, uninitialized]);
        }
        self.initialize_cell(symbol, variable, uninitialized);
        variable
    }
    fn initialize_cell(&mut self, symbol: SymbolId, variable: Variable, value: ClifValue) {
        if !self.captured.contains(&symbol) {
            return;
        }
        let cell = self.call_helper(Helper::CellNew, &[value]);
        self.check_error();
        let slot = self.b.ins().iconst(types::I64, variable.as_u32() as i64);
        self.call_helper(Helper::LocalSet, &[slot, cell]);
        self.b.def_var(variable, cell);
        self.scopes.cells.insert(variable.as_u32());
    }

    pub(super) fn install_capture(&mut self, symbol: SymbolId, cell: ClifValue) {
        let variable = self.b.declare_var(types::I64);
        self.scopes.insert_function(symbol, variable);
        let slot = self.b.ins().iconst(types::I64, variable.as_u32() as i64);
        self.call_helper(Helper::LocalSet, &[slot, cell]);
        self.b.def_var(variable, cell);
        self.scopes.cells.insert(variable.as_u32());
    }

    pub(super) fn capture_cell(&mut self, symbol: SymbolId) -> Option<ClifValue> {
        let variable = self.local(symbol)?;
        self.scopes
            .cells
            .contains(&variable.as_u32())
            .then(|| self.b.use_var(variable))
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

fn collect_lexical_symbols(statements: &[Statement<'_>], symbols: &mut Vec<SymbolId>) {
    for statement in statements {
        if let Statement::ClassDeclaration(class) = statement {
            symbols.push(class.id.as_ref().unwrap().symbol_id.get().unwrap());
        }
        if let Statement::VariableDeclaration(declaration) = statement
            && declaration.kind != VariableDeclarationKind::Var
        {
            collect_declaration_symbols(declaration, symbols);
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
            _ => {}
        }
    }
}
