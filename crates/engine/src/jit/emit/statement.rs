use cranelift_codegen::ir::SourceLoc;
use cranelift_codegen::ir::{BlockArg, InstBuilder, types};
use oxc_ast::ast::{
    BindingPattern, ObjectPattern, Statement, VariableDeclaration, VariableDeclarationKind,
};
use oxc_span::GetSpan;

use crate::error::{Error, unsupported_at};
use crate::jit::helpers::Helper;
use crate::value::Value;

use super::FunctionCompiler;

impl FunctionCompiler<'_, '_> {
    pub(super) fn emit_block(&mut self, statements: &[Statement<'_>]) -> Result<(), Error> {
        self.enter_block(statements);
        self.declare_block_functions(statements);
        let result = self
            .initialize_function_declarations(statements)
            .and_then(|_| self.emit_statements(statements));
        self.leave_block();
        result
    }

    pub(super) fn emit_function_body(&mut self, statements: &[Statement<'_>]) -> Result<(), Error> {
        self.enter_block(statements);
        let result = self
            .initialize_function_declarations(statements)
            .and_then(|_| self.emit_statements(statements));
        self.leave_block();
        result
    }

    pub(super) fn initialize_function_declarations(
        &mut self,
        statements: &[Statement<'_>],
    ) -> Result<(), Error> {
        for statement in statements {
            if let Statement::FunctionDeclaration(function) = statement {
                let symbol = function.id.as_ref().unwrap().symbol_id.get().unwrap();
                let variable = self
                    .local(symbol)
                    .expect("function declaration binding must exist");
                let value = self.make_closure(function.span, false, function.generator)?;
                self.initialize_local(variable, value);
                self.clear_temporaries();
            }
        }
        Ok(())
    }

    pub(super) fn emit_statements(&mut self, statements: &[Statement<'_>]) -> Result<(), Error> {
        for statement in statements {
            if self.terminated {
                break;
            }
            if !matches!(statement, Statement::FunctionDeclaration(_)) {
                self.emit_statement(statement)?;
            }
        }
        Ok(())
    }

    pub(super) fn emit_statement(&mut self, statement: &Statement<'_>) -> Result<(), Error> {
        let previous = std::mem::replace(&mut self.source_span, statement.span());
        self.b.set_srcloc(SourceLoc::new(self.source_span.start));
        let result = self.emit_statement_inner(statement);
        self.b.set_srcloc(SourceLoc::new(previous.start));
        self.source_span = previous;
        result
    }

    fn emit_statement_inner(&mut self, statement: &Statement<'_>) -> Result<(), Error> {
        match statement {
            Statement::ClassDeclaration(class) => self.emit_class_declaration(class),
            Statement::BlockStatement(block) => self.emit_block(&block.body),
            Statement::EmptyStatement(_) | Statement::DebuggerStatement(_) => Ok(()),
            Statement::ExpressionStatement(statement) => {
                if let oxc_ast::ast::Expression::YieldExpression(expression) = &statement.expression
                {
                    let generator = self.generator.ok_or_else(|| {
                        unsupported_at(expression.span, "yield outside generator")
                    })?;
                    let value = match &expression.argument {
                        Some(argument) => self.emit_expression(argument)?,
                        None => self.constant(Value::UNDEFINED),
                    };
                    let (state, resume) = self.suspensions[&expression.span.start];
                    let state = self.b.ins().iconst(types::I64, state as i64);
                    let result =
                        self.call_helper(Helper::GeneratorSuspend, &[generator, value, state]);
                    self.b.ins().return_(&[result]);
                    self.resume_blocks.push(resume);
                    self.switch(resume);
                    for variable in self.scopes.variables() {
                        let slot = self.b.ins().iconst(types::I64, variable.as_u32() as i64);
                        let value = self.call_helper(Helper::LocalGet, &[slot]);
                        self.b.def_var(variable, value);
                    }
                    self.clear_temporaries();
                    return Ok(());
                }
                self.emit_expression(&statement.expression)?;
                self.clear_temporaries();
                Ok(())
            }
            Statement::ThrowStatement(statement) => {
                let value = self.emit_expression(&statement.argument)?;
                self.call_helper(Helper::Throw, &[value]);
                self.check_error();
                let undefined = self.constant(Value::UNDEFINED);
                self.jump(self.return_block, &[BlockArg::Value(undefined)]);
                Ok(())
            }
            Statement::TryStatement(statement) => {
                if statement.finalizer.is_some() {
                    return Err(unsupported_at(statement.span, "finally"));
                }
                let handler = statement
                    .handler
                    .as_ref()
                    .ok_or_else(|| unsupported_at(statement.span, "try without catch"))?;
                if handler.param.as_ref().is_some_and(|param| {
                    !matches!(param.pattern, BindingPattern::BindingIdentifier(_))
                }) {
                    return Err(unsupported_at(handler.span, "destructuring catch binding"));
                }
                let catch = self.b.create_block();
                let done = self.b.create_block();
                self.handlers.push(catch);
                self.emit_block(&statement.block.body)?;
                self.handlers.pop();
                if !self.terminated {
                    self.jump(done, &[]);
                }
                self.switch(catch);
                let dummy = self.constant(Value::UNDEFINED);
                let exception = self.call_helper(Helper::Catch, &[dummy]);
                self.scopes.push();
                if let Some(param) = &handler.param {
                    if let BindingPattern::BindingIdentifier(identifier) = &param.pattern {
                        let variable =
                            self.declare_lexical_binding(identifier.symbol_id.get().unwrap());
                        self.initialize_local(variable, exception);
                    }
                }
                self.clear_temporaries();
                self.emit_block(&handler.body.body)?;
                self.leave_block();
                if !self.terminated {
                    self.jump(done, &[]);
                }
                self.switch(done);
                Ok(())
            }
            Statement::ReturnStatement(statement) => {
                let value = match &statement.argument {
                    Some(expression) => self.emit_expression(expression)?,
                    None => self.constant(Value::UNDEFINED),
                };
                self.jump(self.return_block, &[BlockArg::Value(value)]);
                Ok(())
            }
            Statement::VariableDeclaration(declaration) => {
                self.emit_variable_declaration(declaration)
            }
            Statement::IfStatement(statement) => {
                let region = self.prepare_bit_loop(statement);
                let test = self.emit_expression(&statement.test)?;
                let probe = region.as_ref().and_then(|_| self.probe_bit_loop(test));
                let condition = self.truthy(test);
                self.clear_temporaries();
                let consequent = self.b.create_block();
                let alternate = self.b.create_block();
                let done = self.b.create_block();
                self.brif(condition, consequent, &[], alternate, &[]);

                self.switch(consequent);
                if let (Some(region), Some(probe)) = (&region, probe) {
                    self.record_bit_loop(region);
                    let generic = self.b.create_block();
                    self.emit_bit_loop(region, test, probe, generic, done);
                    self.switch(generic);
                }
                self.emit_statement(&statement.consequent)?;
                if !self.terminated {
                    self.jump(done, &[]);
                }

                self.switch(alternate);
                if let Some(statement) = &statement.alternate {
                    self.emit_statement(statement)?;
                }
                if !self.terminated {
                    self.jump(done, &[]);
                }

                self.switch(done);
                Ok(())
            }
            Statement::SwitchStatement(statement) => self.emit_switch(statement),
            Statement::WhileStatement(statement) => self.emit_while(statement),
            Statement::DoWhileStatement(statement) => self.emit_do_while(statement),
            Statement::ForStatement(statement) => self.emit_for(statement),
            Statement::ForOfStatement(statement) => self.emit_for_of(statement),
            Statement::ForInStatement(statement) => self.emit_for_in(statement),
            Statement::BreakStatement(statement) => {
                if statement.label.is_some() {
                    return Err(unsupported_at(statement.span, "labeled break"));
                }
                let target = self
                    .control_targets
                    .last()
                    .ok_or_else(|| unsupported_at(statement.span, "break outside loop"))?
                    .0;
                self.clear_temporaries();
                self.jump(target, &[]);
                Ok(())
            }
            Statement::ContinueStatement(statement) => {
                if statement.label.is_some() {
                    return Err(unsupported_at(statement.span, "labeled continue"));
                }
                let target = self
                    .control_targets
                    .iter()
                    .rev()
                    .find_map(|(_, target)| *target)
                    .ok_or_else(|| unsupported_at(statement.span, "continue outside loop"))?;
                self.clear_temporaries();
                self.jump(target, &[]);
                Ok(())
            }
            Statement::FunctionDeclaration(function) => Err(unsupported_at(
                function.span,
                "function declaration outside a block",
            )),
            other => Err(unsupported_at(
                other.span(),
                format!("{} statement", statement_kind(other)),
            )),
        }
    }

    pub(super) fn emit_array_binding(
        &mut self,
        pattern: &oxc_ast::ast::ArrayPattern<'_>,
        iterable: cranelift_codegen::ir::Value,
        kind: VariableDeclarationKind,
    ) -> Result<(), Error> {
        let state = self.call_helper(Helper::IterateNew, &[iterable]);
        self.check_error();
        for element in &pattern.elements {
            let value = self.call_helper(Helper::IterateNext, &[state]);
            self.check_error();
            if let Some(BindingPattern::BindingIdentifier(identifier)) = element {
                let exhausted = self.b.ins().icmp_imm(
                    cranelift_codegen::ir::condcodes::IntCC::Equal,
                    value,
                    Value::UNINITIALIZED.bits() as i64,
                );
                let undefined = self.constant(Value::UNDEFINED);
                let value = self.b.ins().select(exhausted, undefined, value);
                let symbol = identifier.symbol_id.get().unwrap();
                let variable = if kind == VariableDeclarationKind::Var {
                    self.resolve_function_binding(symbol)
                } else {
                    self.resolve_lexical_binding(symbol)
                }
                .expect("destructuring binding must be declared");
                self.initialize_local(variable, value);
            }
        }
        Ok(())
    }

    pub(super) fn validate_object_binding(pattern: &ObjectPattern<'_>) -> Result<(), Error> {
        for property in &pattern.properties {
            if property.computed {
                return Err(unsupported_at(
                    property.key.span(),
                    "computed object destructuring key",
                ));
            }
            let name = property
                .key
                .static_name()
                .ok_or_else(|| unsupported_at(property.key.span(), "object destructuring key"))?;
            if name.as_str().is_none() {
                return Err(unsupported_at(
                    property.key.span(),
                    "lone surrogate in object destructuring key",
                ));
            }
            if !matches!(property.value, BindingPattern::BindingIdentifier(_)) {
                return Err(unsupported_at(
                    property.value.span(),
                    "object destructuring pattern",
                ));
            }
        }
        if let Some(rest) = &pattern.rest {
            return Err(unsupported_at(rest.span, "object destructuring rest"));
        }
        Ok(())
    }

    pub(super) fn emit_object_binding(
        &mut self,
        pattern: &ObjectPattern<'_>,
        object: cranelift_codegen::ir::Value,
        kind: VariableDeclarationKind,
    ) -> Result<(), Error> {
        Self::validate_object_binding(pattern)?;
        for property in &pattern.properties {
            let name = property.key.static_name().unwrap();
            let name = name.as_str().unwrap();
            let BindingPattern::BindingIdentifier(identifier) = &property.value else {
                unreachable!()
            };
            let value = self.property_access(object, name, None);
            let symbol = identifier.symbol_id.get().unwrap();
            let variable = if kind == VariableDeclarationKind::Var {
                self.resolve_function_binding(symbol)
            } else {
                self.resolve_lexical_binding(symbol)
            }
            .expect("destructuring binding must be declared");
            self.initialize_local(variable, value);
        }
        Ok(())
    }

    pub(super) fn emit_variable_declaration(
        &mut self,
        declaration: &VariableDeclaration<'_>,
    ) -> Result<(), Error> {
        for declarator in &declaration.declarations {
            if let BindingPattern::ObjectPattern(pattern) = &declarator.id {
                let Some(initializer) = &declarator.init else {
                    return Err(unsupported_at(
                        declarator.span,
                        "object destructuring without initializer",
                    ));
                };
                let object = self.emit_expression(initializer)?;
                self.emit_object_binding(pattern, object, declaration.kind)?;
                self.clear_temporaries();
                continue;
            }
            if let BindingPattern::ArrayPattern(pattern) = &declarator.id {
                if pattern.rest.is_some()
                    || pattern
                        .elements
                        .iter()
                        .flatten()
                        .any(|element| !matches!(element, BindingPattern::BindingIdentifier(_)))
                {
                    return Err(unsupported_at(
                        declarator.span,
                        "array destructuring pattern",
                    ));
                }
                let Some(initializer) = &declarator.init else {
                    return Err(unsupported_at(
                        declarator.span,
                        "array destructuring without initializer",
                    ));
                };
                let iterable = self.emit_expression(initializer)?;
                self.emit_array_binding(pattern, iterable, declaration.kind)?;
                self.clear_temporaries();
                continue;
            }
            let BindingPattern::BindingIdentifier(identifier) = &declarator.id else {
                return Err(unsupported_at(declarator.span, "destructuring declaration"));
            };
            let symbol = identifier.symbol_id.get().unwrap();
            let variable = if declaration.kind == VariableDeclarationKind::Var {
                self.resolve_function_binding(symbol)
                    .expect("function binding must be hoisted")
            } else {
                self.resolve_lexical_binding(symbol)
                    .expect("lexical binding must be declared")
            };
            let value = match &declarator.init {
                Some(expression) => self.emit_expression(expression)?,
                None if declaration.kind == VariableDeclarationKind::Var => continue,
                None => self.constant(Value::UNDEFINED),
            };
            self.initialize_local(variable, value);
            self.clear_temporaries();
        }
        Ok(())
    }
}

pub(crate) fn statement_kind(statement: &Statement<'_>) -> &'static str {
    match statement {
        Statement::BlockStatement(_) => "block",
        Statement::BreakStatement(_) => "break",
        Statement::ContinueStatement(_) => "continue",
        Statement::DebuggerStatement(_) => "debugger",
        Statement::DoWhileStatement(_) => "do-while",
        Statement::EmptyStatement(_) => "empty",
        Statement::ExpressionStatement(_) => "expression",
        Statement::ForInStatement(_) => "for-in",
        Statement::ForOfStatement(_) => "for-of",
        Statement::ForStatement(_) => "for",
        Statement::IfStatement(_) => "if",
        Statement::LabeledStatement(_) => "labeled",
        Statement::ReturnStatement(_) => "return",
        Statement::SwitchStatement(_) => "switch",
        Statement::ThrowStatement(_) => "throw",
        Statement::TryStatement(_) => "try",
        Statement::WhileStatement(_) => "while",
        Statement::WithStatement(_) => "with",
        Statement::VariableDeclaration(_) => "variable declaration",
        Statement::FunctionDeclaration(_) => "function declaration",
        Statement::ClassDeclaration(_) => "class declaration",
        _ => "statement",
    }
}
