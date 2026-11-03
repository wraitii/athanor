use super::FunctionCompiler;
use crate::error::{Error, unsupported_at};
use crate::jit::helpers::Helper;
use crate::value::Value;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{InstBuilder, types};
use oxc_ast::ast::{
    BindingPattern, DoWhileStatement, ForInStatement, ForOfStatement, ForStatement,
    ForStatementInit, ForStatementLeft, VariableDeclarationKind, WhileStatement,
};
use oxc_span::GetSpan;

impl FunctionCompiler<'_, '_> {
    pub(super) fn emit_while(&mut self, statement: &WhileStatement<'_>) -> Result<(), Error> {
        let header = self.b.create_block();
        let body = self.b.create_block();
        let done = self.b.create_block();
        self.jump(header, &[]);

        self.switch(header);
        let test = self.emit_expression(&statement.test)?;
        let condition = self.truthy(test);
        self.clear_temporaries();
        self.brif(condition, body, &[], done, &[]);

        self.switch(body);
        self.control_targets.push((done, Some(header)));
        self.emit_statement(&statement.body)?;
        self.control_targets.pop();
        if !self.terminated {
            self.jump(header, &[]);
        }

        self.switch(done);
        Ok(())
    }
    pub(super) fn emit_do_while(&mut self, statement: &DoWhileStatement<'_>) -> Result<(), Error> {
        let body = self.b.create_block();
        let test = self.b.create_block();
        let done = self.b.create_block();
        self.jump(body, &[]);

        self.switch(body);
        self.control_targets.push((done, Some(test)));
        self.emit_statement(&statement.body)?;
        self.control_targets.pop();
        if !self.terminated {
            self.jump(test, &[]);
        }

        self.switch(test);
        let value = self.emit_expression(&statement.test)?;
        let condition = self.truthy(value);
        self.clear_temporaries();
        self.brif(condition, body, &[], done, &[]);

        self.switch(done);
        Ok(())
    }
    pub(super) fn emit_for(&mut self, statement: &ForStatement<'_>) -> Result<(), Error> {
        let lexical_scope = matches!(
            &statement.init,
            Some(ForStatementInit::VariableDeclaration(declaration))
                if declaration.kind != VariableDeclarationKind::Var
        );
        if lexical_scope {
            if let Some(ForStatementInit::VariableDeclaration(declaration)) = &statement.init {
                for declarator in &declaration.declarations {
                    let identifiers: Vec<_> = match &declarator.id {
                        BindingPattern::BindingIdentifier(identifier) => vec![identifier.as_ref()],
                        BindingPattern::ArrayPattern(pattern) => pattern
                            .elements
                            .iter()
                            .flatten()
                            .filter_map(|element| {
                                if let BindingPattern::BindingIdentifier(identifier) = element {
                                    Some(identifier.as_ref())
                                } else {
                                    None
                                }
                            })
                            .collect(),
                        _ => Vec::new(),
                    };
                    if identifiers.iter().any(|identifier| {
                        self.captured.contains(&identifier.symbol_id.get().unwrap())
                    }) {
                        return Err(unsupported_at(declarator.span, "captured loop binding"));
                    }
                }
            }
        }
        if let Some(init) = &statement.init {
            match init {
                ForStatementInit::VariableDeclaration(declaration) => {
                    if lexical_scope {
                        self.enter_for_declaration(declaration);
                    }
                    self.emit_variable_declaration(declaration)?;
                }
                other => {
                    let expression = other
                        .as_expression()
                        .ok_or_else(|| unsupported_at(statement.span, "for loop init"))?;
                    self.emit_expression(expression)?;
                    self.clear_temporaries();
                }
            }
        }

        let header = self.b.create_block();
        let body = self.b.create_block();
        let update = self.b.create_block();
        let done = self.b.create_block();
        self.jump(header, &[]);

        self.switch(header);
        let condition = match &statement.test {
            Some(expression) => {
                let value = self.emit_expression(expression)?;
                self.truthy(value)
            }
            None => self.b.ins().iconst(types::I8, 1),
        };
        self.clear_temporaries();
        self.brif(condition, body, &[], done, &[]);

        self.switch(body);
        self.control_targets.push((done, Some(update)));
        self.emit_statement(&statement.body)?;
        self.control_targets.pop();
        if !self.terminated {
            self.jump(update, &[]);
        }

        self.switch(update);
        if let Some(expression) = &statement.update {
            self.emit_expression(expression)?;
            self.clear_temporaries();
        }
        self.jump(header, &[]);

        self.switch(done);
        if lexical_scope {
            self.leave_block();
        }
        Ok(())
    }

    pub(super) fn emit_for_in(&mut self, statement: &ForInStatement<'_>) -> Result<(), Error> {
        let ForStatementLeft::VariableDeclaration(declaration) = &statement.left else {
            return Err(unsupported_at(
                statement.left.span(),
                "for-in assignment target",
            ));
        };
        let [declarator] = declaration.declarations.as_slice() else {
            return Err(unsupported_at(declaration.span, "for-in declaration"));
        };
        let BindingPattern::BindingIdentifier(identifier) = &declarator.id else {
            return Err(unsupported_at(declarator.span, "for-in destructuring"));
        };
        let symbol = identifier.symbol_id.get().unwrap();
        let lexical = declaration.kind != VariableDeclarationKind::Var;
        if lexical {
            self.scopes.push();
            self.declare_lexical_binding(symbol);
        }
        let iterable = self.emit_expression(&statement.right)?;
        let state = self.call_helper(Helper::EnumerateNew, &[iterable]);
        self.check_error();
        let iterator = self.b.declare_var(types::I64);
        self.store_local(iterator, state);
        self.clear_temporaries();
        let header = self.b.create_block();
        let body = self.b.create_block();
        let done = self.b.create_block();
        self.jump(header, &[]);
        self.switch(header);
        let state = self.b.use_var(iterator);
        let value = self.call_helper(Helper::EnumerateNext, &[state]);
        self.check_error();
        let finished =
            self.b
                .ins()
                .icmp_imm(IntCC::Equal, value, Value::UNINITIALIZED.bits() as i64);
        self.brif(finished, done, &[], body, &[]);
        self.switch(body);
        if lexical {
            self.scopes.push();
            let variable = self.declare_lexical_binding(symbol);
            self.initialize_local(variable, value);
        } else {
            let variable = self.resolve_function_binding(symbol).unwrap();
            self.initialize_local(variable, value);
        }
        self.clear_temporaries();
        self.control_targets.push((done, Some(header)));
        self.emit_statement(&statement.body)?;
        self.control_targets.pop();
        if lexical {
            self.leave_block();
        }
        if !self.terminated {
            self.jump(header, &[]);
        }
        self.switch(done);
        self.clear_temporaries();
        let undefined = self.constant(Value::UNDEFINED);
        self.store_local(iterator, undefined);
        if lexical {
            self.leave_block();
        }
        Ok(())
    }
    pub(super) fn emit_for_of(&mut self, statement: &ForOfStatement<'_>) -> Result<(), Error> {
        if statement.r#await {
            return Err(unsupported_at(statement.span, "for await"));
        }
        let ForStatementLeft::VariableDeclaration(declaration) = &statement.left else {
            return Err(unsupported_at(
                statement.left.span(),
                "for-of assignment target",
            ));
        };
        let [declarator] = declaration.declarations.as_slice() else {
            return Err(unsupported_at(declaration.span, "for-of declaration"));
        };
        let identifiers = match &declarator.id {
            BindingPattern::BindingIdentifier(identifier) => vec![identifier.as_ref()],
            BindingPattern::ArrayPattern(pattern) if pattern.rest.is_none() => {
                let mut identifiers = Vec::new();
                for element in pattern.elements.iter().flatten() {
                    let BindingPattern::BindingIdentifier(identifier) = element else {
                        return Err(unsupported_at(
                            declarator.span,
                            "for-of destructuring pattern",
                        ));
                    };
                    identifiers.push(identifier.as_ref());
                }
                identifiers
            }
            _ => {
                return Err(unsupported_at(
                    declarator.span,
                    "for-of destructuring pattern",
                ));
            }
        };
        let lexical = declaration.kind != VariableDeclarationKind::Var;
        if lexical {
            self.scopes.push();
            for identifier in &identifiers {
                self.declare_lexical_binding(identifier.symbol_id.get().unwrap());
            }
        }
        let iterable = self.emit_expression(&statement.right)?;
        let state = self.call_helper(Helper::IterateNew, &[iterable]);
        self.check_error();
        let iterator = self.b.declare_var(types::I64);
        self.store_local(iterator, state);
        self.clear_temporaries();
        let header = self.b.create_block();
        let body = self.b.create_block();
        let done = self.b.create_block();
        self.jump(header, &[]);
        self.switch(header);
        let state = self.b.use_var(iterator);
        let value = self.call_helper(Helper::IterateNext, &[state]);
        self.check_error();
        let finished =
            self.b
                .ins()
                .icmp_imm(IntCC::Equal, value, Value::UNINITIALIZED.bits() as i64);
        self.brif(finished, done, &[], body, &[]);
        self.switch(body);
        if lexical {
            self.scopes.push();
            for identifier in &identifiers {
                self.declare_lexical_binding(identifier.symbol_id.get().unwrap());
            }
        }
        match &declarator.id {
            BindingPattern::BindingIdentifier(identifier) => {
                let symbol = identifier.symbol_id.get().unwrap();
                let variable = if lexical {
                    self.resolve_lexical_binding(symbol)
                } else {
                    self.resolve_function_binding(symbol)
                }
                .unwrap();
                self.initialize_local(variable, value);
            }
            BindingPattern::ArrayPattern(pattern) => {
                self.emit_array_binding(pattern, value, declaration.kind)?;
            }
            _ => unreachable!(),
        }
        self.clear_temporaries();
        self.control_targets.push((done, Some(header)));
        self.emit_statement(&statement.body)?;
        self.control_targets.pop();
        if lexical {
            self.leave_block();
        }
        if !self.terminated {
            self.jump(header, &[]);
        }
        self.switch(done);
        self.clear_temporaries();
        let undefined = self.constant(Value::UNDEFINED);
        self.store_local(iterator, undefined);
        if lexical {
            self.leave_block();
        }
        Ok(())
    }
}
