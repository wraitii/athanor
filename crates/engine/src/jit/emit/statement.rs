use cranelift_codegen::ir::{BlockArg, InstBuilder, types};
use oxc_ast::ast::{
    BindingPattern, ForStatement, ForStatementInit, Statement, VariableDeclaration,
    VariableDeclarationKind,
};
use oxc_span::GetSpan;

use crate::error::{Error, unsupported_at};
use crate::value::Value;

use super::FunctionCompiler;

impl FunctionCompiler<'_, '_> {
    pub(super) fn emit_block(&mut self, statements: &[Statement<'_>]) -> Result<(), Error> {
        self.enter_block(statements);
        let result = self.emit_statements(statements);
        self.leave_block();
        result
    }

    pub(super) fn emit_statements(&mut self, statements: &[Statement<'_>]) -> Result<(), Error> {
        for statement in statements {
            if self.terminated {
                break;
            }
            self.emit_statement(statement)?;
        }
        Ok(())
    }

    fn emit_statement(&mut self, statement: &Statement<'_>) -> Result<(), Error> {
        match statement {
            Statement::BlockStatement(block) => self.emit_block(&block.body),
            Statement::EmptyStatement(_) | Statement::DebuggerStatement(_) => Ok(()),
            Statement::ExpressionStatement(statement) => {
                self.emit_expression(&statement.expression)?;
                self.clear_temporaries();
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
                let test = self.emit_expression(&statement.test)?;
                let condition = self.truthy(test);
                self.clear_temporaries();
                let consequent = self.b.create_block();
                let alternate = self.b.create_block();
                let done = self.b.create_block();
                self.brif(condition, consequent, &[], alternate, &[]);

                self.switch(consequent);
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
            Statement::WhileStatement(statement) => {
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
                self.emit_statement(&statement.body)?;
                if !self.terminated {
                    self.jump(header, &[]);
                }

                self.switch(done);
                Ok(())
            }
            Statement::DoWhileStatement(statement) => {
                let body = self.b.create_block();
                let test = self.b.create_block();
                let done = self.b.create_block();
                self.jump(body, &[]);

                self.switch(body);
                self.emit_statement(&statement.body)?;
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
            Statement::ForStatement(statement) => self.emit_for(statement),
            Statement::FunctionDeclaration(function) => {
                Err(unsupported_at(function.span, "nested function declaration"))
            }
            other => Err(unsupported_at(
                other.span(),
                format!("{} statement", statement_kind(other)),
            )),
        }
    }

    fn emit_for(&mut self, statement: &ForStatement<'_>) -> Result<(), Error> {
        let lexical_scope = matches!(
            &statement.init,
            Some(ForStatementInit::VariableDeclaration(declaration))
                if declaration.kind != VariableDeclarationKind::Var
        );
        if lexical_scope {
            if let Some(ForStatementInit::VariableDeclaration(declaration)) = &statement.init {
                for declarator in &declaration.declarations {
                    if let BindingPattern::BindingIdentifier(identifier) = &declarator.id {
                        if self.captured.contains(&identifier.symbol_id.get().unwrap()) {
                            return Err(unsupported_at(declarator.span, "captured loop binding"));
                        }
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
        self.emit_statement(&statement.body)?;
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

    fn emit_variable_declaration(
        &mut self,
        declaration: &VariableDeclaration<'_>,
    ) -> Result<(), Error> {
        for declarator in &declaration.declarations {
            let symbol = match &declarator.id {
                BindingPattern::BindingIdentifier(identifier) => {
                    identifier.symbol_id.get().unwrap()
                }
                _ => {
                    return Err(unsupported_at(declarator.span, "destructuring declaration"));
                }
            };
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
