use std::collections::HashSet;

use oxc_ast::ast::{ArrowFunctionExpression, Expression, Function, Statement, YieldExpression};
use oxc_ast_visit::{Visit, walk};

use crate::error::{Error, unsupported_at};

pub(super) struct ResumePlan {
    pub(super) suspensions: Vec<u32>,
}

impl ResumePlan {
    pub(super) fn lower(statements: &[Statement<'_>]) -> Result<Self, Error> {
        let yields: Vec<_> = statements
            .iter()
            .filter_map(|statement| {
                let Statement::ExpressionStatement(statement) = statement else {
                    return None;
                };
                let Expression::YieldExpression(expression) = &statement.expression else {
                    return None;
                };
                Some(expression.span.start)
            })
            .collect();
        let mut validation = Validate {
            allowed: yields.iter().copied().collect(),
            error: None,
        };
        for statement in statements {
            validation.visit_statement(statement);
        }
        if let Some(error) = validation.error {
            return Err(error);
        }
        Ok(Self {
            suspensions: yields,
        })
    }
}

struct Validate {
    allowed: HashSet<u32>,
    error: Option<Error>,
}

impl<'a> Visit<'a> for Validate {
    fn visit_function(&mut self, _: &Function<'a>, _: oxc_syntax::scope::ScopeFlags) {}
    fn visit_arrow_function_expression(&mut self, _: &ArrowFunctionExpression<'a>) {}
    fn visit_yield_expression(&mut self, expression: &YieldExpression<'a>) {
        if expression.delegate || !self.allowed.contains(&expression.span.start) {
            self.error = Some(unsupported_at(
                expression.span,
                "only top-level yield statements are supported",
            ));
        }
        walk::walk_yield_expression(self, expression);
    }
}
