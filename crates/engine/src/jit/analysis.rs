use std::collections::{HashMap, HashSet};

use oxc_ast::ast::{ArrowFunctionExpression, Function, IdentifierReference, Program};
use oxc_ast_visit::{Visit, walk};
use oxc_semantic::Scoping;
use oxc_span::Span;
use oxc_syntax::{
    scope::{ScopeFlags, ScopeId},
    symbol::SymbolId,
};

#[derive(Default)]
pub(super) struct CaptureAnalysis {
    pub(super) functions: HashMap<u32, FunctionCaptures>,
    pub(super) captured: HashSet<SymbolId>,
}

pub(super) struct FunctionCaptures {
    pub(super) captures: Vec<SymbolId>,
}

struct ActiveFunction {
    scope: ScopeId,
    start: u32,
}

struct Analyzer<'s> {
    scoping: &'s Scoping,
    active: Vec<ActiveFunction>,
    result: CaptureAnalysis,
}

impl CaptureAnalysis {
    pub(super) fn analyze<'a>(program: &Program<'a>, scoping: &Scoping) -> Self {
        let mut analyzer = Analyzer {
            scoping,
            active: Vec::new(),
            result: Self::default(),
        };
        analyzer.visit_program(program);
        analyzer.result
    }
}

impl Analyzer<'_> {
    fn enter(&mut self, scope: ScopeId, span: Span) {
        self.result.functions.insert(
            span.start,
            FunctionCaptures {
                captures: Vec::new(),
            },
        );
        self.active.push(ActiveFunction {
            scope,
            start: span.start,
        });
    }
}

impl<'a> Visit<'a> for Analyzer<'_> {
    fn visit_function(&mut self, function: &Function<'a>, flags: ScopeFlags) {
        self.enter(function.scope_id.get().unwrap(), function.span);
        walk::walk_function(self, function, flags);
        self.active.pop();
    }

    fn visit_arrow_function_expression(&mut self, function: &ArrowFunctionExpression<'a>) {
        self.enter(function.scope_id.get().unwrap(), function.span);
        walk::walk_arrow_function_expression(self, function);
        self.active.pop();
    }

    fn visit_identifier_reference(&mut self, identifier: &IdentifierReference<'a>) {
        let Some(symbol) = identifier
            .reference_id
            .get()
            .and_then(|id| self.scoping.get_reference(id).symbol_id())
        else {
            return;
        };
        let binding_scope = self.scoping.symbol_scope_id(symbol);
        if binding_scope == self.scoping.root_scope_id() {
            return;
        }
        for function in self.active.iter().rev() {
            if binding_scope == function.scope
                || self
                    .scoping
                    .scope_is_descendant_of(binding_scope, function.scope)
            {
                break;
            }
            let captures = &mut self
                .result
                .functions
                .get_mut(&function.start)
                .unwrap()
                .captures;
            if !captures.contains(&symbol) {
                captures.push(symbol);
            }
            self.result.captured.insert(symbol);
        }
    }
}
