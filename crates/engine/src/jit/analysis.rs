use crate::{FxHashMap, FxHashSet};

use oxc_ast::ast::{
    ArrowFunctionExpression, Class, Function, IdentifierReference, Program, ThisExpression,
};
use oxc_ast_visit::{Visit, walk};
use oxc_semantic::Scoping;
use oxc_span::Span;
use oxc_syntax::{
    scope::{ScopeFlags, ScopeId},
    symbol::SymbolId,
};

#[derive(Default)]
pub(super) struct CaptureAnalysis {
    pub(super) functions: FxHashMap<u32, FunctionCaptures>,
    pub(super) captured: FxHashSet<SymbolId>,
    pub(super) class_name_write: Option<Span>,
    pub(super) arguments_bindings: FxHashMap<u32, SymbolId>,
    pub(super) arguments_references: FxHashMap<u32, SymbolId>,
}

pub(super) struct FunctionCaptures {
    pub(super) captures: Vec<SymbolId>,
    pub(super) lexical_this: bool,
}

struct ActiveFunction {
    scope: ScopeId,
    start: u32,
    arrow: bool,
}

struct Analyzer<'s> {
    scoping: &'s Scoping,
    active: Vec<ActiveFunction>,
    classes: Vec<SymbolId>,
    result: CaptureAnalysis,
}

impl CaptureAnalysis {
    fn add_capture(&mut self, function: u32, symbol: SymbolId) {
        let captures = &mut self.functions.get_mut(&function).unwrap().captures;
        if !captures.contains(&symbol) {
            captures.push(symbol);
        }
        self.captured.insert(symbol);
    }

    pub(super) fn analyze<'a>(program: &Program<'a>, scoping: &Scoping) -> Self {
        let mut analyzer = Analyzer {
            scoping,
            active: Vec::new(),
            classes: Vec::new(),
            result: Self::default(),
        };
        analyzer.visit_program(program);
        let used: FxHashSet<_> = analyzer
            .result
            .arguments_references
            .values()
            .copied()
            .collect();
        analyzer
            .result
            .arguments_bindings
            .retain(|_, symbol| used.contains(symbol));
        analyzer.result
    }
}

impl Analyzer<'_> {
    fn enter(&mut self, scope: ScopeId, span: Span, arrow: bool) {
        self.result.functions.insert(
            span.start,
            FunctionCaptures {
                captures: Vec::new(),
                lexical_this: false,
            },
        );
        if !arrow {
            let symbol =
                SymbolId::new(self.scoping.symbols_len() + self.result.arguments_bindings.len());
            self.result.arguments_bindings.insert(span.start, symbol);
        }
        self.active.push(ActiveFunction {
            scope,
            start: span.start,
            arrow,
        });
    }
}

impl<'a> Visit<'a> for Analyzer<'_> {
    fn visit_class(&mut self, class: &Class<'a>) {
        let symbol = class.id.as_ref().and_then(|id| id.symbol_id.get());
        if let Some(symbol) = symbol {
            self.classes.push(symbol);
        }
        walk::walk_class(self, class);
        if symbol.is_some() {
            self.classes.pop();
        }
    }

    fn visit_function(&mut self, function: &Function<'a>, flags: ScopeFlags) {
        self.enter(function.scope_id.get().unwrap(), function.span, false);
        walk::walk_function(self, function, flags);
        self.active.pop();
    }

    fn visit_arrow_function_expression(&mut self, function: &ArrowFunctionExpression<'a>) {
        self.enter(function.scope_id.get().unwrap(), function.span, true);
        walk::walk_arrow_function_expression(self, function);
        self.active.pop();
    }

    fn visit_this_expression(&mut self, _: &ThisExpression) {
        for function in self
            .active
            .iter()
            .rev()
            .take_while(|function| function.arrow)
        {
            self.result
                .functions
                .get_mut(&function.start)
                .unwrap()
                .lexical_this = true;
        }
    }

    fn visit_identifier_reference(&mut self, identifier: &IdentifierReference<'a>) {
        let reference = identifier
            .reference_id
            .get()
            .map(|id| self.scoping.get_reference(id));
        let resolved = reference.and_then(|reference| reference.symbol_id());
        if identifier.name == "arguments"
            && let Some(owner_index) = self.active.iter().rposition(|function| !function.arrow)
        {
            let owner = &self.active[owner_index];
            let is_local = resolved.is_some_and(|symbol| {
                let scope = self.scoping.symbol_scope_id(symbol);
                scope == owner.scope || self.scoping.scope_is_descendant_of(scope, owner.scope)
            });
            if !is_local {
                let symbol = self.result.arguments_bindings[&owner.start];
                self.result
                    .arguments_references
                    .insert(identifier.span.start, symbol);
                for function in &self.active[owner_index + 1..] {
                    self.result.add_capture(function.start, symbol);
                }
                return;
            }
        }
        let Some(symbol) = resolved else {
            return;
        };
        if self.classes.contains(&symbol) && reference.is_some_and(|reference| reference.is_write())
        {
            self.result.class_name_write.get_or_insert(identifier.span);
        }
        let binding_scope = self.scoping.symbol_scope_id(symbol);
        if binding_scope == self.scoping.root_scope_id() && !self.classes.contains(&symbol) {
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
            self.result.add_capture(function.start, symbol);
        }
    }
}
