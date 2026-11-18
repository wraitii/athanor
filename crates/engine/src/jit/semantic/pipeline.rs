use super::runtime::{
    arrays as array_checks, cells, helpers, numeric, properties, storage, typed, values, views,
};
use super::specialization::{
    call_variants, calls, observations, properties as property_accesses, reads, regions, shapes,
    writes,
};
use crate::{FxHashMap, FxHashSet};
use cranelift_codegen::ir::Type;
use cranelift_codegen::isa::CallConv;
use js::{arrays, captures, comparisons, construction, regexps, string_split};
use oxc_ast::ast::{Expression, FormalParameters, Statement};
use oxc_semantic::Scoping;
use oxc_span::Span;
use oxc_syntax::symbol::SymbolId;
use transforms::{conversions, reuse};

use super::{
    FunctionState, Graph, Inst, RegionFallback, Result, SpecializationPlan, Value, borrow, boxing,
    build, js, legalize, low_level, osr, ownership, repr, transforms,
};

impl FunctionState {
    pub(super) fn lower_typed_storage(&mut self) -> Result<()> {
        let typed = typed::lower_memory(self);
        let metadata = storage::lower_metadata(self);
        if typed || metadata {
            self.graph.verify()?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(in crate::jit) fn lower_runtime(&mut self) -> Result<()> {
        self.lower_runtime_observed(false, &mut |_, _| {})
    }

    pub(in crate::jit) fn lower_runtime_observed(
        &mut self,
        baseline: bool,
        after: &mut dyn FnMut(&str, &FunctionState),
    ) -> Result<()> {
        after("input", self);
        array_checks::generic_cold(self);
        self.lower_js_observed(baseline, after)?;
        array_checks::generic_cold(self);
        if self.optimize {
            self.lower_typed_storage()?;
        }
        after("typed-storage", self);
        self.graph.strip_hints();
        cells::lower(self);
        after("cells", self);
        self.invalidate_cfg();
        transforms::promote(&mut self.graph, &mut self.notes);
        self.graph.verify()?;
        after("bindings", self);
        if !baseline {
            properties::lower(self)?;
        }
        after("properties", self);
        properties::lower_accesses(self)?;
        after("property-accesses", self);
        cells::lower_accesses(self);
        after("cell-accesses", self);
        self.lower_typed_storage()?;
        after("memory", self);
        numeric::operations(self);
        let dispatched = numeric::dispatch(self);
        if dispatched {
            numeric::operations(self);
        }
        after("numeric", self);
        array_checks::lower_checks(self);
        after("array-checks", self);
        self.ensure_facts();
        // Helper lowering erases the receivers needed to prove borrows.
        let borrows = borrow::borrows(&mut self.graph, &self.facts, &mut self.notes);
        helpers::lower(self);
        after("helpers", self);
        let plan = ownership::plan(&mut self.graph, &mut self.facts, borrows);
        if super::verification_enabled() {
            ownership::verify(&self.graph, &self.facts, &plan)?;
        }
        self.graph.ownership = Some(plan);
        after("ownership", self);
        properties::lower_loads(self);
        after("property-loads", self);
        storage::lower_array_loads(self);
        after("array-loads", self);
        storage::lower_stores(self);
        after("stores", self);
        values::lower(self);
        after("values", self);
        self.ensure_facts();
        boxing::conditions(&mut self.graph, &mut self.facts);
        after("conditions", self);
        repr::assign(&mut self.graph, &self.facts);
        after("repr", self);
        boxing::insert(&mut self.graph, &mut self.facts);
        after("boxing", self);
        conversions::simplify(&mut self.graph, &self.facts);
        conversions::hoist(&mut self.graph, &self.facts);
        after("conversions", self);
        self.graph.verify()?;
        Ok(())
    }

    pub(super) fn instrument_osr(
        &mut self,
        source: usize,
        feedback: Option<&calls::Repository>,
    ) -> Result<()> {
        self.osr =
            osr::Plan::discover_observed(&self.graph, source, feedback).map(std::sync::Arc::new);
        if let Some(plan) = self.osr.clone() {
            self.invalidate_cfg();
            plan.instrument(&mut self.graph)?;
        }
        Ok(())
    }
}

pub(in crate::jit) const SPECIALIZATION_PASSES: &[&str] =
    &["inline", "captures", "versioning", "osr"];

pub(in crate::jit) const RUNTIME_PASSES: &[&str] = &[
    "input",
    "property-specialization",
    "indexed-accesses",
    "normalise",
    "load-reuse",
    "indexed-checks",
    "property-caches",
    "js-cleanup",
    "string-concat",
    "numeric-specialization",
    "js",
    "typed-storage",
    "cells",
    "bindings",
    "properties",
    "property-accesses",
    "cell-accesses",
    "memory",
    "numeric",
    "array-checks",
    "helpers",
    "ownership",
    "property-loads",
    "array-loads",
    "stores",
    "values",
    "conditions",
    "repr",
    "boxing",
    "conversions",
];

pub(in crate::jit) const LOW_LEVEL_PASSES: &[&str] = &[
    "finalize-runtime",
    "cleanup",
    "ownership-lower",
    "unbox",
    "scalar-semantics",
    "scalar-cleanup",
    "aarch64-select",
    "legalize-convert",
    "bits",
    "cranelift-memory",
];

pub(in crate::jit) fn lower_low_level(
    graph: &mut Graph,
    ctx: Value,
    call_conv: CallConv,
    pointer: Type,
    after: &mut dyn FnMut(&str, &Graph),
) -> Result<()> {
    super::regex::lower(graph);
    super::runtime::cleanup::lower(graph);
    after("cleanup", graph);
    super::runtime::ownership::lower(graph, ctx, call_conv, pointer);
    after("ownership-lower", graph);
    repr::unbox(&mut low_level::Builder::new(graph))?;
    after("unbox", graph);
    graph.verify()?;
    legalize::prepare(graph);
    repr::lower_conversions(&mut low_level::Builder::new(graph))?;
    after("scalar-semantics", graph);
    reuse::lowered(graph);
    after("scalar-cleanup", graph);
    graph.verify()
}

pub(in crate::jit) fn lower_global(
    statements: &[Statement<'_>],
    span: Span,
    scoping: &Scoping,
    captured: &FxHashSet<SymbolId>,
    closures: &FxHashMap<u32, (cranelift_module::FuncId, Vec<SymbolId>, bool)>,
    arguments_references: &FxHashMap<u32, SymbolId>,
    ssa_opt: bool,
    continuations: bool,
    initial_only: bool,
    report_regions: bool,
    after: &mut dyn FnMut(&str, &FunctionState),
) -> Result<FunctionState> {
    let graph = build::lower_global(
        statements,
        span,
        scoping,
        captured,
        closures,
        arguments_references,
    )?;
    let mut function = FunctionState::new(graph, Vec::new(), ssa_opt);
    canonicalize(&mut function)?;
    let planned = inline_and_plan_specialization(function, ssa_opt, None, 1)?;
    let versioning = if initial_only {
        Versioning::Generic
    } else {
        Versioning::Speculate {
            fallback: None,
            continuations,
        }
    };
    specialize_observed(
        planned,
        versioning,
        report_regions,
        None,
        None,
        false,
        after,
    )
}

pub(in crate::jit) fn build_canonical(
    params: &FormalParameters<'_>,
    statements: &[Statement<'_>],
    expression: Option<&Expression<'_>>,
    span: Span,
    scoping: &Scoping,
    captured: &FxHashSet<SymbolId>,
    captures: &[SymbolId],
    closures: &FxHashMap<u32, (cranelift_module::FuncId, Vec<SymbolId>, bool)>,
    arguments_binding: Option<SymbolId>,
    arguments_references: &FxHashMap<u32, SymbolId>,
    generator: bool,
    ssa_opt: bool,
) -> Result<FunctionState> {
    let graph = build::lower_parts(
        params,
        statements,
        expression,
        span,
        scoping,
        captured,
        captures,
        closures,
        arguments_binding,
        arguments_references,
        generator,
    )?;
    let mut function = FunctionState::new(graph, Vec::new(), ssa_opt);
    canonicalize(&mut function)?;
    Ok(function)
}

#[cfg(test)]
pub(in crate::jit) fn analyze(
    function: &oxc_ast::ast::Function<'_>,
    scoping: &Scoping,
    captured: &FxHashSet<SymbolId>,
) -> Result<FunctionState> {
    optimize_graph(build::lower(function, scoping, captured)?, true)
}

fn canonicalize(function: &mut FunctionState) -> Result<()> {
    if !function.optimize {
        return Ok(());
    }
    function.invalidate_cfg();
    transforms::promote(&mut function.graph, &mut function.notes);
    function.graph.verify()?;
    js::typeof_comparisons(&mut function.graph);
    js::simplify(function);
    function.graph.verify()
}

pub(in crate::jit) fn canonicalize_graph(graph: Graph, ssa_opt: bool) -> Result<FunctionState> {
    let mut function = FunctionState::new(graph, Vec::new(), ssa_opt);
    canonicalize(&mut function)?;
    Ok(function)
}

#[cfg(test)]
fn optimize_graph(graph: Graph, ssa_opt: bool) -> Result<FunctionState> {
    optimize_canonical(canonicalize_graph(graph, ssa_opt)?, ssa_opt, None)
}

#[cfg(test)]
pub(in crate::jit) fn optimize_canonical(
    function: FunctionState,
    ssa_opt: bool,
    specialization: Option<(&calls::Repository, usize)>,
) -> Result<FunctionState> {
    lower_planned(
        plan_specialization(function, ssa_opt, specialization)?,
        Versioning::Speculate {
            fallback: None,
            continuations: true,
        },
        false,
    )
}

#[derive(Default)]
struct Expansions {
    arrays: FxHashSet<Inst>,
    constructors: FxHashSet<Inst>,
    splits: FxHashSet<Inst>,
    regexps: FxHashSet<Inst>,
}

impl Expansions {
    fn calls(
        &mut self,
        graph: &mut Graph,
        bodies: &calls::Repository,
        source: usize,
        notes: &mut Vec<String>,
    ) -> Result<()> {
        js::arguments::lower(graph);
        arrays::expand(graph, source, notes, &mut self.arrays)?;
        string_split::expand(graph, &mut self.splits);
        regexps::expand(graph, bodies, source, &mut self.regexps);
        Ok(())
    }

    fn constructors(&mut self, graph: &mut Graph) {
        construction::resolve(graph);
        construction::expand(graph, &mut self.constructors);
    }
}

#[cfg(test)]
pub(in crate::jit) fn plan_specialization(
    function: FunctionState,
    ssa_opt: bool,
    specialization: Option<(&calls::Repository, usize)>,
) -> Result<FunctionState> {
    inline_and_plan_specialization(function, ssa_opt, specialization, 1)
}

pub(in crate::jit) fn inline_and_plan_specialization(
    function: FunctionState,
    ssa_opt: bool,
    specialization: Option<(&calls::Repository, usize)>,
    inline_multiplier: usize,
) -> Result<FunctionState> {
    inline_and_plan_inner(function, ssa_opt, specialization, inline_multiplier, true)
}

pub(super) fn inline_and_plan_inner(
    mut function: FunctionState,
    ssa_opt: bool,
    specialization: Option<(&calls::Repository, usize)>,
    inline_multiplier: usize,
    select_call_variants: bool,
) -> Result<FunctionState> {
    function.optimize = ssa_opt && !function.graph.has_handlers && !function.graph.generator;
    if !function.optimize {
        return Ok(function);
    }
    let mut inline_budget = calls::Budget::new(inline_multiplier);
    let mut expansions = Expansions::default();
    if let Some((bodies, source)) = specialization {
        function.invalidate_cfg();
        expansions.calls(&mut function.graph, bodies, source, &mut function.notes)?;
        if select_call_variants
            && (inline_multiplier != 0 || bodies.call_variants && bodies.call_specialization)
        {
            calls::dispatch_methods(&mut function.graph, bodies, source);
        }
        let requests = calls::requests(&function.graph, bodies, source, &inline_budget);
        let plan = calls::plan_round(
            &function.graph,
            bodies,
            source,
            &requests,
            &mut inline_budget,
        );
        calls::apply(
            &mut function.graph,
            bodies,
            source,
            plan,
            &mut function.notes,
        )?;
        js::simplify(&mut function);
        function.invalidate_cfg();
        expansions.calls(&mut function.graph, bodies, source, &mut function.notes)?;
        function.graph.verify()?;
    }
    function.invalidate_cfg();
    expansions.constructors(&mut function.graph);
    if let Some((bodies, source)) = specialization {
        for _ in 0..calls::SPECIALIZATION_ROUNDS {
            if select_call_variants && inline_multiplier != 0 {
                calls::dispatch_methods(&mut function.graph, bodies, source);
            }
            let requests = calls::requests(&function.graph, bodies, source, &inline_budget);
            let plan = calls::plan_round(
                &function.graph,
                bodies,
                source,
                &requests,
                &mut inline_budget,
            );
            if plan.is_empty() {
                break;
            }
            function.invalidate_cfg();
            calls::apply(
                &mut function.graph,
                bodies,
                source,
                plan,
                &mut function.notes,
            )?;
            js::simplify(&mut function);
            function.invalidate_cfg();
            expansions.calls(&mut function.graph, bodies, source, &mut function.notes)?;
            expansions.constructors(&mut function.graph);
            function.graph.verify()?;
        }
    }
    function.invalidate_effects();
    if let Some((bodies, _)) = specialization {
        calls::install_summaries(&mut function.graph, bodies, &mut function.notes);
    }
    function.invalidate_values();
    let mut call_variants = Vec::new();
    if let Some((bodies, source)) = specialization {
        calls::direct_calls(&mut function.graph, bodies, source);
        if select_call_variants {
            call_variants = call_variants::apply(&mut function.graph, bodies, source)?;
        }
    }
    function.invalidate_cfg();
    reuse::reuse_loads(&mut function);
    let numbers = observations::plan(&mut function.graph, specialization);
    let reads = specialization.map_or_else(Vec::new, |(bodies, source)| {
        reads::plan(&function.graph, bodies, source)
    });
    let stores = specialization.map_or_else(Vec::new, |(bodies, source)| {
        reads::plan_stores(&function.graph, bodies, source)
    });
    let shapes = specialization.map_or_else(shapes::Plan::default, |(bodies, source)| {
        shapes::plan(&function.graph, bodies, source)
    });
    let mut conditional = specialization
        .map_or_else(calls::ConditionalPlan::default, |(bodies, _)| {
            calls::conditional_plan(&function.graph, bodies, &mut function.notes)
        });
    if let Some((bodies, source)) = specialization {
        conditional.native_calls =
            arrays::push_calls(&mut function.graph, bodies, source, &mut function.notes)?;
    }
    function.plan = SpecializationPlan {
        call_variants,
        conditional,
        inlines: inline_budget.inlines,
        reads,
        stores,
        shapes,
        numbers,
    };
    Ok(function)
}

#[derive(Clone, Copy)]
pub(in crate::jit) enum Versioning {
    Generic,
    Speculate {
        fallback: Option<usize>,
        continuations: bool,
    },
}

pub(in crate::jit) fn specialize(
    function: FunctionState,
    versioning: Versioning,
    report_regions: bool,
    osr_source: Option<usize>,
) -> Result<FunctionState> {
    specialize_observed(
        function,
        versioning,
        report_regions,
        osr_source,
        None,
        false,
        &mut |_, _| {},
    )
}

#[cfg(test)]
pub(in crate::jit) fn lower_planned(
    function: FunctionState,
    versioning: Versioning,
    report_regions: bool,
) -> Result<FunctionState> {
    let mut function = specialize(function, versioning, report_regions, None)?;
    function.lower_js()?;
    function.ensure_facts();
    borrow::borrows(&mut function.graph, &function.facts, &mut function.notes);
    Ok(function)
}

pub(in crate::jit) fn specialize_observed(
    mut function: FunctionState,
    versioning: Versioning,
    report_regions: bool,
    osr_source: Option<usize>,
    osr_feedback: Option<&calls::Repository>,
    baseline: bool,
    after: &mut dyn FnMut(&str, &FunctionState),
) -> Result<FunctionState> {
    let osr_feedback = match versioning {
        Versioning::Generic => None,
        Versioning::Speculate { .. } => osr_feedback,
    };
    after("inline", &function);
    if !function.optimize {
        function.ensure_facts();
        after("captures", &function);
        after("versioning", &function);
        if let Some(source) = osr_source {
            function.instrument_osr(source, osr_feedback)?;
        }
        after("osr", &function);
        return Ok(function);
    }
    function.graph.entry_guards.clear();
    function.invalidate_cfg();
    function.graph.megamorphic_gets = function
        .plan
        .shapes
        .megamorphic_sites()
        .iter()
        .copied()
        .collect();
    if baseline {
        property_accesses::assume_natives(&mut function.graph);
    } else {
        property_accesses::assume(&mut function.graph, &function.plan.shapes);
        reads::assume(&mut function.graph, &function.plan.reads);
        reads::assume(&mut function.graph, &function.plan.stores);
    }
    captures::sink_closures(&mut function);
    if captures::lazy_captures(&mut function) {
        function.invalidate_cfg();
        transforms::promote(&mut function.graph, &mut function.notes);
        js::simplify(&mut function);
    }
    function.graph.verify()?;
    captures::forward_captured_parameters(&mut function);
    if cells::lower(&mut function) {
        transforms::promote(&mut function.graph, &mut function.notes);
    }
    function.graph.verify()?;
    function.invalidate_cfg();
    after("captures", &function);
    let regions = match versioning {
        Versioning::Generic => {
            regions::materialize_properties(&mut function.graph)?;
            Vec::new()
        }
        Versioning::Speculate {
            fallback,
            continuations,
        } => {
            let mut regions = regions::version_regions(
                &mut function.graph,
                &function.plan.conditional,
                fallback,
                continuations,
                &mut function.notes,
                report_regions,
                &function.plan.shapes,
            )?;
            function.graph.verify()?;
            if regions.is_empty() {
                regions.extend(regions::version_string_leaves(
                    &mut function.graph,
                    continuations,
                    &mut function.notes,
                )?);
            }
            regions
        }
    };
    if report_regions {
        for (id, region) in &function.graph.regions {
            let mut blocks: Vec<_> = region.blocks.iter().copied().collect();
            blocks.sort();
            let blocks = blocks
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let fallback = match region.fallback {
                Some(RegionFallback::Region(id)) => id.to_string(),
                Some(RegionFallback::CompiledEntry(address)) => {
                    format!("compiled entry {address:#x}")
                }
                None => "none".into(),
            };
            function
                .notes
                .push(format!("{id}: blocks [{blocks}], fallback {fallback}"));
        }
    }
    function.graph.strip_assumes();
    function.specialized_regions = Some(regions);
    after("versioning", &function);
    if let Some(source) = osr_source {
        function.instrument_osr(source, osr_feedback)?;
    }
    after("osr", &function);
    Ok(function)
}

impl FunctionState {
    #[cfg(test)]
    pub(in crate::jit::semantic) fn lower_js(&mut self) -> Result<()> {
        self.lower_js_observed(false, &mut |_, _| {})
    }

    pub(in crate::jit::semantic) fn rewrite_properties(&mut self) -> Result<()> {
        if !self.optimize || self.specialized_regions.is_none() {
            return Ok(());
        }
        property_accesses::rewrite(&mut self.graph)
    }

    fn lower_js_observed(
        &mut self,
        baseline: bool,
        after: &mut dyn FnMut(&str, &FunctionState),
    ) -> Result<()> {
        let Some(regions) = self.specialized_regions.take().filter(|_| self.optimize) else {
            self.ensure_facts();
            for &pass in RUNTIME_PASSES
                .iter()
                .skip(1)
                .take_while(|&&pass| pass != "typed-storage")
            {
                after(pass, self);
            }
            return Ok(());
        };
        if !baseline {
            property_accesses::rewrite(&mut self.graph)?;
        }
        after("property-specialization", self);
        if !baseline {
            reads::lower(&mut self.graph, &self.plan.reads, &mut self.notes)?;
            writes::lower(&mut self.graph, &self.plan.stores)?;
        }
        after("indexed-accesses", self);
        self.invalidate_cfg();
        js::simplify(self);
        self.graph.verify()?;
        after("normalise", self);
        js::coercions(self);
        js::bit_tests(self);
        self.graph.verify()?;
        reuse::reuse_loads(self);
        self.graph.verify()?;
        for region in &regions {
            regions::hoist(self, *region);
            self.graph.verify()?;
        }
        after("load-reuse", self);
        let bits_changed = numeric::bitwise(self);
        let arrays_changed = array_checks::lower_arrays(self);
        let typed_changed = typed::lower(self)?;
        let indexed_changed = bits_changed || arrays_changed || typed_changed;
        if indexed_changed {
            js::simplify(self);
        }
        after("indexed-checks", self);
        self.invalidate_cfg();
        let properties_changed = !baseline && properties::lower(self)?;
        let kinds_changed = array_checks::lower_array_kinds(self);
        if indexed_changed || properties_changed || kinds_changed {
            reuse::reuse_loads(self);
        }
        after("property-caches", self);
        views::views(self);
        comparisons::expose(self);
        self.graph.verify()?;
        js::simplify(self);
        self.graph.verify()?;
        captures::sink_closures(self);
        captures::lazy_captures(self);
        self.graph.verify()?;
        self.invalidate_cfg();
        string_split::forward_packs(&mut self.graph);
        transforms::trim(&mut self.graph);
        transforms::remove_trivial_params(&mut self.graph);
        self.graph.verify()?;
        after("js-cleanup", self);
        js::strings::concatenate(self);
        self.graph.verify()?;
        after("string-concat", self);
        if numeric::lower(self) {
            reuse::reuse_loads(self);
            js::simplify(self);
            if numeric::relax_checks(self) {
                js::simplify(self);
            }
        }
        for region in &regions {
            regions::hoist(self, *region);
            self.graph.verify()?;
        }
        after("numeric-specialization", self);
        self.ensure_facts();
        after("js", self);
        Ok(())
    }
}

pub(super) fn prepare_continuation(graph: Graph) -> FunctionState {
    let mut function = FunctionState::new(graph, Vec::new(), true);
    js::simplify(&mut function);
    js::coercions(&mut function);
    js::bit_tests(&mut function);
    if array_checks::lower_arrays(&mut function) {
        js::simplify(&mut function);
        reuse::reuse_loads(&mut function);
    }
    views::views(&mut function);
    js::simplify(&mut function);
    function.invalidate_cfg();
    transforms::trim(&mut function.graph);
    transforms::remove_trivial_params(&mut function.graph);
    function.ensure_facts();
    function
}
