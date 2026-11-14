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
use js::{arrays, captures, comparisons, construction, string_split};
use oxc_ast::ast::{Expression, FormalParameters, Statement};
use oxc_semantic::Scoping;
use oxc_span::Span;
use oxc_syntax::symbol::SymbolId;
use transforms::{conversions, reuse};

use super::{
    FunctionState, Graph, Inst, RegionFallback, Result, SpecializationPlan, Value, borrow, build,
    js, legalize, low_level, osr, ownership, repr, transforms,
};

impl FunctionState {
    pub(in crate::jit) fn lower_memory(&mut self) -> Result<()> {
        let typed = typed::lower_memory(self);
        let metadata = storage::lower_metadata(self);
        if typed || metadata {
            self.graph.verify()?;
            self.ensure_facts();
            borrow::borrows(&mut self.graph, &self.facts, &mut self.notes);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(in crate::jit) fn prepare_for_emission(&mut self) -> Result<()> {
        self.prepare_for_emission_observed(&mut |_, _| {})
    }

    pub(in crate::jit) fn prepare_for_emission_observed(
        &mut self,
        after: &mut dyn FnMut(&str, &FunctionState),
    ) -> Result<()> {
        after("input", self);
        self.graph.strip_hints();
        cells::lower(self);
        after("cells", self);
        self.invalidate_cfg();
        transforms::promote(&mut self.graph, &mut self.notes);
        self.graph.verify()?;
        after("bindings", self);
        if properties::lower(self)? {
            self.ensure_facts();
            borrow::borrows(&mut self.graph, &self.facts, &mut self.notes);
        }
        after("properties", self);
        properties::lower_accesses(self)?;
        after("property-accesses", self);
        self.lower_memory()?;
        after("memory", self);
        let lowered = numeric::operations(self);
        let dispatched = numeric::dispatch(self);
        if dispatched {
            numeric::operations(self);
        }
        if lowered || dispatched {
            self.ensure_facts();
            borrow::borrows(&mut self.graph, &self.facts, &mut self.notes);
        }
        after("numeric", self);
        array_checks::lower_checks(self);
        after("array-checks", self);
        self.ensure_facts();
        helpers::lower(self);
        after("helpers", self);
        legalize::conditions(&mut self.graph, &mut self.facts);
        after("conditions", self);
        repr::assign(&mut self.graph, &self.facts);
        after("repr", self);
        legalize::legalize(&mut self.graph, &mut self.facts);
        after("legalize", self);
        conversions::simplify(&mut self.graph, &self.facts);
        conversions::hoist(&mut self.graph, &self.facts);
        conversions::split(&mut self.graph, &self.facts);
        after("conversions", self);
        self.graph.verify()?;
        let plan = ownership::plan(&mut self.graph, &mut self.facts);
        if cfg!(debug_assertions) {
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
        self.graph.verify()?;
        Ok(())
    }

    pub(in crate::jit) fn prepare_osr(&mut self, source: usize) -> Result<()> {
        self.osr = osr::Plan::discover(&self.graph, source).map(std::sync::Arc::new);
        if let Some(plan) = self.osr.clone() {
            self.invalidate_cfg();
            plan.instrument(&mut self.graph)?;
            self.ensure_facts();
            borrow::borrows(&mut self.graph, &self.facts, &mut self.notes);
        }
        Ok(())
    }
}

pub(in crate::jit) const LOWERING_PASSES: &[&str] = &[
    "emit",
    "scratch",
    "repr-lower",
    "cleanup",
    "ownership-lower",
    "bits",
];

pub(in crate::jit) fn lower_emitted(
    graph: &mut Graph,
    ctx: Value,
    call_conv: CallConv,
    pointer: Type,
    after: &mut dyn FnMut(&str, &Graph),
) -> Result<()> {
    repr::lower(&mut low_level::Builder::new(graph))?;
    after("repr-lower", graph);
    super::runtime::cleanup::lower(graph);
    after("cleanup", graph);
    super::runtime::ownership::lower(graph, ctx, call_conv, pointer);
    after("ownership-lower", graph);
    super::bits::expand(graph);
    after("bits", graph);
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
    report_regions: bool,
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
    let planned = plan_canonical(function, ssa_opt, None)?;
    let versioning = Versioning::Speculate {
        fallback: None,
        continuations,
    };
    optimize_planned(planned, versioning, report_regions)
}

pub(in crate::jit) fn canonical_for_emission(
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
    optimize_planned(
        plan_canonical(function, ssa_opt, specialization)?,
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
}

impl Expansions {
    fn calls(&mut self, graph: &mut Graph, source: usize, notes: &mut Vec<String>) -> Result<()> {
        arrays::expand(graph, source, notes, &mut self.arrays)?;
        string_split::expand(graph, &mut self.splits);
        Ok(())
    }

    fn constructors(&mut self, graph: &mut Graph) {
        construction::resolve(graph);
        construction::expand(graph, &mut self.constructors);
    }
}

pub(in crate::jit) fn plan_canonical(
    function: FunctionState,
    ssa_opt: bool,
    specialization: Option<(&calls::Repository, usize)>,
) -> Result<FunctionState> {
    plan_canonical_with_budget(function, ssa_opt, specialization, 1)
}

pub(in crate::jit) fn plan_canonical_with_budget(
    function: FunctionState,
    ssa_opt: bool,
    specialization: Option<(&calls::Repository, usize)>,
    inline_multiplier: usize,
) -> Result<FunctionState> {
    plan_canonical_inner(function, ssa_opt, specialization, inline_multiplier, true)
}

pub(super) fn plan_canonical_inner(
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
        expansions.calls(&mut function.graph, source, &mut function.notes)?;
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
        expansions.calls(&mut function.graph, source, &mut function.notes)?;
        function.graph.verify()?;
    }
    function.invalidate_cfg();
    expansions.constructors(&mut function.graph);
    if let Some((bodies, source)) = specialization {
        for _ in 0..calls::SPECIALIZATION_ROUNDS {
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
            expansions.calls(&mut function.graph, source, &mut function.notes)?;
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

pub(in crate::jit) fn optimize_planned(
    mut function: FunctionState,
    versioning: Versioning,
    report_regions: bool,
) -> Result<FunctionState> {
    if !function.optimize {
        function.ensure_facts();
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
    property_accesses::assume(&mut function.graph, &function.plan.shapes);
    reads::assume(&mut function.graph, &function.plan.reads);
    reads::assume(&mut function.graph, &function.plan.stores);
    captures::sink_closures(&mut function);
    if captures::lazy_captures(&mut function) {
        function.invalidate_cfg();
        transforms::promote(&mut function.graph, &mut function.notes);
        js::simplify(&mut function);
    }
    function.graph.verify()?;
    captures::forward_captured_parameters(&mut function);
    function.graph.verify()?;
    function.invalidate_cfg();
    let regions = match versioning {
        Versioning::Generic => {
            regions::materialize_properties(&mut function.graph)?;
            Vec::new()
        }
        Versioning::Speculate {
            fallback,
            continuations,
        } => {
            let mut regions = regions::version_with_report(
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
    property_accesses::rewrite(&mut function.graph)?;
    reads::lower(
        &mut function.graph,
        &function.plan.reads,
        &mut function.notes,
    )?;
    writes::lower(&mut function.graph, &function.plan.stores)?;
    function.invalidate_cfg();
    regions::normalise_regions(&mut function, &regions)?;
    js::coercions(&mut function);
    js::bit_tests(&mut function);
    function.graph.verify()?;
    reuse::reuse_loads(&mut function);
    function.graph.verify()?;
    for region in &regions {
        regions::hoist(&mut function, *region);
        function.graph.verify()?;
    }
    let bits_changed = numeric::bitwise(&mut function);
    let arrays_changed = array_checks::lower_arrays(&mut function);
    let typed_changed = typed::lower(&mut function)?;
    if bits_changed || arrays_changed || typed_changed {
        js::simplify(&mut function);
        reuse::reuse_loads(&mut function);
    }
    function.invalidate_cfg();
    if properties::lower(&mut function)? {
        reuse::reuse_loads(&mut function);
    }
    if array_checks::lower_array_kinds(&mut function) {
        reuse::reuse_loads(&mut function);
    }
    views::views(&mut function);
    comparisons::expose(&mut function);
    function.graph.verify()?;
    js::simplify(&mut function);
    function.graph.verify()?;
    captures::sink_closures(&mut function);
    captures::lazy_captures(&mut function);
    function.graph.verify()?;
    function.invalidate_cfg();
    string_split::forward_packs(&mut function.graph);
    transforms::trim(&mut function.graph);
    transforms::remove_trivial_params(&mut function.graph);
    function.graph.verify()?;
    for region in &regions {
        regions::hoist(&mut function, *region);
        function.graph.verify()?;
    }
    if numeric::lower(&mut function) {
        reuse::reuse_loads(&mut function);
        for region in &regions {
            regions::hoist(&mut function, *region);
            function.graph.verify()?;
        }
    }
    function.ensure_facts();
    borrow::borrows(&mut function.graph, &function.facts, &mut function.notes);
    Ok(function)
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
    borrow::borrows(&mut function.graph, &function.facts, &mut function.notes);
    function
}
