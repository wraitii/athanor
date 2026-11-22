#[cfg(test)]
use crate::jit::layout::ReadLayout;
use crate::jit::semantic::JsOp;
use crate::{FxHashMap, FxHashSet};
use std::sync::Arc;

use cranelift_module::FuncId;

use oxc_syntax::operator::BinaryOperator;

use super::super::super::effects::{Assumption, Control, Effects, Locations, Memory};
use super::super::super::math::METHODS;
use super::super::{
    CallSummary, Cfg, DirectTarget, Facts, FunctionState, Graph, Inst, Native, Op, Result,
    TerminatorKind, UnaryOp, Value, ValueType,
};

pub(in crate::jit::semantic) const SPECIALIZATION_BUDGET: usize = 128;
pub(in crate::jit::semantic) const SPECIALIZATION_BODY_INSTS: usize = 64;
pub(in crate::jit::semantic) const SPECIALIZATION_ROUNDS: usize = 2;
pub(crate) const DEFAULT_RECOMPILE_INLINE_MULTIPLIER: usize = 2;

#[derive(Clone)]
pub(in crate::jit) struct Repository {
    pub(in crate::jit) call_variants: bool,
    pub(in crate::jit) call_specialization: bool,
    pub(in crate::jit) regex_inline: bool,
    pub(in crate::jit) regex_drop_unused_exec: bool,
    pub(in crate::jit) variants: crate::jit::variants::Registry,
    bodies: FxHashMap<FuncId, Arc<Body>>,
    regions: FxHashSet<FuncId>,
    pub(in crate::jit) number_first: FxHashSet<FuncId>,
    pub(in crate::jit) feedback: FxHashMap<crate::jit::profile::Site, Vec<FuncId>>,
    pub(in crate::jit) method_pairs: FxHashMap<crate::jit::profile::Site, Vec<(u64, u32)>>,
    pub(in crate::jit) native_feedback:
        FxHashMap<crate::jit::profile::Site, crate::heap::NativeFunction>,
    pub(in crate::jit) read_feedback: FxHashMap<
        (crate::jit::profile::Site, crate::jit::profile::ReadKind),
        crate::jit::profile::ReadType,
    >,
    pub(in crate::jit) shape_feedback: FxHashMap<
        (
            crate::jit::profile::Site,
            crate::jit::profile::PropertyAccess,
        ),
        crate::jit::profile::ShapeAccess,
    >,
    shape_fallback: FxHashMap<
        (usize, u32, u32, crate::jit::profile::PropertyAccess),
        crate::jit::profile::ShapeAccess,
    >,
    locations: FxHashMap<(usize, u32), FuncId>,
    globals: FxHashMap<String, FuncId>,
    pub(in crate::jit) sources: FxHashMap<usize, usize>,
}

impl Default for Repository {
    fn default() -> Self {
        Self {
            call_variants: true,
            call_specialization: true,
            regex_inline: true,
            regex_drop_unused_exec: true,
            variants: Default::default(),
            bodies: Default::default(),
            regions: Default::default(),
            number_first: Default::default(),
            feedback: Default::default(),
            method_pairs: Default::default(),
            native_feedback: Default::default(),
            read_feedback: Default::default(),
            shape_feedback: Default::default(),
            shape_fallback: Default::default(),
            locations: Default::default(),
            globals: Default::default(),
            sources: Default::default(),
        }
    }
}

#[derive(Clone)]
struct Body {
    name: String,
    source: usize,
    graph: Graph,
    notes: Vec<String>,
    arrow: bool,
}

impl Repository {
    pub(in crate::jit) fn insert_shape_feedback(
        &mut self,
        site: crate::jit::profile::Site,
        property: crate::jit::profile::PropertyAccess,
        access: crate::jit::profile::ShapeAccess,
    ) {
        self.shape_feedback.insert((site, property), access);
        let key = (site.source, site.start, site.end, property);
        self.shape_fallback
            .entry(key)
            .and_modify(|existing| *existing = existing.merge(access))
            .or_insert(access);
    }

    pub(in crate::jit) fn shape_feedback(
        &self,
        site: crate::jit::profile::Site,
        property: crate::jit::profile::PropertyAccess,
    ) -> Option<crate::jit::profile::ShapeAccess> {
        use crate::jit::profile::ShapeAccess;

        self.shape_feedback
            .get(&(site, property))
            .copied()
            .or_else(|| {
                if site.inline_context == 0 {
                    return None;
                }
                let original = crate::jit::profile::Site {
                    inline_context: 0,
                    ..site
                };
                self.shape_feedback
                    .get(&(original, property))
                    .copied()
                    .or_else(|| {
                        self.shape_fallback
                            .get(&(site.source, site.start, site.end, property))
                            .copied()
                    })
                    .filter(|access| {
                        matches!(
                            access,
                            ShapeAccess::Read { .. } | ShapeAccess::Overwrite { .. }
                        )
                    })
            })
    }

    pub(in crate::jit) fn with_options(options: &crate::EngineOptions) -> Self {
        Self {
            call_variants: options.call_variants,
            call_specialization: options.call_specialization,
            regex_inline: options.regex && options.regex_inline,
            regex_drop_unused_exec: options.regex_drop_unused_exec,
            ..Default::default()
        }
    }

    pub(in crate::jit) fn observed(
        &self,
        site: crate::jit::profile::Site,
        kind: crate::jit::profile::Observation,
    ) -> bool {
        match kind {
            crate::jit::profile::Observation::Call => {
                self.feedback.contains_key(&site) || self.native_feedback.contains_key(&site)
            }
            crate::jit::profile::Observation::Read(kind) => {
                self.read_feedback.contains_key(&(site, kind))
            }
            crate::jit::profile::Observation::NumberFailure => self
                .read_feedback
                .contains_key(&(site, crate::jit::profile::ReadKind::Number)),
            crate::jit::profile::Observation::PrimitiveFailure => self
                .read_feedback
                .contains_key(&(site, crate::jit::profile::ReadKind::Primitive)),
        }
    }

    pub(in crate::jit) fn callees(&self, code: FuncId) -> Vec<FuncId> {
        let Some(body) = self.bodies.get(&code) else {
            return Vec::new();
        };
        self.graph_callees(&body.graph, body.source)
    }

    pub(in crate::jit) fn graph_callees(&self, graph: &Graph, source: usize) -> Vec<FuncId> {
        let mut targets = Vec::new();
        for block in Cfg::new(graph).rpo {
            for &call in &graph.blocks[block].insts {
                let callee = match graph.insts[call].op {
                    Op::Javascript(
                        JsOp::Call { callee, .. }
                        | JsOp::CallPack(callee, ..)
                        | JsOp::Construct { callee, .. }
                        | JsOp::ConstructorCall { callee, .. }
                        | JsOp::ConstructPack(callee, ..)
                        | JsOp::ConstructPackResolved(callee, ..)
                        | JsOp::ConstructorCallPack(callee, ..)
                        | JsOp::ConstructResolved { callee, .. },
                    ) => callee,
                    _ => continue,
                };
                if let Some(target) = call_target(graph, self, source, call, callee)
                    && !targets.contains(&target)
                {
                    targets.push(target);
                }
            }
        }
        targets
    }

    pub(in crate::jit) fn insert(
        &mut self,
        source: usize,
        name: &str,
        code: FuncId,
        function: FunctionState,
    ) {
        self.locations
            .insert((source, function.graph.span.start), code);
        self.globals.insert(name.to_owned(), code);
        self.bodies.insert(
            code,
            Arc::new(Body {
                name: name.to_owned(),
                source,
                graph: function.graph,
                notes: function.notes,
                arrow: false,
            }),
        );
    }

    pub(in crate::jit) fn insert_closure(
        &mut self,
        source: usize,
        code: FuncId,
        function: FunctionState,
        arrow: bool,
    ) {
        self.locations
            .insert((source, function.graph.span.start), code);
        let name = function.graph.name.clone();
        self.bodies.insert(
            code,
            Arc::new(Body {
                name,
                source,
                graph: function.graph,
                notes: function.notes,
                arrow,
            }),
        );
    }

    pub(in crate::jit) fn insert_region(&mut self, source: usize, code: FuncId, graph: Graph) {
        self.regions.insert(code);
        self.bodies.insert(
            code,
            Arc::new(Body {
                name: graph.name.clone(),
                source,
                graph,
                notes: Vec::new(),
                arrow: false,
            }),
        );
    }

    pub(in crate::jit) fn is_region(&self, code: FuncId) -> bool {
        self.regions.contains(&code)
    }

    pub(in crate::jit) fn code(&self, source: usize, start: u32) -> Option<FuncId> {
        self.locations.get(&(source, start)).copied()
    }

    pub(in crate::jit::semantic) fn number_first(&self, source: usize, start: u32) -> bool {
        self.code(source, start)
            .is_some_and(|code| self.number_first.contains(&code))
    }

    pub(in crate::jit) fn emission(&self, code: FuncId) -> Option<(usize, FunctionState)> {
        let body = self.bodies.get(&code)?;
        Some((
            body.source,
            FunctionState::new(body.graph.clone(), body.notes.clone(), true),
        ))
    }

    pub(in crate::jit) fn function(&self, source: usize, start: u32) -> Option<FunctionState> {
        let code = self.locations.get(&(source, start))?;
        self.emission(*code).map(|(_, function)| function)
    }

    pub(in crate::jit) fn lexical_this(&self, code: FuncId) -> bool {
        self.bodies
            .get(&code)
            .is_some_and(|body| body.arrow && super::inlining::uses_this(&body.graph))
    }

    pub(in crate::jit::semantic) fn name(&self, code: FuncId) -> Option<&str> {
        self.bodies.get(&code).map(|body| body.name.as_str())
    }

    pub(in crate::jit::semantic) fn body(&self, code: FuncId) -> Option<&Graph> {
        let body = self.bodies.get(&code)?;
        Some(&body.graph)
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
struct SummaryEffects {
    memory: Memory,
    control: Control,
    globals_read: FxHashSet<String>,
    globals_written: FxHashSet<String>,
    properties_read: FxHashSet<String>,
    properties_written: FxHashSet<String>,
    unresolved: FxHashSet<(FuncId, Inst)>,
    unsupported: bool,
}

impl SummaryEffects {
    fn merge(&mut self, other: &Self) {
        self.memory = self.memory.union(other.memory);
        self.control = self.control.union(other.control);
        self.unsupported |= other.unsupported;
        self.globals_read.extend(other.globals_read.iter().cloned());
        self.globals_written
            .extend(other.globals_written.iter().cloned());
        self.properties_read
            .extend(other.properties_read.iter().cloned());
        self.properties_written
            .extend(other.properties_written.iter().cloned());
        self.unresolved.extend(other.unresolved.iter().copied());
    }

    fn absorb(&mut self, effects: Effects) {
        self.memory = self.memory.union(effects.memory);
        self.control = self.control.union(effects.control);
    }

    fn writes(&self) -> bool {
        !self.memory.writes.is_empty() || !self.properties_written.is_empty()
    }
}

fn assumptions(op: &Op, facts: &Facts) -> &'static [Assumption] {
    const PROPERTY: &[Assumption] = &[Assumption::NoAccessors, Assumption::NoCoercionCallbacks];
    const COERCION: &[Assumption] = &[Assumption::NoCoercionCallbacks];
    let coercing = || {
        let operands = op
            .operands_iter()
            .fold(ValueType::default(), |types, value| {
                types.union(facts.of(value))
            });
        if operands.within(ValueType::PRIMITIVE) {
            &[]
        } else {
            COERCION
        }
    };
    match op {
        Op::Javascript(
            JsOp::Get(..)
            | JsOp::GetIndex(..)
            | JsOp::FillPropertyCache(..)
            | JsOp::Set(..)
            | JsOp::SetIndex(..)
            | JsOp::TypedSet(..)
            | JsOp::Define(..),
        ) => PROPERTY,
        Op::Javascript(JsOp::Spread(_) | JsOp::IteratorNext(_) | JsOp::EnumerateNext(_)) => {
            &[Assumption::DirectIteration]
        }
        Op::Javascript(
            JsOp::ProbeGlobal(_)
            | JsOp::ProbeProperty(..)
            | JsOp::ProbeShapeProperty(..)
            | JsOp::ProbeIndex(..),
        ) => &[Assumption::NoAccessors],
        Op::Javascript(JsOp::View(_) | JsOp::TypedIndex(..)) => &[Assumption::FixedTypedStorage],
        Op::Javascript(JsOp::GuardedCompare {
            operator,
            left,
            right,
            ..
        }) => assumptions(
            &Op::Javascript(JsOp::Binary(*operator, *left, *right)),
            facts,
        ),
        Op::Javascript(JsOp::Binary(
            BinaryOperator::StrictEquality
            | BinaryOperator::StrictInequality
            | BinaryOperator::In
            | BinaryOperator::Instanceof,
            ..,
        )) => &[],
        Op::Javascript(JsOp::Unary(
            UnaryOp::Not | UnaryOp::Typeof | UnaryOp::Increment | UnaryOp::Decrement,
            _,
        )) => &[],
        Op::Javascript(JsOp::CallNative(native, _))
            if METHODS[native.index()].1 == crate::heap::MathFunction::Random =>
        {
            &[]
        }
        Op::Javascript(
            JsOp::Binary(..) | JsOp::ToNumeric(_) | JsOp::Unary(..) | JsOp::CallNative(..),
        ) => coercing(),
        _ => &[],
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
struct Requirements {
    assumptions: FxHashSet<Assumption>,
    identities: FxHashSet<(String, FuncId)>,
    natives: FxHashSet<Native>,
    string_splits: FxHashSet<(FuncId, Inst)>,
}

impl Requirements {
    fn merge(&mut self, other: &Self) {
        self.assumptions.extend(other.assumptions.iter().copied());
        self.identities.extend(other.identities.iter().cloned());
        self.natives.extend(other.natives.iter().copied());
        self.string_splits
            .extend(other.string_splits.iter().copied());
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
struct Summary {
    effects: SummaryEffects,
    requires: Requirements,
}

impl Summary {
    fn merge(&mut self, other: &Self) {
        self.effects.merge(&other.effects);
        self.requires.merge(&other.requires);
    }
}

struct LocalEffects {
    summary: Summary,
    calls: Vec<(Inst, FuncId)>,
}

impl Repository {
    fn local_effects(&self, code: FuncId) -> Option<LocalEffects> {
        let graph = self.body(code)?;
        let facts = Facts::compute(graph);
        let mut effects = SummaryEffects::default();
        let mut requires = Requirements::default();
        if graph.generator || graph.has_handlers {
            effects.absorb(Effects::UNKNOWN);
            effects.unsupported = true;
        }
        let mut calls = Vec::new();
        for block in Cfg::new(graph).rpo {
            for &inst in &graph.blocks[block].insts {
                let op = &graph.insts[inst].op;
                match op {
                    Op::Javascript(JsOp::GlobalRead(name) | JsOp::GlobalReadOptional(name)) => {
                        effects.globals_read.insert(name.clone());
                    }
                    Op::Javascript(
                        JsOp::GlobalWrite(name, _) | JsOp::GlobalInitialize(name, _),
                    ) => {
                        effects.globals_written.insert(name.clone());
                    }
                    Op::Javascript(JsOp::Get(_, name) | JsOp::FillPropertyCache(_, _, name)) => {
                        effects.properties_read.insert(name.clone());
                    }
                    Op::Javascript(JsOp::IsPropertyCached(..) | JsOp::CachedProperty(..)) => {
                        effects.properties_read.insert("*".into());
                    }
                    Op::Javascript(
                        JsOp::Set(_, name, _)
                        | JsOp::Define(_, name, _)
                        | JsOp::DefineHidden(_, name, _),
                    ) => {
                        effects.properties_written.insert(name.clone());
                        effects.memory.writes = effects.memory.writes.union(Locations::HEAP);
                    }
                    Op::Javascript(
                        JsOp::GetIndex(..)
                        | JsOp::ArrayGet(..)
                        | JsOp::ArrayLoad(..)
                        | JsOp::TypedIndex(..),
                    ) => {
                        effects.properties_read.insert("*".into());
                    }
                    Op::Javascript(
                        JsOp::SetIndex(..) | JsOp::TypedSet(..) | JsOp::ArrayAppend(..),
                    ) => {
                        effects.properties_written.insert("*".into());
                        effects.memory.writes = effects.memory.writes.union(Locations::HEAP);
                    }
                    _ => {}
                }
                if let Op::Javascript(JsOp::Call {
                    callee,
                    receiver,
                    args,
                    ..
                }) = op
                {
                    let target = match graph.defining_op(*callee) {
                        Some(Op::Javascript(JsOp::GlobalRead(name))) => {
                            self.globals.get(name).copied().inspect(|&target| {
                                requires.identities.insert((name.clone(), target));
                            })
                        }
                        Some(Op::Javascript(JsOp::Closure(_, target, _, false, _, _, _))) => {
                            Some(*target)
                        }
                        _ => None,
                    };
                    if let Some(target) = target.filter(|target| self.body(*target).is_some()) {
                        effects.control = Control::UNKNOWN;
                        calls.push((inst, target));
                        continue;
                    }
                    if let Some(Op::Javascript(JsOp::Get(object, property))) =
                        graph.defining_op(*callee)
                        && property == "split"
                        && *receiver == Some(*object)
                    {
                        requires.string_splits.insert((code, inst));
                        let lowered = Op::Javascript(JsOp::StringSplit(*object, args.clone()));
                        effects.absorb(graph.contract(&lowered, &facts).effects);
                        requires
                            .assumptions
                            .extend(assumptions(&lowered, &facts).iter().copied());
                        continue;
                    }
                    if let Some(Op::Javascript(JsOp::Get(object, property))) =
                        graph.defining_op(*callee)
                        && let Some(Op::Javascript(JsOp::GlobalRead(global))) =
                            graph.defining_op(*object)
                        && let Some(native) = Native::lookup(global, property)
                    {
                        requires.natives.insert(native);
                        let lowered = Op::Javascript(JsOp::CallNative(native, args.clone()));
                        effects.absorb(graph.contract(&lowered, &facts).effects);
                        requires
                            .assumptions
                            .extend(assumptions(&lowered, &facts).iter().copied());
                        continue;
                    }
                }
                let contract = graph.contract(op, &facts).effects;
                effects.absorb(contract);
                requires
                    .assumptions
                    .extend(assumptions(op, &facts).iter().copied());
                if contract.memory.reenters() {
                    effects.unresolved.insert((code, inst));
                }
            }
        }
        Some(LocalEffects {
            summary: Summary { effects, requires },
            calls,
        })
    }

    fn effect_summaries(
        &self,
        roots: impl IntoIterator<Item = FuncId>,
    ) -> FxHashMap<FuncId, Summary> {
        let mut local = FxHashMap::default();
        let mut pending: Vec<_> = roots.into_iter().collect();
        while let Some(code) = pending.pop() {
            if local.contains_key(&code) {
                continue;
            }
            let Some(body) = self.local_effects(code) else {
                continue;
            };
            pending.extend(body.calls.iter().map(|&(_, target)| target));
            local.insert(code, body);
        }
        let mut summaries: FxHashMap<_, _> = local
            .iter()
            .map(|(&code, body)| (code, body.summary.clone()))
            .collect();
        loop {
            let mut changed = false;
            for (&code, body) in &local {
                let mut summary = body.summary.clone();
                for (_, target) in &body.calls {
                    if let Some(callee) = summaries.get(target) {
                        summary.merge(callee);
                    }
                }
                if summaries.get(&code) != Some(&summary) {
                    summaries.insert(code, summary);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        summaries
    }
    fn obligation_chain(&self, root: FuncId, identities_block: bool) -> String {
        let label = |code| {
            self.bodies.get(&code).map_or_else(
                || code.to_string(),
                |body| format!("{} ({code})", body.name),
            )
        };
        let mut pending = vec![(root, label(root))];
        let mut visited = FxHashSet::default();
        while let Some((code, path)) = pending.pop() {
            if !visited.insert(code) {
                continue;
            }
            let Some(local) = self.local_effects(code) else {
                continue;
            };
            let Summary { effects, requires } = &local.summary;
            if effects.unsupported {
                return format!("{path}: generator or exception handler");
            }
            if let Some((_, inst)) = effects.unresolved.iter().min_by_key(|(_, inst)| *inst) {
                let graph = self.body(code).unwrap();
                return format!(
                    "{path} -> {inst}: unresolved {}",
                    unresolved_operation(graph, &graph.insts[*inst].op)
                );
            }
            if identities_block
                && let Some((name, target)) =
                    requires.identities.iter().min_by_key(|(name, _)| name)
            {
                return format!("{path}: requires global {name} to remain {target}");
            }
            if let Some((_, inst)) = requires.string_splits.iter().min_by_key(|(_, inst)| *inst) {
                return format!(
                    "{path} -> {inst}: method .split requires native String.split identity and its original receiver"
                );
            }
            if let Some(native) = requires.natives.iter().min_by_key(|native| native.index()) {
                return format!("{path}: requires native {native} identity");
            }
            if effects.writes() {
                return format!("{path}: writes state");
            }
            for (inst, target) in local.calls.into_iter().rev() {
                pending.push((target, format!("{path} -> {inst} -> {}", label(target))));
            }
        }
        format!("{root}: missing proof")
    }
}

fn unresolved_operation(graph: &Graph, op: &Op) -> String {
    let (callee, constructor) = match op {
        Op::Javascript(JsOp::Call { callee, .. } | JsOp::CallPack(callee, ..)) => (*callee, false),
        Op::Javascript(
            JsOp::Construct { callee, .. }
            | JsOp::ConstructorCall { callee, .. }
            | JsOp::ConstructPack(callee, ..)
            | JsOp::ConstructPackResolved(callee, ..)
            | JsOp::ConstructorCallPack(callee, ..)
            | JsOp::ConstructResolved { callee, .. },
        ) => (*callee, true),
        _ => return graph.op_text(op),
    };
    match graph.defining_op(callee) {
        Some(Op::Javascript(JsOp::Get(_, name))) if constructor => {
            format!("constructor .{name}: requires runtime constructor identity and effects")
        }
        Some(Op::Javascript(JsOp::Get(_, name))) => {
            format!("method .{name}: requires runtime callee identity and receiver effects")
        }
        Some(Op::Javascript(JsOp::GlobalRead(name) | JsOp::GlobalReadOptional(name)))
            if constructor =>
        {
            format!("constructor {name}: requires runtime constructor identity and effects")
        }
        Some(Op::Javascript(JsOp::GlobalRead(name) | JsOp::GlobalReadOptional(name))) => {
            format!("call {name}: requires runtime callee identity and effects")
        }
        _ if constructor => format!(
            "{}: requires runtime constructor identity and effects",
            graph.op_text(op)
        ),
        _ => format!(
            "{}: requires runtime callee identity and receiver effects",
            graph.op_text(op)
        ),
    }
}

pub(in crate::jit::semantic) fn install_summaries(
    graph: &mut Graph,
    bodies: &Repository,
    notes: &mut Vec<String>,
) {
    let marked: FxHashSet<Value> = graph
        .insts
        .values()
        .filter_map(|data| match data.op {
            Op::Javascript(JsOp::MarkClass(value)) => Some(value),
            _ => None,
        })
        .collect();
    let mut closures = Vec::new();
    let mut roots = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            let Op::Javascript(JsOp::Call { callee, .. }) = graph.insts[inst].op else {
                continue;
            };
            match graph.defining_op(callee) {
                Some(Op::Javascript(JsOp::Closure(_, target, _, false, _, _, _)))
                    if !marked.contains(&callee) =>
                {
                    closures.push((inst, *target));
                    roots.push(*target);
                }
                Some(Op::Javascript(JsOp::GlobalRead(name))) => {
                    roots.extend(bodies.globals.get(name).copied());
                }
                _ => {}
            }
        }
    }
    if roots.is_empty() {
        return;
    }
    let summaries = bodies.effect_summaries(roots);
    let mut installed = FxHashMap::default();
    for (&target, Summary { effects, requires }) in &summaries {
        if effects.unsupported
            || !effects.unresolved.is_empty()
            || effects.writes()
            || !requires.natives.is_empty()
            || !requires.string_splits.is_empty()
        {
            continue;
        }
        let mut required: Vec<_> = requires.identities.iter().cloned().collect();
        required.sort();
        installed.insert(
            target,
            CallSummary {
                memory: effects.memory,
                control: effects.control,
                requires: required,
            },
        );
    }
    for (inst, target) in closures {
        let site = graph.note_inst(inst);
        match installed.get(&target) {
            Some(summary) if summary.requires.is_empty() => {
                notes.push(format!("call {site}: {target} proved closed and read-only"));
            }
            None if !summaries.contains_key(&target) => {}
            _ => {
                notes.push(format!(
                    "call {site}: closed-call proof rejected: {}",
                    bodies.obligation_chain(target, true)
                ));
            }
        }
    }
    graph.summaries = Arc::new(installed);
}

#[derive(Clone, Default, PartialEq, Eq)]
pub(in crate::jit) struct ConditionalPlan {
    pub(in crate::jit::semantic) native_calls: Vec<crate::jit::profile::Site>,
    pub(in crate::jit::semantic) calls: Vec<ConditionalCall>,
}

#[derive(Clone, PartialEq, Eq)]
pub(in crate::jit::semantic) struct ConditionalCall {
    pub(in crate::jit::semantic) inst: Inst,
    pub(in crate::jit::semantic) callee: super::super::Value,
    pub(in crate::jit::semantic) name: String,
    pub(in crate::jit::semantic) target: FuncId,
    pub(in crate::jit::semantic) memory: Memory,
    pub(in crate::jit::semantic) control: Control,
    pub(in crate::jit::semantic) identities: Vec<(String, FuncId)>,
}

pub(in crate::jit::semantic) fn conditional_plan(
    graph: &Graph,
    bodies: &Repository,
    notes: &mut Vec<String>,
) -> ConditionalPlan {
    let mut candidates = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            let Op::Javascript(JsOp::Call { callee, .. }) = graph.insts[inst].op else {
                continue;
            };
            let Some(Op::Javascript(JsOp::GlobalRead(name))) = graph.defining_op(callee) else {
                continue;
            };
            let Some(&target) = bodies.globals.get(name) else {
                continue;
            };
            candidates.push((inst, callee, name.clone(), target));
        }
    }
    let summaries = bodies.effect_summaries(candidates.iter().map(|(_, _, _, target)| *target));
    let mut plan = ConditionalPlan::default();
    for (inst, callee, name, target) in candidates {
        let site = graph.note_inst(inst);
        let Some(summary) = summaries.get(&target) else {
            continue;
        };
        let Summary { effects, requires } = summary;
        if effects.unsupported
            || !effects.unresolved.is_empty()
            || effects.writes()
            || !requires.natives.is_empty()
            || !requires.string_splits.is_empty()
        {
            notes.push(format!(
                "call {site}: conditional proof rejected: {}",
                bodies.obligation_chain(target, false)
            ));
            continue;
        }
        let mut required = requires.identities.clone();
        required.insert((name.clone(), target));
        let mut identities: Vec<_> = required.into_iter().collect();
        identities.sort_by(|a, b| a.0.cmp(&b.0));
        let mut assumptions: Vec<_> = requires
            .assumptions
            .iter()
            .map(ToString::to_string)
            .collect();
        assumptions.sort();
        if !assumptions.is_empty() {
            notes.push(format!(
                "call {site}: conditional effects require {}",
                assumptions.join(", ")
            ));
        }
        plan.calls.push(ConditionalCall {
            inst,
            callee,
            name,
            target,
            memory: effects.memory,
            control: effects.control,
            identities,
        });
    }
    plan
}

pub(in crate::jit::semantic) struct Request {
    pub(in crate::jit::semantic) call: Inst,
    pub(in crate::jit::semantic) target: FuncId,
}

pub(in crate::jit::semantic) struct Plan {
    inlines: Vec<Inline>,
}

pub(in crate::jit::semantic) struct Budget {
    remaining: usize,
    charged: usize,
    body_limit: usize,
    sites: FxHashSet<Inst>,
    pub(in crate::jit::semantic) inlines: Vec<Inline>,
}

impl Default for Budget {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Budget {
    pub(in crate::jit::semantic) fn new(multiplier: usize) -> Self {
        Self {
            remaining: SPECIALIZATION_BUDGET.saturating_mul(multiplier),
            charged: 0,
            body_limit: SPECIALIZATION_BODY_INSTS.saturating_mul(multiplier),
            sites: FxHashSet::default(),
            inlines: Vec::new(),
        }
    }

    /// Refunds estimated costs that simplification folded away once inlined bodies met their
    /// call sites.
    pub(in crate::jit::semantic) fn settle(&mut self, growth: usize) {
        self.remaining += self.charged.saturating_sub(growth);
        self.charged = 0;
    }
}

/// Instructions on paths that can return normally; throwing paths do not count.
pub(in crate::jit::semantic) fn cost(graph: &Graph) -> usize {
    let cfg = Cfg::new(graph);
    let throws = |block: super::super::Block| {
        graph.blocks[block]
            .insts
            .iter()
            .any(|&inst| matches!(graph.insts[inst].op, Op::Javascript(JsOp::Throw(_))))
    };
    let mut pending: Vec<_> = cfg
        .rpo
        .iter()
        .copied()
        .filter(|&block| {
            matches!(
                graph.blocks[block].terminator.as_ref().map(|t| &t.kind),
                Some(TerminatorKind::Return(_) | TerminatorKind::ReturnValues(_))
            ) && !throws(block)
        })
        .collect();
    let mut hot = FxHashSet::default();
    while let Some(block) = pending.pop() {
        if !hot.insert(block) {
            continue;
        }
        pending.extend(
            cfg.preds[block]
                .iter()
                .copied()
                .filter(|&pred| !throws(pred)),
        );
    }
    hot.iter()
        .map(|&block| graph.blocks[block].insts.len())
        .sum()
}

impl Plan {
    pub(in crate::jit::semantic) fn is_empty(&self) -> bool {
        self.inlines.is_empty()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(in crate::jit) struct Inline {
    pub(in crate::jit::semantic) context: u64,
    pub(in crate::jit::semantic) call: Inst,
    pub(in crate::jit::semantic) target: FuncId,
    pub(in crate::jit::semantic) arrow: bool,
    pub(in crate::jit::semantic) guard_code: bool,
    pub(in crate::jit::semantic) source_name: Option<(usize, usize)>,
}

fn eligible(graph: &Graph, body_limit: usize, spread: bool) -> bool {
    if graph.generator || graph.has_handlers {
        return false;
    }
    let mut count = 0;
    for block in Cfg::new(graph).rpo {
        if !matches!(
            graph.blocks[block].terminator.as_ref().map(|t| &t.kind),
            Some(TerminatorKind::Jump(_) | TerminatorKind::Branch(..) | TerminatorKind::Return(_))
        ) {
            return false;
        }
        for &inst in &graph.blocks[block].insts {
            count += 1;
            if count > body_limit
                || spread
                    && matches!(
                        graph.insts[inst].op,
                        Op::Javascript(JsOp::Arguments)
                            | Op::ArgumentCount
                            | Op::Javascript(JsOp::CallForward(..))
                    )
                || matches!(
                    graph.insts[inst].op,
                    Op::Javascript(JsOp::Rest(_) | JsOp::Catch)
                )
            {
                return false;
            }
        }
    }
    true
}

pub(super) fn is_method(
    graph: &Graph,
    callee: super::super::Value,
    receiver: super::super::Value,
) -> bool {
    matches!(graph.defining_op(callee), Some(Op::Javascript(JsOp::Get(object, _))) if *object == receiver)
}

pub(in crate::jit::semantic) fn dispatch_methods(
    graph: &mut Graph,
    bodies: &Repository,
    source: usize,
) {
    use super::super::dispatch::{self, Alternative};
    use super::super::guards::Fact;
    if bodies.method_pairs.is_empty() {
        return;
    }
    let calls: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|block| graph.blocks[block].insts.iter().copied())
        .collect();
    for inst in calls {
        let data = graph.insts[inst].clone();
        let Op::Javascript(JsOp::Call {
            callee,
            receiver: Some(receiver),
            ref args,
            target: None,
        }) = data.op
        else {
            continue;
        };
        if !is_method(graph, callee, receiver) {
            continue;
        }
        let site = crate::jit::profile::Site {
            inline_context: 0,
            source: data.source_name.map_or(source, |name| name.0),
            start: data.span.start,
            end: data.span.end,
        };
        let Some(pairs) = bodies
            .method_pairs
            .get(&site)
            .filter(|pairs| pairs.len() > 1)
        else {
            continue;
        };
        let alternatives: Vec<_> = pairs
            .iter()
            .filter_map(|&(code, shape)| {
                let code = FuncId::from_u32(u32::try_from(code).ok()?);
                let body = bodies.body(code)?;
                let arity = body.arity?;
                let lexical_this = bodies.lexical_this(code);
                let target = DirectTarget {
                    code,
                    variant: crate::jit::entries::VariantId::ordinary(u64::from(code.as_u32())),
                    arity,
                    lexical_this,
                };
                Some(Alternative {
                    requirements: vec![(callee, Fact::Code(code)), (receiver, Fact::Shape(shape))],
                    implementation: (target, shape),
                })
            })
            .collect();
        if alternatives.is_empty() {
            continue;
        }
        graph.method_shapes.insert(inst, None);
        dispatch::apply(
            graph,
            inst,
            alternatives,
            |graph, block, (target, shape)| {
                let (call, result) = graph.append(
                    block,
                    data.span,
                    Op::Javascript(JsOp::Call {
                        callee,
                        receiver: Some(receiver),
                        args: args.clone(),
                        target: Some(target),
                    }),
                    true,
                );
                graph.insts[call].source_name = data.source_name;
                graph.insts[call].inline_context = data.inline_context;
                graph.method_shapes.insert(call, Some(shape));
                result
            },
        );
    }
}

fn call_target(
    graph: &Graph,
    bodies: &Repository,
    source: usize,
    call: Inst,
    callee: super::super::Value,
) -> Option<FuncId> {
    let data = &graph.insts[call];
    if graph.method_shapes.get(&call).is_some_and(Option::is_some)
        && let Op::Javascript(JsOp::Call {
            target: Some(target),
            ..
        }) = data.op
    {
        return Some(target.code);
    }
    let site = crate::jit::profile::Site {
        inline_context: 0,
        source: data.source_name.map_or(source, |name| name.0),
        start: data.span.start,
        end: data.span.end,
    };
    if let Some(targets) = bodies.feedback.get(&site) {
        return match targets.as_slice() {
            [target] => Some(*target),
            _ => None,
        };
    }
    match graph.defining_op(callee) {
        Some(Op::Javascript(JsOp::GlobalRead(name))) => bodies.globals.get(name).copied(),
        Some(Op::Javascript(JsOp::Closure(_, target, _, false, _, _, _))) => Some(*target),
        _ => None,
    }
}

pub(in crate::jit::semantic) fn direct_calls(
    graph: &mut Graph,
    bodies: &Repository,
    source: usize,
) {
    let mut targets = Vec::new();
    for block in Cfg::new(graph).rpo {
        if graph.cold.contains(&block) {
            continue;
        }
        for &call in &graph.blocks[block].insts {
            let Op::Javascript(JsOp::Call { callee, .. }) = graph.insts[call].op else {
                continue;
            };
            let Some(target) = call_target(graph, bodies, source, call, callee) else {
                continue;
            };
            let Some(body) = bodies.bodies.get(&target) else {
                continue;
            };
            let Some(arity) = body.graph.arity else {
                continue;
            };
            let lexical_this = body.arrow && super::inlining::uses_this(&body.graph);
            targets.push((
                call,
                DirectTarget {
                    variant: crate::jit::entries::VariantId::ordinary(target.as_u32() as u64),
                    code: target,
                    arity,
                    lexical_this,
                },
            ));
        }
    }
    for (call, direct) in targets {
        if let Op::Javascript(JsOp::Call { target, .. }) = &mut graph.insts[call].op {
            *target = Some(direct);
        }
    }
}

pub(in crate::jit::semantic) fn requests(
    graph: &Graph,
    bodies: &Repository,
    source: usize,
    budget: &Budget,
) -> Vec<Request> {
    if budget.body_limit == 0 {
        return Vec::new();
    }
    let mut requests = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &call in &graph.blocks[block].insts {
            if graph.method_shapes.get(&call) == Some(&None) {
                continue;
            }
            let callee = match &graph.insts[call].op {
                Op::Javascript(
                    JsOp::Call { callee, .. }
                    | JsOp::CallPack(callee, ..)
                    | JsOp::ConstructorCall { callee, .. }
                    | JsOp::ConstructorCallPack(callee, ..),
                ) => callee,
                _ => continue,
            };
            let Some(target) = call_target(graph, bodies, source, call, *callee) else {
                continue;
            };
            if bodies.body(target).is_some_and(|body| {
                eligible(
                    body,
                    budget.body_limit,
                    matches!(
                        graph.insts[call].op,
                        Op::Javascript(JsOp::CallPack(..) | JsOp::ConstructorCallPack(..))
                    ),
                )
            }) {
                requests.push(Request { call, target });
            }
        }
    }
    requests
}

#[cfg(test)]
pub(in crate::jit::semantic) fn plan(
    graph: &Graph,
    bodies: &Repository,
    source: usize,
    requests: &[Request],
) -> Plan {
    plan_round(graph, bodies, source, requests, &mut Budget::default())
}

pub(in crate::jit::semantic) fn plan_round(
    graph: &Graph,
    bodies: &Repository,
    source: usize,
    requests: &[Request],
    budget: &mut Budget,
) -> Plan {
    let mut inlines = Vec::new();
    for request in requests {
        let Some(body) = bodies.body(request.target).filter(|body| {
            eligible(
                body,
                budget.body_limit,
                matches!(
                    graph.insts[request.call].op,
                    Op::Javascript(JsOp::CallPack(..) | JsOp::ConstructorCallPack(..))
                ),
            )
        }) else {
            continue;
        };
        let callee = match &graph.insts[request.call].op {
            Op::Javascript(
                JsOp::Call { callee, .. }
                | JsOp::CallPack(callee, ..)
                | JsOp::ConstructorCall { callee, .. }
                | JsOp::ConstructorCallPack(callee, ..),
            ) => callee,
            _ => continue,
        };
        let cost = cost(body);
        if cost > budget.remaining || budget.sites.contains(&request.call) {
            continue;
        }
        budget.sites.insert(request.call);
        budget.remaining -= cost;
        budget.charged += cost;
        let body_source = bodies.bodies[&request.target].source;
        let source_name = if body_source == source {
            None
        } else {
            bodies
                .sources
                .get(&body_source)
                .map(|&length| (body_source, length))
        };
        use std::hash::{Hash, Hasher};
        let call = &graph.insts[request.call];
        let mut context = std::hash::DefaultHasher::new();
        (
            call.inline_context,
            call.source_name.map_or(source, |name| name.0),
            call.span.start,
            call.span.end,
            request.target,
        )
            .hash(&mut context);
        if let Some(Some(shape)) = graph.method_shapes.get(&request.call) {
            shape.hash(&mut context);
        }
        inlines.push(Inline {
            context: context.finish().max(1),
            call: request.call,
            target: request.target,
            source_name,
            arrow: bodies.bodies[&request.target].arrow,
            guard_code: !graph.method_shapes.get(&request.call).is_some_and(Option::is_some) && !matches!(graph.defining_op(*callee), Some(Op::Javascript(JsOp::Closure(_, code, _, false, _, _, _)))
                if *code == request.target
                && !graph.insts.values().any(|data| matches!(data.op, Op::Javascript(JsOp::MarkClass(value)) if value == *callee))),
        });
    }
    budget.inlines.extend(inlines.iter().cloned());
    Plan { inlines }
}

pub(in crate::jit::semantic) fn apply(
    graph: &mut Graph,
    bodies: &Repository,
    _source: usize,
    plan: Plan,
    notes: &mut Vec<String>,
) -> Result<()> {
    for inline in plan.inlines {
        let body = bodies.body(inline.target).unwrap();
        let site = graph.note_inst(inline.call);
        super::inlining::apply(graph, body, &inline);
        graph.verify()?;
        notes.push(format!(
            "call {site}: inline {} with {} code check(s)",
            inline.target,
            usize::from(inline.guard_code)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::ValueType;
    use super::*;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;
    use oxc_syntax::operator::BinaryOperator;

    fn fixture(source: &str) -> (FunctionState, Repository) {
        let allocator = Allocator::default();
        let parsed = Parser::new(&allocator, source, SourceType::script()).parse();
        assert!(!parsed.diagnostics.has_errors());
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let mut bodies = Repository::default();
        let mut caller = None;
        for (index, statement) in parsed.program.body.iter().enumerate() {
            let Statement::FunctionDeclaration(function) = statement else {
                panic!()
            };
            let graph = super::super::super::build::lower(
                function,
                semantic.scoping(),
                &FxHashSet::default(),
            )
            .unwrap();
            let canonical = super::super::super::canonicalize_graph(graph, true).unwrap();
            let name = function.id.as_ref().unwrap().name.as_str();
            if name == "main" {
                caller = Some(canonical);
            } else {
                bodies.insert(0, name, FuncId::from_u32(index as u32), canonical);
            }
        }
        (caller.unwrap(), bodies)
    }

    fn operations(graph: &Graph) -> Vec<&Op> {
        Cfg::new(graph)
            .rpo
            .into_iter()
            .flat_map(|block| &graph.blocks[block].insts)
            .map(|&inst| &graph.insts[inst].op)
            .collect()
    }

    #[test]
    fn library_invocation_exposes_packed_target_for_inlining() {
        let (mut canonical, bodies) = fixture(
            "function target(a,b){return this.offset+a+b;} function main(receiver,args){return $callSpread(target,receiver,args);}",
        );
        super::super::super::lower_intrinsics(&mut canonical).unwrap();
        let mut graph = canonical.graph;
        assert!(
            operations(&graph)
                .iter()
                .any(|op| matches!(op, Op::Javascript(JsOp::CallPack(..))))
        );
        assert!(!operations(&graph).iter().any(
            |op| matches!(op, Op::Javascript(JsOp::GlobalRead(name)) if name == "$callSpread")
        ));
        let requests = requests(&graph, &bodies, 0, &Budget::default());
        assert_eq!(requests.len(), 1);
        let plan = plan(&graph, &bodies, 0, &requests);
        apply(&mut graph, &bodies, 0, plan, &mut Vec::new()).unwrap();
        assert!(
            operations(&graph)
                .iter()
                .any(|op| matches!(op, Op::Javascript(JsOp::PackTake(..))))
        );
        assert!(
            operations(&graph)
                .iter()
                .any(|op| matches!(op, Op::Javascript(JsOp::Get(_, name)) if name == "offset"))
        );
        graph.verify().unwrap();
    }

    #[test]
    fn recursive_effects_reach_a_fixed_point_and_keep_obligations() {
        let (_, bodies) = fixture(
            "function first(n){if(n)return second(n-1);return state.value;} function second(n){if(n)return first(n-1);state.value=1;return Math.floor(n);} function main(){return first(2);}",
        );
        let summaries = bodies.effect_summaries([FuncId::from_u32(0)]);
        for code in [FuncId::from_u32(0), FuncId::from_u32(1)] {
            let summary = &summaries[&code];
            assert!(summary.effects.unresolved.is_empty());
            assert!(summary.effects.globals_read.contains("state"));
            assert!(summary.effects.properties_read.contains("value"));
            assert!(summary.effects.properties_written.contains("value"));
            assert_eq!(summary.requires.identities.len(), 2);
            assert_eq!(summary.requires.natives.len(), 1);
        }
    }

    #[test]
    fn string_leaf_versions_only_after_split_native_guard() {
        let (canonical, bodies) = fixture(
            "function Terrain(x,y){this.x=x;this.y=y;} function main(t){return typeof t==='string' ? new Terrain(...t.split('|')) : t;}",
        );
        let analysis =
            super::super::super::optimize_canonical(canonical, true, Some((&bodies, 0))).unwrap();
        let listing = analysis.to_string();
        assert!(
            listing.contains("string split continuation versioned after native identity check"),
            "{listing}"
        );
        assert!(listing.contains("is_string_split"));
        assert!(listing.contains("string_split"));
        assert!(listing.contains("constructor_call_pack"));
        analysis.graph.verify().unwrap();
    }

    #[test]
    fn split_leaf_keeps_default_arguments_in_the_typed_copy() {
        let (canonical, bodies) = fixture(
            "function Terrain(texture,template=undefined){if(texture===undefined)throw new Error('missing');this.texture=texture;this.template=template;} function main(t){return typeof t==='string' ? new Terrain(...t.split('|')) : t;}",
        );
        let analysis =
            super::super::super::optimize_canonical(canonical, true, Some((&bodies, 0))).unwrap();
        let graph = &analysis.graph;
        let cfg = Cfg::new(graph);
        assert!(
            cfg.rpo.iter().any(
                |&block| graph.blocks[block].insts.iter().any(|&inst| matches!(
                    graph.insts[inst].op,
                    Op::Javascript(JsOp::StringSplitParts(..))
                ))
            ),
            "{analysis}"
        );
        let setters: Vec<_> = cfg
            .rpo
            .iter()
            .flat_map(|&block| &graph.blocks[block].insts)
            .filter_map(|&inst| match graph.insts[inst].op {
                Op::Javascript(JsOp::FillPropertyStore(_, cache, ref name, value, _))
                    if name == "template" && !graph.cold.contains(&graph.def_block(cache)) =>
                {
                    Some(value)
                }
                _ => None,
            })
            .collect();
        assert!(!setters.is_empty(), "{analysis}");
        for value in setters {
            let types = analysis.facts.of(value);
            assert!(
                types.within(ValueType::STRING.union(ValueType::UNDEFINED)),
                "{analysis}"
            );
            assert!(types.intersects(ValueType::STRING));
            assert!(types.intersects(ValueType::UNDEFINED));
        }
        graph.verify().unwrap();
    }

    #[test]
    fn split_effects_require_native_identity_before_call_proof() {
        let (canonical, bodies) =
            fixture("function leaf(t){return t.split('|');} function main(t){return leaf(t);}");
        let summaries = bodies.effect_summaries([FuncId::from_u32(0)]);
        let summary = &summaries[&FuncId::from_u32(0)];
        assert!(summary.effects.unresolved.is_empty());
        assert_eq!(summary.requires.string_splits.len(), 1);
        assert!(summary.effects.control.ordered);
        assert!(summary.effects.control.throws);
        let mut notes = Vec::new();
        assert!(
            conditional_plan(&canonical.graph, &bodies, &mut notes)
                .calls
                .is_empty()
        );
        assert!(
            notes
                .iter()
                .any(|note| note.contains("native String.split identity"))
        );
    }

    #[test]
    fn recursive_terrain_method_rejection_names_its_obligation() {
        let (canonical, mut bodies) = fixture(
            "function createTerrain(t){if(typeof t==='string')return t.split('|');return t.map(createTerrain);} function main(t){return createTerrain(t);}",
        );
        Arc::make_mut(bodies.bodies.get_mut(&FuncId::from_u32(0)).unwrap())
            .graph
            .name = "<function>".into();
        let mut notes = Vec::new();
        let plan = conditional_plan(&canonical.graph, &bodies, &mut notes);
        assert!(plan.calls.is_empty());
        let summaries = bodies.effect_summaries([FuncId::from_u32(0)]);
        assert_eq!(
            summaries[&FuncId::from_u32(0)].requires.string_splits.len(),
            1
        );
        assert!(
            notes.iter().any(|note| note.contains("createTerrain")
                && note.contains("method .map")
                && note.contains("runtime callee identity and receiver effects")),
            "{notes:?}"
        );
    }

    #[test]
    fn global_call_proof_rejects_writes_to_guarded_identity() {
        let (canonical, bodies) = fixture(
            "function first(n){if(n)return first(n-1);return 0;} function main(n){first=other;return first(n);}",
        );
        let mut graph = canonical.graph;
        let plan = conditional_plan(&graph, &bodies, &mut Vec::new());
        let original_entry = graph.entry;
        let original_blocks = graph.blocks.len();
        let mut notes = Vec::new();
        assert!(
            super::super::regions::version(&mut graph, &plan, None, &mut notes)
                .unwrap()
                .is_empty()
        );
        assert_eq!(graph.entry, original_entry);
        assert_eq!(graph.blocks.len(), original_blocks);
        assert!(notes.iter().any(|note| note.contains("reenters")));
    }

    fn closed_calls(graph: &Graph) -> Vec<Inst> {
        let facts = Facts::compute(graph);
        Cfg::new(graph)
            .rpo
            .into_iter()
            .flat_map(|block| graph.blocks[block].insts.clone())
            .filter(|&inst| matches!(graph.insts[inst].op, Op::Javascript(JsOp::Call { .. })))
            .filter(|&inst| !graph.inst_contract(inst, &facts).effects.memory.reenters())
            .collect()
    }

    #[test]
    fn recursive_global_calls_are_guarded_as_a_group() {
        let (canonical, bodies) = fixture(
            "function first(n){if(n)return second(n-1);return state.value;} function second(n){if(n)return first(n-1);return state.value;} function main(n){return first(n);}",
        );
        let mut graph = canonical.graph;
        install_summaries(&mut graph, &bodies, &mut Vec::new());
        let plan = conditional_plan(&graph, &bodies, &mut Vec::new());
        assert_eq!(plan.calls.len(), 1);
        assert_eq!(plan.calls[0].identities.len(), 2);
        let regions =
            super::super::regions::version(&mut graph, &plan, None, &mut Vec::new()).unwrap();
        assert_eq!(regions.len(), 1);
        let region = &regions[0];
        let closed = closed_calls(&graph);
        assert_eq!(closed.len(), 1);
        assert!(closed.iter().all(|inst| {
            graph.regions[*region]
                .blocks
                .contains(&graph.insts[*inst].block)
        }));
        assert_eq!(
            operations(&graph)
                .iter()
                .filter(|op| matches!(op, Op::Javascript(JsOp::IsCode(..))))
                .count(),
            2
        );
        graph.verify().unwrap();
    }

    #[test]
    fn conditional_calls_merge_with_loop_and_native_assumptions() {
        let (canonical, bodies) = fixture(
            "function first(n){if(n)return first(n-1);return state.value;} function main(xs,n){let sum=0;for(let i=0;i<n;++i)sum+=Math.floor(xs[i]);return sum+first(n);}",
        );
        let mut graph = canonical.graph;
        install_summaries(&mut graph, &bodies, &mut Vec::new());
        let plan = conditional_plan(&graph, &bodies, &mut Vec::new());
        let generic = plan.calls[0].inst;
        let mut notes = Vec::new();
        let regions = super::super::regions::version(&mut graph, &plan, None, &mut notes).unwrap();
        assert_eq!(regions.len(), 1);
        let closed = closed_calls(&graph);
        assert!(closed.contains(&generic));
        assert_eq!(closed.len(), 1);
        assert!(
            notes
                .iter()
                .any(|note| note.starts_with("function: combined entry guards:"))
        );
        assert!(
            notes
                .iter()
                .any(|note| note == "function: 1 global code checks prove 1 read-only call(s)")
        );
        assert!(
            operations(&graph)
                .iter()
                .any(|op| matches!(op, Op::Javascript(JsOp::CallNative(..))))
        );
        assert_eq!(
            operations(&graph)
                .iter()
                .filter(|op| matches!(op, Op::Javascript(JsOp::IsCode(..))))
                .count(),
            1
        );
        graph.verify().unwrap();
    }

    #[test]
    fn conditional_loop_guards_only_its_own_calls() {
        let (canonical, bodies) = fixture(
            "function first(n){if(n)return first(n-1);return 1;} function second(n){return n;} function main(n){let sum=0;for(let i=0;i<n;++i)sum+=first(i);second=other;return sum+second(n);}",
        );
        let mut graph = canonical.graph;
        install_summaries(&mut graph, &bodies, &mut Vec::new());
        let plan = conditional_plan(&graph, &bodies, &mut Vec::new());
        assert_eq!(plan.calls.len(), 2);
        let mut notes = Vec::new();
        let regions = super::super::regions::version(&mut graph, &plan, None, &mut notes).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(closed_calls(&graph).len(), 1);
        let probes: Vec<_> = operations(&graph)
            .into_iter()
            .filter_map(|op| match op {
                Op::Javascript(JsOp::ProbeGlobal(name)) => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert!(!probes.is_empty());
        assert!(probes.iter().all(|name| *name == "first"));
        assert!(
            notes
                .iter()
                .any(|note| note.starts_with("loop at ") && note.contains("read-only call(s)"))
        );
        graph.verify().unwrap();
    }

    #[test]
    fn conditional_proofs_survive_without_typed_storage_speculation() {
        let (canonical, bodies) = fixture(
            "function first(n){if(n)return first(n-1);return 1;} function main(rows){let sum=0;for(let y=0;y<2;++y){const row=rows[y];for(let x=0;x<2;++x)sum+=row[x]+first(x);}return sum;}",
        );
        let mut graph = canonical.graph;
        install_summaries(&mut graph, &bodies, &mut Vec::new());
        let plan = conditional_plan(&graph, &bodies, &mut Vec::new());
        let regions =
            super::super::regions::version(&mut graph, &plan, None, &mut Vec::new()).unwrap();
        assert!(!regions.is_empty());
        assert!(!closed_calls(&graph).is_empty());
        graph.verify().unwrap();
    }

    #[test]
    fn conditional_proof_rejects_a_changed_callee() {
        let (canonical, bodies) = fixture(
            "function first(n){if(n)return first(n-1);return 1;} function main(n){return first(n);}",
        );
        let mut graph = canonical.graph;
        let plan = conditional_plan(&graph, &bodies, &mut Vec::new());
        let super::super::super::ValueDef::Result(inst, _) = graph.values[plan.calls[0].callee]
        else {
            panic!();
        };
        graph.set_op(inst, Op::Javascript(JsOp::GlobalRead("other".into())));
        let regions =
            super::super::regions::version(&mut graph, &plan, None, &mut Vec::new()).unwrap();
        assert!(regions.is_empty());
        assert!(closed_calls(&graph).is_empty());
        graph.verify().unwrap();
    }

    #[test]
    fn recursive_global_group_with_descendant_write_is_rejected() {
        let (canonical, bodies) = fixture(
            "function first(n){if(n)return second(n-1);return 0;} function second(n){if(n)return first(n-1);first=other;return 0;} function main(n){return first(n);}",
        );
        assert!(
            conditional_plan(&canonical.graph, &bodies, &mut Vec::new())
                .calls
                .is_empty()
        );
    }

    #[test]
    fn unresolved_descendant_blocks_exact_closure_proof() {
        let (canonical, bodies) =
            fixture("function helper(x){return unknown(x);} function main(x){return helper(x);}");
        let mut graph = canonical.graph;
        let request = requests(&graph, &bodies, 0, &Budget::default())
            .pop()
            .unwrap();
        let Op::Javascript(JsOp::Call { callee, .. }) = graph.insts[request.call].op else {
            panic!()
        };
        let super::super::super::ValueDef::Result(def, _) = graph.values[callee] else {
            panic!()
        };
        graph.set_op(
            def,
            Op::Javascript(JsOp::Closure(
                graph.insts[def].span,
                request.target,
                false,
                false,
                false,
                vec![],
                None,
            )),
        );
        install_summaries(&mut graph, &bodies, &mut Vec::new());
        assert!(!closed_calls(&graph).contains(&request.call));
    }

    #[test]
    fn exact_read_only_closure_call_has_a_local_proof() {
        let (canonical, bodies) =
            fixture("function helper(x){return x.value;} function main(x){return helper(x);}");
        let mut graph = canonical.graph;
        let request = requests(&graph, &bodies, 0, &Budget::default())
            .pop()
            .unwrap();
        let Op::Javascript(JsOp::Call { callee, .. }) = graph.insts[request.call].op else {
            panic!()
        };
        let super::super::super::ValueDef::Result(def, _) = graph.values[callee] else {
            panic!()
        };
        graph.set_op(
            def,
            Op::Javascript(JsOp::Closure(
                graph.insts[def].span,
                request.target,
                false,
                false,
                false,
                vec![],
                None,
            )),
        );
        install_summaries(&mut graph, &bodies, &mut Vec::new());
        assert!(closed_calls(&graph).contains(&request.call));
    }

    #[test]
    fn inlined_polymorphic_gets_relearn_local_shapes() {
        use crate::jit::profile::{PropertyAccess, ShapeAccess, Site};
        use crate::jit::variants::{Fact as InputFact, Input};
        for (body, invalidated) in [
            ("return readX(o);", false),
            ("mutate(o);return readX(o);", true),
            ("o.extra=1;return readX(o);", true),
        ] {
            for shape in [None, Some(10), Some(20), Some(30)] {
                let (canonical, mut bodies) = fixture(&format!(
                    "function readX(o){{return o.x;}} function main(o,mutate){{{body}}}"
                ));
                let (_, callee) = bodies.emission(FuncId::from_u32(0)).unwrap();
                for block in Cfg::new(&callee.graph).rpo {
                    for &inst in &callee.graph.blocks[block].insts {
                        let data = &callee.graph.insts[inst];
                        if matches!(data.op, Op::Javascript(JsOp::Get(..))) {
                            bodies.shape_feedback.insert(
                                (
                                    Site {
                                        inline_context: 0,
                                        source: 0,
                                        start: data.span.start,
                                        end: data.span.end,
                                    },
                                    PropertyAccess::Get,
                                ),
                                ShapeAccess::PolymorphicRead {
                                    entries: [
                                        ReadLayout::own(10, 0),
                                        ReadLayout::own(20, 1),
                                        ReadLayout::default(),
                                        ReadLayout::default(),
                                    ],
                                    len: 2,
                                },
                            );
                        }
                    }
                }
                let planned = super::super::super::plan_specialization(
                    FunctionState::new(canonical.graph.clone(), canonical.notes.clone(), true),
                    true,
                    Some((&bodies, 0)),
                )
                .unwrap();
                assert_eq!(planned.plan.inlines.len(), 1);
                assert!(planned.plan.shapes.polymorphic_reads().is_empty());
                for data in planned.graph.insts.values() {
                    if !matches!(&data.op, Op::Javascript(JsOp::Get(_, name)) if name == "x") {
                        continue;
                    }
                    assert_ne!(data.inline_context, 0);
                    bodies.shape_feedback.insert(
                        (
                            Site {
                                inline_context: data.inline_context,
                                source: data.source_name.map_or(0, |name| name.0),
                                start: data.span.start,
                                end: data.span.end,
                            },
                            PropertyAccess::Get,
                        ),
                        ShapeAccess::PolymorphicRead {
                            entries: [
                                ReadLayout::own(10, 0),
                                ReadLayout::own(20, 1),
                                ReadLayout::default(),
                                ReadLayout::default(),
                            ],
                            len: 2,
                        },
                    );
                }
                let mut planned =
                    super::super::super::plan_specialization(canonical, true, Some((&bodies, 0)))
                        .unwrap();
                assert_eq!(planned.plan.shapes.polymorphic_reads().len(), 1);
                if let Some(shape) = shape {
                    super::super::call_variants::assume(
                        &mut planned.graph,
                        &vec![(Input::Argument(0), InputFact::Shape(shape))],
                    );
                }
                let optimized = super::super::super::lower_planned(
                    planned,
                    super::super::super::Versioning::Generic,
                    false,
                )
                .unwrap();
                optimized.graph.verify().unwrap();
                assert!(optimized.graph.regions.is_empty());
                assert!(optimized.graph.entry_guards.is_empty());
                let operations = operations(&optimized.graph);
                assert_eq!(
                    operations
                        .iter()
                        .filter(|op| matches!(op, Op::Javascript(JsOp::ShapedGet(..))))
                        .count(),
                    if invalidated || shape.is_none() {
                        2
                    } else {
                        usize::from(shape != Some(30))
                    }
                );
                assert_eq!(
                    operations
                        .iter()
                        .filter(|op| matches!(op, Op::Javascript(JsOp::IsShape(..))))
                        .count(),
                    if invalidated || shape.is_none() { 2 } else { 0 }
                );
                if let Some(shape) = shape.filter(|shape| *shape != 30 && !invalidated) {
                    assert!(operations.iter().any(|op| matches!(op, Op::Javascript(JsOp::ShapedGet(_, _, index)) if *index == u32::from(shape == 20))));
                    assert!(
                        !operations.iter().any(
                            |op| matches!(op, Op::Javascript(JsOp::FillPropertyCache(_, _, name)) if name == "x")
                        )
                    );
                } else {
                    assert!(
                        operations.iter().any(
                            |op| matches!(op, Op::Javascript(JsOp::FillPropertyCache(_, _, name)) if name == "x")
                        )
                    );
                }
            }
        }
    }

    #[test]
    fn inlined_property_feedback_is_specific_to_the_call() {
        use crate::jit::profile::{PropertyAccess, ShapeAccess, Site};

        let (canonical, mut bodies) =
            fixture("function read(o){return o.x;} function main(a,b){return read(a)+read(b);}");
        let (_, callee) = bodies.emission(FuncId::from_u32(0)).unwrap();
        let read = callee
            .graph
            .insts
            .values()
            .find(|data| matches!(data.op, Op::Javascript(JsOp::Get(..))))
            .unwrap();
        let original = Site {
            inline_context: 0,
            source: 0,
            start: read.span.start,
            end: read.span.end,
        };
        bodies
            .shape_feedback
            .insert((original, PropertyAccess::Get), ShapeAccess::Megamorphic);
        let planned = super::super::super::plan_specialization(
            FunctionState::new(canonical.graph.clone(), canonical.notes.clone(), true),
            true,
            Some((&bodies, 0)),
        )
        .unwrap();
        let contexts: Vec<_> = planned
            .plan
            .inlines
            .iter()
            .map(|inline| inline.context)
            .collect();
        assert_eq!(contexts.len(), 2);
        assert_ne!(contexts[0], contexts[1]);
        assert!(contexts.iter().all(|context| *context != 0));
        assert!(planned.plan.shapes.megamorphic_sites().is_empty());
        for (&inline_context, shape) in contexts.iter().zip([10, 20]) {
            bodies.shape_feedback.insert(
                (
                    Site {
                        inline_context,
                        ..original
                    },
                    PropertyAccess::Get,
                ),
                ShapeAccess::Read(ReadLayout::own(shape, 0)),
            );
        }
        let planned =
            super::super::super::plan_specialization(canonical, true, Some((&bodies, 0))).unwrap();
        assert_eq!(
            planned
                .plan
                .inlines
                .iter()
                .map(|inline| inline.context)
                .collect::<Vec<_>>(),
            contexts
        );
        for (&context, shape) in contexts.iter().zip([10, 20]) {
            assert!(planned.plan.shapes.accesses().contains(&(
                Site {
                    inline_context: context,
                    ..original
                },
                PropertyAccess::Get,
                ReadLayout::own(shape, 0)
            )));
        }
        assert_eq!(
            bodies.shape_feedback(original, PropertyAccess::Get),
            Some(ShapeAccess::Megamorphic)
        );
    }

    #[test]
    fn decision_records_include_nested_and_bounded_recursive_inlines() {
        for (source, count) in [
            (
                "function leaf(x){return x*x;} function middle(x){return leaf(x);} function main(x){return middle(x);}",
                2,
            ),
            (
                "function recurse(n){if(n)return recurse(n-1);return 0;} function main(n){return recurse(n);}",
                SPECIALIZATION_ROUNDS + 1,
            ),
            (
                "function C(v){this.x=v;} function make(v){return new C(v);} function main(a,b){return make(a).x+make(b).x;}",
                4,
            ),
        ] {
            let (canonical, bodies) = fixture(source);
            let first = super::super::super::plan_specialization(
                FunctionState::new(canonical.graph.clone(), canonical.notes.clone(), true),
                true,
                Some((&bodies, 0)),
            )
            .unwrap();
            let second =
                super::super::super::plan_specialization(canonical, true, Some((&bodies, 0)))
                    .unwrap();
            assert_eq!(first.plan.inlines.len(), count);
            assert_eq!(
                first
                    .plan
                    .inlines
                    .iter()
                    .map(|inline| inline.context)
                    .collect::<FxHashSet<_>>()
                    .len(),
                count,
            );
            assert!(first.plan.inlines == second.plan.inlines);
            let analysis = super::super::super::lower_planned(
                first,
                super::super::super::Versioning::Speculate {
                    fallback: None,
                    continuations: true,
                },
                false,
            )
            .unwrap();
            assert!(analysis.plan.inlines == second.plan.inlines);
            analysis.graph.verify().unwrap();
        }
    }

    #[test]
    fn feedback_supplies_guarded_targets_for_property_calls() {
        let (canonical, mut bodies) = fixture(
            "function square(x){return x*x;} function main(x){return Math.square(x)+Math.square(x+1);}",
        );
        let mut graph = canonical.graph;
        assert!(requests(&graph, &bodies, 0, &Budget::default()).is_empty());
        for data in graph.insts.values() {
            if matches!(data.op, Op::Javascript(JsOp::Call { .. })) {
                bodies.feedback.insert(
                    crate::jit::profile::Site {
                        inline_context: 0,
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    vec![FuncId::from_u32(0)],
                );
            }
        }
        let requests = requests(&graph, &bodies, 0, &Budget::default());
        assert_eq!(requests.len(), 2);
        let plan = plan(&graph, &bodies, 0, &requests);
        assert!(plan.inlines.iter().all(|inline| inline.guard_code));
        let mut notes = Vec::new();
        apply(&mut graph, &bodies, 0, plan, &mut notes).unwrap();
        assert_eq!(notes.len(), 2);
        assert_eq!(
            operations(&graph)
                .iter()
                .filter(|op| matches!(op, Op::Javascript(JsOp::IsCode(..))))
                .count(),
            2
        );
        assert_eq!(
            operations(&graph)
                .iter()
                .filter(|op| matches!(
                    op,
                    Op::Javascript(JsOp::Binary(BinaryOperator::Multiplication, ..))
                ))
                .count(),
            2
        );
    }

    #[test]
    fn polymorphic_feedback_keeps_calls_out_of_inlining() {
        for call in ["square(x)", "object.square(x)"] {
            let (canonical, mut bodies) = fixture(&format!(
                "function square(x){{return x*x;}} function other(x){{return x+1;}} function main(x){{return {call};}}"
            ));
            let mut graph = canonical.graph;
            for data in graph.insts.values() {
                if matches!(data.op, Op::Javascript(JsOp::Call { .. })) {
                    bodies.feedback.insert(
                        crate::jit::profile::Site {
                            inline_context: 0,
                            source: 0,
                            start: data.span.start,
                            end: data.span.end,
                        },
                        vec![FuncId::from_u32(0), FuncId::from_u32(1)],
                    );
                }
            }
            assert!(requests(&graph, &bodies, 0, &Budget::default()).is_empty());
            direct_calls(&mut graph, &bodies, 0);
            assert!(
                operations(&graph)
                    .iter()
                    .any(|op| matches!(op, Op::Javascript(JsOp::Call { target: None, .. })))
            );
            let variants = super::super::call_variants::apply(&mut graph, &bodies, 0).unwrap();
            for code in [0, 1] {
                assert!(variants.iter().any(|(_, variant)| {
                    *variant == crate::jit::entries::VariantId::ordinary(code)
                }));
            }
            graph.verify().unwrap();
        }
    }

    #[test]
    fn polymorphic_method_receivers_offer_shapes_to_callees() {
        use crate::jit::layout::ReadLayout;
        use crate::jit::profile::{PropertyAccess, ShapeAccess, Site};
        use crate::jit::variants::{Fact as InputFact, Input};

        let (canonical, mut bodies) = fixture(
            "function read(){return this.x;} function constant(){return 1;} function main(o){return o.method();}",
        );
        let mut graph = canonical.graph;
        let site = |data: &super::super::super::InstData| Site {
            inline_context: 0,
            source: 0,
            start: data.span.start,
            end: data.span.end,
        };
        let read = FuncId::from_u32(0);
        let constant = FuncId::from_u32(1);
        let callee_site = bodies
            .body(read)
            .unwrap()
            .insts
            .values()
            .find(|data| matches!(data.op, Op::Javascript(JsOp::Get(..))))
            .map(site)
            .unwrap();
        let layouts = |first, second| ShapeAccess::PolymorphicRead {
            entries: [
                ReadLayout::own(first, 0),
                ReadLayout::own(second, 1),
                ReadLayout::default(),
                ReadLayout::default(),
            ],
            len: 2,
        };
        bodies
            .shape_feedback
            .insert((callee_site, PropertyAccess::Get), layouts(10, 30));
        for data in graph.insts.values() {
            match data.op {
                Op::Javascript(JsOp::Get(..)) => {
                    bodies
                        .shape_feedback
                        .insert((site(data), PropertyAccess::Get), layouts(10, 20));
                }
                Op::Javascript(JsOp::Call { .. }) => {
                    bodies.feedback.insert(site(data), vec![read, constant]);
                    bodies
                        .method_pairs
                        .insert(site(data), vec![(0, 10), (1, 20)]);
                }
                _ => {}
            }
        }
        dispatch_methods(&mut graph, &bodies, 0);
        let mut inlined = graph.clone();
        let requests = requests(&inlined, &bodies, 0, &Budget::default());
        let plan = plan(&inlined, &bodies, 0, &requests);
        assert_eq!(plan.inlines.len(), 2);
        assert!(plan.inlines.iter().all(|inline| !inline.guard_code));
        apply(&mut inlined, &bodies, 0, plan, &mut Vec::new()).unwrap();
        assert!(
            operations(&inlined)
                .iter()
                .any(|op| matches!(op, Op::Javascript(JsOp::Call { target: None, .. })))
        );
        let selected = super::super::call_variants::apply(&mut graph, &bodies, 0).unwrap();
        let variants: Vec<_> = selected
            .into_iter()
            .filter_map(|(_, id)| bodies.variants.get(id))
            .collect();
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].function, 0);
        assert_eq!(
            variants[0].requirements,
            [(Input::This, InputFact::Shape(10))]
        );
        assert!(
            operations(&graph)
                .iter()
                .any(|op| matches!(op, Op::Javascript(JsOp::IsShape(_, 10))))
        );
    }

    #[test]
    fn varying_method_guards_exit_to_the_original_continuation() {
        let (canonical, mut bodies) = fixture(
            "function method(x){return Math.floor(x);} function main(objects,n){let sum=0;for(let i=0;i<n;++i)sum+=objects[i].method(i);return sum;}",
        );
        for data in canonical.graph.insts.values() {
            if let Op::Javascript(JsOp::Call { callee, .. }) = data.op
                && matches!(canonical.graph.defining_op(callee), Some(Op::Javascript(JsOp::Get(_, name))) if name == "method")
            {
                bodies.feedback.insert(
                    crate::jit::profile::Site {
                        inline_context: 0,
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    vec![FuncId::from_u32(0)],
                );
            }
        }
        for fallback in [None, Some(0x1234)] {
            let planned = super::super::super::plan_specialization(
                FunctionState::new(canonical.graph.clone(), canonical.notes.clone(), true),
                true,
                Some((&bodies, 0)),
            )
            .unwrap();
            let analysis = super::super::super::lower_planned(
                planned,
                super::super::super::Versioning::Speculate {
                    fallback,
                    continuations: true,
                },
                false,
            )
            .unwrap();
            let graph = &analysis.graph;
            graph.verify().unwrap();
            let mut fallbacks = Vec::new();
            for block in Cfg::new(graph).rpo {
                if let Some(super::super::super::Terminator {
                    kind: TerminatorKind::Branch(condition, _, otherwise),
                    ..
                }) = &graph.blocks[block].terminator
                    && matches!(graph.defining_op(*condition), Some(Op::Javascript(JsOp::IsCode(_, code))) if *code == FuncId::from_u32(0))
                {
                    fallbacks.push(otherwise.block);
                }
            }
            assert_eq!(fallbacks.len(), 1);
            assert!(
                graph.blocks[fallbacks[0]]
                    .insts
                    .iter()
                    .any(|&inst| matches!(graph.insts[inst].op, Op::Continuation(..)))
            );
            assert!(matches!(
                graph.blocks[fallbacks[0]].terminator.as_ref().unwrap().kind,
                TerminatorKind::Return(_)
            ));
            assert!(
                operations(graph)
                    .iter()
                    .any(|op| matches!(op, Op::Javascript(JsOp::CallNative(..))))
            );
            assert_eq!(
                Cfg::new(graph)
                    .rpo
                    .iter()
                    .filter(|&&block| {
                        matches!(
                            graph.blocks[block].terminator.as_ref().unwrap().kind,
                            TerminatorKind::Fallback(0x1234)
                        )
                    })
                    .count(),
                usize::from(fallback.is_some())
            );
        }
    }

    #[test]
    fn explicit_request_has_one_inline_body_and_one_fallback() {
        let (canonical, bodies) =
            fixture("function square(x) { return x * x; } function main(x) { return square(x); }");
        let request = requests(&canonical.graph, &bodies, 0, &Budget::default())
            .pop()
            .unwrap();
        let mut graph = canonical.graph.clone();
        let before = canonical
            .graph
            .listing(&Facts::compute(&canonical.graph))
            .to_string();
        let plan = plan(&graph, &bodies, 0, &[request]);
        let mut notes = Vec::new();
        apply(&mut graph, &bodies, 0, plan, &mut notes).unwrap();
        let ops = operations(&graph);
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, Op::Javascript(JsOp::Call { .. })))
                .count(),
            1
        );
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(
                    op,
                    Op::Javascript(JsOp::Binary(BinaryOperator::Multiplication, ..))
                ))
                .count(),
            1
        );
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, Op::Javascript(JsOp::IsCode(..))))
                .count(),
            1
        );
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, Op::Javascript(JsOp::IsNumber(_))))
                .count(),
            0
        );
        assert_eq!(
            canonical
                .graph
                .listing(&Facts::compute(&canonical.graph))
                .to_string(),
            before
        );
        let body = bodies.body(FuncId::from_u32(0)).unwrap();
        assert!(
            !operations(body)
                .iter()
                .any(|op| matches!(op, Op::Javascript(JsOp::IsNumber(_) | JsOp::IsCode(..))))
        );
    }

    #[test]
    fn inlining_accepts_all_argument_types_without_numeric_guards() {
        for argument in ["x", "x * 2", "3", "'s'", "{}", "undefined"] {
            let (canonical, bodies) = fixture(&format!(
                "function sum(a, b) {{ return a + b; }} function main(x) {{ let y = {argument}; return sum(y, y); }}",
            ));
            let mut graph = canonical.graph;
            let requests = requests(&graph, &bodies, 0, &Budget::default());
            let plan = plan(&graph, &bodies, 0, &requests);
            apply(&mut graph, &bodies, 0, plan, &mut Vec::new()).unwrap();
            assert!(operations(&graph).iter().any(|op| matches!(
                op,
                Op::Javascript(JsOp::Binary(BinaryOperator::Addition, ..))
            )));
            let ops = operations(&graph);
            assert_eq!(
                ops.iter()
                    .filter(|op| matches!(op, Op::Javascript(JsOp::IsNumber(_))))
                    .count(),
                0
            );
            assert_eq!(
                ops.iter()
                    .filter(|op| matches!(op, Op::Javascript(JsOp::IsCode(..))))
                    .count(),
                1
            );
        }
    }

    #[test]
    fn recompilation_budget_admits_larger_callees() {
        let body = "x = x.value;".repeat(SPECIALIZATION_BODY_INSTS);
        let (canonical, bodies) = fixture(&format!(
            "function helper(x){{{body}return x;}} function main(x){{return helper(x);}}"
        ));
        let initial = super::super::super::plan_specialization(
            FunctionState::new(canonical.graph.clone(), canonical.notes.clone(), true),
            true,
            Some((&bodies, 0)),
        )
        .unwrap();
        let recompiled = super::super::super::inline_and_plan_specialization(
            canonical,
            true,
            Some((&bodies, 0)),
            2,
            &mut |_, _| {},
        )
        .unwrap();
        assert!(initial.plan.inlines.is_empty());
        assert!(!recompiled.plan.inlines.is_empty());
    }

    #[test]
    fn planner_limits_expansion_and_accepts_general_bodies() {
        let (canonical, bodies) =
            fixture("function square(x) { return x * x; } function main(x) { return square(x); }");
        let request = requests(&canonical.graph, &bodies, 0, &Budget::default())
            .pop()
            .unwrap();
        let duplicates = (0..100)
            .map(|_| Request {
                call: request.call,
                target: request.target,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            plan(&canonical.graph, &bodies, 0, &duplicates)
                .inlines
                .len(),
            1
        );
        assert_eq!(
            plan(&canonical.graph, &bodies, 1, &duplicates)
                .inlines
                .len(),
            1
        );
        let calls = "square(x);".repeat(SPECIALIZATION_BUDGET + 1);
        let (canonical, bodies) = fixture(&format!(
            "function square(x) {{ return x * x; }} function main(x) {{ {calls} return x; }}",
        ));
        let candidates = requests(&canonical.graph, &bodies, 0, &Budget::default());
        let selected = plan(&canonical.graph, &bodies, 0, &candidates);
        assert!(!selected.inlines.is_empty());
        assert!(selected.inlines.len() < candidates.len());
        let expanded = plan_round(
            &canonical.graph,
            &bodies,
            0,
            &candidates,
            &mut Budget::new(2),
        );
        assert!(expanded.inlines.len() > selected.inlines.len());
        assert!(
            plan_round(
                &canonical.graph,
                &bodies,
                0,
                &candidates,
                &mut Budget::new(0)
            )
            .is_empty()
        );
        for body in [
            "return x.value;",
            "return other(x);",
            "globalThis.x = x; return x;",
        ] {
            let (canonical, bodies) = fixture(&format!(
                "function helper(x) {{ {body} }} function main(x) {{ return helper(x); }}",
            ));
            let requests = requests(&canonical.graph, &bodies, 0, &Budget::default());
            assert_eq!(requests.len(), 1);
        }
    }
    #[test]
    fn cross_source_general_calls_and_constructors_are_inlined() {
        let (canonical, bodies) = fixture(
            "function C(x){this.value=x;} function touch(o){o.value+=2;return o;} function main(){C.prototype={};return touch(new C(7)).value;}",
        );
        let mut graph = canonical.graph;
        super::super::super::js::construction::resolve(&mut graph);
        super::super::super::js::construction::expand(&mut graph, &mut FxHashSet::default());
        let requests = requests(&graph, &bodies, 1, &Budget::default());
        assert_eq!(requests.len(), 2);
        let plan = plan(&graph, &bodies, 1, &requests);
        assert_eq!(plan.inlines.len(), 2);
        apply(&mut graph, &bodies, 1, plan, &mut Vec::new()).unwrap();
        graph.verify().unwrap();
    }
}
