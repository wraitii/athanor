use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cranelift_module::FuncId;

use oxc_syntax::operator::BinaryOperator;

use super::super::effects::{Assumption, Control, Effects, Locations, Memory};
use super::super::math::METHODS;
use super::{
    CallSummary, Cfg, DirectTarget, Facts, FunctionState, Graph, Inst, Native, Op, Result,
    TerminatorKind, Types, UnaryOp, Value,
};

pub(super) const SPECIALIZATION_BUDGET: usize = 128;
pub(super) const SPECIALIZATION_BODY_INSTS: usize = 64;
pub(super) const SPECIALIZATION_ROUNDS: usize = 2;
pub(crate) const DEFAULT_RECOMPILE_INLINE_MULTIPLIER: usize = 4;

#[derive(Clone, Default)]
pub(in crate::jit) struct Repository {
    bodies: HashMap<FuncId, Arc<Body>>,
    pub(in crate::jit) number_first: HashSet<FuncId>,
    pub(in crate::jit) feedback: HashMap<crate::jit::profile::Site, FuncId>,
    pub(in crate::jit) native_feedback:
        HashMap<crate::jit::profile::Site, crate::heap::NativeFunction>,
    pub(in crate::jit) read_feedback: HashMap<
        (crate::jit::profile::Site, crate::jit::profile::ReadKind),
        crate::jit::profile::ReadType,
    >,
    pub(in crate::jit) shape_feedback: HashMap<
        (
            crate::jit::profile::Site,
            crate::jit::profile::PropertyAccess,
        ),
        crate::jit::profile::ShapeAccess,
    >,
    locations: HashMap<(usize, u32), FuncId>,
    globals: HashMap<String, FuncId>,
    pub(in crate::jit) sources: HashMap<usize, usize>,
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
                    Op::Call { callee, .. }
                    | Op::CallPack(callee, ..)
                    | Op::Construct { callee, .. }
                    | Op::ConstructorCall { callee, .. }
                    | Op::ConstructPack(callee, ..)
                    | Op::ConstructPackResolved(callee, ..)
                    | Op::ConstructorCallPack(callee, ..)
                    | Op::ConstructResolved { callee, .. } => callee,
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

    pub(in crate::jit) fn code(&self, source: usize, start: u32) -> Option<FuncId> {
        self.locations.get(&(source, start)).copied()
    }

    pub(super) fn number_first(&self, source: usize, start: u32) -> bool {
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

    pub(super) fn name(&self, code: FuncId) -> Option<&str> {
        self.bodies.get(&code).map(|body| body.name.as_str())
    }

    pub(super) fn body(&self, code: FuncId) -> Option<&Graph> {
        let body = self.bodies.get(&code)?;
        Some(&body.graph)
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
struct SummaryEffects {
    memory: Memory,
    control: Control,
    globals_read: HashSet<String>,
    globals_written: HashSet<String>,
    properties_read: HashSet<String>,
    properties_written: HashSet<String>,
    unresolved: HashSet<(FuncId, Inst)>,
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
        let operands = op.operands_iter().fold(Types::default(), |types, value| {
            types.union(facts.of(value))
        });
        if operands.within(Types::PRIMITIVE) {
            &[]
        } else {
            COERCION
        }
    };
    match op {
        Op::Get(..)
        | Op::GetIndex(..)
        | Op::FillPropertyCache(..)
        | Op::Set(..)
        | Op::SetIndex(..)
        | Op::Define(..) => PROPERTY,
        Op::Spread(_) | Op::IteratorNext(_) | Op::EnumerateNext(_) => {
            &[Assumption::DirectIteration]
        }
        Op::ConstructorPrototype(_)
        | Op::ProbeGlobal(_)
        | Op::ProbeProperty(..)
        | Op::ProbeShapeProperty(..)
        | Op::ProbeIndex(..) => &[Assumption::NoAccessors],
        Op::View(_) | Op::TypedIndex(..) => &[Assumption::FixedTypedStorage],
        Op::GuardedCompare {
            operator,
            left,
            right,
            ..
        } => assumptions(&Op::Binary(*operator, *left, *right), facts),
        Op::Binary(
            BinaryOperator::StrictEquality
            | BinaryOperator::StrictInequality
            | BinaryOperator::In
            | BinaryOperator::Instanceof,
            ..,
        ) => &[],
        Op::Unary(UnaryOp::Not | UnaryOp::Typeof | UnaryOp::Increment | UnaryOp::Decrement, _) => {
            &[]
        }
        Op::CallNative(native, _)
            if METHODS[native.index()].1 == crate::heap::MathFunction::Random =>
        {
            &[]
        }
        Op::Binary(..) | Op::ToNumeric(_) | Op::Unary(..) | Op::CallNative(..) => coercing(),
        _ => &[],
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
struct Requirements {
    assumptions: HashSet<Assumption>,
    identities: HashSet<(String, FuncId)>,
    natives: HashSet<Native>,
    string_splits: HashSet<(FuncId, Inst)>,
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
                    Op::GlobalRead(name) | Op::GlobalReadOptional(name) => {
                        effects.globals_read.insert(name.clone());
                    }
                    Op::GlobalWrite(name, _) | Op::GlobalInitialize(name, _) => {
                        effects.globals_written.insert(name.clone());
                    }
                    Op::Get(_, name) | Op::FillPropertyCache(_, _, name) => {
                        effects.properties_read.insert(name.clone());
                    }
                    Op::IsPropertyCached(..) | Op::CachedProperty(..) => {
                        effects.properties_read.insert("*".into());
                    }
                    Op::Set(_, name, _) | Op::Define(_, name, _) | Op::DefineHidden(_, name, _) => {
                        effects.properties_written.insert(name.clone());
                        effects.memory.writes = effects.memory.writes.union(Locations::HEAP);
                    }
                    Op::GetIndex(..)
                    | Op::IndexedGet(..)
                    | Op::ArrayLoad(..)
                    | Op::TypedIndex(..) => {
                        effects.properties_read.insert("*".into());
                    }
                    Op::SetIndex(..) | Op::ArrayAppend(..) => {
                        effects.properties_written.insert("*".into());
                        effects.memory.writes = effects.memory.writes.union(Locations::HEAP);
                    }
                    _ => {}
                }
                if let Op::Call {
                    callee,
                    receiver,
                    args,
                    ..
                } = op
                {
                    let target = match graph.defining_op(*callee) {
                        Some(Op::GlobalRead(name)) => {
                            self.globals.get(name).copied().inspect(|&target| {
                                requires.identities.insert((name.clone(), target));
                            })
                        }
                        Some(Op::Closure(_, target, _, false, _, _, _)) => Some(*target),
                        _ => None,
                    };
                    if let Some(target) = target.filter(|target| self.body(*target).is_some()) {
                        effects.control = Control::UNKNOWN;
                        calls.push((inst, target));
                        continue;
                    }
                    if let Some(Op::Get(object, property)) = graph.defining_op(*callee)
                        && property == "split"
                        && *receiver == Some(*object)
                    {
                        requires.string_splits.insert((code, inst));
                        let lowered = Op::StringSplit(*object, args.clone());
                        effects.absorb(graph.contract(&lowered, &facts).effects);
                        requires
                            .assumptions
                            .extend(assumptions(&lowered, &facts).iter().copied());
                        continue;
                    }
                    if let Some(Op::Get(object, property)) = graph.defining_op(*callee)
                        && let Some(Op::GlobalRead(global)) = graph.defining_op(*object)
                        && let Some(native) = Native::lookup(global, property)
                    {
                        requires.natives.insert(native);
                        let lowered = Op::CallNative(native, args.clone());
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
    ) -> HashMap<FuncId, Summary> {
        let mut local = HashMap::new();
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
        let mut summaries: HashMap<_, _> = local
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
        let mut visited = HashSet::new();
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
        Op::Call { callee, .. } | Op::CallPack(callee, ..) => (*callee, false),
        Op::Construct { callee, .. }
        | Op::ConstructorCall { callee, .. }
        | Op::ConstructPack(callee, ..)
        | Op::ConstructPackResolved(callee, ..)
        | Op::ConstructorCallPack(callee, ..)
        | Op::ConstructResolved { callee, .. } => (*callee, true),
        _ => return graph.op_text(op),
    };
    match graph.defining_op(callee) {
        Some(Op::Get(_, name)) if constructor => {
            format!("constructor .{name}: requires runtime constructor identity and effects")
        }
        Some(Op::Get(_, name)) => {
            format!("method .{name}: requires runtime callee identity and receiver effects")
        }
        Some(Op::GlobalRead(name) | Op::GlobalReadOptional(name)) if constructor => {
            format!("constructor {name}: requires runtime constructor identity and effects")
        }
        Some(Op::GlobalRead(name) | Op::GlobalReadOptional(name)) => {
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

pub(super) fn install_summaries(graph: &mut Graph, bodies: &Repository, notes: &mut Vec<String>) {
    let marked: HashSet<Value> = graph
        .insts
        .values()
        .filter_map(|data| match data.op {
            Op::MarkClass(value) => Some(value),
            _ => None,
        })
        .collect();
    let mut closures = Vec::new();
    let mut roots = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            let Op::Call { callee, .. } = graph.insts[inst].op else {
                continue;
            };
            match graph.defining_op(callee) {
                Some(Op::Closure(_, target, _, false, _, _, _)) if !marked.contains(&callee) => {
                    closures.push((inst, *target));
                    roots.push(*target);
                }
                Some(Op::GlobalRead(name)) => {
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
    let mut installed = HashMap::new();
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
        match installed.get(&target) {
            Some(summary) if summary.requires.is_empty() => {
                notes.push(format!("call {inst}: {target} proved closed and read-only"));
            }
            None if !summaries.contains_key(&target) => {}
            _ => {
                notes.push(format!(
                    "call {inst}: closed-call proof rejected: {}",
                    bodies.obligation_chain(target, true)
                ));
            }
        }
    }
    graph.summaries = Arc::new(installed);
}

#[derive(Clone, Default, PartialEq, Eq)]
pub(in crate::jit) struct ConditionalPlan {
    pub(super) native_calls: Vec<crate::jit::profile::Site>,
    pub(super) calls: Vec<ConditionalCall>,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct ConditionalCall {
    pub(super) inst: Inst,
    pub(super) callee: super::Value,
    pub(super) name: String,
    pub(super) target: FuncId,
    pub(super) memory: Memory,
    pub(super) control: Control,
    pub(super) identities: Vec<(String, FuncId)>,
}

pub(super) fn conditional_plan(
    graph: &Graph,
    bodies: &Repository,
    notes: &mut Vec<String>,
) -> ConditionalPlan {
    let mut candidates = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            let Op::Call { callee, .. } = graph.insts[inst].op else {
                continue;
            };
            let Some(Op::GlobalRead(name)) = graph.defining_op(callee) else {
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
                "call {inst}: conditional proof rejected: {}",
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
                "call {inst}: conditional effects require {}",
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

pub(super) struct Request {
    pub(super) call: Inst,
    pub(super) target: FuncId,
}

pub(super) struct Plan {
    inlines: Vec<Inline>,
}

pub(super) struct Budget {
    remaining: usize,
    body_limit: usize,
    sites: HashSet<Inst>,
    pub(super) inlines: Vec<Inline>,
}

impl Default for Budget {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Budget {
    pub(super) fn new(multiplier: usize) -> Self {
        Self {
            remaining: SPECIALIZATION_BUDGET.saturating_mul(multiplier),
            body_limit: SPECIALIZATION_BODY_INSTS.saturating_mul(multiplier),
            sites: HashSet::new(),
            inlines: Vec::new(),
        }
    }
}

impl Plan {
    pub(super) fn is_empty(&self) -> bool {
        self.inlines.is_empty()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(in crate::jit) struct Inline {
    pub(super) call: Inst,
    pub(super) target: FuncId,
    pub(super) arrow: bool,
    pub(super) guard_code: bool,
    pub(super) source_name: Option<(usize, usize)>,
}

fn eligible(graph: &Graph, body_limit: usize) -> bool {
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
                || matches!(
                    graph.insts[inst].op,
                    Op::Arguments | Op::Rest(_) | Op::Catch
                )
            {
                return false;
            }
        }
    }
    true
}

fn call_target(
    graph: &Graph,
    bodies: &Repository,
    source: usize,
    call: Inst,
    callee: super::Value,
) -> Option<FuncId> {
    let data = &graph.insts[call];
    let site = crate::jit::profile::Site {
        source: data.source_name.map_or(source, |name| name.0),
        start: data.span.start,
        end: data.span.end,
    };
    if let Some(&target) = bodies.feedback.get(&site) {
        return Some(target);
    }
    match graph.defining_op(callee) {
        Some(Op::GlobalRead(name)) => bodies.globals.get(name).copied(),
        Some(Op::Closure(_, target, _, false, _, _, _)) => Some(*target),
        _ => None,
    }
}

pub(super) fn direct_calls(graph: &mut Graph, bodies: &Repository, source: usize) {
    let mut targets = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &call in &graph.blocks[block].insts {
            let Op::Call { callee, .. } = graph.insts[call].op else {
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
                    code: target,
                    arity,
                    lexical_this,
                },
            ));
        }
    }
    for (call, direct) in targets {
        if let Op::Call { target, .. } = &mut graph.insts[call].op {
            *target = Some(direct);
        }
    }
}

pub(super) fn requests(
    graph: &Graph,
    bodies: &Repository,
    source: usize,
    budget: &Budget,
) -> Vec<Request> {
    let mut requests = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &call in &graph.blocks[block].insts {
            let callee = match &graph.insts[call].op {
                Op::Call { callee, .. }
                | Op::ConstructorCall { callee, .. }
                | Op::ConstructorCallPack(callee, ..) => callee,
                _ => continue,
            };
            let Some(target) = call_target(graph, bodies, source, call, *callee) else {
                continue;
            };
            if bodies
                .body(target)
                .is_some_and(|body| eligible(body, budget.body_limit))
            {
                requests.push(Request { call, target });
            }
        }
    }
    requests
}

#[cfg(test)]
pub(super) fn plan(
    graph: &Graph,
    bodies: &Repository,
    source: usize,
    requests: &[Request],
) -> Plan {
    plan_round(graph, bodies, source, requests, &mut Budget::default())
}

pub(super) fn plan_round(
    graph: &Graph,
    bodies: &Repository,
    source: usize,
    requests: &[Request],
    budget: &mut Budget,
) -> Plan {
    let mut inlines = Vec::new();
    for request in requests {
        let Some(body) = bodies
            .body(request.target)
            .filter(|body| eligible(body, budget.body_limit))
        else {
            continue;
        };
        let callee = match &graph.insts[request.call].op {
            Op::Call { callee, .. }
            | Op::ConstructorCall { callee, .. }
            | Op::ConstructorCallPack(callee, ..) => callee,
            _ => continue,
        };
        let cost: usize = Cfg::new(body)
            .rpo
            .iter()
            .map(|&block| body.blocks[block].insts.len())
            .sum();
        if cost > budget.remaining || budget.sites.contains(&request.call) {
            continue;
        }
        budget.sites.insert(request.call);
        budget.remaining -= cost;
        let body_source = bodies.bodies[&request.target].source;
        let source_name = if body_source == source {
            None
        } else {
            bodies
                .sources
                .get(&body_source)
                .map(|&length| (body_source, length))
        };
        inlines.push(Inline {
            call: request.call,
            target: request.target,
            source_name,
            arrow: bodies.bodies[&request.target].arrow,
            guard_code: !matches!(graph.defining_op(*callee), Some(Op::Closure(_, code, _, false, _, _, _))
                if *code == request.target
                && !graph.insts.values().any(|data| matches!(data.op, Op::MarkClass(value) if value == *callee))),
        });
    }
    budget.inlines.extend(inlines.iter().cloned());
    Plan { inlines }
}

pub(super) fn apply(
    graph: &mut Graph,
    bodies: &Repository,
    _source: usize,
    plan: Plan,
    notes: &mut Vec<String>,
) -> Result<()> {
    for inline in plan.inlines {
        let body = bodies.body(inline.target).unwrap();
        super::inlining::apply(graph, body, &inline);
        graph.verify()?;
        let span = graph.insts[inline.call].span;
        notes.push(format!(
            "call at {}..{}: inline {} with {} code check(s)",
            span.start,
            span.end,
            inline.target,
            usize::from(inline.guard_code)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::Types;
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
            let graph =
                super::super::build::lower(function, semantic.scoping(), &HashSet::new()).unwrap();
            let canonical = super::super::canonicalize_graph(graph, true).unwrap();
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
            super::super::optimize_canonical(canonical, true, Some((&bodies, 0))).unwrap();
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
            super::super::optimize_canonical(canonical, true, Some((&bodies, 0))).unwrap();
        let graph = &analysis.graph;
        let cfg = Cfg::new(graph);
        assert!(
            cfg.rpo.iter().any(|&block| graph.blocks[block]
                .insts
                .iter()
                .any(|&inst| matches!(graph.insts[inst].op, Op::StringSplitParts(..)))),
            "{analysis}"
        );
        let setters: Vec<_> = cfg
            .rpo
            .iter()
            .flat_map(|&block| &graph.blocks[block].insts)
            .filter_map(|&inst| match graph.insts[inst].op {
                Op::Set(_, ref name, value)
                    if name == "template" && !graph.cold.contains(&graph.insts[inst].block) =>
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
                types.within(Types::STRING.union(Types::UNDEFINED)),
                "{analysis}"
            );
            assert!(types.intersects(Types::STRING));
            assert!(types.intersects(Types::UNDEFINED));
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
            super::super::optimize::version(&mut graph, &plan, None, &mut notes)
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
            .filter(|&inst| matches!(graph.insts[inst].op, Op::Call { .. }))
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
            super::super::optimize::version(&mut graph, &plan, None, &mut Vec::new()).unwrap();
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
                .filter(|op| matches!(op, Op::IsCode(..)))
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
        let regions = super::super::optimize::version(&mut graph, &plan, None, &mut notes).unwrap();
        assert_eq!(regions.len(), 1);
        let closed = closed_calls(&graph);
        assert!(!closed.contains(&generic));
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
                .any(|op| matches!(op, Op::CallNative(..)))
        );
        assert_eq!(
            operations(&graph)
                .iter()
                .filter(|op| matches!(op, Op::IsCode(..)))
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
        let regions = super::super::optimize::version(&mut graph, &plan, None, &mut notes).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(closed_calls(&graph).len(), 1);
        let probes: Vec<_> = operations(&graph)
            .into_iter()
            .filter_map(|op| match op {
                Op::ProbeGlobal(name) => Some(name.as_str()),
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
            super::super::optimize::version(&mut graph, &plan, None, &mut Vec::new()).unwrap();
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
        let super::super::ValueDef::Result(inst) = graph.values[plan.calls[0].callee] else {
            panic!();
        };
        graph.insts[inst].op = Op::GlobalRead("other".into());
        let regions =
            super::super::optimize::version(&mut graph, &plan, None, &mut Vec::new()).unwrap();
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
        let Op::Call { callee, .. } = graph.insts[request.call].op else {
            panic!()
        };
        let super::super::ValueDef::Result(def) = graph.values[callee] else {
            panic!()
        };
        graph.insts[def].op = Op::Closure(
            graph.insts[def].span,
            request.target,
            false,
            false,
            false,
            vec![],
            None,
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
        let Op::Call { callee, .. } = graph.insts[request.call].op else {
            panic!()
        };
        let super::super::ValueDef::Result(def) = graph.values[callee] else {
            panic!()
        };
        graph.insts[def].op = Op::Closure(
            graph.insts[def].span,
            request.target,
            false,
            false,
            false,
            vec![],
            None,
        );
        install_summaries(&mut graph, &bodies, &mut Vec::new());
        assert!(closed_calls(&graph).contains(&request.call));
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
        ] {
            let (canonical, bodies) = fixture(source);
            let first = super::super::plan_canonical(
                FunctionState::new(canonical.graph.clone(), canonical.notes.clone(), true),
                true,
                Some((&bodies, 0)),
            )
            .unwrap();
            let second = super::super::plan_canonical(canonical, true, Some((&bodies, 0))).unwrap();
            assert_eq!(first.plan.inlines.len(), count);
            assert!(first.plan.inlines == second.plan.inlines);
            let analysis = super::super::optimize_planned(
                first,
                super::super::Versioning::Speculate {
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
            if matches!(data.op, Op::Call { .. }) {
                bodies.feedback.insert(
                    crate::jit::profile::Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    FuncId::from_u32(0),
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
                .filter(|op| matches!(op, Op::IsCode(..)))
                .count(),
            2
        );
        assert_eq!(
            operations(&graph)
                .iter()
                .filter(|op| matches!(op, Op::Binary(BinaryOperator::Multiplication, ..)))
                .count(),
            2
        );
    }

    #[test]
    fn varying_method_guards_exit_to_the_original_continuation() {
        let (canonical, mut bodies) = fixture(
            "function method(x){return Math.floor(x);} function main(objects,n){let sum=0;for(let i=0;i<n;++i)sum+=objects[i].method(i);return sum;}",
        );
        for data in canonical.graph.insts.values() {
            if let Op::Call { callee, .. } = data.op
                && matches!(canonical.graph.defining_op(callee), Some(Op::Get(_, name)) if name == "method")
            {
                bodies.feedback.insert(
                    crate::jit::profile::Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    FuncId::from_u32(0),
                );
            }
        }
        for fallback in [None, Some(0x1234)] {
            let planned = super::super::plan_canonical(
                FunctionState::new(canonical.graph.clone(), canonical.notes.clone(), true),
                true,
                Some((&bodies, 0)),
            )
            .unwrap();
            let analysis = super::super::optimize_planned(
                planned,
                super::super::Versioning::Speculate {
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
                if let Some(super::super::Terminator {
                    kind: TerminatorKind::Branch(condition, _, otherwise),
                    ..
                }) = &graph.blocks[block].terminator
                    && matches!(graph.defining_op(*condition), Some(Op::IsCode(_, code)) if *code == FuncId::from_u32(0))
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
                    .any(|op| matches!(op, Op::CallNative(..)))
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
                .filter(|op| matches!(op, Op::Call { .. }))
                .count(),
            1
        );
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, Op::Binary(BinaryOperator::Multiplication, ..)))
                .count(),
            1
        );
        assert_eq!(
            ops.iter().filter(|op| matches!(op, Op::IsCode(..))).count(),
            1
        );
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, Op::IsNumber(_)))
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
                .any(|op| matches!(op, Op::IsNumber(_) | Op::IsCode(..)))
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
            assert!(
                operations(&graph)
                    .iter()
                    .any(|op| matches!(op, Op::Binary(BinaryOperator::Addition, ..)))
            );
            let ops = operations(&graph);
            assert_eq!(
                ops.iter()
                    .filter(|op| matches!(op, Op::IsNumber(_)))
                    .count(),
                0
            );
            assert_eq!(
                ops.iter().filter(|op| matches!(op, Op::IsCode(..))).count(),
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
        let initial = super::super::plan_canonical(
            FunctionState::new(canonical.graph.clone(), canonical.notes.clone(), true),
            true,
            Some((&bodies, 0)),
        )
        .unwrap();
        let recompiled =
            super::super::plan_canonical_with_budget(canonical, true, Some((&bodies, 0)), 2)
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
        super::super::construction::resolve(&mut graph);
        super::super::construction::expand(&mut graph, &mut HashSet::new());
        let requests = requests(&graph, &bodies, 1, &Budget::default());
        assert_eq!(requests.len(), 2);
        let plan = plan(&graph, &bodies, 1, &requests);
        assert_eq!(plan.inlines.len(), 2);
        apply(&mut graph, &bodies, 1, plan, &mut Vec::new()).unwrap();
        graph.verify().unwrap();
    }
}
