use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cranelift_module::FuncId;

use super::super::effects::{
    Assumption, ERROR, GLOBAL_WRITE, HEAP_WRITE, INTERNAL, Reentry, UNKNOWN,
};
use super::{Canonical, Cfg, DirectTarget, Facts, Graph, Inst, Native, Op, Result, TerminatorKind};

const SPECIALIZATION_BUDGET: usize = 128;
const SPECIALIZATION_BODY_INSTS: usize = 64;
pub(super) const SPECIALIZATION_ROUNDS: usize = 2;
pub(crate) const DEFAULT_RECOMPILE_INLINE_MULTIPLIER: usize = 4;

#[derive(Clone, Default)]
pub(in crate::jit) struct Repository {
    bodies: HashMap<FuncId, Arc<Body>>,
    pub(in crate::jit) feedback: HashMap<crate::jit::planner::Site, FuncId>,
    pub(in crate::jit) read_feedback: HashMap<
        (crate::jit::planner::Site, crate::jit::planner::ReadKind),
        crate::jit::planner::ReadType,
    >,
    locations: HashMap<(usize, u32), FuncId>,
    globals: HashMap<String, FuncId>,
    pub(in crate::jit) sources: HashMap<usize, usize>,
}

#[derive(Clone)]
struct Body {
    name: String,
    source: usize,
    canonical: Canonical,
    arrow: bool,
}

impl Repository {
    pub(in crate::jit) fn callees(&self, code: FuncId) -> Vec<FuncId> {
        let Some(body) = self.bodies.get(&code) else {
            return Vec::new();
        };
        self.graph_callees(&body.canonical.graph, body.source)
    }

    pub(in crate::jit) fn graph_callees(&self, graph: &Graph, source: usize) -> Vec<FuncId> {
        let mut targets = Vec::new();
        for block in Cfg::new(graph).rpo {
            for &call in &graph.blocks[block].insts {
                let callee = match graph.insts[call].op {
                    Op::Call { callee, .. }
                    | Op::CallSpread(callee, ..)
                    | Op::Construct { callee, .. }
                    | Op::ConstructorCall { callee, .. }
                    | Op::ConstructSpread(callee, ..)
                    | Op::ConstructSpreadResolved(callee, ..)
                    | Op::ConstructorCallSpread(callee, ..)
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
        canonical: Canonical,
    ) {
        self.locations
            .insert((source, canonical.graph.span.start), code);
        self.globals.insert(name.to_owned(), code);
        self.bodies.insert(
            code,
            Arc::new(Body {
                name: name.to_owned(),
                source,
                canonical,
                arrow: false,
            }),
        );
    }

    pub(in crate::jit) fn insert_closure(
        &mut self,
        source: usize,
        code: FuncId,
        canonical: Canonical,
        arrow: bool,
    ) {
        self.locations
            .insert((source, canonical.graph.span.start), code);
        let name = canonical.graph.name.clone();
        self.bodies.insert(
            code,
            Arc::new(Body {
                name,
                source,
                canonical,
                arrow,
            }),
        );
    }

    pub(in crate::jit) fn code(&self, source: usize, start: u32) -> Option<FuncId> {
        self.locations.get(&(source, start)).copied()
    }

    pub(in crate::jit) fn emission(&self, code: FuncId) -> Option<(usize, Canonical)> {
        let body = self.bodies.get(&code)?;
        Some((body.source, body.canonical.clone()))
    }

    pub(in crate::jit) fn canonical(&self, source: usize, start: u32) -> Option<&Canonical> {
        let code = self.locations.get(&(source, start))?;
        Some(&self.bodies.get(code)?.canonical)
    }

    pub(in crate::jit) fn lexical_this(&self, code: FuncId) -> bool {
        self.bodies
            .get(&code)
            .is_some_and(|body| body.arrow && super::inlining::uses_this(&body.canonical.graph))
    }

    pub(super) fn body(&self, code: FuncId) -> Option<&Graph> {
        let body = self.bodies.get(&code)?;
        Some(&body.canonical.graph)
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
struct ConditionalEffects {
    flags: u16,
    globals_read: HashSet<String>,
    globals_written: HashSet<String>,
    properties_read: HashSet<String>,
    properties_written: HashSet<String>,
    assumptions: HashSet<Assumption>,
    identities: HashSet<(String, FuncId)>,
    natives: HashSet<Native>,
    string_splits: HashSet<(FuncId, Inst)>,
    unresolved: HashSet<(FuncId, Inst)>,
    unsupported: bool,
}

impl ConditionalEffects {
    fn merge(&mut self, other: &Self) {
        self.flags |= other.flags;
        self.unsupported |= other.unsupported;
        self.globals_read.extend(other.globals_read.iter().cloned());
        self.globals_written
            .extend(other.globals_written.iter().cloned());
        self.properties_read
            .extend(other.properties_read.iter().cloned());
        self.properties_written
            .extend(other.properties_written.iter().cloned());
        self.assumptions.extend(other.assumptions.iter().copied());
        self.identities.extend(other.identities.iter().cloned());
        self.natives.extend(other.natives.iter().copied());
        self.string_splits
            .extend(other.string_splits.iter().copied());
        self.unresolved.extend(other.unresolved.iter().copied());
    }
}

struct LocalEffects {
    effects: ConditionalEffects,
    calls: Vec<(Inst, FuncId)>,
}

impl Repository {
    fn local_effects(&self, code: FuncId) -> Option<LocalEffects> {
        let graph = self.body(code)?;
        let facts = Facts::compute(graph);
        let mut effects = ConditionalEffects::default();
        if graph.generator || graph.has_handlers {
            effects.flags |= UNKNOWN;
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
                    Op::Get(_, name) => {
                        effects.properties_read.insert(name.clone());
                    }
                    Op::Set(_, name, _) | Op::Define(_, name, _) | Op::DefineHidden(_, name, _) => {
                        effects.properties_written.insert(name.clone());
                        effects.flags |= HEAP_WRITE;
                    }
                    Op::GetIndex(..) | Op::IndexedGet(..) | Op::TypedIndex(..) => {
                        effects.properties_read.insert("*".into());
                    }
                    Op::SetIndex(..) | Op::ArrayAppend(..) => {
                        effects.properties_written.insert("*".into());
                        effects.flags |= HEAP_WRITE;
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
                            self.globals.get(name).copied().map(|target| {
                                effects.identities.insert((name.clone(), target));
                                target
                            })
                        }
                        Some(Op::Closure(_, target, _, false, _, _, _)) => Some(*target),
                        _ => None,
                    };
                    if let Some(target) = target.filter(|target| self.body(*target).is_some()) {
                        effects.flags |= INTERNAL | ERROR;
                        calls.push((inst, target));
                        continue;
                    }
                    if let Some(Op::Get(object, property)) = graph.defining_op(*callee)
                        && property == "split"
                        && *receiver == Some(*object)
                    {
                        effects.string_splits.insert((code, inst));
                        let contract =
                            graph.contract(&Op::StringSplit(*object, args.clone()), &facts);
                        effects.flags |= contract.flags;
                        effects
                            .assumptions
                            .extend(contract.assumptions.iter().copied());
                        continue;
                    }
                    if let Some(Op::Get(object, property)) = graph.defining_op(*callee)
                        && let Some(Op::GlobalRead(global)) = graph.defining_op(*object)
                        && let Some(native) = Native::lookup(global, property)
                    {
                        effects.natives.insert(native);
                        let contract =
                            graph.contract(&Op::CallNative(native, args.clone()), &facts);
                        effects.flags |= contract.flags;
                        for assumption in contract.assumptions {
                            effects.assumptions.insert(*assumption);
                        }
                        continue;
                    }
                }
                let contract = graph.contract(op, &facts);
                effects.flags |= contract.flags;
                for assumption in contract.assumptions {
                    effects.assumptions.insert(*assumption);
                }
                if contract.reentry != Reentry::Closed {
                    effects.unresolved.insert((code, inst));
                    effects.flags |= UNKNOWN;
                }
            }
        }
        Some(LocalEffects { effects, calls })
    }

    fn effect_summaries(
        &self,
        roots: impl IntoIterator<Item = FuncId>,
    ) -> HashMap<FuncId, ConditionalEffects> {
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
            .map(|(&code, body)| (code, body.effects.clone()))
            .collect();
        loop {
            let mut changed = false;
            for (&code, body) in &local {
                let mut summary = body.effects.clone();
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
            let effects = &local.effects;
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
                && let Some((name, target)) = effects.identities.iter().min_by_key(|(name, _)| name)
            {
                return format!("{path}: requires global {name} to remain {target}");
            }
            if let Some((_, inst)) = effects.string_splits.iter().min_by_key(|(_, inst)| *inst) {
                return format!(
                    "{path} -> {inst}: method .split requires native String.split identity and its original receiver"
                );
            }
            if let Some(native) = effects.natives.iter().min_by_key(|native| native.index()) {
                return format!("{path}: requires native {native} identity");
            }
            if effects.flags & (GLOBAL_WRITE | HEAP_WRITE) != 0
                || !effects.properties_written.is_empty()
            {
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
        Op::Call { callee, .. } | Op::CallSpread(callee, ..) => (*callee, false),
        Op::Construct { callee, .. }
        | Op::ConstructorCall { callee, .. }
        | Op::ConstructSpread(callee, ..)
        | Op::ConstructSpreadResolved(callee, ..)
        | Op::ConstructorCallSpread(callee, ..)
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

pub(super) fn prove_closed_calls(graph: &mut Graph, bodies: &Repository, notes: &mut Vec<String>) {
    let mut candidates = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            let Op::Call { callee, .. } = graph.insts[inst].op else {
                continue;
            };
            let Some(Op::Closure(_, target, _, false, _, _, _)) = graph.defining_op(callee) else {
                continue;
            };
            if graph
                .insts
                .values()
                .any(|data| matches!(data.op, Op::MarkClass(value) if value == callee))
            {
                continue;
            }
            candidates.push((inst, callee, *target));
        }
    }
    if candidates.is_empty() {
        return;
    }
    let summaries = bodies.effect_summaries(candidates.iter().map(|&(_, _, target)| target));
    for (inst, callee, target) in candidates {
        let Some(summary) = summaries.get(&target) else {
            continue;
        };
        if summary.unsupported
            || !summary.unresolved.is_empty()
            || !summary.identities.is_empty()
            || !summary.natives.is_empty()
            || !summary.string_splits.is_empty()
            || summary.flags & (GLOBAL_WRITE | HEAP_WRITE) != 0
            || !summary.properties_written.is_empty()
        {
            notes.push(format!(
                "call {inst}: closed-call proof rejected: {}",
                bodies.obligation_chain(target, true)
            ));
            continue;
        }
        graph.closed_calls.insert(inst, (callee, summary.flags));
        notes.push(format!("call {inst}: {target} proved closed and read-only"));
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
pub(in crate::jit) struct ConditionalPlan {
    pub(super) calls: Vec<ConditionalCall>,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct ConditionalCall {
    pub(super) inst: Inst,
    pub(super) callee: super::Value,
    pub(super) name: String,
    pub(super) flags: u16,
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
        if summary.unsupported
            || !summary.unresolved.is_empty()
            || !summary.natives.is_empty()
            || !summary.string_splits.is_empty()
            || summary.flags & (GLOBAL_WRITE | HEAP_WRITE) != 0
            || !summary.properties_written.is_empty()
        {
            notes.push(format!(
                "call {inst}: conditional proof rejected: {}",
                bodies.obligation_chain(target, false)
            ));
            continue;
        }
        let mut required = summary.identities.clone();
        required.insert((name.clone(), target));
        let mut identities: Vec<_> = required.into_iter().collect();
        identities.sort_by(|a, b| a.0.cmp(&b.0));
        let mut assumptions: Vec<_> = summary
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
            flags: summary.flags,
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
    let site = crate::jit::planner::Site {
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
            let Some(arity) = body.canonical.graph.arity else {
                continue;
            };
            let lexical_this = body.arrow && super::inlining::uses_this(&body.canonical.graph);
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
                | Op::ConstructorCallSpread(callee, ..) => callee,
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
            | Op::ConstructorCallSpread(callee, ..) => callee,
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

    fn fixture(source: &str) -> (Canonical, Repository) {
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
            let canonical = super::super::canonicalize(graph, true).unwrap();
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
            assert!(summary.unresolved.is_empty());
            assert!(summary.globals_read.contains("state"));
            assert!(summary.properties_read.contains("value"));
            assert!(summary.properties_written.contains("value"));
            assert_eq!(summary.identities.len(), 2);
            assert_eq!(summary.natives.len(), 1);
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
        assert!(listing.contains("constructor_call_spread"));
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
        assert!(summary.unresolved.is_empty());
        assert_eq!(summary.string_splits.len(), 1);
        assert!(summary.flags & super::super::ALLOCATION != 0);
        assert!(summary.flags & super::super::ERROR != 0);
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
            .canonical
            .graph
            .name = "<function>".into();
        let mut notes = Vec::new();
        let plan = conditional_plan(&canonical.graph, &bodies, &mut notes);
        assert!(plan.calls.is_empty());
        let summaries = bodies.effect_summaries([FuncId::from_u32(0)]);
        assert_eq!(summaries[&FuncId::from_u32(0)].string_splits.len(), 1);
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
        assert!(
            notes
                .iter()
                .any(|note| note.contains("writes guarded state"))
        );
    }

    #[test]
    fn recursive_global_calls_are_guarded_as_a_group() {
        let (canonical, bodies) = fixture(
            "function first(n){if(n)return second(n-1);return state.value;} function second(n){if(n)return first(n-1);return state.value;} function main(n){return first(n);}",
        );
        let mut graph = canonical.graph;
        let plan = conditional_plan(&graph, &bodies, &mut Vec::new());
        assert_eq!(plan.calls.len(), 1);
        assert_eq!(plan.calls[0].identities.len(), 2);
        let regions =
            super::super::optimize::version(&mut graph, &plan, None, &mut Vec::new()).unwrap();
        assert_eq!(regions.len(), 1);
        let region = &regions[0];
        assert_eq!(graph.closed_calls.len(), 1);
        assert!(
            graph
                .closed_calls
                .keys()
                .all(|inst| region.contains(&graph.insts[*inst].block))
        );
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
        let plan = conditional_plan(&graph, &bodies, &mut Vec::new());
        let generic = plan.calls[0].inst;
        let mut notes = Vec::new();
        let regions = super::super::optimize::version(&mut graph, &plan, None, &mut notes).unwrap();
        assert_eq!(regions.len(), 1);
        assert!(!graph.closed_calls.contains_key(&generic));
        assert_eq!(graph.closed_calls.len(), 1);
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
        let plan = conditional_plan(&graph, &bodies, &mut Vec::new());
        assert_eq!(plan.calls.len(), 2);
        let mut notes = Vec::new();
        let regions = super::super::optimize::version(&mut graph, &plan, None, &mut notes).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(graph.closed_calls.len(), 1);
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
    fn conditional_proofs_survive_retained_inner_copies() {
        let (canonical, bodies) = fixture(
            "function first(n){if(n)return first(n-1);return 1;} function main(rows){let sum=0;for(let y=0;y<2;++y){const row=rows[y];for(let x=0;x<2;++x)sum+=row[x]+first(x);}return sum;}",
        );
        let mut graph = canonical.graph;
        let plan = conditional_plan(&graph, &bodies, &mut Vec::new());
        let regions =
            super::super::optimize::version(&mut graph, &plan, None, &mut Vec::new()).unwrap();
        assert!(regions.len() > 1);
        assert!(graph.closed_calls.len() > 1);
        let facts = Facts::compute(&graph);
        for &inst in graph.closed_calls.keys() {
            assert_eq!(graph.inst_contract(inst, &facts).reentry, Reentry::Closed);
        }
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
        assert!(graph.closed_calls.is_empty());
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
        prove_closed_calls(&mut graph, &bodies, &mut Vec::new());
        assert!(!graph.closed_calls.contains_key(&request.call));
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
        prove_closed_calls(&mut graph, &bodies, &mut Vec::new());
        assert!(graph.closed_calls.contains_key(&request.call));
        assert_eq!(
            graph
                .inst_contract(request.call, &Facts::compute(&graph))
                .reentry,
            Reentry::Closed
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
        ] {
            let (canonical, bodies) = fixture(source);
            let first =
                super::super::plan_canonical(canonical.clone(), true, Some((&bodies, 0))).unwrap();
            let second = super::super::plan_canonical(canonical, true, Some((&bodies, 0))).unwrap();
            assert_eq!(first.inlines.len(), count);
            assert!(first.inlines == second.inlines);
            let analysis = super::super::optimize_planned(first, None).unwrap();
            assert!(analysis.inlines == second.inlines);
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
                    crate::jit::planner::Site {
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
                    crate::jit::planner::Site {
                        source: 0,
                        start: data.span.start,
                        end: data.span.end,
                    },
                    FuncId::from_u32(0),
                );
            }
        }
        for fallback in [None, Some(0x1234)] {
            let planned =
                super::super::plan_canonical(canonical.clone(), true, Some((&bodies, 0))).unwrap();
            let analysis = super::super::optimize_planned(planned, fallback).unwrap();
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
            assert!(fallbacks.len() >= 2);
            assert!(fallbacks.iter().all(|block| *block == fallbacks[0]));
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
        let initial =
            super::super::plan_canonical(canonical.clone(), true, Some((&bodies, 0))).unwrap();
        let recompiled =
            super::super::plan_canonical_with_budget(canonical, true, Some((&bodies, 0)), 2)
                .unwrap();
        assert!(initial.inlines.is_empty());
        assert!(!recompiled.inlines.is_empty());
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
