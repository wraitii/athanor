use std::collections::{HashMap, HashSet};

use oxc_syntax::operator::BinaryOperator;

use super::super::effects::Locations;
use super::access::{Access, Field};
use super::available;
use super::guards;
use super::js::NUMERIC;
use super::{
    Block, BlockData, Cfg, Constant, Facts, Graph, Inst, Native, Op, Region, RegionFallback,
    RegionId, Result, Target, Terminator, TerminatorKind, TypedRead, Types, UnaryOp, Value,
    ValueDef, resolve,
};
use crate::heap::TypedKind;

const SPECIALIZATION_REGION_LIMIT: usize = 512;

mod analysis;
mod cost;
mod placement;
mod planning;
mod propose;
mod versioning;

use analysis::{Requirement, Writes, closure_failure, conflicts, guarded_accesses, requirements};
#[cfg(test)]
pub(super) use planning::version;
pub(super) use planning::version_with_report;
pub(super) use versioning::repair;
use versioning::{Speculation, apply_version};

pub(super) fn version_string_leaves(
    graph: &mut Graph,
    continuations: bool,
    notes: &mut Vec<String>,
) -> Result<Vec<RegionId>> {
    let initial = Cfg::new(graph);
    let mut sites = Vec::new();
    for &block in &initial.rpo {
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, fast, _),
            ..
        }) = &graph.blocks[block].terminator
        else {
            continue;
        };
        if !matches!(graph.defining_op(*condition), Some(Op::IsStringSplit(_))) {
            continue;
        }
        let Some((receiver, header)) = graph.blocks[fast.block].insts.iter().find_map(|&inst| {
            let Op::StringSplit(receiver, _) = graph.insts[inst].op else {
                return None;
            };
            let Some(Terminator {
                kind: TerminatorKind::Jump(target),
                ..
            }) = &graph.blocks[fast.block].terminator
            else {
                return None;
            };
            Some((receiver, target.block))
        }) else {
            continue;
        };
        let string_branch = initial.rpo.iter().any(|&branch| {
            let Some(Terminator {
                kind: TerminatorKind::Branch(check, yes, no),
                ..
            }) = &graph.blocks[branch].terminator
            else {
                return false;
            };
            matches!(graph.defining_op(*check), Some(Op::IsString(value)) if *value == receiver)
                && yes.block != no.block
                && yes.block != graph.entry
                && initial.preds[yes.block].len() == 1
                && initial.preds[yes.block][0] == branch
                && initial.dominates(yes.block, block)
        });
        if string_branch {
            sites.push((fast.block, header));
        }
    }
    let mut regions = Vec::new();
    for (preheader, header) in sites {
        let cfg = Cfg::new(graph);
        if !cfg.rpo.contains(&preheader) || !cfg.rpo.contains(&header) {
            continue;
        }
        if !matches!(&graph.blocks[preheader].terminator, Some(Terminator { kind: TerminatorKind::Jump(target), .. }) if target.block == header)
        {
            continue;
        }
        let blocks: Vec<_> = cfg
            .rpo
            .iter()
            .copied()
            .filter(|&block| cfg.dominates(header, block))
            .collect();
        let cost: usize = blocks
            .iter()
            .map(|&block| graph.blocks[block].insts.len())
            .sum();
        if cost > SPECIALIZATION_REGION_LIMIT {
            continue;
        }
        let exits: Vec<_> = blocks
            .iter()
            .copied()
            .filter(|&block| cold_exit(graph, block).is_some())
            .collect();
        let exits_true: Vec<_> = blocks
            .iter()
            .copied()
            .filter(|&block| {
                let Some(Terminator {
                    kind: TerminatorKind::Branch(condition, yes, no),
                    ..
                }) = &graph.blocks[block].terminator
                else {
                    return false;
                };
                if yes.block == no.block {
                    return false;
                }
                match graph.defining_op(*condition) {
                    Some(Op::IsUninitialized(_)) => graph.cold.contains(&yes.block),
                    Some(Op::IsUndefined(_)) => true,
                    Some(Op::Binary(BinaryOperator::StrictEquality, left, right)) => {
                        [left, right].into_iter().any(|&value| {
                            matches!(
                                graph.defining_op(value),
                                Some(Op::Constant(Constant::Undefined))
                            )
                        })
                    }
                    _ => false,
                }
            })
            .collect();
        if exits.is_empty() && exits_true.is_empty() {
            continue;
        }
        let mut candidate = graph.clone();
        let speculation = Speculation {
            continuations,
            exits: &exits,
            exits_true: &exits_true,
            ..Default::default()
        };
        let version = apply_version(
            &mut candidate,
            &blocks,
            Some((preheader, header)),
            &speculation,
            None,
        );
        notes.extend(version.notes.iter().cloned());
        let copy = version.body;
        super::js::simplify(&mut candidate);
        candidate.verify()?;
        let reachable: HashSet<_> = Cfg::new(&candidate).rpo.into_iter().collect();
        let copy: HashSet<_> = copy.intersection(&reachable).copied().collect();
        if let Some(reason) = closure_failure(&candidate, &copy, &[], &HashMap::new()) {
            notes.push(format!(
                "string split continuation not versioned: guarded copy {reason}"
            ));
            continue;
        }
        notes.push(format!(
            "string split continuation versioned after native identity check; {} guarded exits",
            exits.len() + exits_true.len()
        ));
        let source = candidate.regions.push(Region {
            blocks: blocks.iter().copied().collect(),
            fallback: None,
        });
        *graph = candidate;
        regions.push(graph.regions.push(Region {
            blocks: copy,
            fallback: Some(RegionFallback::Region(source)),
        }));
        break;
    }
    Ok(regions)
}

fn cold_exit(graph: &Graph, block: Block) -> Option<Block> {
    let Some(Terminator {
        kind: TerminatorKind::Branch(_, _, otherwise),
        ..
    }) = &graph.blocks[block].terminator
    else {
        return None;
    };
    graph
        .cold
        .contains(&otherwise.block)
        .then_some(otherwise.block)
}

fn invariant_guards(
    graph: &Graph,
    inside: &HashSet<Block>,
    scan: &[Block],
    probes: &[Inst],
) -> Vec<Block> {
    let probed: HashSet<Value> = probes
        .iter()
        .filter_map(|inst| graph.insts[*inst].result)
        .collect();
    let invariant =
        |value: Value| !inside.contains(&graph.def_block(value)) || probed.contains(&value);
    scan.iter()
        .copied()
        .filter(|&block| {
            cold_exit(graph, block).is_some()
                && matches!(
                    &graph.blocks[block].terminator,
                    Some(Terminator { kind: TerminatorKind::Branch(condition, ..), .. })
                        if invariant(*condition)
                            || graph.defining_op(*condition).unwrap().operands_iter().all(invariant)
                )
        })
        .collect()
}

fn typed_entry(
    graph: &Graph,
    body: &HashSet<Block>,
    available: &HashSet<Value>,
    facts: &Facts,
    candidates: &[(Value, guards::Fact)],
) -> Vec<(Value, Option<TypedKind>)> {
    let mut typed = HashMap::new();
    for &block in body {
        for &inst in &graph.blocks[block].insts {
            if let Op::GetIndex(object, key) | Op::TypedIndex(object, key, _) = graph.insts[inst].op
                && (!body.contains(&graph.def_block(object)) || available.contains(&object))
                && facts.of(key).within(NUMERIC)
                && let Some(element) = candidates.iter().find_map(|&(value, fact)| match fact {
                    guards::Fact::Typed(element) if value == object => Some(element),
                    _ => None,
                })
            {
                typed.insert(object, element);
            }
        }
    }
    let mut typed: Vec<_> = typed.into_iter().collect();
    typed.sort_by_key(|&(object, _)| object);
    typed
}

pub(super) fn element_guess(graph: &Graph, object: Value) -> Option<TypedKind> {
    graph
        .blocks
        .iter()
        .flat_map(|(_, block)| block.insts.iter())
        .find_map(|&inst| match graph.insts[inst].op {
            Op::ReadHint(value, _, crate::jit::profile::ReadType::Typed(element), _)
                if value == object =>
            {
                Some(element)
            }
            _ => None,
        })
        .or_else(|| element_guess_at(graph, object, 0))
}

fn element_guess_at(graph: &Graph, object: Value, depth: usize) -> Option<TypedKind> {
    if depth > 4 {
        return None;
    }
    if let ValueDef::Param(block, index) = graph.values[object] {
        return graph
            .blocks
            .values()
            .filter_map(|data| data.terminator.as_ref())
            .flat_map(|term| term.kind.targets())
            .filter(|target| target.block == block)
            .find_map(|target| element_guess_at(graph, *target.args.get(index)?, depth + 1));
    }
    if let Some(Op::ConstructorResult(result, _)) = graph.defining_op(object) {
        return element_guess_at(graph, *result, depth + 1);
    }
    let constructor = match graph.defining_op(object)? {
        Op::Construct { callee, .. }
        | Op::ConstructResolved { callee, .. }
        | Op::ConstructorCall { callee, .. } => *callee,
        Op::Call { callee, .. } => match graph.defining_op(*callee)? {
            Op::Get(constructor, name) if name == "from" || name == "of" => *constructor,
            _ => return None,
        },
        _ => return None,
    };
    let Op::GlobalRead(name) = graph.defining_op(constructor)? else {
        return None;
    };
    Some(match name.as_str() {
        "Uint8Array" => TypedKind::Uint8,
        "Uint16Array" => TypedKind::Uint16,
        "Uint32Array" => TypedKind::Uint32,
        "Int8Array" => TypedKind::Int8,
        "Int16Array" => TypedKind::Int16,
        "Int32Array" => TypedKind::Int32,
        "Float32Array" => TypedKind::Float32,
        "Float64Array" => TypedKind::Float64,
        _ => return None,
    })
}

/// Index reads on `objects` inside `region` whose key is a loop counter starting at a
/// non-negative integer, stepping by one and tested against a bound defined before the region,
/// mapped to their receiver and that bound.
fn bounded_reads(
    graph: &Graph,
    cfg: &Cfg,
    region: &HashSet<Block>,
    objects: &[Value],
    available: &HashSet<Value>,
    facts: &Facts,
) -> HashMap<Inst, (Value, Value)> {
    let mut counters = HashMap::new();
    for (header, body) in natural_loops(graph, cfg) {
        if !region.contains(&header) {
            continue;
        }
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, inside, outside),
            ..
        }) = &graph.blocks[header].terminator
        else {
            continue;
        };
        let Some(
            &(Op::Binary(BinaryOperator::LessThan, counter, bound)
            | Op::GuardedCompare {
                operator: BinaryOperator::LessThan,
                left: counter,
                right: bound,
                ..
            }),
        ) = graph.defining_op(*condition)
        else {
            continue;
        };
        if !body.contains(&inside.block)
            || body.contains(&outside.block)
            || region.contains(&graph.def_block(bound)) && !available.contains(&bound)
            || !facts.of(bound).within(Types::NUMBER)
            || !counts_up(graph, cfg, header, &body, counter)
        {
            continue;
        }
        counters.insert(counter, (inside.block, bound));
    }
    let mut reads = HashMap::new();
    for &block in region {
        for &inst in &graph.blocks[block].insts {
            if let Op::GetIndex(object, key) | Op::TypedIndex(object, key, _) = graph.insts[inst].op
                && objects.contains(&object)
                && let Some(&(inside, bound)) = counters.get(&key)
                && cfg.dominates(inside, block)
            {
                reads.insert(inst, (object, bound));
            }
        }
    }
    reads
}

fn counts_up(
    graph: &Graph,
    cfg: &Cfg,
    header: Block,
    body: &HashSet<Block>,
    counter: Value,
) -> bool {
    let Some(index) = graph.blocks[header]
        .params
        .iter()
        .position(|&param| param == counter)
    else {
        return false;
    };
    cfg.preds[header].iter().all(|&pred| {
        let terminator = graph.blocks[pred].terminator.as_ref().unwrap();
        terminator
            .kind
            .targets()
            .iter()
            .filter(|target| target.block == header)
            .all(|target| {
                let arg = target.args[index];
                match graph.defining_op(arg) {
                    Some(Op::Constant(Constant::Number(start))) => {
                        !body.contains(&pred) && *start >= 0.0 && start.fract() == 0.0
                    }
                    Some(&Op::Unary(UnaryOp::Increment, operand)) => operand == counter,
                    Some(&Op::Binary(BinaryOperator::Addition, operand, step)) => {
                        operand == counter
                            && matches!(
                                graph.defining_op(step),
                                Some(Op::Constant(Constant::Number(step))) if *step == 1.0
                            )
                    }
                    _ => false,
                }
            })
    })
}

pub(super) fn preheader(
    graph: &Graph,
    cfg: &Cfg,
    header: Block,
    body: &HashSet<Block>,
) -> Option<Block> {
    let outside: Vec<Block> = cfg.preds[header]
        .iter()
        .copied()
        .filter(|pred| !body.contains(pred))
        .collect();
    let [preheader] = outside[..] else {
        return None;
    };
    matches!(
        graph.blocks[preheader].terminator,
        Some(Terminator {
            kind: TerminatorKind::Jump(_),
            ..
        })
    )
    .then_some(preheader)
}

fn numeric_use(op: &Op, index: usize) -> bool {
    match op {
        Op::GuardedCompare { .. } => index < 2,
        Op::Binary(operator, ..) => !matches!(
            operator,
            BinaryOperator::In
                | BinaryOperator::Instanceof
                | BinaryOperator::Equality
                | BinaryOperator::Inequality
                | BinaryOperator::StrictEquality
                | BinaryOperator::StrictInequality
        ),
        Op::Unary(operator, _) => !matches!(operator, UnaryOp::Not | UnaryOp::Typeof),
        Op::ToNumeric(_) | Op::CallNative(..) => true,
        Op::GetIndex(..)
        | Op::TypedIndex(..)
        | Op::IndexedGet(..)
        | Op::ArrayLoad(..)
        | Op::SetIndex(..) => index == 1,
        _ => false,
    }
}

fn entry_assumes(
    graph: &Graph,
    inside: &HashSet<Block>,
    probes: &[Inst],
    assumed: &HashMap<Value, cranelift_module::FuncId>,
    candidates: &[(Value, guards::Fact)],
) -> Vec<(Value, guards::Fact)> {
    let facts = Facts::assuming(graph, &guards::assumed(&HashMap::new(), assumed));
    let probed: HashSet<Value> = probes
        .iter()
        .filter_map(|inst| graph.insts[*inst].result)
        .collect();
    let mut checks: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|&(value, fact)| {
            if !matches!(fact, guards::Fact::Type(_)) {
                return false;
            }
            let desired = if fact == guards::Fact::Type(Types::INTEGER) {
                Types::INTEGER
            } else {
                NUMERIC
            };
            !facts.of(value).within(desired)
                && (!inside.contains(&graph.def_block(value)) || probed.contains(&value))
        })
        .collect();
    checks.sort_by_key(|&(value, _)| value);
    checks.dedup_by_key(|&mut (value, _)| value);
    checks
}

pub(super) fn reuse_loads(graph: &mut Graph, notes: &mut Vec<String>) {
    if graph.has_handlers || graph.generator {
        return;
    }
    let cfg = Cfg::new(graph);
    let facts = Facts::compute(graph);
    let Some(solved) = available::solve(graph, &facts, &cfg, graph.cold.clone(), false) else {
        return;
    };
    let aliases: HashMap<Value, Value> = solved
        .reused
        .iter()
        .map(|&(inst, value)| (graph.insts[inst].result.unwrap(), value))
        .collect();
    graph.replace_uses(&aliases);
    for &(inst, value) in &solved.reused {
        notes.push(format!(
            "{} reuses {value}",
            graph.op_text(&graph.insts[inst].op)
        ));
        graph.remove_inst(inst);
    }
}

pub(super) fn natural_loops(graph: &Graph, cfg: &Cfg) -> Vec<(Block, HashSet<Block>)> {
    let mut loops: HashMap<Block, HashSet<Block>> = HashMap::new();
    for &block in &cfg.rpo {
        for header in graph.successors(block) {
            if !cfg.dominates(header, block) {
                continue;
            }
            let body = loops
                .entry(header)
                .or_insert_with(|| HashSet::from([header]));
            let mut stack = vec![block];
            while let Some(node) = stack.pop() {
                if body.insert(node) {
                    stack.extend(cfg.preds[node].iter().copied());
                }
            }
        }
    }
    let mut loops: Vec<_> = loops.into_iter().collect();
    loops.sort_by_key(|(header, body)| (body.len(), *header));
    loops
}

pub(super) fn normalise_regions(
    graph: &mut Graph,
    regions: &[RegionId],
    notes: &mut Vec<String>,
) -> Result<()> {
    graph.verify()?;
    super::js::simplify(graph);
    graph.verify()?;
    reuse_loads(graph, notes);
    graph.verify()?;
    for region in regions {
        hoist(graph, *region, notes);
        graph.verify()?;
    }
    Ok(())
}

pub(super) fn hoist(graph: &mut Graph, region: RegionId, notes: &mut Vec<String>) {
    let region = graph.regions[region].blocks.clone();
    let region = &region;
    let cfg = Cfg::new(graph);
    let facts = Facts::compute(graph);
    for (header, body) in natural_loops(graph, &cfg) {
        if !body.is_subset(region) {
            continue;
        }
        let Some(preheader) = preheader(graph, &cfg, header, &body) else {
            continue;
        };
        let writes = Writes::collect(graph, &body);
        let invariant = |graph: &Graph, op: &Op| {
            op.operands_iter()
                .all(|value| !body.contains(&graph.def_block(value)))
        };
        let mut moved = Vec::new();
        for inst in graph.blocks[header].insts.clone() {
            let data = &graph.insts[inst];
            if matches!(data.op, Op::ReadHint(..)) {
                continue;
            }
            let pure = facts.is_pure(inst);
            if data.result.is_some()
                && invariant(graph, &data.op)
                && (pure || writes.preserves(&data.op))
            {
                moved.push(inst);
                graph.move_inst(inst, preheader);
            } else if !pure {
                break;
            }
        }
        for &block in cfg.rpo.iter().filter(|block| body.contains(block)) {
            for inst in graph.blocks[block].insts.clone() {
                let data = &graph.insts[inst];
                if data.result.is_some()
                    && !matches!(data.op, Op::Constant(_))
                    && facts.is_pure(inst)
                    && invariant(graph, &data.op)
                {
                    moved.push(inst);
                    graph.move_inst(inst, preheader);
                }
            }
        }
        for inst in moved {
            let data = &graph.insts[inst];
            notes.push(format!(
                "{} = {} hoisted from {header} to {preheader}",
                data.result.unwrap(),
                graph.op_text(&data.op)
            ));
        }
    }
}

fn probe_reads(
    graph: &Graph,
    inside: &HashSet<Block>,
    written: &HashSet<Block>,
    scan: &[Block],
    inputs: bool,
    assumed: &HashMap<Value, cranelift_module::FuncId>,
) -> Vec<Inst> {
    let writes = Writes::collect_assuming(graph, written, assumed);
    let mut reads = Vec::new();
    let mut probed = HashSet::new();
    for &block in scan {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let input = inputs && matches!(data.op, Op::Argument(_) | Op::This);
            if (input
                || matches!(
                    data.op,
                    Op::Get(..)
                        | Op::ShapedGet(..)
                        | Op::GetIndex(..)
                        | Op::IndexedGet(..)
                        | Op::GlobalRead(_)
                ))
                && data.op.operands_iter().all(|value| {
                    !inside.contains(&graph.def_block(value)) || probed.contains(&value)
                })
                && (input || writes.preserves(&data.op))
            {
                reads.push(inst);
                probed.extend(data.result);
            }
        }
    }
    reads
}

pub(super) fn views(graph: &mut Graph, notes: &mut Vec<String>) {
    let cfg = Cfg::new(graph);
    let facts = Facts::compute(graph);
    let mut loops = natural_loops(graph, &cfg);
    loops.reverse();
    let mut placed: HashMap<(Block, Value), Value> = HashMap::new();
    for &block in &cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let (Op::GetIndex(object, key)
            | Op::TypedIndex(object, key, _)
            | Op::IndexedGet(object, key)
            | Op::ArrayLoad(object, key, _)
            | Op::SetIndex(object, key, _)) = graph.insts[inst].op
            else {
                continue;
            };
            if !facts.of(key).intersects(Types::NUMBER)
                || !facts.of(object).intersects(Types::OBJECT)
            {
                continue;
            }
            let Some(preheader) = loops.iter().find_map(|(header, body)| {
                (body.contains(&block) && !body.contains(&graph.def_block(object)))
                    .then(|| preheader(graph, &cfg, *header, body))
                    .flatten()
            }) else {
                continue;
            };
            let view = match placed.get(&(preheader, object)) {
                Some(&view) => view,
                None => {
                    let span = graph.insts[inst].span;
                    let view = graph.append_value(preheader, span, Op::View(object));
                    notes.push(format!("{view} = view {object} in {preheader}"));
                    placed.insert((preheader, object), view);
                    view
                }
            };
            graph.views.insert(inst, view);
        }
    }
}

pub(super) fn sink_closures(graph: &mut Graph, notes: &mut Vec<String>) {
    let cfg = Cfg::new(graph);
    let loops = natural_loops(graph, &cfg);
    let mut uses: HashMap<Value, HashSet<Block>> = HashMap::new();
    let mut closures = Vec::new();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            if matches!(&data.op, Op::Closure(_, _, _, false, _, _, _)) {
                closures.push(inst);
            }
            for value in data.op.operands_iter() {
                uses.entry(value).or_default().insert(block);
            }
        }
        for value in graph.blocks[block]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .operands_iter()
        {
            uses.entry(value).or_default().insert(block);
        }
    }
    for inst in closures {
        let value = graph.insts[inst].result.unwrap();
        let Some(users) = uses.get(&value) else {
            graph.remove_inst(inst);
            notes.push(format!("unused closure {value} removed after inlining"));
            continue;
        };
        if users.len() != 1 {
            continue;
        }
        let target = *users.iter().next().unwrap();
        let origin = graph.insts[inst].block;
        if target == origin
            || !graph.cold.contains(&target)
            || loops
                .iter()
                .any(|(_, body)| body.contains(&origin) != body.contains(&target))
        {
            continue;
        }
        if matches!(&graph.insts[inst].op, Op::Closure(_, _, _, _, _, captures, _) if !captures.is_empty())
        {
            let user = graph.blocks[target].insts.iter().copied().find(|&user| {
                graph.insts[user]
                    .op
                    .operands_iter()
                    .any(|operand| operand == value)
            });
            if !user
                .is_some_and(|user| super::inlining::direct_captures(graph, user, value).is_some())
            {
                continue;
            }
        }
        graph.remove_inst(inst);
        let position = graph.blocks[target]
            .insts
            .iter()
            .position(|&user| {
                graph.insts[user]
                    .op
                    .operands_iter()
                    .any(|operand| operand == value)
            })
            .unwrap_or(graph.blocks[target].insts.len());
        graph.blocks[target].insts.insert(position, inst);
        graph.insts[inst].block = target;
        notes.push(format!(
            "closure {value} materialized only in fallback {target}"
        ));
    }
}

pub(super) fn forward_captured_parameters(graph: &mut Graph, notes: &mut Vec<String>) {
    let cfg = Cfg::new(graph);
    let mut aliases = HashMap::new();
    let mut removed = Vec::new();
    for &binding in &graph.lazy_captures {
        if graph.scopes[graph.bindings[binding].scope].kind != super::ScopeKind::Parameters {
            continue;
        }
        let mut initializers = Vec::new();
        let mut writes = false;
        let mut escaped = HashSet::new();
        let mut pending = Vec::new();
        for &block in &cfg.rpo {
            for &inst in &graph.blocks[block].insts {
                match &graph.insts[inst].op {
                    Op::Initialize(target, value) if *target == binding => {
                        initializers.push((inst, *value))
                    }
                    Op::Write(target, _) if *target == binding => writes = true,
                    Op::Closure(_, _, _, _, _, captures, _) if captures.contains(&binding) => {
                        pending.push(block)
                    }
                    _ => {}
                }
            }
        }
        if writes || initializers.len() != 1 {
            continue;
        }
        let (initializer, value) = initializers[0];
        let origin = graph.insts[initializer].block;
        while let Some(block) = pending.pop() {
            if escaped.insert(block) {
                pending.extend(graph.successors(block));
            }
        }
        for &block in &cfg.rpo {
            if escaped.contains(&block) || !cfg.dominates(origin, block) {
                continue;
            }
            let mut initialized = block != origin;
            for &inst in &graph.blocks[block].insts {
                if inst == initializer {
                    initialized = true;
                }
                if initialized
                    && matches!(graph.insts[inst].op, Op::Read(target) if target == binding)
                {
                    aliases.insert(graph.insts[inst].result.unwrap(), value);
                    removed.push(inst);
                }
            }
        }
    }
    graph.replace_uses(&aliases);
    for inst in removed {
        notes.push(format!("forwarded captured parameter read at {inst}"));
        graph.remove_inst(inst);
    }
}

pub(super) fn lazy_captures(graph: &mut Graph, notes: &mut Vec<String>) -> bool {
    let mut demoted = false;
    let mut candidates = HashSet::new();
    let mut eager = HashSet::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            match &graph.insts[inst].op {
                Op::Closure(_, _, _, false, _, captures, _) if graph.cold.contains(&block) => {
                    candidates.extend(captures.iter().copied());
                }
                Op::Closure(_, _, _, _, _, captures, _) => eager.extend(captures.iter().copied()),
                _ => {}
            }
        }
    }
    for (binding, data) in &mut graph.bindings {
        if data.captured
            && graph.scopes[data.scope].kind != super::ScopeKind::Captures
            && !eager.contains(&binding)
        {
            if candidates.contains(&binding) {
                if graph.lazy_captures.insert(binding) {
                    notes.push(format!(
                        "binding {binding} materialized as a cell only on closure creation"
                    ));
                }
            } else {
                data.captured = false;
                graph.lazy_captures.remove(&binding);
                demoted = true;
                notes.push(format!("binding {binding} no longer needs a capture cell"));
            }
        }
    }
    demoted
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    fn empty_graph() -> Graph {
        let allocator = Allocator::default();
        let parsed =
            Parser::new(&allocator, "function f(){return 0;}", SourceType::script()).parse();
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        super::super::build::lower(function, semantic.scoping(), &HashSet::new()).unwrap()
    }

    fn heap(base: Value, field: Field) -> Access {
        Access {
            location: Locations::HEAP,
            base: Some(base),
            field,
        }
    }

    fn global(name: &str) -> Access {
        Access {
            location: Locations::GLOBAL,
            base: None,
            field: Field::Named(name.into()),
        }
    }

    #[test]
    fn append_writes_preserve_properties_but_invalidate_elements_and_length() {
        for spread in [false, true] {
            let mut graph = empty_graph();
            let block = graph.entry;
            let span = graph.span;
            let array = graph.append_value(block, span, Op::Argument(0));
            let value = graph.append_value(block, span, Op::Argument(1));
            graph.append(block, span, Op::ArrayAppend(array, value, spread), false);
            let scope = Cfg::new(&graph).rpo.into_iter().collect();
            let writes = Writes::collect(&graph, &scope);
            assert!(writes.clobbers(&heap(array, Field::Elements)));
            assert!(writes.clobbers(&heap(array, Field::Length)));
            assert!(!writes.clobbers(&heap(array, Field::Named("allows".into()))));
            assert!(!writes.clobbers(&global("g_Map")));
        }
    }

    #[test]
    fn shape_checks_survive_slot_stores_but_not_shape_writes_on_the_object() {
        let mut graph = empty_graph();
        let block = graph.entry;
        let span = graph.span;
        let object = graph.append_value(block, span, Op::Argument(0));
        let value = graph.append_value(block, span, Op::Argument(1));
        let first = graph.append(block, span, Op::IsShape(object, 1), true).1;
        graph.append(
            block,
            span,
            Op::ShapedSet(object, "x".into(), 0, value),
            false,
        );
        let (second, _) = graph.append(block, span, Op::IsShape(object, 1), true);
        graph.append(block, span, Op::Set(object, "y".into(), value), false);
        let (third, _) = graph.append(block, span, Op::IsShape(object, 1), true);
        graph.terminate(block, span, TerminatorKind::Return(first.unwrap()));
        graph.verify().unwrap();
        let facts = Facts::compute(&graph);
        let cfg = Cfg::new(&graph);
        let solved =
            super::super::available::solve(&graph, &facts, &cfg, HashSet::new(), false).unwrap();
        let reused: Vec<Inst> = solved.reused.iter().map(|&(inst, _)| inst).collect();
        assert!(reused.contains(&second));
        assert!(!reused.contains(&third));
    }

    #[test]
    fn pack_mutations_do_not_clobber_javascript_properties() {
        let mut graph = empty_graph();
        let block = graph.entry;
        let span = graph.span;
        let pack = graph.append_value(block, span, Op::NewPack);
        let value = graph.append_value(block, span, Op::Argument(0));
        graph.append(block, span, Op::PackAppend(pack, value, false), false);
        graph.append(block, span, Op::PackRelease(pack), false);
        let scope = Cfg::new(&graph).rpo.into_iter().collect();
        let writes = Writes::collect(&graph, &scope);
        assert!(writes.clobbers(&heap(pack, Field::Packs)));
        assert!(!writes.clobbers(&heap(value, Field::Length)));
        assert!(!writes.clobbers(&heap(value, Field::Named("allows".into()))));
        assert!(!writes.clobbers(&global("g_Map")));
    }

    #[test]
    fn fresh_block_parameters_keep_aliases_and_reject_external_inputs_and_escapes() {
        for external in [false, true] {
            let mut graph = empty_graph();
            let entry = graph.entry;
            let span = graph.span;
            let object = graph.append_value(entry, span, Op::NewObject);
            let other = graph.append_value(entry, span, Op::NewObject);
            let argument = graph.append_value(entry, span, Op::Argument(0));
            let check = graph.append_value(entry, span, Op::IsUndefined(argument));
            let left = graph.blocks.push(BlockData::default());
            let right = graph.blocks.push(BlockData::default());
            let merge = graph.blocks.push(BlockData::default());
            let receiver = graph.add_param(merge);
            graph.terminate(
                entry,
                span,
                TerminatorKind::Branch(check, Target::new(left), Target::new(right)),
            );
            for (block, value) in [
                (left, object),
                (right, if external { argument } else { object }),
            ] {
                graph.terminate(
                    block,
                    span,
                    TerminatorKind::Jump(Target {
                        block: merge,
                        args: vec![value],
                    }),
                );
            }
            graph.append(merge, span, Op::Set(receiver, "x".into(), argument), false);
            graph.terminate(merge, span, TerminatorKind::Return(receiver));
            graph.verify().unwrap();
            let scope = Cfg::new(&graph).rpo.into_iter().collect();
            let writes = Writes::collect(&graph, &scope);
            assert!(writes.clobbers(&heap(object, Field::Named("x".into()))));
            assert_eq!(
                writes.clobbers(&heap(other, Field::Named("x".into()))),
                external
            );
            let mut after = graph.clone();
            after.append(merge, span, Op::GlobalWrite("leak".into(), receiver), false);
            let writes = Writes::collect(&after, &scope);
            assert_eq!(
                writes.clobbers(&heap(other, Field::Named("x".into()))),
                external
            );
            for value in [object, receiver] {
                let mut before = graph.clone();
                before.append(entry, span, Op::GlobalWrite("leak".into(), value), false);
                let writes = Writes::collect(&before, &scope);
                assert!(writes.clobbers(&heap(other, Field::Named("x".into()))));
            }
        }
    }

    #[test]
    fn branches_decided_by_a_dominating_edge_on_the_same_condition_fold() {
        for inverted in [false, true] {
            let mut graph = empty_graph();
            let entry = graph.entry;
            let span = graph.span;
            let argument = graph.append_value(entry, span, Op::Argument(0));
            let check = graph.append_value(entry, span, Op::IsUndefined(argument));
            let [yes, no, inner_yes, inner_no] =
                std::array::from_fn(|_| graph.blocks.push(BlockData::default()));
            graph.terminate(
                entry,
                span,
                TerminatorKind::Branch(check, Target::new(yes), Target::new(no)),
            );
            let repeated = if inverted {
                graph.append_value(yes, span, Op::Unary(UnaryOp::Not, check))
            } else {
                check
            };
            graph.terminate(
                yes,
                span,
                TerminatorKind::Branch(repeated, Target::new(inner_yes), Target::new(inner_no)),
            );
            for block in [no, inner_yes, inner_no] {
                let zero = graph.append_value(block, span, Op::Constant(Constant::Number(0.0)));
                graph.terminate(block, span, TerminatorKind::Return(zero));
            }
            graph.verify().unwrap();
            super::super::js::simplify(&mut graph);
            let reachable: HashSet<Block> = Cfg::new(&graph).rpo.into_iter().collect();
            assert_eq!(reachable.contains(&inner_yes), !inverted);
            assert_eq!(reachable.contains(&inner_no), inverted);
        }
    }

    #[test]
    fn summaries_apply_only_while_the_guarded_global_is_unwritten() {
        use super::super::super::effects::{Control, Memory};
        let code = cranelift_module::FuncId::from_u32(0);
        for written in [false, true] {
            let mut graph = empty_graph();
            graph.summaries = std::sync::Arc::new(HashMap::from([(
                code,
                super::super::CallSummary {
                    memory: Memory::default(),
                    control: Control::default(),
                    requires: vec![("f".into(), code)],
                },
            )]));
            let entry = graph.entry;
            let span = graph.span;
            let probe = graph.append_value(entry, span, Op::ProbeGlobal("f".into()));
            let check = graph.append_value(entry, span, Op::IsCode(probe, code));
            let body = graph.blocks.push(BlockData::default());
            let exit = graph.blocks.push(BlockData::default());
            graph.terminate(
                entry,
                span,
                TerminatorKind::Branch(check, Target::new(body), Target::new(exit)),
            );
            let value = graph.append_value(body, span, Op::Constant(Constant::Number(1.0)));
            if written {
                graph.append(body, span, Op::GlobalWrite("f".into(), value), false);
            }
            let (call, result) = graph.append(
                body,
                span,
                Op::Call {
                    callee: probe,
                    receiver: None,
                    args: Vec::new(),
                    target: None,
                },
                true,
            );
            graph.terminate(body, span, TerminatorKind::Return(result.unwrap()));
            let zero = graph.append_value(exit, span, Op::Constant(Constant::Number(0.0)));
            graph.terminate(exit, span, TerminatorKind::Return(zero));
            graph.verify().unwrap();
            let facts = Facts::compute(&graph);
            assert_eq!(facts.memory(call).is_empty(), !written);
        }
    }

    #[test]
    fn fresh_aliases_survive_loop_backedges() {
        let mut graph = empty_graph();
        let entry = graph.entry;
        let span = graph.span;
        let object = graph.append_value(entry, span, Op::NewObject);
        let header = graph.blocks.push(BlockData::default());
        let receiver = graph.add_param(header);
        graph.terminate(
            entry,
            span,
            TerminatorKind::Jump(Target {
                block: header,
                args: vec![object],
            }),
        );
        graph.terminate(
            header,
            span,
            TerminatorKind::Jump(Target {
                block: header,
                args: vec![receiver],
            }),
        );
        let other = graph.append_value(entry, span, Op::NewObject);
        let one = graph.append_value(header, span, Op::Constant(Constant::Number(1.0)));
        graph.append(header, span, Op::Set(receiver, "x".into(), one), false);
        graph.verify().unwrap();
        let scope = HashSet::from([entry, header]);
        let writes = Writes::collect(&graph, &scope);
        assert!(writes.clobbers(&heap(object, Field::Named("x".into()))));
        assert!(writes.clobbers(&heap(receiver, Field::Named("x".into()))));
        assert!(!writes.clobbers(&heap(other, Field::Named("x".into()))));
        graph.append(
            header,
            span,
            Op::GlobalWrite("leak".into(), receiver),
            false,
        );
        let writes = Writes::collect(&graph, &scope);
        assert!(writes.clobbers(&heap(other, Field::Named("x".into()))));
    }

    #[test]
    fn numeric_and_typed_guards_share_one_loop_copy() {
        let allocator = Allocator::default();
        let source = "function f(width){const grid=new Float64Array(4);let sum=0;for(let y=0;y<2;++y){for(let x=0;x<2;++x)sum+=grid[y*width+x];}return sum;}";
        let parsed = Parser::new(&allocator, source, SourceType::script()).parse();
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let analysis =
            super::super::analyze(function, semantic.scoping(), &HashSet::new()).unwrap();
        let graph = &analysis.graph;
        let loops = natural_loops(graph, &Cfg::new(graph));
        let outer = loops
            .iter()
            .filter(|(header, _)| {
                !loops
                    .iter()
                    .any(|(other, body)| other != header && body.contains(header))
            })
            .count();
        assert_eq!(outer, 1);
        assert!(
            super::super::continuation::bodies(graph)
                .iter()
                .any(|body| !natural_loops(&body.graph, &Cfg::new(&body.graph)).is_empty())
        );
        assert_eq!(
            Cfg::new(graph)
                .rpo
                .iter()
                .flat_map(|&block| &graph.blocks[block].insts)
                .filter(|&&inst| matches!(graph.insts[inst].op, Op::IsTyped(..)))
                .count(),
            1
        );
        assert!(
            analysis
                .notes
                .iter()
                .any(|note| note.contains("combined entry guards"))
        );
        graph.verify().unwrap();
    }

    #[test]
    fn fresh_constructor_result_remains_private_only_until_terminal_return() {
        let allocator = Allocator::default();
        let parsed =
            Parser::new(&allocator, "function f(){return 0;}", SourceType::script()).parse();
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let mut graph =
            super::super::build::lower(function, semantic.scoping(), &HashSet::new()).unwrap();
        let block = graph.entry;
        let span = graph.span;
        let prototype = graph.append_value(block, span, Op::Constant(Constant::Null));
        let object = graph.append_value(block, span, Op::NewInstance(prototype));
        let undefined = graph.append_value(block, span, Op::Constant(Constant::Undefined));
        let result = graph.append_value(block, span, Op::ConstructorResult(undefined, object));
        graph.terminate(block, span, TerminatorKind::Return(result));
        let other = graph.append_value(block, span, Op::NewObject);
        let one = graph.append_value(block, span, Op::Constant(Constant::Number(1.0)));
        graph.append(block, span, Op::Set(object, "x".into(), one), false);
        let scope = HashSet::from([block]);
        let writes = Writes::collect(&graph, &scope);
        assert!(!writes.clobbers(&heap(other, Field::Named("x".into()))));
        let position = graph.blocks[block]
            .insts
            .iter()
            .position(|&inst| matches!(graph.insts[inst].op, Op::Set(..)))
            .unwrap();
        graph.insert(
            block,
            position,
            span,
            Op::GlobalWrite("leak".into(), result),
            false,
        );
        let writes = Writes::collect(&graph, &scope);
        assert!(writes.clobbers(&heap(other, Field::Named("x".into()))));
    }
}
