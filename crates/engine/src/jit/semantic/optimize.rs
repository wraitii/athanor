use std::collections::{HashMap, HashSet};

use oxc_syntax::operator::BinaryOperator;

use super::super::effects::{GLOBAL_WRITE, HEAP_WRITE, Reentry};
use super::{
    Block, BlockData, Cfg, Constant, Facts, Graph, Inst, Native, Op, Result, Target, Terminator,
    TerminatorKind, TypedRead, Types, UnaryOp, Value, ValueDef, resolve,
};
use crate::heap::TypedKind;

const SPECIALIZATION_REGION_LIMIT: usize = 512;

mod analysis;
mod planning;
mod simplify;
mod versioning;

#[cfg(test)]
use analysis::unescaped;
use analysis::{Access, Guarded, Requirement, Writes, closure_failure, requirements};
#[cfg(test)]
pub(super) use planning::version;
pub(super) use planning::version_with_report;
pub(super) use simplify::{bit_tests, coercions, simplify, typeof_comparisons};
pub(super) use versioning::repair;
use versioning::{Speculation, apply_version};

pub(super) fn version_string_leaves(
    graph: &mut Graph,
    notes: &mut Vec<String>,
) -> Result<Vec<HashSet<Block>>> {
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
            .filter(|&block| guard_fallback(graph, block).is_some())
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
        simplify(&mut candidate);
        candidate.verify()?;
        let reachable: HashSet<_> = Cfg::new(&candidate).rpo.into_iter().collect();
        let copy: HashSet<_> = copy.intersection(&reachable).copied().collect();
        if let Some(reason) = closure_failure(&candidate, &copy, &Guarded::default()) {
            notes.push(format!(
                "string split continuation not versioned: guarded copy {reason}"
            ));
            continue;
        }
        notes.push(format!(
            "string split continuation versioned after native identity check; {} guarded exits",
            exits.len() + exits_true.len()
        ));
        *graph = candidate;
        regions.push(copy);
        break;
    }
    Ok(regions)
}

fn guard_fallback(graph: &Graph, block: Block) -> Option<Block> {
    let Some(Terminator {
        kind: TerminatorKind::Branch(condition, _, otherwise),
        ..
    }) = &graph.blocks[block].terminator
    else {
        return None;
    };
    let guard = matches!(
        graph.defining_op(*condition),
        Some(
            Op::IsCode(..)
                | Op::IsNumber(_)
                | Op::IsInteger(_)
                | Op::IsTyped(..)
                | Op::IsIndexed(_)
                | Op::IsArrayIndex(..)
                | Op::IsPropertyCached(..)
                | Op::IsArrayIteration(..)
                | Op::IsStringSplit(_)
        )
    );
    (guard && graph.cold.contains(&otherwise.block)).then_some(otherwise.block)
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
            guard_fallback(graph, block).is_some()
                && matches!(
                    &graph.blocks[block].terminator,
                    Some(Terminator { kind: TerminatorKind::Branch(condition, ..), .. })
                        if invariant(*condition)
                            || graph.defining_op(*condition).unwrap().operands_iter().all(invariant)
                )
        })
        .collect()
}

const NUMERIC: Types = Types::NUMBER.union(Types::UNDEFINED);

fn typed_objects(
    graph: &Graph,
    body: &HashSet<Block>,
    available: &HashSet<Value>,
    facts: &Facts,
) -> Vec<Value> {
    let mut objects = HashSet::new();
    for &block in body {
        for &inst in &graph.blocks[block].insts {
            if let Op::GetIndex(object, key) | Op::TypedIndex(object, key, _) = graph.insts[inst].op
                && (!body.contains(&graph.def_block(object)) || available.contains(&object))
                && facts.of(key).within(NUMERIC)
                && element_guess(graph, object).is_some()
            {
                objects.insert(object);
            }
        }
    }
    let mut objects: Vec<_> = objects.into_iter().collect();
    objects.sort();
    objects
}

pub(super) fn element_guess(graph: &Graph, object: Value) -> Option<TypedKind> {
    graph
        .blocks
        .iter()
        .flat_map(|(_, block)| block.insts.iter())
        .find_map(|&inst| match graph.insts[inst].op {
            Op::ReadHint(value, _, crate::jit::planner::ReadType::Typed(element), _)
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

fn entry_checks(
    graph: &Graph,
    inside: &HashSet<Block>,
    scan: &[Block],
    probes: &[Inst],
) -> Vec<(Value, Types)> {
    let facts = Facts::compute(graph);
    let probed: HashSet<Value> = probes
        .iter()
        .filter_map(|inst| graph.insts[*inst].result)
        .collect();
    let hints = super::observations::numeric_hints(graph);
    let mut checks = HashSet::new();
    for &block in scan {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            for (index, value) in op.operands_iter().enumerate() {
                let desired = if hints.get(&value) == Some(&crate::jit::planner::ReadType::Integer)
                {
                    Types::INTEGER
                } else {
                    NUMERIC
                };
                if hints.get(&value) != Some(&crate::jit::planner::ReadType::Unknown)
                    && numeric_use(op, index)
                    && !facts.of(value).within(desired)
                    && (!inside.contains(&graph.def_block(value)) || probed.contains(&value))
                    && !matches!(graph.defining_op(value), Some(Op::Constant(_)))
                {
                    checks.insert(value);
                }
            }
        }
    }
    let mut checks: Vec<Value> = checks.into_iter().collect();
    checks.sort();
    checks
        .into_iter()
        .map(|value| {
            let types = if hints.get(&value) == Some(&crate::jit::planner::ReadType::Integer) {
                Types::INT64
            } else {
                Types::NUMBER
            };
            (value, types)
        })
        .collect()
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum LoadKey {
    ToInt32(Value),
    Constructor(Value),
    Property(Value, String),
    Index(Value, Value),
    Global(String),
    This,
    Argument(usize),
    Binary(BinaryOperator, Value, Value),
}

pub(super) fn reuse_loads(graph: &mut Graph, region: &HashSet<Block>, notes: &mut Vec<String>) {
    let cfg = Cfg::new(graph);
    let writes = Writes::collect(graph, region);
    let facts = &writes.facts;
    let mut aliases = HashMap::new();
    let mut available: HashMap<LoadKey, Vec<(Block, usize, Value)>> = HashMap::new();
    let mut reused = Vec::new();
    for &block in cfg.rpo.iter().filter(|block| region.contains(block)) {
        for (index, &inst) in graph.blocks[block].insts.iter().enumerate() {
            let data = &graph.insts[inst];
            let Some(result) = data.result else {
                continue;
            };
            let key = match &data.op {
                Op::Get(object, name) => {
                    LoadKey::Property(resolve(&aliases, *object), name.clone())
                }
                Op::GetIndex(object, key)
                | Op::TypedIndex(object, key, _)
                | Op::IndexedGet(object, key)
                | Op::ArrayLoad(object, key, _) => {
                    LoadKey::Index(resolve(&aliases, *object), resolve(&aliases, *key))
                }
                Op::ConstructorPrototype(callee) => {
                    LoadKey::Constructor(resolve(&aliases, *callee))
                }
                Op::GlobalRead(name) => LoadKey::Global(name.clone()),
                Op::ToInt32(value) => LoadKey::ToInt32(resolve(&aliases, *value)),
                Op::This => LoadKey::This,
                Op::Argument(index) => LoadKey::Argument(*index),
                Op::Binary(operator, left, right)
                | Op::GuardedCompare {
                    operator,
                    left,
                    right,
                    ..
                } if graph.contract(&data.op, facts).is_pure() => LoadKey::Binary(
                    *operator,
                    resolve(&aliases, *left),
                    resolve(&aliases, *right),
                ),
                _ => continue,
            };
            let preserved = match &key {
                LoadKey::Property(object, _)
                | LoadKey::Index(object, _)
                | LoadKey::Constructor(object) => {
                    !writes.clobbers(*object, &Access::of(&data.op, facts))
                }
                LoadKey::Global(name) => !writes.clobbers_global(name),
                LoadKey::This
                | LoadKey::Argument(_)
                | LoadKey::Binary(..)
                | LoadKey::ToInt32(_) => true,
            };
            if !preserved {
                continue;
            }
            let earlier = available.get(&key).and_then(|defs| {
                defs.iter().find(|(def_block, def_index, _)| {
                    if *def_block == block {
                        *def_index < index
                    } else {
                        cfg.dominates(*def_block, block)
                    }
                })
            });
            match earlier {
                Some(&(_, _, value)) => {
                    aliases.insert(result, value);
                    reused.push((inst, value));
                }
                None => available
                    .entry(key)
                    .or_default()
                    .push((block, index, result)),
            }
        }
    }
    graph.replace_uses(&aliases);
    for (inst, value) in reused {
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

pub(super) fn hoist(graph: &mut Graph, region: &HashSet<Block>, notes: &mut Vec<String>) {
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
            let pure = graph.contract(&data.op, &facts).is_pure();
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
                    && graph.contract(&data.op, &facts).is_pure()
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
) -> Vec<Inst> {
    let writes = Writes::collect(graph, written);
    let mut reads = Vec::new();
    let mut probed = HashSet::new();
    for &block in scan {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let input = inputs && matches!(data.op, Op::Argument(_) | Op::This);
            if (input
                || matches!(
                    data.op,
                    Op::Get(..) | Op::GetIndex(..) | Op::IndexedGet(..) | Op::GlobalRead(_)
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
                    let view = graph
                        .append(preheader, span, Op::View(object), true)
                        .1
                        .unwrap();
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

    #[test]
    fn append_writes_preserve_properties_but_invalidate_elements_and_length() {
        for spread in [false, true] {
            let mut graph = empty_graph();
            let block = graph.entry;
            let span = graph.span;
            let array = graph.append(block, span, Op::Argument(0), true).1.unwrap();
            let value = graph.append(block, span, Op::Argument(1), true).1.unwrap();
            graph.append(block, span, Op::ArrayAppend(array, value, spread), false);
            let scope = Cfg::new(&graph).rpo.into_iter().collect();
            let writes = Writes::collect(&graph, &scope);
            assert!(writes.clobbers(array, &Access::Elements));
            assert!(writes.clobbers(array, &Access::Length));
            assert!(!writes.clobbers(array, &Access::Named("allows".into())));
            assert!(!writes.clobbers_global("g_Map"));
        }
    }

    #[test]
    fn pack_mutations_do_not_clobber_javascript_properties() {
        let mut graph = empty_graph();
        let block = graph.entry;
        let span = graph.span;
        let pack = graph.append(block, span, Op::NewPack, true).1.unwrap();
        let value = graph.append(block, span, Op::Argument(0), true).1.unwrap();
        graph.append(block, span, Op::PackAppend(pack, value, false), false);
        graph.append(block, span, Op::PackRelease(pack), false);
        let scope = Cfg::new(&graph).rpo.into_iter().collect();
        let writes = Writes::collect(&graph, &scope);
        assert!(writes.clobbers(pack, &Access::Packs));
        assert!(!writes.clobbers(value, &Access::Length));
        assert!(!writes.clobbers(value, &Access::Named("allows".into())));
        assert!(!writes.clobbers_global("g_Map"));
    }

    #[test]
    fn fresh_block_parameters_keep_aliases_and_reject_external_inputs_and_escapes() {
        for external in [false, true] {
            let mut graph = empty_graph();
            let entry = graph.entry;
            let span = graph.span;
            let object = graph.append(entry, span, Op::NewObject, true).1.unwrap();
            let other = graph.append(entry, span, Op::NewObject, true).1.unwrap();
            let argument = graph.append(entry, span, Op::Argument(0), true).1.unwrap();
            let check = graph
                .append(entry, span, Op::IsUndefined(argument), true)
                .1
                .unwrap();
            let left = graph.blocks.push(BlockData::default());
            let right = graph.blocks.push(BlockData::default());
            let merge = graph.blocks.push(BlockData::default());
            let receiver = graph.add_param(merge);
            graph.blocks[entry].terminator = Some(Terminator {
                source_name: None,
                span,
                kind: TerminatorKind::Branch(check, Target::new(left), Target::new(right)),
            });
            for (block, value) in [
                (left, object),
                (right, if external { argument } else { object }),
            ] {
                graph.blocks[block].terminator = Some(Terminator {
                    source_name: None,
                    span,
                    kind: TerminatorKind::Jump(Target {
                        block: merge,
                        args: vec![value],
                    }),
                });
            }
            graph.append(merge, span, Op::Set(receiver, "x".into(), argument), false);
            graph.blocks[merge].terminator = Some(Terminator {
                source_name: None,
                span,
                kind: TerminatorKind::Return(receiver),
            });
            graph.verify().unwrap();
            let scope = Cfg::new(&graph).rpo.into_iter().collect();
            let fresh = unescaped(&graph, &scope, &Cfg::new(&graph));
            assert_eq!(fresh.contains_key(&object), !external);
            assert_eq!(fresh.contains_key(&receiver), !external);
            let writes = Writes::collect(&graph, &scope);
            assert!(writes.clobbers(object, &Access::Named("x".into())));
            assert_eq!(writes.clobbers(other, &Access::Named("x".into())), external);
            for value in [object, receiver] {
                let mut escaped = graph.clone();
                escaped.append(merge, span, Op::GlobalWrite("leak".into(), value), false);
                let fresh = unescaped(&escaped, &scope, &Cfg::new(&escaped));
                assert!(!fresh.contains_key(&object));
                assert!(!fresh.contains_key(&receiver));
            }
            let partial = HashSet::from([left, right, merge]);
            assert!(!unescaped(&graph, &partial, &Cfg::new(&graph)).contains_key(&receiver));
        }
    }

    #[test]
    fn fresh_aliases_survive_loop_backedges() {
        let mut graph = empty_graph();
        let entry = graph.entry;
        let span = graph.span;
        let object = graph.append(entry, span, Op::NewObject, true).1.unwrap();
        let header = graph.blocks.push(BlockData::default());
        let receiver = graph.add_param(header);
        graph.blocks[entry].terminator = Some(Terminator {
            source_name: None,
            span,
            kind: TerminatorKind::Jump(Target {
                block: header,
                args: vec![object],
            }),
        });
        graph.blocks[header].terminator = Some(Terminator {
            source_name: None,
            span,
            kind: TerminatorKind::Jump(Target {
                block: header,
                args: vec![receiver],
            }),
        });
        graph.verify().unwrap();
        let scope = HashSet::from([entry, header]);
        let fresh = unescaped(&graph, &scope, &Cfg::new(&graph));
        assert!(fresh.contains_key(&object));
        assert_eq!(fresh.get(&object), fresh.get(&receiver));
        graph.append(
            header,
            span,
            Op::GlobalWrite("leak".into(), receiver),
            false,
        );
        assert!(unescaped(&graph, &scope, &Cfg::new(&graph)).is_empty());
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
        let prototype = graph
            .append(block, span, Op::Constant(Constant::Null), true)
            .1
            .unwrap();
        let object = graph
            .append(block, span, Op::NewInstance(prototype), true)
            .1
            .unwrap();
        let undefined = graph
            .append(block, span, Op::Constant(Constant::Undefined), true)
            .1
            .unwrap();
        let result = graph
            .append(block, span, Op::ConstructorResult(undefined, object), true)
            .1
            .unwrap();
        graph.blocks[block].terminator = Some(Terminator {
            source_name: None,
            span,
            kind: TerminatorKind::Return(result),
        });
        let scope = HashSet::from([block]);
        assert!(unescaped(&graph, &scope, &Cfg::new(&graph)).contains_key(&object));
        graph.append(block, span, Op::GlobalWrite("leak".into(), result), false);
        assert!(!unescaped(&graph, &scope, &Cfg::new(&graph)).contains_key(&object));
    }
}
