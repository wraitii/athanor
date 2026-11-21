use super::super::low_level::{Binary, Op as LowOp};
use super::super::ranges::integer_literal;
use super::super::{
    Block, BlockData, Cfg, Graph, Inst, InstData, Op, Target, TerminatorKind, Value, ValueDef,
};
use super::loops::{natural_loops, preheader};
use crate::jit::semantic::JsOp;
use crate::{FxHashMap, FxHashSet};
use cranelift_codegen::ir::condcodes::{CondCode, IntCC};

const MAX_TRIPS: i64 = 8;
const MAX_INSTS: usize = 512;

struct Counted {
    preheader: Block,
    header: Block,
    blocks: Vec<Block>,
    inside: Target,
    exit: Target,
    trips: i64,
}

/// Fully unrolls innermost loops with a small constant trip count whose body has no calls or
/// side exits, so scalar cleanup sees straight-line code with constant induction values.
pub(in crate::jit::semantic) fn unroll(graph: &mut Graph) -> bool {
    let mut changed = false;
    loop {
        let cfg = Cfg::new(graph);
        let loops = natural_loops(graph, &cfg);
        let Some(counted) = loops.iter().find_map(|(header, body)| {
            let nested = loops
                .iter()
                .any(|(other, _)| other != header && body.contains(other));
            (!nested)
                .then(|| counted(graph, &cfg, *header, body))
                .flatten()
        }) else {
            return changed;
        };
        expand(graph, counted);
        changed = true;
    }
}

fn counted(graph: &Graph, cfg: &Cfg, header: Block, body: &FxHashSet<Block>) -> Option<Counted> {
    let preheader = preheader(graph, cfg, header, body)?;
    let TerminatorKind::Branch(condition, yes, no) =
        &graph.blocks[header].terminator.as_ref()?.kind
    else {
        return None;
    };
    let (inside, exit, stays) = match (body.contains(&yes.block), body.contains(&no.block)) {
        (true, false) => (yes, no, true),
        (false, true) => (no, yes, false),
        _ => return None,
    };
    let latches: Vec<Block> = cfg.preds[header]
        .iter()
        .copied()
        .filter(|pred| body.contains(pred))
        .collect();
    let [latch] = latches[..] else {
        return None;
    };
    let TerminatorKind::Jump(back) = &graph.blocks[latch].terminator.as_ref()?.kind else {
        return None;
    };
    let TerminatorKind::Jump(entry) = &graph.blocks[preheader].terminator.as_ref()?.kind else {
        return None;
    };
    let blocks: Vec<Block> = cfg
        .rpo
        .iter()
        .copied()
        .filter(|block| body.contains(block))
        .collect();
    let mut size = 0;
    for &block in &blocks {
        if graph.cold.contains(&block) || block != header && !enclosed(graph, body, block) {
            return None;
        }
        let insts = &graph.blocks[block].insts;
        if !insts.iter().all(|&inst| cloneable(&graph.insts[inst].op)) {
            return None;
        }
        size += insts.len();
    }
    let Some(Op::LowLevel(LowOp::IntCompare {
        condition: compare,
        left,
        right,
    })) = graph.defining_op(*condition)
    else {
        return None;
    };
    let (index, limit, compare) = match (
        integer_literal(graph, *left),
        integer_literal(graph, *right),
    ) {
        (None, Some(limit)) => (*left, limit, *compare),
        (Some(limit), None) => (*right, limit, compare.swap_args()),
        _ => return None,
    };
    let compare = if stays { compare } else { compare.complement() };
    let position = graph.blocks[header]
        .params
        .iter()
        .position(|&param| param == index)?;
    let start = integer_literal(graph, entry.args[position])?;
    let Some(Op::LowLevel(LowOp::Binary { op, left, right })) =
        graph.defining_op(back.args[position])
    else {
        return None;
    };
    let step = match (op, integer_literal(graph, *right)) {
        (Binary::Add, Some(step)) if *left == index => step,
        (Binary::Subtract, Some(step)) if *left == index => -step,
        _ => return None,
    };
    let trips = trips(start, step, limit, compare)?;
    if trips as usize * size > MAX_INSTS {
        return None;
    }
    Some(Counted {
        preheader,
        header,
        blocks,
        inside: inside.clone(),
        exit: exit.clone(),
        trips,
    })
}

fn enclosed(graph: &Graph, body: &FxHashSet<Block>, block: Block) -> bool {
    matches!(
        graph.blocks[block].terminator.as_ref().map(|t| &t.kind),
        Some(TerminatorKind::Jump(_) | TerminatorKind::Branch(..))
    ) && graph.successors(block).all(|next| body.contains(&next))
}

fn cloneable(op: &Op) -> bool {
    match op {
        Op::LowLevel(op) => matches!(
            op,
            LowOp::Integer { .. }
                | LowOp::Float64(_)
                | LowOp::Unary { .. }
                | LowOp::Binary { .. }
                | LowOp::IntCompare { .. }
                | LowOp::FloatCompare { .. }
                | LowOp::Select { .. }
                | LowOp::ExtractBits { .. }
                | LowOp::InRange { .. }
                | LowOp::TestMask { .. }
                | LowOp::Load { .. }
                | LowOp::Store { .. }
        ),
        Op::Memory(_) | Op::Convert(..) | Op::KeepAlive(_) => true,
        Op::Javascript(JsOp::Constant(_)) => true,
        _ => false,
    }
}

fn trips(start: i64, step: i64, limit: i64, compare: IntCC) -> Option<i64> {
    let holds: fn(i64, i64) -> bool = match compare {
        IntCC::SignedLessThan => |value, limit| value < limit,
        IntCC::SignedLessThanOrEqual => |value, limit| value <= limit,
        IntCC::SignedGreaterThan => |value, limit| value > limit,
        IntCC::SignedGreaterThanOrEqual => |value, limit| value >= limit,
        IntCC::NotEqual => |value, limit| value != limit,
        _ => return None,
    };
    let mut value = start;
    let mut trips = 0;
    while holds(value, limit) {
        trips += 1;
        value += step;
        if trips > MAX_TRIPS || value < i32::MIN as i64 || value > i32::MAX as i64 {
            return None;
        }
    }
    Some(trips)
}

/// Each copy of the header jumps straight into the body, except the last which exits; only header
/// values can be used after the loop, as the header is its only exit.
fn expand(graph: &mut Graph, counted: Counted) {
    let Counted {
        preheader,
        header,
        blocks,
        inside,
        exit,
        trips,
    } = counted;
    let TerminatorKind::Jump(entry) = &graph.blocks[preheader].terminator.as_ref().unwrap().kind
    else {
        unreachable!()
    };
    let mut incoming = entry.args.clone();
    let first = graph.blocks.len();
    let copies: Vec<FxHashMap<Block, Block>> = (0..=trips)
        .map(|_| {
            blocks
                .iter()
                .map(|&block| (block, graph.blocks.push(BlockData::default())))
                .collect()
        })
        .collect();
    let mut values: FxHashMap<Value, Value> = FxHashMap::default();
    for (iteration, copy) in copies.iter().enumerate() {
        let last = iteration as i64 == trips;
        for (&param, &arg) in graph.blocks[header].params.clone().iter().zip(&incoming) {
            values.insert(param, arg);
        }
        for &block in &blocks {
            if last && block != header {
                continue;
            }
            let target = copy[&block];
            if block != header {
                for &param in &graph.blocks[block].params.clone() {
                    let value = graph.add_param(target);
                    graph.reprs[value] = graph.reprs[param];
                    values.insert(param, value);
                }
            }
            for inst in graph.blocks[block].insts.clone() {
                clone_inst(graph, inst, target, &mut values);
            }
        }
        for &block in &blocks {
            if last && block != header {
                continue;
            }
            let mut terminator = graph.blocks[block].terminator.clone().unwrap();
            if block == header {
                let target = if last { &exit } else { &inside };
                terminator.kind = TerminatorKind::Jump(target.clone());
            }
            for value in terminator.kind.operands_mut() {
                *value = values.get(value).copied().unwrap_or(*value);
            }
            for target in terminator.kind.targets_mut() {
                if target.block == header {
                    incoming = std::mem::take(&mut target.args);
                    target.block = copies[iteration + 1][&header];
                } else if let Some(&next) = copy.get(&target.block) {
                    target.block = next;
                }
            }
            graph.blocks[copy[&block]].terminator = Some(terminator);
        }
    }
    let start = copies[0][&header];
    let terminator = graph.blocks[preheader].terminator.as_mut().unwrap();
    terminator.kind = TerminatorKind::Jump(Target::new(start));
    let mut outside: FxHashMap<Value, Value> = FxHashMap::default();
    let defined = graph.blocks[header]
        .params
        .iter()
        .copied()
        .chain(
            graph.blocks[header]
                .insts
                .iter()
                .flat_map(|&inst| graph.insts[inst].results.clone()),
        )
        .collect::<Vec<_>>();
    for value in defined {
        outside.insert(value, values[&value]);
    }
    graph.replace_uses(&outside);
    super::super::dispatch::extend_regions(graph, header, first);
}

pub(super) fn clone_inst(
    graph: &mut Graph,
    inst: Inst,
    block: Block,
    values: &mut FxHashMap<Value, Value>,
) {
    let mut data: InstData = graph.insts[inst].clone();
    for value in data.op.operands_mut() {
        *value = values.get(value).copied().unwrap_or(*value);
    }
    let copy = graph.insts.next_key();
    let results = data.results.clone();
    data.block = block;
    data.results = results
        .iter()
        .enumerate()
        .map(|(index, &result)| {
            let value = graph.values.push(ValueDef::Result(copy, index));
            graph.reprs[value] = graph.reprs[result];
            if graph.canonical_floats.contains(&result) {
                graph.canonical_floats.insert(value);
            }
            if graph.borrowed.contains(&result) {
                graph.borrowed.insert(value);
            }
            values.insert(result, value);
            value
        })
        .collect();
    graph.insts.push(data);
    graph.blocks[block].insts.push(copy);
}
