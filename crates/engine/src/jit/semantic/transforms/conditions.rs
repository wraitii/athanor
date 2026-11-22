use cranelift_codegen::entity::SecondaryMap;

use super::super::low_level::{Binary, Op as LowLevel};
use super::super::{
    Block, BlockData, Cfg, Graph, Inst, Op, Target, Terminator, TerminatorKind, Value, ValueDef,
};
use super::simplify::{fold_constant_branches, fold_dominated_branches, thread_branches};

pub(in crate::jit::semantic) fn split(graph: &mut Graph) -> bool {
    let mut changed = junctions(graph);
    loop {
        let cfg = Cfg::new(graph);
        let mut threaded = fold_constant_branches(graph, &cfg);
        threaded |= fold_dominated_branches(graph, &cfg);
        if threaded {
            changed = true;
            continue;
        }
        if !thread_branches(graph, &cfg) {
            return changed;
        }
        changed = true;
    }
}

fn junctions(graph: &mut Graph) -> bool {
    let uses = graph.use_counts();
    let mut work: Vec<Block> = graph.blocks.keys().collect();
    let mut removed = Vec::new();
    while let Some(block) = work.pop() {
        let Some(Terminator {
            source_name,
            emission_location,
            span,
            kind: TerminatorKind::Branch(condition, yes, no),
        }) = graph.blocks[block].terminator.clone()
        else {
            continue;
        };
        let Some((inst, op, left, right)) = junction(graph, &uses, condition) else {
            continue;
        };
        let middle = graph.blocks.push(BlockData::default());
        if graph.cold.contains(&block) {
            graph.cold.insert(middle);
        }
        let next = Target {
            block: middle,
            args: Vec::new(),
        };
        let (first, second) = match op {
            Binary::BooleanAnd => (next, no.clone()),
            _ => (yes.clone(), next),
        };
        let terminator = |kind| Terminator {
            source_name,
            emission_location,
            span,
            kind,
        };
        graph.blocks[block].terminator =
            Some(terminator(TerminatorKind::Branch(left, first, second)));
        graph.blocks[middle].terminator = Some(terminator(TerminatorKind::Branch(right, yes, no)));
        removed.push(inst);
        work.push(block);
        work.push(middle);
    }
    if removed.is_empty() {
        return false;
    }
    graph.remove_insts(removed);
    true
}

fn junction(
    graph: &Graph,
    uses: &SecondaryMap<Value, u32>,
    condition: Value,
) -> Option<(Inst, Binary, Value, Value)> {
    if uses[condition] != 1 {
        return None;
    }
    let ValueDef::Result(inst, 0) = graph.values[condition] else {
        return None;
    };
    let Op::LowLevel(LowLevel::Binary {
        op: op @ (Binary::BooleanAnd | Binary::BooleanOr),
        left,
        right,
    }) = graph.insts[inst].op
    else {
        return None;
    };
    (predicate(graph, uses, left) && predicate(graph, uses, right))
        .then_some((inst, op, left, right))
}

fn predicate(graph: &Graph, uses: &SecondaryMap<Value, u32>, value: Value) -> bool {
    match graph.defining_op(value) {
        Some(Op::LowLevel(
            LowLevel::IntCompare { .. }
            | LowLevel::FloatCompare { .. }
            | LowLevel::InRange { .. }
            | LowLevel::TestMask { .. },
        )) => true,
        Some(Op::LowLevel(LowLevel::Binary {
            op: Binary::BooleanAnd | Binary::BooleanOr,
            ..
        })) => junction(graph, uses, value).is_some(),
        _ => false,
    }
}
