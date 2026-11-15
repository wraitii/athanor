use super::super::{Facts, FunctionState, Graph, Op, TerminatorKind, Value, available};
use crate::FxHashMap;

pub(in crate::jit::semantic) fn reuse_loads(function: &mut FunctionState) {
    let graph = &function.graph;
    if graph.has_handlers || graph.generator {
        return;
    }
    function.ensure_facts();
    let FunctionState {
        graph,
        facts,
        notes,
        ..
    } = function;
    if reuse(graph, facts, notes) {
        function.invalidate_values();
    }
    guarded_floats(function);
}

pub(in crate::jit::semantic) fn lowered(graph: &mut Graph) {
    let facts = Facts::compute(graph);
    let mut notes = Vec::new();
    reuse(graph, &facts, &mut notes);
    let facts = Facts::compute(graph);
    if super::licm::hoist(graph, &facts, None, &mut notes) {
        let facts = Facts::compute(graph);
        reuse(graph, &facts, &mut notes);
    }
}

fn reuse(graph: &mut Graph, facts: &Facts, notes: &mut Vec<String>) -> bool {
    let cfg = &facts.cfg;
    let Some(solved) = available::solve(graph, facts, cfg, graph.cold.clone(), false) else {
        return false;
    };
    let aliases: FxHashMap<Value, Value> = solved
        .reused
        .iter()
        .map(|&(inst, value)| (graph.insts[inst].result().unwrap(), value))
        .collect();
    graph.replace_uses(&aliases);
    for &(inst, value) in &solved.folded {
        graph.set_op(inst, Op::Constant(super::super::Constant::Boolean(value)));
    }
    for &(inst, value) in &solved.reused {
        notes.push(format!("{} reuses {value}", graph.note_inst(inst)));
        graph.remove_inst(inst);
    }
    !solved.reused.is_empty() || !solved.folded.is_empty()
}

fn guarded_floats(function: &mut FunctionState) {
    let graph = &mut function.graph;
    let cfg = &function.facts.cfg;
    let guards: Vec<_> = cfg
        .rpo
        .iter()
        .filter_map(|&block| {
            let TerminatorKind::Branch(condition, yes, no) =
                &graph.blocks[block].terminator.as_ref()?.kind
            else {
                return None;
            };
            let Op::IntegerCheck(_, float, exact) = graph.defining_op(*condition)? else {
                return None;
            };
            (yes.block != no.block && cfg.preds[yes.block][..] == [block])
                .then_some((yes.block, *float, *exact))
        })
        .collect();
    let mut changed = false;
    for &block in &cfg.rpo {
        // The successful check proves equality without losing the sign of zero.
        let aliases: FxHashMap<_, _> = guards
            .iter()
            .filter(|(success, _, _)| cfg.dominates(*success, block))
            .map(|&(_, float, exact)| (exact, float))
            .collect();
        if aliases.is_empty() {
            continue;
        }
        for inst in graph.blocks[block].insts.clone() {
            for value in graph.insts[inst].op.operands_mut() {
                if let Some(&original) = aliases.get(value) {
                    *value = original;
                    changed = true;
                }
            }
        }
        for value in graph.blocks[block]
            .terminator
            .as_mut()
            .unwrap()
            .kind
            .operands_mut()
        {
            if let Some(&original) = aliases.get(value) {
                *value = original;
                changed = true;
            }
        }
    }
    if changed {
        function.invalidate_values();
    }
}
