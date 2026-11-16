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
    cleanup(graph, facts, Some(notes));
    guarded_floats(function);
}

pub(in crate::jit::semantic) fn lowered(graph: &mut Graph) {
    // Scalar operations have explicit effects and no longer need JS value inference.
    let mut facts = Facts::effects(graph);
    cleanup(graph, &mut facts, None);
    if super::licm::hoist(graph, &facts, None, None) {
        facts.refresh(graph);
        cleanup(graph, &mut facts, None);
    }
    super::dce::eliminate(graph, &facts);
}

fn cleanup(graph: &mut Graph, facts: &mut Facts, mut notes: Option<&mut Vec<String>>) -> bool {
    let expressions = super::cse::eliminate(graph, facts, notes.as_deref_mut());
    if expressions {
        // CSE preserves value types, but aliases can change memory dependencies.
        facts.compute_effects(graph);
    }
    if !reuse(graph, facts, notes.as_deref_mut()) {
        return expressions;
    }
    facts.refresh(graph);
    if super::cse::eliminate(graph, facts, notes) {
        facts.compute_effects(graph);
    }
    true
}

fn reuse(graph: &mut Graph, facts: &Facts, mut notes: Option<&mut Vec<String>>) -> bool {
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
        if let Some(notes) = &mut notes {
            notes.push(format!("{} reuses {value}", graph.note_inst(inst)));
        }
    }
    graph.remove_insts(solved.reused.iter().map(|&(inst, _)| inst));
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
