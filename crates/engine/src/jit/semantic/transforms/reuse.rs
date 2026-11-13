use super::super::{FunctionState, Op, Value, available};
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
    let cfg = &facts.cfg;
    let Some(solved) = available::solve(graph, facts, cfg, graph.cold.clone(), false) else {
        return;
    };
    let aliases: FxHashMap<Value, Value> = solved
        .reused
        .iter()
        .map(|&(inst, value)| (graph.insts[inst].result().unwrap(), value))
        .collect();
    graph.replace_uses(&aliases);
    for &(inst, value) in &solved.folded {
        graph.insts[inst].op = Op::Constant(super::super::Constant::Boolean(value));
    }
    for &(inst, value) in &solved.reused {
        notes.push(format!("{} reuses {value}", graph.note_inst(inst)));
        graph.remove_inst(inst);
    }
    if !solved.reused.is_empty() || !solved.folded.is_empty() {
        function.invalidate_values();
    }
}
