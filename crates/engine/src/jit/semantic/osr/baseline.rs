use crate::FxHashSet;
use crate::jit::baseline_tiering::Inputs;
use crate::jit::semantic::{Block, Cfg, Facts, Graph, Op, Result, Target, boundary};

pub(in crate::jit) fn baseline_headers(graph: &Graph) -> FxHashSet<Block> {
    let cfg = Cfg::new(graph);
    let mut headers = FxHashSet::default();
    for &block in &cfg.rpo {
        for header in graph.successors(block) {
            if header != graph.entry && cfg.dominates(header, block) {
                headers.insert(header);
            }
        }
    }
    headers
}

pub(in crate::jit) fn baseline_resume(graph: &Graph, header: Block) -> Result<(Graph, Inputs)> {
    if graph.generator
        || graph.has_handlers
        || graph.bindings.values().any(|binding| binding.captured)
    {
        return Err("baseline continuation requires ordinary binding state".into());
    }
    let cfg = Cfg::new(graph);
    let reachable = Cfg::from_entry(graph, header);
    for &block in &reachable.rpo {
        for &inst in &graph.blocks[block].insts {
            if matches!(
                graph.insts[inst].op,
                Op::Argument(_)
                    | Op::Restored(_)
                    | Op::Callee
                    | Op::Arguments
                    | Op::CallForward(..)
                    | Op::ArgumentCount
                    | Op::Rest(_)
                    | Op::EnterCaptures(..)
                    | Op::Closure(..)
                    | Op::ClosureCells(..)
                    | Op::CellNew(_)
                    | Op::ClosureThis(_)
                    | Op::OsrStart(..)
            ) {
                return Err("baseline continuation requires independent frame state".into());
            }
        }
    }
    let live = boundary::liveness(graph, &cfg.rpo);
    let target = Target {
        block: header,
        args: graph.blocks[header].params.clone(),
    };
    let values = boundary::inputs(&target, &live);
    if values
        .iter()
        .any(|&value| !cfg.dominates(graph.def_block(value), header))
    {
        return Err("baseline continuation input does not dominate its entry".into());
    }
    let scopes = boundary::scopes(graph)[header].clone();
    let bindings = scopes
        .iter()
        .flat_map(|&scope| graph.scopes[scope].bindings.iter().copied())
        .collect::<Vec<_>>();
    let mut result = graph.clone();
    result.name += ".observing";
    result.borrowed.clear();
    boundary::enter(
        &mut result,
        &target,
        &values,
        &bindings,
        &scopes,
        Some(&Facts::compute(graph)),
    );
    boundary::compact(&mut result);
    result.verify()?;
    Ok((result, Inputs { values, bindings }))
}
