use crate::FxHashSet;

use super::super::specialization::regions::Writes;
use super::super::{Block, Facts, Graph, Op, low_level};
use super::loops::{natural_loops, preheader};

pub(in crate::jit::semantic) fn hoist(
    graph: &mut Graph,
    facts: &Facts,
    scope: Option<&FxHashSet<Block>>,
    mut notes: Option<&mut Vec<String>>,
) -> bool {
    let loops = natural_loops(graph, &facts.cfg);
    let mut changed = false;
    for (header, body) in loops {
        if scope.is_some_and(|scope| !body.is_subset(scope)) {
            continue;
        }
        let cfg = &facts.cfg;
        let Some(preheader) = preheader(graph, cfg, header, &body) else {
            continue;
        };
        let writes = scope.map(|_| Writes::with_facts(graph, &body, facts));
        let invariant = |graph: &Graph, op: &Op| {
            op.operands_iter()
                .all(|value| !body.contains(&graph.def_block(value)))
        };
        let constant = |op: &Op| {
            matches!(
                op,
                Op::LowLevel(low_level::Op::Integer { .. } | low_level::Op::Float64(_))
            )
        };
        let mut moved = Vec::new();
        for inst in graph.blocks[header].insts.clone() {
            let data = &graph.insts[inst];
            if matches!(data.op, Op::ReadHint(..)) || constant(&data.op) {
                continue;
            }
            let pure = facts.is_pure(inst);
            if !data.results.is_empty()
                && invariant(graph, &data.op)
                && (pure
                    || writes
                        .as_ref()
                        .is_some_and(|writes| writes.preserves(graph, inst)))
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
                if !data.results.is_empty()
                    && !constant(&data.op)
                    && (facts.is_pure(inst)
                        || (facts.removable(inst)
                            && writes
                                .as_ref()
                                .is_some_and(|writes| writes.preserves(graph, inst))))
                    && invariant(graph, &data.op)
                {
                    moved.push(inst);
                    graph.move_inst(inst, preheader);
                }
            }
        }
        changed |= !moved.is_empty();
        for inst in moved {
            let Some(notes) = &mut notes else {
                continue;
            };
            let data = &graph.insts[inst];
            notes.push(format!(
                "{} = {} hoisted from {header} to {preheader}",
                data.results
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
                graph.op_text(&data.op)
            ));
        }
    }
    changed
}
