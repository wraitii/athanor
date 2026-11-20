use crate::jit::semantic::JsOp;
use crate::{FxHashMap, FxHashSet};

use super::super::{Graph, Op, Value};

pub(in crate::jit::semantic) fn dead_allocations(graph: &mut Graph) -> bool {
    let uses = graph.use_counts();
    let insts = graph
        .blocks
        .values()
        .flat_map(|block| block.insts.iter().copied())
        .collect::<Vec<_>>();
    let mut dead = insts
        .iter()
        .filter(|&&inst| matches!(graph.insts[inst].op, Op::Javascript(JsOp::NewObject)))
        .filter_map(|&inst| graph.insts[inst].result())
        .collect::<FxHashSet<Value>>();
    loop {
        let mut defines = dead
            .iter()
            .map(|&value| (value, 0))
            .collect::<FxHashMap<Value, u32>>();
        for &inst in &insts {
            let Op::Javascript(JsOp::Define(object, _, value)) = graph.insts[inst].op else {
                continue;
            };
            if !dead.contains(&object) {
                continue;
            }
            *defines.get_mut(&object).unwrap() += 1;
            if let Some(count) = defines.get_mut(&value) {
                *count += 1;
            }
        }
        let before = dead.len();
        dead.retain(|value| defines[value] == uses[*value]);
        if dead.len() == before {
            break;
        }
    }
    if dead.is_empty() {
        return false;
    }
    for inst in insts {
        let removed = match &graph.insts[inst].op {
            Op::Javascript(JsOp::NewObject) => graph.insts[inst]
                .result()
                .is_some_and(|value| dead.contains(&value)),
            Op::Javascript(JsOp::Define(object, ..)) | Op::Assume(object, ..) => {
                dead.contains(object)
            }
            _ => false,
        };
        if removed {
            graph.remove_inst(inst);
        }
    }
    true
}
