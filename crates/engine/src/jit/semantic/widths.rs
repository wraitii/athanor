use crate::FxHashMap;

use super::{Graph, Op, Repr, Value};

pub(in crate::jit) fn choose(graph: &mut Graph) {
    params(graph);
    for repr in graph.reprs.values_mut().flatten() {
        *repr = repr.widest();
    }
    for data in graph.insts.values_mut() {
        if let Op::Convert(_, target) = &mut data.op {
            *target = target.widest();
        }
    }
}

fn concrete(open: Repr, repr: Repr) -> Option<Repr> {
    match (open, repr) {
        (Repr::INTEGER, Repr::Word) => Some(Repr::I64),
        (Repr::INTEGER, Repr::Integer(_)) | (Repr::FLOAT, Repr::Float(_)) if !repr.is_open() => {
            Some(repr)
        }
        _ => None,
    }
}

fn params(graph: &mut Graph) {
    let mut chosen: FxHashMap<Value, Repr> = FxHashMap::default();
    let mut changed = true;
    while changed {
        changed = false;
        for block in graph.blocks.values() {
            let Some(terminator) = &block.terminator else {
                continue;
            };
            for target in terminator.kind.targets() {
                let params = &graph.blocks[target.block].params;
                for (&arg, &param) in target.args.iter().zip(params) {
                    let Some(open) = graph.reprs[param].filter(|repr| repr.is_open()) else {
                        continue;
                    };
                    let width = graph.reprs[arg]
                        .and_then(|repr| concrete(open, repr))
                        .or_else(|| chosen.get(&arg).copied());
                    let Some(width) = width else {
                        continue;
                    };
                    let wider = chosen
                        .get(&param)
                        .is_none_or(|current| current.machine().bits() < width.machine().bits());
                    if wider {
                        chosen.insert(param, width);
                        changed = true;
                    }
                }
            }
        }
    }
    for (param, width) in chosen {
        graph.reprs[param] = Some(width);
    }
}
