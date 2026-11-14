use crate::FxHashMap;

use super::contracts::{Use, repr_contract};
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
    operands(graph);
    edges(graph);
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

fn mismatched(value: Repr, required: Repr) -> bool {
    matches!(
        (value, required),
        (Repr::Integer(_), Repr::Integer(_)) | (Repr::Float(_), Repr::Float(_))
    ) && value != required
}

fn operands(graph: &mut Graph) {
    for block in graph.blocks.keys().collect::<Vec<_>>() {
        let mut position = 0;
        while position < graph.blocks[block].insts.len() {
            let inst = graph.blocks[block].insts[position];
            if matches!(
                graph.insts[inst].op,
                Op::LowLevel(super::low_level::Op::Call { .. })
            ) {
                position += 1;
                continue;
            }
            let uses = repr_contract(graph, inst).uses;
            let operands: Vec<_> = graph.insts[inst].op.operands_iter().collect();
            for (index, (value, use_)) in operands.into_iter().zip(uses).enumerate() {
                let Use::Exact(required) = use_ else {
                    continue;
                };
                if !mismatched(graph.repr(value), required) {
                    continue;
                }
                let span = graph.insts[inst].span;
                let (_, converted) =
                    graph.insert(block, position, span, Op::Convert(value, required), true);
                position += 1;
                *graph.insts[inst].op.operands_mut().nth(index).unwrap() = converted.unwrap();
            }
            position += 1;
        }
    }
}

fn edges(graph: &mut Graph) {
    for block in graph.blocks.keys().collect::<Vec<_>>() {
        let Some(terminator) = graph.blocks[block].terminator.clone() else {
            continue;
        };
        let mut conversions = Vec::new();
        for (target_index, target) in terminator.kind.targets().enumerate() {
            let params = &graph.blocks[target.block].params;
            for (arg_index, (&arg, &param)) in target.args.iter().zip(params).enumerate() {
                let required = graph.repr(param);
                if mismatched(graph.repr(arg), required) {
                    conversions.push((target_index, arg_index, arg, required));
                }
            }
        }
        for (target_index, arg_index, arg, required) in conversions {
            let position = graph.blocks[block].insts.len();
            let (_, converted) = graph.insert(
                block,
                position,
                terminator.span,
                Op::Convert(arg, required),
                true,
            );
            let terminator = graph.blocks[block].terminator.as_mut().unwrap();
            let target = terminator.kind.targets_mut().nth(target_index).unwrap();
            target.args[arg_index] = converted.unwrap();
        }
    }
}
