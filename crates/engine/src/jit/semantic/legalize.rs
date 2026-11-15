use super::contracts::{Use, repr_contract};
use super::{Block, Cfg, Graph, Op, Repr, TerminatorKind, low_level, repr, widths};

pub(super) fn prepare(graph: &mut Graph) {
    widths::choose(graph);
    let cfg = Cfg::new(graph);
    operands(graph, &cfg.rpo);
    edges(graph, &cfg.rpo);
    conditions(graph, &cfg.rpo);
}

pub(in crate::jit) fn convert(graph: &mut Graph) -> Result<(), String> {
    repr::lower(&mut low_level::Builder::new(graph))
}

fn operands(graph: &mut Graph, blocks: &[Block]) {
    for &block in blocks {
        let mut position = 0;
        while position < graph.blocks[block].insts.len() {
            let inst = graph.blocks[block].insts[position];
            let conversions: smallvec::SmallVec<[(usize, super::Value, super::Repr); 4]> = graph
                .insts[inst]
                .op
                .operands_iter()
                .zip(repr_contract(graph, inst).uses)
                .enumerate()
                .filter_map(|(index, (value, use_))| match use_ {
                    Use::Exact(required) if !graph.repr(value).accepts(required) => {
                        Some((index, value, required))
                    }
                    _ => None,
                })
                .collect();
            for (index, value, required) in conversions {
                let data = &graph.insts[inst];
                let (span, source_name, location) =
                    (data.span, data.source_name, data.emission_location);
                let (conversion, converted) =
                    graph.insert(block, position, span, Op::Convert(value, required), true);
                graph.insts[conversion].source_name = source_name;
                graph.insts[conversion].emission_location = location;
                position += 1;
                *graph.insts[inst].op.operands_mut().nth(index).unwrap() = converted.unwrap();
            }
            position += 1;
        }
    }
}

fn edges(graph: &mut Graph, blocks: &[Block]) {
    for &block in blocks {
        let Some(terminator) = graph.blocks[block].terminator.clone() else {
            continue;
        };
        let mut conversions = Vec::new();
        for (target_index, target) in terminator.kind.targets().enumerate() {
            let params = &graph.blocks[target.block].params;
            for (arg_index, (&arg, &param)) in target.args.iter().zip(params).enumerate() {
                let required = graph.repr(param);
                if !graph.repr(arg).accepts(required) {
                    conversions.push((target_index, arg_index, arg, required));
                }
            }
        }
        for (target_index, arg_index, arg, required) in conversions {
            let position = graph.blocks[block].insts.len();
            let (conversion, converted) = graph.insert(
                block,
                position,
                terminator.span,
                Op::Convert(arg, required),
                true,
            );
            graph.insts[conversion].source_name = terminator.source_name;
            graph.insts[conversion].emission_location = terminator.emission_location;
            let terminator = graph.blocks[block].terminator.as_mut().unwrap();
            let target = terminator.kind.targets_mut().nth(target_index).unwrap();
            target.args[arg_index] = converted.unwrap();
        }
    }
}

fn conditions(graph: &mut Graph, blocks: &[Block]) {
    for &block in blocks {
        let Some(terminator) = graph.blocks[block].terminator.clone() else {
            continue;
        };
        let TerminatorKind::Branch(value, ..) = terminator.kind else {
            continue;
        };
        if graph.repr(value) == Repr::Boolean {
            continue;
        }
        let at = graph.blocks[block].insts.len();
        let (inst, converted) = graph.insert(
            block,
            at,
            terminator.span,
            Op::Convert(value, Repr::Boolean),
            true,
        );
        graph.insts[inst].source_name = terminator.source_name;
        graph.insts[inst].emission_location = terminator.emission_location;
        if let TerminatorKind::Branch(value, ..) =
            &mut graph.blocks[block].terminator.as_mut().unwrap().kind
        {
            *value = converted.unwrap();
        }
    }
}
