use std::collections::HashSet;

use super::{BlockData, Cfg, Graph, Op, Target, Terminator, TerminatorKind};

pub(super) fn apply(graph: &mut Graph) -> super::Result<()> {
    let guarded: HashSet<_> = graph
        .insts
        .values()
        .filter_map(|data| match data.op {
            Op::IsCode(value, _) | Op::IsNative(value, _) => Some(value),
            _ => None,
        })
        .collect();
    let reads: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter(|&inst| {
            matches!(
                graph.insts[inst].op,
                Op::Get(_, ref name)
                    if name != "length"
                        && graph.insts[inst]
                            .result
                            .is_some_and(|value| guarded.contains(&value))
            )
        })
        .collect();
    for inst in reads {
        let data = graph.insts[inst].clone();
        let Op::Get(object, name) = data.op else {
            unreachable!();
        };
        let (block, continuation, fallback) = super::inlining::split_call(graph, inst);
        graph.insts[inst].op = Op::FillPropertyCache(object, inst, name.clone());
        let fast = graph.blocks.push(BlockData::default());
        let (check_inst, check) = graph.append(
            block,
            data.span,
            Op::IsPropertyCached(object, inst, name),
            true,
        );
        graph.insts[check_inst].source_name = data.source_name;
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            span: data.span,
            kind: TerminatorKind::Branch(check.unwrap(), Target::new(fast), Target::new(fallback)),
        });
        let (read, value) = graph.append(fast, data.span, Op::CachedProperty(object, inst), true);
        graph.insts[read].source_name = data.source_name;
        graph.blocks[fast].terminator = Some(Terminator {
            source_name: data.source_name,
            span: data.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![value.unwrap()],
            }),
        });
    }
    graph.verify()
}
