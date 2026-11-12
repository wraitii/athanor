use crate::FxHashMap;

use super::guards::Fact;
use super::{Block, BlockData, Graph, Inst, Target, Terminator, TerminatorKind, Value};

pub(super) struct Alternative<T> {
    pub(super) requirements: Vec<(Value, Fact)>,
    pub(super) implementation: T,
}

pub(super) fn split(graph: &mut Graph, inst: Inst) -> (Block, Block, Block) {
    let call = graph.insts[inst].clone();
    let block = call.block;
    let position = graph.blocks[block]
        .insts
        .iter()
        .position(|&current| current == inst)
        .unwrap();
    let continuation = graph.blocks.push(BlockData::default());
    if let Some(original) = call.result() {
        let result = graph.add_param(continuation);
        graph.replace_uses(&FxHashMap::from_iter([(original, result)]));
    }
    let tail = graph.blocks[block].insts.split_off(position + 1);
    graph.blocks[continuation].insts = tail;
    for &inst in &graph.blocks[continuation].insts {
        graph.insts[inst].block = continuation;
    }
    graph.blocks[continuation].terminator = graph.blocks[block].terminator.take();
    let fallback = graph.blocks.push(BlockData::default());
    graph.move_inst(inst, fallback);
    graph.blocks[fallback].terminator = Some(Terminator {
        source_name: call.source_name,
        emission_location: None,
        span: call.span,
        kind: TerminatorKind::Jump(Target {
            block: continuation,
            args: call.result().into_iter().collect(),
        }),
    });
    graph.cold.insert(fallback);
    (block, continuation, fallback)
}

pub(super) fn apply<T>(
    graph: &mut Graph,
    inst: Inst,
    alternatives: Vec<Alternative<T>>,
    mut emit: impl FnMut(&mut Graph, Block, T) -> Option<Value>,
) {
    let alternatives: Vec<_> = alternatives
        .into_iter()
        .filter_map(|alternative| {
            let tests: Option<Vec<_>> = alternative
                .requirements
                .into_iter()
                .map(|(value, fact)| fact.test(value))
                .collect();
            Some((tests?, alternative.implementation))
        })
        .collect();
    if alternatives.is_empty() {
        return;
    }
    let data = graph.insts[inst].clone();
    let (mut current, done, fallback) = split(graph, inst);
    for (tests, implementation) in alternatives {
        let next = graph.blocks.push(BlockData::default());
        for test in tests {
            let check = graph.append_value(current, data.span, test);
            let yes = graph.blocks.push(BlockData::default());
            graph.terminate(
                current,
                data.span,
                TerminatorKind::Branch(check, Target::new(yes), Target::new(next)),
            );
            current = yes;
        }
        let result = emit(graph, current, implementation);
        graph.terminate(
            current,
            data.span,
            TerminatorKind::Jump(Target {
                block: done,
                args: result.into_iter().collect(),
            }),
        );
        current = next;
    }
    graph.terminate(
        current,
        data.span,
        TerminatorKind::Jump(Target::new(fallback)),
    );
}
