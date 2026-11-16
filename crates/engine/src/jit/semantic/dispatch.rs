use super::guards::Fact;
use super::{
    Block, BlockData, Graph, Inst, Op, Target, Terminator, TerminatorKind, Value, ValueDef,
};

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
    if graph.cold.contains(&block) {
        graph.cold.insert(continuation);
    }
    let fallback_result = call.result().map(|original| {
        let result = graph.values.push(ValueDef::Result(inst, 0));
        graph.insts[inst].results[0] = result;
        graph.values[original] = ValueDef::Param(continuation, 0);
        graph.blocks[continuation].params.push(original);
        graph.reprs[result] = graph.reprs[original].take();
        if graph.borrowed.remove(&original) {
            graph.borrowed.insert(result);
        }
        result
    });
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
            args: fallback_result.into_iter().collect(),
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

pub(super) fn append(
    graph: &mut Graph,
    source: &super::InstData,
    block: super::Block,
    op: Op,
) -> super::Value {
    let (inst, value) = graph.append(block, source.span, op, true);
    graph.insts[inst].source_name = source.source_name;
    graph.insts[inst].inline_context = source.inline_context;
    value.unwrap()
}

pub(super) fn branch(
    graph: &mut Graph,
    inst: Inst,
    block: Block,
    condition: Value,
    fallback: Block,
) -> Block {
    let fast = graph.blocks.push(BlockData::default());
    graph.terminate(
        block,
        graph.insts[inst].span,
        TerminatorKind::Branch(condition, Target::new(fast), Target::new(fallback)),
    );
    fast
}

pub(super) fn extend_regions(graph: &mut Graph, block: Block, first: usize) {
    let added: Vec<_> = graph.blocks.keys().skip(first).collect();
    if graph.cold.contains(&block) {
        graph.cold.extend(added.iter().copied());
    }
    for region in graph.regions.values_mut() {
        if region.blocks.contains(&block) {
            region.blocks.extend(added.iter().copied());
        }
    }
}
