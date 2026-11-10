use super::{
    BlockData, Cfg, Constant, Facts, Graph, Op, Target, Terminator, TerminatorKind, TypedRead,
    Types,
};

pub(super) fn lower(graph: &mut Graph) -> super::Result<bool> {
    let facts = Facts::compute(graph);
    let reads: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter(|&inst| {
            matches!(graph.insts[inst].op,
                Op::TypedIndex(_, key, TypedRead { element: Some(_), bounded: false })
                if facts.of(key).intersects(Types::NUMBER) && facts.of(key).within(Types::NUMBER))
        })
        .collect();
    let changed = !reads.is_empty();
    for inst in reads {
        let data = graph.insts[inst].clone();
        let Op::TypedIndex(object, key, mut read) = data.op else {
            unreachable!()
        };
        let (block, continuation, load) = super::inlining::split_call(graph, inst);
        graph.cold.remove(&load);
        let bounds = graph.blocks.push(BlockData::default());
        let failure = graph.blocks.push(BlockData::default());
        for region in graph.regions.values_mut() {
            if region.blocks.contains(&block) {
                region.blocks.extend([continuation, load, bounds, failure]);
            }
        }
        let position = append(graph, &data, block, Op::ArrayPosition(key));
        let exact = append(
            graph,
            &data,
            block,
            if facts.of(key).within(Types::INTEGER) {
                Op::Constant(Constant::Boolean(true))
            } else {
                Op::IsExactPosition(key, position)
            },
        );
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            span: data.span,
            kind: TerminatorKind::Branch(exact, Target::new(bounds), Target::new(failure)),
        });
        let length = append(graph, &data, bounds, Op::IndexedLength(object));
        let inside = append(graph, &data, bounds, Op::IsInBounds(position, length));
        graph.blocks[bounds].terminator = Some(Terminator {
            source_name: data.source_name,
            span: data.span,
            kind: TerminatorKind::Branch(inside, Target::new(load), Target::new(failure)),
        });
        read.bounded = true;
        graph.insts[inst].op = Op::TypedIndex(object, position, read);
        let undefined = append(graph, &data, failure, Op::Constant(Constant::Undefined));
        graph.blocks[failure].terminator = Some(Terminator {
            source_name: data.source_name,
            span: data.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![undefined],
            }),
        });
    }
    graph.verify()?;
    Ok(changed)
}

fn append(
    graph: &mut Graph,
    source: &super::InstData,
    block: super::Block,
    op: Op,
) -> super::Value {
    let (inst, value) = graph.append(block, source.span, op, true);
    graph.insts[inst].source_name = source.source_name;
    value.unwrap()
}
