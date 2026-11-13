use super::{Block, BlockData, Cfg, Graph, Inst, Op, Target, TerminatorKind, Value};

pub(super) fn lower(graph: &mut Graph, planned: &[super::reads::Read]) -> super::Result<()> {
    if graph.has_handlers || graph.generator {
        return Ok(());
    }
    let stores: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter_map(|inst| {
            let data = &graph.insts[inst];
            if !matches!(data.op, Op::SetIndex(..)) {
                return None;
            }
            planned
                .iter()
                .find(|access| {
                    let original = &graph.insts[access.inst];
                    original.span == data.span && original.source_name == data.source_name
                })
                .map(|access| (inst, access.clone()))
        })
        .collect();
    for (inst, access) in stores {
        if access.typed {
            super::typed_lowering::lower_store(graph, inst, access.element.unwrap());
        } else {
            lower_array(graph, inst);
        }
    }
    graph.verify()
}

fn lower_array(graph: &mut Graph, inst: Inst) {
    let data = graph.insts[inst].clone();
    let Op::SetIndex(object, key, value) = data.op else {
        unreachable!()
    };
    let numeric = super::Facts::compute(graph)
        .of(key)
        .within(super::ValueType::NUMBER);
    let first = graph.blocks.len();
    let (block, done, fallback) = super::dispatch::split(graph, inst);
    let fast = if numeric {
        block
    } else {
        let number = graph.append_value(block, data.span, Op::IsNumber(key));
        branch(graph, inst, block, number, fallback)
    };
    let key_number = graph.append_value(fast, data.span, Op::AsNumber(key));
    let position = graph.append_value(fast, data.span, Op::ArrayPosition(key_number));
    let valid = graph.append_value(
        fast,
        data.span,
        Op::IsArrayIndex(object, key_number, position, super::ArrayLayout::Packed),
    );
    let fast = branch(graph, inst, fast, valid, fallback);
    let writable = graph.append_value(fast, data.span, Op::CanOverwrite(object, position));
    let fast = branch(graph, inst, fast, writable, fallback);
    graph.append(
        fast,
        data.span,
        Op::ArrayStore(object, position, value),
        false,
    );
    graph.terminate(fast, data.span, TerminatorKind::Jump(Target::new(done)));
    extend_regions(graph, block, first);
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
