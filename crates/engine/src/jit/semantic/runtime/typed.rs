use super::super::cursor::{Cursor, Insert};
use super::super::{
    BlockData, Constant, FunctionState, Graph, Op, Repr, Target, Terminator, TerminatorKind,
    TypedRead, ValueType,
};

pub(in crate::jit::semantic) fn lower(function: &mut FunctionState) -> super::super::Result<bool> {
    function.ensure_values();
    let graph = &mut function.graph;
    let facts = &function.facts;
    let reads: Vec<_> = facts
        .cfg
        .rpo
        .iter()
        .copied()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter(|&inst| {
            matches!(graph.insts[inst].op,
                Op::TypedIndex(_, key, TypedRead { element: Some(_), bounded: false })
                if facts.of(key).intersects(ValueType::NUMBER) && facts.of(key).within(ValueType::NUMBER))
        })
        .collect();
    let changed = !reads.is_empty();
    for inst in reads {
        let data = graph.insts[inst].clone();
        let Op::TypedIndex(object, key, mut read) = data.op else {
            unreachable!()
        };
        let (block, continuation, load) = super::super::dispatch::split(graph, inst);
        graph.cold.remove(&load);
        let bounds = graph.blocks.push(BlockData::default());
        let failure = graph.blocks.push(BlockData::default());
        for region in graph.regions.values_mut() {
            if region.blocks.contains(&block) {
                region.blocks.extend([continuation, load, bounds, failure]);
            }
        }
        let mut at = Cursor::at_end(graph, block, inst);
        let position = at.add_unassigned(Op::ArrayPosition(key));
        let exact = if facts.of(key).within(ValueType::INTEGER) {
            at.add_unassigned(Op::Constant(Constant::Boolean(true)))
        } else {
            at.add_unassigned(Op::IsExactPosition(key, position))
        };
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Branch(exact, Target::new(bounds), Target::new(failure)),
        });
        let mut at = Cursor::at_end(graph, bounds, inst);
        let length = at.add_unassigned(Op::IndexedLength(object));
        let inside = at.add_unassigned(Op::IsInBounds(position, length));
        graph.blocks[bounds].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Branch(inside, Target::new(load), Target::new(failure)),
        });
        read.bounded = true;
        graph.insts[inst].op = Op::TypedIndex(object, position, read);
        let undefined =
            Cursor::at_end(graph, failure, inst).add_unassigned(Op::Constant(Constant::Undefined));
        graph.blocks[failure].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![undefined],
            }),
        });
    }
    if changed {
        function.invalidate_cfg();
    }
    function.graph.verify()?;
    Ok(changed)
}

pub(in crate::jit::semantic) fn lower_memory(function: &mut FunctionState) -> bool {
    use super::super::{MemoryFormat, MemoryOp};
    use crate::heap::TypedKind;

    let graph = &mut function.graph;
    let mut changed = false;
    for block in super::super::Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            if data.block != block {
                continue;
            }
            let Op::TypedIndex(
                object,
                key,
                TypedRead {
                    element: Some(element),
                    bounded: true,
                },
            ) = data.op
            else {
                continue;
            };
            let format = match element {
                TypedKind::Uint8 => MemoryFormat::U8,
                TypedKind::Uint16 => MemoryFormat::U16,
                TypedKind::Uint32 => MemoryFormat::U32,
                TypedKind::Int8 => MemoryFormat::I8,
                TypedKind::Int16 => MemoryFormat::I16,
                TypedKind::Int32 => MemoryFormat::I32,
                TypedKind::Float32 => MemoryFormat::F32,
                TypedKind::Float64 => MemoryFormat::F64,
            };
            let existing_view = graph.views.remove(&inst);
            let mut at = Cursor::before(graph, inst);
            let view = existing_view.unwrap_or_else(|| at.add_unassigned(Op::View(object)));
            let base = at.add_unassigned(Op::TypedData(view));
            let index = at.add_unassigned(Op::ArrayPosition(key));
            let raw = at.add_unassigned(Op::Memory(MemoryOp::IndexedLoad {
                base,
                index,
                format,
            }));
            graph.insts[inst].op = Op::MemoryNumber(raw, format);
            Cursor::after(graph, inst).add_effect(Op::KeepAlive(object));
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

pub(in crate::jit::semantic) fn lower_store(
    graph: &mut Graph,
    inst: super::super::Inst,
    element: crate::heap::TypedKind,
) {
    use super::super::low_level;
    use cranelift_codegen::ir::{MemFlagsData, types};
    use oxc_syntax::operator::BinaryOperator as B;
    let data = graph.insts[inst].clone();
    let Op::SetIndex(object, key, value) = data.op else {
        unreachable!()
    };
    let facts = super::super::Facts::compute(graph);
    let typed = super::super::guards::holds(
        graph,
        &facts.cfg,
        data.block,
        object,
        super::super::guards::Fact::Typed(Some(element)),
    );
    let first = graph.blocks.len();
    let (block, done, fallback) = super::super::dispatch::split(graph, inst);
    let mut fast = if typed {
        block
    } else {
        let check =
            Cursor::at_end(graph, block, inst).add_unassigned(Op::IsTyped(object, Some(element)));
        super::super::dispatch::branch(graph, inst, block, check, fallback)
    };
    for value in [key, value] {
        if !facts.of(value).within(ValueType::NUMBER) {
            let number = Cursor::at_end(graph, fast, inst).add_unassigned(Op::IsNumber(value));
            fast = super::super::dispatch::branch(graph, inst, fast, number, fallback);
        }
    }
    let key = Cursor::at_end(graph, fast, inst)
        .place(Op::AsNumber(key), true)
        .1
        .unwrap();
    let value = Cursor::at_end(graph, fast, inst)
        .place(Op::AsNumber(value), true)
        .1
        .unwrap();
    let mut at = Cursor::at_end(graph, fast, inst);
    let position = at.add_unassigned(Op::ArrayPosition(key));
    let exact = at.add_unassigned(Op::IsExactPosition(key, position));
    fast = super::super::dispatch::branch(graph, inst, fast, exact, fallback);
    let mut at = Cursor::at_end(graph, fast, inst);
    let length = at.add_unassigned(Op::IndexedLength(object));
    let inside = at.add_unassigned(Op::IsInBounds(position, length));
    fast = super::super::dispatch::branch(graph, inst, fast, inside, fallback);
    let float = matches!(
        element,
        crate::heap::TypedKind::Float32 | crate::heap::TypedKind::Float64
    );
    let stored = if float {
        let number = Cursor::at_end(graph, fast, inst)
            .place(Op::Convert(value, Repr::F64), true)
            .1
            .unwrap();
        if element == crate::heap::TypedKind::Float32 {
            Cursor::at_end(graph, fast, inst).add_unassigned(Op::LowLevel(low_level::Op::Unary {
                op: low_level::Unary::FloatDemote,
                ty: types::F32,
                value: number,
            }))
        } else {
            number
        }
    } else {
        let integer = Cursor::at_end(graph, fast, inst).add_unassigned(Op::ArrayPosition(value));
        let exact =
            Cursor::at_end(graph, fast, inst).add_unassigned(Op::IsExactPosition(value, integer));
        fast = super::super::dispatch::branch(graph, inst, fast, exact, fallback);
        let upper = Cursor::at_end(graph, fast, inst)
            .add_unassigned(Op::Constant(Constant::Number(9223372036854775808.0)));
        let below =
            Cursor::at_end(graph, fast, inst).add_unassigned(Op::Binary(B::LessThan, value, upper));
        fast = super::super::dispatch::branch(graph, inst, fast, below, fallback);
        let ty = match 1 << (element.element() & 3) {
            1 => types::I8,
            2 => types::I16,
            _ => types::I32,
        };
        Cursor::at_end(graph, fast, inst).ins().ireduce(ty, integer)
    };
    let mut at = Cursor::at_end(graph, fast, inst);
    let view = at.add_unassigned(Op::View(object));
    let base = at.ins().load(
        types::I64,
        MemFlagsData::trusted(),
        view,
        crate::heap::DATA_OFFSET,
    );
    let bytes = at
        .ins()
        .iconst(types::I64, (1 << (element.element() & 3)) as i64);
    let offset = at
        .ins()
        .binary(low_level::Binary::Multiply, position, bytes);
    let address = at.ins().iadd(base, offset);
    at.ins().store(MemFlagsData::trusted(), stored, address, 0);
    at.add_effect(Op::KeepAlive(object));
    graph.terminate(fast, data.span, TerminatorKind::Jump(Target::new(done)));
    super::super::dispatch::extend_regions(graph, block, first);
}
