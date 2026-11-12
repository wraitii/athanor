use super::{
    BlockData, Constant, FunctionState, Graph, Op, Target, Terminator, TerminatorKind, TypedRead,
    Types,
};

pub(super) fn lower(function: &mut FunctionState) -> super::Result<bool> {
    let bits_changed = lower_bitwise(function);
    let arrays_changed = lower_arrays(function);
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
                if facts.of(key).intersects(Types::NUMBER) && facts.of(key).within(Types::NUMBER))
        })
        .collect();
    let changed = !reads.is_empty();
    for inst in reads {
        let data = graph.insts[inst].clone();
        let Op::TypedIndex(object, key, mut read) = data.op else {
            unreachable!()
        };
        let (block, continuation, load) = super::dispatch::split(graph, inst);
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
    if changed {
        function.invalidate_cfg();
    }
    function.graph.verify()?;
    Ok(changed || arrays_changed || bits_changed)
}

fn lower_bitwise(function: &mut FunctionState) -> bool {
    use oxc_syntax::operator::BinaryOperator;

    function.ensure_values();
    let facts = &function.facts;
    let graph = &mut function.graph;
    let candidates: Vec<_> = facts
        .cfg
        .rpo
        .iter()
        .flat_map(|&block| graph.blocks[block].insts.iter().copied())
        .filter(|&inst| {
            matches!(graph.insts[inst].op, Op::Binary(
            BinaryOperator::BitwiseAnd | BinaryOperator::BitwiseOR
            | BinaryOperator::BitwiseXOR | BinaryOperator::ShiftLeft
            | BinaryOperator::ShiftRight | BinaryOperator::ShiftRightZeroFill,
            left, right,
        ) if !facts.of(left).within(Types::NUMBER) || !facts.of(right).within(Types::NUMBER))
        })
        .collect();
    for &inst in &candidates {
        let data = graph.insts[inst].clone();
        let Op::Binary(operator, left, right) = data.op else {
            unreachable!()
        };
        let (block, continuation, fallback) = super::dispatch::split(graph, inst);
        let fast = graph.blocks.push(BlockData::default());
        let left_number = append(graph, &data, block, Op::IsNumber(left));
        let right_number = append(graph, &data, block, Op::IsNumber(right));
        let both = append(
            graph,
            &data,
            block,
            Op::BooleanAnd(left_number, right_number),
        );
        graph.terminate(
            block,
            data.span,
            TerminatorKind::Branch(both, Target::new(fast), Target::new(fallback)),
        );
        let left = append(graph, &data, fast, Op::AsNumber(left));
        let right = append(graph, &data, fast, Op::AsNumber(right));
        let left = append(graph, &data, fast, Op::ToInt32(left));
        let right = append(graph, &data, fast, Op::ToInt32(right));
        let result = append(graph, &data, fast, Op::Binary(operator, left, right));
        graph.terminate(
            fast,
            data.span,
            TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![result],
            }),
        );
        for region in graph.regions.values_mut() {
            if region.blocks.contains(&block) {
                region.blocks.extend([continuation, fallback, fast]);
            }
        }
    }
    if !candidates.is_empty() {
        function.invalidate_cfg();
    }
    !candidates.is_empty()
}

pub(super) fn lower_arrays(function: &mut FunctionState) -> bool {
    function.ensure_values();
    let graph = &mut function.graph;
    let facts = &function.facts;
    let mut changed = false;
    for &block in &facts.cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let Op::IsArrayIndex(object, key, position, holey) = data.op else {
                continue;
            };
            let mut at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let mut insert = |op| {
                let (lowered, value) = graph.insert(block, at, data.span, op, true);
                graph.insts[lowered].source_name = data.source_name;
                at += 1;
                value.unwrap()
            };
            let layout = insert(Op::IsArray(object, holey));
            let exact = insert(if facts.of(key).within(Types::INTEGER) {
                Op::Constant(Constant::Boolean(true))
            } else {
                Op::IsExactPosition(key, position)
            });
            let length = insert(Op::IndexedLength(object));
            let inside = insert(Op::IsInBounds(position, length));
            let valid = insert(Op::BooleanAnd(exact, inside));
            graph.insts[inst].op = Op::BooleanAnd(layout, valid);
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
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

pub(super) fn lower_array_kinds(function: &mut FunctionState) -> bool {
    let graph = &mut function.graph;
    let mut changed = false;
    for block in graph.blocks.keys().collect::<Vec<_>>() {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let Op::IsArray(object, holey) = data.op else {
                continue;
            };
            let at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let (load, kind) = graph.insert(block, at, data.span, Op::ArrayKind(object), true);
            graph.insts[load].source_name = data.source_name;
            let expected =
                crate::heap::kind::ARRAY | if holey { crate::heap::kind::HOLEY } else { 0 };
            let (constant, expected) = graph.insert(
                block,
                at + 1,
                data.span,
                Op::Constant(Constant::Number(expected as f64)),
                true,
            );
            graph.insts[constant].source_name = data.source_name;
            graph.insts[inst].op = Op::Binary(
                oxc_syntax::operator::BinaryOperator::StrictEquality,
                kind.unwrap(),
                expected.unwrap(),
            );
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::super::Cfg;
    use super::*;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;
    use std::collections::HashSet;

    #[test]
    fn array_checks_share_validity_until_a_mutation() {
        for mutate in [false, true] {
            let allocator = Allocator::default();
            let parsed = Parser::new(
                &allocator,
                "function check(a, k) { return a[k * 1]; }",
                SourceType::script(),
            )
            .parse();
            let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
                panic!()
            };
            let semantic = oxc_semantic::SemanticBuilder::new_compiler()
                .build(&parsed.program)
                .semantic;
            let graph =
                super::super::build::lower(function, semantic.scoping(), &HashSet::new()).unwrap();
            let mut graph = super::super::canonicalize_graph(graph, true).unwrap().graph;
            let inst = graph
                .insts
                .iter()
                .find_map(|(inst, data)| matches!(data.op, Op::GetIndex(..)).then_some(inst))
                .unwrap();
            let Op::GetIndex(object, key) = graph.insts[inst].op else {
                unreachable!()
            };
            let data = graph.insts[inst].clone();
            let block = graph.def_block(data.result.unwrap());
            let at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let position = graph.insert_value(block, at, data.span, Op::ArrayPosition(key));
            graph.insts[inst].op = Op::IsArrayIndex(object, key, position, false);
            if mutate {
                graph.append(block, data.span, Op::ArrayAppend(object, key, false), false);
            }
            let second = append(
                &mut graph,
                &data,
                block,
                Op::IsArrayIndex(object, key, position, true),
            );
            let result = append(
                &mut graph,
                &data,
                block,
                Op::NewArray(vec![data.result.unwrap(), second]),
            );
            graph.blocks[block].terminator.as_mut().unwrap().kind = TerminatorKind::Return(result);
            let mut function = FunctionState::new(graph, Vec::new(), true);
            lower(&mut function).unwrap();
            super::super::optimize::reuse_loads(&mut function);
            super::super::js::simplify(&mut function);
            let graph = function.graph;
            graph.verify().unwrap();
            let count = |predicate: fn(&Op) -> bool| {
                Cfg::new(&graph)
                    .rpo
                    .iter()
                    .flat_map(|&block| &graph.blocks[block].insts)
                    .filter(|&&inst| predicate(&graph.insts[inst].op))
                    .count()
            };
            assert_eq!(count(|op| matches!(op, Op::IsExactPosition(..))), 1);
            assert_eq!(
                count(|op| matches!(op, Op::IndexedLength(..))),
                if mutate { 2 } else { 1 }
            );
            assert_eq!(
                count(|op| matches!(op, Op::IsInBounds(..))),
                if mutate { 2 } else { 1 }
            );
            assert_eq!(count(|op| matches!(op, Op::IsArrayIndex(..))), 0);
        }
    }
}

pub(super) fn lower_memory(function: &mut FunctionState) -> bool {
    use super::{MemoryFormat, MemoryOp};
    use crate::heap::TypedKind;

    let graph = &mut function.graph;
    let mut changed = false;
    for block in super::Cfg::new(graph).rpo {
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
            let mut at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let mut insert = |op| {
                let (inserted, value) = graph.insert(block, at, data.span, op, true);
                graph.insts[inserted].source_name = data.source_name;
                at += 1;
                value.unwrap()
            };
            let view = existing_view.unwrap_or_else(|| insert(Op::View(object)));
            let base = insert(Op::TypedData(view));
            let index = insert(Op::ArrayPosition(key));
            let raw = insert(Op::Memory(MemoryOp::IndexedLoad {
                base,
                index,
                format,
            }));
            graph.insts[inst].op = Op::MemoryNumber(raw, format);
            let (keep, _) = graph.insert(block, at + 1, data.span, Op::KeepAlive(object), false);
            graph.insts[keep].source_name = data.source_name;
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}
