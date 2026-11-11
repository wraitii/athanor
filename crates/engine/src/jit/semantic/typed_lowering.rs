use super::{
    BlockData, Constant, FunctionState, Graph, Op, Target, Terminator, TerminatorKind, TypedRead,
    Types,
};

pub(super) fn lower(function: &mut FunctionState) -> super::Result<bool> {
    let arrays_changed = lower_arrays(function);
    function.refresh_facts();
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
    Ok(changed || arrays_changed)
}

pub(super) fn lower_arrays(function: &mut FunctionState) -> bool {
    function.refresh_facts();
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
