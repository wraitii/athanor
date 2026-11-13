#[cfg(test)]
use super::super::TerminatorKind;
#[cfg(test)]
use super::super::dispatch::append;
use super::super::{Constant, FunctionState, Op, ValueType};

pub(in crate::jit::semantic) fn lower_arrays(function: &mut FunctionState) -> bool {
    function.ensure_values();
    let graph = &mut function.graph;
    let facts = &function.facts;
    let mut changed = false;
    for &block in &facts.cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let Op::IsArrayIndex(object, key, position, layout) = data.op else {
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
            let matching = insert(Op::IsArray(object, layout));
            let exact = insert(if facts.of(key).within(ValueType::INTEGER) {
                Op::Constant(Constant::Boolean(true))
            } else {
                Op::IsExactPosition(key, position)
            });
            let length = insert(Op::IndexedLength(object));
            let inside = insert(Op::IsInBounds(position, length));
            let valid = insert(Op::BooleanAnd(exact, inside));
            graph.insts[inst].op = Op::BooleanAnd(matching, valid);
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

pub(in crate::jit::semantic) fn lower_array_kinds(function: &mut FunctionState) -> bool {
    let graph = &mut function.graph;
    let mut changed = false;
    for block in graph.blocks.keys().collect::<Vec<_>>() {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let Op::IsArray(object, layout) = data.op else {
                continue;
            };
            let at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let (load, kind) = graph.insert(block, at, data.span, Op::ArrayKind(object), true);
            graph.insts[load].source_name = data.source_name;
            let expected = layout.kind();
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
    use super::super::super::Cfg;
    use super::*;
    use crate::FxHashSet;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;

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
            let graph = super::super::super::build::lower(
                function,
                semantic.scoping(),
                &FxHashSet::default(),
            )
            .unwrap();
            let mut graph = super::super::super::canonicalize_graph(graph, true)
                .unwrap()
                .graph;
            let inst = graph
                .insts
                .iter()
                .find_map(|(inst, data)| matches!(data.op, Op::GetIndex(..)).then_some(inst))
                .unwrap();
            let Op::GetIndex(object, key) = graph.insts[inst].op else {
                unreachable!()
            };
            let data = graph.insts[inst].clone();
            let block = graph.def_block(data.result().unwrap());
            let at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let position = graph.insert_value(block, at, data.span, Op::ArrayPosition(key));
            graph.insts[inst].op = Op::IsArrayIndex(
                object,
                key,
                position,
                super::super::super::ArrayLayout::Packed,
            );
            if mutate {
                graph.append(block, data.span, Op::ArrayAppend(object, key, false), false);
            }
            let second = append(
                &mut graph,
                &data,
                block,
                Op::IsArrayIndex(
                    object,
                    key,
                    position,
                    super::super::super::ArrayLayout::Holey,
                ),
            );
            let result = append(
                &mut graph,
                &data,
                block,
                Op::NewArray(vec![data.result().unwrap(), second]),
            );
            graph.blocks[block].terminator.as_mut().unwrap().kind = TerminatorKind::Return(result);
            let mut function = FunctionState::new(graph, Vec::new(), true);
            lower_arrays(&mut function);
            super::super::super::transforms::reuse::reuse_loads(&mut function);
            super::super::super::js::simplify(&mut function);
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
