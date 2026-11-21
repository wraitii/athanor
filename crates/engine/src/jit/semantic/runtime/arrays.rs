#[cfg(test)]
use super::super::TerminatorKind;
use super::super::cursor::{Cursor, Insert};
#[cfg(test)]
use super::super::dispatch::append;
use super::super::{Constant, FunctionState, Graph, Inst, Op, Repr, Value, ValueType, low_level};
use crate::jit::semantic::JsOp;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};

pub(in crate::jit::semantic) fn generic_cold(function: &mut FunctionState) {
    let graph = &mut function.graph;
    let mut changed = false;
    for block in graph.cold.clone() {
        for inst in graph.blocks[block].insts.clone() {
            let op = match graph.insts[inst].op {
                Op::Javascript(
                    JsOp::ArrayLoad(object, index, _) | JsOp::TypedIndex(object, index, _),
                ) => Op::Javascript(JsOp::GetIndex(object, index)),
                Op::Javascript(
                    JsOp::ArrayStore(object, index, value)
                    | JsOp::TypedSet(object, index, value, _),
                ) => Op::Javascript(JsOp::SetIndex(object, index, value)),
                _ => continue,
            };
            graph.set_op(inst, op);
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
}

pub(in crate::jit::semantic) fn lower_arrays(function: &mut FunctionState) -> bool {
    function.ensure_values();
    let graph = &mut function.graph;
    let facts = &function.facts;
    let mut changed = false;
    let mut insts = Vec::new();
    for &block in &facts.cfg.rpo {
        insts.clone_from(&graph.blocks[block].insts);
        for &inst in &insts {
            let data = &graph.insts[inst];
            let Op::Javascript(JsOp::IsArrayIndex(object, key, position, layout)) = data.op else {
                continue;
            };
            let mut at = Cursor::before(graph, inst);
            let matching = at.add_unassigned(Op::Javascript(JsOp::IsArray(object, layout)));
            let exact = if facts.of(key).within(ValueType::INTEGER) {
                at.add_unassigned(Op::Javascript(JsOp::Constant(Constant::Boolean(true))))
            } else {
                at.add_unassigned(Op::Javascript(JsOp::IsExactPosition(key, position)))
            };
            let length = at.add_unassigned(Op::Javascript(JsOp::IndexedLength(object)));
            let inside = at.add_unassigned(Op::Javascript(JsOp::IsInBounds(position, length)));
            let valid = at.add_unassigned(Op::BooleanAnd(exact, inside));
            graph.set_op(inst, Op::BooleanAnd(matching, valid));
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

pub(in crate::jit::semantic) fn lower_checks(function: &mut FunctionState) -> bool {
    let graph = &mut function.graph;
    let mut changed = false;
    let mut insts = Vec::new();
    for block in graph.blocks.keys().collect::<Vec<_>>() {
        insts.clone_from(&graph.blocks[block].insts);
        for &inst in &insts {
            let op = match graph.insts[inst].op {
                Op::BooleanAnd(left, right) => low_level::Op::Binary {
                    op: low_level::Binary::BooleanAnd,
                    left,
                    right,
                },
                Op::Javascript(JsOp::IsExactPosition(left, right)) => low_level::Op::FloatCompare {
                    condition: FloatCC::Equal,
                    left,
                    right,
                },
                Op::Javascript(JsOp::IsInBounds(position, length)) => low_level::Op::IntCompare {
                    condition: IntCC::UnsignedLessThan,
                    left: integer(graph, inst, position),
                    right: integer(graph, inst, length),
                },
                _ => continue,
            };
            graph.set_op(inst, Op::LowLevel(op));
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

fn integer(graph: &mut Graph, inst: Inst, value: Value) -> Value {
    Cursor::before(graph, inst).add(Op::Convert(value, Repr::I64), Repr::I64)
}

pub(in crate::jit::semantic) fn lower_array_kinds(function: &mut FunctionState) -> bool {
    let graph = &mut function.graph;
    let mut changed = false;
    let mut insts = Vec::new();
    for block in graph.blocks.keys().collect::<Vec<_>>() {
        insts.clone_from(&graph.blocks[block].insts);
        for &inst in &insts {
            let data = &graph.insts[inst];
            let Op::Javascript(JsOp::IsArray(object, layout)) = data.op else {
                continue;
            };
            let mut at = Cursor::before(graph, inst);
            let kind = at.add_unassigned(Op::Javascript(JsOp::ArrayKind(object)));
            let expected = at.add_unassigned(Op::Javascript(JsOp::Constant(Constant::Number(
                layout.kind() as f64,
            ))));
            graph.set_op(
                inst,
                Op::Javascript(JsOp::Binary(
                    oxc_syntax::operator::BinaryOperator::StrictEquality,
                    kind,
                    expected,
                )),
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
                .find_map(|(inst, data)| {
                    matches!(data.op, Op::Javascript(JsOp::GetIndex(..))).then_some(inst)
                })
                .unwrap();
            let Op::Javascript(JsOp::GetIndex(object, key)) = graph.insts[inst].op else {
                unreachable!()
            };
            let data = graph.insts[inst].clone();
            let block = graph.def_block(data.result().unwrap());
            let at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let position = graph.insert_value(
                block,
                at,
                data.span,
                Op::Javascript(JsOp::ArrayPosition(key)),
            );
            graph.set_op(
                inst,
                Op::Javascript(JsOp::IsArrayIndex(
                    object,
                    key,
                    position,
                    super::super::super::ArrayLayout::Packed,
                )),
            );
            if mutate {
                graph.append(
                    block,
                    data.span,
                    Op::Javascript(JsOp::ArrayAppend(object, key, false)),
                    false,
                );
            }
            let second = append(
                &mut graph,
                &data,
                block,
                Op::Javascript(JsOp::IsArrayIndex(
                    object,
                    key,
                    position,
                    super::super::super::ArrayLayout::Holey,
                )),
            );
            let result = append(
                &mut graph,
                &data,
                block,
                Op::Javascript(JsOp::NewArray(vec![data.result().unwrap(), second])),
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
            assert_eq!(
                count(|op| matches!(op, Op::Javascript(JsOp::IsExactPosition(..)))),
                1
            );
            assert_eq!(
                count(|op| matches!(op, Op::Javascript(JsOp::IndexedLength(..)))),
                if mutate { 2 } else { 1 }
            );
            assert_eq!(
                count(|op| matches!(op, Op::Javascript(JsOp::IsInBounds(..)))),
                if mutate { 2 } else { 1 }
            );
            assert_eq!(
                count(|op| matches!(op, Op::Javascript(JsOp::IsArrayIndex(..)))),
                0
            );
        }
    }
}
