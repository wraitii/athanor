use crate::jit::semantic::JsOp;
use oxc_syntax::operator::BinaryOperator;

use super::super::{Constant, Op, TerminatorKind, ValueDef, ValueType};

pub(in crate::jit::semantic) fn bit_tests(function: &mut super::super::FunctionState) {
    function.ensure_values();
    let mut changed = false;
    let graph = &mut function.graph;
    let facts = &function.facts;
    let uses = graph.use_counts();
    for &block in &facts.cfg.rpo {
        let terminator = graph.blocks[block].terminator.as_ref().unwrap();
        let TerminatorKind::Branch(condition, _, _) = terminator.kind else {
            continue;
        };
        // The sign-bit mask may be negative; this rewrite preserves only truthiness.
        if uses[condition] != 1 {
            continue;
        }
        let Some(&Op::Javascript(JsOp::Binary(BinaryOperator::BitwiseAnd, left, right))) =
            graph.defining_op(condition)
        else {
            continue;
        };
        let test = [(left, right), (right, left)]
            .into_iter()
            .find_map(|(word, mask)| {
                let &Op::Javascript(JsOp::Binary(BinaryOperator::ShiftLeft, one, index)) =
                    graph.defining_op(mask)?
                else {
                    return None;
                };
                (matches!(
                    graph.defining_op(one),
                    Some(Op::Javascript(JsOp::Constant(Constant::Number(1.0))))
                ) && facts.of(word).within(ValueType::INT32)
                    && facts.of(index).within(ValueType::INT32))
                .then_some((word, index, one))
            });
        let Some((word, index, one)) = test else {
            continue;
        };
        let ValueDef::Result(inst, _) = graph.values[condition] else {
            unreachable!();
        };
        let data = &graph.insts[inst];
        let span = data.span;
        let source_name = data.source_name;
        let block = data.block;
        let position = graph.blocks[block]
            .insts
            .iter()
            .position(|&candidate| candidate == inst)
            .unwrap();
        let (shift, shifted) = graph.insert(
            block,
            position,
            span,
            Op::Javascript(JsOp::Binary(BinaryOperator::ShiftRight, word, index)),
            true,
        );
        graph.insts[shift].source_name = source_name;
        graph.insts[inst].op = Op::Javascript(JsOp::Binary(
            BinaryOperator::BitwiseAnd,
            shifted.unwrap(),
            one,
        ));
        changed = true;
    }
    if changed {
        function.invalidate_values();
    }
}
