use oxc_syntax::operator::BinaryOperator;

use super::super::{Op, UnaryOp, ValueType};
use super::NUMERIC;

pub(in crate::jit::semantic) fn coercions(function: &mut super::super::FunctionState) {
    function.ensure_values();
    let mut changed = false;
    let graph = &mut function.graph;
    let facts = &function.facts;
    for &block in &facts.cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let op = &graph.insts[inst].op;
            let bitwise = matches!(
                op,
                Op::Unary(UnaryOp::BitwiseNot, _)
                    | Op::Binary(
                        BinaryOperator::BitwiseAnd
                            | BinaryOperator::BitwiseOR
                            | BinaryOperator::BitwiseXOR
                            | BinaryOperator::ShiftLeft
                            | BinaryOperator::ShiftRight
                            | BinaryOperator::ShiftRightZeroFill,
                        ..
                    )
            );
            if !bitwise {
                continue;
            }
            if !op
                .operands_iter()
                .all(|value| facts.of(value).within(NUMERIC))
                || op
                    .operands_iter()
                    .all(|value| facts.of(value).within(ValueType::INT32))
            {
                continue;
            }
            let mut op = op.clone();
            let mut position = graph.blocks[block]
                .insts
                .iter()
                .position(|&candidate| candidate == inst)
                .unwrap();
            let span = graph.insts[inst].span;
            for operand in op.operands_mut() {
                if facts.of(*operand).within(ValueType::INT32) {
                    continue;
                }
                *operand = graph.insert_value(block, position, span, Op::ToInt32(*operand));
                position += 1;
            }
            graph.set_op(inst, op);
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
}
