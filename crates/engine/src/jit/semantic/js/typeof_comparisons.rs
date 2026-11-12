use oxc_syntax::operator::BinaryOperator;

use super::super::{Cfg, Constant, Graph, Op, UnaryOp, ValueDef};

pub(in crate::jit::semantic) fn typeof_comparisons(graph: &mut Graph) {
    let uses = graph.use_counts();
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Op::Binary(operator, left, right) = graph.insts[inst].op else {
                continue;
            };
            if !matches!(
                operator,
                BinaryOperator::StrictEquality | BinaryOperator::StrictInequality
            ) {
                continue;
            }
            let pair = [(left, right), (right, left)]
                .into_iter()
                .find_map(|(value, literal)| {
                    let Op::Unary(UnaryOp::Typeof, operand) = graph.defining_op(value)? else {
                        return None;
                    };
                    let Op::Constant(Constant::String(text)) = graph.defining_op(literal)? else {
                        return None;
                    };
                    let kind = [
                        "string",
                        "number",
                        "boolean",
                        "undefined",
                        "object",
                        "function",
                    ]
                    .iter()
                    .position(|name| text == name)?;
                    (uses[value] == 1).then_some((value, literal, *operand, kind as u8))
                });
            let Some((value, literal, operand, kind)) = pair else {
                continue;
            };
            let ValueDef::Result(typeof_inst, _) = graph.values[value] else {
                unreachable!();
            };
            // Keep TDZ errors at the original typeof evaluation point.
            graph.insts[typeof_inst].op = Op::CheckInitialized(operand);
            graph.insts[typeof_inst].results.clear();
            if uses[literal] == 1 {
                let ValueDef::Result(literal_inst, _) = graph.values[literal] else {
                    unreachable!();
                };
                graph.remove_inst(literal_inst);
            }
            let test = if kind == 0 {
                Op::IsString(operand)
            } else if kind == 1 {
                Op::IsNumber(operand)
            } else {
                Op::IsTypeof(operand, kind)
            };
            graph.insts[inst].op = if operator == BinaryOperator::StrictEquality {
                test
            } else {
                let position = graph.blocks[block]
                    .insts
                    .iter()
                    .position(|&i| i == inst)
                    .unwrap();
                let span = graph.insts[inst].span;
                let (_, result) = graph.insert(block, position, span, test, true);
                Op::Unary(UnaryOp::Not, result.unwrap())
            };
        }
    }
}
