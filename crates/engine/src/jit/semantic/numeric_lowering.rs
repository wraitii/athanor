use oxc_syntax::operator::BinaryOperator;

use super::{Block, FunctionState, Graph, InstData, Op, Types, Value};

pub(super) fn lower(function: &mut FunctionState) -> bool {
    if function.graph.generator || function.graph.has_handlers {
        return false;
    }
    function.ensure_values();
    let facts = &function.facts;
    let graph = &mut function.graph;
    let mut changed = false;
    for block in graph.blocks.keys().collect::<Vec<_>>() {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let value = match data.op {
                Op::IsInteger(value) => value,
                Op::AsInteger(value) if !facts.of(value).within(Types::INTEGER) => value,
                Op::Binary(
                    operator @ (BinaryOperator::Addition
                    | BinaryOperator::Subtraction
                    | BinaryOperator::Multiplication
                    | BinaryOperator::Division),
                    left,
                    right,
                ) if facts.of(left).intersects(Types::NUMBER)
                    && facts.of(left).within(Types::NUMBER)
                    && facts.of(right).intersects(Types::NUMBER)
                    && facts.of(right).within(Types::NUMBER)
                    && !(operator != BinaryOperator::Division
                        && data
                            .result()
                            .is_some_and(|result| facts.of(result).within(Types::INTEGER))) =>
                {
                    let at = graph.blocks[block]
                        .insts
                        .iter()
                        .position(|&i| i == inst)
                        .unwrap();
                    let left = insert(graph, block, at, &data, Op::NumericFloat(left));
                    let right = insert(graph, block, at + 1, &data, Op::NumericFloat(right));
                    graph.insts[inst].op = Op::Binary(operator, left, right);
                    changed = true;
                    continue;
                }
                _ => continue,
            };
            changed = true;
            let at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let float = insert(graph, block, at, &data, Op::NumericFloat(value));
            graph.insts[inst].op = match data.op {
                Op::IsInteger(_) => {
                    let integer = insert(graph, block, at + 1, &data, Op::SaturatingInteger(float));
                    let exact = insert(graph, block, at + 2, &data, Op::NumericFloat(integer));
                    Op::IntegerCheck(value, float, exact)
                }
                Op::AsInteger(_) => {
                    let integer = insert(graph, block, at + 1, &data, Op::SaturatingInteger(float));
                    Op::AsInteger(integer)
                }
                _ => unreachable!(),
            };
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

fn insert(graph: &mut Graph, block: Block, at: usize, data: &InstData, op: Op) -> Value {
    let (inst, value) = graph.insert(block, at, data.span, op, true);
    graph.insts[inst].source_name = data.source_name;
    value.unwrap()
}
