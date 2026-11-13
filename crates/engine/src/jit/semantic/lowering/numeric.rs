use cranelift_codegen::ir::{
    condcodes::{FloatCC, IntCC},
    types,
};
use oxc_syntax::operator::BinaryOperator;

use super::super::low_level::{Binary, Op as LowLevel, Unary};
use super::super::{
    Block, Constant, FunctionState, Graph, InstData, Op, UnaryOp, Value, ValueType,
};
use crate::FxHashMap;

pub(in crate::jit::semantic) fn lower(function: &mut FunctionState) -> bool {
    let checks = lower_checks(function);
    arithmetic(function) || checks
}

fn lower_checks(function: &mut FunctionState) -> bool {
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
                Op::AsInteger(value) if !facts.of(value).within(ValueType::INTEGER) => value,
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

pub(in crate::jit::semantic) fn arithmetic(function: &mut FunctionState) -> bool {
    function.ensure_values();
    let facts = &function.facts;
    let graph = &mut function.graph;
    let numeric = ValueType::NUMBER.union(ValueType::UNDEFINED);
    let mut changed = false;
    for block in facts.cfg.rpo.clone() {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let Op::Binary(operator, mut left, mut right) = data.op else {
                continue;
            };
            if !matches!(
                operator,
                BinaryOperator::Addition
                    | BinaryOperator::Subtraction
                    | BinaryOperator::Multiplication
                    | BinaryOperator::Division
            ) || !facts.of(left).intersects(numeric)
                || !facts.of(left).within(numeric)
                || !facts.of(right).intersects(numeric)
                || !facts.of(right).within(numeric)
            {
                continue;
            }
            let integer = operator != BinaryOperator::Division
                && data
                    .result()
                    .is_some_and(|result| facts.of(result).within(ValueType::INTEGER));
            let op = match (operator, integer) {
                (BinaryOperator::Addition, true) => Binary::Add,
                (BinaryOperator::Subtraction, true) => Binary::Subtract,
                (BinaryOperator::Multiplication, true) => Binary::Multiply,
                (BinaryOperator::Addition, false) => Binary::FloatAdd,
                (BinaryOperator::Subtraction, false) => Binary::FloatSubtract,
                (BinaryOperator::Multiplication, false) => Binary::FloatMultiply,
                (BinaryOperator::Division, _) => Binary::FloatDivide,
                _ => unreachable!(),
            };
            if !integer {
                let at = graph.blocks[block]
                    .insts
                    .iter()
                    .position(|&i| i == inst)
                    .unwrap();
                left = insert(graph, block, at, &data, Op::NumericFloat(left));
                right = insert(graph, block, at + 1, &data, Op::NumericFloat(right));
            }
            graph.insts[inst].op = Op::LowLevel(LowLevel::Binary { op, left, right });
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

pub(in crate::jit::semantic) fn operations(function: &mut FunctionState) -> bool {
    let arithmetic = arithmetic(function);
    comparisons_and_unary(function) || arithmetic
}

fn comparisons_and_unary(function: &mut FunctionState) -> bool {
    function.ensure_values();
    let facts = &function.facts;
    let graph = &mut function.graph;
    let numeric = ValueType::NUMBER.union(ValueType::UNDEFINED);
    let known = |value, types| facts.of(value).intersects(types) && facts.of(value).within(types);
    let mut aliases = FxHashMap::default();
    let mut removed = Vec::new();
    let mut changed = false;
    for block in facts.cfg.rpo.clone() {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let at = graph.blocks[block]
                .insts
                .iter()
                .position(|&i| i == inst)
                .unwrap();
            let mut position = at;
            let mut insert = |op| {
                let value = insert(graph, block, position, &data, op);
                position += 1;
                value
            };
            let replacement = match data.op {
                Op::Binary(operator, left, right)
                | Op::GuardedCompare {
                    operator,
                    left,
                    right,
                    ..
                } => {
                    use BinaryOperator as B;
                    let (integer_condition, float_condition, equality) = match operator {
                        B::LessThan => (IntCC::SignedLessThan, FloatCC::LessThan, false),
                        B::LessEqualThan => (
                            IntCC::SignedLessThanOrEqual,
                            FloatCC::LessThanOrEqual,
                            false,
                        ),
                        B::GreaterThan => (IntCC::SignedGreaterThan, FloatCC::GreaterThan, false),
                        B::GreaterEqualThan => (
                            IntCC::SignedGreaterThanOrEqual,
                            FloatCC::GreaterThanOrEqual,
                            false,
                        ),
                        B::Equality | B::StrictEquality => (IntCC::Equal, FloatCC::Equal, true),
                        B::Inequality | B::StrictInequality => {
                            (IntCC::NotEqual, FloatCC::NotEqual, true)
                        }
                        _ => continue,
                    };
                    let booleans = matches!(operator, B::StrictEquality | B::StrictInequality)
                        && known(left, ValueType::BOOLEAN)
                        && known(right, ValueType::BOOLEAN);
                    let required = if equality { ValueType::NUMBER } else { numeric };
                    if !booleans && !(known(left, required) && known(right, required)) {
                        continue;
                    }
                    if booleans
                        || known(left, ValueType::INTEGER) && known(right, ValueType::INTEGER)
                    {
                        Op::LowLevel(LowLevel::IntCompare {
                            condition: integer_condition,
                            left,
                            right,
                        })
                    } else {
                        let left = insert(Op::NumericFloat(left));
                        let right = insert(Op::NumericFloat(right));
                        Op::LowLevel(LowLevel::FloatCompare {
                            condition: float_condition,
                            left,
                            right,
                        })
                    }
                }
                Op::Unary(UnaryOp::Plus, value) | Op::ToNumeric(value) if known(value, numeric) => {
                    if known(value, ValueType::NUMBER) {
                        aliases.insert(data.result().unwrap(), value);
                        removed.push(inst);
                        changed = true;
                        continue;
                    }
                    Op::NumericFloat(value)
                }
                Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, value)
                    if known(value, numeric) =>
                {
                    let increment = matches!(data.op, Op::Unary(UnaryOp::Increment, _));
                    let integer = data
                        .result()
                        .is_some_and(|value| known(value, ValueType::INTEGER));
                    let value = if integer {
                        value
                    } else {
                        insert(Op::NumericFloat(value))
                    };
                    let one = insert(Op::Constant(Constant::Number(1.0)));
                    let op = match (integer, increment) {
                        (true, true) => Binary::Add,
                        (true, false) => Binary::Subtract,
                        (false, true) => Binary::FloatAdd,
                        (false, false) => Binary::FloatSubtract,
                    };
                    Op::LowLevel(LowLevel::Binary {
                        op,
                        left: value,
                        right: one,
                    })
                }
                Op::Unary(UnaryOp::Negate, value) if known(value, numeric) => {
                    let value = insert(Op::NumericFloat(value));
                    Op::LowLevel(LowLevel::Unary {
                        op: Unary::FloatNeg,
                        ty: types::F64,
                        value,
                    })
                }
                Op::Unary(UnaryOp::BitwiseNot, value) if known(value, numeric) => {
                    let value = insert(Op::ToInt32(value));
                    Op::LowLevel(LowLevel::Unary {
                        op: Unary::BitNot,
                        ty: types::I32,
                        value,
                    })
                }
                Op::Unary(UnaryOp::Not, value)
                    if known(value, numeric) || known(value, ValueType::BOOLEAN) =>
                {
                    if known(value, ValueType::BOOLEAN) {
                        let zero = insert(Op::Constant(Constant::Boolean(false)));
                        Op::LowLevel(LowLevel::IntCompare {
                            condition: IntCC::Equal,
                            left: value,
                            right: zero,
                        })
                    } else if known(value, ValueType::INTEGER) {
                        let zero = insert(Op::Constant(Constant::Number(0.0)));
                        Op::LowLevel(LowLevel::IntCompare {
                            condition: IntCC::Equal,
                            left: value,
                            right: zero,
                        })
                    } else {
                        let value = insert(Op::NumericFloat(value));
                        let zero = insert(Op::Constant(Constant::Number(0.0)));
                        Op::LowLevel(LowLevel::FloatCompare {
                            condition: FloatCC::UnorderedOrEqual,
                            left: value,
                            right: zero,
                        })
                    }
                }
                _ => continue,
            };
            graph.insts[inst].op = replacement;
            changed = true;
        }
    }
    graph.replace_uses(&aliases);
    for inst in removed {
        graph.remove_inst(inst);
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
