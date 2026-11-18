use super::super::dispatch::append;
use super::super::{BlockData, Facts, Graph, Inst, Target, TerminatorKind};
use crate::jit::semantic::JsOp;
use cranelift_codegen::ir::{
    condcodes::{FloatCC, IntCC},
    types,
};
use oxc_syntax::operator::BinaryOperator;

use super::super::cursor::{Cursor, Insert};
use super::super::low_level::{Binary, Op as LowLevel, Unary};
use super::super::ranges::Ranges;
use super::super::{Constant, FunctionState, Op, Repr, UnaryOp, Value, ValueType};
use crate::{FxHashMap, FxHashSet};

pub(in crate::jit::semantic) fn lower(function: &mut FunctionState) -> bool {
    let arithmetic = arithmetic(function);
    lower_checks(function) || arithmetic
}

pub(in crate::jit::semantic) fn relax_checks(function: &mut FunctionState) -> bool {
    function.ensure_cfg();
    let graph = &mut function.graph;
    let uses = graph.use_counts();
    let mut changed = false;
    let mut insts = Vec::new();
    for &block in &function.facts.cfg.rpo {
        insts.clone_from(&graph.blocks[block].insts);
        for &inst in &insts {
            let Op::Javascript(JsOp::IntegerCheck(value, float, exact)) = graph.insts[inst].op
            else {
                continue;
            };
            let Some(Op::Javascript(JsOp::NumericFloat(integer))) = graph.defining_op(exact) else {
                continue;
            };
            // The round trip now serves only the guard, with no integer consumers.
            if uses[exact] == 1
                && uses[*integer] == 1
                && matches!(graph.defining_op(*integer), Some(Op::Javascript(JsOp::SaturatingInteger(input))) if *input == float)
                && matches!(graph.defining_op(float), Some(Op::Javascript(JsOp::NumericFloat(input))) if *input == value)
            {
                graph.set_op(inst, Op::Javascript(JsOp::IsNumber(value)));
                changed = true;
            }
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

fn lower_checks(function: &mut FunctionState) -> bool {
    if function.graph.generator || function.graph.has_handlers {
        return false;
    }
    function.ensure_values();
    let facts = &function.facts;
    let graph = &mut function.graph;
    let mut changed = false;
    let mut insts = Vec::new();
    for block in graph.blocks.keys().collect::<Vec<_>>() {
        insts.clone_from(&graph.blocks[block].insts);
        for &inst in &insts {
            let op = &graph.insts[inst].op;
            let integer_check = matches!(op, Op::Javascript(JsOp::IsInteger(_)));
            let value = match *op {
                Op::Javascript(JsOp::IsInteger(value)) => value,
                Op::Javascript(JsOp::AsInteger(value))
                    if !facts.of(value).within(ValueType::INTEGER) =>
                {
                    value
                }
                _ => continue,
            };
            changed = true;
            let mut at = Cursor::before(graph, inst);
            let float = at.add_unassigned(Op::Javascript(JsOp::NumericFloat(value)));
            let integer = at.add_unassigned(Op::Javascript(JsOp::SaturatingInteger(float)));
            let op = if integer_check {
                let exact = at.add_unassigned(Op::Javascript(JsOp::NumericFloat(integer)));
                Op::Javascript(JsOp::IntegerCheck(value, float, exact))
            } else {
                Op::Javascript(JsOp::AsInteger(integer))
            };
            graph.set_op(inst, op);
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

pub(in crate::jit::semantic) fn arithmetic(function: &mut FunctionState) -> bool {
    function.ensure_values();
    let proved = if function.optimize {
        exact_index_arithmetic(&function.graph, &function.facts)
    } else {
        FxHashSet::default()
    };
    let mut changed = false;
    let mut insts = Vec::new();
    for &block in &function.facts.cfg.rpo {
        insts.clone_from(&function.graph.blocks[block].insts);
        for &inst in &insts {
            changed |= arithmetic_inst(
                &mut function.graph,
                &function.facts,
                inst,
                proved.contains(&inst),
            );
        }
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

fn exact_index_arithmetic(graph: &Graph, facts: &Facts) -> FxHashSet<Inst> {
    let mut ranges = Ranges::new(graph, facts, &[]);
    let mut proved = FxHashSet::default();
    for &block in &facts.cfg.rpo {
        if graph.cold.contains(&block) {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            let Op::Javascript(JsOp::ArrayPosition(index)) = graph.insts[inst].op else {
                continue;
            };
            if !facts.of(index).within(ValueType::INTEGER) {
                ranges.prove_arithmetic(index, &mut proved);
            }
        }
    }
    proved
}

pub(in crate::jit::semantic) fn operations(function: &mut FunctionState) -> bool {
    function.ensure_values();
    let mut changed = false;
    let mut aliases = FxHashMap::default();
    let mut removed = Vec::new();
    let mut insts = Vec::new();
    for &block in &function.facts.cfg.rpo {
        insts.clone_from(&function.graph.blocks[block].insts);
        for &inst in &insts {
            let graph = &mut function.graph;
            let facts = &function.facts;
            changed |= arithmetic_inst(graph, facts, inst, false)
                || integer_operations_inst(graph, facts, inst)
                || natives_inst(graph, facts, inst)
                || comparisons_and_unary_inst(graph, facts, inst, &mut aliases, &mut removed);
        }
    }
    function.graph.replace_uses(&aliases);
    for inst in removed {
        function.graph.remove_inst(inst);
    }
    if changed {
        function.invalidate_values();
    }
    changed
}

fn arithmetic_inst(graph: &mut Graph, facts: &Facts, inst: Inst, proved_integer: bool) -> bool {
    let numeric = ValueType::NUMBER.union(ValueType::UNDEFINED);
    let data = &graph.insts[inst];
    let Op::Javascript(JsOp::Binary(operator, mut left, mut right)) = data.op else {
        return false;
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
        return false;
    }
    let integer = operator != BinaryOperator::Division
        && (proved_integer
            || data
                .result()
                .is_some_and(|result| facts.of(result).within(ValueType::INTEGER)));
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
        let mut at = Cursor::before(graph, inst);
        left = at.add_unassigned(Op::Javascript(JsOp::NumericFloat(left)));
        right = at.add_unassigned(Op::Javascript(JsOp::NumericFloat(right)));
    }
    graph.set_op(inst, Op::LowLevel(LowLevel::Binary { op, left, right }));

    true
}

fn integer_operations_inst(graph: &mut Graph, facts: &Facts, inst: Inst) -> bool {
    let numeric = ValueType::NUMBER.union(ValueType::UNDEFINED);
    let data = &graph.insts[inst];
    let Op::Javascript(JsOp::Binary(operator, left, right)) = data.op else {
        return false;
    };
    let result = data.result().unwrap();
    let op = match operator {
        BinaryOperator::BitwiseAnd => Binary::And,
        BinaryOperator::BitwiseOR => Binary::Or,
        BinaryOperator::BitwiseXOR => Binary::Xor,
        BinaryOperator::ShiftLeft => Binary::ShiftLeft,
        BinaryOperator::ShiftRight => Binary::ShiftRightSigned,
        BinaryOperator::ShiftRightZeroFill => Binary::ShiftRightUnsigned,
        _ => return false,
    };
    if !facts.of(left).within(numeric) || !facts.of(right).within(numeric) {
        return false;
    }
    let mut at = Cursor::before(graph, inst);
    let mut integer = |value: Value| {
        let types = facts.of(value);
        if types != ValueType::default() && types.within(ValueType::INT32) {
            return value;
        }
        at.add_unassigned(Op::Javascript(JsOp::ToInt32(value)))
    };
    let left = integer(left);
    let right = integer(right);
    let binary = LowLevel::Binary { op, left, right };
    if operator == BinaryOperator::ShiftRightZeroFill {
        let value = at.add(Op::LowLevel(binary), Repr::I32);
        let unsigned = LowLevel::Unary {
            op: Unary::FloatFromUnsigned,
            ty: types::F64,
            value,
        };
        graph.set_op(inst, Op::LowLevel(unsigned));
    } else {
        graph.set_op(inst, Op::LowLevel(binary));
        graph.reprs[result] = Some(Repr::I32);
    }

    true
}

fn natives_inst(graph: &mut Graph, facts: &Facts, inst: Inst) -> bool {
    use crate::heap::MathFunction;

    let numeric = ValueType::NUMBER.union(ValueType::UNDEFINED);
    let Op::Javascript(JsOp::CallNative(native, args)) = &graph.insts[inst].op else {
        return false;
    };
    let function = crate::jit::math::METHODS[native.index()].1;
    let unary = match (function, args.len()) {
        (MathFunction::Floor, 1) => Some(Unary::Floor),
        (MathFunction::Ceil, 1) => Some(Unary::Ceil),
        (MathFunction::Sqrt, 1) => Some(Unary::Sqrt),
        (MathFunction::Min | MathFunction::Max, 1..) => None,
        _ => return false,
    };
    if !args.iter().all(|&arg| facts.of(arg).within(numeric)) {
        return false;
    }

    if unary.is_none() && args.len() == 1 {
        graph.set_op(inst, Op::Javascript(JsOp::NumericFloat(args[0])));
        return true;
    }
    let args = args.clone();
    let mut at = Cursor::before(graph, inst);
    let floats: Vec<_> = args
        .iter()
        .map(|&arg| at.add_unassigned(Op::Javascript(JsOp::NumericFloat(arg))))
        .collect();
    let replacement = match unary {
        Some(op) => Op::LowLevel(LowLevel::Unary {
            op,
            ty: types::F64,
            value: floats[0],
        }),
        None => {
            let op = if function == MathFunction::Min {
                Binary::FloatMin
            } else {
                Binary::FloatMax
            };
            let (&last, rest) = floats.split_last().unwrap();
            let mut left = floats[0];
            for &right in rest.iter().skip(1) {
                let binary = LowLevel::Binary { op, left, right };
                left = at.add_unassigned(Op::LowLevel(binary));
            }
            Op::LowLevel(LowLevel::Binary {
                op,
                left,
                right: last,
            })
        }
    };
    graph.set_op(inst, replacement);
    true
}

fn comparisons_and_unary_inst(
    graph: &mut Graph,
    facts: &Facts,
    inst: Inst,
    aliases: &mut FxHashMap<Value, Value>,
    removed: &mut Vec<Inst>,
) -> bool {
    let numeric = ValueType::NUMBER.union(ValueType::UNDEFINED);
    let known = |value, types| facts.of(value).intersects(types) && facts.of(value).within(types);
    if !matches!(
        graph.insts[inst].op,
        Op::Javascript(
            JsOp::Binary(..) | JsOp::GuardedCompare { .. } | JsOp::Unary(..) | JsOp::ToNumeric(..)
        )
    ) {
        return false;
    }
    let data = graph.insts[inst].clone();
    let mut at = Cursor::before(graph, inst);
    let replacement = match data.op {
        Op::Javascript(
            JsOp::Binary(operator, left, right)
            | JsOp::GuardedCompare {
                operator,
                left,
                right,
                ..
            },
        ) => {
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
                B::Inequality | B::StrictInequality => (IntCC::NotEqual, FloatCC::NotEqual, true),
                _ => return false,
            };
            let booleans = matches!(operator, B::StrictEquality | B::StrictInequality)
                && known(left, ValueType::BOOLEAN)
                && known(right, ValueType::BOOLEAN);
            let required = if equality { ValueType::NUMBER } else { numeric };
            if !booleans && !(known(left, required) && known(right, required)) {
                return false;
            }
            if booleans {
                let op = if integer_condition == IntCC::Equal {
                    Binary::BooleanEqual
                } else {
                    Binary::BooleanXor
                };
                Op::LowLevel(LowLevel::Binary { op, left, right })
            } else if known(left, ValueType::INTEGER) && known(right, ValueType::INTEGER) {
                Op::LowLevel(LowLevel::IntCompare {
                    condition: integer_condition,
                    left,
                    right,
                })
            } else {
                let left = at.add_unassigned(Op::Javascript(JsOp::NumericFloat(left)));
                let right = at.add_unassigned(Op::Javascript(JsOp::NumericFloat(right)));
                Op::LowLevel(LowLevel::FloatCompare {
                    condition: float_condition,
                    left,
                    right,
                })
            }
        }
        Op::Javascript(JsOp::Unary(UnaryOp::Plus, value) | JsOp::ToNumeric(value))
            if known(value, numeric) =>
        {
            if known(value, ValueType::NUMBER) {
                aliases.insert(data.result().unwrap(), value);
                removed.push(inst);
                return true;
            }
            Op::Javascript(JsOp::NumericFloat(value))
        }
        Op::Javascript(JsOp::Unary(UnaryOp::Increment | UnaryOp::Decrement, value))
            if known(value, numeric) =>
        {
            let increment = matches!(data.op, Op::Javascript(JsOp::Unary(UnaryOp::Increment, _)));
            let integer = data
                .result()
                .is_some_and(|value| known(value, ValueType::INTEGER));
            let value = if integer {
                value
            } else {
                at.add_unassigned(Op::Javascript(JsOp::NumericFloat(value)))
            };
            let one = at.add_unassigned(Op::Javascript(JsOp::Constant(Constant::Number(1.0))));
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
        Op::Javascript(JsOp::Unary(UnaryOp::Negate, value)) if known(value, numeric) => {
            let value = at.add_unassigned(Op::Javascript(JsOp::NumericFloat(value)));
            Op::LowLevel(LowLevel::Unary {
                op: Unary::FloatNeg,
                ty: types::F64,
                value,
            })
        }
        Op::Javascript(JsOp::Unary(UnaryOp::BitwiseNot, value)) if known(value, numeric) => {
            let value = at.add_unassigned(Op::Javascript(JsOp::ToInt32(value)));
            Op::LowLevel(LowLevel::Unary {
                op: Unary::BitNot,
                ty: types::I32,
                value,
            })
        }
        Op::Javascript(JsOp::Unary(UnaryOp::Not, value))
            if known(value, numeric) || known(value, ValueType::BOOLEAN) =>
        {
            if known(value, ValueType::BOOLEAN) {
                let zero =
                    at.add_unassigned(Op::Javascript(JsOp::Constant(Constant::Boolean(false))));
                Op::LowLevel(LowLevel::Binary {
                    op: Binary::BooleanEqual,
                    left: value,
                    right: zero,
                })
            } else if known(value, ValueType::INTEGER) {
                let zero = at.add_unassigned(Op::Javascript(JsOp::Constant(Constant::Number(0.0))));
                Op::LowLevel(LowLevel::IntCompare {
                    condition: IntCC::Equal,
                    left: value,
                    right: zero,
                })
            } else {
                let value = at.add_unassigned(Op::Javascript(JsOp::NumericFloat(value)));
                let zero = at.add_unassigned(Op::Javascript(JsOp::Constant(Constant::Number(0.0))));
                Op::LowLevel(LowLevel::FloatCompare {
                    condition: FloatCC::UnorderedOrEqual,
                    left: value,
                    right: zero,
                })
            }
        }
        _ => return false,
    };
    graph.set_op(inst, replacement);

    true
}

pub(in crate::jit::semantic) fn bitwise(function: &mut FunctionState) -> bool {
    use oxc_syntax::operator::BinaryOperator;

    function.ensure_values();
    let facts = &function.facts;
    let graph = &mut function.graph;
    let candidates: Vec<_> = facts
        .cfg
        .rpo
        .iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|&block| graph.blocks[block].insts.iter().copied())
        .filter(|&inst| {
            matches!(graph.insts[inst].op, Op::Javascript(JsOp::Binary(
            BinaryOperator::BitwiseAnd | BinaryOperator::BitwiseOR
            | BinaryOperator::BitwiseXOR | BinaryOperator::ShiftLeft
            | BinaryOperator::ShiftRight | BinaryOperator::ShiftRightZeroFill,
            left, right,
        )) if !facts.of(left).within(ValueType::NUMBER) || !facts.of(right).within(ValueType::NUMBER))
        })
        .collect();
    for &inst in &candidates {
        let data = graph.insts[inst].clone();
        let Op::Javascript(JsOp::Binary(operator, left, right)) = data.op else {
            unreachable!()
        };
        let (block, continuation, fallback) = super::super::dispatch::split(graph, inst);
        let fast = graph.blocks.push(BlockData::default());
        let left_number = append(graph, &data, block, Op::Javascript(JsOp::IsNumber(left)));
        let right_number = append(graph, &data, block, Op::Javascript(JsOp::IsNumber(right)));
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
        let left = append(graph, &data, fast, Op::Javascript(JsOp::AsNumber(left)));
        let right = append(graph, &data, fast, Op::Javascript(JsOp::AsNumber(right)));
        let left = append(graph, &data, fast, Op::Javascript(JsOp::ToInt32(left)));
        let right = append(graph, &data, fast, Op::Javascript(JsOp::ToInt32(right)));
        let result = append(
            graph,
            &data,
            fast,
            Op::Javascript(JsOp::Binary(operator, left, right)),
        );
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

pub(in crate::jit::semantic) fn dispatch(function: &mut FunctionState) -> bool {
    function.ensure_values();
    let facts = &function.facts;
    let graph = &mut function.graph;
    let candidates: Vec<_> = facts
        .cfg
        .rpo
        .iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|&block| graph.blocks[block].insts.iter().copied())
        .filter(|&inst| {
            numeric_operands(&graph.insts[inst].op).is_some_and(|operands| {
                operands
                    .iter()
                    .all(|&value| facts.of(value).intersects(ValueType::NUMBER))
            })
        })
        .collect();
    for &inst in &candidates {
        let data = graph.insts[inst].clone();
        let operands = numeric_operands(&data.op).unwrap();
        let first = graph.blocks.len();
        let (block, continuation, fallback) = super::super::dispatch::split(graph, inst);
        let fast = graph.blocks.push(BlockData::default());
        let mut at = Cursor::at_end(graph, block, inst);
        let condition = match data.op {
            Op::Javascript(JsOp::GuardedCompare { numbers, .. }) => {
                at.add_unassigned(Op::BooleanAnd(numbers[0], numbers[1]))
            }
            _ => {
                let checks: Vec<_> = operands
                    .iter()
                    .map(|&value| at.add_unassigned(Op::Javascript(JsOp::IsNumber(value))))
                    .collect();
                let mut condition = checks[0];
                for &check in &checks[1..] {
                    condition = at.add_unassigned(Op::BooleanAnd(condition, check));
                }
                condition
            }
        };
        at.terminate_current(TerminatorKind::Branch(
            condition,
            Target::new(fast),
            Target::new(fallback),
        ));
        let mut at = Cursor::at_end(graph, fast, inst);
        let numbers: Vec<_> = operands
            .iter()
            .map(|&value| at.add_unassigned(Op::Javascript(JsOp::AsNumber(value))))
            .collect();
        let result = at.add_unassigned(numeric_op(&data.op, numbers));
        at.terminate_current(TerminatorKind::Jump(Target {
            block: continuation,
            args: vec![result],
        }));
        super::super::dispatch::extend_regions(graph, block, first);
    }
    if !candidates.is_empty() {
        function.invalidate_cfg();
    }
    !candidates.is_empty()
}

fn numeric_operands(op: &Op) -> Option<Vec<Value>> {
    use BinaryOperator as B;
    Some(match op {
        Op::Javascript(
            JsOp::Binary(
                B::Addition
                | B::Subtraction
                | B::Multiplication
                | B::Division
                | B::LessThan
                | B::LessEqualThan
                | B::GreaterThan
                | B::GreaterEqualThan
                | B::Equality
                | B::Inequality
                | B::StrictEquality
                | B::StrictInequality
                | B::BitwiseAnd
                | B::BitwiseOR
                | B::BitwiseXOR
                | B::ShiftLeft
                | B::ShiftRight
                | B::ShiftRightZeroFill,
                left,
                right,
            )
            | JsOp::GuardedCompare { left, right, .. },
        ) => vec![*left, *right],
        Op::Javascript(JsOp::Unary(UnaryOp::Increment | UnaryOp::Decrement, value)) => vec![*value],
        Op::Javascript(JsOp::CallNative(native, args)) if inline_math(*native, args.len()) => {
            args.clone()
        }
        _ => return None,
    })
}

fn inline_math(native: super::super::Native, arguments: usize) -> bool {
    use crate::heap::MathFunction;
    matches!(
        (crate::jit::math::METHODS[native.index()].1, arguments),
        (
            MathFunction::Floor | MathFunction::Ceil | MathFunction::Sqrt,
            1
        ) | (MathFunction::Min | MathFunction::Max, 1..)
    )
}

fn numeric_op(op: &Op, numbers: Vec<Value>) -> Op {
    match *op {
        Op::Javascript(JsOp::Binary(operator, ..) | JsOp::GuardedCompare { operator, .. }) => {
            Op::Javascript(JsOp::Binary(operator, numbers[0], numbers[1]))
        }
        Op::Javascript(JsOp::Unary(operator, _)) => {
            Op::Javascript(JsOp::Unary(operator, numbers[0]))
        }
        Op::Javascript(JsOp::CallNative(native, _)) => {
            Op::Javascript(JsOp::CallNative(native, numbers))
        }
        _ => unreachable!(),
    }
}
