use super::super::{Facts, Graph, MemoryFormat, Op as GraphOp, Value, ValueType};
use super::{Constant, Op, UnaryOp};
use crate::heap::TypedKind;
use oxc_syntax::operator::BinaryOperator;

pub(in crate::jit::semantic) fn result_types(graph: &Graph, op: &Op, facts: &Facts) -> ValueType {
    match op {
        Op::BindingLoad(_) => ValueType::ALL,
        Op::CellGet(_) => ValueType::VALUE,
        Op::BindingRestore(..) => ValueType::default(),
        Op::Constant(constant) => match constant {
            Constant::Undefined => ValueType::UNDEFINED,
            Constant::Null => ValueType::NULL,
            Constant::Boolean(_) => ValueType::BOOLEAN,
            Constant::Number(value) => number(*value),
            Constant::String(_) => ValueType::STRING,
            Constant::Uninitialized => ValueType::UNINITIALIZED,
        },
        Op::NewObject
        | Op::NewInstance(_)
        | Op::ConstructorResult(..)
        | Op::NewArray(_)
        | Op::NewHoleyArray(_)
        | Op::IteratorNew(_)
        | Op::EnumerateNew(_) => ValueType::OBJECT,
        Op::MapElement(..) => ValueType::ALL,
        Op::IsArrayIteration(..) => ValueType::BOOLEAN,
        Op::IteratorNext(_) | Op::EnumerateNext(_) => ValueType::ALL,
        Op::IsUninitialized(_)
        | Op::IsPropertyCached(..)
        | Op::IsShape(..)
        | Op::IsReadLayout(..) => ValueType::BOOLEAN,
        Op::PropertyLoad(..)
        | Op::RelatedPropertyLoad(..)
        | Op::ProbeProperty(..)
        | Op::ProbeShapeProperty(..)
        | Op::ProbeIndex(..)
        | Op::ConstructorPrototype(_)
        | Op::CachedProperty(..)
        | Op::ShapedGet(..)
        | Op::LayoutGet(..)
        | Op::FillPropertyCache(..) => ValueType::ALL,
        Op::View(_) => ValueType::VIEW,
        Op::TypedData(_) => ValueType::default(),
        Op::MemoryNumber(_, format) => match format {
            MemoryFormat::U32 | MemoryFormat::F32 | MemoryFormat::F64 => ValueType::NUMBER,
            _ => ValueType::INT32,
        },
        Op::TypedIndex(.., read) if read.bounded => match read.element {
            Some(TypedKind::Float32 | TypedKind::Float64 | TypedKind::Uint32) | None => {
                ValueType::NUMBER
            }
            Some(_) => ValueType::INT32,
        },
        Op::TypedIndex(..) => ValueType::NUMBER.union(ValueType::UNDEFINED),
        Op::IsTyped(..)
        | Op::IsIndexed(_)
        | Op::IsOrdinaryArray(_)
        | Op::IsArrayIndex(..)
        | Op::IsArray(..)
        | Op::IsExactPosition(..)
        | Op::IsInBounds(..) => ValueType::BOOLEAN,
        Op::PropertyCache(..) | Op::GlobalCache(..) => ValueType::INT64,
        Op::CanOverwrite(..) => ValueType::BOOLEAN,
        Op::IndexedLength(_) => ValueType::INT64,
        Op::ArrayKind(_) => ValueType::INT32,
        Op::IsNumber(_)
        | Op::IsInteger(_)
        | Op::IsIntegerInRange(..)
        | Op::IntegerCheck(..)
        | Op::IsString(_)
        | Op::Truthy(_)
        | Op::IsStringSplit(_)
        | Op::IsTypeof(..) => ValueType::BOOLEAN,
        Op::NewPack | Op::Spread(_) | Op::StringSplitParts(..) => ValueType::PACK,
        Op::StringConcat(_) => ValueType::STRING,
        Op::StringSplit(..) => ValueType::OBJECT,
        Op::ArrayPosition(_) => ValueType::INT64,
        Op::ToInt32(_) => ValueType::INT32,
        Op::AsInteger(_) => ValueType::INT64,
        Op::NumericFloat(_) | Op::SaturatingInteger(_) => ValueType::NUMBER,
        Op::AsString(_) => ValueType::STRING,
        Op::AsObject(_) => ValueType::OBJECT,
        Op::AsNumber(value) | Op::ToNumeric(value) | Op::Unary(UnaryOp::Plus, value)
            if facts.of(*value).within(ValueType::NUMBER) =>
        {
            facts.of(*value)
        }
        Op::AsNumber(_) | Op::ToNumeric(_) => ValueType::NUMBER,
        Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, value)
            if facts.of(*value).within(ValueType::INTEGER) =>
        {
            ValueType::INT64
        }
        Op::Unary(operator, _) => match operator {
            UnaryOp::Not => ValueType::BOOLEAN,
            UnaryOp::Typeof => ValueType::STRING,
            UnaryOp::Plus => ValueType::NUMBER,
            UnaryOp::BitwiseNot => ValueType::INT32,
            UnaryOp::Negate | UnaryOp::Increment | UnaryOp::Decrement => ValueType::NUMBER,
        },
        Op::Binary(BinaryOperator::Addition | BinaryOperator::Subtraction, left, right)
            if facts.of(*left).within(ValueType::INTEGER)
                && facts.of(*right).within(ValueType::INTEGER) =>
        {
            ValueType::INT64
        }
        Op::Binary(BinaryOperator::Multiplication, left, right)
            if integer_product(graph, facts, *left, *right) =>
        {
            ValueType::INT64
        }
        Op::Binary(BinaryOperator::Addition, left, right)
            if facts.of(*left).within(ValueType::INTEGER)
                && signed_zero_product(graph, facts, *right)
                || facts.of(*right).within(ValueType::INTEGER)
                    && signed_zero_product(graph, facts, *left) =>
        {
            ValueType::INT64
        }
        Op::Binary(operator, left, right) => match operator {
            BinaryOperator::Addition => {
                let (left, right) = (facts.of(*left), facts.of(*right));
                let nonstring = ValueType::NUMBER
                    .union(ValueType::BOOLEAN)
                    .union(ValueType::UNDEFINED)
                    .union(ValueType::NULL);
                if left.within(nonstring) && right.within(nonstring) {
                    ValueType::NUMBER
                } else if left.within(ValueType::STRING) || right.within(ValueType::STRING) {
                    ValueType::STRING
                } else {
                    ValueType::NUMBER.union(ValueType::STRING)
                }
            }
            BinaryOperator::Subtraction
            | BinaryOperator::Multiplication
            | BinaryOperator::Division
            | BinaryOperator::Remainder
            | BinaryOperator::Exponential => ValueType::NUMBER,
            BinaryOperator::ShiftLeft
            | BinaryOperator::ShiftRight
            | BinaryOperator::BitwiseOR
            | BinaryOperator::BitwiseXOR
            | BinaryOperator::BitwiseAnd => ValueType::INT32,
            BinaryOperator::ShiftRightZeroFill => ValueType::NUMBER,
            _ => ValueType::BOOLEAN,
        },
        Op::GuardedCompare { .. } => ValueType::BOOLEAN,
        Op::CallNative(..) => ValueType::NUMBER,
        Op::IsUndefined(_)
        | Op::IsNullish(_)
        | Op::IsNative(..)
        | Op::IsCode(..)
        | Op::GuardValid(_) => ValueType::BOOLEAN,
        _ => ValueType::VALUE,
    }
}

fn number(value: f64) -> ValueType {
    let integral = value.fract() == 0.0 && value >= i32::MIN as f64 && value <= i32::MAX as f64;
    if integral && !(value == 0.0 && value.is_sign_negative()) {
        ValueType::INT32
    } else {
        ValueType::DOUBLE
    }
}

/// An int32 times a narrow integer is exact but may be negative zero, which adding it to an
/// integer absorbs.
pub(in crate::jit::semantic) fn signed_zero_product(
    graph: &Graph,
    facts: &Facts,
    value: Value,
) -> bool {
    let Some(GraphOp::Javascript(Op::Binary(BinaryOperator::Multiplication, left, right))) =
        graph.defining_op(value)
    else {
        return false;
    };
    let int32 = |value| facts.of(value).within(ValueType::INT32);
    int32(*left) && int32(*right) && (narrow(graph, *left) || narrow(graph, *right))
}

fn narrow(graph: &Graph, value: Value) -> bool {
    match graph.defining_op(value) {
        Some(GraphOp::Javascript(Op::MemoryNumber(_, format))) => matches!(
            format,
            MemoryFormat::U8 | MemoryFormat::I8 | MemoryFormat::U16 | MemoryFormat::I16
        ),
        Some(GraphOp::Javascript(Op::TypedIndex(.., read))) => {
            read.bounded
                && matches!(
                    read.element,
                    Some(TypedKind::Uint8 | TypedKind::Int8 | TypedKind::Uint16 | TypedKind::Int16)
                )
        }
        _ => false,
    }
}

fn integer_product(graph: &Graph, facts: &Facts, left: Value, right: Value) -> bool {
    // Positive factors avoid negative zero; 2^31 * 2^22 stays exactly representable.
    let scaled = |value, factor| {
        facts.of(value).within(ValueType::INT32)
            && matches!(graph.defining_op(factor), Some(GraphOp::Javascript(Op::Constant(Constant::Number(n))))
                if *n > 0.0 && n.fract() == 0.0 && *n <= 4_194_304.0)
    };
    scaled(left, right) || scaled(right, left)
}

pub(in crate::jit::semantic) fn integer_arithmetic(
    operator: BinaryOperator,
) -> Option<super::super::low_level::Binary> {
    use super::super::low_level::Binary;
    match operator {
        BinaryOperator::Addition => Some(Binary::Add),
        BinaryOperator::Subtraction => Some(Binary::Subtract),
        BinaryOperator::Multiplication => Some(Binary::Multiply),
        _ => None,
    }
}

pub(in crate::jit::semantic) fn exact_integer_arithmetic(
    op: super::super::low_level::Binary,
    left: super::super::ranges::IntRange,
    right: super::super::ranges::IntRange,
) -> Option<super::super::ranges::IntRange> {
    use super::super::low_level::Binary;
    use super::super::ranges::IntRange;
    if !left.within(IntRange::SAFE) || !right.within(IntRange::SAFE) {
        return None;
    }
    if op == Binary::Multiply
        && ((left.min <= 0 && left.max >= 0 && right.min < 0)
            || (right.min <= 0 && right.max >= 0 && left.min < 0))
    {
        return None;
    }
    let result = left.arithmetic(op, right)?;
    result.within(IntRange::SAFE).then_some(result)
}
