use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{MemFlagsData, Type, types};

use super::super::{Cfg, FunctionState, Graph, Inst, Op, Repr, Value, low_level};
use super::helpers::{insert_before, integer, word};
use crate::heap::{CODE_OFFSET, KIND_OFFSET, SHAPE_OFFSET, TypedKind, kind};
use crate::value::{TAG_FIRST, TAG_SHIFT, TAG_SPAN, TAG_STRING, Value as JsValue};

const TAG_MASK: u64 = 0xffff_0000_0000_0000;
const ADDRESS_MASK: u64 = (1 << crate::value::TAG_SHIFT) - 1;

fn insert(graph: &mut Graph, at: Inst, op: low_level::Op, repr: Repr) -> Value {
    insert_before(graph, at, op, Some(repr)).1.unwrap()
}

pub(in crate::jit::semantic) fn lower(function: &mut FunctionState) {
    let graph = &mut function.graph;
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            if let Op::View(object) = graph.insts[inst].op {
                view(graph, inst, object);
                continue;
            }
            if let Some(op) = predicate(graph, inst) {
                graph.insts[inst].op = Op::LowLevel(op);
                continue;
            }
            let Op::ConstructorResult(result, instance) = graph.insts[inst].op else {
                continue;
            };
            let result = word(graph, inst, result);
            let instance = word(graph, inst, instance);
            let mask = insert(graph, inst, integer(TAG_MASK), Repr::I64);
            let tag = low_level::Op::Binary {
                op: low_level::Binary::And,
                left: result,
                right: mask,
            };
            let tag = insert(graph, inst, tag, Repr::I64);
            let object = integer(crate::value::TAG_OBJECT);
            let object = insert(graph, inst, object, Repr::I64);
            let condition = low_level::Op::IntCompare {
                condition: IntCC::Equal,
                left: tag,
                right: object,
            };
            let condition = insert(graph, inst, condition, Repr::I8);
            let select = low_level::Op::Select {
                condition,
                yes: result,
                no: instance,
            };
            let select = insert(graph, inst, select, Repr::I64);
            graph.insts[inst].op = Op::AsBoxed(select);
        }
    }
}

fn compare(graph: &mut Graph, at: Inst, condition: IntCC, left: Value, right: Value) -> Value {
    let compare = low_level::Op::IntCompare {
        condition,
        left,
        right,
    };
    insert(graph, at, compare, Repr::I8)
}

fn and(graph: &mut Graph, at: Inst, left: Value, right: Value, repr: Repr) -> Value {
    let and = low_level::Op::Binary {
        op: low_level::Binary::And,
        left,
        right,
    };
    insert(graph, at, and, repr)
}

fn view(graph: &mut Graph, inst: Inst, object: Value) {
    graph.insts[inst].op = Op::LowLevel(view_address(graph, inst, object));
    let result = graph.insts[inst].result().unwrap();
    graph.reprs[result] = Some(Repr::I64);
}

fn view_address(graph: &mut Graph, inst: Inst, object: Value) -> low_level::Op {
    let object = word(graph, inst, object);
    let mask = insert(graph, inst, integer(TAG_MASK), Repr::I64);
    let tag = and(graph, inst, object, mask, Repr::I64);
    let expected = insert(graph, inst, integer(crate::value::TAG_OBJECT), Repr::I64);
    let is_object = compare(graph, inst, IntCC::Equal, tag, expected);
    let mask = insert(graph, inst, integer(ADDRESS_MASK), Repr::I64);
    let address = and(graph, inst, object, mask, Repr::I64);
    let zero = insert(graph, inst, integer(0), Repr::I64);
    let present = compare(graph, inst, IntCC::NotEqual, address, zero);
    let valid = and(graph, inst, is_object, present, Repr::I8);
    let empty = integer(crate::heap::NO_ELEMENTS.as_ptr() as u64);
    let empty = insert(graph, inst, empty, Repr::I64);
    low_level::Op::Select {
        condition: valid,
        yes: address,
        no: empty,
    }
}

fn constant(graph: &mut Graph, at: Inst, ty: Type, bits: u64) -> Value {
    insert(
        graph,
        at,
        low_level::Op::Integer { ty, bits },
        Repr::from_machine(ty),
    )
}

fn binary(
    graph: &mut Graph,
    at: Inst,
    op: low_level::Binary,
    left: Value,
    right: Value,
    repr: Repr,
) -> Value {
    insert(graph, at, low_level::Op::Binary { op, left, right }, repr)
}

fn unary(graph: &mut Graph, at: Inst, op: low_level::Unary, ty: Type, value: Value) -> Value {
    let unary = low_level::Op::Unary { op, ty, value };
    insert(graph, at, unary, Repr::from_machine(ty))
}

fn int_compare(condition: IntCC, left: Value, right: Value) -> low_level::Op {
    low_level::Op::IntCompare {
        condition,
        left,
        right,
    }
}

fn tag(graph: &mut Graph, at: Inst, word: Value) -> Value {
    let shift = constant(graph, at, types::I64, TAG_SHIFT as u64);
    let op = low_level::Binary::ShiftRightUnsigned;
    binary(graph, at, op, word, shift, Repr::I64)
}

fn is_number(graph: &mut Graph, at: Inst, word: Value) -> low_level::Op {
    let tag = tag(graph, at, word);
    let first = constant(graph, at, types::I64, TAG_FIRST.wrapping_neg());
    let offset = binary(graph, at, low_level::Binary::Add, tag, first, Repr::I64);
    let span = constant(graph, at, types::I64, TAG_SPAN);
    int_compare(IntCC::UnsignedGreaterThan, offset, span)
}

fn equals(graph: &mut Graph, at: Inst, word: Value, expected: JsValue) -> low_level::Op {
    let expected = constant(graph, at, types::I64, expected.bits());
    int_compare(IntCC::Equal, word, expected)
}

fn view_field(graph: &mut Graph, at: Inst, object: Value, offset: i32) -> Value {
    let view = view_address(graph, at, object);
    let view = insert(graph, at, view, Repr::I64);
    let load = low_level::Op::Load {
        ty: types::I32,
        address: view,
        offset,
        flags: MemFlagsData::trusted(),
    };
    insert(graph, at, load, Repr::I32)
}

fn field_is(
    graph: &mut Graph,
    at: Inst,
    object: Value,
    offset: i32,
    expected: u32,
) -> low_level::Op {
    let field = view_field(graph, at, object, offset);
    let expected = constant(graph, at, types::I32, u64::from(expected));
    int_compare(IntCC::Equal, field, expected)
}

fn has_kind(graph: &mut Graph, at: Inst, object: Value, mask: u32) -> low_level::Op {
    let kind = view_field(graph, at, object, KIND_OFFSET);
    let mask = constant(graph, at, types::I32, u64::from(mask));
    let masked = binary(graph, at, low_level::Binary::And, kind, mask, Repr::I32);
    let zero = constant(graph, at, types::I32, 0);
    int_compare(IntCC::NotEqual, masked, zero)
}

fn integer_check(
    graph: &mut Graph,
    at: Inst,
    value: Value,
    float: Value,
    exact: Value,
) -> low_level::Op {
    let word = word(graph, at, value);
    let numeric = is_number(graph, at, word);
    let numeric = insert(graph, at, numeric, Repr::I8);
    let integral = low_level::Op::FloatCompare {
        condition: FloatCC::Equal,
        left: float,
        right: exact,
    };
    let integral = insert(graph, at, integral, Repr::I8);
    let magnitude = unary(graph, at, low_level::Unary::FloatAbs, types::F64, float);
    let limit = low_level::Op::Float64(9007199254740991.0f64.to_bits());
    let limit = insert(graph, at, limit, Repr::F64);
    let safe = low_level::Op::FloatCompare {
        condition: FloatCC::LessThanOrEqual,
        left: magnitude,
        right: limit,
    };
    let safe = insert(graph, at, safe, Repr::I8);
    let negative_zero = constant(graph, at, types::I64, i64::MIN as u64);
    let positive_zero = int_compare(IntCC::NotEqual, word, negative_zero);
    let positive_zero = insert(graph, at, positive_zero, Repr::I8);
    let valid = and(graph, at, numeric, integral, Repr::I8);
    let valid = and(graph, at, valid, safe, Repr::I8);
    low_level::Op::Binary {
        op: low_level::Binary::And,
        left: valid,
        right: positive_zero,
    }
}

fn predicate(graph: &mut Graph, inst: Inst) -> Option<low_level::Op> {
    let op = graph.insts[inst].op.clone();
    let boxed = |graph: &Graph, value: Value| graph.repr(value) == Repr::Boxed;
    let falsy = low_level::Op::Integer {
        ty: types::I8,
        bits: 0,
    };
    Some(match op {
        Op::IntegerCheck(value, float, exact) => integer_check(graph, inst, value, float, exact),
        Op::IsInteger(value) => {
            let bits = word(graph, inst, value);
            let float = unary(graph, inst, low_level::Unary::Bitcast, types::F64, bits);
            let saturate = low_level::Unary::SignedSaturatingFromFloat;
            let integer = unary(graph, inst, saturate, types::I64, float);
            let from_signed = low_level::Unary::FloatFromSigned;
            let exact = unary(graph, inst, from_signed, types::F64, integer);
            integer_check(graph, inst, value, float, exact)
        }
        Op::IsNumber(value) if boxed(graph, value) => {
            let word = word(graph, inst, value);
            is_number(graph, inst, word)
        }
        Op::IsNumber(value) => low_level::Op::Integer {
            ty: types::I8,
            bits: u64::from(graph.repr(value) != Repr::I8),
        },
        Op::IsUninitialized(value) | Op::IsUndefined(value) | Op::IsNullish(value)
            if !boxed(graph, value) =>
        {
            falsy
        }
        Op::IsUninitialized(value) => {
            let word = word(graph, inst, value);
            equals(graph, inst, word, JsValue::UNINITIALIZED)
        }
        Op::IsUndefined(value) => {
            let word = word(graph, inst, value);
            equals(graph, inst, word, JsValue::UNDEFINED)
        }
        Op::IsNullish(value) => {
            let word = word(graph, inst, value);
            let undefined = equals(graph, inst, word, JsValue::UNDEFINED);
            let undefined = insert(graph, inst, undefined, Repr::I8);
            let null = equals(graph, inst, word, JsValue::NULL);
            let null = insert(graph, inst, null, Repr::I8);
            low_level::Op::Binary {
                op: low_level::Binary::Or,
                left: undefined,
                right: null,
            }
        }
        Op::IsString(value) => {
            let word = word(graph, inst, value);
            let tag = tag(graph, inst, word);
            let string = constant(graph, inst, types::I64, TAG_STRING >> TAG_SHIFT);
            int_compare(IntCC::Equal, tag, string)
        }
        Op::IsCode(value, code) => field_is(graph, inst, value, CODE_OFFSET, code.as_u32() + 1),
        Op::IsShape(object, shape) => field_is(graph, inst, object, SHAPE_OFFSET, shape),
        Op::IsTyped(object, Some(element)) => {
            let expected = kind::TYPED | TypedKind::element(element);
            field_is(graph, inst, object, KIND_OFFSET, expected)
        }
        Op::IsTyped(object, None) => has_kind(graph, inst, object, kind::TYPED),
        Op::IsIndexed(object) => has_kind(graph, inst, object, kind::ARRAY | kind::TYPED),
        _ => return None,
    })
}
