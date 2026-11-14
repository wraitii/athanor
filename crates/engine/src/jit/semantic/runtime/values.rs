use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{MemFlagsData, types};

use super::super::cursor::{Cursor, Insert};
use super::super::{Cfg, FunctionState, Graph, Inst, Op, Repr, Value, low_level};
use crate::heap::{CODE_OFFSET, KIND_OFFSET, SHAPE_OFFSET, TypedKind, kind};
use crate::value::{TAG_FIRST, TAG_SHIFT, TAG_SPAN, TAG_STRING, Value as JsValue};

const TAG_MASK: u64 = 0xffff_0000_0000_0000;
const ADDRESS_MASK: u64 = (1 << crate::value::TAG_SHIFT) - 1;

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
            let mut at = Cursor::before(graph, inst);
            let result = at.as_word(result);
            let instance = at.as_word(instance);
            let mask = at.ins().iconst(types::I64, TAG_MASK as i64);
            let tag = at.ins().band(result, mask);
            let object = at.ins().iconst(types::I64, crate::value::TAG_OBJECT as i64);
            let condition = at.ins().icmp(IntCC::Equal, tag, object);
            let select = at.ins().select(condition, result, instance);
            graph.insts[inst].op = Op::AsBoxed(select);
        }
    }
}

fn view(graph: &mut Graph, inst: Inst, object: Value) {
    let (valid, address, empty) = view_parts(&mut Cursor::before(graph, inst), object);
    graph.insts[inst].op = Op::LowLevel(low_level::Op::Select {
        condition: valid,
        yes: address,
        no: empty,
    });
    let result = graph.insts[inst].result().unwrap();
    graph.reprs[result] = Some(Repr::I64);
}

fn view_parts(at: &mut Cursor, object: Value) -> (Value, Value, Value) {
    let object = at.as_word(object);
    let mask = at.ins().iconst(types::I64, TAG_MASK as i64);
    let tag = at.ins().band(object, mask);
    let expected = at.ins().iconst(types::I64, crate::value::TAG_OBJECT as i64);
    let is_object = at.ins().icmp(IntCC::Equal, tag, expected);
    let mask = at.ins().iconst(types::I64, ADDRESS_MASK as i64);
    let address = at.ins().band(object, mask);
    let zero = at.ins().iconst(types::I64, 0);
    let present = at.ins().icmp(IntCC::NotEqual, address, zero);
    let valid = at.ins().band(is_object, present);
    let empty = at
        .ins()
        .iconst(types::I64, crate::heap::NO_ELEMENTS.as_ptr() as i64);
    (valid, address, empty)
}

fn tag(at: &mut Cursor, word: Value) -> Value {
    at.ins().ushr_imm_s(word, TAG_SHIFT as i64)
}

fn number_range(at: &mut Cursor, word: Value) -> (Value, Value) {
    let tag = tag(at, word);
    let offset = at.ins().iadd_imm_s(tag, TAG_FIRST.wrapping_neg() as i64);
    let span = at.ins().iconst(types::I64, TAG_SPAN as i64);
    (offset, span)
}

fn equals(graph: &mut Graph, inst: Inst, value: Value, expected: JsValue) -> low_level::Op {
    let mut at = Cursor::before(graph, inst);
    let word = at.as_word(value);
    let right = at.ins().iconst(types::I64, expected.bits() as i64);
    low_level::Op::IntCompare {
        condition: IntCC::Equal,
        left: word,
        right,
    }
}

fn view_field(at: &mut Cursor, object: Value, offset: i32) -> Value {
    let (valid, address, empty) = view_parts(at, object);
    let view = at.ins().select(valid, address, empty);
    at.ins()
        .load(types::I32, MemFlagsData::trusted(), view, offset)
}

fn field_is(at: &mut Cursor, object: Value, offset: i32, expected: u32) -> low_level::Op {
    let field = view_field(at, object, offset);
    let expected = at.ins().iconst(types::I32, i64::from(expected));
    low_level::Op::IntCompare {
        condition: IntCC::Equal,
        left: field,
        right: expected,
    }
}

fn has_kind(at: &mut Cursor, object: Value, mask: u32) -> low_level::Op {
    let kind = view_field(at, object, KIND_OFFSET);
    let masked = at.ins().band_imm_u(kind, i64::from(mask));
    let zero = at.ins().iconst(types::I32, 0);
    low_level::Op::IntCompare {
        condition: IntCC::NotEqual,
        left: masked,
        right: zero,
    }
}

fn integer_check(at: &mut Cursor, value: Value, float: Value, exact: Value) -> low_level::Op {
    let word = at.as_word(value);
    let (offset, span) = number_range(at, word);
    let numeric = at.ins().icmp(IntCC::UnsignedGreaterThan, offset, span);
    let integral = at.ins().fcmp(FloatCC::Equal, float, exact);
    let magnitude = at.add(
        Op::LowLevel(low_level::Op::Unary {
            op: low_level::Unary::FloatAbs,
            ty: types::F64,
            value: float,
        }),
        Repr::F64,
    );
    let limit = at.ins().f64const(9007199254740991.0);
    let safe = at.ins().fcmp(FloatCC::LessThanOrEqual, magnitude, limit);
    let negative_zero = at.ins().iconst(types::I64, i64::MIN);
    let positive_zero = at.ins().icmp(IntCC::NotEqual, word, negative_zero);
    let valid = at.ins().band(numeric, integral);
    let valid = at.ins().band(valid, safe);
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
        Op::IntegerCheck(value, float, exact) => {
            integer_check(&mut Cursor::before(graph, inst), value, float, exact)
        }
        Op::IsInteger(value) => {
            let mut at = Cursor::before(graph, inst);
            let bits = at.as_word(value);
            let float = at.ins().bitcast(types::F64, MemFlagsData::new(), bits);
            let integer = at.ins().fcvt_to_sint_sat(types::I64, float);
            let exact = at.ins().fcvt_from_sint(types::F64, integer);
            integer_check(&mut at, value, float, exact)
        }
        Op::IsNumber(value) if boxed(graph, value) => {
            let mut at = Cursor::before(graph, inst);
            let word = at.as_word(value);
            let (offset, span) = number_range(&mut at, word);
            low_level::Op::IntCompare {
                condition: IntCC::UnsignedGreaterThan,
                left: offset,
                right: span,
            }
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
        Op::IsUninitialized(value) => equals(graph, inst, value, JsValue::UNINITIALIZED),
        Op::IsUndefined(value) => equals(graph, inst, value, JsValue::UNDEFINED),
        Op::IsNullish(value) => {
            let mut at = Cursor::before(graph, inst);
            let word = at.as_word(value);
            let undefined =
                at.ins()
                    .icmp_imm_s(IntCC::Equal, word, JsValue::UNDEFINED.bits() as i64);
            let null = at
                .ins()
                .icmp_imm_s(IntCC::Equal, word, JsValue::NULL.bits() as i64);
            low_level::Op::Binary {
                op: low_level::Binary::Or,
                left: undefined,
                right: null,
            }
        }
        Op::IsString(value) => {
            let mut at = Cursor::before(graph, inst);
            let word = at.as_word(value);
            let tag = tag(&mut at, word);
            let string = at
                .ins()
                .iconst(types::I64, (TAG_STRING >> TAG_SHIFT) as i64);
            low_level::Op::IntCompare {
                condition: IntCC::Equal,
                left: tag,
                right: string,
            }
        }
        Op::IsCode(value, code) => field_is(
            &mut Cursor::before(graph, inst),
            value,
            CODE_OFFSET,
            code.as_u32() + 1,
        ),
        Op::IsShape(object, shape) => field_is(
            &mut Cursor::before(graph, inst),
            object,
            SHAPE_OFFSET,
            shape,
        ),
        Op::IsTyped(object, Some(element)) => {
            let expected = kind::TYPED | TypedKind::element(element);
            field_is(
                &mut Cursor::before(graph, inst),
                object,
                KIND_OFFSET,
                expected,
            )
        }
        Op::IsTyped(object, None) => {
            has_kind(&mut Cursor::before(graph, inst), object, kind::TYPED)
        }
        Op::IsIndexed(object) => has_kind(
            &mut Cursor::before(graph, inst),
            object,
            kind::ARRAY | kind::TYPED,
        ),
        _ => return None,
    })
}
