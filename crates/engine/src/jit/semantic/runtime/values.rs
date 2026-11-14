use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{MemFlagsData, types};

use super::super::cursor::{Cursor, Insert};
use super::super::{Cfg, Facts, FunctionState, Graph, Inst, Op, Repr, Value, ValueType, low_level};
use super::layout;
use crate::FxHashMap;
use crate::heap::{CODE_OFFSET, KIND_OFFSET, SHAPE_OFFSET, TypedKind, kind};
use crate::value::Value as JsValue;

pub(in crate::jit::semantic) fn lower(function: &mut FunctionState) {
    function.ensure_values();
    let graph = &mut function.graph;
    let facts = &function.facts;
    let mut aliases = FxHashMap::default();
    let mut removed = Vec::new();
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            if let Op::View(object) = graph.insts[inst].op {
                let result = graph.insts[inst].result().unwrap();
                let view = view(graph, inst, object);
                aliases.insert(result, view);
                removed.push(inst);
                continue;
            }
            if let Some(value) = predicate(graph, facts, inst) {
                let result = graph.insts[inst].result().unwrap();
                aliases.insert(result, value);
                removed.push(inst);
                continue;
            }
            let Op::ConstructorResult(result, instance) = graph.insts[inst].op else {
                continue;
            };
            let mut at = Cursor::before(graph, inst);
            let result = at.as_word(result);
            let instance = at.as_word(instance);
            let condition = layout::is_object(&mut at, result);
            let select = at.ins().select(condition, result, instance);
            graph.insts[inst].op = Op::AsBoxed(select);
        }
    }
    graph.replace_uses(&aliases);
    for inst in removed {
        graph.remove_inst(inst);
    }
}

fn view(graph: &mut Graph, inst: Inst, object: Value) -> Value {
    let mut at = Cursor::before(graph, inst);
    let word = at.as_word(object);
    layout::object_view(&mut at, word)
}

fn equals(at: &mut Cursor, value: Value, expected: JsValue) -> Value {
    let word = at.as_word(value);
    at.ins()
        .icmp_imm_s(IntCC::Equal, word, expected.bits() as i64)
}

fn view_field(at: &mut Cursor, object: Value, offset: i32) -> Value {
    let object = at.as_word(object);
    let view = layout::object_view(at, object);
    at.ins()
        .load(types::I32, MemFlagsData::trusted(), view, offset)
}

fn field_is(at: &mut Cursor, object: Value, offset: i32, expected: u32) -> Value {
    let field = view_field(at, object, offset);
    let expected = at.ins().iconst(types::I32, i64::from(expected));
    at.ins().icmp(IntCC::Equal, field, expected)
}

fn has_kind(at: &mut Cursor, object: Value, mask: u32) -> Value {
    let kind = view_field(at, object, KIND_OFFSET);
    at.ins().test_mask(kind, u64::from(mask))
}

fn integer_check(at: &mut Cursor, value: Value, float: Value, exact: Value) -> Value {
    let word = at.as_word(value);
    let numeric = layout::is_number(at, word);
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
    at.ins().band(valid, positive_zero)
}

fn scalar(types: ValueType) -> bool {
    types != ValueType::default()
        && (types.within(ValueType::BOOLEAN) || types.within(ValueType::NUMBER))
}

fn predicate(graph: &mut Graph, facts: &Facts, inst: Inst) -> Option<Value> {
    let op = graph.insts[inst].op.clone();
    let boxed = |_: &Graph, value: Value| !scalar(facts.of(value));
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
            layout::is_number(&mut at, word)
        }
        Op::IsNumber(value) => {
            let bits = i64::from(!facts.of(value).within(ValueType::BOOLEAN));
            let mut at = Cursor::before(graph, inst);
            at.ins().iconst(types::I8, bits)
        }
        Op::IsUninitialized(value) | Op::IsUndefined(value) | Op::IsNullish(value)
            if !boxed(graph, value) =>
        {
            let mut at = Cursor::before(graph, inst);
            at.ins().iconst(types::I8, 0)
        }
        Op::IsUninitialized(value) => equals(
            &mut Cursor::before(graph, inst),
            value,
            JsValue::UNINITIALIZED,
        ),
        Op::IsUndefined(value) => {
            equals(&mut Cursor::before(graph, inst), value, JsValue::UNDEFINED)
        }
        Op::IsNullish(value) => {
            let mut at = Cursor::before(graph, inst);
            let word = at.as_word(value);
            let undefined =
                at.ins()
                    .icmp_imm_s(IntCC::Equal, word, JsValue::UNDEFINED.bits() as i64);
            let null = at
                .ins()
                .icmp_imm_s(IntCC::Equal, word, JsValue::NULL.bits() as i64);
            at.ins().bor(undefined, null)
        }
        Op::IsString(value) => {
            let mut at = Cursor::before(graph, inst);
            let word = at.as_word(value);
            layout::is_string(&mut at, word)
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
