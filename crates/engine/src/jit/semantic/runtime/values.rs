use crate::jit::semantic::JsOp;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{MemFlagsData, types};

use super::super::cursor::{Cursor, Insert};
use super::super::{Facts, FunctionState, Graph, Inst, Op, Repr, Value, ValueType, low_level};
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
    let mut insts = Vec::new();
    for &block in &facts.cfg.rpo {
        insts.clone_from(&graph.blocks[block].insts);
        for &inst in &insts {
            if let Op::Javascript(JsOp::View(object)) = graph.insts[inst].op {
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
            let Op::Javascript(JsOp::ConstructorResult(result, instance)) = graph.insts[inst].op
            else {
                continue;
            };
            let mut at = Cursor::before(graph, inst);
            let result = at.as_word(result);
            let instance = at.as_word(instance);
            let condition = layout::is_object(&mut at, result);
            let select = at.ins().select(condition, result, instance);
            graph.set_op(inst, Op::AsBoxed(select));
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
    let valid = at.ins().boolean_and(numeric, integral);
    let valid = at.ins().boolean_and(valid, safe);
    at.ins().boolean_and(valid, positive_zero)
}

fn scalar(types: ValueType) -> bool {
    types != ValueType::default()
        && (types.within(ValueType::BOOLEAN) || types.within(ValueType::NUMBER))
}

fn predicate(graph: &mut Graph, facts: &Facts, inst: Inst) -> Option<Value> {
    let op = &graph.insts[inst].op;
    let boxed = |_: &Graph, value: Value| !scalar(facts.of(value));
    Some(match *op {
        Op::Javascript(JsOp::IntegerCheck(value, float, exact)) => {
            integer_check(&mut Cursor::before(graph, inst), value, float, exact)
        }
        Op::Javascript(JsOp::IsInteger(value)) => {
            let mut at = Cursor::before(graph, inst);
            let bits = at.as_word(value);
            let float = at.ins().bitcast(types::F64, MemFlagsData::new(), bits);
            let integer = at.ins().fcvt_to_sint_sat(types::I64, float);
            let exact = at.ins().fcvt_from_sint(types::F64, integer);
            integer_check(&mut at, value, float, exact)
        }
        Op::Javascript(JsOp::IsNumber(value)) if boxed(graph, value) => {
            let mut at = Cursor::before(graph, inst);
            let word = at.as_word(value);
            layout::is_number(&mut at, word)
        }
        Op::Javascript(JsOp::IsNumber(value)) => {
            let number = !facts.of(value).within(ValueType::BOOLEAN);
            let mut at = Cursor::before(graph, inst);
            at.ins().boolean(number)
        }
        Op::Javascript(
            JsOp::IsUninitialized(value) | JsOp::IsUndefined(value) | JsOp::IsNullish(value),
        ) if !boxed(graph, value) => {
            let mut at = Cursor::before(graph, inst);
            at.ins().boolean(false)
        }
        Op::Javascript(JsOp::IsUninitialized(value)) => equals(
            &mut Cursor::before(graph, inst),
            value,
            JsValue::UNINITIALIZED,
        ),
        Op::Javascript(JsOp::IsUndefined(value)) => {
            equals(&mut Cursor::before(graph, inst), value, JsValue::UNDEFINED)
        }
        Op::Javascript(JsOp::IsNullish(value)) => {
            let mut at = Cursor::before(graph, inst);
            let word = at.as_word(value);
            let undefined =
                at.ins()
                    .icmp_imm_s(IntCC::Equal, word, JsValue::UNDEFINED.bits() as i64);
            let null = at
                .ins()
                .icmp_imm_s(IntCC::Equal, word, JsValue::NULL.bits() as i64);
            at.ins().boolean_or(undefined, null)
        }
        Op::Javascript(JsOp::IsString(value)) => {
            let mut at = Cursor::before(graph, inst);
            let word = at.as_word(value);
            layout::is_string(&mut at, word)
        }
        Op::Javascript(JsOp::IsCode(value, code)) => field_is(
            &mut Cursor::before(graph, inst),
            value,
            CODE_OFFSET,
            code.as_u32() + 1,
        ),
        Op::Javascript(JsOp::IsShape(object, shape)) => field_is(
            &mut Cursor::before(graph, inst),
            object,
            SHAPE_OFFSET,
            shape,
        ),
        Op::Javascript(JsOp::IsReadLayout(object, layout)) => {
            let mut at = Cursor::before(graph, inst);
            let hit = field_is(&mut at, object, SHAPE_OFFSET, layout.shape);
            match layout.storage {
                None => hit,
                Some(storage) => {
                    // A failed receiver check may leave the recorded storage dead or reused.
                    let holder = at.ins().iconst(types::I64, storage.identity as i64);
                    let empty = at
                        .ins()
                        .iconst(types::I64, crate::heap::NO_ELEMENTS.as_ptr() as i64);
                    let holder = at.ins().select(hit, holder, empty);
                    let shape =
                        at.ins()
                            .load(types::I32, MemFlagsData::trusted(), holder, SHAPE_OFFSET);
                    let expected = at.ins().iconst(types::I32, i64::from(storage.shape));
                    let valid = at.ins().icmp(IntCC::Equal, shape, expected);
                    at.ins().boolean_and(hit, valid)
                }
            }
        }
        Op::Javascript(JsOp::IsTyped(object, Some(element))) => {
            let expected = kind::TYPED | TypedKind::element(element);
            field_is(
                &mut Cursor::before(graph, inst),
                object,
                KIND_OFFSET,
                expected,
            )
        }
        Op::Javascript(JsOp::IsTyped(object, None)) => {
            has_kind(&mut Cursor::before(graph, inst), object, kind::TYPED)
        }
        Op::Javascript(JsOp::IsOrdinaryArray(object)) => {
            has_kind(&mut Cursor::before(graph, inst), object, kind::ARRAY)
        }
        Op::Javascript(JsOp::IsIndexed(object)) => has_kind(
            &mut Cursor::before(graph, inst),
            object,
            kind::ARRAY | kind::TYPED,
        ),
        _ => return None,
    })
}
