use crate::jit::semantic::JsOp;
use cranelift_codegen::ir::{MemFlagsData, types};

use super::super::cursor::{Cursor, Insert};
use super::super::{Facts, FunctionState, Op, Repr, TypedRead, Value, ValueType, low_level};
use crate::jit::helpers::Helper;
use crate::value::Value as JsValue;

enum Arg {
    Value(Value, Repr),
    Integer(i64),
    PropertyCache(super::super::Inst, String),
    GlobalCache(String),
    Boxed(JsValue),
    Vector(Vec<Value>),
}

struct Call {
    helper: Helper,
    args: Vec<Arg>,
    produces: Option<Repr>,
}

fn boxed(values: &[Value]) -> Vec<Arg> {
    values
        .iter()
        .map(|&value| Arg::Value(value, Repr::Boxed))
        .collect()
}

fn vector(values: &[Value]) -> [Arg; 2] {
    [
        Arg::Vector(values.to_vec()),
        Arg::Integer(values.len() as i64),
    ]
}

fn call(helper: Helper, args: Vec<Arg>, produces: bool) -> Option<Call> {
    Some(Call {
        helper,
        args,
        produces: produces.then_some(Repr::Boxed),
    })
}

fn key(facts: &Facts, key: Value) -> (Helper, Arg) {
    let types = facts.of(key);
    if types != ValueType::default() && types.within(ValueType::NUMBER) {
        return (Helper::IndexGetNumeric, Arg::Value(key, Repr::F64));
    }
    (Helper::IndexGet, Arg::Value(key, Repr::Boxed))
}

fn select(inst: super::super::Inst, op: &Op, facts: &Facts) -> Option<Call> {
    let Op::Javascript(op) = op else {
        return None;
    };
    match op {
        JsOp::Get(object, name) => call(
            Helper::PropertyGetCached,
            vec![
                Arg::Value(*object, Repr::Boxed),
                Arg::PropertyCache(inst, name.clone()),
            ],
            true,
        ),
        JsOp::Set(object, name, value)
        | JsOp::Define(object, name, value)
        | JsOp::ShapedSet(object, name, _, value) => call(
            Helper::PropertySetCached,
            vec![
                Arg::Value(*object, Repr::Boxed),
                Arg::PropertyCache(inst, name.clone()),
                Arg::Value(*value, Repr::Boxed),
            ],
            false,
        ),
        JsOp::ProbeProperty(object, name) => call(
            Helper::SemanticPropertyProbeCached,
            vec![
                Arg::Value(*object, Repr::Boxed),
                Arg::PropertyCache(inst, name.clone()),
            ],
            true,
        ),
        JsOp::GlobalRead(name) | JsOp::GlobalReadOptional(name) | JsOp::ProbeGlobal(name) => call(
            match op {
                JsOp::GlobalRead(_) => Helper::GlobalGetBorrowed,
                JsOp::GlobalReadOptional(_) => Helper::GlobalGetOptionalBorrowed,
                _ => Helper::SemanticGlobalProbe,
            },
            vec![Arg::GlobalCache(name.clone())],
            true,
        ),
        JsOp::GetIndex(object, index)
        | JsOp::ArrayGet(object, index)
        | JsOp::TypedIndex(object, index, TypedRead { element: None, .. }) => {
            let (helper, index) = key(facts, *index);
            call(helper, vec![Arg::Value(*object, Repr::Boxed), index], true)
        }
        JsOp::Delete(object, index) => call(Helper::Delete, boxed(&[*object, *index]), true),
        JsOp::SetIndex(object, index, value) => {
            call(Helper::IndexSet, boxed(&[*object, *index, *value]), false)
        }
        JsOp::ClosureThis(value) => call(Helper::ClosureThis, boxed(&[*value]), true),
        JsOp::IsArrayIteration(callee, receiver, method) => {
            let mut args = boxed(&[*callee, *receiver]);
            args.push(Arg::Integer(*method as i64));
            call(Helper::ArrayIterationGuard, args, true)
        }
        JsOp::MapElement(receiver, index) => call(
            Helper::ArrayMapElement,
            vec![
                Arg::Value(*receiver, Repr::Boxed),
                Arg::Value(*index, Repr::I64),
            ],
            true,
        ),
        JsOp::Throw(value) => call(Helper::Throw, boxed(&[*value]), false),
        JsOp::PropertyKey(value) => call(Helper::PropertyKey, boxed(&[*value]), true),
        JsOp::MarkClass(value) => call(Helper::MarkClass, boxed(&[*value]), false),
        JsOp::DisableConstructor(value) => {
            call(Helper::DisableConstructor, boxed(&[*value]), false)
        }
        JsOp::IsTypeof(value, kind) => {
            let mut args = boxed(&[*value]);
            args.push(Arg::Boxed(JsValue::number(*kind as f64)));
            call(Helper::IsTypeof, args, true)
        }
        JsOp::IsNative(value, native) => {
            let mut args = boxed(&[*value]);
            args.push(Arg::Integer(native.index() as i64));
            call(Helper::SemanticMathNative, args, true)
        }
        JsOp::IsStringSplit(value) => call(Helper::IsStringSplit, boxed(&[*value]), true),
        JsOp::ProbeIndex(object, key) => {
            call(Helper::SemanticIndexProbe, boxed(&[*object, *key]), true)
        }
        JsOp::ConstructorPrototype(callee) => {
            call(Helper::ConstructorPrototype, boxed(&[*callee]), true)
        }
        JsOp::IteratorNew(value) => call(Helper::IterateNew, boxed(&[*value]), true),
        JsOp::IteratorNext(value) => call(Helper::IterateNext, boxed(&[*value]), true),
        JsOp::EnumerateNew(value) => call(Helper::EnumerateNew, boxed(&[*value]), true),
        JsOp::EnumerateNext(value) => call(Helper::EnumerateNext, boxed(&[*value]), true),
        JsOp::ArrayAppend(array, value, spread) => call(
            if *spread {
                Helper::ArrayExtend
            } else {
                Helper::ArrayAppend
            },
            boxed(&[*array, *value]),
            false,
        ),
        JsOp::PackAppend(pack, value, spread) => call(
            if *spread {
                Helper::PackExtend
            } else {
                Helper::PackAppend
            },
            boxed(&[*pack, *value]),
            false,
        ),
        JsOp::PackRelease(pack) => call(Helper::PackRelease, boxed(&[*pack]), false),
        JsOp::CallPack(callee, receiver, args) => {
            call(Helper::CallPack, boxed(&[*callee, *receiver, *args]), true)
        }
        JsOp::ConstructPack(callee, args) => {
            call(Helper::ConstructPack, boxed(&[*callee, *args]), true)
        }
        JsOp::ConstructorCallPack(callee, receiver, args) => call(
            Helper::ConstructorCallPack,
            boxed(&[*callee, *receiver, *args]),
            true,
        ),
        JsOp::ConstructPackResolved(callee, prototype, args) => call(
            Helper::ConstructPackResolved,
            boxed(&[*callee, *prototype, *args]),
            true,
        ),
        JsOp::Construct { callee, args } => {
            let mut call_args = boxed(&[*callee]);
            call_args.extend(vector(args));
            call(Helper::Construct, call_args, true)
        }
        JsOp::ConstructResolved {
            callee,
            prototype,
            args,
        } => {
            let mut call_args = boxed(&[*callee, *prototype]);
            call_args.extend(vector(args));
            call(Helper::ConstructResolved, call_args, true)
        }
        JsOp::ConstructorCall {
            callee,
            receiver,
            args,
        } => {
            let mut call_args = boxed(&[*callee, *receiver]);
            call_args.extend(vector(args));
            call(Helper::ConstructorCall, call_args, true)
        }
        JsOp::StringSplit(receiver, args) => {
            let mut call_args = vec![Arg::Integer(0), Arg::Value(*receiver, Repr::Boxed)];
            call_args.extend(vector(args));
            call(Helper::StringSplit, call_args, true)
        }
        JsOp::NewArray(args) => call(Helper::ArrayNew, vector(args).into(), true),
        JsOp::StringConcat(args) => call(Helper::StringConcat, vector(args).into(), true),
        JsOp::CellNew(value) => call(Helper::CellNew, boxed(&[*value]), true),
        JsOp::CellGet(cell) => call(Helper::CellGet, boxed(&[*cell]), true),
        JsOp::CellSet(cell, value) => call(Helper::CellSet, boxed(&[*cell, *value]), false),
        JsOp::CaptureGet(closure, index) => call(
            Helper::CaptureGet,
            vec![
                Arg::Value(*closure, Repr::Boxed),
                Arg::Integer(*index as i64),
            ],
            true,
        ),
        JsOp::NewHoleyArray(length) => call(Helper::ArrayMapNew, boxed(&[*length]), true),
        JsOp::NewInstance(prototype) => call(Helper::NewInstance, boxed(&[*prototype]), true),
        JsOp::NewObject => call(
            Helper::ObjectNew,
            vec![Arg::Boxed(JsValue::UNDEFINED)],
            true,
        ),
        JsOp::PackTake(pack, index) => call(Helper::PackTake, boxed(&[*pack, *index]), true),
        _ => None,
    }
}

fn stack_vector(at: &mut Cursor, values: &[Value]) -> Value {
    if values.is_empty() {
        return at.ins().iconst(types::I64, 0);
    }
    let graph = at.graph_mut();
    let slot = low_level::StackSlot(graph.low_level_slots.len());
    graph.low_level_slots.push(low_level::StackSlotData {
        size: (values.len() * 8) as u32,
        align_shift: 3,
    });
    let address = at.ins().stack_addr(types::I64, slot, 0);
    for (index, &value) in values.iter().enumerate() {
        let value = at.add_unassigned(Op::Convert(value, Repr::Boxed));
        let value = at.as_word(value);
        at.ins()
            .store(MemFlagsData::trusted(), value, address, (index * 8) as i32);
    }
    address
}

pub(in crate::jit::semantic) fn lower(function: &mut FunctionState) {
    function.ensure_cfg();
    let graph = &mut function.graph;
    let mut insts = Vec::new();
    for &block in &function.facts.cfg.rpo {
        insts.clone_from(&graph.blocks[block].insts);
        for &inst in &insts {
            let Some(mut call) = select(inst, &graph.insts[inst].op, &function.facts) else {
                continue;
            };
            if call.helper == Helper::PropertyGetCached
                && function.plan.shapes.megamorphic(graph, inst)
            {
                call.helper = Helper::PropertyGetMegamorphic;
            }
            let contract = graph.contract(&graph.insts[inst].op, &function.facts);
            let result_type = graph.insts[inst]
                .result()
                .map_or(ValueType::default(), |value| function.facts.of(value));
            let mut args = Vec::with_capacity(call.args.len());
            let mut uses = Vec::with_capacity(call.args.len());
            let mut kept = Vec::new();
            let mut at = Cursor::before(graph, inst);
            for arg in call.args {
                let (value, repr) = match arg {
                    Arg::Value(value, repr) => (value, repr),
                    Arg::GlobalCache(name) => (
                        at.add_unassigned(Op::Javascript(JsOp::GlobalCache(name))),
                        Repr::Word,
                    ),
                    Arg::PropertyCache(site, name) => (
                        at.add_unassigned(Op::Javascript(JsOp::PropertyCache(site, name))),
                        Repr::Word,
                    ),
                    Arg::Integer(bits) => (at.ins().iconst(types::I64, bits), Repr::Word),
                    Arg::Boxed(value) => {
                        let word = at.ins().iconst(types::I64, value.bits() as i64);
                        (at.as_boxed(word), Repr::Boxed)
                    }
                    Arg::Vector(values) => {
                        let address = stack_vector(&mut at, &values);
                        kept.extend(values);
                        (address, Repr::Word)
                    }
                };
                args.push(value);
                uses.push(repr);
            }
            graph.set_op(
                inst,
                Op::HelperCall {
                    helper: call.helper,
                    args,
                    uses,
                    produces: call.produces,
                    result_type,
                    effects: contract.effects,
                },
            );
            let mut at = Cursor::after(graph, inst);
            for value in kept.into_iter().rev() {
                at.add_effect(Op::KeepAlive(value));
            }
        }
    }
}
