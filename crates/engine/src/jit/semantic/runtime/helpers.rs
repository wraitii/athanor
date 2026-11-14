use cranelift_codegen::ir::{MemFlagsData, types};

use super::super::{
    Cfg, Facts, FunctionState, Graph, Inst, Op, Repr, TypedRead, Value, ValueType, low_level,
};
use crate::jit::helpers::Helper;
use crate::value::Value as JsValue;

enum Arg {
    Value(Value, Repr),
    Integer(i64),
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

fn select(op: &Op, facts: &Facts) -> Option<Call> {
    match op {
        Op::GetIndex(object, index)
        | Op::ArrayGet(object, index)
        | Op::TypedIndex(object, index, TypedRead { element: None, .. }) => {
            let (helper, index) = key(facts, *index);
            call(helper, vec![Arg::Value(*object, Repr::Boxed), index], true)
        }
        Op::SetIndex(object, index, value) => {
            call(Helper::IndexSet, boxed(&[*object, *index, *value]), false)
        }
        Op::ClosureThis(value) => call(Helper::ClosureThis, boxed(&[*value]), true),
        Op::IsArrayIteration(callee, receiver, method) => {
            let mut args = boxed(&[*callee, *receiver]);
            args.push(Arg::Integer(*method as i64));
            call(Helper::ArrayIterationGuard, args, true)
        }
        Op::MapElement(receiver, index) => call(
            Helper::ArrayMapElement,
            vec![
                Arg::Value(*receiver, Repr::Boxed),
                Arg::Value(*index, Repr::I64),
            ],
            true,
        ),
        Op::Throw(value) => call(Helper::Throw, boxed(&[*value]), false),
        Op::PropertyKey(value) => call(Helper::PropertyKey, boxed(&[*value]), true),
        Op::MarkClass(value) => call(Helper::MarkClass, boxed(&[*value]), false),
        Op::DisableConstructor(value) => call(Helper::DisableConstructor, boxed(&[*value]), false),
        Op::IsTypeof(value, kind) => {
            let mut args = boxed(&[*value]);
            args.push(Arg::Boxed(JsValue::number(*kind as f64)));
            call(Helper::IsTypeof, args, true)
        }
        Op::IsNative(value, native) => {
            let mut args = boxed(&[*value]);
            args.push(Arg::Integer(native.index() as i64));
            call(Helper::SemanticMathNative, args, true)
        }
        Op::IsStringSplit(value) => call(Helper::IsStringSplit, boxed(&[*value]), true),
        Op::ProbeIndex(object, key) => {
            call(Helper::SemanticIndexProbe, boxed(&[*object, *key]), true)
        }
        Op::ConstructorPrototype(callee) => {
            call(Helper::ConstructorPrototype, boxed(&[*callee]), true)
        }
        Op::IteratorNew(value) => call(Helper::IterateNew, boxed(&[*value]), true),
        Op::IteratorNext(value) => call(Helper::IterateNext, boxed(&[*value]), true),
        Op::EnumerateNew(value) => call(Helper::EnumerateNew, boxed(&[*value]), true),
        Op::EnumerateNext(value) => call(Helper::EnumerateNext, boxed(&[*value]), true),
        Op::ArrayAppend(array, value, spread) => call(
            if *spread {
                Helper::ArrayExtend
            } else {
                Helper::ArrayAppend
            },
            boxed(&[*array, *value]),
            false,
        ),
        Op::PackAppend(pack, value, spread) => call(
            if *spread {
                Helper::PackExtend
            } else {
                Helper::PackAppend
            },
            boxed(&[*pack, *value]),
            false,
        ),
        Op::PackRelease(pack) => call(Helper::PackRelease, boxed(&[*pack]), false),
        Op::CallPack(callee, receiver, args) => {
            call(Helper::CallPack, boxed(&[*callee, *receiver, *args]), true)
        }
        Op::ConstructPack(callee, args) => {
            call(Helper::ConstructPack, boxed(&[*callee, *args]), true)
        }
        Op::ConstructorCallPack(callee, receiver, args) => call(
            Helper::ConstructorCallPack,
            boxed(&[*callee, *receiver, *args]),
            true,
        ),
        Op::ConstructPackResolved(callee, prototype, args) => call(
            Helper::ConstructPackResolved,
            boxed(&[*callee, *prototype, *args]),
            true,
        ),
        Op::Construct { callee, args } => {
            let mut call_args = boxed(&[*callee]);
            call_args.extend(vector(args));
            call(Helper::Construct, call_args, true)
        }
        Op::ConstructResolved {
            callee,
            prototype,
            args,
        } => {
            let mut call_args = boxed(&[*callee, *prototype]);
            call_args.extend(vector(args));
            call(Helper::ConstructResolved, call_args, true)
        }
        Op::ConstructorCall {
            callee,
            receiver,
            args,
        } => {
            let mut call_args = boxed(&[*callee, *receiver]);
            call_args.extend(vector(args));
            call(Helper::ConstructorCall, call_args, true)
        }
        Op::StringSplit(receiver, args) => {
            let mut call_args = vec![Arg::Integer(0), Arg::Value(*receiver, Repr::Boxed)];
            call_args.extend(vector(args));
            call(Helper::StringSplit, call_args, true)
        }
        Op::NewArray(args) => call(Helper::ArrayNew, vector(args).into(), true),
        Op::NewHoleyArray(length) => call(Helper::ArrayMapNew, boxed(&[*length]), true),
        Op::NewInstance(prototype) => call(Helper::NewInstance, boxed(&[*prototype]), true),
        Op::NewObject => call(
            Helper::ObjectNew,
            vec![Arg::Boxed(JsValue::UNDEFINED)],
            true,
        ),
        Op::PackTake(pack, index) => call(Helper::PackTake, boxed(&[*pack, *index]), true),
        _ => None,
    }
}

pub(super) fn insert_before(
    graph: &mut Graph,
    inst: Inst,
    op: low_level::Op,
    repr: Option<Repr>,
) -> (Inst, Option<Value>) {
    let data = &graph.insts[inst];
    let (block, span) = (data.block, data.span);
    let at = position(graph, inst);
    let (inserted, value) = graph.insert(block, at, span, Op::LowLevel(op), repr.is_some());
    if let (Some(value), Some(repr)) = (value, repr) {
        graph.reprs[value] = Some(repr);
    }
    (inserted, value)
}

fn position(graph: &Graph, inst: Inst) -> usize {
    graph.blocks[graph.insts[inst].block]
        .insts
        .iter()
        .position(|&i| i == inst)
        .unwrap()
}

pub(super) fn integer(bits: u64) -> low_level::Op {
    low_level::Op::Integer {
        ty: types::I64,
        bits,
    }
}

fn constant(graph: &mut Graph, inst: Inst, bits: u64, repr: Repr) -> Value {
    insert_before(graph, inst, integer(bits), Some(repr))
        .1
        .unwrap()
}

pub(super) fn as_boxed(graph: &mut Graph, inst: Inst, word: Value) -> Value {
    let data = &graph.insts[inst];
    let (block, span, source_name) = (data.block, data.span, data.source_name);
    let at = position(graph, inst);
    let (inserted, value) = graph.insert(block, at, span, Op::AsBoxed(word), true);
    graph.insts[inserted].source_name = source_name;
    let value = value.unwrap();
    graph.reprs[value] = Some(Repr::Boxed);
    value
}

pub(super) fn word(graph: &mut Graph, inst: Inst, value: Value) -> Value {
    let data = &graph.insts[inst];
    let (block, span, source_name) = (data.block, data.span, data.source_name);
    let at = position(graph, inst);
    let (inserted, word) = graph.insert(block, at, span, Op::AsWord(value), true);
    graph.insts[inserted].source_name = source_name;
    let word = word.unwrap();
    graph.reprs[word] = Some(Repr::I64);
    word
}

fn stack_vector(graph: &mut Graph, inst: Inst, values: &[Value]) -> Value {
    if values.is_empty() {
        return constant(graph, inst, 0, Repr::I64);
    }
    let slot = low_level::StackSlot(graph.low_level_slots.len());
    graph.low_level_slots.push(low_level::StackSlotData {
        size: (values.len() * 8) as u32,
        align_shift: 3,
    });
    let address = low_level::Op::StackAddress {
        slot,
        offset: 0,
        ty: types::I64,
    };
    let address = insert_before(graph, inst, address, Some(Repr::I64))
        .1
        .unwrap();
    for (index, &value) in values.iter().enumerate() {
        let (convert, _) = graph.insert(
            graph.insts[inst].block,
            position(graph, inst),
            graph.insts[inst].span,
            Op::Convert(value, Repr::Boxed),
            true,
        );
        let value = graph.insts[convert].result().unwrap();
        let value = word(graph, inst, value);
        let store = low_level::Op::Store {
            value,
            address,
            offset: (index * 8) as i32,
            flags: MemFlagsData::trusted(),
        };
        insert_before(graph, inst, store, None);
    }
    address
}

pub(in crate::jit::semantic) fn lower(function: &mut FunctionState) {
    let graph = &mut function.graph;
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Some(call) = select(&graph.insts[inst].op, &function.facts) else {
                continue;
            };
            let contract = graph.contract(&graph.insts[inst].op, &function.facts);
            let mut args = Vec::with_capacity(call.args.len());
            let mut uses = Vec::with_capacity(call.args.len());
            let mut kept = Vec::new();
            for arg in call.args {
                let (value, repr) = match arg {
                    Arg::Value(value, repr) => (value, repr),
                    Arg::Integer(bits) => {
                        (constant(graph, inst, bits as u64, Repr::I64), Repr::I64)
                    }
                    Arg::Boxed(value) => {
                        let word = constant(graph, inst, value.bits(), Repr::I64);
                        (as_boxed(graph, inst, word), Repr::Boxed)
                    }
                    Arg::Vector(values) => {
                        let address = stack_vector(graph, inst, &values);
                        kept.extend(values);
                        (address, Repr::I64)
                    }
                };
                args.push(value);
                uses.push(repr);
            }
            graph.insts[inst].op = Op::HelperCall {
                helper: call.helper,
                args,
                uses,
                produces: call.produces,
                contract,
            };
            let at = position(graph, inst);
            let span = graph.insts[inst].span;
            for value in kept {
                graph.insert(block, at + 1, span, Op::KeepAlive(value), false);
            }
        }
    }
}
