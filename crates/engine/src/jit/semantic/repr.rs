use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{MemFlagsData, Type, types};

use super::low_level::{Builder, Op as LowOp, Unary};
use super::{
    Constant, Facts, Graph, Inst, MemoryFormat, MemoryOp, Op, TerminatorKind, Value, ValueDef,
    ValueType, resolve,
};
use crate::FxHashMap;
use crate::value::Value as JsValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit) enum Repr {
    Boxed,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl Repr {
    pub(in crate::jit) fn of(types: ValueType) -> Self {
        if types == ValueType::default() {
            Self::Boxed
        } else if types.within(ValueType::BOOLEAN) {
            Self::I8
        } else if types.within(ValueType::INT32) {
            Self::I32
        } else if types.within(ValueType::INTEGER) {
            Self::I64
        } else if types.within(ValueType::NUMBER) {
            Self::F64
        } else {
            Self::Boxed
        }
    }

    pub(in crate::jit) fn admits(self, types: ValueType) -> bool {
        match self {
            Self::Boxed => true,
            Self::I8 => types.within(ValueType::BOOLEAN),
            Self::I32 => types.within(ValueType::INT32),
            Self::I64 => types.within(ValueType::INTEGER),
            Self::F64 => types.within(ValueType::NUMBER),
            Self::I16 | Self::F32 => false,
        }
    }

    pub(in crate::jit) fn coerce(self, types: ValueType) -> ValueType {
        if self.admits(types) {
            return types;
        }
        match self {
            Self::Boxed => types,
            Self::I8 => ValueType::BOOLEAN,
            Self::I16 | Self::I32 => ValueType::INT32,
            Self::I64 => ValueType::INT64,
            Self::F32 | Self::F64 => ValueType::NUMBER,
        }
    }

    pub(in crate::jit) fn machine(self) -> Type {
        match self {
            Self::Boxed | Self::I64 => types::I64,
            Self::I8 => types::I8,
            Self::I16 => types::I16,
            Self::I32 => types::I32,
            Self::F32 => types::F32,
            Self::F64 => types::F64,
        }
    }

    pub(in crate::jit) fn from_machine(ty: Type) -> Self {
        match ty {
            types::I8 => Self::I8,
            types::I16 => Self::I16,
            types::I32 => Self::I32,
            types::I64 => Self::I64,
            types::F32 => Self::F32,
            types::F64 => Self::F64,
            _ => panic!("unsupported machine type {ty}"),
        }
    }
}

impl MemoryFormat {
    pub(in crate::jit) fn repr(self) -> Repr {
        match self {
            Self::U8 | Self::I8 => Repr::I8,
            Self::U16 | Self::I16 => Repr::I16,
            Self::U32 | Self::I32 => Repr::I32,
            Self::F32 => Repr::F32,
            Self::F64 => Repr::F64,
        }
    }
}

pub(super) fn assign(graph: &mut Graph, facts: &Facts) {
    for value in graph.values.keys() {
        if graph.reprs[value].is_none() {
            graph.reprs[value] = Some(representation(graph, facts, value));
        }
    }
}

fn representation(graph: &Graph, facts: &Facts, value: super::Value) -> Repr {
    match graph.defining_op(value) {
        Some(Op::LowLevel(
            super::low_level::Op::Load { ty, .. }
            | super::low_level::Op::Integer { ty, .. }
            | super::low_level::Op::Unary { ty, .. },
        )) => Repr::from_machine(*ty),
        Some(Op::LowLevel(super::low_level::Op::Binary { op, left, .. }))
            if !op.is_float() && !op.is_integer_arithmetic() =>
        {
            representation(graph, facts, *left)
        }
        Some(Op::LowLevel(super::low_level::Op::Select { yes, no, .. })) => {
            let repr = |value: Value| {
                graph.reprs[value].unwrap_or_else(|| representation(graph, facts, value))
            };
            let (yes, no) = (repr(*yes), repr(*no));
            debug_assert_eq!(yes, no, "select arms disagree");
            yes
        }
        Some(Op::TypedData(_) | Op::Memory(MemoryOp::Address { .. })) => Repr::I64,
        Some(Op::Memory(MemoryOp::IndexedLoad { format, .. } | MemoryOp::Load { format, .. })) => {
            format.repr()
        }
        Some(Op::SaturatingInteger(_)) => Repr::I64,
        _ => Repr::of(facts.of(value)),
    }
}

#[derive(Clone, Copy)]
pub(in crate::jit) enum Step {
    Unbox(Repr),
    Box,
    Convert(Repr),
}

impl Step {
    pub(in crate::jit) fn op(self, value: Value) -> (Op, Repr) {
        match self {
            Self::Unbox(repr) => (Op::Unbox(value, repr), repr),
            Self::Box => (Op::Box(value), Repr::Boxed),
            Self::Convert(repr) => (Op::Convert(value, repr), repr),
        }
    }
}

pub(in crate::jit) fn steps(from: Repr, to: Repr) -> Vec<Step> {
    match (from, to) {
        _ if from == to => Vec::new(),
        (Repr::Boxed, Repr::F64 | Repr::I8) => vec![Step::Unbox(to)],
        (Repr::Boxed, _) => vec![Step::Unbox(Repr::F64), Step::Convert(to)],
        (Repr::F64 | Repr::I8, Repr::Boxed) => vec![Step::Box],
        (_, Repr::Boxed) => vec![Step::Convert(Repr::F64), Step::Box],
        _ => vec![Step::Convert(to)],
    }
}

pub(in crate::jit) fn lower(builder: &mut Builder<'_>) -> Result<(), String> {
    verify_words(builder.graph)?;
    let mut aliases = FxHashMap::default();
    for block in builder.graph.blocks.keys().collect::<Vec<_>>() {
        let mut position = 0;
        while position < builder.graph.blocks[block].insts.len() {
            let inst = builder.graph.blocks[block].insts[position];
            if let Op::Constant(constant) = &builder.graph.insts[inst].op {
                let result = builder.graph.insts[inst].result().unwrap();
                let repr = builder.graph.repr(result);
                let op = constant_op(constant, repr);
                builder.graph.insts[inst].op = Op::LowLevel(op);
                if repr == Repr::Boxed {
                    builder.graph.reprs[result] = Some(Repr::I64);
                }
                position += 1;
                continue;
            }
            if let Op::AsBoxed(word) | Op::AsWord(word) = builder.graph.insts[inst].op {
                let result = builder.graph.insts[inst].result().unwrap();
                aliases.insert(result, resolve(&aliases, word));
                builder.graph.blocks[block].insts.remove(position);
                continue;
            }
            let Some((input, step)) = expansion(builder.graph, inst) else {
                position += 1;
                continue;
            };
            let from = builder.graph.repr(input);
            let input = resolve(&aliases, input);
            let data = &builder.graph.insts[inst];
            let (span, location, source_name) =
                (data.span, data.emission_location, data.source_name);
            let result = data.result().unwrap();
            builder.graph.blocks[block].insts.remove(position);
            builder.insert_at(block, position);
            builder.set_span(span);
            builder.set_location(location);
            builder.set_source_name(source_name);
            aliases.insert(result, apply(builder, input, from, step)?);
            position = builder.cursor().unwrap();
        }
    }
    builder.graph.replace_uses(&aliases);
    words(builder.graph)
}

fn runtime(op: &LowOp) -> bool {
    matches!(
        op,
        LowOp::Reference { .. }
            | LowOp::FrameStore { .. }
            | LowOp::Cleanup { .. }
            | LowOp::Scratch { .. }
            | LowOp::Call { .. }
    )
}

fn verify_words(graph: &Graph) -> Result<(), String> {
    let boxed = |value: Value| graph.reprs[value] == Some(Repr::Boxed);
    for (block, data) in graph.blocks.iter() {
        for &inst in &data.insts {
            let Op::LowLevel(op) = &graph.insts[inst].op else {
                continue;
            };
            if runtime(op) {
                continue;
            }
            if let Some(value) = graph.insts[inst].op.operands_iter().find(|&v| boxed(v)) {
                return Err(format!("{inst} ({op:?}) reads boxed {value} as a word"));
            }
        }
        let Some(terminator) = &data.terminator else {
            continue;
        };
        if let TerminatorKind::Branch(condition, ..) = terminator.kind
            && boxed(condition)
        {
            return Err(format!("{block} branches on boxed {condition}"));
        }
        for target in terminator.kind.targets() {
            let params = &graph.blocks[target.block].params;
            for (&arg, &param) in target.args.iter().zip(params) {
                if graph.reprs[arg] != graph.reprs[param] {
                    return Err(format!(
                        "{block} passes {arg}:{:?} to {param}:{:?}",
                        graph.reprs[arg], graph.reprs[param]
                    ));
                }
            }
        }
    }
    Ok(())
}

fn words(graph: &mut Graph) -> Result<(), String> {
    let mut live = Vec::new();
    for data in graph.blocks.values() {
        live.extend(data.params.iter().copied());
        for &inst in &data.insts {
            live.extend(graph.insts[inst].results.iter().copied());
        }
    }
    for value in live {
        if graph.reprs[value] != Some(Repr::Boxed) {
            continue;
        }
        let word = match graph.values[value] {
            ValueDef::Param(..) => true,
            ValueDef::Result(inst, _) => {
                matches!(graph.insts[inst].op, Op::LowLevel(LowOp::FrameStore { .. }))
            }
        };
        if !word {
            return Err(format!(
                "{value} ({:?}) is still boxed after representation lowering",
                graph.defining_op(value)
            ));
        }
        graph.reprs[value] = Some(Repr::I64);
    }
    Ok(())
}

fn expansion(graph: &Graph, inst: Inst) -> Option<(Value, Step)> {
    let data = &graph.insts[inst];
    match data.op {
        Op::Box(input) => Some((input, Step::Box)),
        Op::Unbox(input, to) => Some((input, Step::Unbox(to))),
        Op::Convert(input, to) => Some((input, Step::Convert(to))),
        Op::NumericFloat(input)
        | Op::SaturatingInteger(input)
        | Op::AsInteger(input)
        | Op::ArrayPosition(input)
        | Op::ToInt32(input)
        | Op::AsNumber(input) => Some((input, Step::Convert(graph.repr(data.result()?)))),
        _ => None,
    }
}

fn apply(builder: &mut Builder<'_>, value: Value, from: Repr, step: Step) -> Result<Value, String> {
    let flags = MemFlagsData::new();
    Ok(match step {
        Step::Unbox(Repr::F64) => builder.ins().bitcast(types::F64, flags, value),
        Step::Unbox(Repr::I8) => {
            builder
                .ins()
                .icmp_imm_s(IntCC::Equal, value, JsValue::TRUE.bits() as i64)
        }
        Step::Unbox(repr) => unreachable!("unbox to {repr:?}"),
        Step::Box => box_value(builder, value),
        Step::Convert(to) => {
            if from != to && (from == Repr::Boxed || to == Repr::Boxed) {
                return Err(format!(
                    "{value}: conversion from {from:?} to {to:?} must box or unbox explicitly"
                ));
            }
            convert(builder, value, from, to)
        }
    })
}

fn box_value(builder: &mut Builder<'_>, value: Value) -> Value {
    let flags = MemFlagsData::new();
    match builder.graph.repr(value) {
        Repr::I8 => {
            let mut ins = builder.ins();
            let yes = ins.iconst(types::I64, JsValue::TRUE.bits() as i64);
            let no = ins.iconst(types::I64, JsValue::FALSE.bits() as i64);
            ins.select(value, yes, no)
        }
        Repr::F64 => match builder.graph.defining_op(value) {
            Some(Op::LowLevel(LowOp::Float64(bits))) => {
                let bits = JsValue::number(f64::from_bits(*bits)).bits();
                builder.ins().iconst(types::I64, bits as i64)
            }
            Some(Op::LowLevel(LowOp::Unary {
                op: Unary::FloatFromSigned | Unary::FloatFromUnsigned,
                ..
            })) => builder.ins().bitcast(types::I64, flags, value),
            _ => {
                let mut ins = builder.ins();
                let unordered = ins.fcmp(FloatCC::Unordered, value, value);
                let bits = ins.bitcast(types::I64, flags, value);
                let nan = ins.iconst(types::I64, JsValue::number(f64::NAN).bits() as i64);
                ins.select(unordered, nan, bits)
            }
        },
        repr => unreachable!("box from {repr:?}"),
    }
}

fn constant_op(constant: &Constant, repr: Repr) -> LowOp {
    let integer = |ty, bits| LowOp::Integer { ty, bits };
    match (constant, repr) {
        (Constant::Number(number), Repr::F64) => LowOp::Float64(number.to_bits()),
        (Constant::Number(number), Repr::I64) => integer(types::I64, *number as i64 as u64),
        (Constant::Number(number), Repr::I32) => integer(types::I32, *number as i32 as i64 as u64),
        (Constant::Boolean(value), Repr::I8) => integer(types::I8, u64::from(*value)),
        (constant, Repr::Boxed) => integer(types::I64, boxed(constant).bits()),
        (constant, repr) => unreachable!("constant {constant:?} as {repr:?}"),
    }
}

fn boxed(constant: &Constant) -> JsValue {
    match constant {
        Constant::Undefined => JsValue::UNDEFINED,
        Constant::Null => JsValue::NULL,
        Constant::Boolean(value) => JsValue::boolean(*value),
        Constant::Number(value) => JsValue::number(*value),
        Constant::Uninitialized => JsValue::UNINITIALIZED,
        Constant::String(_) => unreachable!("string constants are lowered at runtime"),
    }
}

fn fold(builder: &mut Builder<'_>, value: Value, from: Repr, to: Repr) -> Option<Value> {
    let constant = match (builder.graph.defining_op(value)?, from) {
        (Op::LowLevel(LowOp::Integer { bits, .. }), Repr::I32) => Ok(i64::from(*bits as i32)),
        (Op::LowLevel(LowOp::Integer { bits, .. }), Repr::I64) => Ok(*bits as i64),
        (Op::LowLevel(LowOp::Float64(bits)), Repr::F64) => Err(f64::from_bits(*bits)),
        _ => return None,
    };
    let mut ins = builder.ins();
    Some(match (constant, to) {
        (Ok(integer), Repr::I64) => ins.iconst(types::I64, integer),
        (Ok(integer), Repr::I32) => ins.iconst(types::I32, i64::from(integer as i32)),
        (Ok(integer), Repr::F64) => ins.f64const(integer as f64),
        (Err(number), Repr::I64) => ins.iconst(types::I64, number as i64),
        (Err(number), Repr::I32) => {
            ins.iconst(types::I32, i64::from(crate::jit::helpers::to_int32(number)))
        }
        _ => return None,
    })
}

fn convert(builder: &mut Builder<'_>, value: Value, from: Repr, to: Repr) -> Value {
    if from == to {
        return value;
    }
    if let Some(folded) = fold(builder, value, from, to) {
        return folded;
    }
    let mut ins = builder.ins();
    match (from, to) {
        (Repr::I64 | Repr::I32, Repr::F64) => ins.fcvt_from_sint(types::F64, value),
        (Repr::F64, Repr::I32) => to_int32(builder, value),
        (Repr::I8, Repr::I32) => ins.uextend(types::I32, value),
        (Repr::I8, Repr::F64) => {
            let wide = ins.uextend(types::I32, value);
            ins.fcvt_from_uint(types::F64, wide)
        }
        (Repr::I64 | Repr::I32, Repr::I8) => ins.icmp_imm_s(IntCC::NotEqual, value, 0),
        (Repr::F64, Repr::I8) => {
            let zero = ins.f64const(0.0);
            let negative = ins.fcmp(FloatCC::LessThan, value, zero);
            let positive = ins.fcmp(FloatCC::GreaterThan, value, zero);
            ins.bor(negative, positive)
        }
        (Repr::I32, Repr::I64) => ins.sextend(types::I64, value),
        (Repr::F64, Repr::I64) => ins.fcvt_to_sint_sat(types::I64, value),
        (Repr::I64, Repr::I32) => ins.ireduce(types::I32, value),
        _ => unreachable!("conversion from {from:?} to {to:?}"),
    }
}

pub(in crate::jit) fn to_int32(builder: &mut Builder<'_>, float: Value) -> Value {
    let mut ins = builder.ins();
    let scale = ins.f64const(1.0 / 4294967296.0);
    let scaled = ins.fmul(float, scale);
    let wraps = ins.trunc(scaled);
    let modulus = ins.f64const(4294967296.0);
    let wrapped = ins.fmul(wraps, modulus);
    let remainder = ins.fsub(float, wrapped);
    let integer = ins.fcvt_to_sint_sat(types::I64, remainder);
    ins.ireduce(types::I32, integer)
}
