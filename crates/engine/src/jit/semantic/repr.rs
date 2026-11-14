use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{MemFlagsData, Type, types};

use super::low_level::{Builder, Op as LowOp, Unary};
use super::{
    Constant, Facts, Graph, Inst, MemoryFormat, MemoryOp, Op, TerminatorKind, TypedRead, Value,
    ValueDef, ValueType, resolve,
};
use crate::FxHashMap;
use crate::value::Value as JsValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::jit) enum IntegerWidth {
    Any,
    W8,
    W16,
    W32,
    W64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::jit) enum FloatWidth {
    Any,
    W32,
    W64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::jit) enum Repr {
    Boxed,
    Word,
    Boolean,
    Integer(IntegerWidth),
    Float(FloatWidth),
}

impl std::fmt::Debug for Repr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match *self {
            Self::Boxed => "Boxed",
            Self::Word => "Word",
            Self::Boolean => "Boolean",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::Integer(IntegerWidth::Any) => "Integer",
            Self::Float(FloatWidth::Any) => "Float",
        })
    }
}

impl Repr {
    pub(in crate::jit) const I8: Self = Self::Integer(IntegerWidth::W8);
    pub(in crate::jit) const I16: Self = Self::Integer(IntegerWidth::W16);
    pub(in crate::jit) const I32: Self = Self::Integer(IntegerWidth::W32);
    pub(in crate::jit) const I64: Self = Self::Integer(IntegerWidth::W64);
    pub(in crate::jit) const F32: Self = Self::Float(FloatWidth::W32);
    pub(in crate::jit) const F64: Self = Self::Float(FloatWidth::W64);
    pub(in crate::jit) const INTEGER: Self = Self::Integer(IntegerWidth::Any);
    pub(in crate::jit) const FLOAT: Self = Self::Float(FloatWidth::Any);

    pub(in crate::jit) fn of(types: ValueType) -> Self {
        if types == ValueType::default() {
            Self::Boxed
        } else if types.within(ValueType::BOOLEAN) {
            Self::Boolean
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
            Self::Boolean => types.within(ValueType::BOOLEAN),
            Self::I32 => types.within(ValueType::INT32),
            Self::I64 => types.within(ValueType::INTEGER),
            Self::F64 => types.within(ValueType::NUMBER),
            Self::Word | Self::Integer(_) | Self::Float(_) => false,
        }
    }

    pub(in crate::jit) fn coerce(self, types: ValueType) -> ValueType {
        if self.admits(types) {
            return types;
        }
        match self {
            Self::Boxed => types,
            Self::Boolean => ValueType::BOOLEAN,
            Self::Word | Self::I64 | Self::INTEGER => ValueType::INT64,
            Self::Integer(_) => ValueType::INT32,
            Self::Float(_) => ValueType::NUMBER,
        }
    }

    pub(in crate::jit) fn machine(self) -> Type {
        match self {
            Self::Boxed | Self::Word => types::I64,
            Self::Boolean => types::I8,
            Self::Integer(IntegerWidth::W8) => types::I8,
            Self::Integer(IntegerWidth::W16) => types::I16,
            Self::Integer(IntegerWidth::W32) => types::I32,
            Self::Integer(IntegerWidth::W64) => types::I64,
            Self::Float(FloatWidth::W32) => types::F32,
            Self::Float(FloatWidth::W64) => types::F64,
            Self::Integer(IntegerWidth::Any) | Self::Float(FloatWidth::Any) => {
                panic!("{self:?} has no machine type before its width is chosen")
            }
        }
    }

    pub(in crate::jit) fn numeric(ty: Type) -> Self {
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

    /// Transitional: low-level ops still carry machine types, so their values get the
    /// family those types overwhelmingly mean there.
    pub(in crate::jit) fn from_machine(ty: Type) -> Self {
        match ty {
            types::I64 => Self::Word,
            _ => Self::numeric(ty),
        }
    }

    pub(in crate::jit) fn fits(self, ty: Type) -> bool {
        match self {
            Self::INTEGER => ty.is_int(),
            Self::FLOAT => ty.is_float(),
            repr => repr.machine() == ty,
        }
    }

    pub(in crate::jit) fn compatible(self, other: Self) -> bool {
        match (self, other) {
            (Self::INTEGER | Self::FLOAT, Self::INTEGER | Self::FLOAT) => self == other,
            (Self::INTEGER | Self::FLOAT, concrete) => self.fits(concrete.machine()),
            (concrete, open) => open.fits(concrete.machine()),
        }
    }

    pub(in crate::jit) fn accepts(self, required: Self) -> bool {
        self == required
            || matches!(
                (self, required),
                (Self::Word, Self::I64)
                    | (Self::I64, Self::Word)
                    | (Self::INTEGER, Self::Integer(_) | Self::Word)
                    | (Self::Integer(_) | Self::Word, Self::INTEGER)
                    | (Self::FLOAT, Self::Float(_))
                    | (Self::Float(_), Self::FLOAT)
            )
    }

    pub(in crate::jit) fn is_open(self) -> bool {
        matches!(self, Self::INTEGER | Self::FLOAT)
    }

    pub(in crate::jit) fn widest(self) -> Self {
        match self {
            Self::INTEGER => Self::I64,
            Self::FLOAT => Self::F64,
            repr => repr,
        }
    }

    /// Transitional: low-level 64-bit integers are still typed as words until low-level ops
    /// stop carrying machine types.
    fn as_integer(self) -> Self {
        match self {
            Self::Word => Self::I64,
            repr => repr,
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
            let repr = representation(graph, facts, value);
            if std::env::var_os("JS_JIT_REPR_ASSIGN").is_some() {
                let kind = match graph.defining_op(value) {
                    None => "param".to_owned(),
                    Some(op) => {
                        let text = format!("{op:?}");
                        let end = text
                            .match_indices(|c: char| !c.is_alphanumeric())
                            .nth(if matches!(op, Op::LowLevel(_)) { 1 } else { 0 })
                            .map_or(text.len(), |(i, _)| i);
                        text[..end].to_owned()
                    }
                };
                eprintln!("repr-assign {kind} {repr:?}");
            }
            graph.reprs[value] = Some(repr);
        }
    }
}

pub(in crate::jit) fn intrinsic(op: &Op) -> Option<Repr> {
    Some(match op {
        Op::LowLevel(
            LowOp::Load { ty, .. }
            | LowOp::AtomicLoad { ty, .. }
            | LowOp::Integer { ty, .. }
            | LowOp::Unary {
                op: Unary::Bitcast,
                ty,
                ..
            }
            | LowOp::StackAddress { ty, .. }
            | LowOp::StackPointer { ty },
        ) => Repr::from_machine(*ty),
        Op::LowLevel(LowOp::Unary { ty, .. }) => Repr::numeric(*ty),
        Op::LowLevel(LowOp::Float64(_)) => Repr::F64,
        Op::LowLevel(
            LowOp::IntCompare { .. }
            | LowOp::FloatCompare { .. }
            | LowOp::InRange { .. }
            | LowOp::TestMask { .. },
        ) => Repr::Boolean,
        Op::LowLevel(LowOp::Binary { op, .. }) if op.is_float() => Repr::F64,
        Op::LowLevel(LowOp::Binary { op, .. }) if op.is_boolean() => Repr::Boolean,
        Op::LowLevel(LowOp::Binary { .. }) => Repr::I64,
        Op::TypedData(_) | Op::Memory(MemoryOp::Address { .. }) => Repr::Word,
        Op::Memory(MemoryOp::IndexedLoad { format, .. } | MemoryOp::Load { format, .. }) => {
            format.repr()
        }
        Op::LowLevel(_) => return None,
        Op::PropertyCache(..) | Op::AsWord(_) => Repr::Word,
        Op::MemoryNumber(_, format) => match format {
            MemoryFormat::U32 | MemoryFormat::F32 | MemoryFormat::F64 => Repr::F64,
            _ => Repr::I32,
        },
        Op::Convert(_, repr) | Op::Unbox(_, repr) => *repr,
        Op::NumericFloat(_) => Repr::F64,
        Op::AsNumber(_) => return None,
        Op::SaturatingInteger(_) | Op::AsInteger(_) | Op::ArrayPosition(_) => Repr::I64,
        Op::ToInt32(_) | Op::ArrayKind(_) => Repr::I32,
        Op::HelperCall { produces, .. } => produces.unwrap_or(Repr::Boxed),
        Op::IntegerCheck(..)
        | Op::IsInteger(_)
        | Op::IsCode(..)
        | Op::IsShape(..)
        | Op::IsTyped(..)
        | Op::IsPropertyCached(..)
        | Op::IsIndexed(_)
        | Op::IsString(_)
        | Op::Truthy(_)
        | Op::IsNumber(_)
        | Op::IsUninitialized(_)
        | Op::IsUndefined(_)
        | Op::IsNullish(_)
        | Op::CanOverwrite(..)
        | Op::IsArray(..) => Repr::Boolean,
        Op::TypedIndex(
            _,
            _,
            TypedRead {
                element: Some(kind),
                bounded: true,
            },
        ) => element(*kind),
        _ => Repr::Boxed,
    })
}

pub(in crate::jit) fn fits(constant: &Constant, repr: Repr) -> bool {
    match (constant, repr) {
        (Constant::Number(_), Repr::F64 | Repr::FLOAT) | (Constant::Boolean(_), Repr::Boolean) => {
            true
        }
        (&Constant::Number(number), Repr::I32) => {
            number == f64::from(number as i32) && !(number == 0.0 && number.is_sign_negative())
        }
        (&Constant::Number(number), Repr::I64 | Repr::INTEGER) => {
            number == number as i64 as f64 && !(number == 0.0 && number.is_sign_negative())
        }
        _ => false,
    }
}

pub(super) fn element(kind: crate::heap::TypedKind) -> Repr {
    use crate::heap::TypedKind as K;
    match kind {
        K::Float64 | K::Float32 | K::Uint32 => Repr::F64,
        _ => Repr::I32,
    }
}

fn representation(graph: &Graph, facts: &Facts, value: super::Value) -> Repr {
    let op = graph.defining_op(value);
    if let Some(repr) = op.and_then(intrinsic) {
        return repr;
    }
    if let Some(&Op::AsNumber(input)) = op {
        return graph.reprs[input].unwrap_or_else(|| representation(graph, facts, input));
    }
    match Repr::of(facts.of(value)) {
        Repr::Integer(_) => Repr::INTEGER,
        Repr::Float(_) => Repr::FLOAT,
        repr => repr,
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
        (Repr::Boxed, Repr::F64 | Repr::Boolean) => vec![Step::Unbox(to)],
        (Repr::Boxed, _) => vec![Step::Unbox(Repr::F64), Step::Convert(to)],
        (Repr::F64 | Repr::Boolean, Repr::Boxed) => vec![Step::Box],
        (_, Repr::Boxed) => vec![Step::Convert(Repr::F64), Step::Box],
        _ => vec![Step::Convert(to)],
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Words,
    Machine,
}

pub(in crate::jit) fn unbox(builder: &mut Builder<'_>) -> Result<(), String> {
    verify_words(builder.graph)?;
    rewrite(builder, Stage::Words)?;
    words(builder.graph)
}

pub(in crate::jit) fn lower(builder: &mut Builder<'_>) -> Result<(), String> {
    if let Some(value) = builder
        .graph
        .values
        .keys()
        .find(|&value| builder.graph.reprs[value] == Some(Repr::Boxed))
    {
        return Err(format!(
            "{value} ({:?}) is still boxed after runtime lowering",
            builder.graph.defining_op(value)
        ));
    }
    rewrite(builder, Stage::Machine)
}

fn rewrite(builder: &mut Builder<'_>, stage: Stage) -> Result<(), String> {
    let mut aliases = FxHashMap::default();
    for block in builder.graph.blocks.keys().collect::<Vec<_>>() {
        let mut position = 0;
        while position < builder.graph.blocks[block].insts.len() {
            let inst = builder.graph.blocks[block].insts[position];
            if let Op::Constant(constant) = &builder.graph.insts[inst].op {
                let result = builder.graph.insts[inst].result().unwrap();
                let repr = builder.graph.repr(result);
                if (stage == Stage::Words) == (repr == Repr::Boxed) {
                    builder.graph.insts[inst].op = Op::LowLevel(constant_op(constant, repr));
                    if repr == Repr::Boxed {
                        builder.graph.reprs[result] = Some(Repr::Word);
                    }
                }
                position += 1;
                continue;
            }
            if let Op::AsBoxed(word) | Op::AsWord(word) | Op::AsNumber(word) =
                builder.graph.insts[inst].op
            {
                let result = builder.graph.insts[inst].result().unwrap();
                aliases.insert(result, resolve(&aliases, word));
                builder.graph.blocks[block].insts.remove(position);
                continue;
            }
            let Some((input, step)) = expansion(builder.graph, inst, stage) else {
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
    Ok(())
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
                if (graph.repr(arg) == Repr::Boxed) != (graph.repr(param) == Repr::Boxed) {
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
    let reprs = std::mem::take(&mut graph.reprs);
    for &value in &live {
        graph.reprs[value] = reprs[value];
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
        graph.reprs[value] = Some(Repr::Word);
    }
    Ok(())
}

fn expansion(graph: &Graph, inst: Inst, stage: Stage) -> Option<(Value, Step)> {
    let data = &graph.insts[inst];
    match (stage, &data.op) {
        (Stage::Words, &Op::Box(input)) => Some((input, Step::Box)),
        (Stage::Words, &Op::Unbox(input, to)) => Some((input, Step::Unbox(to))),
        (Stage::Words, _) => None,
        (Stage::Machine, &Op::Convert(input, to)) => Some((input, Step::Convert(to))),
        (
            Stage::Machine,
            &(Op::NumericFloat(input)
            | Op::SaturatingInteger(input)
            | Op::AsInteger(input)
            | Op::ArrayPosition(input)
            | Op::ToInt32(input)),
        ) => Some((input, Step::Convert(graph.repr(data.result()?)))),
        _ => None,
    }
}

fn apply(builder: &mut Builder<'_>, value: Value, from: Repr, step: Step) -> Result<Value, String> {
    let flags = MemFlagsData::new();
    Ok(match step {
        Step::Unbox(Repr::F64) => builder.ins().bitcast(types::F64, flags, value),
        Step::Unbox(Repr::Boolean) => {
            builder
                .ins()
                .icmp_imm_s(IntCC::Equal, value, JsValue::TRUE.bits() as i64)
        }
        Step::Unbox(repr) => {
            let float = builder.ins().bitcast(types::F64, flags, value);
            builder.semantic(Op::Convert(float, repr), repr)
        }
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
    let value = match builder.graph.repr(value) {
        Repr::Word | Repr::Integer(_) | Repr::FLOAT | Repr::F32 => {
            builder.semantic(Op::Convert(value, Repr::F64), Repr::F64)
        }
        _ => value,
    };
    match builder.graph.repr(value) {
        Repr::Boolean => {
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
            Some(Op::Constant(Constant::Number(number))) => {
                let bits = JsValue::number(*number).bits();
                builder.ins().iconst(types::I64, bits as i64)
            }
            _ if from_integer(builder.graph, value) => {
                builder.ins().bitcast(types::I64, flags, value)
            }
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

fn from_integer(graph: &Graph, value: Value) -> bool {
    let input = match graph.defining_op(value) {
        Some(Op::LowLevel(LowOp::Unary {
            op: Unary::FloatFromSigned | Unary::FloatFromUnsigned,
            ..
        })) => return true,
        Some(Op::Convert(input, _) | Op::NumericFloat(input)) => *input,
        _ => return false,
    };
    matches!(graph.repr(input), Repr::Integer(_) | Repr::Boolean)
}

fn constant_op(constant: &Constant, repr: Repr) -> LowOp {
    let integer = |ty, bits| LowOp::Integer { ty, bits };
    match (constant, repr) {
        (Constant::Number(number), Repr::F64) => LowOp::Float64(number.to_bits()),
        (Constant::Number(number), Repr::I64) => integer(types::I64, *number as i64 as u64),
        (Constant::Number(number), Repr::I32) => integer(types::I32, *number as i32 as i64 as u64),
        (Constant::Boolean(value), Repr::Boolean) => integer(types::I8, u64::from(*value)),
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
    let (from, to) = (from.as_integer(), to.as_integer());
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
        (Repr::Boolean, Repr::I32) => ins.uextend(types::I32, value),
        (Repr::Boolean, Repr::F64) => {
            let wide = ins.uextend(types::I32, value);
            ins.fcvt_from_uint(types::F64, wide)
        }
        (Repr::Integer(_), Repr::Boolean) => ins.icmp_imm_s(IntCC::NotEqual, value, 0),
        (Repr::F64, Repr::Boolean) => {
            let zero = ins.f64const(0.0);
            let negative = ins.fcmp(FloatCC::LessThan, value, zero);
            let positive = ins.fcmp(FloatCC::GreaterThan, value, zero);
            ins.boolean_or(negative, positive)
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
