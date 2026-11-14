use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types;

use super::super::low_level::{Builder, Op as LowOp};
use super::super::{Constant, Op, Repr, Value};

pub(in crate::jit::semantic) fn required(from: Repr, to: Repr) -> bool {
    matches!(
        (from.widest(), to.widest()),
        (
            Repr::F64,
            Repr::I32 | Repr::I64 | Repr::Word | Repr::Boolean
        ) | (Repr::Integer(_) | Repr::Word, Repr::Boolean)
    )
}

pub(in crate::jit::semantic) fn convert(
    builder: &mut Builder<'_>,
    value: Value,
    from: Repr,
    to: Repr,
) -> Value {
    let (from, to) = (from.widest(), to.widest());
    if from == to {
        return value;
    }
    let number = match builder.graph.defining_op(value) {
        Some(Op::Constant(Constant::Number(number))) => Some(*number),
        Some(Op::LowLevel(LowOp::Float64(bits))) => Some(f64::from_bits(*bits)),
        _ => None,
    };
    if from == Repr::F64
        && let Some(number) = number
    {
        let mut ins = builder.ins();
        match to {
            Repr::I32 => {
                return ins.iconst(types::I32, i64::from(crate::jit::helpers::to_int32(number)));
            }
            Repr::I64 | Repr::Word => return ins.iconst(types::I64, number as i64),
            _ => {}
        }
    }
    let mut ins = builder.ins();
    match (from, to) {
        (Repr::F64, Repr::I32) => to_int32(builder, value),
        (Repr::F64, Repr::I64 | Repr::Word) => ins.fcvt_to_sint_sat(types::I64, value),
        (Repr::Integer(_) | Repr::Word, Repr::Boolean) => ins.icmp_imm_s(IntCC::NotEqual, value, 0),
        (Repr::F64, Repr::Boolean) => {
            let zero = ins.f64const(0.0);
            let negative = ins.fcmp(FloatCC::LessThan, value, zero);
            let positive = ins.fcmp(FloatCC::GreaterThan, value, zero);
            ins.boolean_or(negative, positive)
        }
        _ => builder.semantic(Op::Convert(value, to), to),
    }
}

fn to_int32(builder: &mut Builder<'_>, float: Value) -> Value {
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
