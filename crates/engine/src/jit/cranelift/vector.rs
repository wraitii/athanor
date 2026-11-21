use cranelift_codegen::ir::{
    Endianness, InstBuilder, MemFlagsData, Type, Value as ClifValue, types,
};
use cranelift_frontend::FunctionBuilder;

use crate::jit::semantic::low_level::{Binary, Vector};
use crate::jit::semantic::{MemoryFormat, Value};

pub(super) fn emit(
    builder: &mut FunctionBuilder<'_>,
    vector: &Vector,
    get: impl Fn(Value) -> ClifValue,
) -> Vec<ClifValue> {
    let trusted = MemFlagsData::trusted();
    let lanes = MemFlagsData::new().with_endianness(Endianness::Little);
    match *vector {
        Vector::Load {
            ty,
            base,
            index,
            format,
        } => {
            let address = address(builder, get(base), get(index), format);
            let loaded = match format {
                MemoryFormat::U8 | MemoryFormat::I8 => {
                    let word = builder.ins().load(types::I32, trusted, address, 0);
                    let bytes = builder.ins().scalar_to_vector(types::I32X4, word);
                    let bytes = builder.ins().bitcast(types::I8X16, lanes, bytes);
                    let halves = widen(builder, format, bytes);
                    widen(builder, format, halves)
                }
                MemoryFormat::I16 => builder.ins().sload16x4(trusted, address, 0),
                MemoryFormat::U16 => builder.ins().uload16x4(trusted, address, 0),
                _ => builder.ins().load(ty, trusted, address, 0),
            };
            vec![loaded]
        }
        Vector::Store {
            base,
            index,
            value,
            format,
        } => {
            let address = address(builder, get(base), get(index), format);
            let value = get(value);
            let stored = match format.bytes() {
                1 => {
                    let halves = narrow(builder, value, 0xff);
                    let bytes = builder.ins().unarrow(halves, halves);
                    let words = builder.ins().bitcast(types::I32X4, lanes, bytes);
                    builder.ins().extractlane(words, 0)
                }
                2 => {
                    let halves = narrow(builder, value, 0xffff);
                    let words = builder.ins().bitcast(types::I64X2, lanes, halves);
                    builder.ins().extractlane(words, 0)
                }
                _ => value,
            };
            builder.ins().store(trusted, stored, address, 0);
            Vec::new()
        }
        Vector::Splat { ty, value } => vec![builder.ins().splat(ty, get(value))],
        Vector::Binary { op, left, right } => {
            let (left, right) = (get(left), get(right));
            vec![match op {
                Binary::Add => builder.ins().iadd(left, right),
                Binary::Subtract => builder.ins().isub(left, right),
                Binary::Multiply => builder.ins().imul(left, right),
                Binary::And => builder.ins().band(left, right),
                Binary::Or => builder.ins().bor(left, right),
                Binary::Xor => builder.ins().bxor(left, right),
                Binary::ShiftLeft => builder.ins().ishl(left, right),
                Binary::ShiftRightSigned => builder.ins().sshr(left, right),
                Binary::ShiftRightUnsigned => builder.ins().ushr(left, right),
                _ => unreachable!("unsupported vector operation {op:?}"),
            }]
        }
    }
}

fn address(
    builder: &mut FunctionBuilder<'_>,
    base: ClifValue,
    index: ClifValue,
    format: MemoryFormat,
) -> ClifValue {
    let offset = builder.ins().imul_imm_s(index, format.bytes() as i64);
    builder.ins().iadd(base, offset)
}

fn widen(builder: &mut FunctionBuilder<'_>, format: MemoryFormat, value: ClifValue) -> ClifValue {
    match format {
        MemoryFormat::I8 => builder.ins().swiden_low(value),
        _ => builder.ins().uwiden_low(value),
    }
}

/// Truncates each 32-bit lane to its low bits, packed into the low lanes of a narrower vector.
fn narrow(builder: &mut FunctionBuilder<'_>, value: ClifValue, mask: i64) -> ClifValue {
    let ty: Type = builder.func.dfg.value_type(value);
    let mask = builder.ins().iconst(types::I32, mask);
    let mask = builder.ins().splat(ty, mask);
    let masked = builder.ins().band(value, mask);
    builder.ins().unarrow(masked, masked)
}
