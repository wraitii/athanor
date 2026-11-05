use cranelift_codegen::ir::{InstBuilder, MemFlagsData, Value, condcodes::IntCC, types};
use cranelift_frontend::FunctionBuilder;

pub(super) fn unbox_int32(b: &mut FunctionBuilder<'_>, value: Value) -> (Value, Value) {
    // Saturation keeps every boxed bit pattern safe; only exact round trips pass.
    let float = b.ins().bitcast(types::F64, MemFlagsData::new(), value);
    let integer = b.ins().fcvt_to_sint_sat(types::I32, float);
    let boxed = box_int32(b, integer);
    let exact = b.ins().icmp(IntCC::Equal, boxed, value);
    (integer, exact)
}

pub(super) fn box_int32(b: &mut FunctionBuilder<'_>, value: Value) -> Value {
    let float = b.ins().fcvt_from_sint(types::F64, value);
    b.ins().bitcast(types::I64, MemFlagsData::new(), float)
}
