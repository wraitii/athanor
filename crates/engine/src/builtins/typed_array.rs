use super::{arg, fail};
use crate::heap::TypedKind;
use crate::jit::session::{Held, JitCtx};
use crate::value::Value;

pub(crate) fn numeric(ctx: &mut JitCtx, value: Value) -> Option<f64> {
    if value.is_object() {
        fail(
            ctx,
            "typed array object-to-number conversion is unsupported",
        );
        return None;
    }
    if let Some(text) = ctx.global.heap().text(value) {
        return Some(crate::jit::conversion::string_number(&text));
    }
    Some(value.to_number())
}

pub(crate) fn from(ctx: *mut JitCtx, kind: TypedKind, args: &[Value]) -> Value {
    let array = super::array::from(ctx, args);
    let context = unsafe { &mut *ctx };
    if context.error.is_some() || context.exception_pending() {
        return Value::UNDEFINED;
    }
    let heap = context.global.heap().clone();
    let length = heap.len(array).unwrap();
    let Some(result) = heap.typed_new(kind, length) else {
        return fail(
            context,
            "typed array allocation failed or exceeds 256 MiB limit",
        );
    };
    let mut allocation = Held::new(&heap);
    allocation.push(result);
    for index in 0..length {
        let value = heap.get(array, index).unwrap();
        let Some(number) = numeric(context, value) else {
            return Value::UNDEFINED;
        };
        if !heap.typed_set(result, index, number) {
            return fail(context, "cannot set typed array element");
        }
    }
    allocation.take();
    result
}

pub(crate) fn construct(ctx: &mut JitCtx, kind: TypedKind, args: &[Value]) -> Value {
    let argument = arg(args, 0);
    let length = if argument.is_undefined() {
        0.0
    } else if let Some(value) = argument.as_number() {
        if value.is_nan() { 0.0 } else { value.trunc() }
    } else {
        return fail(ctx, "typed array constructor requires a numeric length");
    };
    if !length.is_finite() || length < 0.0 || length > usize::MAX as f64 {
        return fail(ctx, "invalid typed array length");
    }
    match ctx.global.heap().typed_new(kind, length as usize) {
        Some(value) => value,
        None => fail(
            ctx,
            "typed array allocation failed or exceeds 256 MiB limit",
        ),
    }
}

fn bound(ctx: &mut JitCtx, value: Value, length: usize, default: usize) -> Option<usize> {
    if value.is_undefined() {
        return Some(default);
    }
    let number = numeric(ctx, value)?;
    let number = if number.is_nan() { 0.0 } else { number.trunc() };
    Some(if number < 0.0 {
        (length as f64 + number).max(0.0) as usize
    } else {
        number.min(length as f64) as usize
    })
}

pub(crate) fn fill(ctx: &mut JitCtx, receiver: Value, args: &[Value]) -> Value {
    let heap = ctx.global.heap().clone();
    if heap.typed_kind(receiver).is_none() {
        return fail(ctx, "typed array fill requires a typed array receiver");
    }
    let Some(number) = numeric(ctx, arg(args, 0)) else {
        return Value::UNDEFINED;
    };
    let length = heap.len(receiver).unwrap();
    let Some(start) = bound(ctx, arg(args, 1), length, 0) else {
        return Value::UNDEFINED;
    };
    let Some(end) = bound(ctx, arg(args, 2), length, length) else {
        return Value::UNDEFINED;
    };
    heap.typed_fill(receiver, start, end, number);
    ctx.retain(receiver)
}

pub(super) fn call(
    ctx: *mut JitCtx,
    function: crate::heap::NativeFunction,
    receiver: Value,
    args: &[Value],
) -> Value {
    use crate::heap::NativeFunction;
    match function {
        NativeFunction::TypedConstructor(_) => {
            fail(unsafe { &mut *ctx }, "typed array constructor requires new")
        }
        NativeFunction::TypedFrom(kind) => from(ctx, kind, args),
        NativeFunction::TypedFill => fill(unsafe { &mut *ctx }, receiver, args),
        _ => unreachable!(),
    }
}
