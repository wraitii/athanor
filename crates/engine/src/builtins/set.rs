use super::{NativeFunction, arg, fail};
use crate::jit::session::JitCtx;
use crate::value::Value;

pub(super) fn construct(ctx: *mut JitCtx, args: &[Value]) -> Value {
    let ctx = unsafe { &mut *ctx };
    if !arg(args, 0).is_nullish() {
        return fail(ctx, "Set iterable construction is unsupported");
    }
    match ctx.global.heap().set_new() {
        Some(value) => value,
        None => fail(ctx, "heap exhausted"),
    }
}

pub(super) fn call(
    ctx: *mut JitCtx,
    function: NativeFunction,
    receiver: Value,
    args: &[Value],
) -> Value {
    let ctx = unsafe { &mut *ctx };
    if function == NativeFunction::Set {
        return fail(ctx, "Set requires new");
    }
    if ctx.global.heap().set_size(receiver).is_none() {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "Set method requires a Set receiver".into(),
        )));
        return Value::UNDEFINED;
    }
    let argument = arg(args, 0);
    match function {
        NativeFunction::SetHas => {
            Value::boolean(ctx.global.heap().set_has(receiver, argument).unwrap())
        }
        NativeFunction::SetDelete => {
            Value::boolean(ctx.global.heap().set_delete(receiver, argument).unwrap())
        }
        NativeFunction::SetAdd => {
            if !ctx.global.heap().set_add(receiver, argument) {
                ctx.error = Some(Box::new(crate::Error::Runtime("invalid Set entry".into())));
                return Value::UNDEFINED;
            }
            ctx.retain(receiver)
        }
        NativeFunction::SetClear => {
            if !ctx.global.heap().set_clear(receiver) {
                ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            }
            Value::UNDEFINED
        }
        _ => unreachable!(),
    }
}
