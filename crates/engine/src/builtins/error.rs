use super::{NativeFunction, arg};
use crate::jit::session::JitCtx;
use crate::value::Value;

pub(super) fn call(
    ctx: *mut JitCtx,
    _function: NativeFunction,
    _receiver: Value,
    args: &[Value],
) -> Value {
    construct(ctx, args)
}

pub(super) fn construct(ctx: *mut JitCtx, args: &[Value]) -> Value {
    let ctx = unsafe { &mut *ctx };
    let argument = arg(args, 0);
    let heap = ctx.global.heap();
    let text = if argument.is_undefined() {
        String::new()
    } else if argument.is_object() {
        let Some(text) = heap.error_text(argument) else {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "object-to-primitive conversion is unsupported".into(),
            )));
            return Value::UNDEFINED;
        };
        text
    } else {
        heap.text(argument).unwrap_or_else(|| argument.to_string())
    };
    let Some(message) = heap.string(&text) else {
        ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
        return Value::UNDEFINED;
    };
    let Some(name) = heap.string("Error") else {
        heap.release(message);
        ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
        return Value::UNDEFINED;
    };
    let result = heap.error(message, name);
    heap.release(message);
    heap.release(name);
    match result {
        Some(value) => value,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}
