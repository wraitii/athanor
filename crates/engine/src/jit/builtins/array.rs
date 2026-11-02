use super::super::session::JitCtx;
use super::{arg, fail};
use crate::heap::NativeFunction;
use crate::value::Value;

use super::super::helpers::{js_call, js_truthy};

pub(super) fn call(
    ctx: *mut JitCtx,
    function: NativeFunction,
    receiver: Value,
    args: &[Value],
) -> Value {
    use NativeFunction::*;
    let context = unsafe { &mut *ctx };
    let heap = context.global.heap().clone();
    let arg = |index: usize| arg(args, index);
    if !heap.is_array(receiver) {
        return fail(context, "array method requires an array receiver");
    }
    let length = heap.len(receiver).unwrap();
    if function == ArrayJoin {
        let separator = if arg(0) == Value::UNDEFINED {
            ",".to_owned()
        } else if let Some(text) = heap.text(arg(0)) {
            text
        } else if !arg(0).is_object() {
            arg(0).to_string()
        } else {
            return fail(context, "object conversion in join is unsupported");
        };
        let mut parts = Vec::with_capacity(length);
        for index in 0..length {
            let value = heap.get(receiver, index).unwrap();
            if value == Value::UNDEFINED || value == Value::NULL {
                parts.push(String::new());
            } else if let Some(text) = heap.text(value) {
                parts.push(text);
            } else if value.is_object() {
                return fail(context, "object conversion in join is unsupported");
            } else {
                parts.push(value.to_string());
            }
        }
        return match heap.string(&parts.join(&separator)) {
            Some(value) => context.own(value),
            None => fail(context, "heap exhausted"),
        };
    }
    let callback = arg(0);
    if heap.closure_code(callback).is_none() && heap.native(callback).is_none() {
        return fail(context, "array callback is not callable");
    }
    if args.len() > 1 && arg(1) != Value::UNDEFINED {
        return fail(context, "array callback thisArg is unsupported");
    }
    let initial = if function == ArrayMap {
        vec![Value::UNDEFINED; length]
    } else {
        Vec::new()
    };
    let Some(result) = heap.array(&initial) else {
        return fail(context, "heap exhausted");
    };
    context.own(result);
    if function == ArrayMap {
        for index in 0..length {
            heap.mark_hole(result, index);
        }
    }
    for index in 0..length {
        if !heap.present(receiver, index) {
            continue;
        }
        let value = heap.get(receiver, index).unwrap();
        unsafe { &mut *ctx }.retain_temp(value);
        let callback_args = [value, Value::number(index as f64), receiver];
        let mapped = js_call(ctx, callback, Value::UNDEFINED, callback_args.as_ptr(), 3);
        let context = unsafe { &mut *ctx };
        if context.error.is_some() || context.exception_pending() {
            return Value::UNDEFINED;
        }
        if function == ArrayMap {
            heap.set(result, index, mapped);
        } else if js_truthy(ctx, mapped) == Value::TRUE {
            heap.set(result, heap.len(result).unwrap(), value);
        }
    }
    return result;
}
