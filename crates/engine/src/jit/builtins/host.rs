use super::super::session::JitCtx;
use super::{arg, fail};
use crate::heap::NativeFunction;
use crate::value::Value;

fn text(context: &mut JitCtx, value: Value) -> Option<String> {
    let heap = context.global.heap();
    if let Some(text) = heap.text(value).or_else(|| heap.error_text(value)) {
        return Some(text);
    }
    if value.is_object() {
        fail(
            context,
            "object conversion in host arguments is unsupported",
        );
        return None;
    }
    Some(value.to_string())
}

fn host_error(context: &mut JitCtx, message: &str) -> Value {
    let heap = context.global.heap().clone();
    let Some(message) = heap.string(message) else {
        return fail(context, "heap exhausted");
    };
    let Some(name) = heap.string("Error") else {
        heap.release(message);
        return fail(context, "heap exhausted");
    };
    let error = heap.error(message, name);
    heap.release(message);
    heap.release(name);
    let Some(error) = error else {
        return fail(context, "heap exhausted");
    };
    context.throw(error);
    heap.release(error);
    Value::UNDEFINED
}

pub(super) fn call(ctx: *mut JitCtx, function: NativeFunction, args: &[Value]) -> Value {
    let context = unsafe { &mut *ctx };
    let required = if function == NativeFunction::HostEvaluateSource {
        2
    } else {
        1
    };
    if args.len() < required {
        return host_error(context, "missing host function argument");
    }
    let Some(first) = text(context, arg(args, 0)) else {
        return Value::UNDEFINED;
    };
    if function == NativeFunction::HostEvaluateSource {
        let Some(filename) = text(context, arg(args, 1)) else {
            return Value::UNDEFINED;
        };
        return context.evaluate(&first, &filename);
    }
    let result = match function {
        NativeFunction::HostReadFile => context.engine.host.as_ref().unwrap().read_file(&first),
        NativeFunction::HostListFiles => {
            let filter = if args.len() < 2 {
                "*".into()
            } else {
                let Some(filter) = text(context, arg(args, 1)) else {
                    return Value::UNDEFINED;
                };
                filter
            };
            let recursive = super::super::helpers::js_truthy(ctx, arg(args, 2)) == Value::TRUE;
            let context = unsafe { &mut *ctx };
            context
                .engine
                .host
                .as_ref()
                .unwrap()
                .list_files(&first, &filter, recursive)
        }
        _ => unreachable!(),
    };
    let context = unsafe { &mut *ctx };
    match result {
        Ok(text) => match context.global.heap().string(&text) {
            Some(value) => context.own(value),
            None => fail(context, "heap exhausted"),
        },
        Err(error) => host_error(context, &error),
    }
}
