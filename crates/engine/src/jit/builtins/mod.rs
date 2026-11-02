mod array;
mod host;
mod string;

use super::session::JitCtx;
use crate::heap::NativeFunction;
use crate::value::Value;

fn arg(args: &[Value], index: usize) -> Value {
    args.get(index).copied().unwrap_or(Value::UNDEFINED)
}
fn fail(context: &mut JitCtx, message: &str) -> Value {
    context.error = Some(crate::Error::Runtime(message.into()));
    Value::UNDEFINED
}

pub(super) fn property_get(ctx: &mut JitCtx, object: Value, name: &str) -> Option<Value> {
    use crate::heap::NativeFunction::*;
    let heap = ctx.global.heap();
    let function = if heap.is_array(object) {
        match name {
            "filter" => ArrayFilter,
            "map" => ArrayMap,
            "join" => ArrayJoin,
            _ => return None,
        }
    } else if heap.text(object).is_some() {
        match name {
            "slice" => StringSlice,
            "split" => StringSplit,
            "startsWith" => StringStartsWith,
            "replace" => StringReplace,
            _ => return None,
        }
    } else {
        return None;
    };
    Some(match heap.native_function(function) {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
            Value::UNDEFINED
        }
    })
}

pub(super) fn call(
    ctx: *mut JitCtx,
    function: NativeFunction,
    receiver: Value,
    args: &[Value],
) -> Value {
    use NativeFunction::*;
    match function {
        HostReadFile | HostListFiles | HostEvaluateSource => host::call(ctx, function, args),
        ArrayFilter | ArrayMap | ArrayJoin => array::call(ctx, function, receiver, args),
        StringSlice | StringSplit | StringStartsWith | StringReplace => {
            string::call(ctx, function, receiver, args)
        }
        _ => unreachable!(),
    }
}
