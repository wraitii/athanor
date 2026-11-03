pub(super) mod array;
pub(super) mod console;
mod date;
mod host;
pub(super) mod json;
pub(super) mod math;
mod number;
pub(super) mod object;
mod regexp;
mod string;
pub(super) mod typed_array;

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
    let function = if heap.native(object) == Some(Array) {
        match name {
            "from" => ArrayFrom,
            "isArray" => ArrayIsArray,
            _ => return None,
        }
    } else if let Some(NativeFunction::TypedConstructor(kind)) = heap.native(object) {
        match name {
            "from" => TypedFrom(kind),
            _ => return None,
        }
    } else if heap.typed_kind(object).is_some() && name == "fill" {
        TypedFill
    } else if object.is_number() {
        match name {
            "toFixed" => NumberToFixed,
            "toString" => NumberToString,
            _ => return None,
        }
    } else if heap.regexp_compiled(object).is_some() {
        match name {
            "test" => RegExpTest,
            "exec" => RegExpExec,
            _ => return None,
        }
    } else if heap.is_array(object) {
        match name {
            "filter" => ArrayFilter,
            "map" => ArrayMap,
            "every" => ArrayEvery,
            "some" => ArraySome,
            "sort" => ArraySort,
            "join" => ArrayJoin,
            "pop" => ArrayPop,
            "indexOf" => ArrayIndexOf,
            "concat" => ArrayConcat,
            "push" => ArrayPush,
            "includes" => ArrayIncludes,
            "reverse" => ArrayReverse,
            "slice" => ArraySlice,
            "flat" => ArrayFlat,
            "flatMap" => ArrayFlatMap,
            "fill" => ArrayFill,
            _ => return None,
        }
    } else if heap.text(object).is_some() {
        match name {
            "match" => StringMatch,
            "slice" => StringSlice,
            "split" => StringSplit,
            "startsWith" => StringStartsWith,
            "replace" => StringReplace,
            "padStart" => StringPadStart,
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
        Print | ConsoleLog => console::call(ctx, function, args),
        DateNow => date::now(),
        ArrayIsArray => Value::boolean(unsafe { &*ctx }.global.heap().is_array(arg(args, 0))),
        TypedFrom(kind) => typed_array::from(unsafe { &mut *ctx }, kind, args),
        TypedFill => typed_array::fill(unsafe { &mut *ctx }, receiver, args),
        ArrayFilter | ArrayMap | ArrayEvery | ArraySome | ArraySort | ArrayJoin | ArrayPop
        | ArrayIndexOf | ArrayConcat | ArrayPush | ArrayIncludes | ArrayReverse | ArraySlice
        | ArrayFlat | ArrayFlatMap | ArrayFill | ArrayFrom => {
            array::call(ctx, function, receiver, args)
        }
        RegExpTest | RegExpExec | StringMatch => regexp::call(ctx, function, receiver, args),
        NumberToFixed | NumberToString => number::call(ctx, function, receiver, args),
        StringSlice | StringSplit | StringStartsWith | StringReplace | StringPadStart => {
            string::call(ctx, function, receiver, args)
        }
        _ => unreachable!(),
    }
}
