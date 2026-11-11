use super::super::session::JitCtx;
use super::{arg, fail};
use crate::heap::NativeFunction;
use crate::value::Value;

/// Returns an owned reference without registering a temporary.
pub(in crate::jit) fn split(ctx: *mut JitCtx, receiver: Value, args: &[Value]) -> Value {
    split_result(ctx, receiver, args, Value::UNINITIALIZED)
}

pub(in crate::jit) fn split_result(
    ctx: *mut JitCtx,
    receiver: Value,
    args: &[Value],
    destination: Value,
) -> Value {
    let heap = unsafe { &*ctx }.global.heap().clone();
    let arg = |index: usize| arg(args, index);
    if heap.string_is_empty(receiver).is_none() {
        return fail(
            unsafe { &mut *ctx },
            "string method requires a string receiver",
        );
    }
    if args.len() > 1 && arg(1) != Value::UNDEFINED {
        return fail(unsafe { &mut *ctx }, "split limit is unsupported");
    }
    let separator = if arg(0) == Value::UNDEFINED {
        None
    } else if heap.string_is_empty(arg(0)).is_some() {
        Some(arg(0))
    } else {
        return fail(unsafe { &mut *ctx }, "split separator must be a string");
    };
    let mut values = unsafe { super::super::pack::buffer(ctx) };
    if heap
        .split_string_into(receiver, separator, &mut values.values)
        .is_none()
    {
        return fail(unsafe { &mut *ctx }, "heap exhausted");
    }
    let values = std::mem::take(&mut values.values);
    match if destination == Value::UNINITIALIZED {
        heap.array_from_owned(values)
    } else {
        unsafe { super::super::pack::store(ctx, destination, values) }
    } {
        Some(value) => value,
        None => fail(unsafe { &mut *ctx }, "heap exhausted"),
    }
}

pub(in crate::jit) fn call(
    ctx: *mut JitCtx,
    function: NativeFunction,
    receiver: Value,
    args: &[Value],
) -> Value {
    use NativeFunction::*;
    if function == StringSplit {
        let value = split(ctx, receiver, args);
        return value;
    }
    let context = unsafe { &mut *ctx };
    let heap = context.global.heap().clone();
    let arg = |index: usize| arg(args, index);
    let Some(text) = heap.text(receiver) else {
        return fail(context, "string method requires a string receiver");
    };
    let string_arg = |value: Value| heap.text(value);
    let result = match function {
        StringSlice => {
            let chars: Vec<char> = text.chars().collect();
            let convert = |value: Value, default: usize| -> Option<usize> {
                if value == Value::UNDEFINED {
                    return Some(default);
                }
                let n = value.as_number()?;
                let n = if n.is_nan() { 0.0 } else { n.trunc() };
                Some(if n < 0.0 {
                    (chars.len() as f64 + n).max(0.0) as usize
                } else {
                    (n as usize).min(chars.len())
                })
            };
            let (Some(start), Some(end)) = (convert(arg(0), 0), convert(arg(1), chars.len()))
            else {
                return fail(context, "slice indices must be numbers");
            };
            chars[start..end.max(start)].iter().collect::<String>()
        }
        StringStartsWith => {
            let Some(search) = string_arg(arg(0)) else {
                return fail(context, "startsWith requires a string argument");
            };
            let position = if arg(1) == Value::UNDEFINED {
                0
            } else if let Some(n) = arg(1).as_number() {
                n.max(0.0) as usize
            } else {
                return fail(context, "startsWith position must be a number");
            };
            return Value::boolean(
                text.chars()
                    .skip(position)
                    .collect::<String>()
                    .starts_with(&search),
            );
        }
        StringReplace => {
            if heap.regexp_state(arg(0)).is_some() {
                return super::regexp::replace(context, &text, arg(0), arg(1));
            }
            let (Some(search), Some(replacement)) = (string_arg(arg(0)), string_arg(arg(1))) else {
                return fail(context, "replace requires literal string arguments");
            };
            if replacement.contains('$') {
                return fail(context, "replace substitution patterns are unsupported");
            }
            text.replacen(&search, &replacement, 1)
        }
        StringPadStart => {
            let target = if arg(0).is_undefined() {
                0
            } else if let Some(number) = arg(0).as_number() {
                if number.is_nan() || number <= 0.0 {
                    0
                } else if !number.is_finite() || number > usize::MAX as f64 {
                    return fail(context, "invalid padStart length");
                } else {
                    number.floor() as usize
                }
            } else {
                return fail(context, "padStart length must be a number or undefined");
            };
            let length = text.chars().count();
            if target <= length {
                return match heap.string(&text) {
                    Some(value) => value,
                    None => fail(context, "heap exhausted"),
                };
            }
            let padding = if arg(1).is_undefined() {
                " ".to_owned()
            } else if let Some(padding) = string_arg(arg(1)) {
                padding
            } else {
                return fail(context, "padStart padding must be a string or undefined");
            };
            if padding.is_empty() {
                return match heap.string(&text) {
                    Some(value) => value,
                    None => fail(context, "heap exhausted"),
                };
            }
            let padding = padding
                .chars()
                .cycle()
                .take(target - length)
                .collect::<String>();
            format!("{padding}{text}")
        }
        _ => unreachable!(),
    };
    match heap.string(&result) {
        Some(value) => value,
        None => fail(context, "heap exhausted"),
    }
}
