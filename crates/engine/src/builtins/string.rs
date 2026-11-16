use super::{arg, fail};
use crate::heap::NativeFunction;
use crate::jit::session::JitCtx;
use crate::value::Value;

/// Returns an owned reference without registering a temporary.
pub(crate) fn split(ctx: *mut JitCtx, receiver: Value, args: &[Value]) -> Value {
    split_result(ctx, receiver, args, Value::UNINITIALIZED)
}

pub(crate) fn split_result(
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
    let mut values = unsafe { crate::jit::pack::buffer(ctx) };
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
        unsafe { crate::jit::pack::store(ctx, destination, values) }
    } {
        Some(value) => value,
        None => fail(unsafe { &mut *ctx }, "heap exhausted"),
    }
}

pub(crate) fn call(
    ctx: *mut JitCtx,
    function: NativeFunction,
    receiver: Value,
    args: &[Value],
) -> Value {
    use NativeFunction::*;
    if function == StringSplit {
        return split(ctx, receiver, args);
    }
    let context = unsafe { &mut *ctx };
    let heap = context.global.heap().clone();
    let arg = |index: usize| arg(args, index);
    let Some(text) = heap.text(receiver) else {
        return fail(context, "string method requires a string receiver");
    };
    let string_arg = |value: Value| heap.text(value);
    let result = match function {
        StringCharAt | StringCharCodeAt => {
            let value = arg(0);
            if value.is_object() {
                return fail(context, "character index object coercion is unsupported");
            }
            let index = crate::jit::helpers::primitive_number(ctx, value);
            let index = if index.is_nan() { 0.0 } else { index.trunc() };
            let character = if index < 0.0 || !index.is_finite() {
                None
            } else {
                text.chars().nth(index as usize)
            };
            if function == StringCharCodeAt {
                return Value::number(character.map_or(f64::NAN, |c| c as u32 as f64));
            }
            character.map(|c| c.to_string()).unwrap_or_default()
        }
        StringSubstring | StringSubstr => {
            let length = text.chars().count();
            let integer = |value: Value, default: f64| -> Option<f64> {
                if value.is_undefined() {
                    return Some(default);
                }
                if value.is_object() {
                    return None;
                }
                let n = crate::jit::helpers::primitive_number(ctx, value);
                Some(if n.is_nan() { 0.0 } else { n.trunc() })
            };
            let (Some(start), Some(end)) = (integer(arg(0), 0.0), integer(arg(1), length as f64))
            else {
                return fail(context, "string range object coercion is unsupported");
            };
            let (start, end) = if function == StringSubstring {
                let a = start.max(0.0).min(length as f64) as usize;
                let b = end.max(0.0).min(length as f64) as usize;
                (a.min(b), a.max(b))
            } else {
                let start = if start < 0.0 {
                    (length as f64 + start).max(0.0)
                } else {
                    start.min(length as f64)
                } as usize;
                let count = end.max(0.0).min((length - start) as f64) as usize;
                (start, start + count)
            };
            text.chars().skip(start).take(end - start).collect()
        }
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

pub(super) fn primitive_text(ctx: &mut JitCtx, value: Value) -> Option<String> {
    let heap = ctx.global.heap();
    if let Some(text) = heap.text(value).or_else(|| heap.error_text(value)) {
        return Some(text);
    }
    if value.is_object() {
        fail(ctx, "object-to-primitive conversion is unsupported");
        return None;
    }
    Some(value.to_string())
}

pub(super) fn convert(
    ctx: *mut JitCtx,
    _function: NativeFunction,
    _receiver: Value,
    args: &[Value],
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let value = arg(args, 0);
    if value.is_string() {
        return ctx.retain(value);
    }
    let text = if args.is_empty() {
        String::new()
    } else {
        let Some(text) = primitive_text(ctx, value) else {
            return Value::UNDEFINED;
        };
        text
    };
    match ctx.global.heap().string(&text) {
        Some(value) => value,
        None => fail(ctx, "heap exhausted"),
    }
}

pub(super) fn from_char_code(
    ctx: *mut JitCtx,
    _function: NativeFunction,
    _receiver: Value,
    args: &[Value],
) -> Value {
    let mut units = Vec::with_capacity(args.len());
    for &value in args {
        if value.is_object() {
            return fail(
                unsafe { &mut *ctx },
                "character code object coercion is unsupported",
            );
        }
        let n = crate::jit::helpers::primitive_number(ctx, value);
        units.push(crate::jit::helpers::to_int32(n) as u16);
    }
    let ctx = unsafe { &mut *ctx };
    let Ok(text) = String::from_utf16(&units) else {
        return fail(ctx, "unpaired surrogate character codes are unsupported");
    };
    match ctx.global.heap().string(&text) {
        Some(value) => value,
        None => fail(ctx, "heap exhausted"),
    }
}
