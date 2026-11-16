use super::{arg, fail};
use crate::heap::NativeFunction;
use crate::jit::session::JitCtx;
use crate::value::Value;

fn byte_offset(text: &str, index: f64) -> Option<usize> {
    let index = if index.is_nan() || index <= 0.0 {
        0
    } else {
        index.trunc() as usize
    };
    text.char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(text.len()))
        .nth(index)
}

fn next_offset(text: &str, end: usize) -> Option<usize> {
    text.get(end..)?
        .chars()
        .next()
        .map(|character| end + character.len_utf8())
}

fn capture_array(ctx: &mut JitCtx, text: &str, found: regress::Match) -> Value {
    let heap = ctx.global.heap().clone();
    let mut values = Vec::with_capacity(found.captures.len() + 1);
    for range in std::iter::once(Some(found.range)).chain(found.captures) {
        let value = if let Some(range) = range {
            let Some(piece) = text.get(range) else {
                for value in values {
                    heap.release(value);
                }
                return fail(ctx, "regular expression returned an invalid UTF-8 range");
            };
            let Some(value) = heap.string(piece) else {
                for value in values {
                    heap.release(value);
                }
                return fail(ctx, "heap exhausted");
            };
            value
        } else {
            Value::UNDEFINED
        };
        values.push(value);
    }
    let array = heap.array(&values);
    for value in values {
        heap.release(value);
    }
    match array {
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
    let (regexp, input) = if function == NativeFunction::StringMatch {
        (arg(args, 0), receiver)
    } else {
        (receiver, arg(args, 0))
    };
    let heap = ctx.global.heap().clone();
    let Some(regex) = heap.regexp_compiled(regexp) else {
        return fail(
            ctx,
            "regular expression method requires a RegExp receiver or argument",
        );
    };
    let Some(text) = heap.text(input) else {
        return fail(ctx, "regular expression matching requires a string");
    };
    let (global, last_index) = heap.regexp_state(regexp).unwrap();
    if function == NativeFunction::StringMatch && global {
        heap.regexp_set_last_index(regexp, 0.0);
        let mut offset = 0;
        let mut values = Vec::new();
        while let Some(found) = regex.find_from(&text, offset).next() {
            let Some(piece) = text.get(found.range.clone()) else {
                for value in values {
                    heap.release(value);
                }
                return fail(ctx, "regular expression returned an invalid UTF-8 range");
            };
            let Some(value) = heap.string(piece) else {
                for value in values {
                    heap.release(value);
                }
                return fail(ctx, "heap exhausted");
            };
            values.push(value);
            if found.range.is_empty() {
                let Some(next) = next_offset(&text, found.range.end) else {
                    break;
                };
                offset = next;
            } else {
                offset = found.range.end;
            }
        }
        if values.is_empty() {
            return Value::NULL;
        }
        let array = heap.array(&values);
        for value in values {
            heap.release(value);
        }
        return match array {
            Some(value) => value,
            None => fail(ctx, "heap exhausted"),
        };
    }
    let offset = if global {
        byte_offset(&text, last_index)
    } else {
        Some(0)
    };
    let found = offset.and_then(|offset| regex.find_from(&text, offset).next());
    if global {
        let end = match &found {
            Some(found) => match text.get(..found.range.end) {
                Some(prefix) => prefix.chars().count() as f64,
                None => return fail(ctx, "regular expression returned an invalid UTF-8 range"),
            },
            None => 0.0,
        };
        heap.regexp_set_last_index(regexp, end);
    }
    if function == NativeFunction::RegExpTest {
        return Value::boolean(found.is_some());
    }
    match found {
        Some(found) => capture_array(ctx, &text, found),
        None => Value::NULL,
    }
}

pub(super) fn replace(ctx: &mut JitCtx, text: &str, regexp: Value, replacement: Value) -> Value {
    let heap = ctx.global.heap().clone();
    let Some(replacement) = heap.text(replacement) else {
        return fail(ctx, "replace requires a literal string replacement");
    };
    if replacement.contains('$') {
        return fail(ctx, "replace substitution patterns are unsupported");
    }
    let Some(regex) = heap.regexp_compiled(regexp) else {
        return fail(ctx, "replace requires a RegExp");
    };
    let (global, _) = heap.regexp_state(regexp).unwrap();
    if global {
        heap.regexp_set_last_index(regexp, 0.0);
    }
    let mut output = String::new();
    let mut previous = 0;
    let mut offset = 0;
    while let Some(found) = regex.find_from(text, offset).next() {
        let Some(prefix) = text.get(previous..found.range.start) else {
            return fail(ctx, "regular expression returned an invalid UTF-8 range");
        };
        if text.get(found.range.clone()).is_none() {
            return fail(ctx, "regular expression returned an invalid UTF-8 range");
        }
        output.push_str(prefix);
        output.push_str(&replacement);
        previous = found.range.end;
        if !global {
            break;
        }
        if found.range.is_empty() {
            let Some(next) = next_offset(text, found.range.end) else {
                break;
            };
            offset = next;
        } else {
            offset = found.range.end;
        }
    }
    let Some(suffix) = text.get(previous..) else {
        return fail(ctx, "regular expression returned an invalid UTF-8 range");
    };
    output.push_str(suffix);
    match heap.string(&output) {
        Some(value) => value,
        None => fail(ctx, "heap exhausted"),
    }
}
