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

fn match_ranges(pattern: &crate::regex::Pattern, text: &str, global: bool) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut offset = 0;
    while let Some(found) = pattern.find(text, offset) {
        let empty = found.is_empty();
        let end = found.end;
        ranges.push(found);
        if !global {
            break;
        }
        offset = if empty {
            let Some(next) = next_offset(text, end) else {
                break;
            };
            next
        } else {
            end
        };
    }
    ranges
}

fn pieces(
    ctx: &mut JitCtx,
    input: Value,
    ranges: impl IntoIterator<Item = Option<std::ops::Range<usize>>>,
) -> Value {
    let heap = ctx.global.heap().clone();
    let mut values = Vec::new();
    for range in ranges {
        let value = match range {
            Some(range) => match heap.substring(input, range) {
                Some(value) => value,
                None => {
                    heap.release_all(&values);
                    return fail(ctx, "heap exhausted");
                }
            },
            None => Value::UNDEFINED,
        };
        values.push(value);
    }
    match heap.array_from_owned(values) {
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
    let (global, last_index) = heap.regexp_state(regexp).unwrap();
    if function == NativeFunction::StringMatch && global {
        heap.regexp_set_last_index(regexp, 0.0);
        let Some(ranges) = heap.with_text(input, |text| match_ranges(&regex, text, true)) else {
            return fail(ctx, "regular expression matching requires a string");
        };
        if ranges.is_empty() {
            return Value::NULL;
        }
        return pieces(ctx, input, ranges.into_iter().map(Some));
    }
    let Some((found, end)) = heap.with_text(input, |text| {
        let offset = if global {
            byte_offset(text, last_index)
        } else {
            Some(0)
        };
        let found = offset.and_then(|offset| {
            if function == NativeFunction::RegExpTest {
                return regex.find(text, offset).map(|range| crate::regex::Captures {
                    range,
                    groups: Vec::new(),
                });
            }
            regex.captures(text, offset)
        });
        let end = match &found {
            Some(found) if global => text[..found.range.end].chars().count() as f64,
            _ => 0.0,
        };
        (found, end)
    }) else {
        return fail(ctx, "regular expression matching requires a string");
    };
    if global {
        heap.regexp_set_last_index(regexp, end);
    }
    if function == NativeFunction::RegExpTest {
        return Value::boolean(found.is_some());
    }
    match found {
        Some(found) => pieces(
            ctx,
            input,
            std::iter::once(Some(found.range)).chain(found.groups),
        ),
        None => Value::NULL,
    }
}

pub(super) fn replace(ctx: &mut JitCtx, input: Value, regexp: Value, replacement: Value) -> Value {
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
    let Some(output) = heap.with_text(input, |text| {
        let ranges = match_ranges(&regex, text, global);
        if ranges.is_empty() {
            return None;
        }
        let mut output = String::with_capacity(text.len());
        let mut previous = 0;
        for range in ranges {
            output.push_str(&text[previous..range.start]);
            output.push_str(&replacement);
            previous = range.end;
        }
        output.push_str(&text[previous..]);
        Some(output)
    }) else {
        return fail(ctx, "string method requires a string receiver");
    };
    let Some(output) = output else {
        heap.retain(input);
        return input;
    };
    match heap.string(&output) {
        Some(value) => value,
        None => fail(ctx, "heap exhausted"),
    }
}
