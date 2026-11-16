use std::cmp::Ordering;

use super::{arg, fail};
use crate::heap::NativeFunction;
use crate::jit::session::{Held, JitCtx};
use crate::value::Value;

use crate::jit::helpers::{js_call, js_iterate_new, js_iterate_next, js_truthy, strict_equals};

fn relative_index(value: Value, length: usize, default: usize) -> Option<usize> {
    if value.is_undefined() {
        return Some(default);
    }
    let number = value.as_number()?;
    let number = if number.is_nan() { 0.0 } else { number.trunc() };
    Some(if number < 0.0 {
        (length as f64 + number).max(0.0) as usize
    } else {
        number.min(length as f64) as usize
    })
}

fn to_length(ctx: &mut JitCtx, value: Value) -> Option<usize> {
    if value.is_undefined() {
        return Some(0);
    }
    let heap = ctx.global.heap().clone();
    let number = if let Some(number) = value.as_number() {
        number
    } else if let Some(text) = heap.text(value) {
        crate::jit::conversion::string_number(&text)
    } else if value.is_object() {
        fail(ctx, "Array.from length must be a number");
        return None;
    } else {
        value.to_number()
    };
    if number.is_nan() || number <= 0.0 {
        return Some(0);
    }
    let number = number.floor();
    if !number.is_finite() || number > usize::MAX as f64 {
        fail(ctx, "invalid Array.from length");
        return None;
    }
    Some(number as usize)
}

fn array_like_length(ctx: &mut JitCtx, value: Value) -> Option<usize> {
    let heap = ctx.global.heap().clone();
    if let Some(length) = heap.len(value) {
        return Some(length);
    }
    if value == Value::GLOBAL_OBJECT {
        let length = ctx
            .global
            .get_object_raw("length")
            .unwrap_or(Value::UNDEFINED);
        return to_length(ctx, length);
    }
    if value.is_object() {
        let Some(length) = heap.property(value, "length") else {
            fail(ctx, "Array.from source is not iterable or array-like");
            return None;
        };
        return to_length(ctx, length);
    }
    Some(0)
}

fn array_like_get(ctx: &mut JitCtx, value: Value, index: usize) -> Option<Value> {
    let heap = ctx.global.heap().clone();
    if heap.is_array(value) || heap.typed_kind(value).is_some() {
        return heap.get(value, index);
    }
    let name = index.to_string();
    if value == Value::GLOBAL_OBJECT {
        return Some(ctx.global.get_object_raw(&name).unwrap_or(Value::UNDEFINED));
    }
    if value.is_object() {
        return heap.property(value, &name);
    }
    Some(Value::UNDEFINED)
}

pub(crate) fn from(ctx: *mut JitCtx, args: &[Value]) -> Value {
    let context = unsafe { &mut *ctx };
    let heap = context.global.heap().clone();
    let source = arg(args, 0);
    let mapper = arg(args, 1);
    let this_arg = arg(args, 2);
    if source.is_nullish() {
        return fail(context, "Array.from requires a non-null source");
    }
    if !mapper.is_undefined()
        && heap.closure_code(mapper).is_none()
        && heap.native(mapper).is_none()
    {
        return fail(context, "Array.from mapper is not callable");
    }
    let mut values = Held::new(&heap);
    let mut index = 0;
    if heap.len(source).is_some() {
        let mut iteration = Held::new(&heap);
        let state = iteration.push(js_iterate_new(ctx, source));
        if context.error.is_some() || context.exception_pending() {
            return Value::UNDEFINED;
        }
        loop {
            let value = js_iterate_next(ctx, state);
            if context.error.is_some() || context.exception_pending() {
                return Value::UNDEFINED;
            }
            if value == Value::UNINITIALIZED {
                break;
            }
            let mapped = if mapper.is_undefined() {
                value
            } else {
                let callback_args = [value, Value::number(index as f64)];
                let mapped = js_call(ctx, mapper, this_arg, callback_args.as_ptr(), 2);
                heap.release(value);
                if context.error.is_some() || context.exception_pending() {
                    return Value::UNDEFINED;
                }
                mapped
            };
            values.push(mapped);
            index += 1;
        }
    } else {
        let Some(length) = array_like_length(context, source) else {
            return Value::UNDEFINED;
        };
        if !values.reserve(length) {
            return fail(context, "Array.from allocation failed");
        }
        for index in 0..length {
            let Some(value) = array_like_get(context, source, index) else {
                return Value::UNDEFINED;
            };
            let value = context.retain(value);
            let mapped = if mapper.is_undefined() {
                value
            } else {
                let callback_args = [value, Value::number(index as f64)];
                let mapped = js_call(ctx, mapper, this_arg, callback_args.as_ptr(), 2);
                heap.release(value);
                if context.error.is_some() || context.exception_pending() {
                    return Value::UNDEFINED;
                }
                mapped
            };
            values.push(mapped);
        }
    }
    match heap.array_from_owned(values.take()) {
        Some(value) => value,
        None => fail(context, "heap exhausted"),
    }
}

pub(crate) fn construct(ctx: *mut JitCtx, args: &[Value]) -> Value {
    let context = unsafe { &mut *ctx };
    let heap = context.global.heap().clone();
    if args.len() == 1
        && let Some(number) = args[0].as_number()
    {
        if !number.is_finite() || number < 0.0 || number.fract() != 0.0 || number > u32::MAX as f64
        {
            return fail(context, "invalid array length");
        }
        return match heap.array_of_length(number as usize) {
            Some(value) => value,
            None => fail(context, "array allocation failed"),
        };
    }
    match heap.array(args) {
        Some(value) => value,
        None => fail(context, "heap exhausted"),
    }
}

fn flatten_into(
    heap: &crate::heap::Heap,
    result: Value,
    source: Value,
    depth: usize,
    next: &mut usize,
) -> bool {
    let length = heap.len(source).unwrap();
    for index in 0..length {
        if !heap.present(source, index) {
            continue;
        }
        let value = heap.get(source, index).unwrap();
        if depth > 0 && heap.is_array(value) {
            if !flatten_into(heap, result, value, depth - 1, next) {
                return false;
            }
        } else {
            if !heap.set(result, *next, value) {
                return false;
            }
            *next += 1;
        }
    }
    true
}

fn sort_number(heap: &crate::heap::Heap, value: Value) -> Option<f64> {
    if let Some(number) = value.as_number() {
        return Some(number);
    }
    if let Some(text) = heap.text(value) {
        return Some(crate::jit::conversion::string_number(&text));
    }
    if value.is_object() {
        return None;
    }
    Some(value.to_number())
}

fn sort_text(heap: &crate::heap::Heap, value: Value) -> Option<String> {
    if let Some(number) = value.as_number() {
        return Some(crate::value::number_text(number));
    }
    if let Some(text) = heap.text(value) {
        return Some(text);
    }
    if value.is_object() {
        return None;
    }
    Some(value.to_string())
}

fn sort(ctx: *mut JitCtx, receiver: Value, args: &[Value]) -> Value {
    let context = unsafe { &mut *ctx };
    let heap = context.global.heap().clone();
    let length = heap.len(receiver).unwrap();
    let comparator = arg(args, 0);
    let has_comparator = !comparator.is_undefined();
    if has_comparator
        && heap.closure_code(comparator).is_none()
        && heap.native(comparator).is_none()
    {
        return fail(context, "sort comparator is not callable");
    }
    let mut values = Vec::new();
    let mut kept = Held::new(&heap);
    let mut undefined = 0;
    for index in 0..length {
        if !heap.present(receiver, index) {
            continue;
        }
        let value = heap.get(receiver, index).unwrap();
        if value.is_undefined() {
            undefined += 1;
            continue;
        }
        values.push(kept.push(context.retain(value)));
    }
    let mut failed = false;
    if has_comparator {
        values.sort_by(|left, right| {
            if failed {
                return Ordering::Equal;
            }
            let callback_args = [*left, *right];
            let result = js_call(ctx, comparator, Value::UNDEFINED, callback_args.as_ptr(), 2);
            let context = unsafe { &mut *ctx };
            if context.error.is_some() || context.exception_pending() {
                failed = true;
                return Ordering::Equal;
            }
            let number = sort_number(&heap, result);
            heap.release(result);
            let Some(number) = number else {
                context.error = Some(Box::new(crate::Error::Runtime(
                    "object conversion in sort is unsupported".into(),
                )));
                failed = true;
                return Ordering::Equal;
            };
            if number.is_nan() || number == 0.0 {
                Ordering::Equal
            } else if number < 0.0 {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
    } else {
        values.sort_by(|left, right| {
            if failed {
                return Ordering::Equal;
            }
            let Some(left) = sort_text(&heap, *left) else {
                failed = true;
                return Ordering::Equal;
            };
            let Some(right) = sort_text(&heap, *right) else {
                failed = true;
                return Ordering::Equal;
            };
            left.cmp(&right)
        });
    }
    if failed {
        return fail(context, "object conversion in sort is unsupported");
    }
    for (index, value) in values.iter().enumerate() {
        if !heap.set(receiver, index, *value) {
            return fail(context, "cannot write sorted array element");
        }
    }
    let undefined_start = values.len();
    let holes_start = undefined_start + undefined;
    for index in undefined_start..holes_start {
        if !heap.set(receiver, index, Value::UNDEFINED) {
            return fail(context, "cannot write sorted array element");
        }
    }
    for index in holes_start..length {
        if !heap.set(receiver, index, Value::UNDEFINED) {
            return fail(context, "cannot write sorted array element");
        }
        heap.mark_hole(receiver, index);
    }
    context.retain(receiver)
}

fn flat(ctx: *mut JitCtx, receiver: Value, args: &[Value]) -> Value {
    let context = unsafe { &mut *ctx };
    let heap = context.global.heap().clone();
    let depth = if arg(args, 0).is_undefined() {
        1
    } else if let Some(number) = arg(args, 0).as_number() {
        if number.is_nan() || number <= 0.0 {
            0
        } else if number == f64::INFINITY || number > usize::MAX as f64 {
            usize::MAX
        } else {
            number.floor() as usize
        }
    } else {
        return fail(context, "flat depth must be a number or undefined");
    };
    let Some(result) = heap.array(&[]) else {
        return fail(context, "heap exhausted");
    };
    let mut allocation = Held::new(&heap);
    allocation.push(result);
    let mut next = 0;
    if !flatten_into(&heap, result, receiver, depth, &mut next) {
        return fail(context, "cannot append flat element");
    }
    allocation.take();
    result
}

pub(super) fn call(
    ctx: *mut JitCtx,
    function: NativeFunction,
    receiver: Value,
    args: &[Value],
) -> Value {
    use NativeFunction::*;
    if function == Array {
        return construct(ctx, args);
    }
    if function == ArrayIsArray {
        return Value::boolean(unsafe { &*ctx }.global.heap().is_array(arg(args, 0)));
    }
    if function == ArrayFrom {
        return from(ctx, args);
    }
    let context = unsafe { &mut *ctx };
    let heap = context.global.heap().clone();
    let arg = |index: usize| arg(args, index);
    if !heap.is_array(receiver) {
        return fail(context, "array method requires an array receiver");
    }
    let length = heap.len(receiver).unwrap();
    if function == ArrayPush {
        for (offset, value) in args.iter().copied().enumerate() {
            if !heap.set(receiver, length + offset, value) {
                return fail(context, "cannot append array element");
            }
        }
        return Value::number((length + args.len()) as f64);
    }
    if function == ArrayReverse {
        heap.reverse_array(receiver);
        return context.retain(receiver);
    }
    if function == ArrayFlat {
        return flat(ctx, receiver, args);
    }
    if function == ArraySort {
        return sort(ctx, receiver, args);
    }
    if function == ArrayFill {
        let Some(start) = relative_index(arg(1), length, 0) else {
            return fail(context, "fill bounds must be numbers or undefined");
        };
        let Some(end) = relative_index(arg(2), length, length) else {
            return fail(context, "fill bounds must be numbers or undefined");
        };
        for index in start..end {
            if !heap.set(receiver, index, arg(0)) {
                return fail(context, "cannot fill array element");
            }
        }
        return context.retain(receiver);
    }
    if function == ArrayIncludes {
        if length == 0 {
            return Value::FALSE;
        }
        let Some(start) = relative_index(arg(1), length, 0) else {
            return fail(context, "includes fromIndex must be a number or undefined");
        };
        for index in start..length {
            let value = if heap.present(receiver, index) {
                heap.get(receiver, index).unwrap()
            } else {
                Value::UNDEFINED
            };
            let both_nan = value.as_number().is_some_and(f64::is_nan)
                && arg(0).as_number().is_some_and(f64::is_nan);
            if both_nan || strict_equals(&heap, value, arg(0)) {
                return Value::TRUE;
            }
        }
        return Value::FALSE;
    }
    if function == ArraySlice {
        let Some(start) = relative_index(arg(0), length, 0) else {
            return fail(context, "slice bounds must be numbers or undefined");
        };
        let Some(end) = relative_index(arg(1), length, length) else {
            return fail(context, "slice bounds must be numbers or undefined");
        };
        let Some(result) = heap.array(&[]) else {
            return fail(context, "heap exhausted");
        };
        let mut allocation = Held::new(&heap);
        allocation.push(result);
        for (next, index) in (start..end).enumerate() {
            let present = heap.present(receiver, index);
            let value = heap.get(receiver, index).unwrap();
            if !heap.set(result, next, value) {
                return fail(context, "cannot append slice element");
            }
            if !present {
                heap.mark_hole(result, next);
            }
        }
        allocation.take();
        return result;
    }
    if function == ArrayConcat {
        let Some(result) = heap.array(&[]) else {
            return fail(context, "heap exhausted");
        };
        let mut allocation = Held::new(&heap);
        allocation.push(result);
        let mut next = 0;
        for value in std::iter::once(receiver).chain(args.iter().copied()) {
            if let Some(length) = heap.is_array(value).then(|| heap.len(value).unwrap()) {
                for index in 0..length {
                    let present = heap.present(value, index);
                    let element = heap.get(value, index).unwrap();
                    if !heap.set(result, next, element) {
                        return fail(context, "cannot append concat element");
                    }
                    if !present {
                        heap.mark_hole(result, next);
                    }
                    next += 1;
                }
            } else {
                if !heap.set(result, next, value) {
                    return fail(context, "cannot append concat element");
                }
                next += 1;
            }
        }
        allocation.take();
        return result;
    }
    if function == ArrayIndexOf {
        if length == 0 {
            return Value::number(-1.0);
        }
        let from = if arg(1).is_undefined() {
            0.0
        } else if let Some(value) = arg(1).as_number() {
            if value.is_nan() { 0.0 } else { value.trunc() }
        } else {
            return fail(context, "indexOf fromIndex must be a number or undefined");
        };
        if from >= length as f64 {
            return Value::number(-1.0);
        }
        let start = if from < 0.0 {
            (length as f64 + from).max(0.0) as usize
        } else {
            from as usize
        };
        for index in start..length {
            if heap.present(receiver, index)
                && strict_equals(&heap, heap.get(receiver, index).unwrap(), arg(0))
            {
                return Value::number(index as f64);
            }
        }
        return Value::number(-1.0);
    }
    if function == ArrayPop {
        if length == 0 {
            return Value::UNDEFINED;
        }
        let value = context.retain(heap.get(receiver, length - 1).unwrap());
        heap.truncate(receiver, length - 1);
        return value;
    }
    if function == ArrayAt {
        let index = if arg(0).is_undefined() {
            0.0
        } else if let Some(number) = arg(0).as_number() {
            if number.is_nan() { 0.0 } else { number.trunc() }
        } else {
            return fail(context, "at index must be a number or undefined");
        };
        if !index.is_finite() {
            return Value::UNDEFINED;
        }
        let index = if index < 0.0 {
            length as f64 + index
        } else {
            index
        };
        if index < 0.0 || index >= length as f64 {
            return Value::UNDEFINED;
        }
        let index = index as usize;
        let value = if heap.present(receiver, index) {
            heap.get(receiver, index).unwrap()
        } else {
            Value::UNDEFINED
        };
        return context.retain(value);
    }
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
            Some(value) => value,
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
    let mut allocation = Held::new(&heap);
    let result = if matches!(function, ArrayEvery | ArraySome | ArrayFind | ArrayForEach) {
        None
    } else {
        let initial = if function == ArrayMap {
            vec![Value::UNDEFINED; length]
        } else {
            Vec::new()
        };
        let Some(result) = heap.array(&initial) else {
            return fail(context, "heap exhausted");
        };
        allocation.push(result);
        if function == ArrayMap {
            for index in 0..length {
                heap.mark_hole(result, index);
            }
        }
        Some(result)
    };
    let mut iteration = Held::new(&heap);
    for index in 0..length {
        let present = heap.present(receiver, index);
        if !present && function != ArrayFind {
            continue;
        }
        let value = if present {
            heap.get(receiver, index).unwrap()
        } else {
            Value::UNDEFINED
        };
        iteration.release();
        iteration.push(unsafe { &mut *ctx }.retain(value));
        let callback_args = [value, Value::number(index as f64), receiver];
        let mapped = js_call(ctx, callback, Value::UNDEFINED, callback_args.as_ptr(), 3);
        iteration.push(mapped);
        let context = unsafe { &mut *ctx };
        if context.error.is_some() || context.exception_pending() {
            return Value::UNDEFINED;
        }
        if function == ArrayMap {
            heap.set(result.unwrap(), index, mapped);
        } else if function == ArrayFlatMap {
            let result = result.unwrap();
            if heap.is_array(mapped) {
                let mut next = heap.len(result).unwrap();
                if !flatten_into(&heap, result, mapped, 0, &mut next) {
                    return fail(context, "cannot append flatMap element");
                }
            } else if !heap.set(result, heap.len(result).unwrap(), mapped) {
                return fail(context, "cannot append flatMap element");
            }
        } else if function == ArrayEvery {
            if js_truthy(ctx, mapped) != Value::TRUE {
                return Value::FALSE;
            }
        } else if function == ArraySome {
            if js_truthy(ctx, mapped) == Value::TRUE {
                return Value::TRUE;
            }
        } else if function == ArrayFind {
            if js_truthy(ctx, mapped) == Value::TRUE {
                return context.retain(value);
            }
        } else if function == ArrayForEach {
        } else if js_truthy(ctx, mapped) == Value::TRUE {
            let result = result.unwrap();
            heap.set(result, heap.len(result).unwrap(), value);
        }
    }
    match function {
        ArrayEvery => Value::TRUE,
        ArraySome => Value::FALSE,
        ArrayFind | ArrayForEach => Value::UNDEFINED,
        _ => {
            allocation.take();
            result.unwrap()
        }
    }
}
