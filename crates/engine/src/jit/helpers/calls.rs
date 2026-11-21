use super::*;

pub(super) fn argument_values<'a>(
    ctx: *mut JitCtx<'a>,
    arguments: Value,
) -> Option<super::super::pack::OwnedValues<'a>> {
    let mut values = unsafe { super::super::pack::buffer(ctx) };
    if unsafe { &*ctx }
        .global
        .heap()
        .owned_indexed_values_into(arguments, &mut values.values)
        .is_some()
    {
        return Some(values);
    }
    let state = js_iterate_new(ctx, arguments);
    let result = loop {
        if unsafe { &*ctx }.error.is_some() {
            break None;
        }
        let value = js_iterate_next(ctx, state);
        if unsafe { &*ctx }.error.is_some() {
            break None;
        }
        if value == Value::UNINITIALIZED {
            break Some(values);
        }
        values.values.push(value);
    };
    unsafe { &*ctx }.global.heap().release(state);
    result
}

pub(in crate::jit) extern "C" fn js_call_pack(
    ctx: *mut JitCtx,
    callable: Value,
    receiver: Value,
    arguments: Value,
) -> Value {
    let values = unsafe { super::super::pack::take(ctx, arguments) };
    let argv = values.values.as_ptr();
    let argc = values.values.len();
    js_call(ctx, callable, receiver, argv, argc as u64)
}

pub(super) extern "C" fn js_construct_pack(
    ctx: *mut JitCtx,
    callable: Value,
    arguments: Value,
) -> Value {
    let values = unsafe { super::super::pack::take(ctx, arguments) };
    let argv = values.values.as_ptr();
    let argc = values.values.len();
    js_construct(ctx, callable, argv, argc as u64)
}

pub(crate) extern "C" fn js_call(
    ctx: *mut JitCtx,
    closure: Value,
    receiver: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    if unsafe { &*ctx }.global.heap().is_class(closure) {
        unsafe { &mut *ctx }.language_error("TypeError", "class constructor requires new");
        return Value::UNDEFINED;
    }
    invoke(ctx, closure, receiver, argv, argc)
}

pub(super) fn invoke(
    ctx: *mut JitCtx,
    closure: Value,
    receiver: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    if let Some(function) = unsafe { &*ctx }.global.heap().native(closure) {
        return native_call(ctx, function, receiver, argv, argc);
    }
    let entry = {
        let context = unsafe { &mut *ctx };
        let entry = context
            .global
            .heap()
            .closure_code(closure)
            .map(|code| context.entry(code.0))
            .transpose();
        let entry = match entry {
            Ok(entry) => entry.flatten(),
            Err(error) => {
                context.error = Some(Box::new(error));
                return Value::UNDEFINED;
            }
        };
        let Some(entry) = entry else {
            context.language_error("TypeError", "value is not callable in this engine");
            return Value::UNDEFINED;
        };
        entry
    };
    let receiver = unsafe { &*ctx }
        .global
        .heap()
        .lexical_this(closure)
        .unwrap_or(receiver);
    entry.call(ctx, closure, receiver, argv, argc)
}

pub(super) extern "C" fn js_new_instance(ctx: *mut JitCtx, prototype: Value) -> Value {
    let context = unsafe { &mut *ctx };
    match context.global.heap().object_with_prototype(prototype) {
        Some(instance) => instance,
        None => {
            context.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

pub(super) extern "C" fn js_constructor_call(
    ctx: *mut JitCtx,
    constructor: Value,
    instance: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    invoke(ctx, constructor, instance, argv, argc)
}

pub(super) extern "C" fn js_constructor_call_pack(
    ctx: *mut JitCtx,
    constructor: Value,
    instance: Value,
    arguments: Value,
) -> Value {
    let values = unsafe { super::super::pack::take(ctx, arguments) };
    let argv = values.values.as_ptr();
    let argc = values.values.len();
    invoke(ctx, constructor, instance, argv, argc as u64)
}

pub(super) extern "C" fn js_construct_resolved(
    ctx: *mut JitCtx,
    constructor: Value,
    prototype: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    if prototype == Value::UNINITIALIZED {
        return js_construct(ctx, constructor, argv, argc);
    }
    let context = unsafe { &mut *ctx };
    let Some(instance) = context.global.heap().object_with_prototype(prototype) else {
        context.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
        return Value::UNDEFINED;
    };
    invoke_constructor(ctx, constructor, instance, argv, argc)
}

pub(super) extern "C" fn js_construct_pack_resolved(
    ctx: *mut JitCtx,
    constructor: Value,
    prototype: Value,
    arguments: Value,
) -> Value {
    let values = unsafe { super::super::pack::take(ctx, arguments) };
    let argv = values.values.as_ptr();
    let argc = values.values.len();
    js_construct_resolved(ctx, constructor, prototype, argv, argc as u64)
}

pub(super) fn invoke_constructor(
    ctx: *mut JitCtx,
    constructor: Value,
    instance: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    let result = invoke(ctx, constructor, instance, argv, argc);
    let context = unsafe { &*ctx };
    let heap = context.global.heap();
    if context.error.is_some() || context.exception_pending() {
        heap.release(result);
        heap.release(instance);
        return Value::UNDEFINED;
    }
    if result.is_object() {
        heap.release(instance);
        return result;
    }
    heap.release(result);
    instance
}

pub(super) extern "C" fn js_construct(
    ctx: *mut JitCtx,
    constructor: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    let context = unsafe { &mut *ctx };
    if let Some(function) = context.global.heap().native(constructor) {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return crate::builtins::construct(ctx, function, args);
    }
    if context.global.heap().constructable(constructor) {
        let Some(instance) = context.global.heap().constructed_object(constructor) else {
            context.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            return Value::UNDEFINED;
        };
        return invoke_constructor(ctx, constructor, instance, argv, argc);
    }
    context.error = Some(Box::new(crate::Error::Runtime(
        "unsupported constructor".into(),
    )));
    Value::UNDEFINED
}

pub(super) fn native_call(
    ctx: *mut JitCtx,
    function: crate::heap::NativeFunction,
    receiver: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    let args = if argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(argv, argc as usize) }
    };
    crate::builtins::call(ctx, function, receiver, args)
}
