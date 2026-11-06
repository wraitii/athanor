use super::*;

pub(super) fn argument_values(ctx: *mut JitCtx, arguments: Value) -> Option<Vec<Value>> {
    if let Some(values) = unsafe { &*ctx }
        .global
        .heap()
        .owned_indexed_values(arguments)
    {
        return Some(values);
    }
    let state = js_iterate_new(ctx, arguments);
    if unsafe { &*ctx }.error.is_some() {
        return None;
    }
    let mut values = Vec::new();
    loop {
        let value = js_iterate_next(ctx, state);
        if unsafe { &*ctx }.error.is_some() {
            return None;
        }
        if value == Value::UNINITIALIZED {
            for &value in &values {
                unsafe { &*ctx }.global.heap().retain(value);
            }
            return Some(values);
        }
        values.push(value);
    }
}

pub(in crate::jit) extern "C" fn js_call_spread(
    ctx: *mut JitCtx,
    callable: Value,
    receiver: Value,
    arguments: Value,
) -> Value {
    let Some(values) = argument_values(ctx, arguments) else {
        return Value::UNDEFINED;
    };
    let result = js_call(
        ctx,
        callable,
        receiver,
        values.as_ptr(),
        values.len() as u64,
    );
    unsafe { &*ctx }.global.heap().release_all(&values);
    result
}

pub(super) extern "C" fn js_construct_spread(
    ctx: *mut JitCtx,
    callable: Value,
    arguments: Value,
) -> Value {
    let Some(values) = argument_values(ctx, arguments) else {
        return Value::UNDEFINED;
    };
    let result = js_construct(ctx, callable, values.as_ptr(), values.len() as u64);
    unsafe { &*ctx }.global.heap().release_all(&values);
    result
}

pub(in crate::jit) extern "C" fn js_call(
    ctx: *mut JitCtx,
    closure: Value,
    receiver: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    if unsafe { &*ctx }.global.heap().is_class(closure) {
        unsafe { &mut *ctx }.error = Some(Box::new(crate::Error::Runtime(
            "class constructor requires new".into(),
        )));
        return Value::UNDEFINED;
    }
    invoke(ctx, closure, receiver, argv, argc)
}

pub(in crate::jit) extern "C" fn js_call_owned(
    ctx: *mut JitCtx,
    closure: Value,
    receiver: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    let value = js_call(ctx, closure, receiver, argv, argc);
    unsafe { &mut *ctx }.disown(value)
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
            context.error = Some(Box::new(crate::Error::Runtime(
                "value is not callable in this engine".into(),
            )));
            return Value::UNDEFINED;
        };
        entry
    };
    let receiver = unsafe { &*ctx }
        .global
        .heap()
        .lexical_this(closure)
        .unwrap_or(receiver);
    let value = entry.call(ctx, closure, receiver, argv, argc);
    unsafe { &mut *ctx }.own(value)
}

pub(super) extern "C" fn js_new_instance(ctx: *mut JitCtx, prototype: Value) -> Value {
    let value = js_new_instance_owned(ctx, prototype);
    unsafe { &mut *ctx }.own(value)
}

pub(super) extern "C" fn js_new_instance_owned(ctx: *mut JitCtx, prototype: Value) -> Value {
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

pub(super) extern "C" fn js_constructor_call_spread(
    ctx: *mut JitCtx,
    constructor: Value,
    instance: Value,
    arguments: Value,
) -> Value {
    let Some(values) = argument_values(ctx, arguments) else {
        return Value::UNDEFINED;
    };
    let result = invoke(
        ctx,
        constructor,
        instance,
        values.as_ptr(),
        values.len() as u64,
    );
    unsafe { &*ctx }.global.heap().release_all(&values);
    result
}

pub(super) extern "C" fn js_constructor_prototype(ctx: *mut JitCtx, constructor: Value) -> Value {
    unsafe { &*ctx }
        .global
        .heap()
        .constructor_prototype(constructor)
        .unwrap_or(Value::UNINITIALIZED)
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

pub(super) extern "C" fn js_construct_spread_resolved(
    ctx: *mut JitCtx,
    constructor: Value,
    prototype: Value,
    arguments: Value,
) -> Value {
    let Some(values) = argument_values(ctx, arguments) else {
        return Value::UNDEFINED;
    };
    let result = js_construct_resolved(
        ctx,
        constructor,
        prototype,
        values.as_ptr(),
        values.len() as u64,
    );
    unsafe { &*ctx }.global.heap().release_all(&values);
    result
}

pub(super) fn invoke_constructor(
    ctx: *mut JitCtx,
    constructor: Value,
    instance: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    unsafe { &mut *ctx }.own(instance);
    let result = invoke(ctx, constructor, instance, argv, argc);
    let context = unsafe { &*ctx };
    if context.error.is_some() || context.exception_pending() {
        return Value::UNDEFINED;
    }
    if result.is_object() { result } else { instance }
}

pub(super) extern "C" fn js_construct(
    ctx: *mut JitCtx,
    constructor: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    if ctx.global.heap().native(constructor) == Some(crate::heap::NativeFunction::Array) {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return crate::jit::builtins::array::construct(ctx, args);
    }
    if let Some(crate::heap::NativeFunction::TypedConstructor(kind)) =
        ctx.global.heap().native(constructor)
    {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return crate::jit::builtins::typed_array::construct(ctx, kind, args);
    }
    if ctx.global.heap().constructable(constructor) {
        let Some(instance) = ctx.global.heap().constructed_object(constructor) else {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            return Value::UNDEFINED;
        };
        return invoke_constructor(ctx, constructor, instance, argv, argc);
    }
    if ctx.global.heap().native(constructor) == Some(crate::heap::NativeFunction::Set) {
        let argument = if argc == 0 {
            Value::UNDEFINED
        } else {
            unsafe { *argv }
        };
        if !argument.is_nullish() {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "Set iterable construction is unsupported".into(),
            )));
            return Value::UNDEFINED;
        }
        return match ctx.global.heap().set_new() {
            Some(value) => ctx.own(value),
            None => {
                ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
                Value::UNDEFINED
            }
        };
    }
    if ctx.global.heap().native(constructor) != Some(crate::heap::NativeFunction::Error) {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "unsupported constructor".into(),
        )));
        return Value::UNDEFINED;
    }
    let argument = if argc == 0 {
        Value::UNDEFINED
    } else {
        unsafe { *argv }
    };
    let heap = ctx.global.heap();
    let text = if argument.is_undefined() {
        String::new()
    } else if argument.is_object() {
        let Some(text) = heap.error_text(argument) else {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "object-to-primitive conversion is unsupported".into(),
            )));
            return Value::UNDEFINED;
        };
        text
    } else {
        heap.text(argument).unwrap_or_else(|| argument.to_string())
    };
    let Some(message) = heap.string(&text) else {
        ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
        return Value::UNDEFINED;
    };
    let Some(name) = heap.string("Error") else {
        heap.release(message);
        ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
        return Value::UNDEFINED;
    };
    let result = heap.error(message, name);
    heap.release(message);
    heap.release(name);
    match result {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

pub(super) fn native_call(
    ctx: *mut JitCtx,
    function: crate::heap::NativeFunction,
    receiver: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    use crate::heap::NativeFunction;
    if matches!(
        function,
        NativeFunction::JsonParse | NativeFunction::JsonStringify
    ) {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return crate::jit::builtins::json::call(unsafe { &mut *ctx }, function, args);
    }
    if matches!(
        function,
        NativeFunction::ObjectKeys
            | NativeFunction::ObjectValues
            | NativeFunction::ObjectGetOwnPropertyNames
            | NativeFunction::ObjectFreeze
            | NativeFunction::ObjectIsFrozen
    ) {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return crate::jit::builtins::object::call(unsafe { &mut *ctx }, function, args);
    }
    if let NativeFunction::Math(operation) = function {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return crate::jit::builtins::math::call(unsafe { &mut *ctx }, operation, args);
    }
    if matches!(function, NativeFunction::TypedConstructor(_)) {
        unsafe { &mut *ctx }.error = Some(Box::new(crate::Error::Runtime(
            "typed array constructor requires new".into(),
        )));
        return Value::UNDEFINED;
    }
    if function == NativeFunction::Array {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return crate::jit::builtins::array::construct(unsafe { &mut *ctx }, args);
    }
    if function == NativeFunction::HostSeedRandom {
        let argument = if argc == 0 {
            Value::UNDEFINED
        } else {
            unsafe { *argv }
        };
        if !numeric_operands(ctx, &[argument]) {
            return Value::UNDEFINED;
        }
        let seed = to_int32(primitive_number(ctx, argument)) as u32;
        unsafe { &*ctx }.engine.seed_random(seed);
        return Value::UNDEFINED;
    }
    if function == NativeFunction::Error {
        let constructor = unsafe { &*ctx }
            .global
            .heap()
            .native_function(function)
            .unwrap();
        let value = js_construct(ctx, constructor, argv, argc);
        unsafe { &*ctx }.global.heap().release(constructor);
        return value;
    }
    if function == NativeFunction::GeneratorNext {
        if argc != 0 && unsafe { *argv } != Value::UNDEFINED {
            unsafe { &mut *ctx }.error = Some(Box::new(crate::Error::Runtime(
                "generator resume arguments are unsupported".into(),
            )));
            return Value::UNDEFINED;
        }
        return generator_next(ctx, receiver);
    }
    if !matches!(
        function,
        NativeFunction::Set
            | NativeFunction::SetHas
            | NativeFunction::SetDelete
            | NativeFunction::SetAdd
            | NativeFunction::SetClear
    ) {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return crate::jit::builtins::call(ctx, function, receiver, args);
    }
    let ctx = unsafe { &mut *ctx };
    if function == NativeFunction::Set {
        ctx.error = Some(Box::new(crate::Error::Runtime("Set requires new".into())));
        return Value::UNDEFINED;
    }
    if ctx.global.heap().set_size(receiver).is_none() {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "Set method requires a Set receiver".into(),
        )));
        return Value::UNDEFINED;
    }
    let argument = if argc == 0 {
        Value::UNDEFINED
    } else {
        unsafe { *argv }
    };
    match function {
        NativeFunction::SetHas => {
            Value::boolean(ctx.global.heap().set_has(receiver, argument).unwrap())
        }
        NativeFunction::SetDelete => {
            Value::boolean(ctx.global.heap().set_delete(receiver, argument).unwrap())
        }
        NativeFunction::SetAdd => {
            if !ctx.global.heap().set_add(receiver, argument) {
                ctx.error = Some(Box::new(crate::Error::Runtime("invalid Set entry".into())));
                return Value::UNDEFINED;
            }
            ctx.retain_temp(receiver)
        }
        NativeFunction::SetClear => {
            if !ctx.global.heap().set_clear(receiver) {
                ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            }
            Value::UNDEFINED
        }
        _ => unreachable!(),
    }
}
