use super::*;

pub(super) extern "C" fn js_generator_new(
    ctx: *mut JitCtx,
    closure: Value,
    receiver: Value,
    arguments: Value,
    code: u64,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx
        .global
        .heap()
        .generator(crate::heap::CodeId(code), closure, arguments, receiver)
    {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

pub(super) extern "C" fn js_generator_closure(ctx: *mut JitCtx, generator: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let closure = ctx.global.heap().generator_parts(generator).unwrap().1;
    ctx.retain_temp(closure)
}

pub(super) extern "C" fn js_generator_state(ctx: *mut JitCtx) -> u64 {
    unsafe { &*ctx }.resume_state() as u64
}

pub(super) extern "C" fn js_local_get(ctx: *mut JitCtx, slot: u64) -> Value {
    unsafe { &mut *ctx }.local_get(slot as usize)
}

pub(super) extern "C" fn js_generator_suspend(
    ctx: *mut JitCtx,
    generator: Value,
    value: Value,
    state: u64,
) -> Value {
    unsafe { &mut *ctx }.suspend(generator, value, state as usize)
}

pub(super) extern "C" fn js_generator_complete(
    ctx: *mut JitCtx,
    generator: Value,
    value: Value,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let value = ctx.exit(value);
    ctx.global.heap().generator_complete(generator);
    value
}

pub(super) fn generator_result(ctx: &mut JitCtx, value: Value, done: bool) -> Value {
    let Some(result) = ctx.global.heap().object() else {
        ctx.global.heap().release(value);
        ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
        return Value::UNDEFINED;
    };
    ctx.global.heap().set_property(result, "value", value);
    ctx.global
        .heap()
        .set_property(result, "done", Value::boolean(done));
    ctx.global.heap().release(value);
    ctx.own(result)
}

pub(super) fn generator_next(ctx: *mut JitCtx, generator: Value) -> Value {
    use crate::heap::{GENERATOR_DONE, GENERATOR_RUNNING};
    let context = unsafe { &mut *ctx };
    let Some(state) = context.global.heap().generator_state(generator) else {
        context.error = Some(Box::new(crate::Error::Runtime(
            "next requires a generator receiver".into(),
        )));
        return Value::UNDEFINED;
    };
    if state == GENERATOR_DONE {
        return generator_result(context, Value::UNDEFINED, true);
    }
    if state == GENERATOR_RUNNING {
        context.error = Some(Box::new(crate::Error::Runtime(
            "generator is already running".into(),
        )));
        return Value::UNDEFINED;
    }
    let (code, _, arguments) = context.global.heap().generator_parts(generator).unwrap();
    let entry = match context.entry(code.0) {
        Ok(entry) => entry,
        Err(error) => {
            context.error = Some(Box::new(error));
            return Value::UNDEFINED;
        }
    };
    let Some(entry) = entry else {
        context.error = Some(Box::new(crate::Error::Runtime(
            "generator code is unavailable".into(),
        )));
        return Value::UNDEFINED;
    };
    let args: Vec<_> = (0..context.global.heap().len(arguments).unwrap())
        .map(|index| context.global.heap().get(arguments, index).unwrap())
        .collect();
    let receiver = context.global.heap().generator_receiver(generator).unwrap();
    let (state, locals) = context.global.heap().generator_begin(generator).unwrap();
    if let Err(locals) = context.resume(locals, state) {
        context
            .global
            .heap()
            .generator_suspend(generator, state, locals);
        return Value::UNDEFINED;
    }
    let value = entry.call(ctx, generator, receiver, args.as_ptr(), args.len() as u64);
    let context = unsafe { &mut *ctx };
    if context.error.is_some() || context.exception_pending() {
        context.global.heap().release(value);
        return Value::UNDEFINED;
    }
    let done = context.global.heap().generator_state(generator) == Some(GENERATOR_DONE);
    generator_result(context, value, done)
}
