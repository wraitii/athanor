use super::*;

pub(super) extern "C" fn js_generator_new(
    ctx: *mut JitCtx,
    closure: Value,
    receiver: Value,
    arguments: Value,
    plan: u64,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().generator(
        unsafe { &*(plan as *const std::sync::Arc<crate::continuation::Plan>) }.clone(),
        closure,
        arguments,
        receiver,
    ) {
        Some(value) => value,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
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
    result
}

pub(crate) fn generator_next(ctx: *mut JitCtx, generator: Value) -> Value {
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
    let (plan, closure, arguments) = context.global.heap().generator_parts(generator).unwrap();
    let region = &plan.entries[state];
    let receiver = context.global.heap().generator_receiver(generator).unwrap();
    let (_, locals) = context.global.heap().generator_begin(generator).unwrap();
    let args: Vec<_> = if state == 0 {
        (0..context.global.heap().len(arguments).unwrap())
            .map(|index| context.global.heap().get(arguments, index).unwrap())
            .collect()
    } else {
        locals.clone()
    };
    let result = super::super::continuation::run(ctx, region, closure, receiver, &args);
    let context = unsafe { &mut *ctx };
    context.global.heap().release_all(&locals);
    let Some((exit, mut output)) = result else {
        context.global.heap().generator_complete(generator);
        return Value::UNDEFINED;
    };
    let done = match exit {
        crate::continuation::Exit::Complete => {
            output.truncate(1);
            context.global.heap().generator_complete(generator);
            true
        }
        crate::continuation::Exit::Suspend(target) => {
            output.truncate(1 + plan.entries[target].inputs);
            let locals = output.split_off(1);
            context
                .global
                .heap()
                .generator_suspend(generator, target, locals);
            false
        }
    };
    generator_result(context, output[0], done)
}
