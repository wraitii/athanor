use crate::continuation::{Entry, Exit};
use crate::jit::session::JitCtx;
use crate::value::Value;

pub(super) fn run(
    ctx: *mut JitCtx,
    region: &Entry,
    closure: Value,
    receiver: Value,
    inputs: &[Value],
) -> Option<(Exit, Vec<Value>)> {
    let context = unsafe { &mut *ctx };
    let entry = match context.entry(region.code.0) {
        Ok(Some(entry)) => entry,
        Ok(None) => {
            context.error = Some(Box::new(crate::Error::Runtime(
                "continuation code is unavailable".into(),
            )));
            return None;
        }
        Err(error) => {
            context.error = Some(Box::new(error));
            return None;
        }
    };
    // Inputs remain borrowed from the driver while successful exits publish owned outputs.
    let mut buffer = vec![Value::UNDEFINED; inputs.len().max(region.outputs)];
    buffer[..inputs.len()].copy_from_slice(inputs);
    let result = entry.call(ctx, closure, receiver, buffer.as_ptr(), inputs.len() as u64);
    let context = unsafe { &mut *ctx };
    if context.error.is_some() || context.exception_pending() {
        context.global.heap().release(result);
        return None;
    }
    let index = result
        .as_number()
        .expect("continuation exit must be numeric") as usize;
    Some((region.exits[index], buffer))
}
