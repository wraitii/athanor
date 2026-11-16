use super::{NativeFunction, arg, fail};
use crate::jit::helpers::generator_next;
use crate::jit::session::JitCtx;
use crate::value::Value;

pub(super) fn call(
    ctx: *mut JitCtx,
    _function: NativeFunction,
    receiver: Value,
    args: &[Value],
) -> Value {
    if arg(args, 0) != Value::UNDEFINED {
        return fail(
            unsafe { &mut *ctx },
            "generator resume arguments are unsupported",
        );
    }
    generator_next(ctx, receiver)
}
