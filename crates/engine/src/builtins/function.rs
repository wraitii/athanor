use super::{NativeFunction, fail};
use crate::jit::session::JitCtx;
use crate::value::Value;

pub(super) fn call(
    ctx: *mut JitCtx,
    _function: NativeFunction,
    _receiver: Value,
    _args: &[Value],
) -> Value {
    fail(
        unsafe { &mut *ctx },
        "dynamic Function construction is unsupported",
    )
}
