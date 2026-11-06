use std::cell::Cell;

use crate::jit::session::JitCtx;
use crate::value::Value;

pub(crate) const DIRECT_ENTRIES: usize = 1 << 20;

/// Indexed by code ID. Zero means callers must take the generic path, either because the code
/// has no fixed-arity entry or because feedback asked for its recompilation on the next generic
/// call. The table never moves, so generated code embeds its address.
pub(crate) struct DirectEntries(Box<[Cell<usize>]>);

pub(crate) fn zeroed_cells(len: usize) -> Box<[Cell<usize>]> {
    let zeroed = vec![0usize; len].into_boxed_slice();
    // Cell<usize> is repr(transparent); starting from zeroed memory leaves the pages untouched.
    unsafe { Box::from_raw(Box::into_raw(zeroed) as *mut [Cell<usize>]) }
}

impl DirectEntries {
    pub(crate) fn new() -> Self {
        Self(zeroed_cells(DIRECT_ENTRIES))
    }

    pub(crate) fn address(&self) -> *const usize {
        self.0.as_ptr().cast()
    }

    pub(crate) fn publish(&self, code: u64, entry: usize) {
        if let Some(slot) = self.0.get(code as usize) {
            slot.set(entry);
        }
    }

    pub(crate) fn withdraw(&self, code: u64) {
        self.publish(code, 0);
    }
}

type Generic = extern "C" fn(*mut JitCtx, Value, Value, *const Value, u64) -> Value;
type Adapter = extern "C" fn(*mut JitCtx, Value, Value, *const Value, u64, usize) -> Value;

/// Code with an adapter takes its arguments in registers under the tail calling convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) code: usize,
    pub(crate) adapter: Option<usize>,
}

impl Entry {
    pub(crate) fn generic(code: usize) -> Self {
        Self {
            code,
            adapter: None,
        }
    }

    pub(crate) fn call(
        self,
        ctx: *mut JitCtx,
        closure: Value,
        receiver: Value,
        argv: *const Value,
        argc: u64,
    ) -> Value {
        match self.adapter {
            Some(adapter) => {
                let adapter: Adapter = unsafe { std::mem::transmute(adapter) };
                adapter(ctx, closure, receiver, argv, argc, self.code)
            }
            None => {
                let function: Generic = unsafe { std::mem::transmute(self.code) };
                function(ctx, closure, receiver, argv, argc)
            }
        }
    }
}
