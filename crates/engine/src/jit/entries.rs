use std::cell::Cell;

use crate::jit::session::JitCtx;
use crate::value::Value;

pub(crate) const DIRECT_ENTRIES: usize = 1 << 20;
pub(crate) const DIRECT_ENTRY_SIZE: usize = 2 * std::mem::size_of::<usize>();

/// Each code ID has a fixed-arity entry and an argument adapter. Zero entries require runtime
/// dispatch, including when feedback requests recompilation. Adapters stay zero for closures
/// requiring lexical receiver resolution. The table never moves.
pub(crate) struct DirectEntries(Box<[Cell<usize>]>);

pub(crate) fn zeroed_cells(len: usize) -> Box<[Cell<usize>]> {
    let zeroed = vec![0usize; len].into_boxed_slice();
    // Cell<usize> is repr(transparent); starting from zeroed memory leaves the pages untouched.
    unsafe { Box::from_raw(Box::into_raw(zeroed) as *mut [Cell<usize>]) }
}

impl DirectEntries {
    pub(crate) fn new() -> Self {
        Self(zeroed_cells(DIRECT_ENTRIES * 2))
    }

    pub(crate) fn address(&self) -> *const usize {
        self.0.as_ptr().cast()
    }

    pub(crate) fn publish(&self, code: u64, entry: Entry) {
        if code < DIRECT_ENTRIES as u64 {
            let index = code as usize * 2;
            self.0[index].set(entry.code);
            self.0[index + 1].set(if entry.lexical_this {
                0
            } else {
                entry.adapter.unwrap_or(0)
            });
        }
    }

    pub(crate) fn withdraw(&self, code: u64) {
        if code < DIRECT_ENTRIES as u64 {
            let index = code as usize * 2;
            self.0[index].set(0);
            self.0[index + 1].set(0);
        }
    }
}

type Generic = extern "C" fn(*mut JitCtx, Value, Value, *const Value, u64) -> Value;
type Adapter = extern "C" fn(*mut JitCtx, Value, Value, *const Value, u64, usize) -> Value;

/// Code with an adapter takes its arguments in registers under the tail calling convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) code: usize,
    pub(crate) adapter: Option<usize>,
    pub(crate) lexical_this: bool,
}

impl Entry {
    pub(crate) fn generic(code: usize) -> Self {
        Self {
            code,
            adapter: None,
            lexical_this: false,
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
