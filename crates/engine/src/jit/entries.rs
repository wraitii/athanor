use crate::FxHashMap;
use std::cell::Cell;

use crate::jit::session::JitCtx;
use crate::value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum VariantId {
    Ordinary(u64),
    Specialized(usize),
}

impl VariantId {
    // Ordinary slots retain function indices for generic dispatch stubs.
    pub(crate) fn ordinary(function: u64) -> Self {
        Self::Ordinary(function)
    }

    pub(crate) fn index(self) -> usize {
        match self {
            Self::Ordinary(function) => {
                usize::try_from(function).expect("variant ID fits the address space")
            }
            Self::Specialized(_) => panic!("specialized variants have dedicated entry cells"),
        }
    }
}

#[derive(Default)]
pub(crate) struct Variants(Vec<Option<Entry>>, FxHashMap<VariantId, Entry>);

impl Variants {
    pub(crate) fn get(&self, variant: VariantId) -> Option<Entry> {
        match variant {
            VariantId::Ordinary(function) => self
                .0
                .get(usize::try_from(function).ok()?)
                .copied()
                .flatten(),
            VariantId::Specialized(_) => self.1.get(&variant).copied(),
        }
    }

    pub(crate) fn publish(&mut self, variant: VariantId, entry: Entry) {
        if matches!(variant, VariantId::Specialized(_)) {
            self.1.insert(variant, entry);
            return;
        }
        let index = variant.index();
        if self.0.len() <= index {
            self.0.resize(index + 1, None);
        }
        self.0[index] = Some(entry);
    }
}

pub(crate) const DIRECT_ENTRIES: usize = 1 << 20;
pub(crate) const DIRECT_ENTRY_SIZE: usize = 4 * std::mem::size_of::<usize>();
pub(crate) const DIRECT_ADAPTER_OFFSET: usize = std::mem::size_of::<usize>();
pub(crate) const DIRECT_ARITY_OFFSET: usize = 2 * std::mem::size_of::<usize>();
pub(crate) const DIRECT_CALLS_OFFSET: usize = 3 * std::mem::size_of::<usize>();
pub(crate) const RECOMPILE_WARMUP: usize = 8;
pub(crate) const RECOMPILE_PENDING: usize = 16;
const WORDS: usize = DIRECT_ENTRY_SIZE / std::mem::size_of::<usize>();

/// Each variant slot has a fixed-arity entry, an argument adapter, and its arity plus one.
/// Ordinary slots also hold the logical function's saturating completed-call count and pending
/// flag. Zero entries require runtime dispatch, including pending feedback in synchronous mode. Adapters and arities stay zero for closures requiring lexical
/// receiver resolution. The table never moves, and withdrawing an entry preserves its count.
pub(crate) struct DirectEntries(Box<[Cell<usize>]>);

pub(crate) fn zeroed_cells(len: usize) -> Box<[Cell<usize>]> {
    let zeroed = vec![0usize; len].into_boxed_slice();
    // Cell<usize> is repr(transparent); starting from zeroed memory leaves the pages untouched.
    unsafe { Box::from_raw(Box::into_raw(zeroed) as *mut [Cell<usize>]) }
}

impl DirectEntries {
    pub(crate) fn new() -> Self {
        Self(zeroed_cells(DIRECT_ENTRIES * WORDS))
    }

    pub(crate) fn address(&self) -> *const usize {
        self.0.as_ptr().cast()
    }

    pub(crate) fn publish(&self, variant: VariantId, entry: Entry) {
        if matches!(variant, VariantId::Ordinary(function) if function < DIRECT_ENTRIES as u64) {
            let index = variant.index() * WORDS;
            let adapter = entry.adapter.filter(|_| !entry.lexical_this);
            self.0[index].set(entry.code);
            self.0[index + 1].set(adapter.unwrap_or(0));
            self.0[index + 2].set(adapter.map_or(0, |_| usize::from(entry.arity) + 1));
        }
    }

    pub(crate) fn withdraw(&self, variant: VariantId) {
        if matches!(variant, VariantId::Ordinary(function) if function < DIRECT_ENTRIES as u64) {
            let index = variant.index() * WORDS;
            for word in &self.0[index..index + 3] {
                word.set(0);
            }
        }
    }

    pub(crate) fn warmed(&self, code: u64) -> bool {
        code < DIRECT_ENTRIES as u64
            && self.0[code as usize * WORDS + 3].get() & (RECOMPILE_PENDING - 1) >= RECOMPILE_WARMUP
    }

    pub(crate) fn called(&self, code: u64) -> bool {
        code < DIRECT_ENTRIES as u64
            && self.0[code as usize * WORDS + 3].get() & (RECOMPILE_PENDING - 1) != 0
    }

    pub(crate) fn warm(&self, code: u64) {
        if code >= DIRECT_ENTRIES as u64 {
            return;
        }
        let slot = &self.0[code as usize * WORDS + 3];
        slot.set((slot.get() & RECOMPILE_PENDING) | RECOMPILE_WARMUP);
    }

    pub(crate) fn set_pending(&self, code: u64, pending: bool) {
        if code >= DIRECT_ENTRIES as u64 {
            return;
        }
        let slot = &self.0[code as usize * WORDS + 3];
        let value = slot.get();
        slot.set(if pending {
            value | RECOMPILE_PENDING
        } else {
            value & !RECOMPILE_PENDING
        });
    }
}

type Generic = extern "C" fn(*mut JitCtx, Value, Value, *const Value, u64) -> Value;
type Adapter = extern "C" fn(*mut JitCtx, Value, Value, *const Value, u64, usize) -> Value;

/// Code with an adapter takes `arity` arguments in registers under the tail calling convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) code: usize,
    pub(crate) adapter: Option<usize>,
    pub(crate) arity: u16,
    pub(crate) lexical_this: bool,
}

impl Entry {
    pub(crate) fn generic(code: usize) -> Self {
        Self {
            code,
            adapter: None,
            arity: 0,
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
