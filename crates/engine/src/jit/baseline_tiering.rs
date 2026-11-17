use std::cell::{Cell, RefCell};
use std::sync::Arc;

use super::semantic::{Block, Graph, Value as GraphValue};
use crate::value::Value;

pub(crate) const DEFAULT_WARMUP: u64 = 128;

pub(in crate::jit) struct Request {
    pub(in crate::jit) site: usize,
    pub(in crate::jit) graph: Arc<Graph>,
}

pub(crate) struct Site {
    pub(in crate::jit) graph: Arc<Graph>,
    pub(in crate::jit) header: Block,
    pub(in crate::jit) source_name: (usize, usize),
    pub(in crate::jit) entries: Cell<u64>,
    pub(in crate::jit) entry: Cell<usize>,
    pub(in crate::jit) requested: Cell<bool>,
    pub(in crate::jit) transfers: Cell<u64>,
    pub(in crate::jit) inputs: RefCell<Option<Inputs>>,
}

pub(in crate::jit) struct Inputs {
    pub(in crate::jit) values: Vec<GraphValue>,
    pub(in crate::jit) bindings: Vec<super::semantic::Binding>,
}

pub(in crate::jit) const ENTRIES_OFFSET: i32 = std::mem::offset_of!(Site, entries) as i32;
pub(in crate::jit) const ENTRY_OFFSET: i32 = std::mem::offset_of!(Site, entry) as i32;

impl Site {
    pub(in crate::jit) fn new(
        graph: Arc<Graph>,
        header: Block,
        source_name: (usize, usize),
    ) -> Self {
        Self {
            graph,
            header,
            source_name,
            entries: Cell::new(0),
            entry: Cell::new(0),
            requested: Cell::new(false),
            transfers: Cell::new(0),
            inputs: RefCell::new(None),
        }
    }
}

pub(in crate::jit) extern "C" fn poll(
    ctx: *mut super::session::JitCtx,
    site: *const Site,
    slots: *const Value,
    receiver: Value,
    result: *mut Value,
) -> u64 {
    let site = unsafe { &*site };
    let entry = unsafe { &*ctx }.engine.tiering.baseline_osr_entry(site);
    if entry == 0 {
        return 0;
    }
    let inputs = site.inputs.borrow();
    let inputs = inputs.as_ref().unwrap();
    // The suspended baseline frame owns these inputs until the continuation returns.
    let mut args = Vec::with_capacity(inputs.values.len() + inputs.bindings.len());
    for value in &inputs.values {
        args.push(unsafe { *slots.add(value.as_u32() as usize) });
    }
    for binding in &inputs.bindings {
        args.push(unsafe { *slots.add(site.graph.values.len() + binding.as_u32() as usize) });
    }
    unsafe { &*ctx }.engine.tiering.baseline_osr_entered(site);
    let function: unsafe extern "C" fn(
        *mut super::session::JitCtx,
        Value,
        Value,
        *const Value,
        u64,
    ) -> Value = unsafe { std::mem::transmute(entry) };
    unsafe {
        *result = function(
            ctx,
            Value::UNDEFINED,
            receiver,
            args.as_ptr(),
            args.len() as u64,
        );
    }
    1
}
