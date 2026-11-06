use std::cell::Cell;
use std::sync::Arc;

use super::semantic::osr::Plan;

pub(crate) struct Site {
    pub(crate) owner: u64,
    pub(in crate::jit) plan: Arc<Plan>,
    pub(crate) candidate: usize,
    pub(crate) entries: Cell<u64>,
    pub(crate) entry: Cell<usize>,
    pub(crate) requested: Cell<bool>,
    pub(crate) transfers: Cell<u64>,
}

pub(in crate::jit) const ENTRIES_OFFSET: i32 = std::mem::offset_of!(Site, entries) as i32;
pub(in crate::jit) const ENTRY_OFFSET: i32 = std::mem::offset_of!(Site, entry) as i32;

impl Site {
    pub(crate) fn header(&self) -> String {
        self.plan.loops[self.candidate].header.to_string()
    }

    pub(in crate::jit) fn new(owner: u64, plan: Arc<Plan>, candidate: usize) -> Self {
        Self {
            owner,
            plan,
            candidate,
            entries: Cell::new(0),
            entry: Cell::new(0),
            requested: Cell::new(false),
            transfers: Cell::new(0),
        }
    }
}
