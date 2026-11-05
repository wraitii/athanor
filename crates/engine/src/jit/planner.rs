use std::collections::{HashMap, HashSet};

use crate::heap::TypedKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Site {
    pub(super) source: usize,
    pub(super) start: u32,
    pub(super) end: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ReadKind {
    Result,
    Receiver,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadType {
    Array,
    Typed(TypedKind),
}

#[repr(C)]
pub(crate) struct ObservationCache {
    pub(crate) recorded: u64,
    pub(crate) owner: u64,
    pub(crate) site: Site,
}

#[derive(Default)]
pub(crate) struct Planner {
    targets: HashMap<Site, u64>,
    observed: HashSet<(u64, Site)>,
    reads: HashMap<(Site, ReadKind), ReadType>,
    observed_reads: HashSet<(u64, Site, ReadKind)>,
    pending: Vec<bool>,
}

impl Planner {
    pub(crate) fn observe(&mut self, owner: u64, site: Site, target: u64) {
        if self.observed.insert((owner, site)) {
            self.targets.entry(site).or_insert(target);
            let owner = owner as usize;
            self.pending
                .resize(self.pending.len().max(owner + 1), false);
            self.pending[owner] = true;
        }
    }

    pub(crate) fn observe_get(
        &mut self,
        owner: u64,
        site: Site,
        kind: ReadKind,
        observed: ReadType,
    ) {
        if self.observed_reads.insert((owner, site, kind)) {
            self.reads.entry((site, kind)).or_insert(observed);
            let owner = owner as usize;
            self.pending
                .resize(self.pending.len().max(owner + 1), false);
            self.pending[owner] = true;
        }
    }

    pub(super) fn reads(&self) -> impl Iterator<Item = ((Site, ReadKind), ReadType)> + '_ {
        self.reads.iter().map(|(&site, &element)| (site, element))
    }

    pub(crate) fn take(&mut self, owner: u64) -> bool {
        self.pending
            .get_mut(owner as usize)
            .is_some_and(|pending| std::mem::replace(pending, false))
    }

    pub(super) fn targets(&self) -> impl Iterator<Item = (Site, u64)> + '_ {
        self.targets.iter().map(|(&site, &target)| (site, target))
    }
}
