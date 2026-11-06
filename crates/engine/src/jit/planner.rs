use std::collections::{HashMap, HashSet};

use rustc_hash::FxHashMap;

use crate::heap::TypedKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Site {
    pub(super) source: usize,
    pub(super) start: u32,
    pub(super) end: u32,
}

impl std::fmt::Display for Site {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
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

/// Identifies a call site reaching a miss stub. Guarded sites only get there when their
/// direct target does not match.
pub(crate) struct CallSite {
    pub(crate) site: Site,
    pub(crate) guarded: bool,
}

#[derive(Clone)]
pub(crate) struct CallProfile {
    pub(crate) generic: u64,
    pub(crate) guarded: u64,
    pub(crate) native: u64,
    pub(crate) targets: Vec<(u64, u64)>,
}

impl CallProfile {
    fn new() -> Self {
        Self {
            generic: 0,
            guarded: 0,
            native: 0,
            targets: Vec::new(),
        }
    }

    fn merge(&mut self, other: &Self) {
        self.generic += other.generic;
        self.guarded += other.guarded;
        self.native += other.native;
        for &(code, count) in &other.targets {
            self.record(code, count);
        }
    }

    fn record(&mut self, code: u64, count: u64) {
        match self.targets.iter_mut().find(|(target, _)| *target == code) {
            Some((_, total)) => *total += count,
            None => self.targets.push((code, count)),
        }
    }

    pub(crate) fn calls(&self) -> u64 {
        self.generic + self.guarded
    }

    pub(crate) fn target_count(&self) -> usize {
        self.targets.len() + usize::from(self.native != 0)
    }
}

#[derive(Default)]
pub(crate) struct Planner {
    calls: FxHashMap<*const CallSite, (Site, CallProfile)>,
    targets: HashMap<Site, u64>,
    observed: HashSet<(u64, Site)>,
    reads: HashMap<(Site, ReadKind), ReadType>,
    observed_reads: HashSet<(u64, Site, ReadKind)>,
    pending: Vec<bool>,
    pending_count: usize,
}

impl Planner {
    pub(crate) fn observe(&mut self, owner: u64, site: Site, target: u64) -> bool {
        if !self.observed.insert((owner, site)) {
            return false;
        }
        self.targets.entry(site).or_insert(target);
        true
    }

    pub(crate) fn observe_get(
        &mut self,
        owner: u64,
        site: Site,
        kind: ReadKind,
        observed: ReadType,
    ) -> bool {
        if !self.observed_reads.insert((owner, site, kind)) {
            return false;
        }
        self.reads.entry((site, kind)).or_insert(observed);
        true
    }

    pub(super) fn reads(&self) -> impl Iterator<Item = ((Site, ReadKind), ReadType)> + '_ {
        self.reads.iter().map(|(&site, &element)| (site, element))
    }

    pub(crate) fn take(&mut self, owner: u64) -> bool {
        let pending = self
            .pending
            .get_mut(owner as usize)
            .is_some_and(|pending| std::mem::replace(pending, false));
        self.pending_count -= usize::from(pending);
        pending
    }

    pub(crate) fn request(&mut self, owner: u64) -> bool {
        let owner = owner as usize;
        self.pending
            .resize(self.pending.len().max(owner + 1), false);
        let requested = !self.pending[owner];
        self.pending_count += usize::from(requested);
        self.pending[owner] = true;
        requested
    }

    pub(crate) fn has_pending(&self) -> bool {
        self.pending_count != 0
    }

    pub(crate) fn record_call(&mut self, site: &CallSite, code_word: u64, count: u64) {
        let (_, profile) = self
            .calls
            .entry(site)
            .or_insert_with(|| (site.site, CallProfile::new()));
        if site.guarded {
            profile.guarded += count;
        } else {
            profile.generic += count;
        }
        match code_word.checked_sub(1) {
            Some(code) => profile.record(code, count),
            None => profile.native += count,
        }
    }

    pub(crate) fn calls(&self) -> Vec<(Site, CallProfile)> {
        let mut merged: HashMap<Site, CallProfile> = HashMap::new();
        for (site, profile) in self.calls.values() {
            merged
                .entry(*site)
                .or_insert_with(CallProfile::new)
                .merge(profile);
        }
        merged.into_iter().collect()
    }

    pub(super) fn targets(&self) -> impl Iterator<Item = (Site, u64)> + '_ {
        self.targets.iter().map(|(&site, &target)| (site, target))
    }
}
