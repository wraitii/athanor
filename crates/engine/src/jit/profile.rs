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
    Number,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadType {
    Unknown,
    Integer,
    Number,
    Array,
    Typed(TypedKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShapeAccess {
    Read { shape: u32, index: u32 },
    Overwrite { shape: u32, index: u32 },
    Varying,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PropertyAccess {
    Get,
    Set,
}

impl ShapeAccess {
    fn merge(self, other: Self) -> Self {
        if self == other { self } else { Self::Varying }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Observation {
    Call,
    Read(ReadKind),
    NumberFailure,
}

pub(crate) const NUMBER_FAILURE_LIMIT: u64 = 8;

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
pub(crate) struct Profile {
    calls: FxHashMap<*const CallSite, (Site, CallProfile)>,
    targets: HashMap<Site, u64>,
    natives: HashMap<Site, crate::heap::NativeFunction>,
    observed: HashSet<(u64, Site)>,
    reads: HashMap<(Site, ReadKind), ReadType>,
    observed_reads: HashSet<(u64, Site, ReadKind)>,
    shapes: HashMap<(Site, PropertyAccess), ShapeAccess>,
    number_first: HashSet<u64>,
}

#[derive(Default)]
pub(crate) struct Recompiles {
    pending: Vec<bool>,
    pending_count: usize,
}

impl Profile {
    pub(crate) fn observe(&mut self, owner: u64, site: Site, target: u64) -> bool {
        if !self.observed.insert((owner, site)) {
            return false;
        }
        self.targets.entry(site).or_insert(target);
        true
    }

    pub(crate) fn observe_native(
        &mut self,
        owner: u64,
        site: Site,
        target: crate::heap::NativeFunction,
    ) -> bool {
        if !self.observed.insert((owner, site)) {
            return false;
        }
        self.natives.entry(site).or_insert(target);
        true
    }

    pub(super) fn natives(&self) -> impl Iterator<Item = (Site, crate::heap::NativeFunction)> + '_ {
        self.natives.iter().map(|(&site, &target)| (site, target))
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

    pub(crate) fn observe_number_failure(
        &mut self,
        owner: u64,
        site: Site,
        observed: ReadType,
    ) -> bool {
        let key = (site, ReadKind::Number);
        let known = self.reads.get(&key).copied();
        let policy_changed = (observed == ReadType::Number || known == Some(ReadType::Integer))
            && self.number_first.insert(owner);
        if known == Some(ReadType::Unknown) || known == Some(observed) {
            return policy_changed;
        }
        self.reads.insert(key, observed);
        true
    }

    pub(super) fn number_first(&self) -> impl Iterator<Item = u64> + '_ {
        self.number_first.iter().copied()
    }

    pub(crate) fn observe_shape(
        &mut self,
        site: Site,
        property: PropertyAccess,
        access: ShapeAccess,
    ) -> Option<ShapeAccess> {
        let known = self.shapes.get(&(site, property)).copied();
        let merged = known.map_or(access, |known| known.merge(access));
        if known == Some(merged) {
            return None;
        }
        self.shapes.insert((site, property), merged);
        (known.is_some() || merged != ShapeAccess::Varying).then_some(merged)
    }

    pub(super) fn shapes(
        &self,
    ) -> impl Iterator<Item = ((Site, PropertyAccess), ShapeAccess)> + '_ {
        self.shapes.iter().map(|(&site, &access)| (site, access))
    }

    pub(super) fn reads(&self) -> impl Iterator<Item = ((Site, ReadKind), ReadType)> + '_ {
        self.reads.iter().map(|(&site, &element)| (site, element))
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

impl Recompiles {
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
}
