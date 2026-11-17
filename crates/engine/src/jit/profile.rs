use crate::jit::layout::ReadLayout;
use crate::{FxHashMap, FxHashSet};

use crate::heap::TypedKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Site {
    pub(super) inline_context: u64,
    pub(super) source: usize,
    pub(super) start: u32,
    pub(super) end: u32,
}

impl std::fmt::Display for Site {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..{}", self.start, self.end)?;
        if self.inline_context != 0 {
            write!(f, " inline={:x}", self.inline_context)?;
        }
        Ok(())
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
    Addition,
    Read(ReadLayout),
    PolymorphicRead { entries: [ReadLayout; 4], len: u8 },
    Overwrite { shape: u32, index: u32 },
    Varying,
    Megamorphic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PropertyAccess {
    Get,
    Set,
}

impl ShapeAccess {
    pub(crate) fn read(self, preferred: Option<u32>) -> Option<ReadLayout> {
        match self {
            Self::Read(layout) => Some(layout),
            Self::PolymorphicRead { entries, len } => entries[..usize::from(len)]
                .iter()
                .copied()
                .find(|layout| Some(layout.shape) == preferred),
            _ => None,
        }
    }

    fn merge(self, other: Self) -> Self {
        let reads = |access| match access {
            Self::Read(layout) => Some(([layout; 4], 1)),
            Self::PolymorphicRead { entries, len } => Some((entries, len)),
            _ => None,
        };
        if let (Some((mut entries, mut len)), Some((incoming, count))) = (reads(self), reads(other))
        {
            for &entry in &incoming[..usize::from(count)] {
                if entries[..usize::from(len)].contains(&entry) {
                    continue;
                }
                if usize::from(len) == entries.len()
                    || entries[..usize::from(len)]
                        .iter()
                        .any(|known| known.shape == entry.shape)
                {
                    return Self::Varying;
                }
                entries[usize::from(len)] = entry;
                len += 1;
            }
            if len > 1 {
                entries[usize::from(len)..].fill(ReadLayout::default());
                entries[..usize::from(len)].sort_unstable();
                return Self::PolymorphicRead { entries, len };
            }
        }
        match (self, other) {
            (Self::Megamorphic, _) | (_, Self::Megamorphic) => Self::Megamorphic,
            _ if self == other => self,
            _ => Self::Varying,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Observation {
    Call,
    Read(ReadKind),
    NumberFailure,
}

pub(crate) struct Update {
    pub(crate) changed: bool,
    pub(crate) complete: bool,
}

pub(crate) const NUMBER_FAILURE_LIMIT: u64 = 8;
pub(crate) const CALL_TARGET_LIMIT: usize = 4;

#[repr(C)]
pub(crate) struct ObservationCache {
    pub(crate) complete: u64,
    pub(crate) last_callee: u64,
    pub(crate) failures: u64,
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
    revision: u64,
    calls: FxHashMap<*const CallSite, (Site, CallProfile)>,
    targets: FxHashMap<Site, Vec<u64>>,
    call_versions: FxHashMap<(u64, Site), usize>,
    natives: FxHashMap<Site, crate::heap::NativeFunction>,
    observed: FxHashSet<(u64, Site)>,
    reads: FxHashMap<(Site, ReadKind), ReadType>,
    observed_reads: FxHashSet<(u64, Site, ReadKind)>,
    shapes: FxHashMap<(Site, PropertyAccess), ShapeAccess>,
    shape_versions: FxHashMap<(u64, Site, PropertyAccess), ShapeAccess>,
    number_first: FxHashSet<u64>,
}

#[derive(Default)]
pub(crate) struct Recompiles {
    pending: Vec<bool>,
    pending_count: usize,
}

impl Profile {
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn observe(&mut self, owner: u64, site: Site, target: u64) -> Update {
        let targets = self.targets.entry(site).or_default();
        if targets.len() < CALL_TARGET_LIMIT && !targets.contains(&target) {
            targets.push(target);
            self.revision += 1;
        }
        let changed =
            self.call_versions.insert((owner, site), targets.len()) != Some(targets.len());
        Update {
            changed,
            complete: targets.len() == CALL_TARGET_LIMIT,
        }
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
        if let std::collections::hash_map::Entry::Vacant(entry) = self.natives.entry(site) {
            entry.insert(target);
            self.revision += 1;
        }
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
        if let std::collections::hash_map::Entry::Vacant(entry) = self.reads.entry((site, kind)) {
            entry.insert(observed);
            self.revision += 1;
        }
        true
    }

    pub(crate) fn observe_number_failure(
        &mut self,
        owner: u64,
        site: Site,
        observed: ReadType,
    ) -> Update {
        let key = (site, ReadKind::Number);
        let known = self.reads.get(&key).copied();
        let policy_changed = (observed == ReadType::Number || known == Some(ReadType::Integer))
            && self.number_first.insert(owner);
        let changed = known != Some(ReadType::Unknown) && known != Some(observed);
        if changed {
            self.reads.insert(key, observed);
        }
        if changed || policy_changed {
            self.revision += 1;
        }
        Update {
            changed: changed || policy_changed,
            complete: self.reads.get(&key) == Some(&ReadType::Unknown),
        }
    }

    pub(super) fn number_first(&self) -> impl Iterator<Item = u64> + '_ {
        self.number_first.iter().copied()
    }

    pub(crate) fn observe_shape(
        &mut self,
        owner: u64,
        site: Site,
        property: PropertyAccess,
        access: ShapeAccess,
    ) -> Option<ShapeAccess> {
        let known = self.shapes.get(&(site, property)).copied();
        let merged = known.map_or(access, |known| known.merge(access));
        let changed = self.shape_versions.insert((owner, site, property), merged) != Some(merged);
        if known != Some(merged) {
            self.shapes.insert((site, property), merged);
            self.revision += 1;
        }
        (changed && (known.is_some() || merged != ShapeAccess::Varying)).then_some(merged)
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
        let mut merged: FxHashMap<Site, CallProfile> = FxHashMap::default();
        for (site, profile) in self.calls.values() {
            merged
                .entry(*site)
                .or_insert_with(CallProfile::new)
                .merge(profile);
        }
        merged.into_iter().collect()
    }

    pub(super) fn targets(&self) -> impl Iterator<Item = (Site, &[u64])> + '_ {
        self.targets
            .iter()
            .map(|(&site, targets)| (site, targets.as_slice()))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn site() -> Site {
        Site {
            inline_context: 0,
            source: 1,
            start: 0,
            end: 10,
        }
    }

    #[test]
    fn call_feedback_grows_to_a_bound_and_updates_each_owner() {
        let mut profile = Profile::default();
        assert!(profile.observe(1, site(), 10).changed);
        assert!(!profile.observe(1, site(), 10).changed);
        assert!(profile.observe(2, site(), 10).changed);
        for target in 11..14 {
            assert!(profile.observe(1, site(), target).changed);
        }
        let overflow = profile.observe(1, site(), 14);
        assert!(!overflow.changed);
        assert!(overflow.complete);
        let other_owner = profile.observe(2, site(), 14);
        assert!(other_owner.changed);
        assert!(other_owner.complete);
        assert!(!profile.observe(2, site(), 14).changed);
        assert_eq!(profile.targets().next().unwrap().1, &[10, 11, 12, 13]);
    }

    #[test]
    fn number_feedback_widens_and_stops_at_unknown() {
        let mut profile = Profile::default();
        profile.observe_get(1, site(), ReadKind::Number, ReadType::Integer);
        let widened = profile.observe_number_failure(1, site(), ReadType::Number);
        assert!(widened.changed);
        assert!(!widened.complete);
        assert!(
            !profile
                .observe_number_failure(1, site(), ReadType::Number)
                .changed
        );
        let disabled = profile.observe_number_failure(1, site(), ReadType::Unknown);
        assert!(disabled.changed);
        assert!(disabled.complete);
        let repeated = profile.observe_number_failure(1, site(), ReadType::Number);
        assert!(!repeated.changed);
        assert!(repeated.complete);
        assert_eq!(profile.reads().next().unwrap().1, ReadType::Unknown);
    }

    #[test]
    fn related_read_feedback_preserves_storage_and_rejects_conflicts() {
        use crate::jit::layout::RelatedStorage;
        let inherited = ReadLayout {
            shape: 10,
            index: 2,
            storage: Some(RelatedStorage {
                identity: 128,
                shape: 20,
            }),
        };
        let mut profile = Profile::default();
        assert_eq!(
            profile.observe_shape(1, site(), PropertyAccess::Get, ShapeAccess::Read(inherited)),
            Some(ShapeAccess::Read(inherited))
        );
        assert_eq!(
            profile.observe_shape(1, site(), PropertyAccess::Get, ShapeAccess::Read(inherited)),
            None
        );
        let own = ReadLayout::own(30, 1);
        profile.observe_shape(1, site(), PropertyAccess::Get, ShapeAccess::Read(own));
        let access = profile.shapes().next().unwrap().1;
        assert_eq!(access.read(Some(10)), Some(inherited));
        assert_eq!(access.read(Some(30)), Some(own));
        assert_eq!(access.read(None), None);
        let changed = ReadLayout {
            storage: Some(RelatedStorage {
                identity: 128,
                shape: 21,
            }),
            ..inherited
        };
        assert_eq!(
            profile.observe_shape(1, site(), PropertyAccess::Get, ShapeAccess::Read(changed)),
            Some(ShapeAccess::Varying)
        );
    }

    #[test]
    fn own_read_feedback_retains_bounded_shape_alternatives() {
        let mut profile = Profile::default();
        for shape in [30, 10, 20, 40] {
            assert!(
                profile
                    .observe_shape(
                        1,
                        site(),
                        PropertyAccess::Get,
                        ShapeAccess::Read(ReadLayout::own(shape, shape / 10))
                    )
                    .is_some()
            );
        }
        for shape in [10, 20, 30, 40] {
            assert!(
                profile
                    .observe_shape(
                        1,
                        site(),
                        PropertyAccess::Get,
                        ShapeAccess::Read(ReadLayout::own(shape, shape / 10))
                    )
                    .is_none()
            );
            let access = profile.shapes().next().unwrap().1;
            assert_eq!(
                access.read(Some(shape)),
                Some(ReadLayout::own(shape, shape / 10))
            );
            assert_eq!(access.read(None), None);
        }
        let fifth = ShapeAccess::Read(ReadLayout::own(50, 5));
        assert_eq!(
            profile.observe_shape(1, site(), PropertyAccess::Get, fifth),
            Some(ShapeAccess::Varying)
        );
        assert!(
            profile
                .observe_shape(1, site(), PropertyAccess::Get, fifth)
                .is_none()
        );
        assert_eq!(
            profile.observe_shape(1, site(), PropertyAccess::Get, ShapeAccess::Megamorphic),
            Some(ShapeAccess::Megamorphic)
        );
        assert!(
            profile
                .observe_shape(1, site(), PropertyAccess::Get, fifth)
                .is_none()
        );
    }
}
