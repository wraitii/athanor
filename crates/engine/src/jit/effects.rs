#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Effects {
    pub(super) memory: Memory,
    pub(super) control: Control,
}

impl Effects {
    pub(super) const UNKNOWN: Self = Self {
        memory: Memory::UNKNOWN,
        control: Control::UNKNOWN,
    };

    pub(super) fn of(flags: u16) -> Self {
        Self {
            memory: Memory::of(flags),
            control: Control::of(flags),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Assumption {
    NoAccessors,
    NoCoercionCallbacks,
    DirectIteration,
    FixedTypedStorage,
}

impl std::fmt::Display for Assumption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NoAccessors => "property access does not invoke accessors",
            Self::NoCoercionCallbacks => "coercion does not invoke JS callbacks",
            Self::DirectIteration => "iteration and enumeration use direct engine state",
            Self::FixedTypedStorage => "typed array storage never moves or resizes",
        })
    }
}

pub(super) const HEAP_READ: u16 = 1;
pub(super) const HEAP_WRITE: u16 = 2;
pub(super) const GLOBAL_READ: u16 = 4;
pub(super) const GLOBAL_WRITE: u16 = 8;
pub(super) const INTERNAL: u16 = 16;
pub(super) const ALLOCATION: u16 = 32;
pub(super) const ERROR: u16 = 64;
pub(super) const CODE_READ: u16 = 128;
pub(super) const CODE_WRITE: u16 = 256;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct Locations(u8);

impl Locations {
    pub(super) const HEAP: Self = Self(1);
    pub(super) const GLOBAL: Self = Self(2);
    pub(super) const CODE: Self = Self(4);
    pub(super) const ALL: Self = Self::HEAP.union(Self::GLOBAL).union(Self::CODE);

    pub(super) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub(super) fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub(super) fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for Locations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names = [
            (Self::HEAP, "heap"),
            (Self::GLOBAL, "global"),
            (Self::CODE, "code"),
        ]
        .into_iter()
        .filter(|(location, _)| self.intersects(*location))
        .map(|(_, name)| name)
        .collect::<Vec<_>>();
        if names.is_empty() {
            return f.write_str("-");
        }
        f.write_str(&names.join("|"))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Memory {
    pub(super) reads: Locations,
    pub(super) writes: Locations,
    pub(super) unbounded: bool,
}

impl Memory {
    pub(super) const UNKNOWN: Self = Self {
        reads: Locations::ALL,
        writes: Locations::ALL,
        unbounded: true,
    };

    pub(super) fn of(flags: u16) -> Self {
        let locations = |heap, global, code| {
            let mut locations = Locations::default();
            if flags & heap != 0 {
                locations = locations.union(Locations::HEAP);
            }
            if flags & global != 0 {
                locations = locations.union(Locations::GLOBAL);
            }
            if flags & code != 0 {
                locations = locations.union(Locations::CODE);
            }
            locations
        };
        Self {
            reads: locations(HEAP_READ, GLOBAL_READ, CODE_READ),
            writes: locations(HEAP_WRITE, GLOBAL_WRITE, CODE_WRITE),
            unbounded: false,
        }
    }

    pub(super) fn union(self, other: Self) -> Self {
        Self {
            reads: self.reads.union(other.reads),
            writes: self.writes.union(other.writes),
            unbounded: self.unbounded || other.unbounded,
        }
    }

    pub(super) fn reenters(self) -> bool {
        self.unbounded
    }

    pub(super) fn writes_data(self) -> bool {
        self.writes
            .intersects(Locations::HEAP.union(Locations::GLOBAL))
    }

    pub(super) fn writes_state(self) -> bool {
        !self.writes.is_empty()
    }

    pub(super) fn is_empty(self) -> bool {
        self.reads.is_empty() && self.writes.is_empty()
    }
}

impl std::fmt::Display for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reads={} writes={}", self.reads, self.writes)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Control {
    pub(super) throws: bool,
    pub(super) ordered: bool,
}

impl Control {
    pub(super) const UNKNOWN: Self = Self {
        throws: true,
        ordered: true,
    };

    pub(super) fn of(flags: u16) -> Self {
        Self {
            throws: flags & ERROR != 0,
            ordered: flags & (INTERNAL | ALLOCATION) != 0,
        }
    }

    pub(super) fn union(self, other: Self) -> Self {
        Self {
            throws: self.throws || other.throws,
            ordered: self.ordered || other.ordered,
        }
    }

    pub(super) fn is_empty(self) -> bool {
        !self.throws && !self.ordered
    }
}

impl std::fmt::Display for Control {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "throws={} ordered={}", self.throws, self.ordered)
    }
}
