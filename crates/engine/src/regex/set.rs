pub(super) const MAX: u32 = 0x10ffff;
const SURROGATES: (u32, u32) = (0xd800, 0xdfff);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Set(Vec<(u32, u32)>);

impl Set {
    pub(super) fn range(start: u32, end: u32) -> Self {
        Self::from(vec![(start, end)])
    }

    pub(super) fn from(mut ranges: Vec<(u32, u32)>) -> Self {
        ranges.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
        for (start, end) in ranges {
            match merged.last_mut() {
                Some(last) if start <= last.1.saturating_add(1) => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }
        Self(merged)
    }

    pub(super) fn ranges(&self) -> &[(u32, u32)] {
        &self.0
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(super) fn union(&self, other: &Self) -> Self {
        Self::from(self.0.iter().chain(&other.0).copied().collect())
    }

    pub(super) fn complement(&self) -> Self {
        let mut ranges = Vec::new();
        let mut next = 0;
        for &(start, end) in &self.0 {
            if start > next {
                ranges.push((next, start - 1));
            }
            next = end + 1;
        }
        if next <= MAX {
            ranges.push((next, MAX));
        }
        Self(ranges)
    }

    pub(super) fn intersect(&self, other: &Self) -> Self {
        self.complement().union(&other.complement()).complement()
    }

    pub(super) fn minus(&self, other: &Self) -> Self {
        self.intersect(&other.complement())
    }

    pub(super) fn without_surrogates(&self) -> Self {
        self.minus(&Self::range(SURROGATES.0, SURROGATES.1))
    }

    pub(super) fn ascii_case_closure(&self) -> Self {
        let mut ranges = self.0.clone();
        for &(start, end) in &self.0 {
            for (from, to) in [(b'a', b'A'), (b'A', b'a')] {
                let (low, high) = (start.max(from as u32), end.min(from as u32 + 25));
                if low <= high {
                    let shift = |c: u32| c - from as u32 + to as u32;
                    ranges.push((shift(low), shift(high)));
                }
            }
        }
        Self::from(ranges)
    }
}

pub(super) fn is_surrogate(value: u32) -> bool {
    (SURROGATES.0..=SURROGATES.1).contains(&value)
}

pub(super) fn digits() -> Set {
    Set::range('0' as u32, '9' as u32)
}

pub(super) fn word() -> Set {
    Set::from(vec![
        ('0' as u32, '9' as u32),
        ('A' as u32, 'Z' as u32),
        ('_' as u32, '_' as u32),
        ('a' as u32, 'z' as u32),
    ])
}

pub(super) fn line_terminators() -> Set {
    Set::from(vec![(0x0a, 0x0a), (0x0d, 0x0d), (0x2028, 0x2029)])
}

pub(super) fn whitespace() -> Set {
    Set::from(vec![
        (0x09, 0x0d),
        (0x20, 0x20),
        (0xa0, 0xa0),
        (0x1680, 0x1680),
        (0x2000, 0x200a),
        (0x2028, 0x2029),
        (0x202f, 0x202f),
        (0x205f, 0x205f),
        (0x3000, 0x3000),
        (0xfeff, 0xfeff),
    ])
}
