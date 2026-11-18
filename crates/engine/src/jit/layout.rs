#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ReadLayout {
    pub(crate) shape: u32,
    pub(crate) index: u32,
    pub(crate) storage: Option<RelatedStorage>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct RelatedStorage {
    // A matching receiver layout must keep this storage alive. Its identity alone
    // never permits dereferencing it; the receiver check must succeed first.
    pub(crate) identity: u64,
    pub(crate) shape: u32,
    pub(crate) intermediate: Option<StorageGuard>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct StorageGuard {
    pub(crate) identity: u64,
    pub(crate) shape: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LayoutGuard {
    pub(crate) shape: u32,
    pub(crate) storage: Option<RelatedStorage>,
}

impl ReadLayout {
    pub(crate) const fn own(shape: u32, index: u32) -> Self {
        Self {
            shape,
            index,
            storage: None,
        }
    }
}
