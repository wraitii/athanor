use crate::heap::CodeId;

pub(crate) struct Plan {
    pub(crate) entries: Box<[Entry]>,
}

pub(crate) struct Entry {
    pub(crate) code: CodeId,
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) exits: Box<[Exit]>,
}

#[derive(Clone, Copy)]
pub(crate) enum Exit {
    Complete,
    Suspend(usize),
}
