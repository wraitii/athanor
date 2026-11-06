use std::mem::MaybeUninit;

/// Caches and records owned by one compiled function. Generated code loads `base` once at
/// entry and addresses entries relative to it, so nothing may be pushed after `seal`.
pub(crate) struct FunctionData {
    base: *mut Word,
    words: Vec<Word>,
    drops: Vec<(usize, unsafe fn(*mut Word))>,
    pinned: Vec<Box<dyn std::any::Any>>,
}

type Word = MaybeUninit<u64>;

pub(crate) const FUNCTION_DATA_BASE_OFFSET: i32 = std::mem::offset_of!(FunctionData, base) as i32;

impl FunctionData {
    pub(crate) fn new() -> Box<Self> {
        Box::new(Self {
            base: std::ptr::null_mut(),
            words: Vec::new(),
            drops: Vec::new(),
            pinned: Vec::new(),
        })
    }

    pub(crate) fn push<T>(&mut self, value: T) -> i64 {
        assert!(std::mem::align_of::<T>() <= std::mem::align_of::<u64>());
        assert!(self.base.is_null());
        let offset = self.words.len();
        let words = std::mem::size_of::<T>().div_ceil(std::mem::size_of::<u64>());
        self.words.resize(offset + words, MaybeUninit::zeroed());
        unsafe { self.words.as_mut_ptr().add(offset).cast::<T>().write(value) };
        if std::mem::needs_drop::<T>() {
            self.drops.push((offset, drop_in_place::<T>));
        }
        (offset * std::mem::size_of::<u64>()) as i64
    }

    /// For records referenced at hot call sites: an embedded constant is cheaper there than
    /// keeping `base` live in a register.
    pub(crate) fn pin<T: 'static>(&mut self, value: T) -> *const T {
        let value = Box::new(value);
        let address = &raw const *value;
        self.pinned.push(value);
        address
    }

    pub(crate) fn seal(&mut self) {
        self.base = self.words.as_mut_ptr();
    }

    #[cfg(test)]
    pub(crate) fn pinned<T: 'static>(&self) -> impl Iterator<Item = &T> {
        self.pinned
            .iter()
            .filter_map(|record| record.downcast_ref())
    }
}

unsafe fn drop_in_place<T>(value: *mut Word) {
    unsafe { std::ptr::drop_in_place(value.cast::<T>()) };
}

impl Drop for FunctionData {
    fn drop(&mut self) {
        for &(offset, drop) in &self.drops {
            unsafe { drop(self.words.as_mut_ptr().add(offset)) };
        }
    }
}
