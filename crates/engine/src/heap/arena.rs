use std::ptr::NonNull;

use super::Payload;

pub(crate) const SLOT_SIZE: usize = 128;
const FIRST_CHUNK: usize = 256;
const LARGEST_CHUNK: usize = 1 << 18;
const ADDRESS_LIMIT: usize = 1 << 48;

#[repr(C, align(128))]
pub(super) struct Slot {
    pub(super) references: u32,
    pub(super) kind: u32,
    pub(super) shape: u32,
    // Code id + 1 of a closure that may be called without `new`, otherwise zero, so generated
    // call guards compare one word.
    pub(super) code: u32,
    // Generated code reads elements or shaped property values through these; HeapData::sync
    // refreshes them after any change that can move or resize that storage.
    pub(super) data: *const u8,
    pub(super) length: usize,
    pub(super) payload: Payload,
}

const _: () = assert!(std::mem::size_of::<Slot>() == SLOT_SIZE);

pub(crate) const REFERENCES_OFFSET: i32 = std::mem::offset_of!(Slot, references) as i32;
pub(crate) const KIND_OFFSET: i32 = std::mem::offset_of!(Slot, kind) as i32;
pub(crate) const SHAPE_OFFSET: i32 = std::mem::offset_of!(Slot, shape) as i32;
pub(crate) const CODE_OFFSET: i32 = std::mem::offset_of!(Slot, code) as i32;
pub(crate) const DATA_OFFSET: i32 = std::mem::offset_of!(Slot, data) as i32;
pub(crate) const LENGTH_OFFSET: i32 = std::mem::offset_of!(Slot, length) as i32;

pub(crate) static NO_ELEMENTS: [u64; SLOT_SIZE / 8] = [0; SLOT_SIZE / 8];

impl Slot {
    fn vacant() -> Self {
        Self {
            references: 0,
            kind: 0,
            shape: 0,
            code: 0,
            data: std::ptr::null(),
            length: 0,
            payload: Payload::Vacant,
        }
    }
}

struct Chunk {
    start: NonNull<Slot>,
    capacity: usize,
}

impl Chunk {
    fn address(&self) -> usize {
        self.start.as_ptr() as usize
    }

    fn contains(&self, address: usize) -> bool {
        address >= self.address() && address < self.address() + self.capacity * SLOT_SIZE
    }
}

#[derive(Default)]
pub(super) struct Arena {
    chunks: Vec<Chunk>,
    used: usize,
    free: Vec<NonNull<Slot>>,
}

impl Arena {
    pub(super) fn allocate(&mut self) -> Option<NonNull<Slot>> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        if self
            .chunks
            .last()
            .is_none_or(|chunk| self.used == chunk.capacity)
        {
            self.grow()?;
        }
        let chunk = self.chunks.last().unwrap();
        let slot = unsafe { chunk.start.add(self.used) };
        self.used += 1;
        Some(slot)
    }

    fn grow(&mut self) -> Option<()> {
        let capacity = self
            .chunks
            .last()
            .map_or(FIRST_CHUNK, |chunk| (chunk.capacity * 2).min(LARGEST_CHUNK));
        let slots: Box<[Slot]> = (0..capacity).map(|_| Slot::vacant()).collect();
        let start = NonNull::new(Box::into_raw(slots).cast::<Slot>()).unwrap();
        let chunk = Chunk { start, capacity };
        if chunk.address() + capacity * SLOT_SIZE > ADDRESS_LIMIT {
            unsafe { drop_chunk(&chunk) };
            return None;
        }
        self.chunks.push(chunk);
        self.used = 0;
        Some(())
    }

    pub(super) fn free(&mut self, slot: NonNull<Slot>) {
        self.free.push(slot);
    }

    // Chunks stay allocated for the arena's lifetime, so a stale id still reaches a
    // zero-reference slot; only addresses that never came from this arena are unchecked.
    pub(super) fn get(&self, address: u64) -> NonNull<Slot> {
        debug_assert!(self.owns(address));
        NonNull::new(address as *mut Slot).unwrap()
    }

    pub(super) fn owns(&self, address: u64) -> bool {
        let address = address as usize;
        address % SLOT_SIZE == 0 && self.chunks.iter().any(|chunk| chunk.contains(address))
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        match self.chunks.split_last() {
            Some((_, full)) => full.iter().map(|chunk| chunk.capacity).sum::<usize>() + self.used,
            None => 0,
        }
    }

    #[cfg(test)]
    pub(super) fn slots(&self) -> impl Iterator<Item = &Slot> {
        let last = self.chunks.len().saturating_sub(1);
        self.chunks
            .iter()
            .enumerate()
            .flat_map(move |(index, chunk)| {
                let count = if index == last {
                    self.used
                } else {
                    chunk.capacity
                };
                unsafe { std::slice::from_raw_parts(chunk.start.as_ptr(), count) }.iter()
            })
    }
}

unsafe fn drop_chunk(chunk: &Chunk) {
    drop(unsafe {
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            chunk.start.as_ptr(),
            chunk.capacity,
        ))
    });
}

impl Drop for Arena {
    fn drop(&mut self) {
        for chunk in &self.chunks {
            unsafe { drop_chunk(chunk) };
        }
    }
}
