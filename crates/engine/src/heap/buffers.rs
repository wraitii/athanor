const MAX_BUFFER_BYTES: usize = 64 << 10;
const MAX_CACHED_BYTES: usize = 1 << 20;
const MAX_PER_CLASS: usize = 128;
const CLASSES: usize = 17;

pub(super) struct BufferPool<T> {
    classes: [Vec<Vec<T>>; CLASSES],
    bytes: usize,
}

impl<T> Default for BufferPool<T> {
    fn default() -> Self {
        Self {
            classes: std::array::from_fn(|_| Vec::new()),
            bytes: 0,
        }
    }
}

impl<T> BufferPool<T> {
    pub(super) fn take(&mut self, length: usize) -> Option<Vec<T>> {
        if length == 0 {
            return Some(Vec::new());
        }
        let capacity = length.checked_next_power_of_two()?;
        let bytes = capacity.checked_mul(std::mem::size_of::<T>())?;
        let capacity = if bytes <= MAX_BUFFER_BYTES {
            let class = capacity.trailing_zeros() as usize;
            if let Some(buffer) = self.classes[class].pop() {
                self.bytes -= buffer.capacity() * std::mem::size_of::<T>();
                return Some(buffer);
            }
            capacity
        } else {
            length
        };
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(capacity).ok()?;
        Some(buffer)
    }

    pub(super) fn put(&mut self, buffer: Vec<T>) {
        debug_assert!(buffer.is_empty());
        let capacity = buffer.capacity();
        let bytes = capacity.saturating_mul(std::mem::size_of::<T>());
        if capacity == 0 || bytes > MAX_BUFFER_BYTES || self.bytes + bytes > MAX_CACHED_BYTES {
            return;
        }
        // Round down so irregular capacities still satisfy the class's power-of-two request.
        let class = capacity.ilog2() as usize;
        let buffers = &mut self.classes[class];
        if buffers.len() == MAX_PER_CLASS {
            return;
        }
        self.bytes += bytes;
        buffers.push(buffer);
    }
}
