use super::*;

/// A host-owned value that keeps its runtime and any referenced allocation alive.
pub struct OwnedValue {
    pub(super) value: Value,
    pub(super) heap: Heap,
}

impl OwnedValue {
    pub(crate) fn from_owned(value: Value, heap: Heap) -> Self {
        Self { value, heap }
    }
    /// Returns a borrowed token; keep this handle alive and use the token only with its engine.
    pub fn value(&self) -> Value {
        self.value
    }

    pub fn text(&self) -> Option<String> {
        self.heap.text(self.value)
    }
}

impl Clone for OwnedValue {
    fn clone(&self) -> Self {
        assert!(self.heap.retain(self.value));
        Self {
            value: self.value,
            heap: self.heap.clone(),
        }
    }
}

impl Drop for OwnedValue {
    fn drop(&mut self) {
        self.heap.release(self.value);
        self.heap.drain(64);
    }
}
