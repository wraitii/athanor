use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::value::{TAG_OBJECT, Value};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Default)]
pub(crate) struct Heap(Rc<RefCell<HeapData>>);

#[derive(Default)]
struct HeapData {
    arrays: HashMap<u64, Box<Array>>,
    pending: VecDeque<u64>,
}

struct Array {
    references: usize,
    elements: Vec<Value>,
}

impl Heap {
    pub(crate) fn array(&self, elements: &[Value]) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !elements.iter().all(|value| data.valid(*value)) {
            return None;
        }
        #[allow(deprecated)]
        let id = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| {
                (id < (1 << 48)).then_some(id + 1)
            })
            .ok()?;
        for value in elements {
            data.retain(*value);
        }
        data.arrays.insert(
            id,
            Box::new(Array {
                references: 1,
                elements: elements.to_vec(),
            }),
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn valid(&self, value: Value) -> bool {
        if !value.is_object() && !value.is_string() {
            return true;
        }
        self.0.borrow().valid(value)
    }

    pub(crate) fn retain(&self, value: Value) -> bool {
        if !value.is_object() && !value.is_string() {
            return true;
        }
        self.0.borrow_mut().retain(value)
    }

    pub(crate) fn release(&self, value: Value) {
        if !value.is_object() {
            return;
        }
        self.0.borrow_mut().release(value);
    }

    pub(crate) fn get(&self, array: Value, index: usize) -> Option<Value> {
        let data = self.0.borrow();
        let array = data.arrays.get(&array_id(array)?)?;
        if array.references == 0 {
            return None;
        }
        Some(
            array
                .elements
                .get(index)
                .copied()
                .unwrap_or(Value::UNDEFINED),
        )
    }

    pub(crate) fn len(&self, array: Value) -> Option<usize> {
        let data = self.0.borrow();
        let array = data.arrays.get(&array_id(array)?)?;
        (array.references != 0).then_some(array.elements.len())
    }

    pub(crate) fn set(&self, array: Value, index: usize, value: Value) -> bool {
        let mut data = self.0.borrow_mut();
        let Some(id) = array_id(array) else {
            return false;
        };
        let Some(len) = data
            .arrays
            .get(&id)
            .filter(|array| array.references != 0)
            .map(|array| array.elements.len())
        else {
            return false;
        };
        if index > len || !data.retain(value) {
            return false;
        }
        let array = data.arrays.get_mut(&id).unwrap();
        if index == len {
            array.elements.push(value);
        } else {
            let old = std::mem::replace(&mut array.elements[index], value);
            data.release(old);
        }
        true
    }

    pub(crate) fn truncate(&self, array: Value, length: usize) -> bool {
        let mut data = self.0.borrow_mut();
        let Some(id) = array_id(array) else {
            return false;
        };
        let Some(array) = data.arrays.get_mut(&id) else {
            return false;
        };
        if array.references == 0 || length > array.elements.len() {
            return false;
        }
        let removed: Vec<_> = array.elements.drain(length..).collect();
        for value in removed {
            data.release(value);
        }
        true
    }

    pub(crate) fn drain(&self, mut budget: usize) {
        let mut data = self.0.borrow_mut();
        while budget > 0 {
            let Some(id) = data.pending.pop_front() else {
                break;
            };
            let Some(array) = data.arrays.get_mut(&id) else {
                continue;
            };
            if array.references != 0 {
                continue;
            }
            if let Some(value) = array.elements.pop() {
                data.pending.push_back(id);
                data.release(value);
            } else {
                data.arrays.remove(&id);
            }
            budget -= 1;
        }
    }
}

fn array_id(value: Value) -> Option<u64> {
    let bits = value.bits();
    (value.is_object() && bits != TAG_OBJECT).then_some(bits & 0x0000_ffff_ffff_ffff)
}

impl HeapData {
    fn valid(&self, value: Value) -> bool {
        match array_id(value) {
            Some(id) => self
                .arrays
                .get(&id)
                .is_some_and(|array| array.references != 0),
            None => !value.is_string(),
        }
    }

    fn retain(&mut self, value: Value) -> bool {
        match array_id(value) {
            Some(id) => match self.arrays.get_mut(&id) {
                Some(array) if array.references != 0 => {
                    array.references += 1;
                    true
                }
                _ => false,
            },
            None => !value.is_string(),
        }
    }

    fn release(&mut self, value: Value) {
        let Some(id) = array_id(value) else { return };
        let Some(array) = self.arrays.get_mut(&id) else {
            return;
        };
        assert!(array.references > 0);
        array.references -= 1;
        if array.references == 0 {
            self.pending.push_back(id);
        }
    }
}

/// A host-owned value that keeps its runtime and any referenced allocation alive.
pub struct OwnedValue {
    value: Value,
    heap: Heap,
}

impl OwnedValue {
    pub(crate) fn from_owned(value: Value, heap: Heap) -> Self {
        Self { value, heap }
    }
    /// Returns a borrowed token; keep this handle alive while using the token.
    pub fn value(&self) -> Value {
        self.value
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrays_retain_edges_and_replace_safely() {
        let heap = Heap::default();
        let child = heap.array(&[]).unwrap();
        let parent = heap.array(&[child]).unwrap();
        heap.release(child);
        assert!(heap.set(parent, 0, child));
        heap.drain(100);
        assert_eq!(heap.get(parent, 0), Some(child));
        heap.release(parent);
        heap.drain(100);
        assert!(!heap.valid(parent));
        assert!(!heap.valid(child));
    }

    #[test]
    fn destruction_is_iterative_and_budgeted() {
        let heap = Heap::default();
        let mut root = heap.array(&[]).unwrap();
        for _ in 0..10000 {
            let parent = heap.array(&[root]).unwrap();
            heap.release(root);
            root = parent;
        }
        heap.release(root);
        heap.drain(1);
        assert!(!heap.valid(root));
        assert!(!heap.retain(root));
        assert!(
            heap.0
                .borrow()
                .arrays
                .contains_key(&array_id(root).unwrap())
        );
        heap.drain(30000);
        assert!(!heap.valid(root));
        assert!(heap.0.borrow().arrays.is_empty());
    }

    #[test]
    fn host_root_keeps_array_alive_and_rejects_stale_tokens() {
        let heap = Heap::default();
        let value = heap.array(&[]).unwrap();
        let root = OwnedValue::from_owned(value, heap.clone());
        let copy = root.clone();
        drop(root);
        assert!(heap.valid(value));
        drop(copy);
        assert!(!heap.valid(value));
        assert!(!heap.retain(value));
        assert_eq!(heap.get(value, 0), None);
    }

    #[test]
    fn handles_are_checked_across_heaps() {
        let first = Heap::default();
        let second = Heap::default();
        let value = first.array(&[]).unwrap();
        assert!(!second.valid(value));
        assert!(second.array(&[value]).is_none());
    }
}
