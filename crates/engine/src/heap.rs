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
    cells: HashMap<u64, Box<Cell>>,
    closures: HashMap<u64, Box<Closure>>,
    pending: VecDeque<u64>,
}

struct Array {
    references: usize,
    elements: Vec<Value>,
}

struct Cell {
    references: usize,
    value: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CodeId(pub(crate) u64);

struct Closure {
    references: usize,
    code: CodeId,
    captures: Vec<Value>,
}

impl Heap {
    pub(crate) fn array(&self, elements: &[Value]) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !elements.iter().all(|value| data.valid(*value)) {
            return None;
        }
        let id = next_id()?;
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

    pub(crate) fn cell(&self, value: Value) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = next_id()?;
        if !data.retain(value) {
            return None;
        }
        data.cells.insert(
            id,
            Box::new(Cell {
                references: 1,
                value,
            }),
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn closure(&self, code: CodeId, captures: &[Value]) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !captures.iter().all(|value| {
            heap_id(*value)
                .and_then(|id| data.cells.get(&id))
                .is_some_and(|cell| cell.references != 0)
        }) {
            return None;
        }
        let id = next_id()?;
        for value in captures {
            data.retain(*value);
        }
        data.closures.insert(
            id,
            Box::new(Closure {
                references: 1,
                code,
                captures: captures.to_vec(),
            }),
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    #[cfg(test)]
    pub(crate) fn closure_parts(&self, closure: Value) -> Option<(CodeId, Vec<Value>)> {
        let data = self.0.borrow();
        let closure = data.closures.get(&heap_id(closure)?)?;
        (closure.references != 0).then(|| (closure.code, closure.captures.clone()))
    }

    pub(crate) fn closure_code(&self, closure: Value) -> Option<CodeId> {
        let data = self.0.borrow();
        let closure = data.closures.get(&heap_id(closure)?)?;
        (closure.references != 0).then_some(closure.code)
    }

    pub(crate) fn capture(&self, closure: Value, index: usize) -> Option<Value> {
        let data = self.0.borrow();
        let closure = data.closures.get(&heap_id(closure)?)?;
        if closure.references == 0 {
            return None;
        }
        closure.captures.get(index).copied()
    }

    pub(crate) fn cell_get(&self, cell: Value) -> Option<Value> {
        let data = self.0.borrow();
        let cell = data.cells.get(&heap_id(cell)?)?;
        (cell.references != 0).then_some(cell.value)
    }

    pub(crate) fn cell_set(&self, cell: Value, value: Value) -> bool {
        let mut data = self.0.borrow_mut();
        let Some(id) = heap_id(cell) else {
            return false;
        };
        if !data.cells.get(&id).is_some_and(|cell| cell.references != 0) || !data.retain(value) {
            return false;
        }
        let old = std::mem::replace(&mut data.cells.get_mut(&id).unwrap().value, value);
        data.release(old);
        true
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
        let array = data.arrays.get(&heap_id(array)?)?;
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
        let array = data.arrays.get(&heap_id(array)?)?;
        (array.references != 0).then_some(array.elements.len())
    }

    pub(crate) fn set(&self, array: Value, index: usize, value: Value) -> bool {
        let mut data = self.0.borrow_mut();
        let Some(id) = heap_id(array) else {
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
        let Some(id) = heap_id(array) else {
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
            if let Some(array) = data.arrays.get_mut(&id) {
                if array.references != 0 {
                    continue;
                }
                if let Some(value) = array.elements.pop() {
                    data.pending.push_back(id);
                    data.release(value);
                } else {
                    data.arrays.remove(&id);
                }
            } else if let Some(cell) = data.cells.remove(&id) {
                if cell.references != 0 {
                    data.cells.insert(id, cell);
                    continue;
                }
                data.release(cell.value);
            } else if let Some(closure) = data.closures.get_mut(&id) {
                if closure.references != 0 {
                    continue;
                }
                if let Some(value) = closure.captures.pop() {
                    data.pending.push_back(id);
                    data.release(value);
                } else {
                    data.closures.remove(&id);
                }
            }
            budget -= 1;
        }
    }
}

fn next_id() -> Option<u64> {
    #[allow(deprecated)]
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| {
            (id < (1 << 48)).then_some(id + 1)
        })
        .ok()
}

fn heap_id(value: Value) -> Option<u64> {
    let bits = value.bits();
    (value.is_object() && bits != TAG_OBJECT).then_some(bits & 0x0000_ffff_ffff_ffff)
}

impl HeapData {
    fn references(&mut self, id: u64) -> Option<&mut usize> {
        if let Some(array) = self.arrays.get_mut(&id) {
            return Some(&mut array.references);
        }
        if let Some(cell) = self.cells.get_mut(&id) {
            return Some(&mut cell.references);
        }
        self.closures
            .get_mut(&id)
            .map(|closure| &mut closure.references)
    }

    fn valid(&self, value: Value) -> bool {
        match heap_id(value) {
            Some(id) => {
                self.arrays.get(&id).is_some_and(|v| v.references != 0)
                    || self.cells.get(&id).is_some_and(|v| v.references != 0)
                    || self.closures.get(&id).is_some_and(|v| v.references != 0)
            }
            None => !value.is_string(),
        }
    }

    fn retain(&mut self, value: Value) -> bool {
        match heap_id(value) {
            Some(id) => match self.references(id) {
                Some(references) if *references != 0 => {
                    *references += 1;
                    true
                }
                _ => false,
            },
            None => !value.is_string(),
        }
    }

    fn release(&mut self, value: Value) {
        let Some(id) = heap_id(value) else { return };
        let Some(references) = self.references(id) else {
            return;
        };
        assert!(*references > 0);
        *references -= 1;
        if *references == 0 {
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
    fn closures_share_cells_and_retain_captured_values() {
        let heap = Heap::default();
        let first = heap.array(&[]).unwrap();
        let cell = heap.cell(first).unwrap();
        let left = heap.closure(CodeId(1), &[cell]).unwrap();
        let right = heap.closure(CodeId(2), &[cell]).unwrap();
        heap.release(first);
        heap.release(cell);
        let captured = heap.closure_parts(left).unwrap().1[0];
        let replacement = heap.array(&[]).unwrap();
        assert!(heap.cell_set(captured, replacement));
        heap.release(replacement);
        heap.drain(100);
        assert!(!heap.valid(first));
        assert_eq!(
            heap.cell_get(heap.closure_parts(right).unwrap().1[0]),
            Some(replacement)
        );
        assert!(heap.cell_set(captured, replacement));
        heap.release(left);
        heap.drain(100);
        assert!(heap.valid(replacement));
        heap.release(right);
        heap.drain(100);
        assert!(!heap.valid(captured));
        assert!(!heap.valid(replacement));
    }

    #[test]
    fn closure_disposal_is_budgeted_by_capture_edges() {
        let heap = Heap::default();
        let cells: Vec<_> = (0..100)
            .map(|_| heap.cell(Value::UNDEFINED).unwrap())
            .collect();
        let closure = heap.closure(CodeId(1), &cells).unwrap();
        for cell in &cells {
            heap.release(*cell);
        }
        heap.release(closure);
        heap.drain(1);
        assert_eq!(
            heap.0
                .borrow()
                .closures
                .get(&heap_id(closure).unwrap())
                .unwrap()
                .captures
                .len(),
            99
        );
        assert!(cells.iter().any(|cell| heap.valid(*cell)));
        heap.drain(1000);
        assert!(cells.iter().all(|cell| !heap.valid(*cell)));
        assert!(heap.0.borrow().closures.is_empty());
        assert!(heap.0.borrow().cells.is_empty());
    }

    #[test]
    fn closure_cell_cycle_does_not_keep_heap_alive() {
        let heap = Heap::default();
        let weak = Rc::downgrade(&heap.0);
        let cell = heap.cell(Value::UNDEFINED).unwrap();
        let closure = heap.closure(CodeId(1), &[cell]).unwrap();
        assert!(heap.cell_set(cell, closure));
        heap.release(cell);
        heap.release(closure);
        heap.drain(100);
        assert!(heap.valid(cell));
        assert!(heap.valid(closure));
        drop(heap);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn closure_rejects_invalid_or_foreign_captures() {
        let heap = Heap::default();
        let other = Heap::default();
        let cell = heap.cell(Value::UNDEFINED).unwrap();
        let array = heap.array(&[]).unwrap();
        assert!(heap.closure(CodeId(1), &[array]).is_none());
        assert!(other.closure(CodeId(1), &[cell]).is_none());
        heap.release(cell);
        assert!(heap.closure(CodeId(1), &[cell]).is_none());
        heap.release(array);
    }

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
        assert!(heap.0.borrow().arrays.contains_key(&heap_id(root).unwrap()));
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
