use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::Value;
use crate::heap::{Heap, OwnedValue};

#[derive(Clone)]
pub struct Environment {
    data: Rc<RefCell<EnvironmentData>>,
    heap: Heap,
}

struct EnvironmentData {
    bindings: HashMap<String, Value>,
    parent: Option<Environment>,
    heap: Heap,
}

impl Default for Environment {
    fn default() -> Self {
        let heap = Heap::default();
        Self {
            data: Rc::new(RefCell::new(EnvironmentData {
                bindings: HashMap::new(),
                parent: None,
                heap: heap.clone(),
            })),
            heap,
        }
    }
}

impl Drop for EnvironmentData {
    fn drop(&mut self) {
        for value in self.bindings.values() {
            self.heap.release(*value);
        }
    }
}

impl Environment {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn child(&self) -> Self {
        Self {
            data: Rc::new(RefCell::new(EnvironmentData {
                bindings: HashMap::new(),
                parent: Some(self.clone()),
                heap: self.heap.clone(),
            })),
            heap: self.heap.clone(),
        }
    }

    pub fn define(&self, name: impl Into<String>, value: Value) -> bool {
        use std::collections::hash_map::Entry;
        match self.data.borrow_mut().bindings.entry(name.into()) {
            Entry::Vacant(entry) if self.heap.retain(value) => {
                entry.insert(value);
                true
            }
            _ => false,
        }
    }

    /// Returns an owning handle to the binding, if present.
    pub fn get(&self, name: &str) -> Option<OwnedValue> {
        let value = self.get_raw(name)?;
        self.heap
            .retain(value)
            .then(|| OwnedValue::from_owned(value, self.heap.clone()))
    }

    pub(crate) fn get_raw(&self, name: &str) -> Option<Value> {
        let data = self.data.borrow();
        match data.bindings.get(name) {
            Some(value) => Some(*value),
            None => data.parent.as_ref().and_then(|parent| parent.get_raw(name)),
        }
    }

    pub fn set(&self, name: &str, value: Value) -> bool {
        let mut data = self.data.borrow_mut();
        if let Some(binding) = data.bindings.get_mut(name) {
            if !self.heap.retain(value) {
                return false;
            }
            let old = std::mem::replace(binding, value);
            self.heap.release(old);
            return true;
        }
        data.parent
            .as_ref()
            .is_some_and(|parent| parent.set(name, value))
    }

    pub(crate) fn heap(&self) -> &Heap {
        &self.heap
    }
}
