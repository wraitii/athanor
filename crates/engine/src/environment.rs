use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cell::RefCell;
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
    lexical: HashMap<String, (Value, bool)>,
    readonly: HashSet<String>,
    parent: Option<Environment>,
    heap: Heap,
}

impl Default for Environment {
    fn default() -> Self {
        let heap = Heap::default();
        Self {
            data: Rc::new(RefCell::new(EnvironmentData {
                bindings: HashMap::default(),
                lexical: HashMap::default(),
                readonly: HashSet::default(),
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
        for (value, _) in self.lexical.values() {
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
                bindings: HashMap::default(),
                lexical: HashMap::default(),
                readonly: HashSet::default(),
                parent: Some(self.clone()),
                heap: self.heap.clone(),
            })),
            heap: self.heap.clone(),
        }
    }

    /// Defines a binding, retaining its value. Rejects values outside this runtime's live heap.
    /// Heap tokens must satisfy the lifetime and runtime requirements of [`Value`].
    pub fn define(&self, name: impl Into<String>, value: Value) -> bool {
        use std::collections::hash_map::Entry;
        if !self.heap.owns(value) {
            return false;
        }
        match self.data.borrow_mut().bindings.entry(name.into()) {
            Entry::Vacant(entry) if self.heap.retain(value) => {
                entry.insert(value);
                self.heap.invalidate();
                true
            }
            _ => false,
        }
    }

    pub(crate) fn define_readonly(&self, name: &str, value: Value) -> bool {
        if !self.define(name, value) {
            return false;
        }
        self.data.borrow_mut().readonly.insert(name.to_owned());
        true
    }

    fn is_readonly(&self, name: &str) -> bool {
        let data = self.data.borrow();
        if data.bindings.contains_key(name) {
            return data.readonly.contains(name);
        }
        data.parent
            .as_ref()
            .is_some_and(|parent| parent.is_readonly(name))
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
        match data
            .lexical
            .get(name)
            .map(|binding| &binding.0)
            .or_else(|| data.bindings.get(name))
        {
            Some(value) => Some(*value),
            None => data.parent.as_ref().and_then(|parent| parent.get_raw(name)),
        }
    }

    pub(crate) fn get_object_raw(&self, name: &str) -> Option<Value> {
        let data = self.data.borrow();
        data.bindings.get(name).copied().or_else(|| {
            data.parent
                .as_ref()
                .and_then(|parent| parent.get_object_raw(name))
        })
    }

    pub(crate) fn has_lexical(&self, name: &str) -> bool {
        self.data.borrow().lexical.contains_key(name)
    }

    pub(crate) fn define_lexical(&self, name: String, mutable: bool) {
        self.heap.invalidate();
        self.data
            .borrow_mut()
            .lexical
            .insert(name, (Value::UNINITIALIZED, mutable));
    }

    pub(crate) fn initialize_lexical(&self, name: &str, value: Value) -> bool {
        let mut data = self.data.borrow_mut();
        let Some(binding) = data.lexical.get_mut(name) else {
            return false;
        };
        if binding.0 != Value::UNINITIALIZED || !self.heap.retain(value) {
            return false;
        }
        binding.0 = value;
        true
    }

    pub(crate) fn set_binding(&self, name: &str, value: Value) -> Result<(), &'static str> {
        let mut data = self.data.borrow_mut();
        if let Some(binding) = data.lexical.get_mut(name) {
            if binding.0 == Value::UNINITIALIZED {
                return Err("accessed before initialization");
            }
            if !binding.1 {
                return Err("is constant");
            }
            if !self.heap.retain(value) {
                return Err("has invalid value");
            }
            let old = std::mem::replace(&mut binding.0, value);
            self.replaced(old);
            return Ok(());
        }
        drop(data);
        if self.set(name, value) {
            Ok(())
        } else if self.is_readonly(name) {
            Err("is constant")
        } else {
            Err("is unknown")
        }
    }

    /// Replaces a binding, retaining its value. Rejects values outside this runtime's live heap.
    /// Heap tokens must satisfy the lifetime and runtime requirements of [`Value`].
    pub fn set(&self, name: &str, value: Value) -> bool {
        if !self.heap.owns(value) {
            return false;
        }
        let mut data = self.data.borrow_mut();
        if data.readonly.contains(name) {
            return false;
        }
        if let Some(binding) = data.bindings.get_mut(name) {
            if !self.heap.retain(value) {
                return false;
            }
            let old = std::mem::replace(binding, value);
            self.replaced(old);
            return true;
        }
        data.parent
            .as_ref()
            .is_some_and(|parent| parent.set(name, value))
    }

    fn replaced(&self, old: Value) {
        if old.is_object() {
            self.heap.invalidate();
        }
        self.heap.release(old);
    }

    pub(crate) fn heap(&self) -> &Heap {
        &self.heap
    }
}
