use crate::{FxHashMap, FxHashSet};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::Value;
use crate::heap::{Heap, OwnedValue};

#[derive(Clone)]
pub struct Environment {
    data: Rc<RefCell<EnvironmentData>>,
    heap: Heap,
    revision: Rc<Cell<u64>>,
}

struct EnvironmentData {
    // JIT binding locations must survive map growth; Cell permits reads through cached pointers.
    bindings: FxHashMap<String, Box<Cell<Value>>>,
    lexical: FxHashMap<String, (Box<Cell<Value>>, bool)>,
    readonly: FxHashSet<String>,
    parent: Option<Environment>,
    heap: Heap,
}

impl EnvironmentData {
    fn binding(&self, name: &str) -> Option<&Cell<Value>> {
        self.lexical
            .get(name)
            .map(|binding| binding.0.as_ref())
            .or_else(|| self.bindings.get(name).map(Box::as_ref))
    }
}

impl Default for Environment {
    fn default() -> Self {
        let heap = Heap::default();
        Self {
            data: Rc::new(RefCell::new(EnvironmentData {
                bindings: FxHashMap::default(),
                lexical: FxHashMap::default(),
                readonly: FxHashSet::default(),
                parent: None,
                heap: heap.clone(),
            })),
            heap,
            revision: Rc::new(Cell::new(0)),
        }
    }
}

impl Drop for EnvironmentData {
    fn drop(&mut self) {
        for value in self.bindings.values() {
            self.heap.release(value.get());
        }
        for (value, _) in self.lexical.values() {
            self.heap.release(value.get());
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
                bindings: FxHashMap::default(),
                lexical: FxHashMap::default(),
                readonly: FxHashSet::default(),
                parent: Some(self.clone()),
                heap: self.heap.clone(),
            })),
            heap: self.heap.clone(),
            revision: self.revision.clone(),
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
                entry.insert(Box::new(Cell::new(value)));
                self.invalidate_bindings();
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
        match data.binding(name) {
            Some(value) => Some(value.get()),
            None => data.parent.as_ref().and_then(|parent| parent.get_raw(name)),
        }
    }

    pub(crate) fn identity(&self) -> *const () {
        Rc::as_ptr(&self.data).cast()
    }

    pub(crate) fn revision_ptr(&self) -> *const u64 {
        self.revision.as_ptr()
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision.get()
    }

    fn invalidate_bindings(&self) {
        // Descendants share this revision because declarations can shadow inherited bindings.
        self.revision.set(self.revision.get().wrapping_add(1));
    }

    pub(crate) fn binding_slot(&self, name: &str) -> Option<*const Value> {
        let data = self.data.borrow();
        data.binding(name)
            .map(|value| value.as_ptr().cast_const())
            .or_else(|| {
                data.parent
                    .as_ref()
                    .and_then(|parent| parent.binding_slot(name))
            })
    }

    pub(crate) fn get_object_raw(&self, name: &str) -> Option<Value> {
        let data = self.data.borrow();
        data.bindings
            .get(name)
            .map(|value| value.get())
            .or_else(|| {
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
        self.invalidate_bindings();
        if let Some((old, _)) = self
            .data
            .borrow_mut()
            .lexical
            .insert(name, (Box::new(Cell::new(Value::UNINITIALIZED)), mutable))
        {
            self.heap.release(old.get());
        }
    }

    pub(crate) fn initialize_lexical(&self, name: &str, value: Value) -> bool {
        let mut data = self.data.borrow_mut();
        let Some(binding) = data.lexical.get_mut(name) else {
            return false;
        };
        if binding.0.get() != Value::UNINITIALIZED || !self.heap.retain(value) {
            return false;
        }
        binding.0.set(value);
        true
    }

    pub(crate) fn set_binding(&self, name: &str, value: Value) -> Result<(), &'static str> {
        let mut data = self.data.borrow_mut();
        if let Some(binding) = data.lexical.get_mut(name) {
            if binding.0.get() == Value::UNINITIALIZED {
                return Err("accessed before initialization");
            }
            if !binding.1 {
                return Err("is constant");
            }
            if !self.heap.retain(value) {
                return Err("has invalid value");
            }
            let old = binding.0.replace(value);
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
            let old = binding.replace(value);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_locations_survive_growth_and_value_replacement() {
        let environment = Environment::new();
        assert!(environment.define("value", Value::number(1.0)));
        let slot = environment.binding_slot("value").unwrap();
        for i in 0..1000 {
            assert!(environment.define(format!("other{i}"), Value::number(i as f64)));
        }
        assert_eq!(environment.binding_slot("value"), Some(slot));
        let revision = environment.revision();
        assert!(environment.set("value", Value::number(2.0)));
        assert_eq!(environment.revision(), revision);
        assert_eq!(unsafe { *slot }, Value::number(2.0));
    }

    #[test]
    fn resolution_revision_tracks_parent_and_lexical_shadowing() {
        let parent = Environment::new();
        assert!(parent.define("value", Value::number(1.0)));
        let child = parent.child();
        assert_eq!(child.binding_slot("value"), parent.binding_slot("value"));
        let revision = child.revision();
        parent.define_lexical("value".into(), true);
        assert_ne!(child.revision(), revision);
        assert_eq!(child.get_raw("value"), Some(Value::UNINITIALIZED));
        let slot = child.binding_slot("value").unwrap();
        let revision = child.revision();
        assert!(parent.initialize_lexical("value", Value::number(2.0)));
        assert_eq!(child.revision(), revision);
        assert_eq!(unsafe { *slot }, Value::number(2.0));
        assert!(child.define("value", Value::number(3.0)));
        assert_ne!(child.revision(), revision);
        assert_ne!(child.binding_slot("value"), Some(slot));
        assert_eq!(child.get_raw("value"), Some(Value::number(3.0)));
    }
}
