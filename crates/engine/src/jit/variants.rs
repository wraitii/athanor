use crate::FxHashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::entries::{Entry, VariantId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Input {
    This,
    Argument(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Fact {
    Number,
    Integer,
    Shape(u32),
}

pub(crate) type Requirements = Vec<(Input, Fact)>;

// Generated code addresses the first field directly; the request helper receives the same
// address as a Variant pointer. The registry keeps every Arc alive until engine shutdown.
#[repr(C)]
pub(crate) struct Variant {
    pub(crate) entry: AtomicUsize,
    pub(crate) function: u64,
    pub(crate) requirements: Requirements,
}

impl Variant {
    pub(crate) fn id(&self) -> VariantId {
        VariantId::Specialized(&self.entry as *const AtomicUsize as usize)
    }

    pub(crate) fn publish(&self, entry: Entry) {
        self.entry.store(entry.code, Ordering::Release);
    }
}

#[derive(Clone, Default)]
pub(crate) struct Registry(Arc<Mutex<FxHashMap<(u64, Requirements), Arc<Variant>>>>);

impl Registry {
    pub(crate) fn intern(&self, function: u64, requirements: Requirements) -> Option<Arc<Variant>> {
        if requirements.is_empty() {
            return None;
        }
        let mut variants = self.0.lock().unwrap();
        let key = (function, requirements);
        if let Some(variant) = variants.get(&key) {
            return Some(variant.clone());
        }
        if variants.len() >= 128 || variants.keys().filter(|key| key.0 == function).count() >= 4 {
            return None;
        }
        let variant = Arc::new(Variant {
            entry: AtomicUsize::new(0),
            function,
            requirements: key.1.clone(),
        });
        variants.insert(key, variant.clone());
        Some(variant)
    }

    pub(crate) fn for_function(&self, function: u64) -> Vec<Arc<Variant>> {
        self.0
            .lock()
            .unwrap()
            .values()
            .filter(|variant| variant.function == function)
            .cloned()
            .collect()
    }

    pub(crate) fn get(&self, id: VariantId) -> Option<Arc<Variant>> {
        if matches!(id, VariantId::Ordinary(_)) {
            return None;
        }
        self.0
            .lock()
            .unwrap()
            .values()
            .find(|variant| variant.id() == id)
            .cloned()
    }
}
