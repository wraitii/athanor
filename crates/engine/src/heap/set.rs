use super::*;

#[derive(Hash, PartialEq, Eq)]
pub(super) enum SetKey {
    String(String),
    Bits(u64),
}

pub(super) struct Set {
    pub(super) entries: HashMap<SetKey, Value>,
}

impl Heap {
    pub(crate) fn set_new(&self) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = data.reserve()?;
        data.store(id, Payload::Set(Box::new(Set {
            entries: HashMap::default(),
        })));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn set_size(&self, value: Value) -> Option<usize> {
        let data = self.0.borrow();
        let set = data.set(heap_id(value)?)?;
        Some(set.entries.len())
    }

    pub(super) fn set_key(&self, value: Value) -> Option<SetKey> {
        if !self.valid(value) {
            return None;
        }
        Some(if let Some(text) = self.text(value) {
            SetKey::String(text)
        } else if let Some(number) = value.as_number() {
            SetKey::Bits(Value::number(if number == 0.0 { 0.0 } else { number }).bits())
        } else {
            SetKey::Bits(value.bits())
        })
    }

    pub(crate) fn set_has(&self, set: Value, value: Value) -> Option<bool> {
        let key = self.set_key(value)?;
        let data = self.0.borrow();
        let set = data.set(heap_id(set)?)?;
        Some(set.entries.contains_key(&key))
    }

    pub(crate) fn set_add(&self, set: Value, value: Value) -> bool {
        let Some(key) = self.set_key(value) else {
            return false;
        };
        let Some(id) = heap_id(set) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(set) = data.set(id) else {
            return false;
        };
        if set.entries.contains_key(&key) {
            return true;
        }
        if !data.retain(value) {
            return false;
        }
        data.set_mut(id).unwrap().entries.insert(key, value);
        true
    }

    pub(crate) fn set_delete(&self, set: Value, value: Value) -> Option<bool> {
        let key = self.set_key(value)?;
        let mut data = self.0.borrow_mut();
        let set = data.set_mut(heap_id(set)?)?;
        let removed = set.entries.remove(&key);
        if let Some(value) = removed {
            data.release(value);
        }
        Some(removed.is_some())
    }

    pub(crate) fn set_clear(&self, set: Value) -> bool {
        let Some(id) = heap_id(set) else { return false };
        let mut data = self.0.borrow_mut();
        let Some(set) = data.set(id) else {
            return false;
        };
        if set.entries.is_empty() {
            return true;
        }
        let Some(disposal) = data.reserve() else {
            return false;
        };
        let entries = std::mem::take(&mut data.set_mut(id).unwrap().entries);
        data.store(disposal, Payload::Set(Box::new(Set { entries })));
        data.slot_mut(disposal).references = 0;
        data.pending.push_back(disposal);
        true
    }
}
