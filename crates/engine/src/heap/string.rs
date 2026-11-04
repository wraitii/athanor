use super::*;

impl Heap {
    pub(crate) fn interned(&self, text: &'static str) -> Option<Value> {
        let mut data = self.0.borrow_mut();
        let id = if let Some((_, id)) = data.interned.iter().find(|(name, _)| *name == text) {
            *id
        } else {
            let id = data.reserve()?;
            data.store(id, Payload::String(text.into()));
            data.interned.push((text, id));
            id
        };
        data.slot_mut(id).references += 1;
        Some(Value::from_bits(TAG_STRING | id))
    }

    pub(crate) fn string(&self, text: &str) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = data.reserve()?;
        data.store(id, Payload::String(text.into()));
        Some(Value::from_bits(TAG_STRING | id))
    }

    pub(crate) fn next_character(&self, value: Value, offset: usize) -> Option<(char, usize)> {
        if !value.is_string() {
            return None;
        }
        let data = self.0.borrow();
        let text = data.string(heap_id(value)?)?;
        let character = text.get(offset..)?.chars().next()?;
        Some((character, offset + character.len_utf8()))
    }

    pub(crate) fn string_is_empty(&self, value: Value) -> Option<bool> {
        if !value.is_string() {
            return None;
        }
        let data = self.0.borrow();
        let text = data.string(heap_id(value)?)?;
        Some(text.is_empty())
    }

    pub(crate) fn text(&self, value: Value) -> Option<String> {
        if !value.is_string() {
            return None;
        }
        let data = self.0.borrow();
        let text = data.string(heap_id(value)?)?;
        Some(text.clone())
    }
}
