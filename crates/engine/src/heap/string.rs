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

    // Appended values are owned by the caller, including partial results on failure.
    pub(crate) fn split_string_into(
        &self,
        value: Value,
        separator: Option<Value>,
        values: &mut Vec<Value>,
    ) -> Option<()> {
        assert!(values.is_empty());
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = heap_id(value)?;
        let text = data.string(id)?;
        let ranges = if let Some(separator) = separator {
            let separator = data.string(heap_id(separator)?)?;
            if separator.is_empty() {
                text.char_indices()
                    .map(|(start, character)| start..start + character.len_utf8())
                    .collect::<Vec<_>>()
            } else if text.contains(separator) {
                let mut start = 0;
                let mut ranges = Vec::new();
                for (end, _) in text.match_indices(separator) {
                    ranges.push(start..end);
                    start = end + separator.len();
                }
                ranges.push(start..text.len());
                ranges
            } else {
                data.retain(value);
                values.push(value);
                return Some(());
            }
        } else {
            data.retain(value);
            values.push(value);
            return Some(());
        };
        // The caller keeps the source alive while draining and allocating pieces.
        drop(data);
        self.drain(64usize.saturating_mul(ranges.len()));
        let mut data = self.0.borrow_mut();
        values.reserve(ranges.len());
        for range in ranges {
            let piece_id = data.reserve()?;
            let piece = data.string(id).unwrap()[range].to_owned();
            data.store(piece_id, Payload::String(piece));
            values.push(Value::from_bits(TAG_STRING | piece_id));
        }
        Some(())
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

    pub(crate) fn compare_strings(&self, left: Value, right: Value) -> std::cmp::Ordering {
        let data = self.0.borrow();
        let left = data.string(heap_id(left).unwrap()).unwrap();
        let right = data.string(heap_id(right).unwrap()).unwrap();
        left.cmp(right)
    }

    pub(crate) fn strings_equal(&self, left: Value, right: Value) -> bool {
        let data = self.0.borrow();
        let left = heap_id(left).and_then(|id| data.string(id));
        let right = heap_id(right).and_then(|id| data.string(id));
        left == right
    }
}
