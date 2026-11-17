use super::*;

pub(super) struct RegExp {
    pub(super) pattern: Rc<crate::regex::Pattern>,
    pub(super) global: bool,
    pub(super) last_index: f64,
}

pub(crate) fn regexp_flags(bits: u8) -> regress::Flags {
    regress::Flags {
        icase: bits & 1 != 0,
        multiline: bits & 2 != 0,
        dot_all: bits & 4 != 0,
        unicode: bits & 8 != 0,
        ..Default::default()
    }
}

fn split_ranges(pattern: &crate::regex::Pattern, text: &str) -> Vec<Option<std::ops::Range<usize>>> {
    if text.is_empty() {
        return match pattern.find(text, 0) {
            Some(_) => Vec::new(),
            None => vec![Some(0..0)],
        };
    }
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut offset = 0;
    while let Some(found) = pattern.captures(text, offset) {
        if found.range.start >= text.len() {
            break;
        }
        if found.range.end == start {
            let Some(character) = text[found.range.start..].chars().next() else {
                break;
            };
            offset = found.range.start + character.len_utf8();
            continue;
        }
        pieces.push(Some(start..found.range.start));
        pieces.extend(found.groups);
        start = found.range.end;
        offset = start;
    }
    pieces.push(Some(start..text.len()));
    pieces
}

impl Heap {
    pub(crate) fn regexp(&self, pattern: &str, flags: u8) -> Result<Value, String> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let key = (pattern.to_owned(), flags);
        let compiled = if let Some(compiled) = data.regexp_cache.get(&key) {
            compiled.clone()
        } else {
            let compiled = Rc::new(crate::regex::Pattern::new(pattern, flags)?);
            data.regexp_cache.insert(key, compiled.clone());
            compiled
        };
        let id = data.reserve().ok_or("heap exhausted")?;
        data.store(
            id,
            Payload::RegExp(Box::new(RegExp {
                pattern: compiled,
                global: crate::regex::Flags::from_bits(flags).global,
                last_index: 0.0,
            })),
        );
        Ok(Value::from_bits(TAG_OBJECT | id))
    }

    // Appended values are owned by the caller, including partial results on failure.
    pub(crate) fn regexp_split_into(
        &self,
        value: Value,
        regexp: Value,
        values: &mut Vec<Value>,
    ) -> Option<()> {
        assert!(values.is_empty());
        let pattern = self.regexp_compiled(regexp)?;
        let pieces = self.with_text(value, |text| split_ranges(&pattern, text))?;
        // The caller keeps the source alive while draining and allocating pieces.
        self.drain(64usize.saturating_mul(pieces.len()));
        let mut data = self.0.borrow_mut();
        let id = heap_id(value)?;
        values.reserve(pieces.len());
        for piece in pieces {
            let Some(range) = piece else {
                values.push(Value::UNDEFINED);
                continue;
            };
            let piece = data.string(id)?.get(range)?.to_owned();
            let piece_id = data.reserve()?;
            data.store_string(piece_id, piece);
            values.push(Value::from_bits(TAG_STRING | piece_id));
        }
        Some(())
    }

    pub(crate) fn regexp_compiled(&self, value: Value) -> Option<Rc<crate::regex::Pattern>> {
        let data = self.0.borrow();
        let regexp = data.regexp(heap_id(value)?)?;
        Some(regexp.pattern.clone())
    }

    pub(crate) fn regexp_state(&self, value: Value) -> Option<(bool, f64)> {
        let data = self.0.borrow();
        let regexp = data.regexp(heap_id(value)?)?;
        Some((regexp.global, regexp.last_index))
    }

    pub(crate) fn regexp_set_last_index(&self, value: Value, index: f64) -> bool {
        let Some(id) = heap_id(value) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(regexp) = data.regexp_mut(id) else {
            return false;
        };
        regexp.last_index = index;
        true
    }
}
