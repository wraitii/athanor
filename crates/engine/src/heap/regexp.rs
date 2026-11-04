use super::*;

pub(super) struct RegExp {
    pub(super) regex: Rc<regress::Regex>,
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

impl Heap {
    pub(crate) fn regexp(&self, pattern: &str, flags: u8) -> Result<Value, String> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let key = (pattern.to_owned(), flags);
        let regex = if let Some(regex) = data.regexp_cache.get(&key) {
            regex.clone()
        } else {
            let regex = Rc::new(
                regress::Regex::with_flags(pattern, regexp_flags(flags))
                    .map_err(|error| error.to_string())?,
            );
            data.regexp_cache.insert(key, regex.clone());
            regex
        };
        let id = data.reserve().ok_or("heap exhausted")?;
        data.store(id, Payload::RegExp(Box::new(RegExp {
            regex,
            global: flags & 16 != 0,
            last_index: 0.0,
        })));
        Ok(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn regexp_compiled(&self, value: Value) -> Option<Rc<regress::Regex>> {
        let data = self.0.borrow();
        let regexp = data.regexp(heap_id(value)?)?;
        Some(regexp.regex.clone())
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
