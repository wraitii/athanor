use super::*;

impl Heap {
    pub(crate) fn native_function(&self, function: NativeFunction) -> Option<Value> {
        let mut data = self.0.borrow_mut();
        let id = if let Some((_, id)) = data.natives.iter().find(|(kind, _)| *kind == function) {
            *id
        } else {
            let id = data.reserve()?;
            data.store(id, Payload::Native(function));
            data.natives.push((function, id));
            id
        };
        data.slot_mut(id).references += 1;
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn native(&self, value: Value) -> Option<NativeFunction> {
        let data = self.0.borrow();
        let slot = data.live(heap_id(value)?)?;
        match &slot.payload {
            Payload::Native(function) => Some(*function),
            _ => None,
        }
    }
}
