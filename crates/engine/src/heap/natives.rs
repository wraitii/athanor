use super::*;

pub(super) struct Native {
    pub(super) function: NativeFunction,
    pub(super) properties: Properties,
}

impl Heap {
    pub(crate) fn install_object_prototype(&self, prototype: Value) -> bool {
        let mut data = self.0.borrow_mut();
        if data.object_prototype.is_some()
            || heap_id(prototype).and_then(|id| data.object(id)).is_none()
        {
            return false;
        }
        data.retain(prototype);
        data.object_prototype = Some(prototype);
        data.object_shape = data.shapes.root(Some(prototype));
        data.empty_object_shape = data.shapes.root(None);
        true
    }

    pub(crate) fn install_function_prototype(&self, prototype: Value) -> bool {
        let mut data = self.0.borrow_mut();
        if data.function_prototype.is_some()
            || heap_id(prototype).and_then(|id| data.object(id)).is_none()
        {
            return false;
        }
        data.retain(prototype);
        data.function_prototype = Some(prototype);
        true
    }

    pub(crate) fn install_array_prototype(&self, prototype: Value) -> bool {
        let mut data = self.0.borrow_mut();
        if data.array_prototype.is_some()
            || heap_id(prototype).and_then(|id| data.array(id)).is_none()
        {
            return false;
        }
        data.retain(prototype);
        data.array_prototype = Some(prototype);
        data.array_shape = data.shapes.array_root(Some(prototype));
        data.sync(heap_id(prototype).unwrap());
        true
    }

    pub(crate) fn install_string_prototype(&self, prototype: Value) -> bool {
        let mut data = self.0.borrow_mut();
        if data.string_prototype.is_some()
            || heap_id(prototype).and_then(|id| data.object(id)).is_none()
        {
            return false;
        }
        data.retain(prototype);
        data.string_prototype = Some(prototype);
        data.string_shape = data.shapes.string_root(prototype);
        true
    }

    pub(crate) fn native_function(&self, function: NativeFunction) -> Option<Value> {
        let mut data = self.0.borrow_mut();
        let id = if let Some((_, id)) = data.natives.iter().find(|(kind, _)| *kind == function) {
            *id
        } else {
            let id = data.reserve()?;
            data.store_shaped(
                id,
                Payload::Native(Native {
                    function,
                    properties: Properties::default(),
                }),
            );
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
            Payload::Native(native) => Some(native.function),
            _ => None,
        }
    }
}
