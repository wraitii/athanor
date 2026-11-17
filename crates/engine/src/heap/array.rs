use super::*;

const MAX_ARRAY_LENGTH: usize = (256 << 20) / std::mem::size_of::<Value>();

pub(crate) const HOLE: Value = Value::UNINITIALIZED;

pub(super) struct Array {
    pub(super) elements: Vec<Value>,
    pub(super) holes: usize,
    pub(super) properties: Option<Box<Properties>>,
    pub(super) prototype: Option<Value>,
}

impl Array {
    pub(super) fn get(&self, index: usize) -> Value {
        match self.elements.get(index) {
            Some(&value) if value != HOLE => value,
            _ => Value::UNDEFINED,
        }
    }

    pub(super) fn present(&self, index: usize) -> bool {
        self.elements.get(index).is_some_and(|value| *value != HOLE)
    }
}

impl Heap {
    pub(crate) fn has_indexed_prototypes(&self) -> bool {
        self.0.borrow().indexed_prototypes
    }

    // Consumes the element references on both success and failure.
    pub(crate) fn array_from_owned(&self, elements: Vec<Value>) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let Some(id) = data.reserve() else {
            for value in elements {
                data.release(value);
            }
            return None;
        };
        let holes = elements.iter().filter(|value| **value == HOLE).count();
        data.store_array(id, elements, holes);
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    #[cfg(test)]
    pub(crate) fn owned_indexed_values(&self, value: Value) -> Option<Vec<Value>> {
        let mut values = Vec::new();
        self.owned_indexed_values_into(value, &mut values)?;
        Some(values)
    }

    // The snapshot owns its references independently of mutations to the source.
    pub(crate) fn owned_indexed_values_into(
        &self,
        value: Value,
        values: &mut Vec<Value>,
    ) -> Option<()> {
        assert!(values.is_empty());
        let mut data = self.0.borrow_mut();
        let id = heap_id(value)?;
        if let Some(array) = data.array(id) {
            values.extend((0..array.elements.len()).map(|index| data.array_element(id, index)));
        } else {
            let array = data.typed_array(id)?;
            values.extend(
                (0..array.storage.len())
                    .map(|index| Value::number(array.storage.get(index).unwrap())),
            );
        }
        for &value in values.iter() {
            assert!(data.retain(value));
        }
        Some(())
    }

    pub(crate) fn take_array_element(&self, array: Value, index: usize) -> Value {
        let mut data = self.0.borrow_mut();
        let array = data.array_mut(heap_id(array).unwrap()).unwrap();
        assert_eq!(array.holes, 0);
        array
            .elements
            .get_mut(index)
            .map(|value| std::mem::replace(value, Value::UNDEFINED))
            .unwrap_or(Value::UNDEFINED)
    }

    pub(crate) fn take_array_values(&self, array: Value) -> Vec<Value> {
        let mut data = self.0.borrow_mut();
        let id = heap_id(array).unwrap();
        let array = data.array_mut(id).unwrap();
        assert_eq!(array.holes, 0);
        let values = std::mem::take(&mut array.elements);
        data.sync(id);
        values
    }

    pub(crate) fn append_array_owned(&self, array: Value, values: &mut Vec<Value>) {
        let mut data = self.0.borrow_mut();
        let id = heap_id(array).unwrap();
        data.array_mut(id).unwrap().elements.append(values);
        data.sync(id);
    }

    pub(crate) fn push_array_owned(&self, array: Value, value: Value) {
        let mut data = self.0.borrow_mut();
        let id = heap_id(array).unwrap();
        data.array_mut(id).unwrap().elements.push(value);
        data.sync(id);
    }

    pub(crate) fn extend_array_values(&self, destination: Value, values: &[Value]) -> bool {
        let mut data = self.0.borrow_mut();
        let Some(destination) = heap_id(destination).filter(|&id| data.array(id).is_some()) else {
            return false;
        };
        if data
            .array_mut(destination)
            .unwrap()
            .elements
            .try_reserve(values.len())
            .is_err()
        {
            return false;
        }
        for &value in values {
            assert!(data.retain(value));
            data.array_mut(destination).unwrap().elements.push(value);
        }
        data.sync(destination);
        true
    }

    pub(crate) fn array(&self, elements: &[Value]) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !elements.iter().all(|value| data.valid(*value)) {
            return None;
        }
        let mut buffer = data.array_buffers.take(elements.len())?;
        buffer.extend_from_slice(elements);
        let id = data.reserve()?;
        for value in elements {
            data.retain(*value);
        }
        data.store_array(
            id,
            buffer,
            elements.iter().filter(|value| **value == HOLE).count(),
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn array_of_length(&self, length: usize) -> Option<Value> {
        if length > MAX_ARRAY_LENGTH {
            return None;
        }
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let mut elements = data.array_buffers.take(length)?;
        elements.resize(length, HOLE);
        let id = data.reserve()?;
        data.store_array(id, elements, length);
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn read_numeric_index(&self, receiver: Value, number: f64) -> Option<IndexRead> {
        let data = self.0.borrow();
        let id = heap_id(receiver)?;
        let payload = &data.live(id)?.payload;
        match payload {
            Payload::TypedArray(array) => {
                let value = element_index(number, usize::MAX as f64)
                    .and_then(|index| array.storage.get(index))
                    .map(Value::number)
                    .unwrap_or(Value::UNDEFINED);
                Some(IndexRead::Primitive(value))
            }
            Payload::Array(_) => {
                let index = element_index(number, u32::MAX as f64)?;
                let value = data.array_element(id, index);
                Some(if value.is_string() || value.is_object() {
                    IndexRead::Borrowed(value)
                } else {
                    IndexRead::Primitive(value)
                })
            }
            Payload::String(text) => Some(
                element_index(number, u32::MAX as f64)
                    .and_then(|index| text.chars().nth(index))
                    .map(IndexRead::Character)
                    .unwrap_or(IndexRead::Primitive(Value::UNDEFINED)),
            ),
            _ => None,
        }
    }

    pub(crate) fn get(&self, array: Value, index: usize) -> Option<Value> {
        let data = self.0.borrow();
        let id = heap_id(array)?;
        if let Some(array) = data.typed_array(id) {
            return Some(
                array
                    .storage
                    .get(index)
                    .map(Value::number)
                    .unwrap_or(Value::UNDEFINED),
            );
        }
        data.array(id)?;
        Some(data.array_element(id, index))
    }

    pub(crate) fn present(&self, array: Value, index: usize) -> bool {
        let data = self.0.borrow();
        let Some(id) = heap_id(array) else {
            return false;
        };
        let Some(array) = data.array(id) else {
            return false;
        };
        array.present(index)
            || (data.indexed_prototypes
                && array
                    .prototype
                    .and_then(heap_id)
                    .is_some_and(|id| data.indexed_property_present(id, &index.to_string())))
    }

    pub(crate) fn mark_hole(&self, array: Value, index: usize) {
        let mut data = self.0.borrow_mut();
        let id = heap_id(array).unwrap();
        let array = data.array_mut(id).unwrap();
        let old = std::mem::replace(&mut array.elements[index], HOLE);
        if old != HOLE {
            array.holes += 1;
            data.release(old);
        }
        data.sync(id);
    }

    pub(crate) fn is_array(&self, value: Value) -> bool {
        let data = self.0.borrow();
        heap_id(value).and_then(|id| data.array(id)).is_some()
    }

    pub(crate) fn len(&self, array: Value) -> Option<usize> {
        let data = self.0.borrow();
        if array.is_string() {
            let text = data.string(heap_id(array)?)?;
            return Some(text.chars().count());
        }
        let id = heap_id(array)?;
        if let Some(array) = data.typed_array(id) {
            return Some(array.storage.len());
        }
        let array = data.array(id)?;
        Some(array.elements.len())
    }

    pub(crate) fn set(&self, array: Value, index: usize, value: Value) -> bool {
        if index >= MAX_ARRAY_LENGTH {
            return false;
        }
        let mut data = self.0.borrow_mut();
        let Some(id) = heap_id(array) else {
            return false;
        };
        let Some(len) = data.array(id).map(|array| array.elements.len()) else {
            return false;
        };
        if index >= len {
            let Some(length) = index.checked_add(1) else {
                return false;
            };
            if data
                .array_mut(id)
                .unwrap()
                .elements
                .try_reserve(length - len)
                .is_err()
            {
                return false;
            }
        }
        if !data.retain(value) {
            return false;
        }
        data.indexed_prototype_write(id);
        let array = data.array_mut(id).unwrap();
        if index >= len {
            array.elements.resize(index, HOLE);
            array.holes += index - len;
            array.elements.push(value);
        } else {
            let old = std::mem::replace(&mut array.elements[index], value);
            if old == HOLE {
                array.holes -= 1;
            }
            data.release(old);
        }
        data.sync(id);
        true
    }

    pub(crate) fn reverse_array(&self, value: Value) -> bool {
        let Some(id) = heap_id(value) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(array) = data.array_mut(id) else {
            return false;
        };
        array.elements.reverse();
        true
    }

    pub(crate) fn truncate(&self, array: Value, length: usize) -> bool {
        let mut data = self.0.borrow_mut();
        let Some(id) = heap_id(array) else {
            return false;
        };
        let Some(array) = data.array_mut(id) else {
            return false;
        };
        if length > array.elements.len() {
            return false;
        }
        let removed: Vec<_> = array.elements.drain(length..).collect();
        array.holes -= removed.iter().filter(|value| **value == HOLE).count();
        for value in removed {
            data.release(value);
        }
        data.sync(id);
        true
    }
}

impl HeapData {
    pub(super) fn indexed_prototype_write(&mut self, id: u64) {
        if self.indexed_prototypes
            || ![self.array_prototype, self.object_prototype]
                .into_iter()
                .flatten()
                .any(|value| heap_id(value) == Some(id))
        {
            return;
        }
        // Existing holey fast paths assume missing elements cannot be inherited.
        self.indexed_prototypes = true;
        self.epoch += 1;
        let arrays: Vec<_> = self
            .arena
            .slots()
            .filter(|slot| slot.references > 0 && matches!(slot.payload, Payload::Array(_)))
            .map(|slot| slot as *const _ as u64)
            .collect();
        for id in arrays {
            self.slot_mut(id).kind |= kind::PROTOTYPE_INDEXED;
        }
    }

    fn indexed_property_present(&self, mut id: u64, name: &str) -> bool {
        loop {
            let value = Value::from_bits(TAG_OBJECT | id);
            if self.enumeration_presence(value, name).is_some() {
                return true;
            }
            let Some(parent) = self.property_prototype(id).and_then(heap_id) else {
                return false;
            };
            id = parent;
        }
    }

    fn array_element(&self, id: u64, index: usize) -> Value {
        let array = self.array(id).unwrap();
        if array.present(index) || !self.indexed_prototypes {
            return array.get(index);
        }
        array
            .prototype
            .and_then(heap_id)
            .and_then(|id| self.object_property(id, &index.to_string()))
            .unwrap_or(Value::UNDEFINED)
    }

    fn store_array(&mut self, id: u64, elements: Vec<Value>, holes: usize) {
        let prototype = self.array_prototype.or(self.object_prototype);
        if let Some(prototype) = prototype {
            self.retain(prototype);
        }
        if self.array_shape == NO_SHAPE {
            self.array_shape = self.shapes.array_root(prototype);
        }
        let shape = self.array_shape;
        self.store(
            id,
            Payload::Array(Array {
                elements,
                holes,
                properties: None,
                prototype,
            }),
        );
        self.slot_mut(id).shape = shape;
    }
}
