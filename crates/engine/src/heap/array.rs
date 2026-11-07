use super::*;

pub(crate) const HOLE: Value = Value::UNINITIALIZED;

pub(super) struct Array {
    pub(super) elements: Vec<Value>,
    pub(super) holes: usize,
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
        data.store(id, Payload::Array(Array { elements, holes }));
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
            values.extend((0..array.elements.len()).map(|index| array.get(index)));
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
        let id = data.reserve()?;
        for value in elements {
            data.retain(*value);
        }
        data.store(
            id,
            Payload::Array(Array {
                elements: elements.to_vec(),
                holes: elements.iter().filter(|value| **value == HOLE).count(),
            }),
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn array_of_length(&self, length: usize) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let mut elements = Vec::new();
        elements.try_reserve_exact(length).ok()?;
        elements.resize(length, HOLE);
        let id = data.reserve()?;
        data.store(
            id,
            Payload::Array(Array {
                elements,
                holes: length,
            }),
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn supports_indexed_storage(
        &self,
        receiver: Value,
        key: Value,
        contract: IndexedStorageContract,
    ) -> bool {
        match contract {
            IndexedStorageContract::DirectNumericIndexStorage => {
                self.direct_numeric_index(receiver, key)
            }
        }
    }

    pub(super) fn direct_numeric_index(&self, receiver: Value, key: Value) -> bool {
        let Some(number) = key.as_number() else {
            return false;
        };
        let data = self.0.borrow();
        let Some(payload) = heap_id(receiver)
            .and_then(|id| data.live(id))
            .map(|slot| &slot.payload)
        else {
            return false;
        };
        match payload {
            Payload::TypedArray(_) => true,
            Payload::Array(_) => element_index(number, u32::MAX as f64).is_some(),
            _ => false,
        }
    }

    pub(crate) fn read_numeric_index(&self, receiver: Value, number: f64) -> Option<IndexRead> {
        let data = self.0.borrow();
        let payload = &data.live(heap_id(receiver)?)?.payload;
        match payload {
            Payload::TypedArray(array) => {
                let value = element_index(number, usize::MAX as f64)
                    .and_then(|index| array.storage.get(index))
                    .map(Value::number)
                    .unwrap_or(Value::UNDEFINED);
                Some(IndexRead::Primitive(value))
            }
            Payload::Array(array) => {
                let index = element_index(number, u32::MAX as f64)?;
                let value = array.get(index);
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
        Some(data.array(id)?.get(index))
    }

    pub(crate) fn present(&self, array: Value, index: usize) -> bool {
        let data = self.0.borrow();
        heap_id(array)
            .and_then(|id| data.array(id))
            .is_some_and(|array| array.present(index))
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
        let mut data = self.0.borrow_mut();
        let Some(id) = heap_id(array) else {
            return false;
        };
        let Some(len) = data.array(id).map(|array| array.elements.len()) else {
            return false;
        };
        if index > len || !data.retain(value) {
            return false;
        }
        let array = data.array_mut(id).unwrap();
        if index == len {
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
