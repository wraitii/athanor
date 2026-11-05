use super::*;

pub(super) struct Cell {
    pub(super) value: Value,
}

pub(super) struct Closure {
    pub(super) code: CodeId,
    pub(super) captures: Vec<Value>,
    pub(super) lexical_this: Option<Value>,
    pub(super) constructable: bool,
    pub(super) class: bool,
    pub(super) properties: Properties,
}

impl Heap {
    pub(crate) fn cell(&self, value: Value) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !data.valid(value) {
            return None;
        }
        let id = data.reserve()?;
        data.retain(value);
        data.store(id, Payload::Cell(Cell { value }));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn closure(&self, code: CodeId, captures: &[Value]) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !captures
            .iter()
            .all(|value| heap_id(*value).and_then(|id| data.cell(id)).is_some())
        {
            return None;
        }
        let id = data.reserve()?;
        for value in captures {
            data.retain(*value);
        }
        data.store_shaped(
            id,
            Payload::Closure(Box::new(Closure {
                code,
                captures: captures.to_vec(),
                lexical_this: None,
                constructable: true,
                class: false,
                properties: Properties::default(),
            })),
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    #[cfg(test)]
    pub(crate) fn closure_parts(&self, closure: Value) -> Option<(CodeId, Vec<Value>)> {
        let data = self.0.borrow();
        let closure = data.closure(heap_id(closure)?)?;
        Some((closure.code, closure.captures.clone()))
    }

    pub(crate) fn bind_this(&self, closure: Value, receiver: Value) -> bool {
        let Some(id) = heap_id(closure) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        if data.closure(id).is_none() || !data.retain(receiver) {
            return false;
        }
        let closure = data.closure_mut(id).unwrap();
        closure.constructable = false;
        let old = closure.lexical_this.replace(receiver);
        if let Some(old) = old {
            data.release(old);
        }
        true
    }

    pub(crate) fn disable_constructor(&self, closure: Value) {
        self.0
            .borrow_mut()
            .closure_mut(heap_id(closure).unwrap())
            .unwrap()
            .constructable = false;
    }

    pub(crate) fn mark_class(&self, closure: Value) {
        self.0
            .borrow_mut()
            .closure_mut(heap_id(closure).unwrap())
            .unwrap()
            .class = true;
    }

    pub(crate) fn is_class(&self, closure: Value) -> bool {
        heap_id(closure)
            .and_then(|id| self.0.borrow().closure(id).map(|closure| closure.class))
            .unwrap_or(false)
    }

    pub(crate) fn constructable(&self, closure: Value) -> bool {
        heap_id(closure).is_some_and(|id| {
            self.0
                .borrow()
                .closure(id)
                .is_some_and(|closure| closure.constructable)
        })
    }

    pub(crate) fn lexical_this(&self, closure: Value) -> Option<Value> {
        let data = self.0.borrow();
        let closure = data.closure(heap_id(closure)?)?;

        closure.lexical_this
    }

    pub(crate) fn closure_code(&self, closure: Value) -> Option<CodeId> {
        let data = self.0.borrow();
        let closure = data.closure(heap_id(closure)?)?;
        Some(closure.code)
    }

    pub(crate) fn capture(&self, closure: Value, index: usize) -> Option<Value> {
        let data = self.0.borrow();
        let closure = data.closure(heap_id(closure)?)?;

        closure.captures.get(index).copied()
    }

    pub(crate) fn cell_get(&self, cell: Value) -> Option<Value> {
        let data = self.0.borrow();
        let cell = data.cell(heap_id(cell)?)?;
        Some(cell.value)
    }

    pub(crate) fn cell_set(&self, cell: Value, value: Value) -> bool {
        let mut data = self.0.borrow_mut();
        let Some(id) = heap_id(cell) else {
            return false;
        };
        if data.cell(id).is_none() || !data.retain(value) {
            return false;
        }
        let old = std::mem::replace(&mut data.cell_mut(id).unwrap().value, value);
        data.release(old);
        true
    }
}
