use super::*;

pub(super) struct Generator {
    pub(super) code: CodeId,
    pub(super) closure: Value,
    pub(super) arguments: Value,
    pub(super) receiver: Value,
    pub(super) state: usize,
    pub(super) locals: Vec<Value>,
}

impl Heap {
    pub(crate) fn generator(
        &self,
        code: CodeId,
        closure: Value,
        arguments: Value,
        receiver: Value,
    ) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !data.valid(closure) || !data.valid(arguments) || !data.valid(receiver) {
            return None;
        }
        let id = data.reserve()?;
        data.retain(closure);
        data.retain(arguments);
        data.retain(receiver);
        data.store(id, Payload::Generator(Box::new(Generator {
            code,
            closure,
            arguments,
            receiver,
            state: 0,
            locals: Vec::new(),
        })));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn generator_receiver(&self, value: Value) -> Option<Value> {
        let data = self.0.borrow();
        let generator = data.generator(heap_id(value)?)?;
        Some(generator.receiver)
    }

    pub(crate) fn generator_state(&self, value: Value) -> Option<usize> {
        let data = self.0.borrow();
        let generator = data.generator(heap_id(value)?)?;
        Some(generator.state)
    }

    pub(crate) fn generator_parts(&self, value: Value) -> Option<(CodeId, Value, Value)> {
        let data = self.0.borrow();
        let generator = data.generator(heap_id(value)?)?;
        Some((generator.code, generator.closure, generator.arguments))
    }

    pub(crate) fn generator_begin(&self, value: Value) -> Option<(usize, Vec<Value>)> {
        let mut data = self.0.borrow_mut();
        let generator = data.generator_mut(heap_id(value)?)?;
        if generator.state >= GENERATOR_RUNNING {
            return None;
        }
        let state = std::mem::replace(&mut generator.state, GENERATOR_RUNNING);
        Some((state, std::mem::take(&mut generator.locals)))
    }

    pub(crate) fn generator_suspend(&self, value: Value, state: usize, locals: Vec<Value>) {
        let mut data = self.0.borrow_mut();
        let generator = data.generator_mut(heap_id(value).unwrap()).unwrap();
        generator.state = state;
        generator.locals = locals;
    }

    pub(crate) fn generator_complete(&self, value: Value) {
        let mut data = self.0.borrow_mut();
        let generator = data.generator_mut(heap_id(value).unwrap()).unwrap();
        generator.state = GENERATOR_DONE;
        let closure = std::mem::replace(&mut generator.closure, Value::UNDEFINED);
        let arguments = std::mem::replace(&mut generator.arguments, Value::UNDEFINED);
        let receiver = std::mem::replace(&mut generator.receiver, Value::UNDEFINED);
        data.release(closure);
        data.release(arguments);
        data.release(receiver);
    }
}
