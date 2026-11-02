use std::collections::HashMap;

use crate::environment::Environment;
use crate::error::Error;
use crate::value::Value;

#[derive(Default)]
struct Frame {
    locals: Vec<Value>,
    temps: Vec<Value>,
}

#[repr(C)]
pub(crate) struct JitCtx<'a> {
    pub(crate) global: Environment,
    pub(crate) error: Option<Error>,
    frames: Vec<Frame>,
    code: &'a HashMap<u64, usize>,
}

impl<'a> JitCtx<'a> {
    fn new(global: &Environment, code: &'a HashMap<u64, usize>) -> Self {
        Self {
            global: global.clone(),
            error: None,
            frames: Vec::new(),
            code,
        }
    }

    pub(crate) fn entry(&self, code: u64) -> Option<usize> {
        self.code.get(&code).copied()
    }

    pub(crate) fn enter(&mut self) {
        self.frames.push(Frame::default());
    }
    pub(crate) fn own(&mut self, value: Value) -> Value {
        if value.is_object() && value != Value::GLOBAL_OBJECT {
            self.frames.last_mut().unwrap().temps.push(value);
        }
        value
    }
    pub(crate) fn retain_temp(&mut self, value: Value) -> Value {
        if !self.global.heap().retain(value) {
            self.error = Some(Error::Runtime("invalid heap value".into()));
            return Value::UNDEFINED;
        }
        self.own(value)
    }
    pub(crate) fn clear(&mut self) {
        for value in self.frames.last_mut().unwrap().temps.drain(..) {
            self.global.heap().release(value);
        }
    }
    pub(crate) fn local_set(&mut self, slot: usize, value: Value) -> Value {
        if !self.global.heap().retain(value) {
            self.error = Some(Error::Runtime("invalid heap value".into()));
            return Value::UNDEFINED;
        }
        let frame = self.frames.last_mut().unwrap();
        frame
            .locals
            .resize(frame.locals.len().max(slot + 1), Value::UNDEFINED);
        let old = std::mem::replace(&mut frame.locals[slot], value);
        self.global.heap().release(old);
        value
    }
    pub(crate) fn exit(&mut self, value: Value) -> Value {
        let value = if self.global.heap().retain(value) {
            value
        } else {
            self.error = Some(Error::Runtime("invalid heap value".into()));
            Value::UNDEFINED
        };
        let frame = self.frames.pop().unwrap();
        for owned in frame.locals.into_iter().chain(frame.temps) {
            self.global.heap().release(owned);
        }
        value
    }
}

pub(crate) unsafe fn call(
    entry: usize,
    args: &[Value],
    global: &Environment,
    code: &HashMap<u64, usize>,
    closure: Value,
) -> Result<Value, Error> {
    let mut ctx = JitCtx::new(global, code);
    let argv = if args.is_empty() {
        std::ptr::null()
    } else {
        args.as_ptr().cast::<u64>()
    };
    let function: extern "C" fn(*mut JitCtx, Value, *const u64, u64) -> u64 =
        unsafe { std::mem::transmute(entry) };
    let value = Value::from_bits(function(&mut ctx, closure, argv, args.len() as u64));
    match ctx.error {
        Some(error) => {
            global.heap().release(value);
            Err(error)
        }
        None => Ok(value),
    }
}
