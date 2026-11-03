use crate::Engine;

use crate::environment::Environment;
use crate::error::Error;
use crate::value::Value;

#[derive(Default)]
struct Frame {
    locals: Vec<Value>,
    temps: Vec<Value>,
    resume_state: usize,
}

#[repr(C)]
pub(crate) struct JitCtx<'a> {
    pub(crate) global: Environment,
    pub(crate) error: Option<Error>,
    pub(crate) error_origin: Option<(String, u32, u32)>,
    exception: Option<Value>,
    frames: Vec<Frame>,
    pub(crate) engine: &'a Engine,
}

impl<'a> JitCtx<'a> {
    fn new(engine: &'a Engine) -> Self {
        Self {
            global: engine.global().clone(),
            error: None,
            error_origin: None,
            exception: None,
            frames: Vec::new(),
            engine,
        }
    }

    pub(crate) fn evaluate(&mut self, source: &str, filename: &str) -> Value {
        let entry = match self.engine.prepare_source(source, filename) {
            Ok(Some(entry)) => entry,
            Ok(None) => return Value::UNDEFINED,
            Err(error) => {
                self.error = Some(error);
                return Value::UNDEFINED;
            }
        };
        let function: extern "C" fn(*mut JitCtx, Value, Value, *const Value, u64) -> Value =
            unsafe { std::mem::transmute(entry) };
        let value = function(
            self,
            Value::UNDEFINED,
            Value::GLOBAL_OBJECT,
            std::ptr::null(),
            0,
        );
        self.own(value)
    }

    pub(crate) fn throw(&mut self, value: Value) {
        if !self.global.heap().retain(value) {
            self.error = Some(Error::Runtime("invalid thrown value".into()));
            return;
        }
        if let Some(old) = self.exception.replace(value) {
            self.global.heap().release(old);
        }
    }

    pub(crate) fn exception_pending(&self) -> bool {
        self.exception.is_some()
    }

    pub(crate) fn catch(&mut self) -> Value {
        let value = self.exception.take().unwrap();
        self.clear();
        self.own(value)
    }

    pub(crate) fn entry(&self, code: u64) -> Option<usize> {
        self.engine.entry(code)
    }

    pub(crate) fn resume(&mut self, locals: Vec<Value>, state: usize) {
        self.frames.push(Frame {
            locals,
            temps: Vec::new(),
            resume_state: state,
        });
    }

    pub(crate) fn resume_state(&self) -> usize {
        self.frames.last().unwrap().resume_state
    }

    pub(crate) fn local_get(&mut self, slot: usize) -> Value {
        let value = self
            .frames
            .last()
            .unwrap()
            .locals
            .get(slot)
            .copied()
            .unwrap_or(Value::UNDEFINED);
        self.retain_temp(value)
    }

    pub(crate) fn suspend(&mut self, generator: Value, value: Value, state: usize) -> Value {
        if !self.global.heap().retain(value) {
            self.error = Some(Error::Runtime("invalid yielded value".into()));
            return self.exit(Value::UNDEFINED);
        }
        self.clear();
        let frame = self.frames.pop().unwrap();
        self.global
            .heap()
            .generator_suspend(generator, state, frame.locals);
        value
    }

    pub(crate) fn enter(&mut self) {
        self.frames.push(Frame::default());
    }
    pub(crate) fn own(&mut self, value: Value) -> Value {
        if (value.is_object() && value != Value::GLOBAL_OBJECT) || value.is_string() {
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
    engine: &Engine,
    closure: Value,
    receiver: Value,
) -> Result<Value, Error> {
    let global = engine.global();
    let mut ctx = JitCtx::new(engine);
    let argv = if args.is_empty() {
        std::ptr::null()
    } else {
        args.as_ptr().cast::<u64>()
    };
    let function: extern "C" fn(*mut JitCtx, Value, Value, *const u64, u64) -> u64 =
        unsafe { std::mem::transmute(entry) };
    let value = Value::from_bits(function(
        &mut ctx,
        closure,
        receiver,
        argv,
        args.len() as u64,
    ));
    if let Some(exception) = ctx.exception.take() {
        let message = global
            .heap()
            .text(exception)
            .or_else(|| global.heap().error_text(exception))
            .unwrap_or_else(|| exception.to_string());
        global.heap().release(exception);
        if ctx.error.is_none() {
            ctx.error = Some(Error::Runtime(format!("uncaught exception: {message}")));
        }
    }
    match ctx.error {
        Some(error) => {
            global.heap().release(value);
            match (error, ctx.error_origin) {
                (Error::Runtime(message), Some((filename, start, end))) => Err(Error::Runtime(
                    format!("{filename}:{start}..{end}: {message}"),
                )),
                (error, _) => Err(error),
            }
        }
        None => Ok(value),
    }
}
