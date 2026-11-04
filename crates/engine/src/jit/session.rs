use crate::Engine;

use crate::environment::Environment;
use crate::error::Error;
use crate::value::Value;

struct Frame {
    locals: usize,
    temps: usize,
    index_roots: Vec<Option<[Value; 3]>>,
    resume_state: usize,
}

#[repr(C)]
pub(crate) struct JitCtx<'a> {
    pub(crate) global: Environment,
    pub(crate) error: Option<Box<Error>>,
    pub(crate) error_origin: Option<(String, u32, u32)>,
    exception: Value,
    frames: Vec<Frame>,
    locals: Vec<Value>,
    locals_base: *mut Value,
    temps: Vec<Value>,
    pending: u64,
    epoch: *const u64,
    pub(crate) engine: &'a Engine,
}

pub(crate) const ERROR_OFFSET: i32 = std::mem::offset_of!(JitCtx<'static>, error) as i32;
pub(crate) const EXCEPTION_OFFSET: i32 = std::mem::offset_of!(JitCtx<'static>, exception) as i32;
pub(crate) const LOCALS_OFFSET: i32 = std::mem::offset_of!(JitCtx<'static>, locals_base) as i32;
pub(crate) const PENDING_OFFSET: i32 = std::mem::offset_of!(JitCtx<'static>, pending) as i32;
pub(crate) const EPOCH_OFFSET: i32 = std::mem::offset_of!(JitCtx<'static>, epoch) as i32;

impl<'a> JitCtx<'a> {
    fn new(engine: &'a Engine) -> Self {
        Self {
            global: engine.global().clone(),
            error: None,
            error_origin: None,
            exception: Value::UNINITIALIZED,
            frames: Vec::new(),
            locals: Vec::new(),
            locals_base: std::ptr::null_mut(),
            temps: Vec::new(),
            pending: 0,
            epoch: engine.global().heap().epoch(),
            engine,
        }
    }

    pub(crate) fn evaluate(&mut self, source: &str, filename: &str) -> Value {
        let entry = match self.engine.prepare_source(source, filename) {
            Ok(Some(entry)) => entry,
            Ok(None) => return Value::UNDEFINED,
            Err(error) => {
                self.error = Some(Box::new(error));
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
            self.error = Some(Box::new(Error::Runtime("invalid thrown value".into())));
            return;
        }
        let old = std::mem::replace(&mut self.exception, value);
        self.global.heap().release(old);
    }

    pub(crate) fn exception_pending(&self) -> bool {
        self.exception != Value::UNINITIALIZED
    }

    pub(crate) fn catch(&mut self) -> Value {
        let value = std::mem::replace(&mut self.exception, Value::UNINITIALIZED);
        self.clear();
        self.own(value)
    }

    pub(crate) fn entry(&self, code: u64) -> Option<usize> {
        self.engine.entry(code)
    }

    pub(crate) fn resume(&mut self, locals: Vec<Value>, state: usize) {
        self.push_frame(state);
        self.locals.extend(locals);
        self.sync_locals();
    }

    fn push_frame(&mut self, resume_state: usize) {
        self.frames.push(Frame {
            locals: self.locals.len(),
            temps: self.temps.len(),
            index_roots: Vec::new(),
            resume_state,
        });
    }

    fn sync_locals(&mut self) {
        self.locals_base = match self.frames.last() {
            Some(frame) => self.locals.as_mut_ptr().wrapping_add(frame.locals),
            None => std::ptr::null_mut(),
        };
        self.sync_temps();
    }

    fn sync_temps(&mut self) {
        let base = self.frames.last().map_or(0, |frame| frame.temps);
        self.pending = (self.temps.len() - base) as u64;
    }

    fn frame(&self) -> &Frame {
        self.frames.last().unwrap()
    }

    pub(crate) fn resume_state(&self) -> usize {
        self.frames.last().unwrap().resume_state
    }

    pub(crate) fn local_get(&mut self, slot: usize) -> Value {
        let value = self
            .locals
            .get(self.frame().locals + slot)
            .copied()
            .unwrap_or(Value::UNDEFINED);
        self.retain_temp(value)
    }

    pub(crate) fn suspend(&mut self, generator: Value, value: Value, state: usize) -> Value {
        if !self.global.heap().retain(value) {
            self.error = Some(Box::new(Error::Runtime("invalid yielded value".into())));
            return self.exit(Value::UNDEFINED);
        }
        self.clear();
        let frame = self.frames.pop().unwrap();
        for values in frame.index_roots.into_iter().flatten() {
            for value in values {
                self.global.heap().release(value);
            }
        }
        let locals = self.locals.split_off(frame.locals);
        self.sync_locals();
        self.global
            .heap()
            .generator_suspend(generator, state, locals);
        value
    }

    pub(crate) fn enter(&mut self, size: usize) {
        self.push_frame(0);
        self.locals
            .resize(self.locals.len() + size, Value::UNDEFINED);
        self.sync_locals();
    }
    pub(crate) fn own(&mut self, value: Value) -> Value {
        if (value.is_object() && value != Value::GLOBAL_OBJECT) || value.is_string() {
            self.temps.push(value);
            self.pending += 1;
        }
        value
    }
    pub(crate) fn retain_temp(&mut self, value: Value) -> Value {
        if !self.global.heap().retain(value) {
            self.error = Some(Box::new(Error::Runtime("invalid heap value".into())));
            return Value::UNDEFINED;
        }
        self.own(value)
    }
    pub(crate) fn clear(&mut self) {
        let base = self.frame().temps;
        self.global.heap().release_all(&self.temps[base..]);
        self.temps.truncate(base);
        self.pending = 0;
    }
    pub(crate) fn root_index_read(
        &mut self,
        slot: usize,
        receiver: Value,
        key: Value,
        result: Value,
    ) -> bool {
        let heap = self.global.heap();
        if !heap.supports_indexed_storage(
            receiver,
            key,
            crate::heap::IndexedStorageContract::DirectNumericIndexStorage,
        ) {
            return false;
        }
        let values = [receiver, key, result];
        let mut retained = 0;
        for value in values {
            if !heap.retain(value) {
                for value in values.into_iter().take(retained) {
                    heap.release(value);
                }
                return false;
            }
            retained += 1;
        }
        let roots = &mut self.frames.last_mut().unwrap().index_roots;
        roots.resize(roots.len().max(slot + 1), None);
        if let Some(old) = roots[slot].replace(values) {
            for value in old {
                heap.release(value);
            }
        }
        true
    }

    fn take_ownership(&mut self, value: Value) -> bool {
        if self.temps.len() > self.frame().temps && self.temps.last() == Some(&value) {
            self.temps.pop();
            self.pending -= 1;
            return true;
        }
        if self.global.heap().retain(value) {
            return true;
        }
        self.error = Some(Box::new(Error::Runtime("invalid heap value".into())));
        false
    }

    pub(crate) fn local_replace(&mut self, slot: usize, value: Value, old: Value) -> Value {
        if !self.take_ownership(value) {
            let index = self.frame().locals + slot;
            self.locals[index] = old;
            return Value::UNDEFINED;
        }
        self.global.heap().release(old);
        value
    }

    pub(crate) fn local_set(&mut self, slot: usize, value: Value) -> Value {
        if !self.take_ownership(value) {
            return Value::UNDEFINED;
        }
        let frame = self.frames.last().unwrap();
        let index = frame.locals + slot;
        if index >= self.locals.len() {
            self.locals.resize(index + 1, Value::UNDEFINED);
            self.sync_locals();
        }
        let old = std::mem::replace(&mut self.locals[index], value);
        self.global.heap().release(old);
        value
    }
    pub(crate) fn exit(&mut self, value: Value) -> Value {
        let value = if self.global.heap().retain(value) {
            value
        } else {
            self.error = Some(Box::new(Error::Runtime("invalid heap value".into())));
            Value::UNDEFINED
        };
        let frame = self.frames.pop().unwrap();
        let heap = self.global.heap();
        heap.release_all(&self.locals[frame.locals..]);
        self.locals.truncate(frame.locals);
        heap.release_all(&self.temps[frame.temps..]);
        self.temps.truncate(frame.temps);
        for roots in frame.index_roots.into_iter().flatten() {
            heap.release_all(&roots);
        }
        self.sync_locals();
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
    let exception = std::mem::replace(&mut ctx.exception, Value::UNINITIALIZED);
    if exception != Value::UNINITIALIZED {
        let message = global
            .heap()
            .text(exception)
            .or_else(|| global.heap().error_text(exception))
            .unwrap_or_else(|| exception.to_string());
        global.heap().release(exception);
        if ctx.error.is_none() {
            ctx.error = Some(Box::new(Error::Runtime(format!(
                "uncaught exception: {message}"
            ))));
        }
    }
    match ctx.error {
        Some(error) => {
            global.heap().release(value);
            match (*error, ctx.error_origin) {
                (Error::Runtime(message), Some((filename, start, end))) => Err(Error::Runtime(
                    format!("{filename}:{start}..{end}: {message}"),
                )),
                (error, _) => Err(error),
            }
        }
        None => Ok(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_read_roots_replace_and_release_on_error_exit() {
        let engine = Engine::new().unwrap();
        let mut ctx = JitCtx::new(&engine);
        let heap = engine.global().heap();
        let first = heap.object().unwrap();
        let first_array = heap.array(&[first]).unwrap();
        let second = heap.object().unwrap();
        let second_array = heap.array(&[second]).unwrap();
        let key = Value::number(0.0);
        ctx.enter(0);
        assert!(ctx.root_index_read(0, first_array, key, first));
        heap.release(first_array);
        heap.release(first);
        heap.drain(100);
        assert!(heap.valid(first_array) && heap.valid(first));
        assert!(ctx.root_index_read(0, second_array, key, second));
        assert_eq!(ctx.frames.last().unwrap().index_roots.len(), 1);
        heap.drain(100);
        assert!(!heap.valid(first_array) && !heap.valid(first));
        heap.release(second_array);
        heap.release(second);
        ctx.error = Some(Box::new(Error::Runtime("test error".into())));
        ctx.exit(Value::UNDEFINED);
        heap.drain(100);
        assert!(!heap.valid(second_array) && !heap.valid(second));
    }

    #[test]
    fn dispatcher_releases_rooted_index_receiver_on_return_and_error() {
        let mut engine = Engine::new().unwrap();
        engine
            .compile(
                r#"
            function run(n) {
                if (!this.items[0]) return 0;
                let sum = 0;
                for (let i = 0; i < Math.floor(n); ++i) sum += this.items[0].value;
                return sum;
            }
            function fail(n) {
                if (!this.items[0]) return 0;
                for (let i = 0; i < Math.floor(n); ++i) this.items[0] * {};
                return 0;
            }
            function invoke(holder) { return holder.run(2); }
        "#,
                "rooted-index-release.js",
            )
            .unwrap();
        let heap = engine.global().heap();
        for (name, error) in [("run", false), ("fail", true)] {
            let item = heap.object().unwrap();
            assert!(heap.set_property(item, "value", Value::number(7.0)));
            let array = heap.array(&[item]).unwrap();
            let holder = heap.object().unwrap();
            assert!(heap.set_property(holder, "items", array));
            assert!(heap.set_property(holder, "run", engine.global().get_raw(name).unwrap()));
            assert_eq!(engine.call("invoke", &[holder]).is_err(), error);
            heap.release(holder);
            heap.release(array);
            heap.release(item);
            heap.drain(1000);
            assert!(!heap.valid(holder) && !heap.valid(array) && !heap.valid(item));
        }
    }

    #[test]
    fn indexed_read_roots_reject_native_method_and_string_keys() {
        let engine = Engine::new().unwrap();
        let mut ctx = JitCtx::new(&engine);
        let heap = engine.global().heap();
        let array = heap.array(&[]).unwrap();
        let key = heap.string("map").unwrap();
        let method = heap
            .native_function(crate::heap::NativeFunction::ArrayMap)
            .unwrap();
        ctx.enter(0);
        assert!(!ctx.root_index_read(0, array, key, method));
        assert!(ctx.frames.last().unwrap().index_roots.is_empty());
        heap.release(array);
        heap.release(key);
        heap.release(method);
        ctx.exit(Value::UNDEFINED);
    }
}
