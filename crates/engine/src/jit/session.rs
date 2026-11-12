use std::cell::Cell;

use crate::Engine;
use crate::jit::entries::zeroed_cells;
use crate::jit::helpers::HELPERS;
use crate::jit::profile::CallSite;

use crate::environment::Environment;
use crate::error::Error;
use crate::value::Value;

#[cfg(test)]
struct Frame {
    caller_start: usize,
}

pub(crate) const LOCAL_STACK_SIZE: usize = 1 << 20;
const NATIVE_STACK_BUDGET: usize = 1 << 20;

/// Starts zeroed (+0, a non-heap value), so untouched pages are never committed.
pub(crate) struct LocalSlots {
    storage: Box<[Cell<usize>]>,
    active: Cell<bool>,
}

impl LocalSlots {
    pub(crate) fn new() -> Self {
        Self {
            storage: zeroed_cells(LOCAL_STACK_SIZE),
            active: Cell::new(false),
        }
    }

    fn enter(&self) -> *mut Value {
        assert!(!self.active.replace(true), "nested JIT sessions");
        self.storage.as_ptr().cast_mut().cast()
    }

    fn exit(&self) {
        self.active.set(false);
    }
}

/// Frame slots of every active frame. Every slot above `len` holds a non-heap value, so
/// entering a frame only bumps `len` and exiting resets the slots it wrote.
#[repr(C)]
pub(crate) struct LocalStack {
    base: *mut Value,
    start: usize,
    len: usize,
    limit: usize,
    storage: *mut Value,
}

impl LocalStack {
    fn new(storage: *mut Value) -> Self {
        Self {
            base: storage,
            start: 0,
            len: 0,
            limit: LOCAL_STACK_SIZE,
            storage,
        }
    }

    fn values_mut(&mut self) -> &mut [Value] {
        unsafe { std::slice::from_raw_parts_mut(self.storage, self.len) }
    }

    #[cfg(test)]
    fn set_start(&mut self, start: usize) {
        self.start = start;
        self.base = self.storage.wrapping_add(start);
    }

    #[cfg(test)]
    fn grow(&mut self, len: usize) -> bool {
        if len > self.limit {
            return false;
        }
        let old = self.len;
        self.len = len;
        self.values_mut()[old..].fill(Value::UNDEFINED);
        true
    }
}

/// References owned by native code, released on drop unless taken back.
pub(crate) struct Held {
    heap: crate::heap::Heap,
    values: Vec<Value>,
}

impl Held {
    pub(crate) fn new(heap: &crate::heap::Heap) -> Self {
        Self {
            heap: heap.clone(),
            values: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, value: Value) -> Value {
        self.values.push(value);
        value
    }

    pub(crate) fn reserve(&mut self, additional: usize) -> bool {
        self.values.try_reserve(additional).is_ok()
    }

    pub(crate) fn release(&mut self) {
        self.heap.release_all(&self.values);
        self.values.clear();
    }

    pub(crate) fn take(mut self) -> Vec<Value> {
        std::mem::take(&mut self.values)
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.heap.release_all(&self.values);
    }
}

#[repr(C)]
pub(crate) struct JitCtx<'a> {
    pub(crate) global: Environment,
    pub(crate) error: Option<Box<Error>>,
    pub(crate) error_origin: Option<(String, u32, u32)>,
    exception: Value,
    #[cfg(test)]
    frames: Vec<Frame>,
    locals: LocalStack,
    stack_limit: usize,
    pub(super) pack_buffers: Vec<Vec<Value>>,
    epoch: *const u64,
    global_identity: *const (),
    global_revision: *const u64,
    pub(crate) engine: &'a Engine,
    helpers: [*const u8; HELPERS.len()],
    call_site: *const CallSite,
    call_samples_len: usize,
    call_samples: [CallSample; CALL_SAMPLES],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CallSample {
    site: *const CallSite,
    code: u64,
}

pub(crate) const CALL_SAMPLES: usize = 256;
pub(crate) const CALL_SAMPLE_SIZE: usize = std::mem::size_of::<CallSample>();
pub(crate) const CALL_SAMPLE_CODE_OFFSET: i32 = std::mem::offset_of!(CallSample, code) as i32;

pub(crate) const ERROR_OFFSET: i32 = std::mem::offset_of!(JitCtx<'static>, error) as i32;
pub(crate) const EXCEPTION_OFFSET: i32 = std::mem::offset_of!(JitCtx<'static>, exception) as i32;
const LOCAL_STACK_OFFSET: usize = std::mem::offset_of!(JitCtx<'static>, locals);
pub(crate) const LOCALS_OFFSET: i32 =
    (LOCAL_STACK_OFFSET + std::mem::offset_of!(LocalStack, base)) as i32;
pub(crate) const LOCALS_START_OFFSET: i32 =
    (LOCAL_STACK_OFFSET + std::mem::offset_of!(LocalStack, start)) as i32;
pub(crate) const LOCALS_LEN_OFFSET: i32 =
    (LOCAL_STACK_OFFSET + std::mem::offset_of!(LocalStack, len)) as i32;
pub(crate) const LOCALS_LIMIT_OFFSET: i32 =
    (LOCAL_STACK_OFFSET + std::mem::offset_of!(LocalStack, limit)) as i32;
pub(crate) const STACK_LIMIT_OFFSET: i32 =
    std::mem::offset_of!(JitCtx<'static>, stack_limit) as i32;
pub(crate) const EPOCH_OFFSET: i32 = std::mem::offset_of!(JitCtx<'static>, epoch) as i32;

pub(crate) const GLOBAL_IDENTITY_OFFSET: i32 =
    std::mem::offset_of!(JitCtx<'static>, global_identity) as i32;
pub(crate) const GLOBAL_REVISION_OFFSET: i32 =
    std::mem::offset_of!(JitCtx<'static>, global_revision) as i32;
pub(crate) const HELPERS_OFFSET: i32 = std::mem::offset_of!(JitCtx<'static>, helpers) as i32;
pub(crate) const CALL_SITE_OFFSET: i32 = std::mem::offset_of!(JitCtx<'static>, call_site) as i32;
pub(crate) const CALL_SAMPLES_LEN_OFFSET: i32 =
    std::mem::offset_of!(JitCtx<'static>, call_samples_len) as i32;
pub(crate) const CALL_SAMPLES_OFFSET: i32 =
    std::mem::offset_of!(JitCtx<'static>, call_samples) as i32;

impl<'a> JitCtx<'a> {
    pub(super) fn new(engine: &'a Engine) -> Self {
        let marker = 0u8;
        let stack_pointer = std::hint::black_box(&raw const marker) as usize;
        Self {
            global: engine.global().clone(),
            error: None,
            error_origin: None,
            exception: Value::UNINITIALIZED,
            #[cfg(test)]
            frames: Vec::new(),
            locals: LocalStack::new(engine.locals.enter()),
            stack_limit: stack_pointer.saturating_sub(NATIVE_STACK_BUDGET),
            pack_buffers: Vec::new(),
            epoch: engine.global().heap().epoch(),
            global_identity: engine.global().identity(),
            global_revision: engine.global().revision_ptr(),
            engine,
            helpers: std::array::from_fn(|index| HELPERS[index].pointer),
            call_site: std::ptr::null(),
            call_samples_len: 0,
            call_samples: [CallSample {
                site: std::ptr::null(),
                code: 0,
            }; CALL_SAMPLES],
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
        function(
            self,
            Value::UNDEFINED,
            Value::GLOBAL_OBJECT,
            std::ptr::null(),
            0,
        )
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
        std::mem::replace(&mut self.exception, Value::UNINITIALIZED)
    }

    pub(crate) fn flush_call_samples(&mut self) {
        let len = std::mem::take(&mut self.call_samples_len);
        let samples = self.call_samples[..len]
            .iter()
            .filter(|sample| !sample.site.is_null())
            .map(|sample| (unsafe { &*sample.site }, sample.code));
        self.engine.tiering.record_calls(samples);
    }

    pub(crate) fn has_code(&self, code: u64) -> bool {
        self.engine.tiering.has_code(code)
    }

    pub(crate) fn entry(
        &self,
        code: u64,
    ) -> Result<Option<crate::jit::entries::Entry>, crate::Error> {
        self.engine.tiering.entry(code)
    }

    #[cfg(test)]
    fn push_frame(&mut self) {
        self.frames.push(Frame {
            caller_start: self.locals.start,
        });
        self.locals.set_start(self.locals.len);
    }

    #[cfg(test)]
    fn pop_frame(&mut self) {
        let frame = self.frames.pop().unwrap();
        self.locals.len = self.locals.start;
        self.locals.set_start(frame.caller_start);
    }

    pub(crate) fn overflow(&mut self) {
        self.error = Some(Box::new(Error::Runtime("too much recursion".into())));
    }

    #[cfg(test)]
    pub(crate) fn enter(&mut self, size: usize) {
        self.push_frame();
        if !self.locals.grow(self.locals.len + size) {
            self.overflow();
        }
    }
    pub(crate) fn retain(&mut self, value: Value) -> Value {
        if self.global.heap().retain(value) {
            return value;
        }
        self.error = Some(Box::new(Error::Runtime("invalid heap value".into())));
        Value::UNDEFINED
    }

    pub(crate) fn release_frame_locals(&mut self) {
        let start = self.locals.start;
        self.global
            .heap()
            .release_frame(&mut self.locals.values_mut()[start..]);
    }

    fn take_reference(&mut self, value: Value) -> bool {
        if self.global.heap().retain(value) {
            return true;
        }
        self.error = Some(Box::new(Error::Runtime("invalid heap value".into())));
        false
    }

    pub(crate) fn local_replace(&mut self, slot: usize, value: Value, old: Value) -> Value {
        if !self.take_reference(value) {
            let index = self.locals.start + slot;
            self.locals.values_mut()[index] = old;
            return Value::UNDEFINED;
        }
        self.global.heap().release(old);
        value
    }

    #[cfg(test)]
    pub(crate) fn exit(&mut self, value: Value) -> Value {
        let value = if self.global.heap().retain(value) {
            value
        } else {
            self.error = Some(Box::new(Error::Runtime("invalid heap value".into())));
            Value::UNDEFINED
        };
        self.release_frame_locals();
        self.pop_frame();
        value
    }
}

impl Drop for JitCtx<'_> {
    fn drop(&mut self) {
        self.flush_call_samples();
        self.locals.values_mut().fill(Value::UNDEFINED);
        self.engine.locals.exit();
    }
}

pub(crate) unsafe fn call(
    entry: crate::jit::entries::Entry,
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
        args.as_ptr()
    };
    let value = entry.call(&mut ctx, closure, receiver, argv, args.len() as u64);
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
    match ctx.error.take() {
        Some(error) => {
            global.heap().release(value);
            match (*error, ctx.error_origin.take()) {
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
    use crate::jit::helpers::{GlobalCache, Helper};

    #[test]
    fn owned_allocations_transfer_one_reference_to_frame_slots() {
        for (helper, input) in [
            (Helper::ObjectNewOwned, Value::UNDEFINED),
            (Helper::NewInstanceOwned, Value::UNDEFINED),
            (Helper::ArrayMapNewOwned, Value::number(2.0)),
        ] {
            let engine = Engine::new().unwrap();
            let heap = engine.global().heap();
            let mut ctx = JitCtx::new(&engine);
            ctx.enter(1);
            let allocate: extern "C" fn(*mut JitCtx, Value) -> Value =
                unsafe { std::mem::transmute(HELPERS[helper as usize].pointer) };
            let value = allocate(&mut ctx, input);
            assert!(ctx.error.is_none());
            ctx.locals.values_mut()[0] = value;
            heap.drain(100);
            assert!(heap.valid(value));
            ctx.exit(Value::UNDEFINED);
            heap.drain(100);
            assert!(!heap.valid(value));
        }
    }

    #[test]
    fn owned_arrays_and_splits_keep_their_elements_without_temporaries() {
        for text in ["grass", "grass|tree"] {
            let engine = Engine::new().unwrap();
            let heap = engine.global().heap();
            let mut ctx = JitCtx::new(&engine);
            ctx.enter(2);
            let source = heap.string(text).unwrap();
            let separator = heap.string("|").unwrap();
            let split: extern "C" fn(*mut JitCtx, u64, Value, *const Value, u64) -> Value =
                unsafe { std::mem::transmute(HELPERS[Helper::StringSplitOwned as usize].pointer) };
            let parts = split(&mut ctx, 0, source, &separator, 1);
            ctx.locals.values_mut()[0] = parts;
            let first = heap.get(parts, 0).unwrap();
            let array: extern "C" fn(*mut JitCtx, *const Value, u64) -> Value =
                unsafe { std::mem::transmute(HELPERS[Helper::ArrayNewOwned as usize].pointer) };
            let values = array(&mut ctx, &first, 1);
            ctx.locals.values_mut()[1] = values;
            assert!(ctx.error.is_none());
            heap.release(source);
            heap.release(separator);
            heap.drain(100);
            assert_eq!(heap.text(first).as_deref(), Some("grass"));
            assert_eq!(heap.get(values, 0), Some(first));
            ctx.exit(Value::UNDEFINED);
            heap.drain(100);
            for value in [parts, values, first, source, separator] {
                assert!(!heap.valid(value));
            }
        }
    }

    #[test]
    fn borrowed_global_misses_do_not_register_or_retain_results() {
        for helper in [Helper::GlobalGetBorrowed, Helper::GlobalGetOptionalBorrowed] {
            let engine = Engine::new().unwrap();
            let heap = engine.global().heap();
            let mut ctx = JitCtx::new(&engine);
            let value = heap.object().unwrap();
            assert!(engine.global().define("value", value));
            heap.release(value);
            let mut cache = GlobalCache::new("value");
            let read: extern "C" fn(*mut JitCtx, *mut GlobalCache) -> Value =
                unsafe { std::mem::transmute(HELPERS[helper as usize].pointer) };
            assert_eq!(read(&mut ctx, &mut cache), value);
            assert!(ctx.error.is_none());
            assert!(engine.global().set("value", Value::UNDEFINED));
            heap.drain(100);
            assert!(!heap.valid(value));
        }
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
}
