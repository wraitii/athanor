use std::cell::Cell;

use crate::Engine;
use crate::jit::entries::zeroed_cells;
use crate::jit::helpers::HELPERS;
use crate::jit::profile::CallSite;

use crate::environment::Environment;
use crate::error::Error;
use crate::value::Value;

/// A frame entered through the runtime rather than by generated code: generator resumption
/// and tests. Generated frames keep the caller's state in their own SSA values instead.
struct Frame {
    caller_start: usize,
    caller_floor: usize,
    resume_state: usize,
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

    fn values(&self) -> &[Value] {
        unsafe { std::slice::from_raw_parts(self.storage, self.len) }
    }

    fn values_mut(&mut self) -> &mut [Value] {
        unsafe { std::slice::from_raw_parts_mut(self.storage, self.len) }
    }

    fn set_start(&mut self, start: usize) {
        self.start = start;
        self.base = self.storage.wrapping_add(start);
    }

    fn grow(&mut self, len: usize) -> bool {
        if len > self.limit {
            return false;
        }
        let old = self.len;
        self.len = len;
        self.values_mut()[old..].fill(Value::UNDEFINED);
        true
    }

    fn frame(&self) -> &[Value] {
        &self.values()[self.start..]
    }

    fn reset_frame(&mut self) {
        let start = self.start;
        self.values_mut()[start..].fill(Value::UNDEFINED);
    }
}

/// Values owned until the next clear. Generated code pushes and pops in place: `storage` is
/// always full, `len` counts live entries and `floor` is the current frame's first entry.
#[repr(C)]
pub(crate) struct TempStack {
    values: *mut Value,
    len: usize,
    capacity: usize,
    floor: usize,
    storage: Vec<Value>,
}

impl TempStack {
    fn new() -> Self {
        let mut storage = vec![Value::UNDEFINED; 64];
        Self {
            values: storage.as_mut_ptr(),
            len: 0,
            capacity: storage.len(),
            floor: 0,
            storage,
        }
    }

    fn push(&mut self, value: Value) {
        if self.len == self.capacity {
            self.storage.resize(self.capacity * 2, Value::UNDEFINED);
            self.values = self.storage.as_mut_ptr();
            self.capacity = self.storage.len();
        }
        self.storage[self.len] = value;
        self.len += 1;
    }

    fn above(&self, base: usize) -> &[Value] {
        &self.storage[base..self.len]
    }

    fn top(&self) -> Option<Value> {
        (self.len > self.floor).then(|| self.storage[self.len - 1])
    }

    fn pop_if(&mut self, value: Value) -> bool {
        let popped = self.top() == Some(value);
        self.len -= usize::from(popped);
        popped
    }
}

#[repr(C)]
pub(crate) struct JitCtx<'a> {
    pub(crate) global: Environment,
    pub(crate) error: Option<Box<Error>>,
    pub(crate) error_origin: Option<(String, u32, u32)>,
    exception: Value,
    frames: Vec<Frame>,
    locals: LocalStack,
    stack_limit: usize,
    temps: TempStack,
    pub(super) pack_buffers: Vec<Vec<Value>>,
    index_root: [Value; 3],
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
pub(crate) const INDEX_ROOT_OFFSET: i32 = std::mem::offset_of!(JitCtx<'static>, index_root) as i32;
const TEMPS_OFFSET: usize = std::mem::offset_of!(JitCtx<'static>, temps);
pub(crate) const TEMPS_VALUES_OFFSET: i32 =
    (TEMPS_OFFSET + std::mem::offset_of!(TempStack, values)) as i32;
pub(crate) const TEMPS_LEN_OFFSET: i32 =
    (TEMPS_OFFSET + std::mem::offset_of!(TempStack, len)) as i32;
pub(crate) const TEMPS_CAPACITY_OFFSET: i32 =
    (TEMPS_OFFSET + std::mem::offset_of!(TempStack, capacity)) as i32;
pub(crate) const TEMPS_FLOOR_OFFSET: i32 =
    (TEMPS_OFFSET + std::mem::offset_of!(TempStack, floor)) as i32;
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
            frames: Vec::new(),
            locals: LocalStack::new(engine.locals.enter()),
            stack_limit: stack_pointer.saturating_sub(NATIVE_STACK_BUDGET),
            temps: TempStack::new(),
            pack_buffers: Vec::new(),
            index_root: [Value::UNDEFINED; 3],
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

    pub(crate) fn resume(&mut self, locals: Vec<Value>, state: usize) -> Result<(), Vec<Value>> {
        self.push_frame(state);
        let start = self.locals.len;
        if !self.locals.grow(start + locals.len()) {
            self.pop_frame();
            self.overflow();
            return Err(locals);
        }
        self.locals.values_mut()[start..].copy_from_slice(&locals);
        Ok(())
    }

    fn push_frame(&mut self, resume_state: usize) {
        self.frames.push(Frame {
            caller_start: self.locals.start,
            caller_floor: self.temps.floor,
            resume_state,
        });
        self.locals.set_start(self.locals.len);
        self.temps.floor = self.temps.len;
    }

    fn pop_frame(&mut self) {
        let frame = self.frames.pop().unwrap();
        self.locals.len = self.locals.start;
        self.locals.set_start(frame.caller_start);
        self.temps.floor = frame.caller_floor;
    }

    pub(crate) fn overflow(&mut self) {
        self.error = Some(Box::new(Error::Runtime("too much recursion".into())));
    }

    pub(crate) fn resume_state(&self) -> usize {
        self.frames.last().unwrap().resume_state
    }

    pub(crate) fn local_get(&mut self, slot: usize) -> Value {
        let value = self
            .locals
            .frame()
            .get(slot)
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
        let locals = self.locals.frame().to_vec();
        self.locals.reset_frame();
        self.pop_frame();
        self.global
            .heap()
            .generator_suspend(generator, state, locals);
        value
    }

    #[cfg(test)]
    pub(crate) fn enter(&mut self, size: usize) {
        self.push_frame(0);
        if !self.locals.grow(self.locals.len + size) {
            self.overflow();
        }
    }
    pub(crate) fn own(&mut self, value: Value) -> Value {
        if (value.is_object() && value != Value::GLOBAL_OBJECT) || value.is_string() {
            self.temps.push(value);
        }
        value
    }

    /// Turns a temporary result into a reference owned by the caller.
    pub(crate) fn disown(&mut self, value: Value) -> Value {
        if value == Value::GLOBAL_OBJECT
            || self.temps.pop_if(value)
            || self.global.heap().retain(value)
        {
            return value;
        }
        self.error = Some(Box::new(Error::Runtime("invalid heap value".into())));
        Value::UNDEFINED
    }

    pub(crate) fn retain_temp(&mut self, value: Value) -> Value {
        if !self.global.heap().retain(value) {
            self.error = Some(Box::new(Error::Runtime("invalid heap value".into())));
            return Value::UNDEFINED;
        }
        self.own(value)
    }
    pub(crate) fn clear(&mut self) {
        let base = self.temps.floor;
        self.global.heap().release_all(self.temps.above(base));
        self.temps.len = base;
    }
    pub(crate) fn root_index_read(&mut self, receiver: Value, key: Value, result: Value) -> bool {
        let heap = self.global.heap();
        if !heap.supports_indexed_storage(
            receiver,
            key,
            crate::heap::IndexedStorageContract::DirectNumericIndexStorage,
        ) {
            return false;
        }
        let values = [receiver, key, result];
        for (retained, value) in values.into_iter().enumerate() {
            if !heap.retain(value) {
                for value in values.into_iter().take(retained) {
                    heap.release(value);
                }
                return false;
            }
        }
        let old = std::mem::replace(&mut self.index_root, values);
        heap.release_all(&old);
        true
    }

    /// Only call-free functions root an indexed read, so at most one frame owns the root.
    pub(crate) fn release_index_root(&mut self) {
        let old = std::mem::replace(&mut self.index_root, [Value::UNDEFINED; 3]);
        self.global.heap().release_all(&old);
    }

    pub(crate) fn release_frame_locals(&mut self) {
        let start = self.locals.start;
        self.global
            .heap()
            .release_frame(&mut self.locals.values_mut()[start..]);
    }

    fn take_ownership(&mut self, value: Value) -> bool {
        if self.temps.top() == Some(value) {
            self.temps.len -= 1;
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
            let index = self.locals.start + slot;
            self.locals.values_mut()[index] = old;
            return Value::UNDEFINED;
        }
        self.global.heap().release(old);
        value
    }

    pub(crate) fn local_set(&mut self, slot: usize, value: Value) -> Value {
        if !self.take_ownership(value) {
            return Value::UNDEFINED;
        }
        let index = self.locals.start + slot;
        if index >= self.locals.len && !self.locals.grow(index + 1) {
            self.global.heap().release(value);
            self.overflow();
            return Value::UNDEFINED;
        }
        let old = std::mem::replace(&mut self.locals.values_mut()[index], value);
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
        self.release_frame_locals();
        self.clear();
        self.release_index_root();
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
            assert_eq!(ctx.temps.len, 0);
            ctx.locals.values_mut()[0] = value;
            ctx.clear();
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
            assert_eq!(ctx.temps.len, 0);
            heap.release(source);
            heap.release(separator);
            ctx.clear();
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
            assert_eq!(ctx.temps.len, 0);
            assert!(engine.global().set("value", Value::UNDEFINED));
            ctx.clear();
            heap.drain(100);
            assert!(!heap.valid(value));
        }
    }

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
        assert!(ctx.root_index_read(first_array, key, first));
        heap.release(first_array);
        heap.release(first);
        heap.drain(100);
        assert!(heap.valid(first_array) && heap.valid(first));
        assert!(ctx.root_index_read(second_array, key, second));
        assert!(ctx.index_root[0] != Value::UNDEFINED);
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
        assert!(!ctx.root_index_read(array, key, method));
        assert!(ctx.index_root[0] == Value::UNDEFINED);
        heap.release(array);
        heap.release(key);
        heap.release(method);
        ctx.exit(Value::UNDEFINED);
    }
}
