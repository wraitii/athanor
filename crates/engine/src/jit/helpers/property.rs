use super::*;
use crate::jit::profile::{PropertyAccess, ShapeAccess};

#[repr(C)]
pub(crate) struct GlobalCache {
    pub(crate) identity: *const (),
    pub(crate) revision: u64,
    pub(crate) slot: *const Value,
    name: Box<str>,
    // Keep the environment identity and its parent chain alive while generated code retains slots.
    owner: Option<crate::environment::Environment>,
}

impl GlobalCache {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            identity: std::ptr::null(),
            revision: 0,
            slot: std::ptr::null(),
            name: name.into(),
            owner: None,
        }
    }

    fn resolve(&mut self, global: &crate::environment::Environment) -> Option<Value> {
        self.owner = Some(global.clone());
        self.identity = global.identity();
        self.revision = global.revision();
        self.slot = global.binding_slot(&self.name).unwrap_or(std::ptr::null());
        (!self.slot.is_null()).then(|| unsafe { *self.slot })
    }
}

pub(super) extern "C" fn js_global_get(ctx: *mut JitCtx, cache: *mut GlobalCache) -> Value {
    global_get(ctx, cache, false, true)
}

pub(super) extern "C" fn js_global_get_optional(
    ctx: *mut JitCtx,
    cache: *mut GlobalCache,
) -> Value {
    global_get(ctx, cache, true, true)
}

pub(super) extern "C" fn js_global_get_borrowed(
    ctx: *mut JitCtx,
    cache: *mut GlobalCache,
) -> Value {
    global_get(ctx, cache, false, false)
}

pub(super) extern "C" fn js_global_get_optional_borrowed(
    ctx: *mut JitCtx,
    cache: *mut GlobalCache,
) -> Value {
    global_get(ctx, cache, true, false)
}

pub(super) fn global_get(
    ctx: *mut JitCtx,
    cache: *mut GlobalCache,
    optional: bool,
    retain: bool,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let cache = unsafe { &mut *cache };
    match cache.resolve(&ctx.global) {
        Some(Value::UNINITIALIZED) => {
            ctx.error = Some(Box::new(crate::Error::Runtime(format!(
                "binding `{}` accessed before initialization",
                cache.name
            ))));
            Value::UNDEFINED
        }
        Some(value) if retain => ctx.retain_temp(value),
        Some(value) => value,
        None if optional => Value::UNDEFINED,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(format!(
                "unknown variable `{}`",
                cache.name
            ))));
            Value::UNDEFINED
        }
    }
}

pub(super) extern "C" fn js_global_set(
    ctx: *mut JitCtx,
    ptr: *const u8,
    len: u64,
    value: Value,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if let Err(reason) = ctx.global.set_binding(name, value) {
        ctx.error = Some(Box::new(crate::Error::Runtime(format!(
            "binding `{name}` {reason}"
        ))));
    }
    value
}

pub(super) extern "C" fn js_global_initialize(
    ctx: *mut JitCtx,
    ptr: *const u8,
    len: u64,
    value: Value,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if !ctx.global.initialize_lexical(name, value) {
        ctx.error = Some(Box::new(crate::Error::Runtime(format!(
            "cannot initialize binding `{name}`"
        ))));
    }
    value
}

pub(super) extern "C" fn js_property_get(
    ctx: *mut JitCtx,
    object: Value,
    ptr: *const u8,
    len: u64,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if object == Value::GLOBAL_OBJECT {
        let value = ctx.global.get_object_raw(name).unwrap_or(Value::UNDEFINED);
        return ctx.retain_temp(value);
    }
    match ctx.global.heap().read_property(object, name) {
        Some(PropertyRead::Value(value)) => return ctx.retain_temp(value),
        Some(PropertyRead::Native(function)) => {
            return match ctx.global.heap().native_function(function) {
                Some(value) => ctx.own(value),
                None => {
                    ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
                    Value::UNDEFINED
                }
            };
        }
        Some(PropertyRead::Prototype) => {
            if let Some(value) = ctx.global.heap().property(object, name) {
                return ctx.retain_temp(value);
            }
        }
        None => {}
    }
    ctx.error = Some(Box::new(crate::Error::Runtime(format!(
        "cannot read property {name:?} of {}",
        ctx.global.heap().receiver_kind(object)
    ))));
    Value::UNDEFINED
}

/// A named property read site. Generated code compares a receiver's shape with `shape` and
/// the holder's (the receiver itself when `holder` is 0) with `holder_shape`; on a match it
/// loads the holder's shaped value at `index`. A miss refills the cache from the receiver.
/// An empty cache expects a dictionary receiver holding a property with no shape, which none do.
#[repr(C)]
pub(crate) struct PropertyCache {
    pub(crate) initialized: u32,
    pub(crate) shape: u32,
    pub(crate) index: u32,
    pub(crate) holder: u64,
    pub(crate) holder_shape: u32,
    pub(crate) builtin_kind: u32,
    pub(crate) builtin_value: u64,
    pub(crate) name: Box<str>,
    write: crate::heap::PropertyWriteCache,
    observer: Option<(u64, crate::jit::profile::Site)>,
    reported: Option<ShapeAccess>,
}

pub(crate) const CACHE_INITIALIZED_OFFSET: i32 =
    std::mem::offset_of!(PropertyCache, initialized) as i32;
pub(crate) const CACHE_SHAPE_OFFSET: i32 = std::mem::offset_of!(PropertyCache, shape) as i32;
pub(crate) const CACHE_INDEX_OFFSET: i32 = std::mem::offset_of!(PropertyCache, index) as i32;
pub(crate) const CACHE_HOLDER_OFFSET: i32 = std::mem::offset_of!(PropertyCache, holder) as i32;
pub(crate) const CACHE_HOLDER_SHAPE_OFFSET: i32 =
    std::mem::offset_of!(PropertyCache, holder_shape) as i32;
pub(crate) const CACHE_BUILTIN_KIND_OFFSET: i32 =
    std::mem::offset_of!(PropertyCache, builtin_kind) as i32;
pub(crate) const CACHE_BUILTIN_VALUE_OFFSET: i32 =
    std::mem::offset_of!(PropertyCache, builtin_value) as i32;

const CACHE_ADDITION_OFFSET: i32 = std::mem::offset_of!(PropertyCache, write) as i32
    + std::mem::offset_of!(crate::heap::PropertyWriteCache, addition) as i32;
pub(crate) const CACHE_ADDITION_SOURCE_OFFSET: i32 =
    CACHE_ADDITION_OFFSET + std::mem::offset_of!(crate::heap::PropertyAdditionCache, source) as i32;
pub(crate) const CACHE_ADDITION_DESTINATION_OFFSET: i32 = CACHE_ADDITION_OFFSET
    + std::mem::offset_of!(crate::heap::PropertyAdditionCache, destination) as i32;
pub(crate) const CACHE_ADDITION_INDEX_OFFSET: i32 =
    CACHE_ADDITION_OFFSET + std::mem::offset_of!(crate::heap::PropertyAdditionCache, index) as i32;

const CACHE_OVERWRITE_OFFSET: i32 = std::mem::offset_of!(PropertyCache, write) as i32
    + std::mem::offset_of!(crate::heap::PropertyWriteCache, overwrite) as i32;
pub(crate) const CACHE_OVERWRITE_SHAPE_OFFSET: i32 = CACHE_OVERWRITE_OFFSET
    + std::mem::offset_of!(crate::heap::PropertyOverwriteCache, shape) as i32;
pub(crate) const CACHE_OVERWRITE_INDEX_OFFSET: i32 = CACHE_OVERWRITE_OFFSET
    + std::mem::offset_of!(crate::heap::PropertyOverwriteCache, index) as i32;

impl PropertyCache {
    pub(crate) fn new(name: &str, observer: Option<(u64, crate::jit::profile::Site)>) -> Self {
        Self {
            initialized: 0,
            shape: crate::heap::DICTIONARY,
            index: 0,
            holder: 0,
            holder_shape: crate::heap::NO_SHAPE,
            builtin_kind: 0,
            builtin_value: Value::UNDEFINED.bits(),
            write: Default::default(),
            name: name.into(),
            observer,
            reported: None,
        }
    }

    fn builtin(&mut self, heap: &crate::heap::Heap, receiver: Value) -> Option<Value> {
        let (kind, function) = heap.builtin_property(receiver, &self.name)?;
        let value = heap.native_function(function)?;
        // The engine's native-function cache owns the reference backing this borrowed value.
        heap.release(value);
        self.builtin_kind = kind;
        self.builtin_value = value.bits();
        Some(value)
    }

    fn fill(
        &mut self,
        engine: &crate::Engine,
        heap: &crate::heap::Heap,
        receiver: Value,
    ) -> Option<Value> {
        if let Some(value) = self.builtin(heap, receiver) {
            self.initialized = 1;
            self.observe(engine, PropertyAccess::Get, ShapeAccess::Varying);
            return Some(value);
        }
        if let Some(found) = heap.cached_property(receiver, &self.name) {
            self.initialized = 1;
            self.shape = found.shape;
            self.index = found.index;
            self.holder = found.holder;
            self.holder_shape = found.holder_shape;
            self.observe(
                engine,
                PropertyAccess::Get,
                if found.holder == 0 {
                    ShapeAccess::Read {
                        shape: found.shape,
                        index: found.index,
                    }
                } else {
                    ShapeAccess::Varying
                },
            );
        } else {
            self.observe(engine, PropertyAccess::Get, ShapeAccess::Varying);
        }
        None
    }

    fn observe(&mut self, engine: &crate::Engine, property: PropertyAccess, access: ShapeAccess) {
        let Some((owner, site)) = self.observer else {
            return;
        };
        if self.reported == Some(ShapeAccess::Varying) || self.reported == Some(access) {
            return;
        }
        self.reported = Some(access);
        engine.tiering.observe_shape(owner, site, property, access);
    }
}

pub(super) extern "C" fn js_property_get_cached(
    ctx: *mut JitCtx,
    object: Value,
    cache: *mut PropertyCache,
) -> Value {
    let cache = unsafe { &mut *cache };
    let context = unsafe { &*ctx };
    if let Some(value) = cache.fill(context.engine, context.global.heap(), object) {
        return unsafe { &mut *ctx }.retain_temp(value);
    }
    js_property_get(ctx, object, cache.name.as_ptr(), cache.name.len() as u64)
}

pub(super) extern "C" fn js_property_cache_prime(
    ctx: *mut JitCtx,
    object: Value,
    cache: *mut PropertyCache,
) -> Value {
    let context = unsafe { &*ctx };
    unsafe { &mut *cache }.fill(context.engine, context.global.heap(), object);
    Value::UNDEFINED
}

pub(super) extern "C" fn js_property_set(
    ctx: *mut JitCtx,
    object: Value,
    ptr: *const u8,
    len: u64,
    value: Value,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    property_set(ctx, object, name, value, None)
}

pub(super) extern "C" fn js_property_set_cached(
    ctx: *mut JitCtx,
    object: Value,
    cache: *mut PropertyCache,
    value: Value,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let cache = unsafe { &mut *cache };
    cache.write.last = crate::heap::PropertyWrite::None;
    let result = property_set(ctx, object, &cache.name, value, Some(&mut cache.write));
    let access = match cache.write.last {
        crate::heap::PropertyWrite::Overwrite { shape, index } => {
            ShapeAccess::Overwrite { shape, index }
        }
        _ => ShapeAccess::Varying,
    };
    cache.observe(ctx.engine, PropertyAccess::Set, access);
    result
}

pub(super) fn property_set(
    ctx: &mut JitCtx,
    object: Value,
    name: &str,
    value: Value,
    cache: Option<&mut crate::heap::PropertyWriteCache>,
) -> Value {
    if object != Value::GLOBAL_OBJECT {
        if name == "lastIndex" && ctx.global.heap().regexp_state(object).is_some() {
            if let Some(index) = value.as_number() {
                ctx.global.heap().regexp_set_last_index(object, index);
            } else {
                ctx.error = Some(Box::new(crate::Error::Runtime(
                    "RegExp lastIndex requires a number".into(),
                )));
            }
            return value;
        }
        let stored = match cache {
            Some(cache) => ctx
                .global
                .heap()
                .set_property_cached(object, name, value, cache),
            None => ctx.global.heap().set_property(object, name, value),
        };
        if stored {
            return value;
        }
        if name == "length"
            && array_index(value).is_some_and(|length| ctx.global.heap().truncate(object, length))
        {
            return value;
        }
        ctx.error = Some(Box::new(crate::Error::Runtime(format!(
            "cannot assign property {name:?} of {}",
            ctx.global.heap().receiver_kind(object)
        ))));
        return Value::UNDEFINED;
    }
    if !ctx.global.set(name, value) {
        ctx.global.define(name, value);
    }
    value
}

pub(super) extern "C" fn js_property_define_hidden(
    ctx: *mut JitCtx,
    object: Value,
    ptr: *const u8,
    len: u64,
    value: Value,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if !ctx
        .global
        .heap()
        .define_property(object, name, value, false)
    {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "cannot define property".into(),
        )));
    }
    value
}

pub(super) fn array_index(value: Value) -> Option<usize> {
    let number = value.as_number()?;
    (number.is_finite() && number >= 0.0 && number.fract() == 0.0 && number < u32::MAX as f64)
        .then_some(number as usize)
}
pub(super) fn named_array_index(name: &str) -> Option<u32> {
    name.parse::<u32>()
        .ok()
        .filter(|index| *index != u32::MAX && index.to_string() == name)
}
pub(super) fn typed_numeric_index(ctx: &JitCtx, key: Value) -> Option<Option<usize>> {
    let number = if let Some(number) = key.as_number() {
        number
    } else {
        let text = ctx.global.heap().text(key)?;
        if text == "-0" {
            return Some(None);
        }
        let number = crate::jit::conversion::string_number(&text);
        let canonical = if number == 0.0 {
            "0".to_owned()
        } else {
            dragonbox_ecma::Buffer::new().format(number).to_owned()
        };
        if text != canonical {
            return None;
        }
        number
    };
    Some(
        if number.is_finite()
            && number >= 0.0
            && number.fract() == 0.0
            && number < usize::MAX as f64
        {
            Some(number as usize)
        } else {
            None
        },
    )
}

pub(super) extern "C" fn js_index_root(
    ctx: *mut JitCtx,
    receiver: Value,
    key: Value,
    result: Value,
) -> Value {
    Value::boolean(unsafe { &mut *ctx }.root_index_read(receiver, key, result))
}

pub(super) extern "C" fn js_index_get(ctx: *mut JitCtx, array: Value, index: Value) -> Value {
    if let Some(number) = index.as_number() {
        return js_index_get_numeric(ctx, array, number);
    }
    index_get_slow(unsafe { &mut *ctx }, array, index)
}

pub(super) extern "C" fn js_index_get_numeric(ctx: *mut JitCtx, array: Value, index: f64) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().read_numeric_index(array, index) {
        Some(IndexRead::Primitive(value)) => value,
        Some(IndexRead::Borrowed(value)) => ctx.retain_temp(value),
        Some(IndexRead::Character(character)) => index_get_character(ctx, character),
        None => index_get_slow(ctx, array, Value::number(index)),
    }
}

// The caller proves a typed receiver and an integral, in-bounds index.
pub(super) extern "C" fn js_typed_index_get(ctx: *mut JitCtx, array: Value, index: u64) -> Value {
    unsafe { &*ctx }
        .global
        .heap()
        .get(array, index as usize)
        .unwrap_or(Value::UNDEFINED)
}

#[cold]
#[inline(never)]
pub(super) fn index_get_character(ctx: &mut JitCtx, character: char) -> Value {
    match ctx.global.heap().string(character.encode_utf8(&mut [0; 4])) {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "string heap exhausted".into(),
            )));
            Value::UNDEFINED
        }
    }
}

#[cold]
#[inline(never)]
pub(super) fn index_get_slow(ctx: &mut JitCtx, array: Value, index: Value) -> Value {
    if ctx.global.heap().typed_kind(array).is_some() {
        if let Some(index) = typed_numeric_index(ctx, index) {
            return index
                .and_then(|index| ctx.global.heap().get(array, index))
                .unwrap_or(Value::UNDEFINED);
        }
        let Some(name) = property_key(ctx, index) else {
            return Value::UNDEFINED;
        };
        return js_property_get(ctx, array, name.as_ptr(), name.len() as u64);
    }
    if index.is_string() && (ctx.global.heap().is_array(array) || array.is_string()) {
        let name = ctx.global.heap().text(index).unwrap();
        if let Some(index) = named_array_index(&name) {
            return js_index_get_numeric(ctx, array, index as f64);
        }
    }
    let Some(name) = property_key(ctx, index) else {
        return Value::UNDEFINED;
    };
    js_property_get(ctx, array, name.as_ptr(), name.len() as u64)
}
pub(super) extern "C" fn js_index_set(
    ctx: *mut JitCtx,
    array: Value,
    index: Value,
    value: Value,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    if ctx.global.heap().typed_kind(array).is_some() {
        if let Some(index) = typed_numeric_index(ctx, index) {
            let Some(number) = crate::jit::builtins::typed_array::numeric(ctx, value) else {
                return Value::UNDEFINED;
            };
            if let Some(index) = index {
                ctx.global.heap().typed_set(array, index, number);
            }
            return value;
        }
        let Some(name) = property_key(ctx, index) else {
            return Value::UNDEFINED;
        };
        return js_property_set(ctx, array, name.as_ptr(), name.len() as u64, value);
    }
    let index = if index.is_string() && ctx.global.heap().is_array(array) {
        let name = ctx.global.heap().text(index).unwrap();
        named_array_index(&name)
            .map(|index| Value::number(index as f64))
            .unwrap_or(index)
    } else {
        index
    };
    if ctx.global.heap().is_array(array)
        && let Some(index) = array_index(index)
    {
        if !ctx.global.heap().set(array, index, value) {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "unsupported array index or receiver".into(),
            )));
        }
        return value;
    }
    let Some(name) = property_key(ctx, index) else {
        return Value::UNDEFINED;
    };
    property_set(ctx, array, &name, value, None)
}

pub(super) extern "C" fn js_semantic_global_probe(
    ctx: *mut JitCtx,
    cache: *mut GlobalCache,
) -> Value {
    let global = &unsafe { &*ctx }.global;
    unsafe { &mut *cache }
        .resolve(global)
        .unwrap_or(Value::UNINITIALIZED)
}

pub(super) extern "C" fn js_semantic_property_probe(
    ctx: *mut JitCtx,
    object: Value,
    ptr: *const u8,
    len: u64,
) -> Value {
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if object == Value::GLOBAL_OBJECT {
        return Value::UNINITIALIZED;
    }
    match unsafe { &*ctx }.global.heap().read_property(object, name) {
        Some(PropertyRead::Value(value)) => value,
        _ => Value::UNINITIALIZED,
    }
}

pub(super) extern "C" fn js_semantic_property_probe_cached(
    ctx: *mut JitCtx,
    object: Value,
    cache: *mut PropertyCache,
) -> Value {
    let cache = unsafe { &mut *cache };
    let context = unsafe { &*ctx };
    if let Some(value) = cache.fill(context.engine, context.global.heap(), object) {
        return value;
    }
    js_semantic_property_probe(ctx, object, cache.name.as_ptr(), cache.name.len() as u64)
}

pub(super) extern "C" fn js_semantic_index_probe(
    ctx: *mut JitCtx,
    object: Value,
    key: Value,
) -> Value {
    let heap = unsafe { &*ctx }.global.heap();
    if let Some(number) = key.as_number() {
        match heap.read_numeric_index(object, number) {
            Some(IndexRead::Primitive(value) | IndexRead::Borrowed(value)) => return value,
            Some(IndexRead::Character(_)) => return Value::UNINITIALIZED,
            None => {}
        }
    }
    if heap.typed_kind(object).is_some() {
        let context = unsafe { &*ctx };
        if let Some(index) = typed_numeric_index(context, key) {
            return index
                .and_then(|index| heap.get(object, index))
                .unwrap_or(Value::UNDEFINED);
        }
    }
    if key.is_string()
        && (heap.is_array(object) || object.is_string())
        && let Some(index) = heap.text(key).as_deref().and_then(named_array_index)
    {
        return match heap.read_numeric_index(object, index as f64) {
            Some(IndexRead::Primitive(value) | IndexRead::Borrowed(value)) => value,
            _ => Value::UNINITIALIZED,
        };
    }
    if key.is_object() || key == Value::UNINITIALIZED {
        return Value::UNINITIALIZED;
    }
    let name = heap.text(key).unwrap_or_else(|| key.to_string());
    js_semantic_property_probe(ctx, object, name.as_ptr(), name.len() as u64)
}

pub(super) fn property_key(ctx: &mut JitCtx, value: Value) -> Option<String> {
    if value.is_object() {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "object-to-property-key conversion is unsupported".into(),
        )));
        return None;
    }
    Some(ctx.global.heap().text(value).unwrap_or_else(|| {
        if value.as_number() == Some(0.0) {
            "0".into()
        } else {
            value.to_string()
        }
    }))
}

pub(super) extern "C" fn js_property_key(ctx: *mut JitCtx, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let Some(name) = property_key(ctx, value) else {
        return Value::UNDEFINED;
    };
    match ctx.global.heap().string(&name) {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}
