use super::*;
use crate::jit::layout::ReadLayout;
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

pub(super) extern "C" fn js_global_get_borrowed(
    ctx: *mut JitCtx,
    cache: *mut GlobalCache,
) -> Value {
    global_get(ctx, cache, false)
}

pub(super) extern "C" fn js_global_get_optional_borrowed(
    ctx: *mut JitCtx,
    cache: *mut GlobalCache,
) -> Value {
    global_get(ctx, cache, true)
}

pub(super) fn global_get(ctx: *mut JitCtx, cache: *mut GlobalCache, optional: bool) -> Value {
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
    property_get(ctx, object, name, true)
}

fn property_get(ctx: &mut JitCtx, object: Value, name: &str, owned: bool) -> Value {
    property_get_with_cache(ctx, object, name, owned, None)
}

fn property_get_with_cache(
    ctx: &mut JitCtx,
    object: Value,
    name: &str,
    owned: bool,
    cache: Option<&mut crate::heap::PropertyReadCache>,
) -> Value {
    if object == Value::GLOBAL_OBJECT {
        let value = ctx.global.get_object_raw(name).unwrap_or(Value::UNDEFINED);
        return if owned { ctx.retain(value) } else { value };
    }
    let read = match cache {
        Some(cache) => ctx.global.heap().read_property_cached(object, name, cache),
        None => ctx.global.heap().read_property(object, name),
    };
    match read {
        Some(PropertyRead::Value(value)) => return if owned { ctx.retain(value) } else { value },
        Some(PropertyRead::Native(function)) => {
            return match ctx.global.heap().native_function(function) {
                Some(value) => {
                    if !owned {
                        // The native-function cache keeps its own reference.
                        ctx.global.heap().release(value);
                    }
                    value
                }
                None => {
                    ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
                    Value::UNDEFINED
                }
            };
        }
        Some(PropertyRead::Prototype) => {
            if let Some(value) = ctx.global.heap().property(object, name) {
                return if owned { ctx.retain(value) } else { value };
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

pub(crate) const PROPERTY_CACHE_LIMIT: usize = 4;
const PROPERTY_CACHE_MISS_LIMIT: u32 = 8;

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct PropertyCacheEntry {
    shape: u32,
    index: u32,
    holder: u64,
    holder_shape: u32,
    builtin_kind: u32,
    builtin_value: u64,
}

impl PropertyCacheEntry {
    const EMPTY: Self = Self {
        shape: crate::heap::DICTIONARY,
        index: 0,
        holder: 0,
        holder_shape: crate::heap::NO_SHAPE,
        builtin_kind: 0,
        builtin_value: Value::UNDEFINED.bits(),
    };

    fn same_receiver(&self, other: &Self) -> bool {
        self.builtin_kind == other.builtin_kind
            && (self.builtin_kind != 0 || self.shape == other.shape)
    }
}

#[repr(C)]
pub(crate) struct PropertyCache {
    entry: PropertyCacheEntry,
    // The box stays allocated, including after saturation; generated code reads its nullable pointer.
    overflow: Option<Box<[PropertyCacheEntry; PROPERTY_CACHE_LIMIT - 1]>>,
    len: u32,
    megamorphic: u32,
    misses: u32,
    pub(crate) name: Box<str>,
    read: crate::heap::PropertyReadCache,
    write: crate::heap::PropertyWriteCache,
    observer: Option<(u64, crate::jit::profile::Site)>,
    reported: Option<ShapeAccess>,
}

pub(crate) const CACHE_LEN_OFFSET: i32 = std::mem::offset_of!(PropertyCache, len) as i32;
pub(crate) const CACHE_MEGAMORPHIC_OFFSET: i32 =
    std::mem::offset_of!(PropertyCache, megamorphic) as i32;
pub(crate) const CACHE_OVERFLOW_OFFSET: i32 = std::mem::offset_of!(PropertyCache, overflow) as i32;
pub(crate) const CACHE_ENTRY_OFFSET: i32 = std::mem::offset_of!(PropertyCache, entry) as i32;
pub(crate) const CACHE_ENTRY_SIZE: i64 = std::mem::size_of::<PropertyCacheEntry>() as i64;
pub(crate) const CACHE_SHAPE_OFFSET: i32 = std::mem::offset_of!(PropertyCacheEntry, shape) as i32;
pub(crate) const CACHE_INDEX_OFFSET: i32 = std::mem::offset_of!(PropertyCacheEntry, index) as i32;
pub(crate) const CACHE_HOLDER_OFFSET: i32 = std::mem::offset_of!(PropertyCacheEntry, holder) as i32;
pub(crate) const CACHE_HOLDER_SHAPE_OFFSET: i32 =
    std::mem::offset_of!(PropertyCacheEntry, holder_shape) as i32;
pub(crate) const CACHE_BUILTIN_KIND_OFFSET: i32 =
    std::mem::offset_of!(PropertyCacheEntry, builtin_kind) as i32;
pub(crate) const CACHE_BUILTIN_VALUE_OFFSET: i32 =
    std::mem::offset_of!(PropertyCacheEntry, builtin_value) as i32;

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
    pub(crate) fn new(
        name: &str,
        observer: Option<(u64, crate::jit::profile::Site)>,
        megamorphic: bool,
    ) -> Self {
        Self {
            entry: PropertyCacheEntry::EMPTY,
            overflow: None,
            len: 0,
            megamorphic: u32::from(megamorphic),
            misses: 0,
            read: Default::default(),
            write: Default::default(),
            name: name.into(),
            observer,
            reported: None,
        }
    }

    fn insert(&mut self, entry: PropertyCacheEntry) {
        if self.len == 0 {
            self.entry = entry;
            self.len = 1;
            return;
        }
        if self.entry.same_receiver(&entry) {
            self.entry = entry;
            return;
        }
        if let Some(overflow) = self.overflow.as_mut() {
            for current in overflow.iter_mut().take(self.len as usize - 1) {
                if current.same_receiver(&entry) {
                    *current = entry;
                    return;
                }
            }
        }
        if self.len as usize == PROPERTY_CACHE_LIMIT {
            self.megamorphic = 1;
            return;
        }
        let overflow = self
            .overflow
            .get_or_insert_with(|| Box::new([PropertyCacheEntry::EMPTY; PROPERTY_CACHE_LIMIT - 1]));
        overflow[self.len as usize - 1] = entry;
        self.len += 1;
    }

    fn lookup(&self, heap: &crate::heap::Heap, receiver: Value) -> Option<Value> {
        if self.megamorphic != 0 {
            return None;
        }
        let entries = std::iter::once(&self.entry)
            .take(self.len.min(1) as usize)
            .chain(
                self.overflow
                    .iter()
                    .flat_map(|entries| entries.iter().take(self.len.saturating_sub(1) as usize)),
            );
        for entry in entries {
            if entry.builtin_kind != 0 {
                if heap.has_kind(receiver, entry.builtin_kind) {
                    return Some(Value::from_bits(entry.builtin_value));
                }
                continue;
            }
            if let Some(value) = heap.cached_property_hit(
                receiver,
                entry.shape,
                entry.holder,
                entry.holder_shape,
                entry.index,
            ) {
                return Some(value);
            }
        }
        None
    }

    fn fill(
        &mut self,
        engine: &crate::Engine,
        heap: &crate::heap::Heap,
        receiver: Value,
    ) -> Option<Value> {
        if self.megamorphic != 0 {
            return None;
        }
        if let Some((kind, function)) = heap.builtin_property(receiver, &self.name) {
            let value = heap.native_function(function)?;
            // The engine's native-function cache owns the reference backing this borrowed value.
            heap.release(value);
            self.insert(PropertyCacheEntry {
                builtin_kind: kind,
                builtin_value: value.bits(),
                ..PropertyCacheEntry::EMPTY
            });
            self.observe(engine, PropertyAccess::Get, ShapeAccess::Varying);
            return Some(value);
        }
        if let Some(found) = heap.cached_property(receiver, &self.name) {
            self.insert(PropertyCacheEntry {
                shape: found.shape,
                index: found.index,
                holder: found.holder,
                holder_shape: found.holder_shape,
                ..PropertyCacheEntry::EMPTY
            });
            self.observe(
                engine,
                PropertyAccess::Get,
                ShapeAccess::Read(ReadLayout {
                    shape: found.shape,
                    index: found.index,
                    storage: (found.holder != 0).then_some(crate::jit::layout::RelatedStorage {
                        identity: found.holder,
                        shape: found.holder_shape,
                    }),
                }),
            );
            return Some(found.value);
        }
        self.misses += 1;
        if self.misses == PROPERTY_CACHE_MISS_LIMIT {
            self.megamorphic = 1;
        }
        self.observe(engine, PropertyAccess::Get, ShapeAccess::Varying);
        None
    }

    fn observe(&mut self, engine: &crate::Engine, property: PropertyAccess, access: ShapeAccess) {
        let Some((owner, site)) = self.observer else {
            return;
        };
        let access = if self.megamorphic != 0 {
            ShapeAccess::Megamorphic
        } else {
            access
        };
        if self.reported == Some(ShapeAccess::Megamorphic)
            || self.reported == Some(access)
            || self.reported == Some(ShapeAccess::Varying) && access != ShapeAccess::Megamorphic
        {
            return;
        }
        let access = self
            .reported
            .map_or(access, |reported| reported.merge(access));
        if self.reported == Some(access) {
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
    if let Some(value) = cache.lookup(context.global.heap(), object) {
        return value;
    }
    if let Some(value) = cache.fill(context.engine, context.global.heap(), object) {
        return value;
    }
    property_get_with_cache(
        unsafe { &mut *ctx },
        object,
        &cache.name,
        false,
        Some(&mut cache.read),
    )
}

pub(super) extern "C" fn js_property_get_megamorphic(
    ctx: *mut JitCtx,
    object: Value,
    cache: *mut PropertyCache,
) -> Value {
    let cache = unsafe { &mut *cache };
    property_get_with_cache(
        unsafe { &mut *ctx },
        object,
        &cache.name,
        false,
        Some(&mut cache.read),
    )
}

pub(super) extern "C" fn js_property_get_baseline(
    ctx: *mut JitCtx,
    object: Value,
    cache: *mut PropertyCache,
) -> Value {
    let value = js_property_get_cached(ctx, object, cache);
    unsafe { &mut *ctx }.retain(value)
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
        crate::heap::PropertyWrite::Addition => ShapeAccess::Addition,
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
        Some(IndexRead::Borrowed(value)) => ctx.retain(value),
        Some(IndexRead::Character(character)) => index_get_character(ctx, character),
        None => index_get_slow(ctx, array, Value::number(index)),
    }
}

#[cold]
#[inline(never)]
pub(super) fn index_get_character(ctx: &mut JitCtx, character: char) -> Value {
    match ctx.global.heap().string(character.encode_utf8(&mut [0; 4])) {
        Some(value) => value,
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
            let Some(number) = crate::builtins::typed_array::numeric(ctx, value) else {
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
        Some(value) => value,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::{Engine, EngineOptions};

    fn engine(source: &str) -> Engine {
        let mut engine = Engine::with_options(EngineOptions {
            inlining: false,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "read-cache.js").unwrap();
        engine
    }

    #[test]
    fn cache_feedback_merges_alternating_shapes() {
        let engine = engine("");
        let site = crate::jit::profile::Site {
            inline_context: 0,
            source: 0,
            start: 0,
            end: 0,
        };
        let mut cache = PropertyCache::new("x", Some((0, site)), false);
        for shape in [10, 20, 10, 20] {
            cache.observe(
                &engine,
                PropertyAccess::Set,
                ShapeAccess::Overwrite { shape, index: 0 },
            );
        }
        assert_eq!(cache.reported, Some(ShapeAccess::Varying));

        let mut cache = PropertyCache::new("x", Some((0, site)), false);
        for shape in [10, 20, 10, 20] {
            cache.observe(
                &engine,
                PropertyAccess::Get,
                ShapeAccess::Read(ReadLayout::own(shape, 0)),
            );
        }
        let reported = cache.reported.unwrap();
        assert_eq!(reported.read(Some(10)), Some(ReadLayout::own(10, 0)));
        assert_eq!(reported.read(Some(20)), Some(ReadLayout::own(20, 0)));
        cache.megamorphic = 1;
        cache.observe(
            &engine,
            PropertyAccess::Get,
            ShapeAccess::Read(ReadLayout::own(10, 0)),
        );
        assert_eq!(cache.reported, Some(ShapeAccess::Megamorphic));
    }

    #[test]
    fn read_cache_keeps_four_shapes_and_stops_refilling_on_overflow() {
        let engine = engine(
            "var a={x:1}; var b={pad:0,x:2}; var c={x:3,tail:0}; var d={one:0,two:0,x:4}; var e={x:5,last:0,other:0};",
        );
        let heap = engine.global().heap();
        let mut cache = PropertyCache::new("x", None, false);
        for (index, name) in ["a", "b", "c", "d"].into_iter().enumerate() {
            let object = engine.global().get_raw(name).unwrap();
            assert_eq!(
                cache.fill(&engine, heap, object),
                Some(Value::number((index + 1) as f64))
            );
            assert_eq!(cache.len, (index + 1) as u32);
            assert_eq!(cache.megamorphic, 0);
            assert_eq!(cache.overflow.is_none(), index == 0);
        }
        for name in ["d", "c", "b", "a"] {
            let object = engine.global().get_raw(name).unwrap();
            assert_eq!(cache.lookup(heap, object), heap.property(object, "x"));
            assert_eq!(cache.len, 4);
            assert_eq!(cache.megamorphic, 0);
        }
        let fifth = engine.global().get_raw("e").unwrap();
        assert_eq!(cache.fill(&engine, heap, fifth), Some(Value::number(5.0)));
        assert_eq!(cache.megamorphic, 1);
        let first = engine.global().get_raw("a").unwrap();
        assert_eq!(cache.fill(&engine, heap, first), None);
        assert_eq!(cache.lookup(heap, first), None);
        assert_eq!(cache.len, 4);
    }

    #[test]
    fn prototype_updates_refresh_an_entry_without_spending_another_shape() {
        let engine = engine(
            "function C(){} C.prototype.x=1; var a=new C(); function update(){C.prototype.pad=0;C.prototype.x=2;}",
        );
        let object = engine.global().get_raw("a").unwrap();
        let mut cache = PropertyCache::new("x", None, false);
        assert_eq!(
            cache.fill(&engine, engine.global().heap(), object),
            Some(Value::number(1.0))
        );
        let old_holder_shape = cache.entry.holder_shape;
        assert_eq!(
            cache.lookup(engine.global().heap(), object),
            Some(Value::number(1.0))
        );
        engine.call("update", &[]).unwrap();
        assert_eq!(cache.lookup(engine.global().heap(), object), None);
        assert_eq!(
            cache.fill(&engine, engine.global().heap(), object),
            Some(Value::number(2.0))
        );
        assert_ne!(cache.entry.holder_shape, old_holder_shape);
        assert_eq!(
            cache.lookup(engine.global().heap(), object),
            Some(Value::number(2.0))
        );
        assert_eq!(cache.len, 1);
        assert!(cache.overflow.is_none());
    }

    #[test]
    fn function_prototype_reads_cache_and_follow_mutations() {
        let engine = engine(
            "function f(){} function g(){} var plain={};
            function replace(){Function.prototype.apply=8;}
            function grow(){Function.prototype.extra=1;}
            function shadow(){f.apply=9;}",
        );
        let f = engine.global().get_raw("f").unwrap();
        let g = engine.global().get_raw("g").unwrap();
        let plain = engine.global().get_raw("plain").unwrap();
        let heap = engine.global().heap();
        let mut cache = PropertyCache::new("apply", None, false);
        let method = heap.property(f, "apply").unwrap();
        assert_eq!(cache.fill(&engine, heap, f), Some(method));
        assert_eq!(cache.lookup(heap, g), Some(method));
        assert_eq!(cache.lookup(heap, plain), None);
        engine.call("replace", &[]).unwrap();
        assert_eq!(cache.lookup(heap, f), Some(Value::number(8.0)));
        engine.call("grow", &[]).unwrap();
        assert_eq!(cache.lookup(heap, f), None);
        assert_eq!(cache.fill(&engine, heap, f), Some(Value::number(8.0)));
        engine.call("shadow", &[]).unwrap();
        assert_eq!(cache.lookup(heap, f), None);
        assert_eq!(cache.lookup(heap, g), Some(Value::number(8.0)));
        assert_eq!(cache.fill(&engine, heap, f), Some(Value::number(9.0)));
        assert_eq!(cache.megamorphic, 0);
    }

    #[test]
    fn function_prototype_cache_preserves_lazy_constructor_prototypes() {
        let engine = engine(
            "Function.prototype.prototype=123; function f(){}
            function C(){} C.prototype=Function.prototype; var inherited=new C();",
        );
        let heap = engine.global().heap();
        let f = engine.global().get_raw("f").unwrap();
        let inherited = engine.global().get_raw("inherited").unwrap();
        let mut cache = PropertyCache::new("prototype", None, false);
        assert_eq!(
            cache.fill(&engine, heap, inherited),
            Some(Value::number(123.0))
        );
        assert_eq!(cache.lookup(heap, f), None);
        assert_eq!(cache.fill(&engine, heap, f), None);
        let prototype = heap.property(f, "prototype").unwrap();
        assert!(prototype.is_object());
        assert_eq!(cache.fill(&engine, heap, f), Some(prototype));
        assert_eq!(cache.lookup(heap, f), Some(prototype));
    }

    #[test]
    fn builtin_cache_hits_require_the_receiver_kind() {
        let engine = engine("var a=[1]; var b=[2]; var s='text'; var o={slice:7};");
        let heap = engine.global().heap();
        let a = engine.global().get_raw("a").unwrap();
        let b = engine.global().get_raw("b").unwrap();
        let s = engine.global().get_raw("s").unwrap();
        let o = engine.global().get_raw("o").unwrap();
        let mut cache = PropertyCache::new("slice", None, false);
        assert_eq!(cache.lookup(heap, a), None);
        let array_method = cache.fill(&engine, heap, a).unwrap();
        assert_eq!(cache.lookup(heap, b), Some(array_method));
        assert_eq!(cache.lookup(heap, s), None);
        assert_eq!(cache.lookup(heap, o), None);
        assert_eq!(cache.lookup(heap, Value::NULL), None);
        let string_method = cache.fill(&engine, heap, s).unwrap();
        assert_ne!(array_method, string_method);
        assert_eq!(cache.lookup(heap, s), Some(string_method));
        assert_eq!(cache.lookup(heap, a), Some(array_method));
        assert_eq!(cache.fill(&engine, heap, o), Some(Value::number(7.0)));
        assert_eq!(cache.lookup(heap, o), Some(Value::number(7.0)));
    }

    #[test]
    fn repeated_uncacheable_reads_switch_to_megamorphic_mode() {
        let engine = engine("var dictionary={x:7,0:0};");
        let object = engine.global().get_raw("dictionary").unwrap();
        let mut cache = PropertyCache::new("x", None, false);
        for _ in 0..PROPERTY_CACHE_MISS_LIMIT {
            assert_eq!(cache.fill(&engine, engine.global().heap(), object), None);
        }
        assert_eq!(cache.megamorphic, 1);
        assert_eq!(cache.len, 0);
        assert!(cache.overflow.is_none());
        let misses = cache.misses;
        cache.fill(&engine, engine.global().heap(), object);
        assert_eq!(cache.misses, misses);
    }
}
