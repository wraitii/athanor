use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::value::{TAG_OBJECT, TAG_STRING, Value};

#[derive(Clone, Default)]
pub(crate) struct Heap(Rc<RefCell<HeapData>>);

#[derive(Default)]
struct HeapData {
    slots: Vec<Slot>,
    free: Vec<usize>,
    natives: Vec<(NativeFunction, u64)>,
    regexp_cache: HashMap<(String, u8), Rc<regress::Regex>>,
    pending: VecDeque<u64>,
}

struct Slot {
    references: usize,
    payload: Payload,
}

enum Payload {
    Vacant,
    Reserved,
    Native(NativeFunction),
    String(String),
    Generator(Box<Generator>),
    RegExp(Box<RegExp>),
    Enumerator(Box<Enumerator>),
    Set(Box<Set>),
    Object(Box<Object>),
    Array(Box<Array>),
    TypedArray(Box<TypedArray>),
    Cell(Box<Cell>),
    Closure(Box<Closure>),
}

pub(crate) enum PropertyRead {
    Value(Value),
    Native(NativeFunction),
    Prototype,
}

pub(crate) const GENERATOR_DONE: usize = usize::MAX;
pub(crate) const GENERATOR_RUNNING: usize = usize::MAX - 1;

struct RegExp {
    regex: Rc<regress::Regex>,
    global: bool,
    last_index: f64,
}

struct Enumerator {
    current: Option<Value>,
    keys: Vec<String>,
    cursor: usize,
    visited: HashSet<String>,
}

struct Generator {
    code: CodeId,
    closure: Value,
    arguments: Value,
    receiver: Value,
    state: usize,
    locals: Vec<Value>,
}

#[derive(Default)]
struct PropertyTable {
    slots: Vec<PropertySlot>,
    index: HashMap<String, usize>,
    revision: u64,
}

struct PropertySlot {
    name: String,
    value: Value,
    enumerable: bool,
}

fn array_index_name(name: &str) -> Option<u32> {
    name.parse::<u32>()
        .ok()
        .filter(|index| *index != u32::MAX && index.to_string() == name)
}

impl PropertyTable {
    fn get(&self, name: &str) -> Option<&Value> {
        self.index.get(name).map(|index| &self.slots[*index].value)
    }

    fn contains_key(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn set(&mut self, name: &str, value: Value, enumerable: Option<bool>) -> Option<Value> {
        if let Some(index) = self.index.get(name).copied() {
            let slot = &mut self.slots[index];
            if let Some(enumerable) = enumerable
                && slot.enumerable != enumerable
            {
                slot.enumerable = enumerable;
                self.revision = self.revision.wrapping_add(1);
            }
            return Some(std::mem::replace(&mut slot.value, value));
        }
        self.index.insert(name.to_owned(), self.slots.len());
        self.slots.push(PropertySlot {
            name: name.to_owned(),
            value,
            enumerable: enumerable.unwrap_or(true),
        });
        self.revision = self.revision.wrapping_add(1);
        None
    }

    fn keys(&self) -> Vec<String> {
        let mut indices: Vec<_> = self
            .slots
            .iter()
            .filter_map(|slot| array_index_name(&slot.name).map(|index| (index, slot.name.clone())))
            .collect();
        indices.sort_unstable_by_key(|(index, _)| *index);
        indices
            .into_iter()
            .map(|(_, name)| name)
            .chain(
                self.slots
                    .iter()
                    .filter(|slot| array_index_name(&slot.name).is_none())
                    .map(|slot| slot.name.clone()),
            )
            .collect()
    }

    fn enumerable(&self, name: &str) -> Option<bool> {
        self.index
            .get(name)
            .map(|index| self.slots[*index].enumerable)
    }

    fn pop(&mut self) -> Option<Value> {
        let slot = self.slots.pop()?;
        self.index.remove(&slot.name);
        Some(slot.value)
    }
}

struct Object {
    properties: PropertyTable,
    error: bool,
    prototype: Option<Value>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeFunction {
    DateNow,
    ObjectKeys,
    ObjectValues,
    ObjectGetOwnPropertyNames,
    ObjectFreeze,
    ObjectIsFrozen,
    Array,
    ArrayFrom,
    ArrayIsArray,
    TypedConstructor(TypedKind),
    TypedFrom(TypedKind),
    TypedFill,
    Math(MathFunction),
    JsonParse,
    JsonStringify,
    RegExpTest,
    RegExpExec,
    StringMatch,
    HostSeedRandom,
    HostReadFile,
    Print,
    ConsoleLog,
    HostListFiles,
    HostEvaluateSource,
    Error,
    Set,
    SetAdd,
    SetHas,
    SetDelete,
    SetClear,
    GeneratorNext,
    ArrayFilter,
    ArrayMap,
    ArrayEvery,
    ArraySome,
    ArrayFind,
    ArrayForEach,
    ArrayAt,
    ArraySort,
    ArrayJoin,
    ArrayPop,
    ArrayIndexOf,
    ArrayConcat,
    ArrayPush,
    ArrayIncludes,
    ArrayReverse,
    ArraySlice,
    ArrayFlat,
    ArrayFlatMap,
    ArrayFill,
    NumberToFixed,
    NumberToString,
    StringSlice,
    StringSplit,
    StringStartsWith,
    StringReplace,
    StringPadStart,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MathFunction {
    Random,
    Abs,
    Ceil,
    Floor,
    Round,
    Trunc,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Exp,
    Log,
    Pow,
    Min,
    Max,
    Imul,
    Sign,
}

#[derive(Hash, PartialEq, Eq)]
enum SetKey {
    String(String),
    Bits(u64),
}

struct Set {
    entries: HashMap<SetKey, Value>,
}

pub(crate) const MAX_TYPED_ARRAY_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TypedKind {
    Uint8,
    Uint16,
    Uint32,
    Int8,
    Int16,
    Int32,
    Float32,
    Float64,
}

struct TypedArray {
    kind: TypedKind,
    storage: TypedStorage,
}

enum TypedStorage {
    Uint8(Vec<u8>),
    Uint16(Vec<u16>),
    Uint32(Vec<u32>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

impl TypedStorage {
    fn new(kind: TypedKind, length: usize) -> Option<Self> {
        macro_rules! allocate {
            ($variant:ident, $ty:ty) => {{
                if length.checked_mul(std::mem::size_of::<$ty>())? > MAX_TYPED_ARRAY_BYTES {
                    return None;
                }
                let mut values = Vec::<$ty>::new();
                values.try_reserve_exact(length).ok()?;
                values.resize(length, 0 as $ty);
                Some(Self::$variant(values))
            }};
        }
        match kind {
            TypedKind::Uint8 => allocate!(Uint8, u8),
            TypedKind::Uint16 => allocate!(Uint16, u16),
            TypedKind::Uint32 => allocate!(Uint32, u32),
            TypedKind::Int8 => allocate!(Int8, i8),
            TypedKind::Int16 => allocate!(Int16, i16),
            TypedKind::Int32 => allocate!(Int32, i32),
            TypedKind::Float32 => allocate!(Float32, f32),
            TypedKind::Float64 => allocate!(Float64, f64),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Uint8(values) => values.len(),
            Self::Uint16(values) => values.len(),
            Self::Uint32(values) => values.len(),
            Self::Int8(values) => values.len(),
            Self::Int16(values) => values.len(),
            Self::Int32(values) => values.len(),
            Self::Float32(values) => values.len(),
            Self::Float64(values) => values.len(),
        }
    }
    fn get(&self, index: usize) -> Option<f64> {
        match self {
            Self::Uint8(values) => values.get(index).map(|value| *value as f64),
            Self::Uint16(values) => values.get(index).map(|value| *value as f64),
            Self::Uint32(values) => values.get(index).map(|value| *value as f64),
            Self::Int8(values) => values.get(index).map(|value| *value as f64),
            Self::Int16(values) => values.get(index).map(|value| *value as f64),
            Self::Int32(values) => values.get(index).map(|value| *value as f64),
            Self::Float32(values) => values.get(index).map(|value| *value as f64),
            Self::Float64(values) => values.get(index).map(|value| *value as f64),
        }
    }
    fn integer(number: f64) -> u32 {
        if number.is_finite() && number != 0.0 {
            number.trunc().rem_euclid(4294967296.0) as u32
        } else {
            0
        }
    }
    fn set(&mut self, index: usize, number: f64) {
        match self {
            Self::Uint8(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number) as u8;
                }
            }
            Self::Uint16(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number) as u16;
                }
            }
            Self::Uint32(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number) as u32;
                }
            }
            Self::Int8(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number) as i8;
                }
            }
            Self::Int16(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number) as i16;
                }
            }
            Self::Int32(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = Self::integer(number) as i32;
                }
            }
            Self::Float32(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = number as f32;
                }
            }
            Self::Float64(values) => {
                if let Some(value) = values.get_mut(index) {
                    *value = number as f64;
                }
            }
        }
    }
    fn fill(&mut self, start: usize, end: usize, number: f64) {
        if start >= end {
            return;
        }
        match self {
            Self::Uint8(values) => values[start..end].fill(Self::integer(number) as u8),
            Self::Uint16(values) => values[start..end].fill(Self::integer(number) as u16),
            Self::Uint32(values) => values[start..end].fill(Self::integer(number) as u32),
            Self::Int8(values) => values[start..end].fill(Self::integer(number) as i8),
            Self::Int16(values) => values[start..end].fill(Self::integer(number) as i16),
            Self::Int32(values) => values[start..end].fill(Self::integer(number) as i32),
            Self::Float32(values) => values[start..end].fill(number as f32),
            Self::Float64(values) => values[start..end].fill(number as f64),
        }
    }
}

struct Array {
    elements: Vec<Value>,
    holes: HashSet<usize>,
}

struct Cell {
    value: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CodeId(pub(crate) u64);

struct Closure {
    code: CodeId,
    captures: Vec<Value>,
    lexical_this: Option<Value>,
    constructable: bool,
    class: bool,
    properties: PropertyTable,
}

pub(crate) fn regexp_flags(bits: u8) -> regress::Flags {
    regress::Flags {
        icase: bits & 1 != 0,
        multiline: bits & 2 != 0,
        dot_all: bits & 4 != 0,
        unicode: bits & 8 != 0,
        ..Default::default()
    }
}

impl Heap {
    pub(crate) fn regexp(&self, pattern: &str, flags: u8) -> Result<Value, String> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let key = (pattern.to_owned(), flags);
        let regex = if let Some(regex) = data.regexp_cache.get(&key) {
            regex.clone()
        } else {
            let regex = Rc::new(
                regress::Regex::with_flags(pattern, regexp_flags(flags))
                    .map_err(|error| error.to_string())?,
            );
            data.regexp_cache.insert(key, regex.clone());
            regex
        };
        let id = data.reserve().ok_or("heap exhausted")?;
        data.slots[(id - 1) as usize].payload = Payload::RegExp(Box::new(RegExp {
            regex,
            global: flags & 16 != 0,
            last_index: 0.0,
        }));
        Ok(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn regexp_compiled(&self, value: Value) -> Option<Rc<regress::Regex>> {
        let data = self.0.borrow();
        let regexp = data.regexp(heap_id(value)?)?;
        Some(regexp.regex.clone())
    }

    pub(crate) fn regexp_state(&self, value: Value) -> Option<(bool, f64)> {
        let data = self.0.borrow();
        let regexp = data.regexp(heap_id(value)?)?;
        Some((regexp.global, regexp.last_index))
    }

    pub(crate) fn regexp_set_last_index(&self, value: Value, index: f64) -> bool {
        let Some(id) = heap_id(value) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(regexp) = data.regexp_mut(id) else {
            return false;
        };
        regexp.last_index = index;
        true
    }

    pub(crate) fn receiver_kind(&self, value: Value) -> &'static str {
        if self.is_array(value) {
            return "array";
        }
        if self.typed_kind(value).is_some() {
            return "typed array";
        }
        if self.regexp_state(value).is_some() {
            return "regexp";
        }
        if self.closure_code(value).is_some() {
            return "function";
        }
        if self.native(value).is_some() {
            return "native function";
        }
        if self.set_size(value).is_some() {
            return "set";
        }
        if self.generator_state(value).is_some() {
            return "generator";
        }
        if value == Value::GLOBAL_OBJECT {
            return "global object";
        }
        if value.is_null() {
            return "null";
        }
        value.type_name()
    }

    pub(crate) fn typed_new(&self, kind: TypedKind, length: usize) -> Option<Value> {
        let storage = TypedStorage::new(kind, length)?;
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = data.reserve()?;
        data.slots[(id - 1) as usize].payload =
            Payload::TypedArray(Box::new(TypedArray { kind, storage }));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn typed_kind(&self, value: Value) -> Option<TypedKind> {
        let data = self.0.borrow();
        let array = data.typed_array(heap_id(value)?)?;
        Some(array.kind)
    }

    pub(crate) fn typed_set(&self, value: Value, index: usize, number: f64) -> bool {
        let Some(id) = heap_id(value) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(array) = data.typed_array_mut(id) else {
            return false;
        };
        array.storage.set(index, number);
        true
    }

    pub(crate) fn typed_fill(&self, value: Value, start: usize, end: usize, number: f64) -> bool {
        let Some(id) = heap_id(value) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(array) = data.typed_array_mut(id) else {
            return false;
        };
        array.storage.fill(start, end, number);
        true
    }

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
        data.slots[(id - 1) as usize].payload = Payload::Generator(Box::new(Generator {
            code,
            closure,
            arguments,
            receiver,
            state: 0,
            locals: Vec::new(),
        }));
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

    pub(crate) fn native_function(&self, function: NativeFunction) -> Option<Value> {
        let mut data = self.0.borrow_mut();
        let id = if let Some((_, id)) = data.natives.iter().find(|(kind, _)| *kind == function) {
            *id
        } else {
            let id = data.reserve()?;
            data.slots[(id - 1) as usize].payload = Payload::Native(function);
            data.natives.push((function, id));
            id
        };
        data.slots[(id - 1) as usize].references += 1;
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn native(&self, value: Value) -> Option<NativeFunction> {
        let data = self.0.borrow();
        let slot = data.live(heap_id(value)?)?;
        match &slot.payload {
            Payload::Native(function) => Some(*function),
            _ => None,
        }
    }

    pub(crate) fn set_new(&self) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = data.reserve()?;
        data.slots[(id - 1) as usize].payload = Payload::Set(Box::new(Set {
            entries: HashMap::default(),
        }));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn set_size(&self, value: Value) -> Option<usize> {
        let data = self.0.borrow();
        let set = data.set(heap_id(value)?)?;
        Some(set.entries.len())
    }

    fn set_key(&self, value: Value) -> Option<SetKey> {
        if !self.valid(value) {
            return None;
        }
        Some(if let Some(text) = self.text(value) {
            SetKey::String(text)
        } else if let Some(number) = value.as_number() {
            SetKey::Bits(Value::number(if number == 0.0 { 0.0 } else { number }).bits())
        } else {
            SetKey::Bits(value.bits())
        })
    }

    pub(crate) fn set_has(&self, set: Value, value: Value) -> Option<bool> {
        let key = self.set_key(value)?;
        let data = self.0.borrow();
        let set = data.set(heap_id(set)?)?;
        Some(set.entries.contains_key(&key))
    }

    pub(crate) fn set_add(&self, set: Value, value: Value) -> bool {
        let Some(key) = self.set_key(value) else {
            return false;
        };
        let Some(id) = heap_id(set) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(set) = data.set(id) else {
            return false;
        };
        if set.entries.contains_key(&key) {
            return true;
        }
        if !data.retain(value) {
            return false;
        }
        data.set_mut(id).unwrap().entries.insert(key, value);
        true
    }

    pub(crate) fn set_delete(&self, set: Value, value: Value) -> Option<bool> {
        let key = self.set_key(value)?;
        let mut data = self.0.borrow_mut();
        let set = data.set_mut(heap_id(set)?)?;
        let removed = set.entries.remove(&key);
        if let Some(value) = removed {
            data.release(value);
        }
        Some(removed.is_some())
    }

    pub(crate) fn set_clear(&self, set: Value) -> bool {
        let Some(id) = heap_id(set) else { return false };
        let mut data = self.0.borrow_mut();
        let Some(set) = data.set(id) else {
            return false;
        };
        if set.entries.is_empty() {
            return true;
        }
        let Some(disposal) = data.reserve() else {
            return false;
        };
        let entries = std::mem::take(&mut data.set_mut(id).unwrap().entries);
        let slot = &mut data.slots[(disposal - 1) as usize];
        slot.references = 0;
        slot.payload = Payload::Set(Box::new(Set { entries }));
        data.pending.push_back(disposal);
        true
    }

    pub(crate) fn object(&self) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = data.reserve()?;
        data.slots[(id - 1) as usize].payload = Payload::Object(Box::new(Object {
            properties: PropertyTable::default(),
            error: false,
            prototype: None,
        }));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn has_properties(&self, value: Value) -> bool {
        heap_id(value).is_some_and(|id| self.0.borrow().object(id).is_some())
    }

    pub(crate) fn error(&self, message: Value, name: Value) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !data.valid(message) || !data.valid(name) {
            return None;
        }
        let id = data.reserve()?;
        data.retain(message);
        data.retain(name);
        let mut properties = PropertyTable::default();
        properties.set("message", message, Some(false));
        properties.set("name", name, Some(false));
        data.slots[(id - 1) as usize].payload = Payload::Object(Box::new(Object {
            properties,
            error: true,
            prototype: None,
        }));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    /// Returned values are borrowed; native functions and lazy prototypes are resolved by the caller.
    pub(crate) fn read_property(&self, receiver: Value, name: &str) -> Option<PropertyRead> {
        use NativeFunction::*;

        if receiver.is_number() {
            return Some(PropertyRead::Native(match name {
                "toFixed" => NumberToFixed,
                "toString" => NumberToString,
                _ => return None,
            }));
        }
        let data = self.0.borrow();
        let payload = &data.live(heap_id(receiver)?)?.payload;
        let function = match payload {
            Payload::Object(object) => {
                return data.object_property(object, name).map(PropertyRead::Value);
            }
            Payload::Closure(closure) => {
                if let Some(value) = closure.properties.get(name) {
                    return Some(PropertyRead::Value(*value));
                }
                return Some(if closure.constructable && name == "prototype" {
                    PropertyRead::Prototype
                } else {
                    PropertyRead::Value(Value::UNDEFINED)
                });
            }
            Payload::Native(Array) => match name {
                "from" => ArrayFrom,
                "isArray" => ArrayIsArray,
                _ => return None,
            },
            Payload::Native(TypedConstructor(kind)) => match name {
                "from" => TypedFrom(*kind),
                _ => return None,
            },
            Payload::TypedArray(array) => match name {
                "length" => {
                    return Some(PropertyRead::Value(Value::number(
                        array.storage.len() as f64
                    )));
                }
                "fill" => TypedFill,
                _ => return None,
            },
            Payload::RegExp(regexp) => match name {
                "lastIndex" => return Some(PropertyRead::Value(Value::number(regexp.last_index))),
                "test" => RegExpTest,
                "exec" => RegExpExec,
                _ => return None,
            },
            Payload::Array(array) => match name {
                "length" => {
                    return Some(PropertyRead::Value(Value::number(
                        array.elements.len() as f64
                    )));
                }
                "filter" => ArrayFilter,
                "map" => ArrayMap,
                "every" => ArrayEvery,
                "some" => ArraySome,
                "find" => ArrayFind,
                "forEach" => ArrayForEach,
                "at" => ArrayAt,
                "sort" => ArraySort,
                "join" => ArrayJoin,
                "pop" => ArrayPop,
                "indexOf" => ArrayIndexOf,
                "concat" => ArrayConcat,
                "push" => ArrayPush,
                "includes" => ArrayIncludes,
                "reverse" => ArrayReverse,
                "slice" => ArraySlice,
                "flat" => ArrayFlat,
                "flatMap" => ArrayFlatMap,
                "fill" => ArrayFill,
                _ => return None,
            },
            Payload::String(text) => match name {
                "length" => {
                    return Some(PropertyRead::Value(Value::number(
                        text.chars().count() as f64
                    )));
                }
                "match" => StringMatch,
                "slice" => StringSlice,
                "split" => StringSplit,
                "startsWith" => StringStartsWith,
                "replace" => StringReplace,
                "padStart" => StringPadStart,
                _ => return None,
            },
            Payload::Set(set) => match name {
                "size" => {
                    return Some(PropertyRead::Value(Value::number(set.entries.len() as f64)));
                }
                "add" => SetAdd,
                "has" => SetHas,
                "delete" => SetDelete,
                "clear" => SetClear,
                _ => return Some(PropertyRead::Value(Value::UNDEFINED)),
            },
            Payload::Generator(_) => match name {
                "next" => GeneratorNext,
                _ => return Some(PropertyRead::Value(Value::UNDEFINED)),
            },
            _ => return None,
        };
        Some(PropertyRead::Native(function))
    }

    pub(crate) fn property(&self, object: Value, name: &str) -> Option<Value> {
        let id = heap_id(object)?;
        let needs_prototype = {
            let data = self.0.borrow();
            data.closure(id).is_some_and(|closure| {
                closure.constructable
                    && name == "prototype"
                    && !closure.properties.contains_key(name)
            })
        };
        if needs_prototype {
            let prototype = self.object()?;
            self.define_property(object, name, prototype, false);
            self.release(prototype);
        }
        let data = self.0.borrow();
        if let Some(closure) = data.closure(id) {
            return Some({
                closure
                    .properties
                    .get(name)
                    .copied()
                    .unwrap_or(Value::UNDEFINED)
            });
        }
        data.object_property(data.object(id)?, name)
    }

    pub(crate) fn own_property_names(
        &self,
        value: Value,
        enumerable_only: bool,
    ) -> Option<Vec<String>> {
        if value.is_number() || value == Value::TRUE || value == Value::FALSE {
            return Some(Vec::new());
        }
        let data = self.0.borrow();
        let id = heap_id(value)?;
        if let Some(text) = data.string(id) {
            let mut keys: Vec<_> = (0..text.chars().count())
                .map(|index| index.to_string())
                .collect();
            if !enumerable_only {
                keys.push("length".into());
            }
            return Some(keys);
        }
        let mut keys = data.enumeration_keys(value)?;
        if data.is_array(id) {
            keys.retain(|name| {
                array_index_name(name).is_some() || (!enumerable_only && name == "length")
            });
        } else if enumerable_only {
            keys.retain(|name| data.enumeration_presence(value, name) == Some(true));
        } else if data
            .closure(id)
            .is_some_and(|closure| closure.constructable)
        {
            if let Some(index) = keys.iter().position(|name| name == "prototype") {
                let name = keys.remove(index);
                let first_string = keys
                    .iter()
                    .take_while(|name| array_index_name(name).is_some())
                    .count();
                keys.insert(first_string, name);
            }
        }
        Some(keys)
    }

    pub(crate) fn own_properties(&self, value: Value) -> Option<Vec<(String, Value)>> {
        let data = self.0.borrow();
        let object = data.object(heap_id(value)?)?;

        Some(
            object
                .properties
                .keys()
                .into_iter()
                .filter_map(|name| {
                    (object.properties.enumerable(&name) == Some(true))
                        .then(|| (name.clone(), *object.properties.get(&name).unwrap()))
                })
                .collect(),
        )
    }

    pub(crate) fn set_property(&self, object: Value, name: &str, value: Value) -> bool {
        self.write_property(object, name, value, None)
    }

    pub(crate) fn define_property(
        &self,
        object: Value,
        name: &str,
        value: Value,
        enumerable: bool,
    ) -> bool {
        self.write_property(object, name, value, Some(enumerable))
    }

    fn write_property(
        &self,
        object: Value,
        name: &str,
        value: Value,
        enumerable: Option<bool>,
    ) -> bool {
        let Some(id) = heap_id(object) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let supported = data.object(id).is_some() || data.closure(id).is_some();
        if !supported || !data.retain(value) {
            return false;
        }
        let enumerable = if enumerable.is_none()
            && name == "prototype"
            && data.closure(id).is_some_and(|closure| {
                closure.constructable && !closure.properties.contains_key(name)
            }) {
            Some(false)
        } else {
            enumerable
        };
        let properties = if let Some(object) = data.object_mut(id) {
            &mut object.properties
        } else {
            &mut data.closure_mut(id).unwrap().properties
        };
        if let Some(old) = properties.set(name, value, enumerable) {
            data.release(old);
        }
        true
    }

    pub(crate) fn enumerate(&self, value: Value) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let (current, keys) = if value.is_nullish()
            || value.is_number()
            || value == Value::TRUE
            || value == Value::FALSE
        {
            (None, Vec::new())
        } else {
            let keys = data.enumeration_keys(value)?;
            if !data.retain(value) {
                return None;
            }
            (Some(value), keys)
        };
        let Some(id) = data.reserve() else {
            if let Some(value) = current {
                data.release(value);
            }
            return None;
        };
        data.slots[(id - 1) as usize].payload = Payload::Enumerator(Box::new(Enumerator {
            current,
            keys,
            cursor: 0,
            visited: HashSet::default(),
        }));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn enumerate_next(&self, state: Value) -> Option<Option<Value>> {
        let mut data = self.0.borrow_mut();
        let id = heap_id(state)?;
        loop {
            let state = data.enumerator_mut(id)?;

            let Some(current) = state.current else {
                return Some(None);
            };
            if state.cursor < state.keys.len() {
                let name = state.keys[state.cursor].clone();
                state.cursor += 1;
                if state.visited.contains(&name) {
                    continue;
                }
                let Some(enumerable) = data.enumeration_presence(current, &name) else {
                    continue;
                };
                data.enumerator_mut(id)
                    .unwrap()
                    .visited
                    .insert(name.clone());
                if !enumerable {
                    continue;
                }
                drop(data);
                return self.string(&name).map(Some);
            }
            let next = data.enumeration_prototype(current);
            let keys = next
                .and_then(|value| data.enumeration_keys(value))
                .unwrap_or_default();
            if let Some(next) = next {
                if !data.retain(next) {
                    return None;
                }
            }
            let state = data.enumerator_mut(id).unwrap();
            state.current = next;
            state.keys = keys;
            state.cursor = 0;
            data.release(current);
        }
    }

    pub(crate) fn has_property(&self, value: Value, name: &str) -> Option<bool> {
        let data = self.0.borrow();
        let id = heap_id(value)?;
        if let Some(array) = data.typed_array(id) {
            if matches!(name, "length" | "fill") {
                return Some(true);
            }
            return Some(
                array_index_name(name).is_some_and(|index| (index as usize) < array.storage.len()),
            );
        }
        if let Some(closure) = data.closure(id) {
            return Some({
                closure.properties.contains_key(name)
                    || (closure.constructable && name == "prototype")
            });
        }
        if let Some(array) = data.array(id) {
            if matches!(
                name,
                "length"
                    | "filter"
                    | "map"
                    | "every"
                    | "some"
                    | "find"
                    | "forEach"
                    | "at"
                    | "sort"
                    | "join"
                    | "pop"
                    | "indexOf"
                    | "concat"
                    | "push"
                    | "includes"
                    | "reverse"
                    | "slice"
                    | "flat"
                    | "flatMap"
                    | "fill"
            ) {
                return Some(true);
            }
            let index = name
                .parse::<u32>()
                .ok()
                .filter(|index| *index != u32::MAX && index.to_string() == name);
            return Some(index.is_some_and(|index| {
                (index as usize) < array.elements.len() && !array.holes.contains(&(index as usize))
            }));
        }
        let mut current = value;
        loop {
            let object = data.object(heap_id(current)?)?;

            if object.properties.contains_key(name) {
                return Some(true);
            }
            let Some(prototype) = object.prototype else {
                return Some(false);
            };
            current = prototype;
        }
    }

    pub(crate) fn inherits(&self, value: Value, prototype: Value) -> bool {
        let data = self.0.borrow();
        let mut current = value;
        while let Some(object) = heap_id(current).and_then(|id| data.object(id)) {
            let Some(parent) = object.prototype else {
                return false;
            };
            if parent == prototype {
                return true;
            }
            current = parent;
        }
        false
    }

    pub(crate) fn constructed_object(&self, constructor: Value) -> Option<Value> {
        let prototype = self.property(constructor, "prototype")?;
        if prototype.is_object() && !self.has_properties(prototype) {
            return None;
        }
        let instance = self.object()?;
        if prototype.is_object() {
            let mut data = self.0.borrow_mut();
            data.retain(prototype);
            data.object_mut(heap_id(instance)?).unwrap().prototype = Some(prototype);
        }
        Some(instance)
    }

    pub(crate) fn error_text(&self, value: Value) -> Option<String> {
        let data = self.0.borrow();
        let object = data.object(heap_id(value)?)?;
        if !object.error {
            return None;
        }
        let name = object
            .properties
            .get("name")
            .copied()
            .unwrap_or(Value::UNDEFINED);
        let message = object
            .properties
            .get("message")
            .copied()
            .unwrap_or(Value::UNDEFINED);
        drop(data);
        let name = if name.is_undefined() {
            "Error".into()
        } else {
            self.text(name).unwrap_or_else(|| name.to_string())
        };
        let message = if message.is_undefined() {
            String::new()
        } else {
            self.text(message).unwrap_or_else(|| message.to_string())
        };
        Some(if name.is_empty() {
            message
        } else if message.is_empty() {
            name
        } else {
            format!("{name}: {message}")
        })
    }

    pub(crate) fn string(&self, text: &str) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = data.reserve()?;
        data.slots[(id - 1) as usize].payload = Payload::String(text.into());
        Some(Value::from_bits(TAG_STRING | id))
    }

    pub(crate) fn next_character(&self, value: Value, offset: usize) -> Option<(char, usize)> {
        if !value.is_string() {
            return None;
        }
        let data = self.0.borrow();
        let text = data.string(heap_id(value)?)?;
        let character = text.get(offset..)?.chars().next()?;
        Some((character, offset + character.len_utf8()))
    }

    pub(crate) fn string_is_empty(&self, value: Value) -> Option<bool> {
        if !value.is_string() {
            return None;
        }
        let data = self.0.borrow();
        let text = data.string(heap_id(value)?)?;
        Some(text.is_empty())
    }

    pub(crate) fn text(&self, value: Value) -> Option<String> {
        if !value.is_string() {
            return None;
        }
        let data = self.0.borrow();
        let text = data.string(heap_id(value)?)?;
        Some(text.clone())
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
        data.slots[(id - 1) as usize].payload = Payload::Array(Box::new(Array {
            elements: elements.to_vec(),
            holes: HashSet::default(),
        }));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn array_of_length(&self, length: usize) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let mut elements = Vec::new();
        elements.try_reserve_exact(length).ok()?;
        elements.resize(length, Value::UNDEFINED);
        let mut holes = HashSet::default();
        holes.try_reserve(length).ok()?;
        holes.extend(0..length);
        let id = data.reserve()?;
        data.slots[(id - 1) as usize].payload = Payload::Array(Box::new(Array { elements, holes }));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn cell(&self, value: Value) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !data.valid(value) {
            return None;
        }
        let id = data.reserve()?;
        data.retain(value);
        data.slots[(id - 1) as usize].payload = Payload::Cell(Box::new(Cell { value }));
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
        data.slots[(id - 1) as usize].payload = Payload::Closure(Box::new(Closure {
            code,
            captures: captures.to_vec(),
            lexical_this: None,
            constructable: true,
            class: false,
            properties: PropertyTable::default(),
        }));
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

    pub(crate) fn valid(&self, value: Value) -> bool {
        if !value.is_object() && !value.is_string() {
            return true;
        }
        self.0.borrow().valid(value)
    }

    pub(crate) fn retain(&self, value: Value) -> bool {
        if !value.is_object() && !value.is_string() {
            return true;
        }
        self.0.borrow_mut().retain(value)
    }

    pub(crate) fn release(&self, value: Value) {
        if !value.is_object() && !value.is_string() {
            return;
        }
        self.0.borrow_mut().release(value);
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
        let array = data.array(id)?;

        Some(
            array
                .elements
                .get(index)
                .copied()
                .unwrap_or(Value::UNDEFINED),
        )
    }

    pub(crate) fn present(&self, array: Value, index: usize) -> bool {
        let data = self.0.borrow();
        heap_id(array)
            .and_then(|id| data.array(id))
            .is_some_and(|array| index < array.elements.len() && !array.holes.contains(&index))
    }

    pub(crate) fn mark_hole(&self, array: Value, index: usize) {
        let mut data = self.0.borrow_mut();
        let array = data.array_mut(heap_id(array).unwrap()).unwrap();
        array.holes.insert(index);
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
        array.holes.remove(&index);
        if index == len {
            array.elements.push(value);
        } else {
            let old = std::mem::replace(&mut array.elements[index], value);
            data.release(old);
        }
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
        let length = array.elements.len();
        array.elements.reverse();
        for left in 0..length / 2 {
            let right = length - 1 - left;
            let left_hole = array.holes.remove(&left);
            let right_hole = array.holes.remove(&right);
            if left_hole {
                array.holes.insert(right);
            }
            if right_hole {
                array.holes.insert(left);
            }
        }
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
        array.holes.retain(|index| *index < length);
        let removed: Vec<_> = array.elements.drain(length..).collect();
        for value in removed {
            data.release(value);
        }
        true
    }

    pub(crate) fn drain(&self, mut budget: usize) {
        let mut data = self.0.borrow_mut();
        while budget > 0 {
            let Some(id) = data.pending.pop_front() else {
                break;
            };
            let Some(slot) = data.slots.get_mut((id - 1) as usize) else {
                continue;
            };
            if slot.references != 0 {
                continue;
            }
            let edge = match &mut slot.payload {
                Payload::Generator(generator) => generator.locals.pop(),
                Payload::Set(set) => set
                    .entries
                    .extract_if(|_, _| true)
                    .next()
                    .map(|(_, value)| value),
                Payload::Object(object) => object.properties.pop(),
                Payload::Array(array) => array.elements.pop(),
                Payload::Closure(closure) => {
                    closure.properties.pop().or_else(|| closure.captures.pop())
                }
                _ => None,
            };
            if let Some(value) = edge {
                data.pending.push_back(id);
                data.release(value);
            } else {
                let payload =
                    std::mem::replace(&mut data.slots[(id - 1) as usize].payload, Payload::Vacant);
                match payload {
                    Payload::Enumerator(state) => {
                        if let Some(value) = state.current {
                            data.release(value);
                        }
                    }
                    Payload::Generator(generator) => {
                        data.release(generator.closure);
                        data.release(generator.arguments);
                        data.release(generator.receiver);
                    }
                    Payload::Object(object) => {
                        if let Some(value) = object.prototype {
                            data.release(value);
                        }
                    }
                    Payload::Cell(cell) => data.release(cell.value),
                    Payload::Closure(closure) => {
                        if let Some(value) = closure.lexical_this {
                            data.release(value);
                        }
                    }
                    _ => {}
                }
                data.free.push((id - 1) as usize);
            }
            budget -= 1;
        }
    }
}

fn heap_id(value: Value) -> Option<u64> {
    let bits = value.bits();
    ((value.is_object() && bits != TAG_OBJECT) || value.is_string())
        .then_some(bits & 0x0000_ffff_ffff_ffff)
}

impl HeapData {
    fn object_property<'a>(&'a self, mut object: &'a Object, name: &str) -> Option<Value> {
        loop {
            if let Some(value) = object.properties.get(name) {
                return Some(*value);
            }
            let Some(prototype) = object.prototype else {
                return Some(Value::UNDEFINED);
            };
            object = self.object(heap_id(prototype)?)?;
        }
    }

    fn enumeration_keys(&self, value: Value) -> Option<Vec<String>> {
        let id = heap_id(value)?;
        if let Some(object) = self.object(id) {
            return Some(object.properties.keys());
        }
        if let Some(closure) = self.closure(id) {
            let mut keys = closure.properties.keys();
            if closure.constructable && !closure.properties.contains_key("prototype") {
                keys.push("prototype".into());
            }
            return Some(keys);
        }
        if let Some(array) = self.typed_array(id) {
            return Some(
                (0..array.storage.len())
                    .map(|index| index.to_string())
                    .collect(),
            );
        }
        if let Some(array) = self.array(id) {
            let mut keys: Vec<_> = (0..array.elements.len())
                .filter(|index| !array.holes.contains(index))
                .map(|index| index.to_string())
                .collect();
            keys.extend(
                [
                    "length", "filter", "map", "join", "pop", "indexOf", "concat", "push",
                    "includes", "reverse", "slice",
                ]
                .into_iter()
                .map(str::to_owned),
            );
            return Some(keys);
        }
        None
    }

    fn enumeration_presence(&self, value: Value, name: &str) -> Option<bool> {
        let id = heap_id(value)?;
        if let Some(object) = self.object(id) {
            return object.properties.enumerable(name);
        }
        if let Some(closure) = self.closure(id) {
            return closure
                .properties
                .enumerable(name)
                .or_else(|| (closure.constructable && name == "prototype").then_some(false));
        }
        if let Some(array) = self.typed_array(id) {
            return array_index_name(name)
                .and_then(|index| ((index as usize) < array.storage.len()).then_some(true));
        }
        if let Some(array) = self.array(id) {
            if matches!(
                name,
                "length"
                    | "filter"
                    | "map"
                    | "join"
                    | "pop"
                    | "indexOf"
                    | "concat"
                    | "push"
                    | "includes"
                    | "reverse"
                    | "slice"
            ) {
                return Some(false);
            }
            return array_index_name(name).and_then(|index| {
                ((index as usize) < array.elements.len()
                    && !array.holes.contains(&(index as usize)))
                .then_some(true)
            });
        }
        None
    }

    fn enumeration_prototype(&self, value: Value) -> Option<Value> {
        self.object(heap_id(value)?)
            .and_then(|object| object.prototype)
    }

    fn generator(&self, id: u64) -> Option<&Generator> {
        match &self.live(id)?.payload {
            Payload::Generator(value) => Some(value),
            _ => None,
        }
    }
    fn generator_mut(&mut self, id: u64) -> Option<&mut Generator> {
        match &mut self.live_mut(id)?.payload {
            Payload::Generator(value) => Some(value),
            _ => None,
        }
    }
    fn regexp(&self, id: u64) -> Option<&RegExp> {
        match &self.live(id)?.payload {
            Payload::RegExp(value) => Some(value),
            _ => None,
        }
    }
    fn regexp_mut(&mut self, id: u64) -> Option<&mut RegExp> {
        match &mut self.live_mut(id)?.payload {
            Payload::RegExp(value) => Some(value),
            _ => None,
        }
    }
    fn enumerator_mut(&mut self, id: u64) -> Option<&mut Enumerator> {
        match &mut self.live_mut(id)?.payload {
            Payload::Enumerator(value) => Some(value),
            _ => None,
        }
    }
    fn set(&self, id: u64) -> Option<&Set> {
        match &self.live(id)?.payload {
            Payload::Set(value) => Some(value),
            _ => None,
        }
    }
    fn set_mut(&mut self, id: u64) -> Option<&mut Set> {
        match &mut self.live_mut(id)?.payload {
            Payload::Set(value) => Some(value),
            _ => None,
        }
    }
    fn object(&self, id: u64) -> Option<&Object> {
        match &self.live(id)?.payload {
            Payload::Object(value) => Some(value),
            _ => None,
        }
    }
    fn object_mut(&mut self, id: u64) -> Option<&mut Object> {
        match &mut self.live_mut(id)?.payload {
            Payload::Object(value) => Some(value),
            _ => None,
        }
    }
    fn array(&self, id: u64) -> Option<&Array> {
        match &self.live(id)?.payload {
            Payload::Array(value) => Some(value),
            _ => None,
        }
    }
    fn array_mut(&mut self, id: u64) -> Option<&mut Array> {
        match &mut self.live_mut(id)?.payload {
            Payload::Array(value) => Some(value),
            _ => None,
        }
    }
    fn is_array(&self, id: u64) -> bool {
        self.array(id).is_some()
    }
    fn typed_array(&self, id: u64) -> Option<&TypedArray> {
        match &self.live(id)?.payload {
            Payload::TypedArray(value) => Some(value),
            _ => None,
        }
    }
    fn typed_array_mut(&mut self, id: u64) -> Option<&mut TypedArray> {
        match &mut self.live_mut(id)?.payload {
            Payload::TypedArray(value) => Some(value),
            _ => None,
        }
    }
    fn cell(&self, id: u64) -> Option<&Cell> {
        match &self.live(id)?.payload {
            Payload::Cell(value) => Some(value),
            _ => None,
        }
    }
    fn cell_mut(&mut self, id: u64) -> Option<&mut Cell> {
        match &mut self.live_mut(id)?.payload {
            Payload::Cell(value) => Some(value),
            _ => None,
        }
    }
    fn closure(&self, id: u64) -> Option<&Closure> {
        match &self.live(id)?.payload {
            Payload::Closure(value) => Some(value),
            _ => None,
        }
    }
    fn closure_mut(&mut self, id: u64) -> Option<&mut Closure> {
        match &mut self.live_mut(id)?.payload {
            Payload::Closure(value) => Some(value),
            _ => None,
        }
    }
    fn string(&self, id: u64) -> Option<&String> {
        match &self.live(id)?.payload {
            Payload::String(value) => Some(value),
            _ => None,
        }
    }
    // Handle payload zero denotes the global object; slot indices are encoded as index + 1.
    fn reserve(&mut self) -> Option<u64> {
        let id = if let Some(index) = self.free.pop() {
            index
        } else {
            let index = self.slots.len();
            if index as u64 >= (1u64 << 48) - 1 {
                return None;
            }
            self.slots.push(Slot {
                references: 0,
                payload: Payload::Vacant,
            });
            index
        };
        self.slots[id] = Slot {
            references: 1,
            payload: Payload::Reserved,
        };
        Some(id as u64 + 1)
    }

    // Zero-reference slots remain present until incremental destruction finishes.
    fn live(&self, id: u64) -> Option<&Slot> {
        self.slots
            .get(id.checked_sub(1)? as usize)
            .filter(|slot| slot.references != 0)
    }

    fn live_mut(&mut self, id: u64) -> Option<&mut Slot> {
        self.slots
            .get_mut(id.checked_sub(1)? as usize)
            .filter(|slot| slot.references != 0)
    }

    fn valid(&self, value: Value) -> bool {
        match heap_id(value) {
            Some(id) => self.live(id).is_some_and(|slot| {
                matches!(
                    (&slot.payload, value.is_string()),
                    (Payload::String(_), true)
                ) || (!value.is_string()
                    && !matches!(
                        slot.payload,
                        Payload::String(_) | Payload::Vacant | Payload::Reserved
                    ))
            }),
            None => !value.is_string(),
        }
    }

    fn retain(&mut self, value: Value) -> bool {
        let Some(id) = heap_id(value) else {
            return !value.is_string();
        };
        let Some(slot) = self.live_mut(id) else {
            return false;
        };
        slot.references += 1;
        true
    }

    fn release(&mut self, value: Value) {
        let Some(id) = heap_id(value) else { return };
        let Some(slot) = id
            .checked_sub(1)
            .and_then(|index| self.slots.get_mut(index as usize))
        else {
            return;
        };
        if matches!(slot.payload, Payload::Vacant | Payload::Reserved) {
            return;
        }
        assert!(slot.references > 0);
        slot.references -= 1;
        if slot.references == 0 {
            self.pending.push_back(id);
        }
    }
}

/// A host-owned value that keeps its runtime and any referenced allocation alive.
pub struct OwnedValue {
    value: Value,
    heap: Heap,
}

impl OwnedValue {
    pub(crate) fn from_owned(value: Value, heap: Heap) -> Self {
        Self { value, heap }
    }
    /// Returns a borrowed token; keep this handle alive and use the token only with its engine.
    pub fn value(&self) -> Value {
        self.value
    }

    pub fn text(&self) -> Option<String> {
        self.heap.text(self.value)
    }
}

impl Clone for OwnedValue {
    fn clone(&self) -> Self {
        assert!(self.heap.retain(self.value));
        Self {
            value: self.value,
            heap: self.heap.clone(),
        }
    }
}

impl Drop for OwnedValue {
    fn drop(&mut self) {
        self.heap.release(self.value);
        self.heap.drain(64);
    }
}

#[cfg(test)]
impl Payload {
    fn capture_count(&self) -> usize {
        match self {
            Self::Closure(closure) => closure.captures.len(),
            _ => panic!("expected closure"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumeration_respects_hidden_shadowing_and_releases_receivers() {
        let heap = Heap::default();
        let prototype = heap.object().unwrap();
        let object = heap.object().unwrap();
        assert!(heap.set_property(prototype, "hidden", Value::TRUE));
        assert!(heap.set_property(prototype, "inherited", Value::TRUE));
        assert!(heap.define_property(object, "hidden", Value::TRUE, false));
        assert!(heap.set_property(object, "own", Value::TRUE));
        {
            let mut data = heap.0.borrow_mut();
            assert!(data.retain(prototype));
            data.object_mut(heap_id(object).unwrap()).unwrap().prototype = Some(prototype);
        }
        let state = heap.enumerate(object).unwrap();
        heap.release(object);
        heap.release(prototype);
        let first = heap.enumerate_next(state).unwrap().unwrap();
        assert_eq!(heap.text(first).as_deref(), Some("own"));
        heap.release(first);
        let second = heap.enumerate_next(state).unwrap().unwrap();
        assert_eq!(heap.text(second).as_deref(), Some("inherited"));
        heap.release(second);
        assert_eq!(heap.enumerate_next(state), Some(None));
        heap.release(state);
        heap.drain(100);
        assert!(
            heap.0
                .borrow()
                .slots
                .iter()
                .all(|slot| !matches!(slot.payload, Payload::Object(_)))
        );
        assert!(
            heap.0
                .borrow()
                .slots
                .iter()
                .all(|slot| !matches!(slot.payload, Payload::Enumerator(_)))
        );
    }

    #[test]
    fn unfinished_enumeration_releases_receiver_and_property_values() {
        let heap = Heap::default();
        let object = heap.object().unwrap();
        let child = heap.object().unwrap();
        assert!(heap.set_property(object, "child", child));
        let state = heap.enumerate(object).unwrap();
        heap.release(object);
        heap.release(child);
        heap.drain(100);
        assert!(heap.valid(object));
        assert!(heap.valid(child));
        heap.release(state);
        heap.drain(100);
        assert!(!heap.valid(object));
        assert!(!heap.valid(child));
        assert!(
            heap.0
                .borrow()
                .slots
                .iter()
                .all(|slot| !matches!(slot.payload, Payload::Enumerator(_)))
        );
    }

    #[test]
    fn regexp_cache_shares_code_without_retaining_objects() {
        let heap = Heap::default();
        let a = heap.regexp("x+", 0).unwrap();
        let b = heap.regexp("x+", 0).unwrap();
        assert_ne!(a, b);
        assert!(Rc::ptr_eq(
            &heap.regexp_compiled(a).unwrap(),
            &heap.regexp_compiled(b).unwrap()
        ));
        heap.release(a);
        heap.release(b);
        heap.drain(100);
        assert!(
            heap.0
                .borrow()
                .slots
                .iter()
                .all(|slot| !matches!(slot.payload, Payload::RegExp(_)))
        );
        assert_eq!(heap.0.borrow().regexp_cache.len(), 1);
    }

    #[test]
    fn property_layout_revision_ignores_replacement() {
        let mut table = PropertyTable::default();
        table.set("x", Value::TRUE, None);
        let revision = table.revision;
        table.set("x", Value::FALSE, None);
        assert_eq!(table.revision, revision);
        table.set("x", Value::TRUE, Some(false));
        assert_ne!(table.revision, revision);
    }

    #[test]
    fn array_presence_distinguishes_holes_and_undefined() {
        let heap = Heap::default();
        let array = heap.array(&[Value::UNDEFINED, Value::UNDEFINED]).unwrap();
        heap.mark_hole(array, 0);
        assert_eq!(heap.has_property(array, "0"), Some(false));
        assert_eq!(heap.has_property(array, "1"), Some(true));
        assert_eq!(heap.has_property(array, "2"), Some(false));
        heap.release(array);
        heap.drain(100);
    }

    #[test]
    fn instances_retain_prototypes_and_release_their_properties() {
        let heap = Heap::default();
        let constructor = heap.closure(CodeId(1), &[]).unwrap();
        let prototype = heap.property(constructor, "prototype").unwrap();
        let child = heap.object().unwrap();
        assert!(heap.set_property(prototype, "child", child));
        heap.release(child);
        let instance = heap.constructed_object(constructor).unwrap();
        heap.release(constructor);
        heap.drain(100);
        assert!(heap.valid(prototype));
        assert_eq!(heap.property(instance, "child"), Some(child));
        heap.release(instance);
        heap.drain(100);
        assert!(!heap.valid(prototype));
        assert!(!heap.valid(child));
    }

    #[test]
    fn arrow_and_generator_receivers_are_owned_and_released() {
        let heap = Heap::default();
        let receiver = heap.object().unwrap();
        let arrow = heap.closure(CodeId(1), &[]).unwrap();
        assert!(heap.bind_this(arrow, receiver));
        heap.release(receiver);
        assert_eq!(heap.lexical_this(arrow), Some(receiver));
        assert!(heap.valid(receiver));
        assert!(!heap.constructable(arrow));
        heap.release(arrow);
        heap.drain(100);
        assert!(!heap.valid(receiver));

        let receiver = heap.object().unwrap();
        let closure = heap.closure(CodeId(2), &[]).unwrap();
        let arguments = heap.array(&[]).unwrap();
        let generator = heap
            .generator(CodeId(3), closure, arguments, receiver)
            .unwrap();
        heap.release(receiver);
        heap.release(closure);
        heap.release(arguments);
        assert_eq!(heap.generator_receiver(generator), Some(receiver));
        heap.generator_complete(generator);
        assert!(!heap.valid(receiver));
        heap.release(generator);
        heap.drain(100);
        assert!(
            heap.0
                .borrow()
                .slots
                .iter()
                .all(|slot| !matches!(slot.payload, Payload::Generator(_)))
        );
    }

    #[test]
    fn abandoned_generators_release_suspended_frames_incrementally() {
        let heap = Heap::default();
        let closure = heap.closure(CodeId(1), &[]).unwrap();
        let argument = heap.string("argument").unwrap();
        let arguments = heap.array(&[argument]).unwrap();
        let generator = heap
            .generator(CodeId(2), closure, arguments, Value::UNDEFINED)
            .unwrap();
        heap.release(closure);
        heap.release(argument);
        heap.release(arguments);
        assert_eq!(heap.generator_begin(generator), Some((0, Vec::new())));
        assert_eq!(heap.generator_state(generator), Some(GENERATOR_RUNNING));
        assert!(heap.generator_begin(generator).is_none());
        let local = heap.array(&[]).unwrap();
        heap.generator_suspend(generator, 1, vec![local]);
        heap.release(generator);
        heap.drain(1);
        assert!(!heap.valid(local));
        assert!(heap.valid(closure));
        assert!(heap.valid(arguments));
        heap.drain(100);
        assert!(!heap.valid(closure));
        assert!(!heap.valid(arguments));
        assert!(!heap.valid(argument));
        assert!(
            heap.0
                .borrow()
                .slots
                .iter()
                .all(|slot| !matches!(slot.payload, Payload::Generator(_)))
        );
    }

    #[test]
    fn sets_retain_entries_and_clear_disposes_incrementally() {
        let heap = Heap::default();
        let set = heap.set_new().unwrap();
        let first = heap.array(&[]).unwrap();
        let second = heap.array(&[]).unwrap();
        assert!(heap.set_add(set, first));
        assert!(heap.set_add(set, first));
        assert!(heap.set_add(set, second));
        heap.release(first);
        heap.release(second);
        assert_eq!(heap.set_size(set), Some(2));
        assert!(heap.set_clear(set));
        assert_eq!(heap.set_size(set), Some(0));
        assert!(heap.valid(first));
        assert!(heap.valid(second));
        heap.drain(1);
        assert_ne!(heap.valid(first), heap.valid(second));
        heap.drain(100);
        assert!(!heap.valid(first));
        assert!(!heap.valid(second));
        let value = heap.string("é😀").unwrap();
        assert!(heap.set_add(set, value));
        heap.release(value);
        assert_eq!(heap.set_delete(set, value), Some(true));
        assert!(!heap.valid(value));
        heap.release(set);
        heap.drain(100);
        assert!(
            heap.0
                .borrow()
                .slots
                .iter()
                .all(|slot| !matches!(slot.payload, Payload::Set(_)))
        );
    }

    #[test]
    fn error_properties_own_values_and_dispose_incrementally() {
        let heap = Heap::default();
        let message = heap.string("oops").unwrap();
        let name = heap.string("Error").unwrap();
        let error = heap.error(message, name).unwrap();
        heap.release(message);
        heap.release(name);
        assert!(heap.set_property(error, "message", message));
        assert_eq!(heap.error_text(error).as_deref(), Some("Error: oops"));
        let child = heap.array(&[]).unwrap();
        assert!(heap.set_property(error, "extra", child));
        heap.release(child);
        heap.release(error);
        heap.drain(1);
        assert!(!heap.valid(error));
        assert!(matches!(
            heap.0.borrow().slots[(heap_id(error).unwrap() - 1) as usize].payload,
            Payload::Object(_)
        ));
        heap.drain(100);
        assert!(!heap.valid(message));
        assert!(!heap.valid(name));
        assert!(!heap.valid(child));
        assert!(
            heap.0
                .borrow()
                .slots
                .iter()
                .all(|slot| !matches!(slot.payload, Payload::Object(_)))
        );
    }

    #[test]
    fn closures_share_cells_and_retain_captured_values() {
        let heap = Heap::default();
        let first = heap.array(&[]).unwrap();
        let cell = heap.cell(first).unwrap();
        let left = heap.closure(CodeId(1), &[cell]).unwrap();
        let right = heap.closure(CodeId(2), &[cell]).unwrap();
        heap.release(first);
        heap.release(cell);
        let captured = heap.closure_parts(left).unwrap().1[0];
        let replacement = heap.array(&[]).unwrap();
        assert!(heap.cell_set(captured, replacement));
        heap.release(replacement);
        heap.drain(100);
        assert!(!heap.valid(first));
        assert_eq!(
            heap.cell_get(heap.closure_parts(right).unwrap().1[0]),
            Some(replacement)
        );
        assert!(heap.cell_set(captured, replacement));
        heap.release(left);
        heap.drain(100);
        assert!(heap.valid(replacement));
        heap.release(right);
        heap.drain(100);
        assert!(!heap.valid(captured));
        assert!(!heap.valid(replacement));
    }

    #[test]
    fn closure_disposal_is_budgeted_by_capture_edges() {
        let heap = Heap::default();
        let cells: Vec<_> = (0..100)
            .map(|_| heap.cell(Value::UNDEFINED).unwrap())
            .collect();
        let closure = heap.closure(CodeId(1), &cells).unwrap();
        for cell in &cells {
            heap.release(*cell);
        }
        heap.release(closure);
        heap.drain(1);
        assert_eq!(
            heap.0.borrow().slots[(heap_id(closure).unwrap() - 1) as usize]
                .payload
                .capture_count(),
            99
        );
        assert!(cells.iter().any(|cell| heap.valid(*cell)));
        heap.drain(1000);
        assert!(cells.iter().all(|cell| !heap.valid(*cell)));
        assert!(
            heap.0
                .borrow()
                .slots
                .iter()
                .all(|slot| !matches!(slot.payload, Payload::Closure(_)))
        );
        assert!(
            heap.0
                .borrow()
                .slots
                .iter()
                .all(|slot| !matches!(slot.payload, Payload::Cell(_)))
        );
    }

    #[test]
    fn closure_cell_cycle_does_not_keep_heap_alive() {
        let heap = Heap::default();
        let weak = Rc::downgrade(&heap.0);
        let cell = heap.cell(Value::UNDEFINED).unwrap();
        let closure = heap.closure(CodeId(1), &[cell]).unwrap();
        assert!(heap.cell_set(cell, closure));
        heap.release(cell);
        heap.release(closure);
        heap.drain(100);
        assert!(heap.valid(cell));
        assert!(heap.valid(closure));
        drop(heap);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn closure_rejects_invalid_captures() {
        let heap = Heap::default();
        let cell = heap.cell(Value::UNDEFINED).unwrap();
        let array = heap.array(&[]).unwrap();
        assert!(heap.closure(CodeId(1), &[array]).is_none());
        heap.release(cell);
        assert!(heap.closure(CodeId(1), &[cell]).is_none());
        heap.release(array);
    }

    #[test]
    fn arrays_retain_edges_and_replace_safely() {
        let heap = Heap::default();
        let child = heap.array(&[]).unwrap();
        let parent = heap.array(&[child]).unwrap();
        heap.release(child);
        assert!(heap.set(parent, 0, child));
        heap.drain(100);
        assert_eq!(heap.get(parent, 0), Some(child));
        heap.release(parent);
        heap.drain(100);
        assert!(!heap.valid(parent));
        assert!(!heap.valid(child));
    }

    #[test]
    fn destruction_is_iterative_and_budgeted() {
        let heap = Heap::default();
        let mut root = heap.array(&[]).unwrap();
        for _ in 0..10000 {
            let parent = heap.array(&[root]).unwrap();
            heap.release(root);
            root = parent;
        }
        heap.release(root);
        heap.drain(1);
        assert!(!heap.valid(root));
        assert!(!heap.retain(root));
        assert!(matches!(
            heap.0.borrow().slots[(heap_id(root).unwrap() - 1) as usize].payload,
            Payload::Array(_)
        ));
        heap.drain(30000);
        assert!(!heap.valid(root));
        assert!(
            heap.0
                .borrow()
                .slots
                .iter()
                .all(|slot| !matches!(slot.payload, Payload::Array(_)))
        );
    }

    #[test]
    fn host_root_keeps_array_alive_and_rejects_stale_tokens() {
        let heap = Heap::default();
        let value = heap.array(&[]).unwrap();
        let root = OwnedValue::from_owned(value, heap.clone());
        let copy = root.clone();
        drop(root);
        assert!(heap.valid(value));
        drop(copy);
        assert!(!heap.valid(value));
        assert!(!heap.retain(value));
        assert_eq!(heap.get(value, 0), None);
    }

    #[test]
    fn pending_slots_are_reused_only_after_their_edges_are_released() {
        let heap = Heap::default();
        let child = heap.object().unwrap();
        let parent = heap.array(&vec![child; 100]).unwrap();
        heap.release(child);
        heap.release(parent);
        heap.drain(1);
        assert!(!heap.valid(parent));
        assert!(heap.valid(child));
        let other = heap.string("other").unwrap();
        assert_ne!(heap_id(parent), heap_id(other));
        assert!(heap.valid(child));
        heap.drain(1000);
        assert!(!heap.valid(child));
        heap.release(other);
        heap.drain(100);
    }

    #[test]
    fn freed_slots_are_reused_across_kinds_and_allocation_churn_is_bounded() {
        let heap = Heap::default();
        let array = heap.array(&[]).unwrap();
        heap.release(array);
        heap.drain(100);
        let object = heap.object().unwrap();
        assert_eq!(heap_id(array), heap_id(object));
        assert!(heap.set_property(object, "value", Value::TRUE));
        heap.release(object);
        heap.drain(100);
        for _ in 0..1000 {
            let value = heap.string("temporary").unwrap();
            heap.release(value);
        }
        heap.drain(100);
        assert_eq!(heap.0.borrow().slots.len(), 1);
    }
}
