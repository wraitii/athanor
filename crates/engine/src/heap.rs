use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::value::{TAG_OBJECT, TAG_STRING, Value};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Default)]
pub(crate) struct Heap(Rc<RefCell<HeapData>>);

#[derive(Default)]
struct HeapData {
    generators: HashMap<u64, Generator>,
    regexps: HashMap<u64, RegExp>,
    regexp_cache: HashMap<(String, u8), Rc<regress::Regex>>,
    enumerators: HashMap<u64, Enumerator>,
    natives: HashMap<u64, (usize, NativeFunction)>,
    sets: HashMap<u64, Set>,
    objects: HashMap<u64, Object>,
    strings: HashMap<u64, (usize, String)>,
    arrays: HashMap<u64, Box<Array>>,
    typed_arrays: HashMap<u64, TypedArray>,
    cells: HashMap<u64, Box<Cell>>,
    closures: HashMap<u64, Box<Closure>>,
    pending: VecDeque<u64>,
}

pub(crate) const GENERATOR_DONE: usize = usize::MAX;
pub(crate) const GENERATOR_RUNNING: usize = usize::MAX - 1;

struct RegExp {
    references: usize,
    regex: Rc<regress::Regex>,
    global: bool,
    last_index: f64,
}

struct Enumerator {
    references: usize,
    current: Option<Value>,
    keys: Vec<String>,
    cursor: usize,
    visited: HashSet<String>,
}

struct Generator {
    references: usize,
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
    references: usize,
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
    references: usize,
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
    references: usize,
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
    references: usize,
    elements: Vec<Value>,
    holes: HashSet<usize>,
}

struct Cell {
    references: usize,
    value: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CodeId(pub(crate) u64);

struct Closure {
    references: usize,
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
        let id = next_id().ok_or("heap exhausted")?;
        data.regexps.insert(
            id,
            RegExp {
                references: 1,
                regex,
                global: flags & 16 != 0,
                last_index: 0.0,
            },
        );
        Ok(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn regexp_compiled(&self, value: Value) -> Option<Rc<regress::Regex>> {
        let data = self.0.borrow();
        let regexp = data.regexps.get(&heap_id(value)?)?;
        (regexp.references != 0).then(|| regexp.regex.clone())
    }

    pub(crate) fn regexp_state(&self, value: Value) -> Option<(bool, f64)> {
        let data = self.0.borrow();
        let regexp = data.regexps.get(&heap_id(value)?)?;
        (regexp.references != 0).then_some((regexp.global, regexp.last_index))
    }

    pub(crate) fn regexp_set_last_index(&self, value: Value, index: f64) -> bool {
        let Some(id) = heap_id(value) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(regexp) = data
            .regexps
            .get_mut(&id)
            .filter(|regexp| regexp.references != 0)
        else {
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
        let id = next_id()?;
        self.0.borrow_mut().typed_arrays.insert(
            id,
            TypedArray {
                references: 1,
                kind,
                storage,
            },
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn typed_kind(&self, value: Value) -> Option<TypedKind> {
        let data = self.0.borrow();
        let array = data.typed_arrays.get(&heap_id(value)?)?;
        (array.references != 0).then_some(array.kind)
    }

    pub(crate) fn typed_set(&self, value: Value, index: usize, number: f64) -> bool {
        let Some(id) = heap_id(value) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(array) = data
            .typed_arrays
            .get_mut(&id)
            .filter(|array| array.references != 0)
        else {
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
        let Some(array) = data
            .typed_arrays
            .get_mut(&id)
            .filter(|array| array.references != 0)
        else {
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
        let id = next_id()?;
        data.retain(closure);
        data.retain(arguments);
        data.retain(receiver);
        data.generators.insert(
            id,
            Generator {
                references: 1,
                code,
                closure,
                arguments,
                receiver,
                state: 0,
                locals: Vec::new(),
            },
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn generator_receiver(&self, value: Value) -> Option<Value> {
        let data = self.0.borrow();
        let generator = data.generators.get(&heap_id(value)?)?;
        (generator.references != 0).then_some(generator.receiver)
    }

    pub(crate) fn generator_state(&self, value: Value) -> Option<usize> {
        let data = self.0.borrow();
        let generator = data.generators.get(&heap_id(value)?)?;
        (generator.references != 0).then_some(generator.state)
    }

    pub(crate) fn generator_parts(&self, value: Value) -> Option<(CodeId, Value, Value)> {
        let data = self.0.borrow();
        let generator = data.generators.get(&heap_id(value)?)?;
        (generator.references != 0).then_some((
            generator.code,
            generator.closure,
            generator.arguments,
        ))
    }

    pub(crate) fn generator_begin(&self, value: Value) -> Option<(usize, Vec<Value>)> {
        let mut data = self.0.borrow_mut();
        let generator = data.generators.get_mut(&heap_id(value)?)?;
        if generator.references == 0 || generator.state >= GENERATOR_RUNNING {
            return None;
        }
        let state = std::mem::replace(&mut generator.state, GENERATOR_RUNNING);
        Some((state, std::mem::take(&mut generator.locals)))
    }

    pub(crate) fn generator_suspend(&self, value: Value, state: usize, locals: Vec<Value>) {
        let mut data = self.0.borrow_mut();
        let generator = data.generators.get_mut(&heap_id(value).unwrap()).unwrap();
        generator.state = state;
        generator.locals = locals;
    }

    pub(crate) fn generator_complete(&self, value: Value) {
        let mut data = self.0.borrow_mut();
        let generator = data.generators.get_mut(&heap_id(value).unwrap()).unwrap();
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
        let id =
            if let Some((&id, _)) = data.natives.iter().find(|(_, (_, kind))| *kind == function) {
                id
            } else {
                let id = next_id()?;
                data.natives.insert(id, (1, function));
                id
            };
        data.natives.get_mut(&id).unwrap().0 += 1;
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn native(&self, value: Value) -> Option<NativeFunction> {
        self.0
            .borrow()
            .natives
            .get(&heap_id(value)?)
            .map(|(_, function)| *function)
    }

    pub(crate) fn set_new(&self) -> Option<Value> {
        self.drain(64);
        let id = next_id()?;
        self.0.borrow_mut().sets.insert(
            id,
            Set {
                references: 1,
                entries: HashMap::new(),
            },
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn set_size(&self, value: Value) -> Option<usize> {
        let data = self.0.borrow();
        let set = data.sets.get(&heap_id(value)?)?;
        (set.references != 0).then_some(set.entries.len())
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
        let set = data.sets.get(&heap_id(set)?)?;
        (set.references != 0).then(|| set.entries.contains_key(&key))
    }

    pub(crate) fn set_add(&self, set: Value, value: Value) -> bool {
        let Some(key) = self.set_key(value) else {
            return false;
        };
        let Some(id) = heap_id(set) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(set) = data.sets.get(&id).filter(|set| set.references != 0) else {
            return false;
        };
        if set.entries.contains_key(&key) {
            return true;
        }
        if !data.retain(value) {
            return false;
        }
        data.sets.get_mut(&id).unwrap().entries.insert(key, value);
        true
    }

    pub(crate) fn set_delete(&self, set: Value, value: Value) -> Option<bool> {
        let key = self.set_key(value)?;
        let mut data = self.0.borrow_mut();
        let set = data
            .sets
            .get_mut(&heap_id(set)?)
            .filter(|set| set.references != 0)?;
        let removed = set.entries.remove(&key);
        if let Some(value) = removed {
            data.release(value);
        }
        Some(removed.is_some())
    }

    pub(crate) fn set_clear(&self, set: Value) -> bool {
        let Some(id) = heap_id(set) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        let Some(set) = data.sets.get_mut(&id).filter(|set| set.references != 0) else {
            return false;
        };
        if set.entries.is_empty() {
            return true;
        }
        let Some(disposal) = next_id() else {
            return false;
        };
        let entries = std::mem::take(&mut set.entries);
        data.sets.insert(
            disposal,
            Set {
                references: 0,
                entries,
            },
        );
        data.pending.push_back(disposal);
        true
    }

    pub(crate) fn object(&self) -> Option<Value> {
        self.drain(64);
        let id = next_id()?;
        self.0.borrow_mut().objects.insert(
            id,
            Object {
                references: 1,
                properties: PropertyTable::default(),
                error: false,
                prototype: None,
            },
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn has_properties(&self, value: Value) -> bool {
        heap_id(value).is_some_and(|id| {
            self.0
                .borrow()
                .objects
                .get(&id)
                .is_some_and(|object| object.references != 0)
        })
    }

    pub(crate) fn error(&self, message: Value, name: Value) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !data.valid(message) || !data.valid(name) {
            return None;
        }
        let id = next_id()?;
        data.retain(message);
        data.retain(name);
        let mut properties = PropertyTable::default();
        properties.set("message", message, Some(false));
        properties.set("name", name, Some(false));
        data.objects.insert(
            id,
            Object {
                references: 1,
                properties,
                error: true,
                prototype: None,
            },
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn property(&self, object: Value, name: &str) -> Option<Value> {
        let id = heap_id(object)?;
        let needs_prototype = {
            let data = self.0.borrow();
            data.closures.get(&id).is_some_and(|closure| {
                closure.references != 0
                    && closure.constructable
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
        if let Some(closure) = data.closures.get(&id) {
            return (closure.references != 0).then(|| {
                closure
                    .properties
                    .get(name)
                    .copied()
                    .unwrap_or(Value::UNDEFINED)
            });
        }
        let mut current = object;
        loop {
            let object = data.objects.get(&heap_id(current)?)?;
            if object.references == 0 {
                return None;
            }
            if let Some(value) = object.properties.get(name) {
                return Some(*value);
            }
            let Some(prototype) = object.prototype else {
                return Some(Value::UNDEFINED);
            };
            current = prototype;
        }
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
        if let Some((references, text)) = data.strings.get(&id) {
            if *references == 0 {
                return None;
            }
            let mut keys: Vec<_> = (0..text.chars().count())
                .map(|index| index.to_string())
                .collect();
            if !enumerable_only {
                keys.push("length".into());
            }
            return Some(keys);
        }
        let mut keys = data.enumeration_keys(value)?;
        if data.arrays.contains_key(&id) {
            keys.retain(|name| {
                array_index_name(name).is_some() || (!enumerable_only && name == "length")
            });
        } else if enumerable_only {
            keys.retain(|name| data.enumeration_presence(value, name) == Some(true));
        } else if data
            .closures
            .get(&id)
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
        let object = data.objects.get(&heap_id(value)?)?;
        if object.references == 0 {
            return None;
        }
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
        let supported = data
            .objects
            .get(&id)
            .is_some_and(|object| object.references != 0)
            || data
                .closures
                .get(&id)
                .is_some_and(|closure| closure.references != 0);
        if !supported || !data.retain(value) {
            return false;
        }
        let enumerable = if enumerable.is_none()
            && name == "prototype"
            && data.closures.get(&id).is_some_and(|closure| {
                closure.constructable && !closure.properties.contains_key(name)
            }) {
            Some(false)
        } else {
            enumerable
        };
        let properties = if let Some(object) = data.objects.get_mut(&id) {
            &mut object.properties
        } else {
            &mut data.closures.get_mut(&id).unwrap().properties
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
        let Some(id) = next_id() else {
            if let Some(value) = current {
                data.release(value);
            }
            return None;
        };
        data.enumerators.insert(
            id,
            Enumerator {
                references: 1,
                current,
                keys,
                cursor: 0,
                visited: HashSet::new(),
            },
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn enumerate_next(&self, state: Value) -> Option<Option<Value>> {
        let mut data = self.0.borrow_mut();
        let id = heap_id(state)?;
        loop {
            let state = data.enumerators.get_mut(&id)?;
            if state.references == 0 {
                return None;
            }
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
                data.enumerators
                    .get_mut(&id)
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
            let state = data.enumerators.get_mut(&id).unwrap();
            state.current = next;
            state.keys = keys;
            state.cursor = 0;
            data.release(current);
        }
    }

    pub(crate) fn has_property(&self, value: Value, name: &str) -> Option<bool> {
        let data = self.0.borrow();
        let id = heap_id(value)?;
        if let Some(array) = data.typed_arrays.get(&id) {
            if array.references == 0 {
                return None;
            }
            if matches!(name, "length" | "fill") {
                return Some(true);
            }
            return Some(
                array_index_name(name).is_some_and(|index| (index as usize) < array.storage.len()),
            );
        }
        if let Some(closure) = data.closures.get(&id) {
            return (closure.references != 0).then(|| {
                closure.properties.contains_key(name)
                    || (closure.constructable && name == "prototype")
            });
        }
        if let Some(array) = data.arrays.get(&id) {
            if array.references == 0 {
                return None;
            }
            if matches!(
                name,
                "length"
                    | "filter"
                    | "map"
                    | "every"
                    | "some"
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
            let object = data.objects.get(&heap_id(current)?)?;
            if object.references == 0 {
                return None;
            }
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
        while let Some(object) = heap_id(current).and_then(|id| data.objects.get(&id)) {
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
            data.objects.get_mut(&heap_id(instance)?).unwrap().prototype = Some(prototype);
        }
        Some(instance)
    }

    pub(crate) fn error_text(&self, value: Value) -> Option<String> {
        let data = self.0.borrow();
        let object = data.objects.get(&heap_id(value)?)?;
        if object.references == 0 || !object.error {
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
        let id = next_id()?;
        self.0.borrow_mut().strings.insert(id, (1, text.into()));
        Some(Value::from_bits(TAG_STRING | id))
    }

    pub(crate) fn next_character(&self, value: Value, offset: usize) -> Option<(char, usize)> {
        if !value.is_string() {
            return None;
        }
        let data = self.0.borrow();
        let (references, text) = data.strings.get(&heap_id(value)?)?;
        if *references == 0 {
            return None;
        }
        let character = text.get(offset..)?.chars().next()?;
        Some((character, offset + character.len_utf8()))
    }

    pub(crate) fn text(&self, value: Value) -> Option<String> {
        if !value.is_string() {
            return None;
        }
        let data = self.0.borrow();
        let (references, text) = data.strings.get(&heap_id(value)?)?;
        (*references != 0).then(|| text.clone())
    }

    pub(crate) fn array(&self, elements: &[Value]) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !elements.iter().all(|value| data.valid(*value)) {
            return None;
        }
        let id = next_id()?;
        for value in elements {
            data.retain(*value);
        }
        data.arrays.insert(
            id,
            Box::new(Array {
                references: 1,
                elements: elements.to_vec(),
                holes: HashSet::new(),
            }),
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn array_of_length(&self, length: usize) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = next_id()?;
        let mut elements = Vec::new();
        elements.try_reserve_exact(length).ok()?;
        elements.resize(length, Value::UNDEFINED);
        let mut holes = HashSet::new();
        holes.try_reserve(length).ok()?;
        holes.extend(0..length);
        data.arrays.insert(
            id,
            Box::new(Array {
                references: 1,
                elements,
                holes,
            }),
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn cell(&self, value: Value) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = next_id()?;
        if !data.retain(value) {
            return None;
        }
        data.cells.insert(
            id,
            Box::new(Cell {
                references: 1,
                value,
            }),
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn closure(&self, code: CodeId, captures: &[Value]) -> Option<Value> {
        self.drain(64);
        let mut data = self.0.borrow_mut();
        if !captures.iter().all(|value| {
            heap_id(*value)
                .and_then(|id| data.cells.get(&id))
                .is_some_and(|cell| cell.references != 0)
        }) {
            return None;
        }
        let id = next_id()?;
        for value in captures {
            data.retain(*value);
        }
        data.closures.insert(
            id,
            Box::new(Closure {
                references: 1,
                code,
                captures: captures.to_vec(),
                lexical_this: None,
                constructable: true,
                class: false,
                properties: PropertyTable::default(),
            }),
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    #[cfg(test)]
    pub(crate) fn closure_parts(&self, closure: Value) -> Option<(CodeId, Vec<Value>)> {
        let data = self.0.borrow();
        let closure = data.closures.get(&heap_id(closure)?)?;
        (closure.references != 0).then(|| (closure.code, closure.captures.clone()))
    }

    pub(crate) fn bind_this(&self, closure: Value, receiver: Value) -> bool {
        let Some(id) = heap_id(closure) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        if !data
            .closures
            .get(&id)
            .is_some_and(|closure| closure.references != 0)
            || !data.retain(receiver)
        {
            return false;
        }
        let closure = data.closures.get_mut(&id).unwrap();
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
            .closures
            .get_mut(&heap_id(closure).unwrap())
            .unwrap()
            .constructable = false;
    }

    pub(crate) fn mark_class(&self, closure: Value) {
        self.0
            .borrow_mut()
            .closures
            .get_mut(&heap_id(closure).unwrap())
            .unwrap()
            .class = true;
    }

    pub(crate) fn is_class(&self, closure: Value) -> bool {
        heap_id(closure)
            .and_then(|id| {
                self.0
                    .borrow()
                    .closures
                    .get(&id)
                    .map(|closure| closure.references != 0 && closure.class)
            })
            .unwrap_or(false)
    }

    pub(crate) fn constructable(&self, closure: Value) -> bool {
        heap_id(closure).is_some_and(|id| {
            self.0
                .borrow()
                .closures
                .get(&id)
                .is_some_and(|closure| closure.references != 0 && closure.constructable)
        })
    }

    pub(crate) fn lexical_this(&self, closure: Value) -> Option<Value> {
        let data = self.0.borrow();
        let closure = data.closures.get(&heap_id(closure)?)?;
        if closure.references == 0 {
            return None;
        }
        closure.lexical_this
    }

    pub(crate) fn closure_code(&self, closure: Value) -> Option<CodeId> {
        let data = self.0.borrow();
        let closure = data.closures.get(&heap_id(closure)?)?;
        (closure.references != 0).then_some(closure.code)
    }

    pub(crate) fn capture(&self, closure: Value, index: usize) -> Option<Value> {
        let data = self.0.borrow();
        let closure = data.closures.get(&heap_id(closure)?)?;
        if closure.references == 0 {
            return None;
        }
        closure.captures.get(index).copied()
    }

    pub(crate) fn cell_get(&self, cell: Value) -> Option<Value> {
        let data = self.0.borrow();
        let cell = data.cells.get(&heap_id(cell)?)?;
        (cell.references != 0).then_some(cell.value)
    }

    pub(crate) fn cell_set(&self, cell: Value, value: Value) -> bool {
        let mut data = self.0.borrow_mut();
        let Some(id) = heap_id(cell) else {
            return false;
        };
        if !data.cells.get(&id).is_some_and(|cell| cell.references != 0) || !data.retain(value) {
            return false;
        }
        let old = std::mem::replace(&mut data.cells.get_mut(&id).unwrap().value, value);
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
        if let Some(array) = data.typed_arrays.get(&id) {
            if array.references == 0 {
                return None;
            }
            return Some(
                array
                    .storage
                    .get(index)
                    .map(Value::number)
                    .unwrap_or(Value::UNDEFINED),
            );
        }
        let array = data.arrays.get(&id)?;
        if array.references == 0 {
            return None;
        }
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
            .and_then(|id| data.arrays.get(&id))
            .is_some_and(|array| {
                array.references != 0
                    && index < array.elements.len()
                    && !array.holes.contains(&index)
            })
    }

    pub(crate) fn mark_hole(&self, array: Value, index: usize) {
        let mut data = self.0.borrow_mut();
        let array = data.arrays.get_mut(&heap_id(array).unwrap()).unwrap();
        array.holes.insert(index);
    }

    pub(crate) fn is_array(&self, value: Value) -> bool {
        let data = self.0.borrow();
        heap_id(value)
            .and_then(|id| data.arrays.get(&id))
            .is_some_and(|array| array.references != 0)
    }

    pub(crate) fn len(&self, array: Value) -> Option<usize> {
        let data = self.0.borrow();
        if array.is_string() {
            let (references, text) = data.strings.get(&heap_id(array)?)?;
            return (*references != 0).then(|| text.chars().count());
        }
        let id = heap_id(array)?;
        if let Some(array) = data.typed_arrays.get(&id) {
            return (array.references != 0).then_some(array.storage.len());
        }
        let array = data.arrays.get(&id)?;
        (array.references != 0).then_some(array.elements.len())
    }

    pub(crate) fn set(&self, array: Value, index: usize, value: Value) -> bool {
        let mut data = self.0.borrow_mut();
        let Some(id) = heap_id(array) else {
            return false;
        };
        let Some(len) = data
            .arrays
            .get(&id)
            .filter(|array| array.references != 0)
            .map(|array| array.elements.len())
        else {
            return false;
        };
        if index > len || !data.retain(value) {
            return false;
        }
        let array = data.arrays.get_mut(&id).unwrap();
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
        let Some(array) = data
            .arrays
            .get_mut(&id)
            .filter(|array| array.references != 0)
        else {
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
        let Some(array) = data.arrays.get_mut(&id) else {
            return false;
        };
        if array.references == 0 || length > array.elements.len() {
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
            if data
                .typed_arrays
                .get(&id)
                .is_some_and(|array| array.references == 0)
            {
                data.typed_arrays.remove(&id);
            } else if data
                .regexps
                .get(&id)
                .is_some_and(|regexp| regexp.references == 0)
            {
                data.regexps.remove(&id);
            } else if let Some(state) = data.enumerators.get(&id) {
                if state.references != 0 {
                    continue;
                }
                let state = data.enumerators.remove(&id).unwrap();
                if let Some(current) = state.current {
                    data.release(current);
                }
            } else if let Some(generator) = data.generators.get_mut(&id) {
                if generator.references != 0 {
                    continue;
                }
                if let Some(value) = generator.locals.pop() {
                    data.pending.push_back(id);
                    data.release(value);
                } else {
                    let generator = data.generators.remove(&id).unwrap();
                    data.release(generator.closure);
                    data.release(generator.arguments);
                    data.release(generator.receiver);
                }
            } else if let Some(set) = data.sets.get_mut(&id) {
                if set.references != 0 {
                    continue;
                }
                let entry = set.entries.extract_if(|_, _| true).next();
                if let Some((_, value)) = entry {
                    data.pending.push_back(id);
                    data.release(value);
                } else {
                    data.sets.remove(&id);
                }
            } else if let Some(object) = data.objects.get_mut(&id) {
                if object.references != 0 {
                    continue;
                }
                if let Some(value) = object.properties.pop() {
                    data.pending.push_back(id);
                    data.release(value);
                } else {
                    let object = data.objects.remove(&id).unwrap();
                    if let Some(prototype) = object.prototype {
                        data.release(prototype);
                    }
                }
            } else if data
                .strings
                .get(&id)
                .is_some_and(|(references, _)| *references == 0)
            {
                data.strings.remove(&id);
            } else if let Some(array) = data.arrays.get_mut(&id) {
                if array.references != 0 {
                    continue;
                }
                if let Some(value) = array.elements.pop() {
                    data.pending.push_back(id);
                    data.release(value);
                } else {
                    data.arrays.remove(&id);
                }
            } else if let Some(cell) = data.cells.remove(&id) {
                if cell.references != 0 {
                    data.cells.insert(id, cell);
                    continue;
                }
                data.release(cell.value);
            } else if let Some(closure) = data.closures.get_mut(&id) {
                if closure.references != 0 {
                    continue;
                }
                if let Some(value) = closure.properties.pop() {
                    data.pending.push_back(id);
                    data.release(value);
                } else if let Some(value) = closure.captures.pop() {
                    data.pending.push_back(id);
                    data.release(value);
                } else {
                    let closure = data.closures.remove(&id).unwrap();
                    if let Some(receiver) = closure.lexical_this {
                        data.release(receiver);
                    }
                }
            }
            budget -= 1;
        }
    }
}

fn next_id() -> Option<u64> {
    #[allow(deprecated)]
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| {
            (id < (1 << 48)).then_some(id + 1)
        })
        .ok()
}

fn heap_id(value: Value) -> Option<u64> {
    let bits = value.bits();
    ((value.is_object() && bits != TAG_OBJECT) || value.is_string())
        .then_some(bits & 0x0000_ffff_ffff_ffff)
}

impl HeapData {
    fn enumeration_keys(&self, value: Value) -> Option<Vec<String>> {
        let id = heap_id(value)?;
        if let Some(object) = self
            .objects
            .get(&id)
            .filter(|object| object.references != 0)
        {
            return Some(object.properties.keys());
        }
        if let Some(closure) = self
            .closures
            .get(&id)
            .filter(|closure| closure.references != 0)
        {
            let mut keys = closure.properties.keys();
            if closure.constructable && !closure.properties.contains_key("prototype") {
                keys.push("prototype".into());
            }
            return Some(keys);
        }
        if let Some(array) = self
            .typed_arrays
            .get(&id)
            .filter(|array| array.references != 0)
        {
            return Some(
                (0..array.storage.len())
                    .map(|index| index.to_string())
                    .collect(),
            );
        }
        if let Some(array) = self.arrays.get(&id).filter(|array| array.references != 0) {
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
        if let Some(object) = self
            .objects
            .get(&id)
            .filter(|object| object.references != 0)
        {
            return object.properties.enumerable(name);
        }
        if let Some(closure) = self
            .closures
            .get(&id)
            .filter(|closure| closure.references != 0)
        {
            return closure
                .properties
                .enumerable(name)
                .or_else(|| (closure.constructable && name == "prototype").then_some(false));
        }
        if let Some(array) = self
            .typed_arrays
            .get(&id)
            .filter(|array| array.references != 0)
        {
            return array_index_name(name)
                .and_then(|index| ((index as usize) < array.storage.len()).then_some(true));
        }
        if let Some(array) = self.arrays.get(&id).filter(|array| array.references != 0) {
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
        self.objects
            .get(&heap_id(value)?)
            .and_then(|object| object.prototype)
    }

    fn references(&mut self, id: u64) -> Option<&mut usize> {
        if let Some(array) = self.typed_arrays.get_mut(&id) {
            return Some(&mut array.references);
        }
        if let Some(regexp) = self.regexps.get_mut(&id) {
            return Some(&mut regexp.references);
        }
        if let Some(generator) = self.generators.get_mut(&id) {
            return Some(&mut generator.references);
        }
        if let Some(enumerator) = self.enumerators.get_mut(&id) {
            return Some(&mut enumerator.references);
        }
        if let Some((references, _)) = self.natives.get_mut(&id) {
            return Some(references);
        }
        if let Some(set) = self.sets.get_mut(&id) {
            return Some(&mut set.references);
        }
        if let Some(object) = self.objects.get_mut(&id) {
            return Some(&mut object.references);
        }
        if let Some((references, _)) = self.strings.get_mut(&id) {
            return Some(references);
        }
        if let Some(array) = self.arrays.get_mut(&id) {
            return Some(&mut array.references);
        }
        if let Some(cell) = self.cells.get_mut(&id) {
            return Some(&mut cell.references);
        }
        self.closures
            .get_mut(&id)
            .map(|closure| &mut closure.references)
    }

    fn valid(&self, value: Value) -> bool {
        match heap_id(value) {
            Some(id) => {
                self.typed_arrays
                    .get(&id)
                    .is_some_and(|array| array.references != 0)
                    || self
                        .regexps
                        .get(&id)
                        .is_some_and(|regexp| regexp.references != 0)
                    || self
                        .enumerators
                        .get(&id)
                        .is_some_and(|state| state.references != 0)
                    || self
                        .generators
                        .get(&id)
                        .is_some_and(|generator| generator.references != 0)
                    || self
                        .natives
                        .get(&id)
                        .is_some_and(|(references, _)| *references != 0)
                    || self.sets.get(&id).is_some_and(|set| set.references != 0)
                    || self
                        .objects
                        .get(&id)
                        .is_some_and(|object| object.references != 0)
                    || self
                        .strings
                        .get(&id)
                        .is_some_and(|(references, _)| *references != 0)
                    || self.arrays.get(&id).is_some_and(|v| v.references != 0)
                    || self.cells.get(&id).is_some_and(|v| v.references != 0)
                    || self.closures.get(&id).is_some_and(|v| v.references != 0)
            }
            None => !value.is_string(),
        }
    }

    fn retain(&mut self, value: Value) -> bool {
        match heap_id(value) {
            Some(id) => match self.references(id) {
                Some(references) if *references != 0 => {
                    *references += 1;
                    true
                }
                _ => false,
            },
            None => !value.is_string(),
        }
    }

    fn release(&mut self, value: Value) {
        let Some(id) = heap_id(value) else { return };
        let Some(references) = self.references(id) else {
            return;
        };
        assert!(*references > 0);
        *references -= 1;
        if *references == 0 {
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
    /// Returns a borrowed token; keep this handle alive while using the token.
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
            data.objects
                .get_mut(&heap_id(object).unwrap())
                .unwrap()
                .prototype = Some(prototype);
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
        assert!(heap.0.borrow().objects.is_empty());
        assert!(heap.0.borrow().enumerators.is_empty());
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
        assert!(heap.0.borrow().enumerators.is_empty());
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
        assert!(heap.0.borrow().regexps.is_empty());
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
        assert!(heap.0.borrow().generators.is_empty());
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
        assert!(heap.0.borrow().generators.is_empty());
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
        assert!(heap.0.borrow().sets.is_empty());
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
        let other = Heap::default();
        assert!(!other.valid(error));
        assert!(!other.set_property(error, "message", Value::NULL));
        heap.release(error);
        heap.drain(1);
        assert!(!heap.valid(error));
        assert!(
            heap.0
                .borrow()
                .objects
                .contains_key(&heap_id(error).unwrap())
        );
        heap.drain(100);
        assert!(!heap.valid(message));
        assert!(!heap.valid(name));
        assert!(!heap.valid(child));
        assert!(heap.0.borrow().objects.is_empty());
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
            heap.0
                .borrow()
                .closures
                .get(&heap_id(closure).unwrap())
                .unwrap()
                .captures
                .len(),
            99
        );
        assert!(cells.iter().any(|cell| heap.valid(*cell)));
        heap.drain(1000);
        assert!(cells.iter().all(|cell| !heap.valid(*cell)));
        assert!(heap.0.borrow().closures.is_empty());
        assert!(heap.0.borrow().cells.is_empty());
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
    fn closure_rejects_invalid_or_foreign_captures() {
        let heap = Heap::default();
        let other = Heap::default();
        let cell = heap.cell(Value::UNDEFINED).unwrap();
        let array = heap.array(&[]).unwrap();
        assert!(heap.closure(CodeId(1), &[array]).is_none());
        assert!(other.closure(CodeId(1), &[cell]).is_none());
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
        assert!(heap.0.borrow().arrays.contains_key(&heap_id(root).unwrap()));
        heap.drain(30000);
        assert!(!heap.valid(root));
        assert!(heap.0.borrow().arrays.is_empty());
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
    fn handles_are_checked_across_heaps() {
        let first = Heap::default();
        let second = Heap::default();
        let value = first.array(&[]).unwrap();
        assert!(!second.valid(value));
        assert!(second.array(&[value]).is_none());
    }
}
