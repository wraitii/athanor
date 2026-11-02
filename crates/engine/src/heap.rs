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
    natives: HashMap<u64, (usize, NativeFunction)>,
    sets: HashMap<u64, Set>,
    objects: HashMap<u64, Object>,
    strings: HashMap<u64, (usize, String)>,
    arrays: HashMap<u64, Box<Array>>,
    cells: HashMap<u64, Box<Cell>>,
    closures: HashMap<u64, Box<Closure>>,
    pending: VecDeque<u64>,
}

pub(crate) const GENERATOR_DONE: usize = usize::MAX;
pub(crate) const GENERATOR_RUNNING: usize = usize::MAX - 1;

struct Generator {
    references: usize,
    code: CodeId,
    closure: Value,
    arguments: Value,
    receiver: Value,
    state: usize,
    locals: Vec<Value>,
}

struct Object {
    references: usize,
    properties: HashMap<String, Value>,
    error: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeFunction {
    HostReadFile,
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
    ArrayJoin,
    StringSlice,
    StringSplit,
    StringStartsWith,
    StringReplace,
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
                properties: HashMap::new(),
                error: false,
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
        data.objects.insert(
            id,
            Object {
                references: 1,
                properties: HashMap::from([("message".into(), message), ("name".into(), name)]),
                error: true,
            },
        );
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn property(&self, object: Value, name: &str) -> Option<Value> {
        let data = self.0.borrow();
        let object = data.objects.get(&heap_id(object)?)?;
        (object.references != 0).then(|| {
            object
                .properties
                .get(name)
                .copied()
                .unwrap_or(Value::UNDEFINED)
        })
    }

    pub(crate) fn set_property(&self, object: Value, name: &str, value: Value) -> bool {
        let Some(id) = heap_id(object) else {
            return false;
        };
        let mut data = self.0.borrow_mut();
        if !data
            .objects
            .get(&id)
            .is_some_and(|object| object.references != 0)
            || !data.retain(value)
        {
            return false;
        }
        if let Some(old) = data
            .objects
            .get_mut(&id)
            .unwrap()
            .properties
            .insert(name.into(), value)
        {
            data.release(old);
        }
        true
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
        let array = data.arrays.get(&heap_id(array)?)?;
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
        let array = data.arrays.get(&heap_id(array)?)?;
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
            if let Some(generator) = data.generators.get_mut(&id) {
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
                if let Some(name) = object.properties.keys().next().cloned() {
                    let value = object.properties.remove(&name).unwrap();
                    data.pending.push_back(id);
                    data.release(value);
                } else {
                    data.objects.remove(&id);
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
                if let Some(value) = closure.captures.pop() {
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
    fn references(&mut self, id: u64) -> Option<&mut usize> {
        if let Some(generator) = self.generators.get_mut(&id) {
            return Some(&mut generator.references);
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
                self.generators
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
