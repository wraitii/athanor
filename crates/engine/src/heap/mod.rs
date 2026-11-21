use std::cell::RefCell;
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::FxHashMap;

use crate::value::{TAG_OBJECT, TAG_STRING, Value};

mod arena;
mod array;
mod buffers;
mod cycles;
mod function;
mod generator;
pub(crate) mod kind;
mod natives;
mod object;
mod owned;
mod properties;
mod regexp;
mod set;
mod shape;
pub(crate) mod stats;
mod string;
#[cfg(test)]
mod tests;
mod typed;

use arena::*;
use array::*;
use function::*;
use generator::*;
use natives::*;
use object::*;
use properties::*;
use regexp::*;
use set::*;
use shape::*;
use typed::*;

pub(crate) use crate::builtins::{MathFunction, NativeFunction};
pub(crate) use arena::{
    CODE_OFFSET, DATA_OFFSET, INLINE_PROPERTY_LENGTH_OFFSET, KIND_OFFSET, LENGTH_OFFSET,
    NO_ELEMENTS, PROPERTY_DATA_OFFSET, REFERENCES_OFFSET, SHAPE_OFFSET,
};
pub(crate) use array::HOLE;
pub use owned::OwnedValue;
pub(crate) use properties::{
    PropertyAdditionCache, PropertyOverwriteCache, PropertyReadCache, PropertyWrite,
    PropertyWriteCache,
};
pub(crate) use regexp::regexp_flags;
pub(crate) use shape::{DICTIONARY, NO_SHAPE};
pub(crate) use typed::TypedKind;

#[derive(Clone, Default)]
pub(crate) struct Heap(Rc<RefCell<HeapData>>);

struct HeapData {
    arena: Arena,
    atoms: Atoms,
    shapes: Shapes,
    natives: Vec<(NativeFunction, u64)>,
    object_prototype: Option<Value>,
    object_shape: ShapeId,
    empty_object_shape: ShapeId,
    function_prototype: Option<Value>,
    string_prototype: Option<Value>,
    string_shape: ShapeId,
    array_prototype: Option<Value>,
    indexed_prototypes: bool,
    array_shape: ShapeId,
    interned: Vec<(&'static str, u64)>,
    regexp_cache: FxHashMap<(String, u8), Rc<crate::regex::Pattern>>,
    pending: VecDeque<u64>,
    array_buffers: buffers::BufferPool<Value>,
    string_buffers: buffers::BufferPool<u8>,
    epoch: u64,
    stats: Option<Box<stats::Stats>>,
}

impl Default for HeapData {
    fn default() -> Self {
        Self {
            arena: Arena::default(),
            atoms: Atoms::default(),
            shapes: Shapes::default(),
            natives: Vec::new(),
            object_prototype: None,
            object_shape: ShapeId::default(),
            empty_object_shape: ShapeId::default(),
            function_prototype: None,
            string_prototype: None,
            string_shape: NO_SHAPE,
            array_prototype: None,
            indexed_prototypes: false,
            array_shape: NO_SHAPE,
            interned: Vec::new(),
            regexp_cache: FxHashMap::default(),
            pending: VecDeque::new(),
            array_buffers: buffers::BufferPool::default(),
            string_buffers: buffers::BufferPool::default(),
            epoch: 0,
            stats: stats::enabled().then(|| Box::new(stats::Stats::default())),
        }
    }
}

enum Payload {
    Vacant,
    Reserved,
    Native(Native),
    String(String),
    Generator(Box<Generator>),
    RegExp(Box<RegExp>),
    Enumerator(Box<Enumerator>),
    Set(Box<Set>),
    SmallObject(SmallObject),
    Object(Object),
    Array(Array),
    TypedArray(TypedArray),
    Cell(Cell),
    Closure(Box<Closure>),
}

impl Payload {
    fn kind(&self) -> u32 {
        match self {
            Self::Vacant | Self::Reserved => 0,
            Self::Native(_) => kind::NATIVE,
            Self::String(_) => kind::STRING,
            Self::Generator(_) => kind::GENERATOR,
            Self::RegExp(_) => kind::REGEXP,
            Self::Enumerator(_) => kind::ENUMERATOR,
            Self::Set(_) => kind::SET,
            Self::Object(_) => kind::OBJECT,
            Self::SmallObject(_) => kind::OBJECT | kind::INLINE,
            Self::Array(array) if array.holes == 0 => kind::ARRAY,
            Self::Array(_) => kind::ARRAY | kind::HOLEY,
            Self::TypedArray(array) => kind::TYPED | array.kind.element(),
            Self::Cell(_) => kind::CELL,
            Self::Closure(closure) if closure.constructable => kind::CLOSURE | kind::CONSTRUCTOR,
            Self::Closure(_) => kind::CLOSURE,
        }
    }
}

pub(crate) enum PropertyRead {
    Value(Value),
    Native(NativeFunction),
    Prototype,
}

pub(crate) enum IndexRead {
    Primitive(Value),
    Borrowed(Value),
    Character(char),
}

pub(crate) const GENERATOR_DONE: usize = usize::MAX;
pub(crate) const GENERATOR_RUNNING: usize = usize::MAX - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CodeId(pub(crate) u64);

impl Heap {
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

    pub(crate) fn invalidate(&self) {
        self.0.borrow_mut().epoch += 1;
    }

    pub(crate) fn epoch(&self) -> *const u64 {
        unsafe { &raw const (*self.0.as_ptr()).epoch }
    }

    pub(crate) fn owns(&self, value: Value) -> bool {
        let data = self.0.borrow();
        heap_id(value).is_none_or(|id| data.arena.owns(id)) && data.valid(value)
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

    pub(crate) fn release_all(&self, values: &[Value]) {
        let mut owned = values
            .iter()
            .filter(|value| value.is_object() || value.is_string())
            .peekable();
        if owned.peek().is_none() {
            return;
        }
        let mut data = self.0.borrow_mut();
        for &value in owned {
            data.release(value);
        }
    }

    pub(crate) fn release_frame(&self, values: &mut [Value]) {
        let mut data = None;
        for value in values {
            let held = std::mem::replace(value, Value::UNDEFINED);
            let Some(id) = heap_id(held) else { continue };
            data.get_or_insert_with(|| self.0.borrow_mut())
                .release_owned(id, held);
        }
    }

    pub(crate) fn drain(&self, mut budget: usize) {
        let mut data = self.0.borrow_mut();
        let empty = data.pending.is_empty();
        if let Some(stats) = &mut data.stats {
            stats.drains += 1;
            stats.empty_drains += u64::from(empty);
            stats.disposing = true;
        }
        while budget > 0 {
            let Some(id) = data.pending.pop_front() else {
                break;
            };
            if data.slot(id).references != 0 {
                continue;
            }
            while budget > 0 {
                let edge = match &mut data.slot_mut(id).payload {
                    Payload::Generator(generator) => generator.locals.pop(),
                    Payload::Set(set) => set
                        .entries
                        .extract_if(|_, _| true)
                        .next()
                        .map(|(_, value)| value),
                    Payload::Object(object) => object.properties.pop(),
                    Payload::SmallObject(object) => object.pop(),
                    Payload::Array(array) => array.elements.pop().or_else(|| {
                        array
                            .properties
                            .as_mut()
                            .and_then(|properties| properties.pop())
                    }),
                    Payload::Closure(closure) => {
                        closure.properties.pop().or_else(|| closure.captures.pop())
                    }
                    Payload::Native(native) => native.properties.pop(),
                    _ => None,
                };
                budget -= 1;
                if let Some(stats) = &mut data.stats {
                    stats.work += 1;
                    stats.edges += u64::from(edge.is_some());
                    stats.heap_edges +=
                        u64::from(edge.is_some_and(|value| heap_id(value).is_some()));
                }
                if let Some(value) = edge {
                    data.release(value);
                    if budget == 0 {
                        data.pending.push_back(id);
                    }
                    continue;
                }
                let payload = data.vacate(id);
                match payload {
                    Payload::Array(array) => {
                        if let Some(prototype) = array.prototype {
                            data.release(prototype);
                        }
                        data.array_buffers.put(array.elements);
                    }
                    Payload::String(mut text) => {
                        text.clear();
                        data.string_buffers.put(text.into_bytes());
                    }
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
                    Payload::SmallObject(object) => {
                        if let Some(value) = object.prototype() {
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
                break;
            }
        }
        if let Some(stats) = &mut data.stats {
            stats.disposing = false;
        }
    }
}

fn element_index(number: f64, limit: f64) -> Option<usize> {
    if !(number >= 0.0 && number < limit) {
        return None;
    }
    let index = number as usize;
    (index as f64 == number).then_some(index)
}

fn heap_id(value: Value) -> Option<u64> {
    let bits = value.bits();
    ((value.is_object() && bits != TAG_OBJECT) || value.is_string())
        .then_some(bits & 0x0000_ffff_ffff_ffff)
        .filter(|id| *id != 0)
}

impl HeapData {
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
    fn object(&self, id: u64) -> Option<ObjectInfo> {
        match &self.live(id)?.payload {
            Payload::Object(value) => Some(ObjectInfo {
                prototype: value.prototype,
                error: value.error,
            }),
            Payload::SmallObject(value) => Some(ObjectInfo {
                prototype: value.prototype(),
                error: value.error,
            }),
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
    fn reserve(&mut self) -> Option<u64> {
        let slot = self.arena.allocate()?;
        let slot = unsafe { &mut *slot.as_ptr() };
        slot.references = 1;
        slot.payload = Payload::Reserved;
        Some(slot as *mut Slot as u64)
    }

    fn store(&mut self, id: u64, payload: Payload) {
        if let Some(stats) = &mut self.stats {
            stats.allocate(id, payload.cycle_kind());
        }
        self.slot_mut(id).payload = payload;
        self.sync(id);
    }

    fn store_string(&mut self, id: u64, text: String) {
        self.store(id, Payload::String(text));
        self.slot_mut(id).shape = self.string_shape;
    }

    fn sync(&mut self, id: u64) {
        let inherited =
            self.indexed_prototypes || self.array_prototype.and_then(heap_id) == Some(id);
        let slot = self.slot_mut(id);
        slot.kind = slot.payload.kind();
        if inherited && matches!(slot.payload, Payload::Array(_)) {
            slot.kind |= kind::PROTOTYPE_INDEXED;
        }
        slot.code = match &slot.payload {
            Payload::Closure(closure) => callable_code(closure),
            _ => 0,
        };
        slot.property_data = match &slot.payload {
            Payload::SmallObject(object) => object.values.as_ptr().cast(),
            Payload::Object(object) => object.properties.raw().0,
            Payload::Closure(closure) => closure.properties.raw().0,
            Payload::Native(native) => native.properties.raw().0,
            Payload::Array(array) => array
                .properties
                .as_ref()
                .map_or(std::ptr::null(), |properties| properties.raw().0),
            _ => std::ptr::null(),
        };
        (slot.data, slot.length) = match &slot.payload {
            Payload::Array(array) => (array.elements.as_ptr().cast(), array.elements.len()),
            Payload::TypedArray(array) => array.storage.raw(),
            Payload::Object(object) => object.properties.raw(),
            Payload::SmallObject(object) => (object.values.as_ptr().cast(), object.len as usize),
            Payload::Closure(closure) => closure.properties.raw(),
            Payload::Native(native) => native.properties.raw(),
            Payload::Cell(cell) => (std::ptr::from_ref(&cell.value).cast(), 1),
            Payload::String(text) => (text.as_ptr(), text.len()),
            Payload::RegExp(regexp) => (regexp.pattern.reference().address() as *const u8, 0),
            _ => (std::ptr::null(), 0),
        };
    }

    fn store_shaped(&mut self, id: u64, payload: Payload) {
        let root = match &payload {
            Payload::Closure(_) | Payload::Native(_) => {
                self.shapes.closure_root(self.function_prototype)
            }
            _ => self.shapes.root(None),
        };
        self.store(id, payload);
        self.slot_mut(id).shape = root;
    }

    fn vacate(&mut self, id: u64) -> Payload {
        if let Some(stats) = &mut self.stats {
            stats.dispose(id, false);
        }
        let slot = self.slot_mut(id);
        slot.kind = 0;
        slot.shape = NO_SHAPE;
        slot.code = 0;
        slot.data = std::ptr::null();
        slot.property_data = std::ptr::null();
        slot.length = 0;
        let payload = std::mem::replace(&mut slot.payload, Payload::Vacant);
        let slot = NonNull::from(slot);
        self.arena.free(slot);
        payload
    }

    fn slot(&self, id: u64) -> &Slot {
        unsafe { &*self.arena.get(id).as_ptr() }
    }

    fn slot_mut(&mut self, id: u64) -> &mut Slot {
        unsafe { &mut *self.arena.get(id).as_ptr() }
    }

    #[cfg(test)]
    fn slots(&self) -> impl Iterator<Item = &Slot> {
        self.arena.slots()
    }

    // Zero-reference slots remain present until incremental destruction finishes.
    fn live(&self, id: u64) -> Option<&Slot> {
        Some(self.slot(id)).filter(|slot| slot.references != 0)
    }

    fn live_mut(&mut self, id: u64) -> Option<&mut Slot> {
        Some(self.slot_mut(id)).filter(|slot| slot.references != 0)
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

    #[inline(always)]
    fn release_owned(&mut self, id: u64, value: Value) {
        let slot = self.slot_mut(id);
        if slot.references > 1 && !matches!(slot.payload, Payload::Vacant | Payload::Reserved) {
            slot.references -= 1;
            return;
        }
        self.release(value);
    }

    fn release(&mut self, value: Value) {
        let Some(id) = heap_id(value) else { return };
        let slot = self.slot_mut(id);
        if matches!(slot.payload, Payload::Vacant | Payload::Reserved) {
            return;
        }
        assert!(slot.references > 0);
        slot.references -= 1;
        if slot.references == 0 {
            // Disposal pops property values, so no site cache may match while it runs.
            slot.shape = NO_SHAPE;
            self.pending.push_back(id);
            if let Some(stats) = &mut self.stats {
                stats.queue(id);
            }
        }
    }
}
