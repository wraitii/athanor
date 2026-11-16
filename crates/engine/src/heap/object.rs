use super::*;
use crate::FxHashSet;
use crate::builtins::ReceiverKind;

pub(super) struct Enumerator {
    pub(super) current: Option<Value>,
    pub(super) keys: Vec<String>,
    pub(super) cursor: usize,
    pub(super) visited: FxHashSet<String>,
}

pub(super) struct Object {
    pub(super) properties: Properties,
    pub(super) error: bool,
    pub(super) prototype: Option<Value>,
}

pub(super) struct ObjectInfo {
    pub(super) error: bool,
    pub(super) prototype: Option<Value>,
}

pub(super) const INLINE_PROPERTIES: usize = 8;

#[repr(C)]
pub(super) struct SmallObject {
    pub(super) values: [Value; INLINE_PROPERTIES],
    pub(super) prototype: Option<Value>,
    pub(super) error: bool,
    pub(super) len: u8,
}

impl SmallObject {
    fn new(error: bool) -> Self {
        Self {
            values: [Value::UNDEFINED; INLINE_PROPERTIES],
            prototype: None,
            error,
            len: 0,
        }
    }

    pub(super) fn pop(&mut self) -> Option<Value> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.values[self.len as usize])
    }
}

impl Heap {
    pub(crate) fn object(&self) -> Option<Value> {
        self.object_with_prototype(Value::UNDEFINED)
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
        data.store_shaped(id, Payload::SmallObject(SmallObject::new(true)));
        data.put_property(id, "message", message, Some(false));
        data.put_property(id, "name", name, Some(false));
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    /// Where a site cache may find a property, together with its borrowed current value.
    pub(crate) fn cached_property(&self, receiver: Value, name: &str) -> Option<CachedProperty> {
        let data = self.0.borrow();
        let id = heap_id(receiver)?;
        let shape = data.cacheable_shape(id)?;
        let atom = data.atoms.find(name)?;
        let (holder, holder_shape, index) = if let Some(index) = data.shapes.get(shape).slot(atom) {
            (0, shape, index)
        } else {
            let prototype = if data.closure(id).is_some() {
                // Closures share shapes even when only some have a lazy own prototype.
                if name == "prototype" {
                    return None;
                }
                let prototype = data.function_prototype?;
                if !data.shapes.get(shape).has_prototype(prototype) {
                    return None;
                }
                prototype
            } else {
                data.object(id)?.prototype?
            };
            let holder = heap_id(prototype)?;
            let holder_shape = data.cacheable_shape(holder)?;
            let index = data.shapes.get(holder_shape).slot(atom)?;
            (holder, holder_shape, index)
        };
        let owner = if holder == 0 { id } else { holder };
        let PropertyView::Shaped { values, .. } = data.properties(owner)? else {
            return None;
        };
        Some(CachedProperty {
            shape,
            holder,
            holder_shape,
            index: index as u32,
            value: *values.get(index)?,
        })
    }

    pub(crate) fn cached_property_hit(
        &self,
        receiver: Value,
        shape: u32,
        holder: u64,
        holder_shape: u32,
        index: u32,
    ) -> Option<Value> {
        let data = self.0.borrow();
        let id = heap_id(receiver)?;
        if data.cacheable_shape(id)? != shape {
            return None;
        }
        let owner = if holder == 0 { id } else { holder };
        if data.cacheable_shape(owner)? != holder_shape {
            return None;
        }
        let PropertyView::Shaped { values, .. } = data.properties(owner)? else {
            return None;
        };
        values.get(index as usize).copied()
    }

    pub(crate) fn builtin_property(
        &self,
        receiver: Value,
        name: &str,
    ) -> Option<(u32, NativeFunction)> {
        // These payloads resolve methods by kind; native constructors need their identity too.
        let kind = {
            let data = self.0.borrow();
            let slot = data.live(heap_id(receiver)?)?;
            match &slot.payload {
                Payload::Array(_)
                | Payload::TypedArray(_)
                | Payload::String(_)
                | Payload::Set(_)
                | Payload::RegExp(_)
                | Payload::Generator(_) => slot.kind,
                _ => return None,
            }
        };
        let Some(PropertyRead::Native(function)) = self.read_property(receiver, name) else {
            return None;
        };
        Some((kind, function))
    }

    pub(crate) fn has_kind(&self, receiver: Value, kind: u32) -> bool {
        let data = self.0.borrow();
        heap_id(receiver)
            .and_then(|id| data.live(id))
            .is_some_and(|slot| slot.kind == kind)
    }

    /// Returned values are borrowed; native functions and lazy prototypes are resolved by the caller.
    pub(crate) fn read_property(&self, receiver: Value, name: &str) -> Option<PropertyRead> {
        if receiver.is_number() {
            return Some(PropertyRead::Native(ReceiverKind::Number.method(name)?));
        }
        let data = self.0.borrow();
        let payload = &data.live(heap_id(receiver)?)?.payload;
        let function = match payload {
            Payload::Object(_) | Payload::SmallObject(_) => {
                return data
                    .object_property(heap_id(receiver)?, name)
                    .map(PropertyRead::Value);
            }
            Payload::Closure(closure) => {
                if let Some(value) = data.properties(heap_id(receiver)?)?.get(name) {
                    return Some(PropertyRead::Value(value));
                }
                return Some(if closure.constructable && name == "prototype" {
                    PropertyRead::Prototype
                } else {
                    PropertyRead::Value(data.function_property(name))
                });
            }
            Payload::Native(function) => {
                if *function == NativeFunction::Function && name == "prototype" {
                    return Some(PropertyRead::Value(data.function_prototype?));
                }
                match function.static_method(name) {
                    Some(function) => function,
                    None => return Some(PropertyRead::Value(data.function_property(name))),
                }
            }
            Payload::TypedArray(array) => match name {
                "length" => {
                    return Some(PropertyRead::Value(Value::number(
                        array.storage.len() as f64
                    )));
                }
                _ => ReceiverKind::TypedArray.method(name)?,
            },
            Payload::RegExp(regexp) => match name {
                "lastIndex" => return Some(PropertyRead::Value(Value::number(regexp.last_index))),
                _ => ReceiverKind::RegExp.method(name)?,
            },
            Payload::Array(array) => match name {
                "length" => {
                    return Some(PropertyRead::Value(Value::number(
                        array.elements.len() as f64
                    )));
                }
                _ => ReceiverKind::Array.method(name)?,
            },
            Payload::String(text) => match name {
                "length" => {
                    return Some(PropertyRead::Value(Value::number(
                        text.chars().count() as f64
                    )));
                }
                _ => ReceiverKind::String.method(name)?,
            },
            Payload::Set(set) => match name {
                "size" => {
                    return Some(PropertyRead::Value(Value::number(set.entries.len() as f64)));
                }
                _ => match ReceiverKind::Set.method(name) {
                    Some(function) => function,
                    None => return Some(PropertyRead::Value(Value::UNDEFINED)),
                },
            },
            Payload::Generator(_) => match ReceiverKind::Generator.method(name) {
                Some(function) => function,
                None => return Some(PropertyRead::Value(Value::UNDEFINED)),
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
                    && !data.properties(id).unwrap().contains_key(name)
            })
        };
        if needs_prototype {
            let prototype = self.object()?;
            self.define_property(object, name, prototype, false);
            self.release(prototype);
        }
        let data = self.0.borrow();
        if data.closure(id).is_some() {
            return Some(
                data.properties(id)?
                    .get(name)
                    .unwrap_or_else(|| data.function_property(name)),
            );
        }
        data.object(id)?;
        data.object_property(id, name)
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
            && let Some(index) = keys.iter().position(|name| name == "prototype")
        {
            let name = keys.remove(index);
            let first_string = keys
                .iter()
                .take_while(|name| array_index_name(name).is_some())
                .count();
            keys.insert(first_string, name);
        }
        Some(keys)
    }

    pub(crate) fn own_properties(&self, value: Value) -> Option<Vec<(String, Value)>> {
        let data = self.0.borrow();
        let id = heap_id(value)?;
        data.object(id)?;
        let properties = data.properties(id)?;
        Some(
            properties
                .keys()
                .into_iter()
                .filter_map(|name| {
                    (properties.enumerable(&name) == Some(true))
                        .then(|| (name.clone(), properties.get(&name).unwrap()))
                })
                .collect(),
        )
    }

    pub(crate) fn set_property(&self, object: Value, name: &str, value: Value) -> bool {
        self.write_property(object, name, value, None, None)
    }

    pub(crate) fn define_property(
        &self,
        object: Value,
        name: &str,
        value: Value,
        enumerable: bool,
    ) -> bool {
        self.write_property(object, name, value, Some(enumerable), None)
    }

    pub(crate) fn set_property_cached(
        &self,
        object: Value,
        name: &str,
        value: Value,
        cache: &mut PropertyWriteCache,
    ) -> bool {
        self.write_property(object, name, value, None, Some(cache))
    }

    pub(super) fn write_property(
        &self,
        object: Value,
        name: &str,
        value: Value,
        enumerable: Option<bool>,
        cache: Option<&mut PropertyWriteCache>,
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
            && data
                .closure(id)
                .is_some_and(|closure| closure.constructable)
            && !data.properties(id).unwrap().contains_key(name)
        {
            Some(false)
        } else {
            enumerable
        };
        let old = match cache {
            Some(cache) => {
                let atom = cache.atom(self, &mut data, name);
                data.put_property_cached(id, name, value, enumerable, atom, cache)
            }
            None => data.put_property(id, name, value, enumerable),
        };
        if let Some(old) = old {
            if heap_id(old)
                .and_then(|id| data.live(id))
                .is_some_and(|slot| matches!(slot.payload, Payload::Native(_)))
            {
                data.epoch += 1;
            }
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
        data.store(
            id,
            Payload::Enumerator(Box::new(Enumerator {
                current,
                keys,
                cursor: 0,
                visited: FxHashSet::default(),
            })),
        );
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
            if let Some(next) = next
                && !data.retain(next)
            {
                return None;
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
                data.properties(id)?.contains_key(name)
                    || (closure.constructable && name == "prototype")
                    || data.function_has_property(name)
            });
        }
        if let Payload::Native(function) = &data.live(id)?.payload {
            return Some(
                function.static_method(name).is_some()
                    || (*function == NativeFunction::Function && name == "prototype")
                    || data.function_has_property(name),
            );
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
            return Some(index.is_some_and(|index| array.present(index as usize)));
        }
        let mut current = value;
        loop {
            let id = heap_id(current)?;
            let object = data.object(id)?;
            if data.properties(id)?.contains_key(name) {
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

    pub(crate) fn constructor_prototype(&self, constructor: Value) -> Option<Value> {
        let data = self.0.borrow();
        let id = heap_id(constructor)?;
        if !data.closure(id)?.constructable {
            return None;
        }
        // Reading an absent prototype must not trigger its lazy allocation.
        let prototype = data.properties(id)?.get("prototype")?;
        if prototype.is_object() && heap_id(prototype).and_then(|id| data.object(id)).is_none() {
            return None;
        }
        Some(prototype)
    }

    pub(crate) fn object_with_prototype(&self, prototype: Value) -> Option<Value> {
        let prototype = prototype.is_object().then_some(prototype);
        if let Some(id) = self.0.borrow_mut().reuse_object(prototype) {
            return Some(Value::from_bits(TAG_OBJECT | id));
        }
        self.drain(64);
        let mut data = self.0.borrow_mut();
        let id = data.reserve()?;
        if let Some(prototype) = prototype {
            data.retain(prototype);
        }
        let root = data.shapes.root(prototype);
        data.store(
            id,
            Payload::SmallObject(SmallObject {
                prototype,
                ..SmallObject::new(false)
            }),
        );
        data.slot_mut(id).shape = root;
        Some(Value::from_bits(TAG_OBJECT | id))
    }

    pub(crate) fn constructed_object(&self, constructor: Value) -> Option<Value> {
        let prototype = self.property(constructor, "prototype")?;
        if prototype.is_object() && !self.has_properties(prototype) {
            return None;
        }
        self.object_with_prototype(prototype)
    }

    pub(crate) fn error_text(&self, value: Value) -> Option<String> {
        let data = self.0.borrow();
        let id = heap_id(value)?;
        if !data.object(id)?.error {
            return None;
        }
        let properties = data.properties(id)?;
        let name = properties.get("name").unwrap_or(Value::UNDEFINED);
        let message = properties.get("message").unwrap_or(Value::UNDEFINED);
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
}

/// A receiver's shape fixes its prototype's address, so a matching receiver and holder shape
/// fix where the value lives. `holder` is 0 for an own property.
pub(crate) struct CachedProperty {
    pub(crate) value: Value,
    pub(crate) shape: u32,
    pub(crate) holder: u64,
    pub(crate) holder_shape: u32,
    pub(crate) index: u32,
}

impl HeapData {
    fn function_has_property(&self, name: &str) -> bool {
        self.function_prototype
            .and_then(heap_id)
            .and_then(|id| self.properties(id))
            .is_some_and(|properties| properties.contains_key(name))
    }

    fn function_property(&self, name: &str) -> Value {
        self.function_prototype
            .and_then(heap_id)
            .and_then(|id| self.object_property(id, name))
            .unwrap_or(Value::UNDEFINED)
    }

    fn reuse_object(&mut self, prototype: Option<Value>) -> Option<u64> {
        let id = *self.pending.front()?;
        if !matches!(self.slot(id).payload, Payload::SmallObject(_)) {
            return None;
        }
        debug_assert_eq!(self.slot(id).references, 0);
        self.pending.pop_front();
        loop {
            let Payload::SmallObject(object) = &mut self.slot_mut(id).payload else {
                unreachable!();
            };
            let Some(value) = object.pop() else {
                break;
            };
            self.release(value);
        }
        let root = self.shapes.root(prototype);
        if let Some(prototype) = prototype {
            self.retain(prototype);
        }
        let slot = self.slot_mut(id);
        let Payload::SmallObject(object) = &mut slot.payload else {
            unreachable!();
        };
        let previous = std::mem::replace(&mut object.prototype, prototype);
        object.error = false;
        slot.references = 1;
        slot.shape = root;
        slot.code = 0;
        slot.length = 0;
        if let Some(prototype) = previous {
            self.release(prototype);
        }
        Some(id)
    }

    fn cacheable_shape(&self, id: u64) -> Option<ShapeId> {
        Some(self.live(id)?.shape).filter(|&shape| shape != NO_SHAPE && shape != DICTIONARY)
    }

    pub(super) fn object_property(&self, mut id: u64, name: &str) -> Option<Value> {
        loop {
            if let Some(value) = self.properties(id)?.get(name) {
                return Some(value);
            }
            let Some(prototype) = self.object(id)?.prototype else {
                return Some(Value::UNDEFINED);
            };
            id = heap_id(prototype)?;
        }
    }

    pub(super) fn enumeration_keys(&self, value: Value) -> Option<Vec<String>> {
        let id = heap_id(value)?;
        if self.object(id).is_some() {
            return Some(self.properties(id)?.keys());
        }
        if let Some(closure) = self.closure(id) {
            let properties = self.properties(id)?;
            let mut keys = properties.keys();
            if closure.constructable && !properties.contains_key("prototype") {
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
                .filter(|index| array.present(*index))
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

    pub(super) fn enumeration_presence(&self, value: Value, name: &str) -> Option<bool> {
        let id = heap_id(value)?;
        if self.object(id).is_some() {
            return self.properties(id)?.enumerable(name);
        }
        if let Some(closure) = self.closure(id) {
            return self
                .properties(id)?
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
            return array_index_name(name)
                .and_then(|index| array.present(index as usize).then_some(true));
        }
        None
    }

    pub(super) fn enumeration_prototype(&self, value: Value) -> Option<Value> {
        self.object(heap_id(value)?)
            .and_then(|object| object.prototype)
    }
}
