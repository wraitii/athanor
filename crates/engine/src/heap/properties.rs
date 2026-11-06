use super::*;

const SHAPED_LIMIT: usize = 64;

/// A cache for one fixed property name. The weak heap reference prevents address reuse
/// from making atom and shape IDs from a previous heap appear valid.
#[derive(Default)]
pub(crate) struct PropertyWriteCache {
    heap: std::rc::Weak<RefCell<HeapData>>,
    atom: Option<Atom>,
    slot: Option<(ShapeId, Option<usize>)>,
}

impl PropertyWriteCache {
    pub(super) fn atom(&mut self, heap: &Heap, data: &mut HeapData, name: &str) -> Atom {
        if self.heap.as_ptr() != Rc::as_ptr(&heap.0) {
            self.heap = Rc::downgrade(&heap.0);
            self.atom = None;
            self.slot = None;
        }
        *self.atom.get_or_insert_with(|| data.atoms.intern(name))
    }

    fn slot(&mut self, shape: ShapeId, shapes: &Shapes, atom: Atom) -> Option<usize> {
        if let Some((cached, index)) = self.slot
            && cached == shape
        {
            return index;
        }
        let index = shapes.get(shape).slot(atom);
        self.slot = Some((shape, index));
        index
    }
}

pub(super) enum Properties {
    Shaped(Vec<Value>),
    Dictionary(PropertyTable),
}

impl Default for Properties {
    fn default() -> Self {
        Self::Shaped(Vec::new())
    }
}

impl Properties {
    pub(super) fn raw(&self) -> (*const u8, usize) {
        match self {
            Self::Shaped(values) => (values.as_ptr().cast(), values.len()),
            Self::Dictionary(_) => (std::ptr::null(), 0),
        }
    }

    pub(super) fn pop(&mut self) -> Option<Value> {
        match self {
            Self::Shaped(values) => values.pop(),
            Self::Dictionary(table) => table.pop(),
        }
    }
}

pub(super) enum PropertyView<'a> {
    Shaped {
        atoms: &'a Atoms,
        shape: &'a Shape,
        values: &'a [Value],
    },
    Dictionary(&'a PropertyTable),
}

impl PropertyView<'_> {
    fn slot(&self, name: &str) -> Option<usize> {
        let Self::Shaped { atoms, shape, .. } = self else {
            return None;
        };
        shape.slot(atoms.find(name)?)
    }

    pub(super) fn get(&self, name: &str) -> Option<Value> {
        match self {
            Self::Shaped { values, .. } => self.slot(name).map(|slot| values[slot]),
            Self::Dictionary(table) => table.get(name).copied(),
        }
    }

    pub(super) fn contains_key(&self, name: &str) -> bool {
        match self {
            Self::Shaped { .. } => self.slot(name).is_some(),
            Self::Dictionary(table) => table.contains_key(name),
        }
    }

    pub(super) fn enumerable(&self, name: &str) -> Option<bool> {
        match self {
            Self::Shaped { shape, .. } => self.slot(name).map(|slot| shape.keys()[slot].enumerable),
            Self::Dictionary(table) => table.enumerable(name),
        }
    }

    pub(super) fn keys(&self) -> Vec<String> {
        match self {
            Self::Shaped { atoms, shape, .. } => shape
                .keys()
                .iter()
                .map(|key| atoms.name(key.atom).to_owned())
                .collect(),
            Self::Dictionary(table) => table.keys(),
        }
    }
}

impl HeapData {
    pub(super) fn properties(&self, id: u64) -> Option<PropertyView<'_>> {
        let slot = self.live(id)?;
        let properties = match &slot.payload {
            Payload::SmallObject(object) => {
                return Some(PropertyView::Shaped {
                    atoms: &self.atoms,
                    shape: self.shapes.get(slot.shape),
                    values: &object.values[..object.len as usize],
                });
            }
            Payload::Object(object) => &object.properties,
            Payload::Closure(closure) => &closure.properties,
            _ => return None,
        };
        Some(match properties {
            Properties::Shaped(values) => PropertyView::Shaped {
                atoms: &self.atoms,
                shape: self.shapes.get(slot.shape),
                values,
            },
            Properties::Dictionary(table) => PropertyView::Dictionary(table),
        })
    }

    pub(super) fn put_property(
        &mut self,
        id: u64,
        name: &str,
        value: Value,
        enumerable: Option<bool>,
    ) -> Option<Value> {
        let old = self.write_property(id, name, value, enumerable, None, None);
        self.sync(id);
        old
    }

    pub(super) fn put_property_cached(
        &mut self,
        id: u64,
        name: &str,
        value: Value,
        enumerable: Option<bool>,
        atom: Atom,
        cache: &mut PropertyWriteCache,
    ) -> Option<Value> {
        let old = self.write_property(id, name, value, enumerable, Some(atom), Some(cache));
        self.sync(id);
        old
    }

    fn write_property(
        &mut self,
        id: u64,
        name: &str,
        value: Value,
        enumerable: Option<bool>,
        mut atom: Option<Atom>,
        mut cache: Option<&mut PropertyWriteCache>,
    ) -> Option<Value> {
        // Slots live in arena chunks rather than in HeapData, so this borrow is disjoint
        // from the shape and atom tables.
        let slot = unsafe { &mut *self.arena.get(id).as_ptr() };
        if let Payload::SmallObject(object) = &mut slot.payload {
            let atom = *atom.get_or_insert_with(|| self.atoms.intern(name));
            let shape = self.shapes.get(slot.shape);
            let index = match cache.as_deref_mut() {
                Some(cache) => cache.slot(slot.shape, &self.shapes, atom),
                None => shape.slot(atom),
            };
            if let Some(index) = index {
                if let Some(enumerable) = enumerable
                    && shape.keys()[index].enumerable != enumerable
                {
                    slot.shape = self.shapes.transition(slot.shape, Key { atom, enumerable });
                }
                return Some(std::mem::replace(&mut object.values[index], value));
            }
            if (object.len as usize) < INLINE_PROPERTIES && array_index_name(name).is_none() {
                slot.shape = self.shapes.transition(
                    slot.shape,
                    Key {
                        atom,
                        enumerable: enumerable.unwrap_or(true),
                    },
                );
                object.values[object.len as usize] = value;
                object.len += 1;
                return None;
            }
            let mut values = Vec::with_capacity(INLINE_PROPERTIES * 2);
            values.extend_from_slice(&object.values[..object.len as usize]);
            // The replacement takes ownership of the values and prototype without retaining them.
            slot.payload = Payload::Object(Object {
                properties: Properties::Shaped(values),
                error: object.error,
                prototype: object.prototype,
            });
        }
        let properties = match &mut slot.payload {
            Payload::Object(object) => &mut object.properties,
            Payload::Closure(closure) => &mut closure.properties,
            _ => unreachable!(),
        };
        if let Properties::Shaped(values) = properties {
            let atom = *atom.get_or_insert_with(|| self.atoms.intern(name));
            let shape = self.shapes.get(slot.shape);
            let index = match cache.as_deref_mut() {
                Some(cache) => cache.slot(slot.shape, &self.shapes, atom),
                None => shape.slot(atom),
            };
            if let Some(index) = index {
                if let Some(enumerable) = enumerable
                    && shape.keys()[index].enumerable != enumerable
                {
                    slot.shape = self.shapes.transition(slot.shape, Key { atom, enumerable });
                }
                return Some(std::mem::replace(&mut values[index], value));
            }
            if values.len() < SHAPED_LIMIT && array_index_name(name).is_none() {
                let key = Key {
                    atom,
                    enumerable: enumerable.unwrap_or(true),
                };
                slot.shape = self.shapes.transition(slot.shape, key);
                values.push(value);
                return None;
            }
            let mut table = PropertyTable::default();
            for (key, value) in self
                .shapes
                .get(slot.shape)
                .keys()
                .iter()
                .zip(values.drain(..))
            {
                table.set(self.atoms.name(key.atom), value, Some(key.enumerable));
            }
            *properties = Properties::Dictionary(table);
            slot.shape = DICTIONARY;
        }
        let Properties::Dictionary(table) = properties else {
            unreachable!()
        };
        table.set(name, value, enumerable)
    }

    pub(super) fn set_prototype(&mut self, id: u64, prototype: Value) {
        let slot = unsafe { &mut *self.arena.get(id).as_ptr() };
        match &mut slot.payload {
            Payload::Object(object) => object.prototype = Some(prototype),
            Payload::SmallObject(object) => object.prototype = Some(prototype),
            _ => unreachable!(),
        }
        if slot.shape != DICTIONARY {
            slot.shape = self.shapes.rebase(slot.shape, Some(prototype));
        }
    }
}

#[derive(Default)]
pub(super) struct PropertyTable {
    slots: Vec<PropertySlot>,
    index: HashMap<String, usize>,
}

struct PropertySlot {
    name: String,
    value: Value,
    enumerable: bool,
}

pub(super) fn array_index_name(name: &str) -> Option<u32> {
    name.parse::<u32>()
        .ok()
        .filter(|index| *index != u32::MAX && index.to_string() == name)
}

impl PropertyTable {
    pub(super) fn get(&self, name: &str) -> Option<&Value> {
        self.index.get(name).map(|index| &self.slots[*index].value)
    }

    pub(super) fn contains_key(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub(super) fn set(
        &mut self,
        name: &str,
        value: Value,
        enumerable: Option<bool>,
    ) -> Option<Value> {
        if let Some(index) = self.index.get(name).copied() {
            let slot = &mut self.slots[index];
            if let Some(enumerable) = enumerable {
                slot.enumerable = enumerable;
            }
            return Some(std::mem::replace(&mut slot.value, value));
        }
        self.index.insert(name.to_owned(), self.slots.len());
        self.slots.push(PropertySlot {
            name: name.to_owned(),
            value,
            enumerable: enumerable.unwrap_or(true),
        });
        None
    }

    pub(super) fn keys(&self) -> Vec<String> {
        let mut indices = Vec::new();
        let mut names = Vec::new();
        for slot in &self.slots {
            if let Some(index) = array_index_name(&slot.name) {
                indices.push((index, slot.name.clone()));
            } else {
                names.push(slot.name.clone());
            }
        }
        indices.sort_unstable_by_key(|(index, _)| *index);
        indices
            .into_iter()
            .map(|(_, name)| name)
            .chain(names)
            .collect()
    }

    pub(super) fn enumerable(&self, name: &str) -> Option<bool> {
        self.index
            .get(name)
            .map(|index| self.slots[*index].enumerable)
    }

    pub(super) fn pop(&mut self) -> Option<Value> {
        let slot = self.slots.pop()?;
        self.index.remove(&slot.name);
        Some(slot.value)
    }
}
