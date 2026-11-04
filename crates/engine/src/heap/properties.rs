use super::*;

const SHAPED_LIMIT: usize = 64;

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
        // Slots live in arena chunks rather than in HeapData, so this borrow is disjoint
        // from the shape and atom tables.
        let slot = unsafe { &mut *self.arena.get(id).as_ptr() };
        let properties = match &mut slot.payload {
            Payload::Object(object) => &mut object.properties,
            Payload::Closure(closure) => &mut closure.properties,
            _ => unreachable!(),
        };
        if let Properties::Shaped(values) = properties {
            let atom = self.atoms.intern(name);
            let shape = self.shapes.get(slot.shape);
            if let Some(index) = shape.slot(atom) {
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
            for (key, value) in self.shapes.get(slot.shape).keys().iter().zip(values.drain(..)) {
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
        let Payload::Object(object) = &mut slot.payload else {
            unreachable!()
        };
        object.prototype = Some(prototype);
        if slot.shape != DICTIONARY {
            slot.shape = self.shapes.rebase(slot.shape, Some(prototype));
        }
    }
}

#[derive(Default)]
pub(super) struct PropertyTable {
    pub(super) slots: Vec<PropertySlot>,
    pub(super) index: HashMap<String, usize>,
}

pub(super) struct PropertySlot {
    pub(super) name: String,
    pub(super) value: Value,
    pub(super) enumerable: bool,
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
