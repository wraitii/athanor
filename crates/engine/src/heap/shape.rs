use super::*;
use crate::FxHashMap;

pub(super) type Atom = u32;
pub(super) type ShapeId = u32;

pub(crate) const NO_SHAPE: ShapeId = 0;
pub(crate) const DICTIONARY: ShapeId = u32::MAX;
const INDEXED_KEYS: usize = 8;

#[derive(Default)]
pub(super) struct Atoms {
    names: Vec<Rc<str>>,
    index: FxHashMap<Rc<str>, Atom>,
}

impl Atoms {
    pub(super) fn find(&self, name: &str) -> Option<Atom> {
        self.index.get(name).copied()
    }

    pub(super) fn intern(&mut self, name: &str) -> Atom {
        if let Some(atom) = self.find(name) {
            return atom;
        }
        let atom = self.names.len() as Atom;
        let name: Rc<str> = name.into();
        self.names.push(name.clone());
        self.index.insert(name, atom);
        atom
    }

    pub(super) fn name(&self, atom: Atom) -> &str {
        &self.names[atom as usize]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Key {
    pub(super) atom: Atom,
    pub(super) enumerable: bool,
}

pub(super) struct Shape {
    prototype: u64,
    root: ShapeId,
    keys: Vec<Key>,
    index: FxHashMap<Atom, usize>,
}

impl Shape {
    pub(super) fn has_prototype(&self, prototype: Value) -> bool {
        self.prototype == prototype.bits()
    }

    pub(super) fn keys(&self) -> &[Key] {
        &self.keys
    }

    pub(super) fn slot(&self, atom: Atom) -> Option<usize> {
        if self.keys.len() <= INDEXED_KEYS {
            return self.keys.iter().position(|key| key.atom == atom);
        }
        self.index.get(&atom).copied()
    }

    fn with(prototype: u64, keys: Vec<Key>) -> Self {
        let index = if keys.len() > INDEXED_KEYS {
            keys.iter()
                .enumerate()
                .map(|(slot, key)| (key.atom, slot))
                .collect()
        } else {
            FxHashMap::default()
        };
        Self {
            prototype,
            root: NO_SHAPE,
            keys,
            index,
        }
    }
}

pub(super) struct Shapes {
    shapes: Vec<Shape>,
    roots: FxHashMap<(u64, bool), ShapeId>,
    transitions: FxHashMap<(ShapeId, Key), ShapeId>,
    deletions: FxHashMap<(ShapeId, Atom), ShapeId>,
}

impl Default for Shapes {
    fn default() -> Self {
        Self {
            shapes: vec![Shape::with(0, Vec::new())],
            roots: FxHashMap::default(),
            transitions: FxHashMap::default(),
            deletions: FxHashMap::default(),
        }
    }
}

impl Shapes {
    pub(super) fn get(&self, shape: ShapeId) -> &Shape {
        &self.shapes[shape as usize]
    }

    // Roots are keyed by prototype address; a reused address describes its new occupant.
    pub(super) fn root(&mut self, prototype: Option<Value>) -> ShapeId {
        self.root_for(prototype, false)
    }

    pub(super) fn closure_root(&mut self, prototype: Option<Value>) -> ShapeId {
        // Closure roots stay distinct because their constructor prototype is lazy.
        self.root_for(prototype, true)
    }

    fn root_for(&mut self, prototype: Option<Value>, closure: bool) -> ShapeId {
        let prototype = prototype.map_or(0, Value::bits);
        let key = (prototype, closure);
        if let Some(&shape) = self.roots.get(&key) {
            return shape;
        }
        let shape = self.push(Shape::with(prototype, Vec::new()));
        self.shapes[shape as usize].root = shape;
        self.roots.insert(key, shape);
        shape
    }

    pub(super) fn transition(&mut self, shape: ShapeId, key: Key) -> ShapeId {
        if let Some(&next) = self.transitions.get(&(shape, key)) {
            return next;
        }
        let parent = self.get(shape);
        let root = parent.root;
        let mut keys = parent.keys.clone();
        if let Some(slot) = parent.slot(key.atom) {
            if keys[slot] == key {
                return shape;
            }
            keys[slot] = key;
            let next = keys
                .into_iter()
                .fold(root, |id, key| self.transition(id, key));
            self.transitions.insert((shape, key), next);
            return next;
        }
        keys.push(key);
        let mut layout = Shape::with(parent.prototype, keys);
        layout.root = root;
        let next = self.push(layout);
        self.transitions.insert((shape, key), next);
        self.deletions.insert((next, key.atom), shape);
        next
    }

    pub(super) fn delete(&mut self, shape: ShapeId, atom: Atom) -> ShapeId {
        if let Some(&next) = self.deletions.get(&(shape, atom)) {
            return next;
        }
        let layout = self.get(shape);
        if layout.slot(atom).is_none() {
            return shape;
        }
        let root = layout.root;
        let keys = layout.keys.clone();
        let next = keys
            .into_iter()
            .filter(|key| key.atom != atom)
            .fold(root, |id, key| self.transition(id, key));
        self.deletions.insert((shape, atom), next);
        next
    }

    #[cfg(test)]
    pub(super) fn rebase(&mut self, shape: ShapeId, prototype: Option<Value>) -> ShapeId {
        let keys = self.get(shape).keys.clone();
        keys.into_iter().fold(self.root(prototype), |shape, key| {
            self.transition(shape, key)
        })
    }

    fn push(&mut self, shape: Shape) -> ShapeId {
        let id = self.shapes.len() as ShapeId;
        assert!(id < DICTIONARY);
        self.shapes.push(shape);
        id
    }
}
