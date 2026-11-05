use super::*;

pub(super) type Atom = u32;
pub(super) type ShapeId = u32;

pub(crate) const NO_SHAPE: ShapeId = 0;
pub(crate) const DICTIONARY: ShapeId = u32::MAX;
const INDEXED_KEYS: usize = 8;

#[derive(Default)]
pub(super) struct Atoms {
    names: Vec<Rc<str>>,
    index: HashMap<Rc<str>, Atom>,
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
    keys: Vec<Key>,
    index: HashMap<Atom, usize>,
}

impl Shape {
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
            HashMap::default()
        };
        Self {
            prototype,
            keys,
            index,
        }
    }
}

pub(super) struct Shapes {
    shapes: Vec<Shape>,
    roots: HashMap<u64, ShapeId>,
    transitions: HashMap<(ShapeId, Key), ShapeId>,
}

impl Default for Shapes {
    fn default() -> Self {
        Self {
            shapes: vec![Shape::with(0, Vec::new())],
            roots: HashMap::default(),
            transitions: HashMap::default(),
        }
    }
}

impl Shapes {
    pub(super) fn get(&self, shape: ShapeId) -> &Shape {
        &self.shapes[shape as usize]
    }

    // Roots are keyed by prototype address; a reused address describes its new occupant.
    pub(super) fn root(&mut self, prototype: Option<Value>) -> ShapeId {
        let prototype = prototype.map_or(0, Value::bits);
        if let Some(&shape) = self.roots.get(&prototype) {
            return shape;
        }
        let shape = self.push(Shape::with(prototype, Vec::new()));
        self.roots.insert(prototype, shape);
        shape
    }

    pub(super) fn transition(&mut self, shape: ShapeId, key: Key) -> ShapeId {
        if let Some(&next) = self.transitions.get(&(shape, key)) {
            return next;
        }
        let parent = self.get(shape);
        let mut keys = parent.keys.clone();
        match parent.slot(key.atom) {
            Some(slot) => keys[slot] = key,
            None => keys.push(key),
        }
        let next = self.push(Shape::with(parent.prototype, keys));
        self.transitions.insert((shape, key), next);
        next
    }

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
