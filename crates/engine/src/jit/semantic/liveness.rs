use cranelift_codegen::entity::{EntityRef, SecondaryMap};

use super::{Block, Cfg, Graph, Value};

#[derive(Default, Eq)]
pub(super) struct Set {
    words: Vec<u64>,
}

impl Clone for Set {
    fn clone(&self) -> Self {
        Self {
            words: self.words.clone(),
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.words.clone_from(&other.words);
    }
}

impl PartialEq for Set {
    fn eq(&self, other: &Self) -> bool {
        let common = self.words.len().min(other.words.len());
        self.words[..common] == other.words[..common]
            && self.words[common..].iter().all(|&word| word == 0)
            && other.words[common..].iter().all(|&word| word == 0)
    }
}

impl Set {
    pub(super) fn clear(&mut self) {
        self.words.clear();
    }

    pub(super) fn insert(&mut self, value: Value) -> bool {
        let index = value.index() / 64;
        if self.words.len() <= index {
            self.words.resize(index + 1, 0);
        }
        let mask = 1 << (value.index() % 64);
        let added = self.words[index] & mask == 0;
        self.words[index] |= mask;
        added
    }

    pub(super) fn remove(&mut self, value: &Value) -> bool {
        let Some(word) = self.words.get_mut(value.index() / 64) else {
            return false;
        };
        let mask = 1 << (value.index() % 64);
        let removed = *word & mask != 0;
        *word &= !mask;
        removed
    }

    pub(super) fn contains(&self, value: &Value) -> bool {
        self.words
            .get(value.index() / 64)
            .is_some_and(|word| word & (1 << (value.index() % 64)) != 0)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        self.words.iter().enumerate().flat_map(|(index, &word)| {
            let mut word = word;
            std::iter::from_fn(move || {
                if word == 0 {
                    return None;
                }
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                Some(Value::new(index * 64 + bit))
            })
        })
    }

    pub(super) fn union(&mut self, other: &Self) {
        if self.words.len() < other.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (word, other) in self.words.iter_mut().zip(&other.words) {
            *word |= other;
        }
    }

    fn subtract(&mut self, other: &Self) {
        for (word, other) in self.words.iter_mut().zip(&other.words) {
            *word &= !other;
        }
    }
}

impl Extend<Value> for Set {
    fn extend<T: IntoIterator<Item = Value>>(&mut self, values: T) {
        for value in values {
            self.insert(value);
        }
    }
}

pub(super) struct Live {
    pub(super) input: SecondaryMap<Block, Set>,
    pub(super) output: SecondaryMap<Block, Set>,
}

pub(super) fn solve(
    graph: &Graph,
    cfg: &Cfg,
    uses: &SecondaryMap<Block, Set>,
    defs: &SecondaryMap<Block, Set>,
) -> Live {
    let mut input = SecondaryMap::<Block, Set>::new();
    let mut output = SecondaryMap::<Block, Set>::new();
    let mut pending = cfg.rpo.clone();
    let mut queued = SecondaryMap::with_default(true);
    let mut live = Set::default();
    while let Some(block) = pending.pop() {
        queued[block] = false;
        live.clear();
        for successor in graph.successors(block) {
            live.union(&input[successor]);
        }
        output[block].clone_from(&live);
        live.subtract(&defs[block]);
        live.union(&uses[block]);
        if live == input[block] {
            continue;
        }
        std::mem::swap(&mut input[block], &mut live);
        for &pred in &cfg.preds[block] {
            if !queued[pred] {
                queued[pred] = true;
                pending.push(pred);
            }
        }
    }
    Live { input, output }
}
