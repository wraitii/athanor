use crate::{FxHashMap, FxHashSet};

use super::Unsupported;
use super::nfa::{Nfa, State, StateId};

const STATE_LIMIT: usize = 4096;

pub(crate) const DEAD: u32 = 0;

#[derive(Clone, Debug)]
pub(crate) struct Dfa {
    pub(crate) classes: [u8; 256],
    pub(crate) class_count: usize,
    pub(crate) transitions: Vec<u32>,
    pub(crate) matching: Vec<bool>,
    pub(crate) matching_at_end: Vec<bool>,
    pub(crate) start_at_beginning: u32,
    pub(crate) start: u32,
}

impl Dfa {
    pub(crate) fn next(&self, state: u32, byte: u8) -> u32 {
        self.transitions[state as usize * self.class_count + self.classes[byte as usize] as usize]
    }
}

pub(super) fn determinize(nfa: &Nfa) -> Result<Dfa, Unsupported> {
    let (classes, class_count, representatives) = byte_classes(nfa);
    let mut builder = Builder {
        nfa,
        ids: FxHashMap::default(),
        lists: Vec::new(),
        beginning: Vec::new(),
    };
    builder.intern(Vec::new(), false)?;
    let start_at_beginning = builder.intern(builder.closure(&[nfa.start], true, false), true)?;
    let start = builder.intern(builder.closure(&[nfa.start], false, false), false)?;
    let mut transitions = Vec::new();
    let mut state = 0;
    while state < builder.lists.len() {
        let list = builder.lists[state].clone();
        for &byte in &representatives {
            let seeds: Vec<StateId> = list
                .iter()
                .filter_map(|&id| match nfa.states[id as usize] {
                    State::Bytes { start, end, next } if (start..=end).contains(&byte) => {
                        Some(next)
                    }
                    _ => None,
                })
                .collect();
            let next = builder.closure(&seeds, false, false);
            transitions.push(builder.intern(next, false)?);
        }
        state += 1;
    }
    let matching = builder
        .lists
        .iter()
        .map(|list| list.iter().any(|&id| matches!(nfa.states[id as usize], State::Match)))
        .collect();
    let matching_at_end = builder
        .lists
        .iter()
        .zip(&builder.beginning)
        .map(|(list, &beginning)| {
            let resolved = builder.closure(list, beginning, true);
            resolved
                .iter()
                .any(|&id| matches!(nfa.states[id as usize], State::Match))
        })
        .collect();
    Ok(Dfa {
        classes,
        class_count,
        transitions,
        matching,
        matching_at_end,
        start_at_beginning,
        start,
    })
}

fn byte_classes(nfa: &Nfa) -> ([u8; 256], usize, Vec<u8>) {
    let mut boundaries = [false; 257];
    for state in &nfa.states {
        if let State::Bytes { start, end, .. } = *state {
            boundaries[start as usize] = true;
            boundaries[end as usize + 1] = true;
        }
    }
    let mut classes = [0; 256];
    let mut representatives = vec![0];
    for byte in 1..256 {
        if boundaries[byte] {
            representatives.push(byte as u8);
        }
        classes[byte] = (representatives.len() - 1) as u8;
    }
    (classes, representatives.len(), representatives)
}

struct Builder<'a> {
    nfa: &'a Nfa,
    ids: FxHashMap<(bool, Vec<StateId>), u32>,
    lists: Vec<Vec<StateId>>,
    beginning: Vec<bool>,
}

impl Builder<'_> {
    fn intern(&mut self, list: Vec<StateId>, beginning: bool) -> Result<u32, Unsupported> {
        if list.is_empty() && !self.lists.is_empty() {
            return Ok(DEAD);
        }
        let key = (beginning, list);
        if let Some(&id) = self.ids.get(&key) {
            return Ok(id);
        }
        if self.lists.len() >= STATE_LIMIT {
            return Err(Unsupported("automaton too large"));
        }
        let id = self.lists.len() as u32;
        self.lists.push(key.1.clone());
        self.beginning.push(beginning);
        self.ids.insert(key, id);
        Ok(id)
    }

    fn closure(&self, seeds: &[StateId], beginning: bool, end: bool) -> Vec<StateId> {
        let mut list = Vec::new();
        let mut seen = FxHashSet::default();
        let mut stack: Vec<StateId> = seeds.iter().rev().copied().collect();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            match &self.nfa.states[id as usize] {
                State::Bytes { .. } => list.push(id),
                State::Split(targets) => stack.extend(targets.iter().rev()),
                State::Start(next) => {
                    if beginning {
                        stack.push(*next);
                    }
                }
                State::End(next) => {
                    if end {
                        stack.push(*next);
                    } else {
                        list.push(id);
                    }
                }
                State::Match => {
                    list.push(id);
                    break;
                }
            }
        }
        list
    }
}
