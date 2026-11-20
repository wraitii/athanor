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
    pub(crate) matching_before: Vec<bool>,
    pub(crate) matching_at_end: Vec<bool>,
    pub(crate) start_at_beginning: u32,
    pub(crate) start: u32,
    pub(crate) start_after_word: u32,
}

impl Dfa {
    pub(crate) fn next(&self, state: u32, byte: u8) -> u32 {
        self.transitions[self.index(state, byte)]
    }

    pub(crate) fn matches_before(&self, state: u32, byte: u8) -> bool {
        self.matching_before[self.index(state, byte)]
    }

    pub(crate) fn matching(&self, state: u32) -> Option<bool> {
        let row = state as usize * self.class_count;
        let row = &self.matching_before[row..row + self.class_count];
        row.iter()
            .all(|&matching| matching == row[0])
            .then_some(row[0])
    }

    pub(crate) fn may_match(&self, state: u32) -> bool {
        let row = state as usize * self.class_count;
        self.matching_at_end[state as usize]
            || self.matching_before[row..row + self.class_count]
                .iter()
                .any(|&matching| matching)
    }

    fn index(&self, state: u32, byte: u8) -> usize {
        state as usize * self.class_count + self.classes[byte as usize] as usize
    }
}

pub(crate) fn is_word(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Context {
    beginning: bool,
    after_word: bool,
}

pub(super) fn determinize(nfa: &Nfa) -> Result<Dfa, Unsupported> {
    let (classes, class_count, representatives) = byte_classes(nfa);
    let mut builder = Builder {
        nfa,
        ids: FxHashMap::default(),
        states: Vec::new(),
    };
    let initial = |beginning, after_word| Context {
        beginning,
        after_word,
    };
    builder.intern(Vec::new(), initial(false, false))?;
    let entry = |builder: &Builder<'_>, beginning| builder.closure(&[nfa.start], beginning);
    let start_at_beginning = builder.intern(entry(&builder, true), initial(true, false))?;
    let start = builder.intern(entry(&builder, false), initial(false, false))?;
    let start_after_word = builder.intern(entry(&builder, false), initial(false, true))?;
    let mut transitions = Vec::new();
    let mut matching_before = Vec::new();
    let mut state = 0;
    while state < builder.states.len() {
        let (list, context) = builder.states[state].clone();
        for &byte in &representatives {
            let resolved = builder.resolve(&list, context, Some(is_word(byte)));
            matching_before.push(builder.matches(&resolved));
            let seeds: Vec<StateId> = resolved
                .iter()
                .filter_map(|&id| match nfa.states[id as usize] {
                    State::Bytes { start, end, next } if (start..=end).contains(&byte) => {
                        Some(next)
                    }
                    _ => None,
                })
                .collect();
            let next = builder.closure(&seeds, false);
            transitions.push(builder.intern(next, initial(false, is_word(byte)))?);
        }
        state += 1;
    }
    let matching_at_end = builder
        .states
        .iter()
        .map(|(list, context)| builder.matches(&builder.resolve(list, *context, None)))
        .collect();
    Ok(Dfa {
        classes,
        class_count,
        transitions,
        matching_before,
        matching_at_end,
        start_at_beginning,
        start,
        start_after_word,
    })
}

fn byte_classes(nfa: &Nfa) -> ([u8; 256], usize, Vec<u8>) {
    let mut boundaries = [false; 257];
    let mut split = |start: u8, end: u8| {
        boundaries[start as usize] = true;
        boundaries[end as usize + 1] = true;
    };
    for state in &nfa.states {
        match *state {
            State::Bytes { start, end, .. } => split(start, end),
            State::WordBoundary { .. } => {
                for (start, end) in [(b'0', b'9'), (b'A', b'Z'), (b'_', b'_'), (b'a', b'z')] {
                    split(start, end);
                }
            }
            _ => {}
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
    ids: FxHashMap<(Context, Vec<StateId>), u32>,
    states: Vec<(Vec<StateId>, Context)>,
}

#[derive(Clone, Copy)]
enum Lookahead {
    Byte(bool),
    End,
}

impl Builder<'_> {
    fn intern(&mut self, list: Vec<StateId>, mut context: Context) -> Result<u32, Unsupported> {
        if list.is_empty() && !self.states.is_empty() {
            return Ok(DEAD);
        }
        if !list.iter().any(|&id| {
            matches!(
                self.nfa.states[id as usize],
                State::WordBoundary { .. } | State::End(_)
            )
        }) {
            context.after_word = false;
        }
        let key = (context, list);
        if let Some(&id) = self.ids.get(&key) {
            return Ok(id);
        }
        if self.states.len() >= STATE_LIMIT {
            return Err(Unsupported("automaton too large"));
        }
        let id = self.states.len() as u32;
        self.states.push((key.1.clone(), context));
        self.ids.insert(key, id);
        Ok(id)
    }

    fn matches(&self, list: &[StateId]) -> bool {
        list.iter()
            .any(|&id| matches!(self.nfa.states[id as usize], State::Match))
    }

    fn resolve(&self, list: &[StateId], context: Context, next_word: Option<bool>) -> Vec<StateId> {
        let lookahead = match next_word {
            Some(word) => Lookahead::Byte(word),
            None => Lookahead::End,
        };
        self.closure_with(
            list,
            context.beginning,
            Some((context.after_word, lookahead)),
        )
    }

    fn closure(&self, seeds: &[StateId], beginning: bool) -> Vec<StateId> {
        self.closure_with(seeds, beginning, None)
    }

    fn closure_with(
        &self,
        seeds: &[StateId],
        beginning: bool,
        around: Option<(bool, Lookahead)>,
    ) -> Vec<StateId> {
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
                State::Save { next, .. } | State::Reset { next, .. } => stack.push(*next),
                State::Start(next) => {
                    if beginning {
                        stack.push(*next);
                    }
                }
                State::End(next) => match around {
                    Some((_, Lookahead::End)) => stack.push(*next),
                    Some((_, Lookahead::Byte(_))) => {}
                    None => list.push(id),
                },
                State::WordBoundary { negated, next } => match around {
                    Some((before, lookahead)) => {
                        let after = matches!(lookahead, Lookahead::Byte(true));
                        if (before != after) != *negated {
                            stack.push(*next);
                        }
                    }
                    None => list.push(id),
                },
                State::Match => {
                    list.push(id);
                    break;
                }
            }
        }
        list
    }
}
