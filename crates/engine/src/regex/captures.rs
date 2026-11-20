use std::ops::Range;

use super::dfa::is_word;
use super::nfa::{Nfa, State, StateId};

const UNSET: usize = usize::MAX;
const VISITED_LIMIT: usize = 1 << 22;

#[derive(Default)]
pub(crate) struct Scratch {
    visited: Vec<u64>,
    slots: Vec<usize>,
    stack: Vec<Frame>,
}

enum Frame {
    Explore(StateId, usize),
    Restore(usize, usize),
}

pub(super) fn captures(
    nfa: &Nfa,
    groups: usize,
    text: &[u8],
    range: Range<usize>,
    scratch: &mut Scratch,
) -> Option<Vec<Option<Range<usize>>>> {
    let Range { start, end } = range;
    let span = end - start + 1;
    let words = (nfa.states.len() * span).div_ceil(64);
    if words > VISITED_LIMIT {
        return None;
    }
    scratch.visited.clear();
    scratch.visited.resize(words, 0);
    scratch.slots.clear();
    scratch.slots.resize(2 * groups, UNSET);
    scratch.stack.clear();
    scratch.stack.push(Frame::Explore(nfa.start, start));
    while let Some(frame) = scratch.stack.pop() {
        let (mut id, mut position) = match frame {
            Frame::Restore(slot, value) => {
                scratch.slots[slot] = value;
                continue;
            }
            Frame::Explore(id, position) => (id, position),
        };
        loop {
            let index = id as usize * span + position - start;
            let (word, bit) = (index / 64, 1 << (index % 64));
            if scratch.visited[word] & bit != 0 {
                break;
            }
            scratch.visited[word] |= bit;
            match &nfa.states[id as usize] {
                State::Bytes {
                    start: low,
                    end: high,
                    next,
                } => {
                    if position == end || !(*low..=*high).contains(&text[position]) {
                        break;
                    }
                    id = *next;
                    position += 1;
                }
                State::Split(targets) => {
                    let Some((&first, rest)) = targets.split_first() else {
                        break;
                    };
                    scratch.stack.extend(
                        rest.iter()
                            .rev()
                            .map(|&target| Frame::Explore(target, position)),
                    );
                    id = first;
                }
                State::Start(next) if position == 0 => id = *next,
                State::End(next) if position == text.len() => id = *next,
                State::Start(_) | State::End(_) => break,
                State::WordBoundary { negated, next } => {
                    let before = position > 0 && is_word(text[position - 1]);
                    let after = text.get(position).is_some_and(|&byte| is_word(byte));
                    if (before != after) == *negated {
                        break;
                    }
                    id = *next;
                }
                State::Save { slot, next } => {
                    scratch
                        .stack
                        .push(Frame::Restore(*slot, scratch.slots[*slot]));
                    scratch.slots[*slot] = position;
                    id = *next;
                }
                State::Reset { slots, next } => {
                    for slot in slots.clone() {
                        scratch
                            .stack
                            .push(Frame::Restore(slot, scratch.slots[slot]));
                        scratch.slots[slot] = UNSET;
                    }
                    id = *next;
                }
                State::Match => {
                    if position != end {
                        return None;
                    }
                    return Some(
                        scratch
                            .slots
                            .chunks_exact(2)
                            .map(|pair| {
                                (pair[0] != UNSET && pair[1] != UNSET).then(|| pair[0]..pair[1])
                            })
                            .collect(),
                    );
                }
            }
        }
    }
    None
}
