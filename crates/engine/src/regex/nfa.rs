use super::Unsupported;
use super::parse::Node;
use super::set::Set;

const STATE_LIMIT: usize = 50_000;

pub(super) type StateId = u32;

#[derive(Clone, Debug)]
pub(super) enum State {
    Bytes { start: u8, end: u8, next: StateId },
    Split(Vec<StateId>),
    Start(StateId),
    End(StateId),
    Match,
}

pub(super) struct Nfa {
    pub(super) states: Vec<State>,
    pub(super) start: StateId,
}

impl Nfa {
    pub(super) fn new(node: &Node) -> Result<Self, Unsupported> {
        let mut builder = Builder { states: Vec::new() };
        let accept = builder.push(State::Match)?;
        let start = builder.node(node, accept)?;
        Ok(Self {
            states: builder.states,
            start,
        })
    }
}

struct Builder {
    states: Vec<State>,
}

impl Builder {
    fn push(&mut self, state: State) -> Result<StateId, Unsupported> {
        if self.states.len() >= STATE_LIMIT {
            return Err(Unsupported("pattern too large"));
        }
        self.states.push(state);
        Ok((self.states.len() - 1) as StateId)
    }

    fn split(&mut self, targets: Vec<StateId>) -> Result<StateId, Unsupported> {
        if let [target] = targets[..] {
            return Ok(target);
        }
        self.push(State::Split(targets))
    }

    fn node(&mut self, node: &Node, next: StateId) -> Result<StateId, Unsupported> {
        match node {
            Node::Empty => Ok(next),
            Node::Start => self.push(State::Start(next)),
            Node::End => self.push(State::End(next)),
            Node::Set(set) => self.set(set, next),
            Node::Concat(nodes) => nodes
                .iter()
                .rev()
                .try_fold(next, |next, node| self.node(node, next)),
            Node::Alternation(nodes) => {
                let targets = nodes
                    .iter()
                    .map(|node| self.node(node, next))
                    .collect::<Result<_, _>>()?;
                self.split(targets)
            }
            Node::Repeat {
                node,
                min,
                max,
                greedy,
            } => {
                let order = |body: StateId| {
                    if *greedy {
                        vec![body, next]
                    } else {
                        vec![next, body]
                    }
                };
                let mut entry = match max {
                    None => {
                        let loop_state = self.push(State::Split(Vec::new()))?;
                        let body = self.node(node, loop_state)?;
                        self.states[loop_state as usize] = State::Split(order(body));
                        loop_state
                    }
                    Some(max) => {
                        let mut entry = next;
                        for _ in *min..*max {
                            let body = self.node(node, entry)?;
                            entry = self.push(State::Split(order(body)))?;
                        }
                        entry
                    }
                };
                for _ in 0..*min {
                    entry = self.node(node, entry)?;
                }
                Ok(entry)
            }
        }
    }

    fn set(&mut self, set: &Set, next: StateId) -> Result<StateId, Unsupported> {
        let mut targets = Vec::new();
        for sequence in utf8_sequences(set) {
            let mut entry = next;
            for &(start, end) in sequence.iter().rev() {
                entry = self.push(State::Bytes {
                    start,
                    end,
                    next: entry,
                })?;
            }
            targets.push(entry);
        }
        if targets.is_empty() {
            return self.push(State::Split(Vec::new()));
        }
        self.split(targets)
    }
}

fn utf8_sequences(set: &Set) -> Vec<Vec<(u8, u8)>> {
    let mut sequences = Vec::new();
    for &(start, end) in set.ranges() {
        for (low, high) in [(0, 0x7f), (0x80, 0x7ff), (0x800, 0xffff), (0x10000, 0x10ffff)] {
            let (start, end) = (start.max(low), end.min(high));
            if start <= end {
                split_sequence(start, end, &mut sequences);
            }
        }
    }
    sequences
}

fn split_sequence(start: u32, end: u32, sequences: &mut Vec<Vec<(u8, u8)>>) {
    let length = char::from_u32(start).map_or(3, char::len_utf8);
    for trailing in 1..length {
        let mask = (1u32 << (6 * trailing)) - 1;
        if start & !mask == end & !mask {
            continue;
        }
        if start & mask != 0 {
            split_sequence(start, start | mask, sequences);
            split_sequence((start | mask) + 1, end, sequences);
            return;
        }
        if end & mask != mask {
            split_sequence(start, (end & !mask) - 1, sequences);
            split_sequence(end & !mask, end, sequences);
            return;
        }
    }
    let (mut low, mut high) = ([0; 4], [0; 4]);
    let low = encode(start, &mut low);
    let high = encode(end, &mut high);
    sequences.push(low.iter().zip(high).map(|(&a, &b)| (a, b)).collect());
}

fn encode(value: u32, buffer: &mut [u8; 4]) -> &[u8] {
    let length = match value {
        0..=0x7f => 1,
        0x80..=0x7ff => 2,
        0x800..=0xffff => 3,
        _ => 4,
    };
    let leading = [0, 0, 0xc0, 0xe0, 0xf0][length];
    for index in (1..length).rev() {
        buffer[index] = 0x80 | (value >> (6 * (length - 1 - index)) & 0x3f) as u8;
    }
    buffer[0] = leading | (value >> (6 * (length - 1))) as u8;
    &buffer[..length]
}
