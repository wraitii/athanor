use std::collections::VecDeque;

#[derive(Clone, Copy)]
pub(super) enum Input<T> {
    Parameter(usize),
    Value(T),
    Varying,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum State<T> {
    #[default]
    Empty,
    One(T),
    Many,
}

impl<T: Copy + Eq> State<T> {
    fn merge(&mut self, other: Self) -> bool {
        let next = match (*self, other) {
            (Self::Empty, value) | (value, Self::Empty) => value,
            (Self::One(a), Self::One(b)) if a == b => Self::One(a),
            _ => Self::Many,
        };
        if *self == next {
            return false;
        }
        *self = next;
        true
    }
}

// Values identify SSA definitions, not literal constants. Unseeded cycles stay unresolved.
pub(super) fn solve<T: Copy + Eq>(incoming: &[Vec<Input<T>>]) -> Vec<Option<T>> {
    let mut states = vec![State::Empty; incoming.len()];
    let mut users = vec![Vec::new(); incoming.len()];
    let mut work = VecDeque::new();
    for (parameter, inputs) in incoming.iter().enumerate() {
        for input in inputs {
            match *input {
                Input::Parameter(source) => users[source].push(parameter),
                Input::Value(value) => {
                    states[parameter].merge(State::One(value));
                }
                Input::Varying => {
                    states[parameter].merge(State::Many);
                }
            }
        }
        if states[parameter] != State::Empty {
            work.push_back(parameter);
        }
    }
    while let Some(parameter) = work.pop_front() {
        let state = states[parameter];
        for &user in &users[parameter] {
            if states[user].merge(state) {
                work.push_back(user);
            }
        }
    }
    states
        .into_iter()
        .map(|state| match state {
            State::One(value) => Some(value),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::Input::{Parameter, Value, Varying};
    use super::solve;

    #[test]
    fn forwarding_cycles_share_their_single_origin() {
        assert_eq!(
            solve(&[
                vec![Value(42), Parameter(2)],
                vec![Parameter(0), Parameter(2)],
                vec![Parameter(1)],
                vec![Parameter(3)],
            ]),
            vec![Some(42), Some(42), Some(42), None]
        );
    }

    #[test]
    fn conflicts_propagate_through_cycles() {
        assert_eq!(
            solve(&[
                vec![Value(1), Parameter(2)],
                vec![Parameter(0)],
                vec![Parameter(1), Parameter(3)],
                vec![Value(2)],
                vec![Parameter(0), Value(1)],
                vec![Value(1), Varying],
            ]),
            vec![None, None, None, Some(2), None, None]
        );
    }
}
