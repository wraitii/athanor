use std::collections::VecDeque;

#[derive(Clone, Copy)]
pub(super) enum Input<T> {
    Parameter(usize),
    Value(T),
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

pub(super) fn solve<T: Copy + Eq>(starts: &[usize], inputs: &[Input<T>]) -> Vec<Option<T>> {
    let count = starts.len() - 1;
    let mut states = vec![State::Empty; count];
    let mut users = vec![Vec::new(); count];
    let mut work = VecDeque::new();
    for parameter in 0..count {
        for input in &inputs[starts[parameter]..starts[parameter + 1]] {
            match *input {
                Input::Parameter(source) => users[source].push(parameter),
                Input::Value(value) => {
                    states[parameter].merge(State::One(value));
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
    use super::Input::{self, Parameter, Value};

    fn solve(incoming: &[Vec<Input<i32>>]) -> Vec<Option<i32>> {
        let mut starts = vec![0];
        for inputs in incoming {
            starts.push(starts.last().unwrap() + inputs.len());
        }
        let inputs: Vec<_> = incoming.iter().flatten().copied().collect();
        super::solve(&starts, &inputs)
    }

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
            ]),
            vec![None, None, None, Some(2), None]
        );
    }
}
