use crate::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::Hash;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Priority {
    Demanded,
    Optimize,
    Prefetch,
    Unlazify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Ticket(u64);

struct Pending<K, T> {
    key: K,
    priority: Priority,
    dependencies: FxHashSet<Ticket>,
    value: T,
}

pub(super) struct Queue<K, T> {
    next: u64,
    keys: FxHashMap<K, Ticket>,
    outstanding: FxHashSet<Ticket>,
    pending: FxHashMap<Ticket, Pending<K, T>>,
    ready: [BinaryHeap<Reverse<(Priority, Ticket)>>; 2],
}

impl<K: Copy + Eq + Hash, T> Queue<K, T> {
    pub(super) fn new() -> Self {
        Self {
            next: 0,
            keys: FxHashMap::default(),
            outstanding: FxHashSet::default(),
            pending: FxHashMap::default(),
            ready: std::array::from_fn(|_| BinaryHeap::new()),
        }
    }

    pub(super) fn submit(
        &mut self,
        key: K,
        priority: Priority,
        value: T,
        dependencies: &[Ticket],
    ) -> Ticket {
        if let Some(&ticket) = self.keys.get(&key) {
            self.promote(ticket, priority);
            return ticket;
        }
        let ticket = Ticket(self.next);
        self.next = self
            .next
            .checked_add(1)
            .expect("compilation ticket overflow");
        let dependencies: FxHashSet<_> = dependencies
            .iter()
            .copied()
            .filter(|dependency| self.outstanding.contains(dependency))
            .collect();
        for &dependency in &dependencies {
            self.promote(dependency, priority);
        }
        if dependencies.is_empty() {
            self.ready[usize::from(priority == Priority::Demanded)]
                .push(Reverse((priority, ticket)));
        }
        self.pending.insert(
            ticket,
            Pending {
                key,
                priority,
                dependencies,
                value,
            },
        );
        self.keys.insert(key, ticket);
        self.outstanding.insert(ticket);
        ticket
    }

    fn promote(&mut self, ticket: Ticket, priority: Priority) {
        let Some(pending) = self.pending.get_mut(&ticket) else {
            return;
        };
        if priority >= pending.priority {
            return;
        }
        pending.priority = priority;
        let dependencies: Vec<_> = pending.dependencies.iter().copied().collect();
        if dependencies.is_empty() {
            self.ready[usize::from(priority == Priority::Demanded)]
                .push(Reverse((priority, ticket)));
        }
        for dependency in dependencies {
            self.promote(dependency, priority);
        }
    }

    pub(super) fn pending_mut(&mut self, key: K) -> Option<&mut T> {
        let ticket = self.keys.get(&key)?;
        self.pending
            .get_mut(ticket)
            .map(|pending| &mut pending.value)
    }

    pub(super) fn pop(&mut self) -> Option<(Ticket, K, T)> {
        self.pop_lane(false)
    }

    pub(super) fn pop_lazy(&mut self) -> Option<(Ticket, K, T)> {
        self.pop_lane(true)
    }

    fn pop_lane(&mut self, lazy: bool) -> Option<(Ticket, K, T)> {
        while let Some(Reverse((priority, ticket))) = self.ready[usize::from(lazy)].pop() {
            if !self
                .pending
                .get(&ticket)
                .is_some_and(|pending| pending.priority == priority)
            {
                continue;
            }
            let pending = self.pending.remove(&ticket).unwrap();
            return Some((ticket, pending.key, pending.value));
        }
        None
    }

    pub(super) fn complete(&mut self, ticket: Ticket, key: K) {
        assert_eq!(self.keys.remove(&key), Some(ticket));
        assert!(self.outstanding.remove(&ticket));
        for (&waiting, pending) in &mut self.pending {
            if pending.dependencies.remove(&ticket) && pending.dependencies.is_empty() {
                self.ready[usize::from(pending.priority == Priority::Demanded)]
                    .push(Reverse((pending.priority, waiting)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_demands_move_to_the_reserved_lane_without_duplicate_attempts() {
        let mut queue = Queue::new();
        let demand = queue.submit(1, Priority::Unlazify, "demand", &[]);
        queue.submit(2, Priority::Optimize, "loop", &[]);
        assert!(queue.pop_lazy().is_none());
        assert_eq!(
            queue.submit(1, Priority::Demanded, "duplicate", &[]),
            demand
        );
        assert_eq!(queue.pop().unwrap().2, "loop");
        assert!(queue.pop().is_none());
        assert_eq!(queue.pop_lazy(), Some((demand, 1, "demand")));
        assert_eq!(
            queue.submit(1, Priority::Demanded, "duplicate", &[]),
            demand
        );
        assert!(queue.pop_lazy().is_none());
        queue.complete(demand, 1);
        let running = queue.submit(3, Priority::Unlazify, "running", &[]);
        assert_eq!(queue.pop(), Some((running, 3, "running")));
        assert_eq!(
            queue.submit(3, Priority::Demanded, "duplicate", &[]),
            running
        );
        assert!(queue.pop_lazy().is_none());
    }

    #[test]
    fn priorities_are_fifo_and_queued_requests_can_be_promoted() {
        let mut queue = Queue::new();
        let low = queue.submit(1, Priority::Prefetch, "parent", &[]);
        queue.submit(2, Priority::Optimize, "ordinary", &[]);
        queue.submit(3, Priority::Optimize, "loop", &[]);
        assert_eq!(queue.submit(1, Priority::Optimize, "duplicate", &[]), low);
        assert_eq!(queue.pop(), Some((low, 1, "parent")));
        assert_eq!(queue.pop().unwrap().2, "ordinary");
        assert_eq!(queue.pop().unwrap().2, "loop");
        assert!(queue.pop().is_none());
    }

    #[test]
    fn dependencies_inherit_priority_and_wait_for_publication() {
        let mut queue = Queue::new();
        let callee = queue.submit(1, Priority::Prefetch, "callee", &[]);
        let caller = queue.submit(2, Priority::Optimize, "caller", &[callee]);
        queue.submit(3, Priority::Optimize, "unrelated", &[]);
        assert_eq!(queue.pop(), Some((callee, 1, "callee")));
        assert_eq!(queue.pop().unwrap().2, "unrelated");
        assert!(queue.pop().is_none());
        queue.complete(callee, 1);
        assert_eq!(queue.pop(), Some((caller, 2, "caller")));
        queue.complete(caller, 2);
        let next = queue.submit(2, Priority::Optimize, "new version", &[callee]);
        assert_ne!(caller, next);
        assert_eq!(queue.pop(), Some((next, 2, "new version")));
    }

    #[test]
    fn workers_cannot_take_the_same_attempt_twice() {
        let mut queue = Queue::new();
        let ticket = queue.submit(1, Priority::Optimize, 42, &[]);
        assert_eq!(queue.pop(), Some((ticket, 1, 42)));
        assert_eq!(queue.submit(1, Priority::Optimize, 99, &[]), ticket);
        assert!(queue.pop().is_none());
        queue.complete(ticket, 1);
        assert_ne!(queue.submit(1, Priority::Optimize, 99, &[]), ticket);
    }

    #[test]
    fn priority_inheritance_reaches_transitive_dependencies() {
        let mut queue = Queue::new();
        let leaf = queue.submit(1, Priority::Prefetch, "leaf", &[]);
        let middle = queue.submit(2, Priority::Prefetch, "middle", &[leaf]);
        queue.submit(3, Priority::Unlazify, "unrelated", &[]);
        let root = queue.submit(4, Priority::Optimize, "root", &[middle]);
        assert_eq!(queue.pop(), Some((leaf, 1, "leaf")));
        queue.complete(leaf, 1);
        assert_eq!(queue.pop(), Some((middle, 2, "middle")));
        queue.complete(middle, 2);
        assert_eq!(queue.pop(), Some((root, 4, "root")));
        assert_eq!(queue.pop().unwrap().2, "unrelated");
    }
}
