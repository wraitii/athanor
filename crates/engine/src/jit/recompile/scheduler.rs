use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::Hash;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Priority {
    Active,
    Normal,
    NextCall,
    Prefetch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Ticket(u64);

struct Pending<K, T> {
    key: K,
    priority: Priority,
    dependencies: HashSet<Ticket>,
    value: T,
}

pub(super) struct Queue<K, T> {
    next: u64,
    keys: HashMap<K, Ticket>,
    outstanding: HashSet<Ticket>,
    pending: HashMap<Ticket, Pending<K, T>>,
    ready: BinaryHeap<Reverse<(Priority, Ticket)>>,
}

impl<K: Copy + Eq + Hash, T> Queue<K, T> {
    pub(super) fn new() -> Self {
        Self {
            next: 0,
            keys: HashMap::new(),
            outstanding: HashSet::new(),
            pending: HashMap::new(),
            ready: BinaryHeap::new(),
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
        let dependencies: HashSet<_> = dependencies
            .iter()
            .copied()
            .filter(|dependency| self.outstanding.contains(dependency))
            .collect();
        for &dependency in &dependencies {
            self.promote(dependency, priority);
        }
        if dependencies.is_empty() {
            self.ready.push(Reverse((priority, ticket)));
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
            self.ready.push(Reverse((priority, ticket)));
        }
        for dependency in dependencies {
            self.promote(dependency, priority);
        }
    }

    pub(super) fn pop(&mut self) -> Option<(Ticket, K, T)> {
        while let Some(Reverse((priority, ticket))) = self.ready.pop() {
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
                self.ready.push(Reverse((pending.priority, waiting)));
            }
        }
    }

    pub(super) fn cancel_pending(&mut self, key: K) -> Option<Ticket> {
        let ticket = *self.keys.get(&key)?;
        self.pending.remove(&ticket)?;
        self.complete(ticket, key);
        Some(ticket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_releases_dependencies_but_cannot_cancel_running_jobs() {
        let mut queue = Queue::new();
        let callee = queue.submit(1, Priority::Prefetch, "callee", &[]);
        let caller = queue.submit(2, Priority::Normal, "caller", &[callee]);
        assert_eq!(queue.cancel_pending(1), Some(callee));
        assert_eq!(queue.pop(), Some((caller, 2, "caller")));
        assert_eq!(queue.cancel_pending(2), None);
        queue.complete(caller, 2);
        assert!(queue.pop().is_none());
        let next = queue.submit(1, Priority::Prefetch, "retry", &[]);
        queue.submit(3, Priority::NextCall, "next call", &[]);
        assert_eq!(queue.pop().unwrap().2, "next call");
        assert_eq!(queue.pop(), Some((next, 1, "retry")));
    }

    #[test]
    fn priorities_are_fifo_and_queued_requests_can_be_promoted() {
        let mut queue = Queue::new();
        let low = queue.submit(1, Priority::NextCall, "parent", &[]);
        queue.submit(2, Priority::Normal, "ordinary", &[]);
        queue.submit(3, Priority::Active, "loop", &[]);
        assert_eq!(queue.submit(1, Priority::Active, "duplicate", &[]), low);
        assert_eq!(queue.pop(), Some((low, 1, "parent")));
        assert_eq!(queue.pop().unwrap().2, "loop");
        assert_eq!(queue.pop().unwrap().2, "ordinary");
        assert!(queue.pop().is_none());
    }

    #[test]
    fn dependencies_inherit_priority_and_wait_for_publication() {
        let mut queue = Queue::new();
        let callee = queue.submit(1, Priority::NextCall, "callee", &[]);
        let caller = queue.submit(2, Priority::Active, "caller", &[callee]);
        queue.submit(3, Priority::Normal, "unrelated", &[]);
        assert_eq!(queue.pop(), Some((callee, 1, "callee")));
        assert_eq!(queue.pop().unwrap().2, "unrelated");
        assert!(queue.pop().is_none());
        queue.complete(callee, 1);
        assert_eq!(queue.pop(), Some((caller, 2, "caller")));
        queue.complete(caller, 2);
        let next = queue.submit(2, Priority::Normal, "new version", &[callee]);
        assert_ne!(caller, next);
        assert_eq!(queue.pop(), Some((next, 2, "new version")));
    }

    #[test]
    fn workers_cannot_take_the_same_attempt_twice() {
        let mut queue = Queue::new();
        let ticket = queue.submit(1, Priority::Normal, 42, &[]);
        assert_eq!(queue.pop(), Some((ticket, 1, 42)));
        assert_eq!(queue.submit(1, Priority::Active, 99, &[]), ticket);
        assert!(queue.pop().is_none());
        queue.complete(ticket, 1);
        assert_ne!(queue.submit(1, Priority::Normal, 99, &[]), ticket);
    }

    #[test]
    fn priority_inheritance_reaches_transitive_dependencies() {
        let mut queue = Queue::new();
        let leaf = queue.submit(1, Priority::NextCall, "leaf", &[]);
        let middle = queue.submit(2, Priority::NextCall, "middle", &[leaf]);
        queue.submit(3, Priority::Normal, "unrelated", &[]);
        let root = queue.submit(4, Priority::Active, "root", &[middle]);
        assert_eq!(queue.pop(), Some((leaf, 1, "leaf")));
        queue.complete(leaf, 1);
        assert_eq!(queue.pop(), Some((middle, 2, "middle")));
        queue.complete(middle, 2);
        assert_eq!(queue.pop(), Some((root, 4, "root")));
        assert_eq!(queue.pop().unwrap().2, "unrelated");
    }
}
