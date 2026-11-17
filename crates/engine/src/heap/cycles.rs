use super::*;
use crate::FxHashMap;

#[derive(Debug, Default)]
pub(crate) struct CycleStats {
    pub(crate) live: usize,
    pub(crate) arena_bytes: usize,
    pub(crate) scanned_slots: usize,
    pub(crate) unreachable: usize,
    pub(crate) cyclic_components: usize,
    pub(crate) cyclic_nodes: usize,
    pub(crate) unreachable_components: usize,
    pub(crate) unreachable_cyclic_nodes: usize,
    pub(crate) cyclic_kinds: std::collections::BTreeMap<&'static str, usize>,
    pub(crate) unreachable_kinds: std::collections::BTreeMap<&'static str, usize>,
}

impl Payload {
    fn visit_edges(&self, mut visit: impl FnMut(Value)) {
        match self {
            Self::Generator(value) => {
                visit(value.closure);
                visit(value.arguments);
                visit(value.receiver);
                value.locals.iter().copied().for_each(visit);
            }
            Self::Enumerator(value) => value.current.into_iter().for_each(visit),
            Self::Set(value) => value.entries.values().copied().for_each(visit),
            Self::SmallObject(value) => {
                value.prototype().into_iter().for_each(&mut visit);
                value.values[..value.len as usize]
                    .iter()
                    .copied()
                    .for_each(visit);
            }
            Self::Object(value) => {
                value.prototype.into_iter().for_each(&mut visit);
                value.properties.visit_values(&mut visit);
            }
            Self::Array(value) => {
                value.prototype.into_iter().for_each(&mut visit);
                if let Some(properties) = &value.properties {
                    properties.visit_values(&mut visit);
                }
                value.elements.iter().copied().for_each(visit);
            }
            Self::Cell(value) => visit(value.value),
            Self::Closure(value) => {
                value.lexical_this.into_iter().for_each(&mut visit);
                value.properties.visit_values(&mut visit);
                value.captures.iter().copied().for_each(visit);
            }
            Self::Native(value) => value.properties.visit_values(&mut visit),
            Self::Vacant
            | Self::Reserved
            | Self::String(_)
            | Self::RegExp(_)
            | Self::TypedArray(_) => {}
        }
    }

    pub(super) fn cycle_kind(&self) -> &'static str {
        match self {
            Self::Vacant => "vacant",
            Self::Reserved => "reserved",
            Self::Native(_) => "native",
            Self::String(_) => "string",
            Self::Generator(_) => "generator",
            Self::RegExp(_) => "regexp",
            Self::Enumerator(_) => "enumerator",
            Self::Set(_) => "set",
            Self::SmallObject(_) | Self::Object(_) => "object",
            Self::Array(_) => "array",
            Self::TypedArray(_) => "typed array",
            Self::Cell(_) => "cell",
            Self::Closure(_) => "closure",
        }
    }
}

impl Heap {
    pub(crate) fn cycle_stats(&self) -> CycleStats {
        // Pending payloads still own edges; finish disposal before inferring external roots.
        self.drain(usize::MAX);
        let data = self.0.borrow();
        let slots: Vec<_> = data
            .arena
            .slots()
            .filter(|slot| slot.references > 0)
            .collect();
        let indices: FxHashMap<_, _> = slots
            .iter()
            .enumerate()
            .map(|(index, slot)| (*slot as *const Slot as u64, index))
            .collect();
        // Residual counts represent owners outside the heap, including host handles.
        let mut external: Vec<_> = slots.iter().map(|slot| slot.references).collect();
        let mut edges = vec![Vec::new(); slots.len()];
        let mut reverse = vec![Vec::new(); slots.len()];
        for (source, slot) in slots.iter().enumerate() {
            slot.payload.visit_edges(|value| {
                let Some(id) = heap_id(value) else { return };
                let target = indices[&id];
                external[target] = external[target]
                    .checked_sub(1)
                    .expect("heap edges exceed reference count");
                edges[source].push(target);
                reverse[target].push(source);
            });
        }
        let mut reachable = vec![false; slots.len()];
        let mut work: Vec<_> = external
            .iter()
            .enumerate()
            .filter_map(|(index, count)| (*count > 0).then_some(index))
            .collect();
        while let Some(node) = work.pop() {
            if std::mem::replace(&mut reachable[node], true) {
                continue;
            }
            work.extend(edges[node].iter().copied());
        }
        let mut stats = CycleStats {
            live: slots.len(),
            arena_bytes: data.arena.capacity() * SLOT_SIZE,
            scanned_slots: data.arena.len(),
            ..CycleStats::default()
        };
        for (index, slot) in slots.iter().enumerate() {
            if !reachable[index] {
                stats.unreachable += 1;
                *stats
                    .unreachable_kinds
                    .entry(slot.payload.cycle_kind())
                    .or_default() += 1;
            }
        }
        // Iterative Kosaraju traversal separates cycle members from retained acyclic tails.
        let mut seen = vec![false; slots.len()];
        let mut order = Vec::with_capacity(slots.len());
        for root in 0..slots.len() {
            if std::mem::replace(&mut seen[root], true) {
                continue;
            }
            let mut stack = vec![(root, 0)];
            while let Some((node, next)) = stack.last_mut() {
                if *next == edges[*node].len() {
                    order.push(*node);
                    stack.pop();
                    continue;
                }
                let child = edges[*node][*next];
                *next += 1;
                if !std::mem::replace(&mut seen[child], true) {
                    stack.push((child, 0));
                }
            }
        }
        seen.fill(false);
        for root in order.into_iter().rev() {
            if seen[root] {
                continue;
            }
            work.push(root);
            let mut members = Vec::new();
            while let Some(node) = work.pop() {
                if std::mem::replace(&mut seen[node], true) {
                    continue;
                }
                members.push(node);
                work.extend(reverse[node].iter().copied());
            }
            let count = members.len();
            if count == 1 && !edges[root].contains(&root) {
                continue;
            }
            for node in members {
                *stats
                    .cyclic_kinds
                    .entry(slots[node].payload.cycle_kind())
                    .or_default() += 1;
            }
            stats.cyclic_components += 1;
            stats.cyclic_nodes += count;
            if !reachable[root] {
                stats.unreachable_components += 1;
                stats.unreachable_cyclic_nodes += count;
            }
        }
        stats
    }
}
