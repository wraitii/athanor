use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use super::liveness::{self, Segment};
use crate::jit::semantic::{Block, Cfg, Graph, Inst, Value, low_level};
use cranelift_codegen::entity::SecondaryMap;

use super::{RegClass, Target};

pub(crate) struct Hint {
    pub value: Value,
    pub class: RegClass,
    pub reg: u8,
    pub block: Block,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Place {
    Reg(u8),
    Slot(u32),
    Remat,
}

pub(crate) struct Allocation {
    registers: SecondaryMap<Value, Option<u8>>,
    slots: SecondaryMap<Value, Option<u32>>,
    split: SecondaryMap<Value, bool>,
    remat: SecondaryMap<Value, bool>,
    cold: SecondaryMap<Block, bool>,
    boundary: SecondaryMap<Block, Vec<Value>>,
    slot_count: u32,
    pub callee_saved: Vec<(RegClass, u8)>,
}

impl Allocation {
    pub(crate) fn place(&self, value: Value, block: Block) -> Place {
        match self.registers[value] {
            Some(reg) if !(self.split[value] && self.cold[block]) => Place::Reg(reg),
            _ if self.remat[value] => Place::Remat,
            _ => Place::Slot(self.slots[value].expect("values without a register have a slot")),
        }
    }

    pub(crate) fn is_cold(&self, block: Block) -> bool {
        self.cold[block]
    }

    pub(crate) fn boundary(&self, block: Block) -> &[Value] {
        &self.boundary[block]
    }
}

const REGISTERS: usize = 32;

fn occupancy_index(class: RegClass, reg: u8) -> usize {
    match class {
        RegClass::Int => usize::from(reg),
        RegClass::Float | RegClass::Vector => REGISTERS + usize::from(reg),
    }
}

fn free(occupied: &BTreeMap<u32, u32>, segments: &[Segment]) -> bool {
    segments.iter().all(|segment| {
        occupied
            .range(..=segment.end)
            .next_back()
            .is_none_or(|(_, &end)| end < segment.start)
    })
}

fn overlap(a: &[Segment], b: &[Segment]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].end < b[j].start {
            i += 1;
        } else if b[j].end < a[i].start {
            j += 1;
        } else {
            return true;
        }
    }
    false
}

fn assign_slots(mut intervals: Vec<(u32, u32, usize)>, count: usize) -> (Vec<Option<u32>>, u32) {
    intervals.sort_unstable();
    let mut active = BinaryHeap::<Reverse<(u32, u32)>>::new();
    let mut available = BinaryHeap::<Reverse<u32>>::new();
    let mut slots = vec![None; count];
    let mut slot_count = 0;
    for (start, end, index) in intervals {
        while active.peek().is_some_and(|Reverse((end, _))| *end < start) {
            available.push(Reverse(active.pop().unwrap().0.1));
        }
        let slot = available
            .pop()
            .map(|Reverse(slot)| slot)
            .unwrap_or_else(|| {
                let slot = slot_count;
                slot_count += 1;
                slot
            });
        slots[index] = Some(slot);
        active.push(Reverse((end, slot)));
    }
    (slots, slot_count)
}

fn merge_segments(a: &[Segment], b: &[Segment]) -> Vec<Segment> {
    let mut merged = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].start < b[j].start {
            merged.push(a[i]);
            i += 1;
        } else {
            merged.push(b[j]);
            j += 1;
        }
    }
    merged.extend_from_slice(&a[i..]);
    merged.extend_from_slice(&b[j..]);
    merged
}

fn block_weights(
    graph: &Graph,
    cfg: &Cfg,
    cold: &SecondaryMap<Block, bool>,
) -> SecondaryMap<Block, f64> {
    let mut position = SecondaryMap::<Block, u32>::with_default(u32::MAX);
    for (index, &block) in cfg.rpo.iter().enumerate() {
        position[block] = index as u32;
    }
    let mut latches = Vec::new();
    for &block in &cfg.rpo {
        for successor in graph.successors(block) {
            if position[successor] <= position[block] && cfg.dominates(successor, block) {
                latches.push((successor, block));
            }
        }
    }
    latches.sort_by_key(|&(header, _)| header);
    let mut depth = SecondaryMap::<Block, u32>::new();
    let mut seen = SecondaryMap::<Block, u32>::new();
    let mut pending = Vec::new();
    for (stamp, group) in latches.chunk_by(|a, b| a.0 == b.0).enumerate() {
        let stamp = stamp as u32 + 1;
        let header = group[0].0;
        seen[header] = stamp;
        depth[header] += 1;
        pending.extend(group.iter().map(|&(_, latch)| latch));
        while let Some(current) = pending.pop() {
            if seen[current] == stamp {
                continue;
            }
            seen[current] = stamp;
            depth[current] += 1;
            pending.extend(cfg.preds[current].iter().copied());
        }
    }
    let mut weights = SecondaryMap::new();
    for &block in &cfg.rpo {
        weights[block] = if cold[block] {
            0.01
        } else {
            8f64.powi(depth[block].min(6) as i32)
        };
    }
    weights
}

#[derive(Default)]
pub(crate) struct Input {
    pub classes: SecondaryMap<Value, Option<RegClass>>,
    pub remat: SecondaryMap<Value, u32>,
    pub calls: SecondaryMap<Inst, bool>,
    pub copies: Vec<(Value, Value, Block)>,
    pub hints: Vec<Hint>,
}

fn value_weights(graph: &Graph, blocks: &SecondaryMap<Block, f64>) -> SecondaryMap<Value, f64> {
    let mut weights = SecondaryMap::<Value, f64>::new();
    for (block, data) in graph.blocks.iter() {
        let weight = blocks[block];
        for &param in &data.params {
            weights[param] += weight;
        }
        for &inst in &data.insts {
            for value in graph.insts[inst].op.operands_iter() {
                weights[value] += weight;
            }
            for &result in &graph.insts[inst].results {
                weights[result] += weight;
            }
        }
        if let Some(term) = &data.terminator {
            for value in term.kind.operands_iter() {
                weights[value] += weight;
            }
        }
    }
    weights
}

fn copies(
    graph: &Graph,
    blocks: &SecondaryMap<Block, f64>,
    input: &Input,
) -> Vec<(Value, Value, f64)> {
    let mut copies: Vec<_> = input
        .copies
        .iter()
        .map(|&(from, to, block)| (from, to, blocks[block]))
        .collect();
    for (block, data) in graph.blocks.iter() {
        if let Some(term) = &data.terminator {
            for target in term.kind.targets() {
                for (&arg, &param) in target.args.iter().zip(&graph.blocks[target.block].params) {
                    copies.push((arg, param, blocks[block]));
                }
            }
        }
    }
    copies.sort_by(|a, b| b.2.total_cmp(&a.2));
    copies
}

struct Bundle {
    class: RegClass,
    first: Value,
    last: Value,
    size: u32,
    merged: Option<Vec<Segment>>,
    weight: f64,
}

impl Bundle {
    fn single(class: RegClass, value: Value, weight: f64) -> Self {
        Self {
            class,
            first: value,
            last: value,
            size: 1,
            merged: None,
            weight,
        }
    }

    fn segments<'a>(&'a self, liveness: &'a liveness::Liveness) -> &'a [Segment] {
        self.merged
            .as_deref()
            .unwrap_or_else(|| liveness.of(self.first))
    }

    fn members<'a>(
        &self,
        next: &'a SecondaryMap<Value, Option<Value>>,
    ) -> impl Iterator<Item = Value> + 'a {
        std::iter::successors((self.size > 0).then_some(self.first), |&value| next[value])
    }
}

struct Candidate {
    weight: f64,
    first: Value,
    index: usize,
}

impl Candidate {
    fn new(index: usize, bundle: &Bundle) -> Self {
        Self {
            weight: bundle.weight,
            first: bundle.first,
            index,
        }
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight
            .total_cmp(&other.weight)
            .then(other.first.cmp(&self.first))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

fn prefer(preferences: &mut Vec<(u8, f64)>, reg: u8, weight: f64) {
    match preferences.iter_mut().find(|(r, _)| *r == reg) {
        Some((_, total)) => *total += weight,
        None => preferences.push((reg, weight)),
    }
}

pub(crate) fn allocate(
    graph: &Graph,
    cfg: &Cfg,
    order: &[Block],
    target: &Target,
    input: &Input,
) -> Result<Allocation, String> {
    let remat = &input.remat;
    let hints = &input.hints;
    let entry = graph.entry;
    let mut cold = SecondaryMap::new();
    for block in graph.blocks.keys() {
        cold[block] = graph.cold.contains(&block) && block != entry;
    }
    let liveness = liveness::compute(graph, order, cfg, &cold, input)?;
    let blocks = block_weights(graph, cfg, &cold);
    let mut weights = value_weights(graph, &blocks);
    for value in graph.values.keys() {
        if remat[value] > 1 {
            weights[value] *= f64::from(remat[value] - 1) / 2.0;
        }
    }

    let mut bundles = Vec::new();
    let mut bundle_of = SecondaryMap::<Value, Option<usize>>::new();
    let mut next = SecondaryMap::<Value, Option<Value>>::new();
    for value in graph.values.keys() {
        if liveness.of(value).is_empty() {
            continue;
        }
        let class =
            input.classes[value].ok_or_else(|| format!("missing register class for {value}"))?;
        bundle_of[value] = Some(bundles.len());
        bundles.push(Bundle::single(class, value, weights[value]));
    }

    let mut partners = Vec::new();
    for (a, b, weight) in copies(graph, &blocks, input) {
        let (Some(x), Some(y)) = (bundle_of[a], bundle_of[b]) else {
            continue;
        };
        if bundles[x].class != bundles[y].class {
            continue;
        }
        partners.push((a, b, weight));
        partners.push((b, a, weight));
        if x == y
            || overlap(
                bundles[x].segments(&liveness),
                bundles[y].segments(&liveness),
            )
        {
            continue;
        }
        let (keep, gone) = if bundles[x].size >= bundles[y].size {
            (x, y)
        } else {
            (y, x)
        };
        let segments = merge_segments(
            bundles[keep].segments(&liveness),
            bundles[gone].segments(&liveness),
        );
        let empty = Bundle {
            size: 0,
            ..Bundle::single(bundles[gone].class, bundles[gone].first, 0.0)
        };
        let moved = std::mem::replace(&mut bundles[gone], empty);
        for member in moved.members(&next) {
            bundle_of[member] = Some(keep);
        }
        let bundle = &mut bundles[keep];
        next[bundle.last] = Some(moved.first);
        bundle.last = moved.last;
        bundle.size += moved.size;
        bundle.merged = Some(segments);
        bundle.weight += moved.weight;
    }
    partners.sort_by_key(|&(value, _, _)| value);
    let mut partner_range = SecondaryMap::<Value, (u32, u32)>::new();
    for (index, &(value, _, _)) in partners.iter().enumerate() {
        let range = &mut partner_range[value];
        if range.0 == range.1 {
            range.0 = index as u32;
        }
        range.1 = index as u32 + 1;
    }

    let mut hinted = SecondaryMap::<Value, Vec<(u8, f64)>>::new();
    for hint in hints {
        let Some(index) = bundle_of[hint.value] else {
            continue;
        };
        if bundles[index].class == hint.class {
            prefer(&mut hinted[hint.value], hint.reg, blocks[hint.block]);
        }
    }

    let mut occupied = vec![BTreeMap::<u32, u32>::new(); 2 * REGISTERS];
    let mut queue: BinaryHeap<Candidate> = bundles
        .iter()
        .enumerate()
        .filter(|(_, bundle)| bundle.size > 0)
        .map(|(index, bundle)| Candidate::new(index, bundle))
        .collect();
    let mut assigned = vec![None; bundles.len()];
    let mut callee_saved = Vec::new();
    let mut preferences = Vec::new();
    let mut occupy = Vec::new();
    while let Some(Candidate { index, .. }) = queue.pop() {
        let class = bundles[index].class;
        preferences.clear();
        for member in bundles[index].members(&next) {
            for &(reg, weight) in &hinted[member] {
                prefer(&mut preferences, reg, weight);
            }
            let (start, end) = partner_range[member];
            for &(_, other, weight) in &partners[start as usize..end as usize] {
                if let Some(reg) = bundle_of[other].and_then(|other| assigned[other]) {
                    prefer(&mut preferences, reg, weight);
                }
            }
        }
        preferences.sort_by(|a, b| b.1.total_cmp(&a.1));
        let segments = bundles[index].segments(&liveness);
        occupy.clear();
        occupy.extend(segments.iter().filter(|s| !s.cold));
        if occupy.is_empty() {
            occupy.extend_from_slice(segments);
        }
        let crosses_call = occupy.iter().any(|segment| liveness.crosses_call(segment));
        let available = target.registers(class);
        let allocatable =
            |reg: &u8| available.caller_saved.contains(reg) || available.callee_saved.contains(reg);
        let usable = |reg: u8| {
            !(crosses_call && available.caller_saved.contains(&reg))
                && free(&occupied[occupancy_index(class, reg)], &occupy)
        };
        let choice = preferences
            .iter()
            .map(|&(reg, _)| reg)
            .filter(allocatable)
            .chain(available.caller_saved.iter().copied())
            .chain(available.callee_saved.iter().copied())
            .find(|&reg| usable(reg));
        let Some(reg) = choice else {
            if bundles[index].size > 1 {
                let mut member = Some(bundles[index].first);
                bundles[index].size = 0;
                bundles[index].merged = None;
                while let Some(current) = member {
                    member = next[current].take();
                    let single = Bundle::single(class, current, weights[current]);
                    bundle_of[current] = Some(bundles.len());
                    queue.push(Candidate::new(bundles.len(), &single));
                    bundles.push(single);
                    assigned.push(None);
                }
            }
            continue;
        };
        let map = &mut occupied[occupancy_index(class, reg)];
        for segment in &occupy {
            map.insert(segment.start, segment.end);
        }
        assigned[index] = Some(reg);
        if available.callee_saved.contains(&reg) && !callee_saved.contains(&(class, reg)) {
            callee_saved.push((class, reg));
        }
    }
    callee_saved.sort_by_key(|&(class, reg)| occupancy_index(class, reg));

    let mut registers = SecondaryMap::new();
    let mut slots = SecondaryMap::new();
    let mut split = SecondaryMap::new();
    let mut slot_bundles = Vec::new();
    for (index, (bundle, &reg)) in bundles.iter().zip(&assigned).enumerate() {
        if bundle.size == 0 {
            continue;
        }
        let segments = bundle.segments(&liveness);
        let has_hot = segments.iter().any(|s| !s.cold);
        let has_cold = segments.iter().any(|s| s.cold);
        let spillable = bundle.members(&next).any(|member| remat[member] == 0);
        if (reg.is_none() || has_hot && has_cold) && spillable {
            slot_bundles.push((segments[0].start, segments.last().unwrap().end, index));
        }
    }
    let (vector_bundles, slot_bundles): (Vec<_>, Vec<_>) = slot_bundles
        .into_iter()
        .partition(|&(_, _, index)| bundles[index].class == RegClass::Vector);
    let (mut bundle_slots, mut slot_count) = assign_slots(slot_bundles, bundles.len());
    let (vector_slots, vector_count) = assign_slots(vector_bundles, bundles.len());
    if vector_count > 0 {
        // Vector slots take two words each, placed after the word slots.
        let base = slot_count.next_multiple_of(2);
        for (slot, vector) in bundle_slots.iter_mut().zip(vector_slots) {
            if let Some(vector) = vector {
                *slot = Some(base + 2 * vector);
            }
        }
        slot_count = base + 2 * vector_count;
    }
    for (index, (bundle, &reg)) in bundles.iter().zip(&assigned).enumerate() {
        if bundle.size == 0 {
            continue;
        }
        let segments = bundle.segments(&liveness);
        let has_hot = segments.iter().any(|s| !s.cold);
        let has_cold = segments.iter().any(|s| s.cold);
        let divided = reg.is_some() && has_hot && has_cold;
        let slot = bundle_slots[index];
        for member in bundle.members(&next) {
            registers[member] = reg;
            slots[member] = slot;
            split[member] = divided;
        }
    }

    let mut boundary = SecondaryMap::<Block, Vec<Value>>::new();
    for &(value, block) in &liveness.boundary {
        if split[value] {
            boundary[block].push(value);
        }
    }

    let mut rematerialized = SecondaryMap::new();
    for value in graph.values.keys() {
        rematerialized[value] = remat[value] > 0;
    }

    Ok(Allocation {
        registers,
        slots,
        split,
        remat: rematerialized,
        cold,
        boundary,
        slot_count,
        callee_saved,
    })
}

const WORD: u32 = 8;

pub(crate) struct Frame {
    slots: u32,
    stack_slots: Vec<u32>,
    pub saves: u32,
    pub size: u32,
}

impl Frame {
    pub(crate) fn new(graph: &Graph, allocation: &Allocation, outgoing: u32) -> Self {
        let saves = outgoing.next_multiple_of(16);
        let slots = saves + WORD * allocation.callee_saved.len() as u32;
        let mut size = slots + WORD * allocation.slot_count;
        let mut stack_slots = Vec::new();
        for data in &graph.low_level_slots {
            let align = 1u32 << data.align_shift;
            size = size.next_multiple_of(align.max(WORD));
            stack_slots.push(size);
            size += data.size;
        }
        Self {
            slots,
            stack_slots,
            saves,
            size: size.next_multiple_of(16),
        }
    }

    pub(crate) fn slot(&self, index: u32) -> u32 {
        self.slots + WORD * index
    }

    pub(crate) fn stack_slot(&self, slot: low_level::StackSlot) -> u32 {
        self.stack_slots[slot.0]
    }
}

#[cfg(test)]
mod tests {
    use super::assign_slots;

    #[test]
    fn spill_slots_reuse_only_after_inclusive_lifetimes() {
        let (slots, count) = assign_slots(vec![(0, 10, 0), (5, 7, 1), (7, 9, 2), (11, 12, 3)], 4);
        assert_eq!(count, 3);
        assert_ne!(slots[0], slots[1]);
        assert_ne!(slots[1], slots[2]);
        assert_eq!(slots[0], slots[3]);
    }
}
