use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use super::liveness::{self, Segment};
use crate::jit::semantic::{Block, Cfg, Graph, Inst, Op, Value, ValueDef, low_level};
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

pub(crate) fn constant(graph: &Graph, value: Value) -> Option<u64> {
    let ValueDef::Result(inst, _) = graph.values[value] else {
        return None;
    };
    let Op::LowLevel(low_level::Op::Integer { bits, ty }) = graph.insts[inst].op else {
        return None;
    };
    let width = ty.bits();
    Some(if width >= 64 {
        bits
    } else {
        bits & ((1 << width) - 1)
    })
}

const REGISTERS: usize = 32;

fn occupancy_index(class: RegClass, reg: u8) -> usize {
    match class {
        RegClass::Int => usize::from(reg),
        RegClass::Float => REGISTERS + usize::from(reg),
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
    let mut depth = SecondaryMap::<Block, u32>::new();
    let mut latches = std::collections::HashMap::<Block, Vec<Block>>::new();
    for &block in &cfg.rpo {
        for successor in graph.successors(block) {
            if cfg.dominates(successor, block) {
                latches.entry(successor).or_default().push(block);
            }
        }
    }
    for (header, mut pending) in latches {
        let mut seen = std::collections::HashSet::new();
        seen.insert(header);
        while let Some(current) = pending.pop() {
            if !seen.insert(current) {
                continue;
            }
            pending.extend(cfg.preds[current].iter().copied());
        }
        for member in seen {
            depth[member] += 1;
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

pub(crate) type Operands<'a> = &'a dyn Fn(Inst, &mut Vec<Value>);

fn value_weights(
    graph: &Graph,
    blocks: &SecondaryMap<Block, f64>,
    operands: Operands,
    terminator_operands: &dyn Fn(Block, &mut Vec<Value>),
) -> SecondaryMap<Value, f64> {
    let mut weights = SecondaryMap::<Value, f64>::new();
    let mut values = Vec::new();
    for block in graph.blocks.keys() {
        let weight = blocks[block];
        for &param in &graph.blocks[block].params {
            weights[param] += weight;
        }
        for inst in graph.blocks[block].insts.iter().copied() {
            values.clear();
            operands(inst, &mut values);
            for &value in &values {
                weights[value] += weight;
            }
            for &result in &graph.insts[inst].results {
                weights[result] += weight;
            }
        }
        values.clear();
        terminator_operands(block, &mut values);
        for &value in &values {
            weights[value] += weight;
        }
    }
    weights
}

fn copies(graph: &Graph, blocks: &SecondaryMap<Block, f64>) -> Vec<(Value, Value, f64)> {
    let mut copies = Vec::new();
    for block in graph.blocks.keys() {
        let weight = blocks[block];
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            if let Op::LowLevel(low_level::Op::Unary {
                op: low_level::Unary::Bitcast | low_level::Unary::Reduce,
                value,
                ..
            }) = data.op
            {
                copies.push((value, data.results[0], weight));
            }
        }
        if let Some(term) = &graph.blocks[block].terminator {
            for target in term.kind.targets() {
                for (&arg, &param) in target.args.iter().zip(&graph.blocks[target.block].params) {
                    copies.push((arg, param, weight));
                }
            }
        }
    }
    copies.sort_by(|a, b| b.2.total_cmp(&a.2));
    copies
}

struct Bundle {
    class: RegClass,
    members: Vec<Value>,
    segments: Vec<Segment>,
    weight: f64,
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
            first: bundle.members[0],
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
    hints: &[Hint],
    operands: Operands,
    terminator_operands: &dyn Fn(Block, &mut Vec<Value>),
    remat: &SecondaryMap<Value, u32>,
    is_call: &dyn Fn(Inst) -> bool,
) -> Result<Allocation, String> {
    let entry = graph.entry;
    let mut cold = SecondaryMap::new();
    for block in graph.blocks.keys() {
        cold[block] = graph.cold.contains(&block) && block != entry;
    }
    let liveness = liveness::compute(
        graph,
        order,
        cfg,
        &cold,
        remat,
        operands,
        terminator_operands,
        is_call,
    )?;
    let blocks = block_weights(graph, cfg, &cold);
    let mut weights = value_weights(graph, &blocks, operands, terminator_operands);
    for value in graph.values.keys() {
        if remat[value] > 1 {
            weights[value] *= f64::from(remat[value] - 1) / 2.0;
        }
    }

    let mut bundles = Vec::new();
    let mut bundle_of = SecondaryMap::<Value, Option<usize>>::new();
    for (value, segments) in liveness.segments.iter() {
        if segments.is_empty() {
            continue;
        }
        let ty = graph.low_level_types[value].ok_or_else(|| format!("missing type for {value}"))?;
        let class = RegClass::of(ty).ok_or_else(|| format!("value type {ty}"))?;
        bundle_of[value] = Some(bundles.len());
        bundles.push(Bundle {
            class,
            members: vec![value],
            segments: segments.clone(),
            weight: weights[value],
        });
    }

    let mut partners = SecondaryMap::<Value, Vec<(Value, f64)>>::new();
    for (a, b, weight) in copies(graph, &blocks) {
        let (Some(x), Some(y)) = (bundle_of[a], bundle_of[b]) else {
            continue;
        };
        if bundles[x].class != bundles[y].class {
            continue;
        }
        partners[a].push((b, weight));
        partners[b].push((a, weight));
        if x == y || overlap(&bundles[x].segments, &bundles[y].segments) {
            continue;
        }
        let (keep, gone) = if bundles[x].members.len() >= bundles[y].members.len() {
            (x, y)
        } else {
            (y, x)
        };
        let moved = std::mem::take(&mut bundles[gone].members);
        let segments = std::mem::take(&mut bundles[gone].segments);
        let weight = std::mem::take(&mut bundles[gone].weight);
        for &member in &moved {
            bundle_of[member] = Some(keep);
        }
        let bundle = &mut bundles[keep];
        bundle.members.extend(moved);
        bundle.segments = merge_segments(&bundle.segments, &segments);
        bundle.weight += weight;
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
    for class in [RegClass::Int, RegClass::Float] {
        for &reg in target.registers(class).caller_saved {
            let map = &mut occupied[occupancy_index(class, reg)];
            for &call in &liveness.calls {
                map.insert(call, call);
            }
        }
    }

    let mut queue: BinaryHeap<Candidate> = bundles
        .iter()
        .enumerate()
        .filter(|(_, bundle)| !bundle.members.is_empty())
        .map(|(index, bundle)| Candidate::new(index, bundle))
        .collect();
    let mut assigned = vec![None; bundles.len()];
    let mut callee_saved = Vec::new();
    while let Some(Candidate { index, .. }) = queue.pop() {
        let class = bundles[index].class;
        let mut preferences = Vec::new();
        for &member in &bundles[index].members {
            for &(reg, weight) in &hinted[member] {
                prefer(&mut preferences, reg, weight);
            }
            for &(other, weight) in &partners[member] {
                if let Some(reg) = bundle_of[other].and_then(|other| assigned[other]) {
                    prefer(&mut preferences, reg, weight);
                }
            }
        }
        preferences.sort_by(|a, b| b.1.total_cmp(&a.1));
        let segments = &bundles[index].segments;
        let hot: Vec<Segment> = segments.iter().filter(|s| !s.cold).copied().collect();
        let occupy = if hot.is_empty() {
            segments.clone()
        } else {
            hot
        };
        let available = target.registers(class);
        let allocatable =
            |reg: &u8| available.caller_saved.contains(reg) || available.callee_saved.contains(reg);
        let choice = preferences
            .iter()
            .map(|&(reg, _)| reg)
            .filter(allocatable)
            .chain(available.caller_saved.iter().copied())
            .chain(available.callee_saved.iter().copied())
            .find(|&reg| free(&occupied[occupancy_index(class, reg)], &occupy));
        let Some(reg) = choice else {
            if bundles[index].members.len() > 1 {
                for member in std::mem::take(&mut bundles[index].members) {
                    let single = Bundle {
                        class,
                        members: vec![member],
                        segments: liveness.segments[member].clone(),
                        weight: weights[member],
                    };
                    bundle_of[member] = Some(bundles.len());
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
        if bundle.members.is_empty() {
            continue;
        }
        let has_hot = bundle.segments.iter().any(|s| !s.cold);
        let has_cold = bundle.segments.iter().any(|s| s.cold);
        let spillable = bundle.members.iter().any(|&member| remat[member] == 0);
        if (reg.is_none() || has_hot && has_cold) && spillable {
            slot_bundles.push((
                bundle.segments[0].start,
                bundle.segments.last().unwrap().end,
                index,
            ));
        }
    }
    let (bundle_slots, slot_count) = assign_slots(slot_bundles, bundles.len());
    for (index, (bundle, &reg)) in bundles.iter().zip(&assigned).enumerate() {
        if bundle.members.is_empty() {
            continue;
        }
        let has_hot = bundle.segments.iter().any(|s| !s.cold);
        let has_cold = bundle.segments.iter().any(|s| s.cold);
        let divided = reg.is_some() && has_hot && has_cold;
        let slot = bundle_slots[index];
        for &member in &bundle.members {
            registers[member] = reg;
            slots[member] = slot;
            split[member] = divided;
        }
    }

    let mut crossing = SecondaryMap::<Block, bool>::new();
    for &block in order {
        crossing[block] = cfg.preds[block]
            .iter()
            .any(|&pred| cold[pred] != cold[block]);
    }
    let mut boundary = SecondaryMap::<Block, Vec<Value>>::new();
    for (value, segments) in liveness.segments.iter() {
        if !split[value] {
            continue;
        }
        for segment in segments {
            if segment.live_in && crossing[segment.block] {
                boundary[segment.block].push(value);
            }
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
