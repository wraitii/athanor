mod liveness;
pub(crate) mod moves;

use std::collections::BTreeMap;

use cranelift_codegen::dominator_tree::DominatorTree;
use cranelift_codegen::entity::SecondaryMap;
use cranelift_codegen::flowgraph::ControlFlowGraph;
use cranelift_codegen::ir::{
    Block, Function, Inst, InstructionData, Opcode, StackSlot, Type, Value, ValueDef, types,
};
use cranelift_codegen::loop_analysis::LoopAnalysis;

use liveness::Segment;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RegClass {
    Int,
    Float,
}

impl RegClass {
    pub(crate) fn of(ty: Type) -> Option<Self> {
        match ty {
            types::I8 | types::I16 | types::I32 | types::I64 => Some(Self::Int),
            types::F32 | types::F64 => Some(Self::Float),
            _ => None,
        }
    }
}

pub(crate) struct Registers {
    pub caller_saved: &'static [u8],
    pub callee_saved: &'static [u8],
}

pub(crate) struct Target {
    pub int: Registers,
    pub float: Registers,
}

impl Target {
    fn registers(&self, class: RegClass) -> &Registers {
        match class {
            RegClass::Int => &self.int,
            RegClass::Float => &self.float,
        }
    }
}

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
}

pub(crate) struct Allocation {
    registers: SecondaryMap<Value, Option<u8>>,
    slots: SecondaryMap<Value, Option<u32>>,
    split: SecondaryMap<Value, bool>,
    cold: SecondaryMap<Block, bool>,
    boundary: SecondaryMap<Block, Vec<Value>>,
    slot_count: u32,
    pub callee_saved: Vec<(RegClass, u8)>,
}

impl Allocation {
    pub(crate) fn place(&self, value: Value, block: Block) -> Place {
        match self.registers[value] {
            Some(reg) if !(self.split[value] && self.cold[block]) => Place::Reg(reg),
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

pub(crate) fn constant(func: &Function, value: Value) -> Option<i64> {
    let ValueDef::Result(inst, _) = func.dfg.value_def(value) else {
        return None;
    };
    match func.dfg.insts[inst] {
        InstructionData::UnaryImm {
            opcode: Opcode::Iconst,
            imm,
        } => Some(imm.bits()),
        _ => None,
    }
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
    func: &Function,
    cfg: &ControlFlowGraph,
    domtree: &DominatorTree,
    cold: &SecondaryMap<Block, bool>,
) -> SecondaryMap<Block, f64> {
    let mut loops = LoopAnalysis::new();
    loops.compute(func, cfg, domtree);
    let mut weights = SecondaryMap::new();
    for block in func.layout.blocks() {
        weights[block] = if cold[block] {
            0.01
        } else {
            8f64.powi(loops.loop_level(block).level().min(6) as i32)
        };
    }
    weights
}

pub(crate) type Operands<'a> = &'a dyn Fn(Inst, &mut Vec<Value>);

fn value_weights(
    func: &Function,
    blocks: &SecondaryMap<Block, f64>,
    operands: Operands,
) -> SecondaryMap<Value, f64> {
    let mut weights = SecondaryMap::<Value, f64>::new();
    let mut values = Vec::new();
    for block in func.layout.blocks() {
        let weight = blocks[block];
        for &param in func.dfg.block_params(block) {
            weights[param] += weight;
        }
        for inst in func.layout.block_insts(block) {
            operands(inst, &mut values);
            for &value in &values {
                weights[value] += weight;
            }
            for &result in func.dfg.inst_results(inst) {
                weights[result] += weight;
            }
        }
    }
    weights
}

fn copies(func: &Function, blocks: &SecondaryMap<Block, f64>) -> Vec<(Value, Value, f64)> {
    let dfg = &func.dfg;
    let mut copies = Vec::new();
    for block in func.layout.blocks() {
        let weight = blocks[block];
        for inst in func.layout.block_insts(block) {
            let data = &dfg.insts[inst];
            if matches!(data.opcode(), Opcode::Bitcast | Opcode::Ireduce) {
                copies.push((dfg.inst_args(inst)[0], dfg.first_result(inst), weight));
            }
            for call in data.branch_destination(&dfg.jump_tables, &dfg.exception_tables) {
                let params = dfg.block_params(call.block(&dfg.value_lists));
                for (arg, &param) in call.args(&dfg.value_lists).zip(params) {
                    if let Some(arg) = arg.as_value() {
                        copies.push((arg, param, weight));
                    }
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
    preferences: Vec<(u8, f64)>,
}

impl Bundle {
    fn prefer(&mut self, reg: u8, weight: f64) {
        match self.preferences.iter_mut().find(|(r, _)| *r == reg) {
            Some((_, total)) => *total += weight,
            None => self.preferences.push((reg, weight)),
        }
    }
}

pub(crate) fn allocate(
    func: &Function,
    order: &[Block],
    target: &Target,
    hints: &[Hint],
    operands: Operands,
) -> Result<Allocation, String> {
    let cfg = ControlFlowGraph::with_function(func);
    let domtree = DominatorTree::with_function(func, &cfg);
    let entry = func.layout.entry_block();
    let mut cold = SecondaryMap::new();
    for block in func.layout.blocks() {
        cold[block] = func.layout.is_cold(block) && Some(block) != entry;
    }
    let mut constants = SecondaryMap::<Value, bool>::new();
    for value in func.dfg.values() {
        constants[value] = constant(func, value).is_some();
    }
    let liveness = liveness::compute(func, order, &cfg, &cold, &constants, operands)?;
    let blocks = block_weights(func, &cfg, &domtree, &cold);
    let weights = value_weights(func, &blocks, operands);

    let mut bundles = Vec::new();
    let mut bundle_of = SecondaryMap::<Value, Option<usize>>::new();
    for (value, segments) in liveness.segments.iter() {
        if segments.is_empty() {
            continue;
        }
        let ty = func.dfg.value_type(value);
        let class = RegClass::of(ty).ok_or_else(|| format!("value type {ty}"))?;
        bundle_of[value] = Some(bundles.len());
        bundles.push(Bundle {
            class,
            members: vec![value],
            segments: segments.clone(),
            weight: weights[value],
            preferences: Vec::new(),
        });
    }

    let mut partners = Vec::new();
    for (a, b, weight) in copies(func, &blocks) {
        let (Some(x), Some(y)) = (bundle_of[a], bundle_of[b]) else {
            continue;
        };
        if x == y || bundles[x].class != bundles[y].class {
            continue;
        }
        if overlap(&bundles[x].segments, &bundles[y].segments) {
            partners.push((a, b, weight));
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

    for hint in hints {
        let Some(index) = bundle_of[hint.value] else {
            continue;
        };
        let bundle = &mut bundles[index];
        if bundle.class == hint.class {
            bundle.prefer(hint.reg, blocks[hint.block]);
        }
    }

    let mut related = vec![Vec::new(); bundles.len()];
    for (a, b, weight) in partners {
        let (Some(x), Some(y)) = (bundle_of[a], bundle_of[b]) else {
            continue;
        };
        if x != y {
            related[x].push((y, weight));
            related[y].push((x, weight));
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

    let mut queue: Vec<usize> = (0..bundles.len())
        .filter(|&index| !bundles[index].members.is_empty())
        .collect();
    queue.sort_by(|&a, &b| {
        bundles[b]
            .weight
            .total_cmp(&bundles[a].weight)
            .then(bundles[a].members[0].cmp(&bundles[b].members[0]))
    });

    let mut assigned = vec![None; bundles.len()];
    let mut callee_saved = Vec::new();
    for &index in &queue {
        let bundle = &bundles[index];
        let class = bundle.class;
        let hot: Vec<Segment> = bundle
            .segments
            .iter()
            .filter(|s| !s.cold)
            .copied()
            .collect();
        let occupy = if hot.is_empty() {
            &bundle.segments
        } else {
            &hot
        };
        let mut preferences = bundle.preferences.clone();
        for &(other, weight) in &related[index] {
            if let Some(reg) = assigned[other] {
                preferences.push((reg, weight));
            }
        }
        preferences.sort_by(|a, b| b.1.total_cmp(&a.1));
        let available = target.registers(class);
        let allocatable =
            |reg: &u8| available.caller_saved.contains(reg) || available.callee_saved.contains(reg);
        let choice = preferences
            .iter()
            .map(|&(reg, _)| reg)
            .filter(allocatable)
            .chain(available.caller_saved.iter().copied())
            .chain(available.callee_saved.iter().copied())
            .find(|&reg| free(&occupied[occupancy_index(class, reg)], occupy));
        let Some(reg) = choice else {
            continue;
        };
        let map = &mut occupied[occupancy_index(class, reg)];
        for segment in occupy {
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
    let mut slot_count = 0;
    for &index in &queue {
        let bundle = &bundles[index];
        let has_hot = bundle.segments.iter().any(|s| !s.cold);
        let has_cold = bundle.segments.iter().any(|s| s.cold);
        let reg = assigned[index];
        let divided = reg.is_some() && has_hot && has_cold;
        let slot = (reg.is_none() || divided).then(|| {
            slot_count += 1;
            slot_count - 1
        });
        for &member in &bundle.members {
            registers[member] = reg;
            slots[member] = slot;
            split[member] = divided;
        }
    }

    let mut crossing = SecondaryMap::<Block, bool>::new();
    for &block in order {
        crossing[block] = cfg
            .pred_iter(block)
            .any(|pred| cold[pred.block] != cold[block]);
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

    Ok(Allocation {
        registers,
        slots,
        split,
        cold,
        boundary,
        slot_count,
        callee_saved,
    })
}

const WORD: u32 = 8;

pub(crate) struct Frame {
    slots: u32,
    stack_slots: SecondaryMap<StackSlot, u32>,
    pub saves: u32,
    pub size: u32,
}

impl Frame {
    pub(crate) fn new(func: &Function, allocation: &Allocation, outgoing: u32) -> Self {
        let saves = outgoing.next_multiple_of(16);
        let slots = saves + WORD * allocation.callee_saved.len() as u32;
        let mut size = slots + WORD * allocation.slot_count;
        let mut stack_slots = SecondaryMap::new();
        for (slot, data) in func.sized_stack_slots.iter() {
            let align = 1u32 << data.align_shift;
            size = size.next_multiple_of(align.max(WORD));
            stack_slots[slot] = size;
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

    pub(crate) fn stack_slot(&self, slot: StackSlot) -> u32 {
        self.stack_slots[slot]
    }
}
