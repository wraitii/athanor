use std::collections::{HashMap, HashSet};

use cranelift_codegen::entity::SecondaryMap;

use crate::jit::semantic::{
    Block, Cfg, Facts, Graph, Inst, Ownership, Types, Value, borrowed_receiver,
};

const HEAP: Types = Types::STRING.union(Types::OBJECT).union(Types::OTHER);

/// Frame slots for values that outlive their block. Each slot owns exactly its current contents,
/// so storing into a slot that held an earlier value must release it.
pub(super) struct Slots {
    slot: SecondaryMap<Value, Option<u32>>,
    shared: Vec<bool>,
    forwarded: HashSet<Value>,
}

impl Slots {
    pub(super) fn none() -> Self {
        Self {
            slot: SecondaryMap::new(),
            shared: Vec::new(),
            forwarded: HashSet::new(),
        }
    }

    pub(super) fn count(&self) -> u32 {
        self.shared.len() as u32
    }

    pub(super) fn of(&self, value: Value) -> Option<u32> {
        self.slot[value]
    }

    /// Every incoming edge already owns this value in its destination slot.
    pub(super) fn forwarded(&self, value: Value) -> bool {
        self.forwarded.contains(&value)
    }

    /// Whether a store to this value's slot may find an earlier value there.
    pub(super) fn shared(&self, value: Value) -> bool {
        self.slot[value].is_some_and(|slot| self.shared[slot as usize])
    }

    pub(super) fn assigned(&self) -> impl Iterator<Item = (Value, u32)> + '_ {
        self.slot
            .iter()
            .filter_map(|(value, slot)| slot.map(|slot| (value, slot)))
    }

    fn take(&mut self, in_use: &mut Vec<bool>, value: Value) {
        let index = match in_use.iter().position(|used| !used) {
            Some(index) => {
                self.shared[index] = true;
                index
            }
            None => {
                in_use.push(false);
                self.shared.push(false);
                in_use.len() - 1
            }
        };
        in_use[index] = true;
        self.slot[value] = Some(index as u32);
    }

    fn free(&self, in_use: &mut [bool], value: Value) {
        if let Some(slot) = self.slot[value] {
            in_use[slot as usize] = false;
        }
    }
}

/// Assigns slots to owned values by liveness, in reverse postorder so that every live-in value
/// already has one. Params always get a slot, as do results that outlive their block. Other
/// results stay owned by their temporary until the block's clear, unless `prefers` says their
/// operation can skip the temporary when it has a slot.
pub(super) fn assign(
    graph: &Graph,
    cfg: &Cfg,
    owned: impl Fn(Value) -> bool,
    prefers: impl Fn(Value) -> bool,
) -> Slots {
    let mut receivers = HashMap::new();
    for (_, data) in graph.insts.iter() {
        if let Some(result) = data.result()
            && graph.borrowed.contains(&result)
            && let Some(receiver) = borrowed_receiver(&data.op)
        {
            receivers.insert(result, vec![receiver]);
        }
    }
    for &block in &cfg.rpo {
        for (index, &param) in graph.blocks[block].params.iter().enumerate() {
            if graph.borrowed.contains(&param) {
                let incoming = cfg.preds[block]
                    .iter()
                    .flat_map(|&pred| {
                        graph.blocks[pred]
                            .terminator
                            .as_ref()
                            .unwrap()
                            .kind
                            .targets()
                    })
                    .filter(|target| target.block == block)
                    .map(|target| target.args[index])
                    .collect();
                receivers.insert(param, incoming);
            }
        }
    }
    let anchored = |value: Value, live: &mut HashSet<Value>| {
        let mut pending = vec![value];
        let mut visited = HashSet::new();
        while let Some(current) = pending.pop() {
            if !visited.insert(current) {
                continue;
            }
            if owned(current) {
                live.insert(current);
            }
            if let Some(owners) = receivers.get(&current) {
                pending.extend(owners.iter().copied());
            }
        }
    };
    let terminator_uses = |block: Block| {
        let mut uses = HashSet::new();
        if let Some(terminator) = &graph.blocks[block].terminator {
            for value in terminator.kind.operands_iter() {
                anchored(value, &mut uses);
            }
        }
        uses
    };
    let mut live_in: SecondaryMap<Block, HashSet<Value>> = SecondaryMap::new();
    let mut live_out: SecondaryMap<Block, HashSet<Value>> = SecondaryMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for &block in cfg.rpo.iter().rev() {
            let data = &graph.blocks[block];
            let mut live: HashSet<Value> = graph
                .successors(block)
                .into_iter()
                .flat_map(|successor| live_in[successor].iter().copied())
                .collect();
            live_out[block] = live.clone();
            live.extend(terminator_uses(block));
            for &inst in data.insts.iter().rev() {
                let data = &graph.insts[inst];
                if let Some(result) = data.result() {
                    live.remove(&result);
                }
                for value in data.op.operands_iter() {
                    anchored(value, &mut live);
                }
            }
            for param in &data.params {
                live.remove(param);
            }
            if live != live_in[block] {
                live_in[block] = live;
                changed = true;
            }
        }
    }

    let mut slots = Slots::none();
    for &block in &cfg.rpo {
        let data = &graph.blocks[block];
        let mut live = live_out[block].clone();
        live.extend(terminator_uses(block));
        let mut dying = vec![Vec::new(); data.insts.len()];
        let mut used = HashSet::new();
        for (index, &inst) in data.insts.iter().enumerate().rev() {
            let data = &graph.insts[inst];
            if let Some(result) = data.result()
                && live.remove(&result)
            {
                used.insert(result);
            }
            let mut uses = HashSet::new();
            for value in data.op.operands_iter() {
                anchored(value, &mut uses);
            }
            for value in uses {
                if live.insert(value) {
                    dying[index].push(value);
                }
            }
        }
        let mut in_use = vec![false; slots.shared.len()];
        for value in &live_in[block] {
            if let Some(slot) = slots.slot[*value] {
                in_use[slot as usize] = true;
            }
        }
        for &param in &data.params {
            if owned(param) {
                slots.take(&mut in_use, param);
            }
        }
        for &param in &data.params {
            if !live.contains(&param) {
                slots.free(&mut in_use, param);
            }
        }
        let escaping = &live_out[block];
        let terminator = terminator_uses(block);
        for (index, &inst) in data.insts.iter().enumerate() {
            for &value in &dying[index] {
                slots.free(&mut in_use, value);
            }
            if let Some(result) = graph.insts[inst].result()
                && owned(result)
                && (escaping.contains(&result)
                    || terminator.contains(&result)
                    || (prefers(result) && used.contains(&result)))
            {
                slots.take(&mut in_use, result);
            }
        }
    }
    for &block in &cfg.rpo {
        if cfg.preds[block].is_empty() {
            continue;
        }
        for (index, &param) in graph.blocks[block].params.iter().enumerate() {
            let Some(slot) = slots.of(param) else {
                continue;
            };
            let forwarded = cfg.preds[block].iter().all(|&pred| {
                graph.blocks[pred]
                    .terminator
                    .as_ref()
                    .unwrap()
                    .kind
                    .targets()
                    .filter(|target| target.block == block)
                    .all(|target| slots.of(target.args[index]) == Some(slot))
            });
            if forwarded {
                slots.forwarded.insert(param);
            }
        }
    }
    slots
}

/// For each instruction, the results whose references can be released right after it: their
/// last use within their own block, counting the uses of every borrowed value that may alias them.
/// Results used by a terminator or another block keep their references until the block's end.
pub(super) fn releases(graph: &Graph, facts: &Facts, cfg: &Cfg) -> HashMap<Inst, Vec<Value>> {
    let mut aliases: HashMap<Value, Vec<Value>> = HashMap::new();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let Some(result) = data.result() else {
                continue;
            };
            if !facts.of(result).intersects(HEAP) {
                continue;
            }
            if graph.contract(&data.op, facts).result != Ownership::Borrowed {
                continue;
            }
            aliases.insert(result, data.op.operands_iter().collect());
        }
    }
    let anchors: HashMap<Value, Vec<Value>> = aliases
        .keys()
        .map(|&value| {
            let mut pending = vec![value];
            let mut visited = HashSet::new();
            while let Some(current) = pending.pop() {
                if !visited.insert(current) {
                    continue;
                }
                if let Some(owners) = aliases.get(&current) {
                    pending.extend(owners.iter().copied());
                }
            }
            (value, visited.into_iter().collect())
        })
        .collect();
    let mut defined: HashMap<Value, (Block, usize, Inst)> = HashMap::new();
    for &block in &cfg.rpo {
        for (index, &inst) in graph.blocks[block].insts.iter().enumerate() {
            if let Some(result) = graph.insts[inst].result() {
                defined.insert(result, (block, index, inst));
            }
        }
    }
    let mut last: HashMap<Value, Option<usize>> = HashMap::new();
    for &block in &cfg.rpo {
        let data = &graph.blocks[block];
        let escape = |value: Value, last: &mut HashMap<Value, Option<usize>>| {
            for &value in anchored(&anchors, &value) {
                last.insert(value, None);
            }
        };
        for (index, &inst) in data.insts.iter().enumerate() {
            for operand in graph.insts[inst].op.operands_iter() {
                for &value in anchored(&anchors, &operand) {
                    match defined.get(&value) {
                        Some(&(owner, _, _)) if owner == block => {
                            let entry = last.entry(value).or_insert(Some(index));
                            if let Some(position) = entry {
                                *position = (*position).max(index);
                            }
                        }
                        _ => {
                            last.insert(value, None);
                        }
                    }
                }
            }
        }
        if let Some(terminator) = &data.terminator {
            for operand in terminator.kind.operands_iter() {
                escape(operand, &mut last);
            }
        }
    }
    let mut releases: HashMap<Inst, Vec<Value>> = HashMap::new();
    for (&value, &(block, index, inst)) in &defined {
        let position = match last.get(&value) {
            Some(None) => continue,
            Some(Some(position)) => *position,
            None => index,
        };
        let after = if position == index {
            inst
        } else {
            graph.blocks[block].insts[position]
        };
        releases.entry(after).or_default().push(value);
    }
    releases
}

/// A value together with every value it borrows from, directly or not.
fn anchored<'a>(anchors: &'a HashMap<Value, Vec<Value>>, value: &'a Value) -> &'a [Value] {
    match anchors.get(value) {
        Some(values) => values,
        None => std::slice::from_ref(value),
    }
}
