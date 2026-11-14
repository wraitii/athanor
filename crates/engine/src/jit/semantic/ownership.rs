use crate::{FxHashMap, FxHashSet};

use cranelift_codegen::entity::SecondaryMap;
use oxc_span::Span;

use super::borrow::receiver as borrowed_receiver;
use super::contracts::{self, ResultReference};
use super::{
    Block, BlockData, Cfg, Facts, Graph, Inst, InstData, Op, Ownership, Target, Terminator,
    TerminatorKind, Value, ValueType,
};
use crate::jit::helpers::Helper;

const HEAP: ValueType = ValueType::STRING
    .union(ValueType::OBJECT)
    .union(ValueType::OTHER);

/// Frame slots for values that outlive their block. Each slot owns exactly its current contents,
/// so storing into a slot that held an earlier value must release it.
#[derive(Clone)]
pub(in crate::jit) struct Slots {
    slot: SecondaryMap<Value, Option<u32>>,
    shared: Vec<bool>,
    forwarded: FxHashSet<Value>,
}

impl Slots {
    fn none() -> Self {
        Self {
            slot: SecondaryMap::new(),
            shared: Vec::new(),
            forwarded: FxHashSet::default(),
        }
    }

    pub(in crate::jit) fn count(&self) -> u32 {
        self.shared.len() as u32
    }

    pub(in crate::jit) fn of(&self, value: Value) -> Option<u32> {
        self.slot[value]
    }

    /// The value is in its slot on arrival: some incoming edges already own it there, and the
    /// others store it on the way.
    pub(in crate::jit) fn forwarded(&self, value: Value) -> bool {
        self.forwarded.contains(&value)
    }

    /// Whether a store to this value's slot may find an earlier value there.
    pub(in crate::jit) fn shared(&self, value: Value) -> bool {
        self.slot[value].is_some_and(|slot| self.shared[slot as usize])
    }

    pub(in crate::jit) fn assigned(&self) -> impl Iterator<Item = (Value, u32)> + '_ {
        self.slot
            .iter()
            .filter_map(|(value, slot)| slot.map(|slot| (value, slot)))
    }

    fn take(&mut self, in_use: &mut Vec<bool>, value: Value, preferences: &[Value]) {
        let preferred = preferences
            .iter()
            .filter_map(|&peer| self.of(peer))
            .map(|slot| slot as usize)
            .find(|&slot| !in_use[slot]);
        let index = match preferred.or_else(|| in_use.iter().position(|used| !used)) {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(in crate::jit) enum Holder {
    #[default]
    None,
    Caller,
    Borrowed,
    Temporary,
    Slot,
}

#[derive(Clone)]
pub(in crate::jit) struct Plan {
    pub(in crate::jit) holders: SecondaryMap<Value, Holder>,
    pub(in crate::jit) slots: Slots,
    pub(in crate::jit) results: SecondaryMap<Inst, ResultReference>,
}

impl Plan {
    pub(in crate::jit) fn holder(&self, value: Value) -> Holder {
        self.holders[value]
    }
}

pub(super) fn plan(graph: &mut Graph, facts: &mut Facts) -> Plan {
    let cfg = &facts.cfg;
    let (slots, live_out) = assign(
        graph,
        cfg,
        |value| {
            facts.of(value).intersects(HEAP)
                && !graph.borrowed.contains(&value)
                && !graph.caller_rooted(value)
        },
        |value| match graph.defining_op(value) {
            Some(Op::Get(_, name)) => name != "length",
            Some(Op::Read(binding)) => !graph.bindings[*binding].captured,
            Some(
                Op::TypedIndex(..)
                | Op::HelperCall {
                    helper: Helper::IndexGet | Helper::IndexGetNumeric,
                    ..
                }
                | Op::ArrayLoad(..)
                | Op::GlobalRead(_)
                | Op::GlobalReadOptional(_),
            ) => true,
            Some(op) => super::contracts::fresh_reference(op),
            _ => false,
        },
    );
    let mut holders = SecondaryMap::new();
    for value in graph.values.keys() {
        holders[value] = if !facts.of(value).intersects(HEAP) {
            Holder::None
        } else if graph.borrowed.contains(&value) {
            Holder::Borrowed
        } else if graph.caller_rooted(value) {
            Holder::Caller
        } else if slots.of(value).is_some() {
            Holder::Slot
        } else {
            Holder::Temporary
        };
    }
    let mut results = SecondaryMap::new();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let slot = graph.insts[inst]
                .result()
                .is_some_and(|value| holders[value] == Holder::Slot);
            results[inst] = contracts::result_reference(graph, facts, inst, slot);
        }
    }
    let mut releases = releases(graph, facts, cfg);
    for values in releases.values_mut() {
        values.retain(|&value| holders[value] == Holder::Temporary);
    }
    let rpo = cfg.rpo.clone();
    insert(graph, &rpo, &holders, &slots, &releases, &results);
    if transfer(graph, &rpo, &holders, &slots, &live_out) {
        facts.cfg = Cfg::new(graph);
    }
    for &block in &rpo {
        let insts = std::mem::take(&mut graph.blocks[block].insts);
        graph.blocks[block].insts = insts
            .into_iter()
            .filter(|&inst| !matches!(graph.insts[inst].op, Op::KeepAlive(_)))
            .collect();
    }
    Plan {
        holders,
        slots,
        results,
    }
}

/// Checks the placement of the ownership operations: temporaries stay in their block and are
/// released at most once after their last use there, every slot value is stored exactly once,
/// moved values are dead, and every edge into a slot param hands it a reference.
pub(super) fn verify(graph: &Graph, facts: &Facts, plan: &Plan) -> super::Result<()> {
    let holder = |value: Value| plan.holders[value];
    let fail = |message: String| Err(format!("ownership: {message}"));
    let mut stores = FxHashMap::<Value, usize>::default();
    let mut defined = FxHashMap::<Value, Block>::default();
    for &block in &facts.cfg.rpo {
        for &param in &graph.blocks[block].params {
            defined.insert(param, block);
        }
        for &inst in &graph.blocks[block].insts {
            if let Some(result) = graph.insts[inst].result() {
                defined.insert(result, block);
            }
        }
    }
    for &block in &facts.cfg.rpo {
        let data = &graph.blocks[block];
        let terminator = data.terminator.as_ref().unwrap();
        let mut released = FxHashSet::default();
        let mut cleared = FxHashSet::default();
        for &inst in &data.insts {
            let op = &graph.insts[inst].op;
            for value in op.operands_iter() {
                if released.contains(&value) || cleared.contains(&value) {
                    return fail(format!(
                        "{value} used by {inst} after its reference is gone"
                    ));
                }
                if holder(value) == Holder::Temporary && defined.get(&value) != Some(&block) {
                    return fail(format!(
                        "temporary {value} used outside its block by {inst}"
                    ));
                }
            }
            match *op {
                Op::Release(value) => {
                    if holder(value) != Holder::Temporary || !released.insert(value) {
                        return fail(format!("{inst} releases {value}"));
                    }
                }
                Op::SlotStore(value) => {
                    if holder(value) != Holder::Slot {
                        return fail(format!("{inst} stores {value}, which has no slot"));
                    }
                    *stores.entry(value).or_default() += 1;
                }
                Op::SlotClear(value) => {
                    if holder(value) != Holder::Slot || !cleared.insert(value) {
                        return fail(format!("{inst} clears {value}"));
                    }
                }
                Op::ParamStore(param, _) => {
                    if holder(param) != Holder::Slot || !plan.slots.forwarded(param) {
                        return fail(format!("{inst} stores into {param} on an edge"));
                    }
                }
                Op::Retain(value) => {
                    let edge = terminator
                        .kind
                        .targets()
                        .any(|target| target.args.contains(&value));
                    if !matches!(holder(value), Holder::Temporary | Holder::Slot) && !edge {
                        return fail(format!("{inst} retains {value} for no holder"));
                    }
                }
                _ => {}
            }
        }
        for value in terminator.kind.operands_iter() {
            if released.contains(&value) {
                return fail(format!(
                    "{value} used by the terminator of {block} after release"
                ));
            }
            if holder(value) == Holder::Temporary {
                return fail(format!("temporary {value} escapes {block}"));
            }
        }
        for target in terminator.kind.targets() {
            for (&arg, &param) in target.args.iter().zip(&graph.blocks[target.block].params) {
                if holder(param) != Holder::Slot {
                    continue;
                }
                let arrives = plan.slots.of(arg) == plan.slots.of(param);
                if arrives && plan.slots.forwarded(param) {
                    continue;
                }
                let handed = data.insts.iter().any(|&inst| {
                    matches!(graph.insts[inst].op, Op::Retain(value) | Op::SlotClear(value) if value == arg)
                });
                let stored = !plan.slots.forwarded(param)
                    || data.insts.iter().any(|&inst| {
                        matches!(graph.insts[inst].op, Op::ParamStore(to, value) if to == param && value == arg)
                    });
                if !handed || !stored {
                    return fail(format!(
                        "edge {block} -> {} does not hand {arg} to {param}",
                        target.block
                    ));
                }
            }
        }
    }
    for (&value, &block) in &defined {
        if holder(value) != Holder::Slot {
            continue;
        }
        let expected = match graph.defining_op(value) {
            None if plan.slots.forwarded(value) => 0,
            _ => 1,
        };
        if stores.get(&value).copied().unwrap_or(0) != expected {
            return fail(format!("{value} in {block} is not stored exactly once"));
        }
    }
    Ok(())
}

/// Assigns slots to owned values by liveness, in reverse postorder so that every live-in value
/// already has one. Params always get a slot, as do results that outlive their block. Other
/// results stay owned by their temporary until the block's clear, unless `prefers` says their
/// operation can skip the temporary when it has a slot.
fn assign(
    graph: &Graph,
    cfg: &Cfg,
    owned: impl Fn(Value) -> bool,
    prefers: impl Fn(Value) -> bool,
) -> (Slots, SecondaryMap<Block, FxHashSet<Value>>) {
    let mut receivers = FxHashMap::default();
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
    let anchored = |value: Value, live: &mut FxHashSet<Value>| {
        let mut pending = vec![value];
        let mut visited = FxHashSet::default();
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
        let mut uses = FxHashSet::default();
        if let Some(terminator) = &graph.blocks[block].terminator {
            for value in terminator.kind.operands_iter() {
                anchored(value, &mut uses);
            }
        }
        uses
    };
    let mut live_in: SecondaryMap<Block, FxHashSet<Value>> = SecondaryMap::new();
    let mut live_out: SecondaryMap<Block, FxHashSet<Value>> = SecondaryMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for &block in cfg.rpo.iter().rev() {
            let data = &graph.blocks[block];
            let mut live: FxHashSet<Value> = graph
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

    let mut affinities = SecondaryMap::<Value, Vec<Value>>::new();
    for &block in &cfg.rpo {
        for target in graph.blocks[block]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .targets()
        {
            for (&arg, &param) in target.args.iter().zip(&graph.blocks[target.block].params) {
                affinities[arg].push(param);
                affinities[param].push(arg);
            }
        }
    }
    let mut slots = Slots::none();
    for &block in &cfg.rpo {
        let data = &graph.blocks[block];
        let mut live = live_out[block].clone();
        live.extend(terminator_uses(block));
        let mut dying = vec![Vec::new(); data.insts.len()];
        let mut used = FxHashSet::default();
        for (index, &inst) in data.insts.iter().enumerate().rev() {
            let data = &graph.insts[inst];
            if let Some(result) = data.result()
                && live.remove(&result)
            {
                used.insert(result);
            }
            let mut uses = FxHashSet::default();
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
                slots.take(&mut in_use, param, &affinities[param]);
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
                slots.take(&mut in_use, result, &affinities[result]);
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
            let incoming: Vec<_> = cfg.preds[block]
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
                .collect();
            let arrives = |target: &Target| slots.of(target.args[index]) == Some(slot);
            let stores =
                |target: &Target| target.args.iter().all(|&arg| slots.of(arg) != Some(slot));
            if incoming.iter().any(|target| arrives(target))
                && incoming
                    .iter()
                    .all(|target| arrives(target) || stores(target))
            {
                slots.forwarded.insert(param);
            }
        }
    }
    (slots, live_out)
}

/// For each instruction, the results whose references can be released right after it: their
/// last use within their own block, counting the uses of every borrowed value that may alias them.
/// Results used by a terminator or another block keep their references until the block's end.
fn insert(
    graph: &mut Graph,
    rpo: &[Block],
    holders: &SecondaryMap<Value, Holder>,
    slots: &Slots,
    releases: &FxHashMap<Inst, Vec<Value>>,
    results: &SecondaryMap<Inst, ResultReference>,
) {
    for &block in rpo {
        let insts = std::mem::take(&mut graph.blocks[block].insts);
        let mut placed = Vec::with_capacity(insts.len());
        let entering: Vec<Value> = graph.blocks[block]
            .params
            .iter()
            .copied()
            .filter(|&param| holders[param] == Holder::Slot && !slots.forwarded(param))
            .collect();
        let span = graph.span;
        for param in entering {
            placed.push(append(graph, block, span, None, Op::SlotStore(param)));
        }
        for inst in insts {
            placed.push(inst);
            let data = &graph.insts[inst];
            let (span, source_name) = (data.span, data.source_name);
            let mut ops = Vec::new();
            if let Some(result) = data.result() {
                match holders[result] {
                    Holder::Slot => {
                        if !matches!(results[inst], ResultReference::Slot | ResultReference::Move) {
                            ops.push(Op::Retain(result));
                        }
                        ops.push(Op::SlotStore(result));
                    }
                    Holder::Temporary if results[inst] == ResultReference::Borrowed => {
                        ops.push(Op::Retain(result));
                    }
                    _ => {}
                }
            }
            for &value in releases.get(&inst).into_iter().flatten() {
                ops.push(Op::Release(value));
            }
            for op in ops {
                placed.push(append(graph, block, span, source_name, op));
            }
        }
        graph.blocks[block].insts = placed;
    }
}

/// Hands each slot param its reference on the incoming edges: an argument that dies at the jump
/// moves its reference out of its own slot, any other argument is retained. Branch edges that
/// need either get their own block.
fn transfer(
    graph: &mut Graph,
    rpo: &[Block],
    holders: &SecondaryMap<Value, Holder>,
    slots: &Slots,
    live_out: &SecondaryMap<Block, FxHashSet<Value>>,
) -> bool {
    let mut split = false;
    for &block in rpo {
        let terminator = graph.blocks[block].terminator.clone().unwrap();
        let mut passed = FxHashMap::<Value, usize>::default();
        for target in terminator.kind.targets() {
            for &arg in &target.args {
                *passed.entry(arg).or_default() += 1;
            }
        }
        let ops = |graph: &Graph, target: &Target| -> Vec<Op> {
            target
                .args
                .iter()
                .zip(&graph.blocks[target.block].params)
                .filter(|&(_, &param)| holders[param] == Holder::Slot)
                .flat_map(|(&arg, &param)| {
                    let arrives = slots.of(arg) == slots.of(param);
                    if arrives && slots.forwarded(param) {
                        return Vec::new();
                    }
                    let moves = holders[arg] == Holder::Slot
                        && !arrives
                        && !live_out[block].contains(&arg)
                        && passed[&arg] == 1;
                    let store = slots.forwarded(param).then_some(Op::ParamStore(param, arg));
                    if moves {
                        store.into_iter().chain([Op::SlotClear(arg)]).collect()
                    } else {
                        [Op::Retain(arg)].into_iter().chain(store).collect()
                    }
                })
                .collect()
        };
        let (span, source_name) = (terminator.span, terminator.source_name);
        match &terminator.kind {
            TerminatorKind::Jump(target) | TerminatorKind::CheckException(target, _) => {
                for op in ops(graph, target) {
                    let inst = append(graph, block, span, source_name, op);
                    graph.blocks[block].insts.push(inst);
                }
            }
            TerminatorKind::Branch(condition, yes, no) => {
                let mut edge = |graph: &mut Graph, target: &Target| {
                    let ops = ops(graph, target);
                    if ops.is_empty() {
                        return target.clone();
                    }
                    split = true;
                    let edge = graph.blocks.push(BlockData::default());
                    if graph.cold.contains(&block) || graph.cold.contains(&target.block) {
                        graph.cold.insert(edge);
                    }
                    for op in ops {
                        let inst = append(graph, edge, span, source_name, op);
                        graph.blocks[edge].insts.push(inst);
                    }
                    graph.blocks[edge].terminator = Some(Terminator {
                        kind: TerminatorKind::Jump(target.clone()),
                        ..terminator.clone()
                    });
                    Target::new(edge)
                };
                let yes = edge(graph, yes);
                let no = edge(graph, no);
                graph.blocks[block].terminator.as_mut().unwrap().kind =
                    TerminatorKind::Branch(*condition, yes, no);
            }
            _ => {}
        }
    }
    split
}

fn append(
    graph: &mut Graph,
    block: Block,
    span: Span,
    source_name: Option<(usize, usize)>,
    op: Op,
) -> Inst {
    graph.insts.push(InstData {
        source_name,
        emission_location: None,
        span,
        block,
        results: Vec::new(),
        op,
    })
}

fn releases(graph: &Graph, facts: &Facts, cfg: &Cfg) -> FxHashMap<Inst, Vec<Value>> {
    let mut aliases: FxHashMap<Value, Vec<Value>> = FxHashMap::default();
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
    let anchors: FxHashMap<Value, Vec<Value>> = aliases
        .keys()
        .map(|&value| {
            let mut pending = vec![value];
            let mut visited = FxHashSet::default();
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
    let mut defined: FxHashMap<Value, (Block, usize, Inst)> = FxHashMap::default();
    for &block in &cfg.rpo {
        for (index, &inst) in graph.blocks[block].insts.iter().enumerate() {
            if let Some(result) = graph.insts[inst].result() {
                defined.insert(result, (block, index, inst));
            }
        }
    }
    let mut last: FxHashMap<Value, Option<usize>> = FxHashMap::default();
    for &block in &cfg.rpo {
        let data = &graph.blocks[block];
        let escape = |value: Value, last: &mut FxHashMap<Value, Option<usize>>| {
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
    let mut releases: FxHashMap<Inst, Vec<Value>> = FxHashMap::default();
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
fn anchored<'a>(anchors: &'a FxHashMap<Value, Vec<Value>>, value: &'a Value) -> &'a [Value] {
    match anchors.get(value) {
        Some(values) => values,
        None => std::slice::from_ref(value),
    }
}
