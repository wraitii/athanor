use crate::jit::profile::{Observation, ReadKind};
use crate::{FxHashMap, FxHashSet};

use super::super::super::available::LoadKey;
use super::super::super::guards::{Fact, edge_of, implies};
use super::*;

fn retarget(graph: &mut Graph, blocks: &FxHashSet<Block>, from: Value, to: Value) {
    let aliases = FxHashMap::from_iter([(from, to)]);
    for &block in blocks {
        for inst in graph.blocks[block].insts.clone() {
            if !matches!(
                graph.insts[inst].op,
                Op::IsNumber(_)
                    | Op::AsNumber(_)
                    | Op::IsInteger(_)
                    | Op::AsInteger(_)
                    | Op::AsObject(_)
            ) {
                for value in graph.insts[inst].op.operands_mut() {
                    *value = resolve(&aliases, *value);
                }
            }
        }
        if let Some(terminator) = &mut graph.blocks[block].terminator {
            for value in terminator.kind.operands_mut() {
                *value = resolve(&aliases, *value);
            }
        }
    }
}

/// Guarded assumptions and exits for a versioned copy.
#[derive(Default)]
pub(in crate::jit::semantic) struct Speculation<'a> {
    pub(in crate::jit::semantic) locals: &'a [Inst],
    pub(in crate::jit::semantic) fallback: Option<usize>,
    pub(in crate::jit::semantic) continuations: bool,
    pub(in crate::jit::semantic) reads: &'a [Inst],
    pub(in crate::jit::semantic) shaped_reads: &'a [(Inst, u32, u32)],
    pub(in crate::jit::semantic) guards: &'a [Block],
    pub(in crate::jit::semantic) exits: &'a [Block],
    pub(in crate::jit::semantic) exits_true: &'a [Block],
    pub(in crate::jit::semantic) typed_reads: &'a [(Inst, TypedRead)],
    pub(in crate::jit::semantic) assumes: &'a [(Value, Fact)],
    pub(in crate::jit::semantic) code_guards: &'a [(String, cranelift_module::FuncId)],
}

struct NativeCheck {
    callee: Value,
    global: String,
    property: String,
    native: Native,
    captured: Option<Value>,
    object: Option<Value>,
}

#[derive(PartialEq, Eq, Hash)]
enum EntryGuard {
    Fact(Value, Fact),
    Value(Value),
}

impl EntryGuard {
    fn of(op: &Op) -> Option<Self> {
        let (value, fact) = implies(op)?;
        Some(Self::Fact(value, fact))
    }

    fn tested_by(&self, known: &FxHashSet<Self>) -> bool {
        known.contains(self)
            || matches!(self, Self::Fact(value, fact) if known.iter().any(|other| {
                matches!(other, Self::Fact(known_value, known_fact)
                    if known_value == value && known_fact.implies(*fact))
            }))
    }

    fn at_entry(graph: &Graph, entry: Block) -> FxHashSet<Self> {
        // These predicates describe SSA values, so their proofs survive writes and reentry.
        let cfg = Cfg::new(graph);
        cfg.rpo
            .iter()
            .filter_map(|&block| {
                let edge = edge_of(graph, &graph.blocks[block].terminator.as_ref()?.kind)?;
                if cfg.preds[edge.yes].len() != 1 || !cfg.dominates(edge.yes, entry) {
                    return None;
                }
                Some(Self::Fact(edge.value, edge.fact))
            })
            .collect()
    }
}

pub(in crate::jit::semantic) struct Version {
    pub(in crate::jit::semantic) body: FxHashSet<Block>,
    pub(in crate::jit::semantic) notes: Vec<String>,
    pub(in crate::jit::semantic) blocks: FxHashMap<Block, Block>,
    pub(in crate::jit::semantic) insts: FxHashMap<Inst, Inst>,
    pub(in crate::jit::semantic) values: FxHashMap<Value, Value>,
    pub(in crate::jit::semantic) aliases: FxHashMap<Value, Value>,
}

pub(in crate::jit::semantic) struct Canonical<'a> {
    pub(in crate::jit::semantic) graph: &'a Graph,
    pub(in crate::jit::semantic) blocks: FxHashMap<Block, Block>,
    pub(in crate::jit::semantic) values: FxHashMap<Value, Value>,
    pieces: std::rc::Rc<std::cell::RefCell<super::super::super::continuation::Pieces>>,
}

impl<'a> Canonical<'a> {
    pub(in crate::jit::semantic) fn identity(graph: &'a Graph) -> Self {
        Self {
            graph,
            blocks: graph.blocks.keys().map(|block| (block, block)).collect(),
            values: graph.values.keys().map(|value| (value, value)).collect(),
            pieces: Default::default(),
        }
    }

    pub(in crate::jit::semantic) fn copied(&self, version: &Version) -> Self {
        Self {
            graph: self.graph,
            pieces: self.pieces.clone(),
            blocks: self
                .blocks
                .iter()
                .map(|(&original, &current)| {
                    (
                        original,
                        version.blocks.get(&current).copied().unwrap_or(current),
                    )
                })
                .collect(),
            values: self
                .values
                .iter()
                .map(|(&original, &current)| {
                    (
                        original,
                        version.values.get(&current).copied().unwrap_or(current),
                    )
                })
                .collect(),
        }
    }
}

pub(in crate::jit::semantic) fn apply_version(
    graph: &mut Graph,
    region: &[Block],
    entry: Option<(Block, Block)>,
    speculation: &Speculation<'_>,
    canonical: Option<&Canonical<'_>>,
) -> Version {
    let Speculation {
        fallback,
        continuations,
        guards,
        exits,
        exits_true,
        assumes,
        ..
    } = *speculation;
    let region_blocks: FxHashSet<Block> = region.iter().copied().collect();
    let requirements: Vec<NativeCheck> = assumes
        .iter()
        .filter_map(|&(callee, fact)| {
            let Fact::Native(native) = fact else {
                return None;
            };
            let Some(Op::Get(object, property)) = graph.defining_op(callee) else {
                unreachable!()
            };
            let Some(Op::GlobalRead(global)) = graph.defining_op(*object) else {
                unreachable!()
            };
            Some(NativeCheck {
                callee,
                global: global.clone(),
                property: property.clone(),
                native,
                captured: (!region_blocks.contains(&graph.def_block(callee))).then_some(callee),
                object: (!region_blocks.contains(&graph.def_block(*object))).then_some(*object),
            })
        })
        .collect();
    let local_generic = canonical.is_none().then(|| graph.clone());
    let local_canonical = local_generic.as_ref().map(Canonical::identity);
    let canonical = canonical.or(local_canonical.as_ref()).unwrap();
    let mut notes = Vec::new();
    let hints = super::super::observations::numeric_hints(graph);
    let head = entry.map_or(graph.entry, |(_, header)| header);
    let tested = EntryGuard::at_entry(graph, entry.map_or(graph.entry, |(preheader, _)| preheader));
    assert!(fallback.is_none() || entry.is_none());
    let reuse = speculation.locals.is_empty()
        && fallback.is_some()
        && exits.iter().all(|block| guards.contains(block))
        && exits_true.is_empty();
    let mut copy = duplicate(graph, region, reuse, speculation);
    let Entry {
        start,
        previous,
        aliases,
    } = entry_checks(
        graph,
        region,
        entry,
        speculation,
        &requirements,
        &hints,
        head,
        reuse,
        tested,
        &mut copy,
    );
    local_checks(graph, speculation, canonical, &copy, &mut notes);
    if !reuse && continuations {
        wire_failures(graph, region, speculation, canonical, &copy, &mut notes);
    }
    let Copy {
        blocks,
        values,
        insts,
        copies,
    } = copy;

    let mut callees = Vec::new();
    for &block in region {
        for &inst in &graph.blocks[block].insts.clone() {
            let Op::Call { callee, .. } = graph.insts[inst].op else {
                continue;
            };
            let Some(requirement) = requirements.iter().find(|r| r.callee == callee) else {
                continue;
            };
            let copy = insts[&inst];
            if let Op::Call { callee, args, .. } = &graph.insts[copy].op {
                callees.push(*callee);
                graph.insts[copy].op = Op::CallNative(requirement.native, args.clone());
            }
        }
    }
    let inside: FxHashSet<Block> = region.iter().chain(&copies).copied().collect();
    let mut outside = FxHashSet::default();
    for (block, data) in graph.blocks.iter() {
        if inside.contains(&block) {
            continue;
        }
        for &inst in &data.insts {
            outside.extend(graph.insts[inst].op.operands_iter());
        }
        if let Some(terminator) = &data.terminator {
            outside.extend(terminator.kind.operands_iter());
        }
    }
    let pinned: FxHashSet<Value> = values
        .iter()
        .filter(|(original, _)| outside.contains(*original))
        .map(|(_, &copy)| copy)
        .collect();
    let mut uses = graph.use_counts();
    while let Some(value) = callees.pop() {
        if uses[value] != 0 || pinned.contains(&value) {
            continue;
        }
        let ValueDef::Result(inst, _) = graph.values[value] else {
            continue;
        };
        match graph.insts[inst].op {
            Op::Get(object, _) => {
                uses[object] -= 1;
                callees.push(object);
            }
            Op::GlobalRead(_) => {}
            _ => continue,
        }
        uses[value] = u32::MAX;
        graph.remove_inst(inst);
    }
    if !reuse {
        let definitions: FxHashMap<Value, Vec<Value>> = values
            .iter()
            .map(|(&original, &copy)| {
                let live = uses[copy] != u32::MAX;
                (original, if live { vec![copy] } else { Vec::new() })
            })
            .collect();
        repair(graph, &definitions);
    }
    // Preserve the original entry during repair so generic loop updates merge with side exits.
    if !reuse && let Some(previous) = previous {
        for (block, data) in graph.blocks.iter_mut() {
            if block < start {
                continue;
            }
            if let Some(terminator) = &mut data.terminator {
                for target in terminator.kind.targets_mut() {
                    if target.block == head {
                        *target = previous.clone();
                    }
                }
            }
        }
    }
    consume_assumes(graph, region, entry, assumes);
    // Disconnected fallbacks must not contribute writes to the closed copy.
    let reachable = Cfg::new(graph)
        .rpo
        .into_iter()
        .filter(|block| copies.contains(block))
        .collect();
    Version {
        notes,
        body: reachable,
        blocks,
        insts,
        values,
        aliases,
    }
}

struct Copy {
    blocks: FxHashMap<Block, Block>,
    values: FxHashMap<Value, Value>,
    insts: FxHashMap<Inst, Inst>,
    copies: FxHashSet<Block>,
}

struct Entry {
    start: Block,
    previous: Option<Target>,
    aliases: FxHashMap<Value, Value>,
}

fn duplicate(
    graph: &mut Graph,
    region: &[Block],
    reuse: bool,
    speculation: &Speculation<'_>,
) -> Copy {
    let Speculation {
        guards,
        exits,
        exits_true,
        typed_reads,
        ..
    } = *speculation;
    let mut blocks = FxHashMap::default();
    let mut values = FxHashMap::default();
    let mut insts = FxHashMap::default();
    for &block in region {
        let copy = if reuse {
            block
        } else {
            graph.blocks.push(BlockData::default())
        };
        blocks.insert(block, copy);
        for param in graph.blocks[block].params.clone() {
            let mapped = if reuse { param } else { graph.add_param(copy) };
            graph.reprs[mapped] = graph.reprs[param];
            values.insert(param, mapped);
        }
    }
    for &block in region {
        for inst in graph.blocks[block].insts.clone() {
            let data = graph.insts[inst].clone();
            let mut op = data.op.clone();
            if let Some((_, read)) = typed_reads.iter().find(|(original, _)| *original == inst)
                && let Op::GetIndex(object, key) | Op::TypedIndex(object, key, _) = op
            {
                op = Op::TypedIndex(object, key, *read);
            }
            for value in op.operands_mut() {
                *value = values.get(value).copied().unwrap_or(*value);
            }
            let (copy, results) = if reuse {
                graph.insts[inst].op = op;
                (inst, data.results.clone())
            } else {
                graph.append_results(blocks[&block], data.span, op, data.results.len())
            };
            graph.insts[copy].source_name = data.source_name;
            graph.insts[copy].emission_location = data.emission_location;
            for (&old, &new) in data.results.iter().zip(&results) {
                graph.reprs[new] = graph.reprs[old];
                values.insert(old, new);
            }
            insts.insert(inst, copy);
            if let Some(mut read) = graph.property_accesses.get(&inst).cloned() {
                read.cache = read
                    .cache
                    .map(|cache| values.get(&cache).copied().unwrap_or(cache));
                graph.property_accesses.insert(copy, read);
            }
        }
        let mut terminator = graph.blocks[block].terminator.clone().unwrap();
        for value in terminator.kind.operands_mut() {
            *value = values.get(value).copied().unwrap_or(*value);
        }
        for target in terminator.kind.targets_mut() {
            if let Some(&copy) = blocks.get(&target.block) {
                target.block = copy;
            }
        }
        graph.blocks[blocks[&block]].terminator = Some(terminator);
    }
    let copies: FxHashSet<Block> = blocks.values().copied().collect();
    // A varying guard leaves the copy before reentry; SSA repair carries its current values.
    for &block in exits {
        if guards.contains(&block) {
            continue;
        }
        let fallback = cold_exit(graph, block).unwrap();
        let terminator = graph.blocks[blocks[&block]].terminator.as_mut().unwrap();
        let TerminatorKind::Branch(_, _, otherwise) = &mut terminator.kind else {
            unreachable!();
        };
        otherwise.block = fallback;
    }
    for &block in exits_true {
        let Some(Terminator {
            kind: TerminatorKind::Branch(_, fallback, _),
            ..
        }) = &graph.blocks[block].terminator
        else {
            unreachable!()
        };
        let fallback = fallback.block;
        let Some(Terminator {
            kind: TerminatorKind::Branch(_, yes, _),
            ..
        }) = &mut graph.blocks[blocks[&block]].terminator
        else {
            unreachable!()
        };
        yes.block = fallback;
    }
    for &block in region {
        if graph.cold.contains(&block) {
            graph.cold.insert(blocks[&block]);
        }
    }
    if !reuse {
        graph.cold.extend(region.iter().copied());
    }
    Copy {
        blocks,
        values,
        insts,
        copies,
    }
}

fn entry_checks(
    graph: &mut Graph,
    region: &[Block],
    entry: Option<(Block, Block)>,
    speculation: &Speculation<'_>,
    requirements: &[NativeCheck],
    hints: &FxHashMap<Value, crate::jit::profile::ReadType>,
    head: Block,
    reuse: bool,
    mut tested: FxHashSet<EntryGuard>,
    copy: &mut Copy,
) -> Entry {
    let Speculation {
        fallback,
        reads,
        guards,
        assumes,
        code_guards,
        ..
    } = *speculation;
    let Copy {
        blocks,
        values,
        insts,
        copies,
    } = copy;
    let span = graph.span;
    let mut args = match entry {
        Some((preheader, _)) => match &graph.blocks[preheader].terminator {
            Some(Terminator {
                kind: TerminatorKind::Jump(target),
                ..
            }) => target.args.clone(),
            _ => unreachable!(),
        },
        None => Vec::new(),
    };
    let previous = fallback.map(|address| {
        let block = graph.blocks.push(BlockData {
            terminator: Some(Terminator {
                source_name: None,
                emission_location: None,
                span,
                kind: TerminatorKind::Fallback(address),
            }),
            ..Default::default()
        });
        graph.cold.insert(block);
        Target::new(block)
    });
    let original = if reuse {
        previous.clone().unwrap()
    } else {
        Target {
            block: head,
            args: args.clone(),
        }
    };
    let start = graph.blocks.push(BlockData::default());
    let mut current = start;
    let mut guarded_probes = FxHashMap::default();
    for (name, code) in code_guards {
        let global = graph.append_value(current, span, Op::ProbeGlobal(name.clone()));
        guarded_probes.insert(name.clone(), global);
        let check = graph.append_value(current, span, Op::IsCode(global, *code));
        current = continue_if(graph, current, check, &original);
    }
    if requirements
        .iter()
        .any(|r| r.captured.is_none() && r.object.is_none())
    {
        let guard = graph.guards;
        graph.guards += 1;
        let probes = graph.blocks.push(BlockData::default());
        let commit = graph.blocks.push(BlockData::default());
        graph.cold.extend([probes, commit]);
        let validated = graph.blocks.push(BlockData::default());
        let valid = graph.append_value(current, span, Op::GuardValid(guard));
        graph.terminate(
            current,
            span,
            TerminatorKind::Branch(valid, Target::new(validated), Target::new(probes)),
        );
        let mut globals: FxHashMap<String, Value> = FxHashMap::default();
        let mut methods = FxHashSet::default();
        let mut probed = Vec::new();
        for requirement in requirements
            .iter()
            .filter(|r| r.captured.is_none() && r.object.is_none())
        {
            let global = match globals.get(&requirement.global) {
                Some(&value) => value,
                None => {
                    let op = Op::ProbeGlobal(requirement.global.clone());
                    let value = graph.append_value(probes, span, op);
                    globals.insert(requirement.global.clone(), value);
                    value
                }
            };
            if !methods.insert((requirement.global.clone(), requirement.property.clone())) {
                continue;
            }
            let op = Op::ProbeProperty(global, requirement.property.clone());
            let method = graph.append_value(probes, span, op);
            let op = Op::IsNative(method, requirement.native);
            probed.push(graph.append_value(probes, span, op));
        }
        let mut chain = probes;
        for (index, &check) in probed.iter().enumerate() {
            let next = if index + 1 == probed.len() {
                commit
            } else {
                let next = graph.blocks.push(BlockData::default());
                graph.cold.insert(next);
                next
            };
            graph.terminate(
                chain,
                span,
                TerminatorKind::Branch(check, Target::new(next), original.clone()),
            );
            chain = next;
        }
        graph.append(commit, span, Op::GuardCommit(guard), false);
        graph.terminate(commit, span, TerminatorKind::Jump(Target::new(validated)));
        current = validated;
    }
    for requirement in requirements
        .iter()
        .filter(|r| r.captured.is_some() || r.object.is_some())
    {
        let method = match requirement.captured {
            Some(value) => value,
            None => graph.append_value(
                current,
                span,
                Op::ProbeProperty(requirement.object.unwrap(), requirement.property.clone()),
            ),
        };
        let check = graph.append_value(current, span, Op::IsNative(method, requirement.native));
        current = continue_if(graph, current, check, &original);
    }
    let mut probes = FxHashMap::default();
    let mut replacements = FxHashMap::default();
    let mut available: FxHashMap<LoadKey, Value> = guarded_probes
        .into_iter()
        .map(|(name, value)| (LoadKey::Global(name), value))
        .collect();
    for &inst in reads {
        let data = graph.insts[inst].clone();
        let mut op = match data.op.clone() {
            Op::Get(object, name)
            | Op::ShapedGet(object, name, _)
            | Op::ProbeProperty(object, name) => {
                match speculation
                    .shaped_reads
                    .iter()
                    .find(|&&(read, _, _)| read == inst)
                {
                    Some(&(_, shape, index)) => Op::ProbeShapeProperty(object, name, shape, index),
                    None => Op::ProbeProperty(object, name),
                }
            }
            Op::ProbeShapeProperty(..) => data.op.clone(),
            Op::GetIndex(object, key) | Op::ArrayGet(object, key) => Op::ProbeIndex(object, key),
            Op::GlobalRead(name) => Op::ProbeGlobal(name),
            Op::Argument(index) => Op::Argument(index),
            Op::This => Op::This,
            Op::PropertyCache(..) => data.op.clone(),
            _ => unreachable!(),
        };
        for value in op.operands_mut() {
            if let Some(&probe) = probes.get(value) {
                *value = probe;
            }
        }
        let key = match &op {
            Op::ProbeGlobal(name) => LoadKey::Global(name.clone()),
            Op::ProbeProperty(object, name) | Op::ProbeShapeProperty(object, name, ..) => {
                LoadKey::Property(*object, name.clone())
            }
            Op::ProbeIndex(object, key) => LoadKey::Index(*object, *key),
            Op::Argument(index) => LoadKey::Argument(*index),
            Op::This => LoadKey::This,
            Op::PropertyCache(site, _) => LoadKey::PropertyCache(*site),
            _ => unreachable!(),
        };
        let value = match available.get(&key) {
            Some(&value) => value,
            None => {
                let input = matches!(op, Op::Argument(_) | Op::This | Op::PropertyCache(..));
                let value = graph.append_value(current, data.span, op);
                if !input {
                    let failed = graph.append_value(current, data.span, Op::IsUninitialized(value));
                    let next = graph.blocks.push(BlockData::default());
                    graph.blocks[current].terminator = Some(Terminator {
                        source_name: data.source_name,
                        emission_location: None,
                        span: data.span,
                        kind: TerminatorKind::Branch(failed, original.clone(), Target::new(next)),
                    });
                    current = next;
                }
                available.insert(key, value);
                value
            }
        };
        let result = data.result().unwrap();
        replacements.insert(values[&result], value);
        graph.detach_inst(insts[&inst]);
        values.insert(result, value);
        probes.insert(result, value);
    }
    graph.replace_uses(&replacements);
    let copied: FxHashSet<Block> = region.iter().copied().collect();
    let mut objects: FxHashMap<Value, Value> = FxHashMap::default();
    for &block in guards {
        let Some(Terminator {
            source_name,
            emission_location,
            span,
            kind: TerminatorKind::Branch(condition, _, _),
        }) = graph.blocks[block].terminator.clone()
        else {
            unreachable!();
        };
        let copy = blocks[&block];
        let terminator = graph.blocks[copy].terminator.as_mut().unwrap();
        let TerminatorKind::Branch(_, taken, _) = &terminator.kind else {
            unreachable!();
        };
        terminator.kind = TerminatorKind::Jump(taken.clone());
        let (test, key) = if copied.contains(&graph.def_block(condition)) {
            let mut op = graph.defining_op(condition).unwrap().clone();
            for value in op.operands_mut() {
                if copied.contains(&graph.def_block(*value)) {
                    *value = values[value];
                }
            }
            let key = EntryGuard::of(&op);
            let ValueDef::Result(inst, _) = graph.values[condition] else {
                unreachable!();
            };
            graph.insts[insts[&inst]].op = Op::Constant(Constant::Boolean(true));
            (Some(op), key)
        } else {
            let key = graph
                .defining_op(condition)
                .and_then(EntryGuard::of)
                .unwrap_or(EntryGuard::Value(condition));
            (None, Some(key))
        };
        if let Some(key) = key {
            if key.tested_by(&tested) {
                continue;
            }
            tested.insert(key);
        }
        if entry.is_none()
            && let Some((value, fact)) = test
                .as_ref()
                .or_else(|| graph.defining_op(condition))
                .and_then(implies)
        {
            super::super::super::guards::record_entry(graph, value, fact);
        }
        let otherwise = match &test {
            Some(Op::IsPropertyCached(object, site, _)) if fallback.is_some() => {
                let prime = graph.blocks.push(BlockData::default());
                graph.cold.insert(prime);
                let op = Op::PrimePropertyCache(*object, *site);
                graph.append(prime, span, op, false);
                graph.blocks[prime].terminator = Some(Terminator {
                    source_name,
                    emission_location,
                    span,
                    kind: TerminatorKind::Jump(original.clone()),
                });
                Target::new(prime)
            }
            _ => original.clone(),
        };
        let shaped = match &test {
            Some(Op::IsPropertyCached(object, _, false)) if graph.caller_rooted(*object) => {
                Some(*object)
            }
            Some(Op::IsShape(object, _)) if graph.caller_rooted(*object) => Some(*object),
            _ => None,
        };
        let condition = test.map_or(condition, |op| graph.append_value(current, span, op));
        let next = graph.blocks.push(BlockData::default());
        graph.blocks[current].terminator = Some(Terminator {
            source_name,
            emission_location,
            span,
            kind: TerminatorKind::Branch(condition, Target::new(next), otherwise),
        });
        current = next;
        if let Some(object) = shaped
            && !objects.contains_key(&object)
        {
            let alias = graph.append_value(current, span, Op::AsObject(object));
            objects.insert(object, alias);
            retarget(graph, copies, object, alias);
            for arg in &mut args {
                if *arg == object {
                    *arg = alias;
                }
            }
        }
    }
    for &(value, fact) in assumes {
        let checked = values.get(&value).copied().unwrap_or(value);
        let test = match fact {
            Fact::Indexed => Op::IsIndexed(checked),
            Fact::Shape(shape) => Op::IsShape(checked, shape),
            Fact::Cached(cache, builtins) => Op::IsPropertyCached(
                checked,
                values.get(&cache).copied().unwrap_or(cache),
                builtins,
            ),
            _ => continue,
        };
        let guard = EntryGuard::Fact(checked, fact);
        if guard.tested_by(&tested) {
            continue;
        }
        tested.insert(guard);
        let otherwise = if fallback.is_some() {
            if let Op::IsPropertyCached(object, site, _) = &test {
                let prime = graph.blocks.push(BlockData::default());
                graph.cold.insert(prime);
                graph.append(prime, span, Op::PrimePropertyCache(*object, *site), false);
                graph.terminate(prime, span, TerminatorKind::Jump(original.clone()));
                Target::new(prime)
            } else {
                original.clone()
            }
        } else {
            original.clone()
        };
        let shaped = match &test {
            Op::IsPropertyCached(object, _, builtins) => !builtins && graph.caller_rooted(*object),
            Op::IsShape(object, _) => graph.caller_rooted(*object),
            _ => false,
        };
        if entry.is_none() {
            if let Some((value, fact)) = implies(&test) {
                super::super::super::guards::record_entry(graph, value, fact);
            }
        }
        let condition = graph.append_value(current, span, test);
        let next = graph.blocks.push(BlockData::default());
        graph.terminate(
            current,
            span,
            TerminatorKind::Branch(condition, Target::new(next), otherwise),
        );
        current = next;
        if shaped && !objects.contains_key(&checked) {
            let alias = graph.append_value(current, span, Op::AsObject(checked));
            objects.insert(checked, alias);
            retarget(graph, copies, checked, alias);
            for arg in &mut args {
                if *arg == checked {
                    *arg = alias;
                }
            }
        }
    }
    let numeric = || {
        assumes
            .iter()
            .copied()
            .filter(|(_, fact)| matches!(fact, Fact::Type(_)))
    };
    for (value, fact) in numeric() {
        let checked = values.get(&value).copied().unwrap_or(value);
        let guard = EntryGuard::Fact(checked, fact);
        if guard.tested_by(&tested) {
            continue;
        }
        tested.insert(guard);
        if entry.is_none() {
            super::super::super::guards::record_entry(graph, value, fact);
        }
        let check = if fact == Fact::Type(ValueType::INTEGER) {
            Op::IsInteger(checked)
        } else {
            Op::IsNumber(checked)
        };
        let check = graph.append_value(current, span, check);
        let failure = number_failure(
            graph,
            checked,
            super::super::observations::numeric_site(graph, value),
            original.clone(),
            span,
        );
        current = continue_if(graph, current, check, &failure);
    }
    let mut numbers = FxHashMap::default();
    let mut aliases = FxHashMap::default();
    for (value, fact) in numeric() {
        let checked = values.get(&value).copied().unwrap_or(value);
        let number = *numbers.entry(checked).or_insert_with(|| {
            graph.append_value(
                current,
                span,
                if fact == Fact::Type(ValueType::INTEGER) {
                    Op::AsInteger(checked)
                } else {
                    Op::AsNumber(checked)
                },
            )
        });
        if !hints.contains_key(&value)
            && let Some(site) = super::super::observations::numeric_site(graph, value)
        {
            graph.append(
                current,
                span,
                Op::Observe(number, Observation::Read(ReadKind::Number), site),
                false,
            );
            for &block in copies.iter() {
                graph.blocks[block].insts.retain(|&inst| {
                    !matches!(graph.insts[inst].op,
                        Op::Observe(observed, Observation::Read(ReadKind::Number), origin)
                            if observed == checked && origin == site)
                });
            }
        }
        retarget(graph, copies, checked, number);
        aliases.insert(checked, number);
        for arg in &mut args {
            if *arg == checked {
                *arg = number;
            }
        }
        if values.contains_key(&value) {
            values.insert(value, number);
        }
    }
    let mut mapped_typed = Vec::new();
    for &(object, fact) in assumes {
        let Fact::Typed(element) = fact else {
            continue;
        };
        let guard = (values.get(&object).copied().unwrap_or(object), element);
        let key = EntryGuard::Fact(guard.0, Fact::Typed(element));
        if key.tested_by(&tested) {
            continue;
        }
        tested.insert(key);
        mapped_typed.push(guard);
    }
    let mut bounds: Vec<_> = assumes
        .iter()
        .filter_map(|&(object, fact)| {
            let Fact::Bound(bound) = fact else {
                return None;
            };
            Some((
                values.get(&object).copied().unwrap_or(object),
                values.get(&bound).copied().unwrap_or(bound),
            ))
        })
        .collect();
    bounds.sort();
    bounds.dedup();
    current = typed_guards(graph, current, &original, &mapped_typed, &bounds);
    graph.terminate(
        current,
        span,
        TerminatorKind::Jump(Target {
            block: blocks[&head],
            args,
        }),
    );
    match entry {
        Some((preheader, _)) => {
            graph.terminate(preheader, span, TerminatorKind::Jump(Target::new(start)));
        }
        None => graph.entry = start,
    }

    Entry {
        start,
        previous,
        aliases,
    }
}

fn number_failure(
    graph: &mut Graph,
    value: Value,
    site: Option<crate::jit::profile::Site>,
    fallback: Target,
    span: oxc_span::Span,
) -> Target {
    let Some(site) = site else {
        return fallback;
    };
    let failure = graph.blocks.push(BlockData::default());
    graph.cold.insert(failure);
    graph.append(
        failure,
        span,
        Op::Observe(value, Observation::NumberFailure, site),
        false,
    );
    graph.terminate(failure, span, TerminatorKind::Jump(fallback));
    Target::new(failure)
}

fn local_checks(
    graph: &mut Graph,
    speculation: &Speculation<'_>,
    canonical: &Canonical<'_>,
    copy: &Copy,
    notes: &mut Vec<String>,
) {
    let reverse: FxHashMap<_, _> = canonical
        .blocks
        .iter()
        .map(|(&old, &current)| (current, old))
        .collect();
    let facts = Facts::compute(graph);
    for &inst in speculation.locals {
        let mapped = copy.insts[&inst];
        let data = graph.insts[mapped].clone();
        let Op::Assume(value, fact, origin) = data.op else {
            continue;
        };
        let types = fact.mask().unwrap_or(ValueType::default());
        let known = match graph.defining_op(value) {
            Some(Op::AsInteger(_)) => ValueType::INTEGER,
            Some(Op::AsNumber(_)) => ValueType::NUMBER,
            _ => facts.of(value),
        };
        if (fact.mask().is_some() && known.intersects(types) && known.within(types))
            || super::super::super::guards::holds(graph, &facts.cfg, data.block, value, fact)
        {
            continue;
        }
        let block = data.block;
        let cfg = Cfg::new(graph);
        let [before] = cfg.preds[block][..] else {
            continue;
        };
        let mut fallback = Target::new(graph.insts[inst].block);
        if speculation.continuations {
            let original = reverse[&fallback.block];
            let piece = canonical
                .pieces
                .borrow_mut()
                .resume(canonical.graph, &Target::new(original));
            match piece {
                Ok(piece) => {
                    let inputs: Option<Vec<_>> = piece
                        .inputs
                        .iter()
                        .map(|input| {
                            let current = *canonical.values.get(input)?;
                            let mapped = copy.values.get(&current).copied().unwrap_or(current);
                            value_available(graph, &cfg, mapped, before).then_some(mapped)
                        })
                        .collect();
                    if let Some(inputs) = inputs {
                        let stub = graph.blocks.push(BlockData::default());
                        let result = graph.append_value(
                            stub,
                            data.span,
                            Op::Continuation(piece.body.clone(), inputs),
                        );
                        graph.terminate(stub, data.span, TerminatorKind::Return(result));
                        graph.cold.insert(stub);
                        fallback = Target::new(stub);
                        notes.push(format!(
                            "local assumption {inst}: resumes in {}",
                            piece.body.graph.name
                        ));
                    } else {
                        notes.push(format!(
                            "local assumption {inst}: canonical inputs unavailable"
                        ));
                    }
                }
                Err(reason) => notes.push(format!(
                    "local assumption {inst}: continuation unavailable: {reason}"
                )),
            }
        }
        let integer = types.within(ValueType::INTEGER);
        let (guard, check) = graph.append(
            before,
            data.span,
            fact.test(value).expect("local assumption test"),
            true,
        );
        graph.insts[guard].source_name = data.source_name;
        let fallback = number_failure(graph, value, origin.site, fallback, data.span);
        graph.terminate(
            before,
            data.span,
            TerminatorKind::Branch(check.unwrap(), Target::new(block), fallback),
        );
        if fact.mask().is_none() {
            continue;
        }
        let (unbox, number) = graph.insert(
            block,
            0,
            data.span,
            if integer {
                Op::AsInteger(value)
            } else {
                Op::AsNumber(value)
            },
            true,
        );
        graph.insts[unbox].source_name = data.source_name;
        let cfg = Cfg::new(graph);
        let dominated = copy
            .copies
            .iter()
            .copied()
            .filter(|&candidate| cfg.dominates(block, candidate))
            .collect();
        retarget(graph, &dominated, value, number.unwrap());
    }
}

fn wire_failures(
    graph: &mut Graph,
    region: &[Block],
    speculation: &Speculation<'_>,
    canonical: &Canonical<'_>,
    copy: &Copy,
    notes: &mut Vec<String>,
) {
    let Speculation {
        guards,
        exits,
        exits_true,
        ..
    } = *speculation;
    let generic = canonical.graph;
    let span = graph.span;
    let Copy { blocks, values, .. } = copy;
    let region: FxHashSet<_> = region.iter().copied().collect();
    let reverse: FxHashMap<_, _> = canonical
        .blocks
        .iter()
        .map(|(&old, &current)| (current, old))
        .collect();
    let cfg = Cfg::new(graph);
    let facts = Facts::compute(graph);
    let order: FxHashMap<Block, usize> = cfg
        .rpo
        .iter()
        .enumerate()
        .map(|(index, &block)| (block, index))
        .collect();
    let mut exits: Vec<_> = exits
        .iter()
        .filter(|block| !guards.contains(block))
        .map(|&block| (block, false))
        .chain(exits_true.iter().map(|&block| (block, true)))
        .collect();
    exits.sort_by_key(|(block, _)| order.get(&blocks[block]).copied().unwrap_or(usize::MAX));
    let mut pieces = canonical.pieces.borrow_mut();
    let mut stubs: FxHashMap<(usize, Vec<Value>, Option<Value>), Block> = FxHashMap::default();
    let mut resumed: FxHashMap<Block, Block> = FxHashMap::default();
    for &(block, taken) in &exits {
        if let Some(stub) = shared_resume(graph, &cfg, &facts, blocks[&block], &resumed) {
            let TerminatorKind::Branch(_, yes, no) = &mut graph.blocks[blocks[&block]]
                .terminator
                .as_mut()
                .unwrap()
                .kind
            else {
                continue;
            };
            *if taken { yes } else { no } = Target::new(stub);
            resumed.insert(blocks[&block], stub);
            notes.push(format!("guard at {block}: shares an earlier resume"));
            continue;
        }
        let Some(&original) = reverse.get(&block) else {
            notes.push(format!("guard at {block}: canonical block unavailable"));
            continue;
        };
        let TerminatorKind::Branch(_, yes, no) =
            &generic.blocks[original].terminator.as_ref().unwrap().kind
        else {
            notes.push(format!("guard at {block}: canonical branch unavailable"));
            continue;
        };
        let piece = match pieces.resume(generic, if taken { yes } else { no }) {
            Ok(piece) => piece,
            Err(reason) => {
                notes.push(format!(
                    "guard at {block}: continuation unavailable: {reason}"
                ));
                continue;
            }
        };
        let inputs: Option<Vec<_>> = piece
            .inputs
            .iter()
            .map(|value| {
                let current = *canonical.values.get(value)?;
                let mapped = values.get(&current).copied().unwrap_or(current);
                value_available(graph, &cfg, mapped, blocks[&block]).then_some(mapped)
            })
            .collect();
        let Some(inputs) = inputs else {
            notes.push(format!("guard at {block}: canonical inputs unavailable"));
            continue;
        };
        let prime = match &graph.blocks[blocks[&block]]
            .terminator
            .as_ref()
            .unwrap()
            .kind
        {
            TerminatorKind::Branch(condition, ..) => match graph.defining_op(*condition) {
                Some(Op::IsPropertyCached(object, site, _)) => Some((*object, *site)),
                _ => None,
            },
            _ => None,
        };
        let guard_span = graph.blocks[blocks[&block]]
            .terminator
            .as_ref()
            .unwrap()
            .span;
        if crate::jit::bailouts::enabled()
            && let TerminatorKind::Branch(condition, ..) = &graph.blocks[blocks[&block]]
                .terminator
                .as_ref()
                .unwrap()
                .kind
            && let Some(op) = graph.defining_op(*condition)
        {
            crate::jit::bailouts::note_guard(guard_span, graph.op_text(op));
        }
        let key = (
            std::sync::Arc::as_ptr(&piece.body) as usize,
            inputs.clone(),
            prime.as_ref().map(|(_, site)| *site),
        );
        let stub = *stubs.entry(key).or_insert_with(|| {
            let stub = graph.blocks.push(BlockData::default());
            if let Some((object, site)) = prime {
                graph.append(
                    stub,
                    guard_span,
                    Op::PrimePropertyCache(object, site),
                    false,
                );
            }
            let result = graph.append_value(
                stub,
                guard_span,
                Op::Continuation(piece.body.clone(), inputs),
            );
            graph.terminate(stub, span, TerminatorKind::Return(result));
            graph.cold.insert(stub);
            stub
        });
        let TerminatorKind::Branch(_, yes, no) = &mut graph.blocks[blocks[&block]]
            .terminator
            .as_mut()
            .unwrap()
            .kind
        else {
            continue;
        };
        *if taken { yes } else { no } = Target::new(stub);
        resumed.insert(blocks[&block], stub);
        notes.push(format!(
            "guard at {block}: resumes in {}",
            piece.body.graph.name
        ));
    }
    for header in pieces.headers(generic) {
        let cfg = Cfg::new(graph);
        let Some(&current) = canonical
            .blocks
            .get(&header)
            .filter(|current| region.contains(current) && cfg.rpo.contains(current))
        else {
            continue;
        };
        let piece = match pieces.driver(generic, header) {
            Ok(piece) => piece,
            Err(reason) => {
                notes.push(format!(
                    "loop at {current}: continuation unavailable: {reason}"
                ));
                continue;
            }
        };
        let inputs: Option<Vec<_>> = piece
            .inputs
            .iter()
            .map(|value| {
                let current_value = *canonical.values.get(value)?;
                value_available(graph, &cfg, current_value, current).then_some(current_value)
            })
            .collect();
        let Some(inputs) = inputs else {
            notes.push(format!(
                "loop at {current}: generic entry inputs unavailable"
            ));
            continue;
        };
        graph.blocks[current].insts.clear();
        let result = graph.append_value(current, span, Op::Continuation(piece.body, inputs));
        graph.terminate(current, span, TerminatorKind::Return(result));
        notes.push(format!("loop at {current}: separate generic function"));
    }
}

fn consume_assumes(
    graph: &mut Graph,
    blocks: &[Block],
    entry: Option<(Block, Block)>,
    checks: &[(Value, Fact)],
) {
    let scope = blocks
        .iter()
        .copied()
        .chain(entry.map(|(before, _)| before));
    for block in scope.collect::<Vec<_>>() {
        let consumed: Vec<Inst> = graph.blocks[block]
            .insts
            .iter()
            .copied()
            .filter(|&inst| {
                matches!(graph.insts[inst].op, Op::Assume(value, fact, _)
                    if checks.iter().any(|&(checked, known)| checked == value && known.implies(fact)))
            })
            .collect();
        for inst in consumed {
            graph.detach_inst(inst);
        }
    }
}

fn shared_resume(
    graph: &Graph,
    cfg: &Cfg,
    facts: &Facts,
    block: Block,
    resumed: &FxHashMap<Block, Block>,
) -> Option<Block> {
    let mut current = block;
    loop {
        let pure = graph.blocks[current]
            .insts
            .iter()
            .all(|&inst| facts.is_pure(inst));
        if !pure {
            return None;
        }
        let [previous] = cfg.preds[current][..] else {
            return None;
        };
        if let Some(&stub) = resumed.get(&previous) {
            return Some(stub);
        }
        if !matches!(
            graph.blocks[previous].terminator,
            Some(Terminator {
                kind: TerminatorKind::Jump(_),
                ..
            })
        ) {
            return None;
        }
        current = previous;
    }
}

fn value_available(graph: &Graph, cfg: &Cfg, value: Value, block: Block) -> bool {
    let defined = match graph.values[value] {
        ValueDef::Result(inst, _) => graph.blocks[graph.insts[inst].block].insts.contains(&inst),
        ValueDef::Param(block, _) => graph.blocks[block].params.contains(&value),
    };
    defined && cfg.dominates(graph.def_block(value), block)
}

fn continue_if(graph: &mut Graph, block: Block, check: Value, otherwise: &Target) -> Block {
    let next = graph.blocks.push(BlockData::default());
    let kind = TerminatorKind::Branch(check, Target::new(next), otherwise.clone());
    graph.terminate(block, graph.span, kind);
    next
}

fn typed_guards(
    graph: &mut Graph,
    mut current: Block,
    fallback: &Target,
    typed: &[(Value, Option<TypedKind>)],
    bounds: &[(Value, Value)],
) -> Block {
    let span = graph.span;
    for &(object, element) in typed {
        let check = graph.append_value(current, span, Op::IsTyped(object, element));
        current = continue_if(graph, current, check, fallback);
    }
    for &(object, bound) in bounds {
        let length = graph.append_value(current, span, Op::IndexedLength(object));
        let fits = Op::Binary(BinaryOperator::LessEqualThan, bound, length);
        let fits = graph.append_value(current, span, fits);
        current = continue_if(graph, current, fits, fallback);
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_proofs_require_the_success_edge_to_dominate() {
        use oxc_ast::ast::Statement;

        let allocator = oxc_allocator::Allocator::default();
        let parsed = oxc_parser::Parser::new(
            &allocator,
            "function f(xs){if(xs){xs=1;}return 0;}",
            oxc_span::SourceType::script(),
        )
        .parse();
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let mut graph = super::super::super::super::build::lower(
            function,
            semantic.scoping(),
            &FxHashSet::default(),
        )
        .unwrap();
        let xs = graph
            .insts
            .values()
            .find_map(|data| {
                matches!(data.op, Op::Argument(0))
                    .then_some(data.result())
                    .flatten()
            })
            .unwrap();
        let branch = Cfg::new(&graph)
            .rpo
            .into_iter()
            .find(|&block| {
                matches!(
                    graph.blocks[block].terminator.as_ref().unwrap().kind,
                    TerminatorKind::Branch(..)
                )
            })
            .unwrap();
        let condition = graph.append_value(
            branch,
            graph.span,
            Op::IsTyped(xs, Some(TypedKind::Float32)),
        );
        let TerminatorKind::Branch(test, yes, no) =
            &mut graph.blocks[branch].terminator.as_mut().unwrap().kind
        else {
            unreachable!()
        };
        *test = condition;
        let (yes, no) = (yes.block, no.block);
        let proof = EntryGuard::Fact(xs, Fact::Typed(Some(TypedKind::Float32)));
        assert!(EntryGuard::at_entry(&graph, yes).contains(&proof));
        assert!(!EntryGuard::at_entry(&graph, no).contains(&proof));
        assert!(!EntryGuard::at_entry(&graph, branch).contains(&proof));
        let exit = Cfg::new(&graph)
            .rpo
            .into_iter()
            .find(|&block| {
                matches!(
                    graph.blocks[block].terminator.as_ref().unwrap().kind,
                    TerminatorKind::Return(_)
                )
            })
            .unwrap();
        assert!(!EntryGuard::at_entry(&graph, exit).contains(&proof));
    }
}
