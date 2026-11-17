use super::hoisting::AssumeCfg;
use crate::jit::layout::ReadLayout;
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
                    | Op::AsString(_)
                    | Op::IsString(_)
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
    pub(in crate::jit::semantic) reads: &'a [Inst],
    pub(in crate::jit::semantic) shaped_reads: &'a [(Inst, ReadLayout)],
    pub(in crate::jit::semantic) guards: &'a [Block],
    pub(in crate::jit::semantic) exits: &'a [Block],
    pub(in crate::jit::semantic) exits_true: &'a [Block],
    pub(in crate::jit::semantic) typed_reads: &'a [(Inst, TypedRead)],
    pub(in crate::jit::semantic) assumes: &'a [(Value, Fact)],
    pub(in crate::jit::semantic) code_guards: &'a [(String, cranelift_module::FuncId)],
}

struct NativeCheck {
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

struct Copy {
    blocks: FxHashMap<Block, Block>,
    values: FxHashMap<Value, Value>,
    insts: FxHashMap<Inst, Inst>,
    copies: FxHashSet<Block>,
}

fn entry_checks(
    graph: &mut Graph,
    region: &[Block],
    entry: Option<(Block, Block)>,
    speculation: &Speculation<'_>,
    requirements: &[NativeCheck],
    hints: &FxHashMap<Value, crate::jit::profile::ReadType>,
    head: Block,
    mut tested: FxHashSet<EntryGuard>,
    values: &mut FxHashMap<Value, Value>,
    copies: &FxHashSet<Block>,
    original: Target,
) -> AssumeCfg {
    let Speculation {
        reads,
        guards,
        assumes,
        code_guards,
        ..
    } = *speculation;
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
    let first = graph.blocks.len();
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
                    .find(|&&(read, _)| read == inst)
                {
                    Some(&(_, layout)) => Op::ProbeShapeProperty(object, name, layout),
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
        graph.detach_inst(inst);
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
        let copy = block;
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
            graph.set_op(inst, Op::Constant(Constant::Boolean(true)));
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
            Some(Op::IsPropertyCached(object, site, _)) => {
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
            Fact::Array => Op::IsOrdinaryArray(checked),
            Fact::Shape(shape) => Op::IsShape(checked, shape),
            Fact::ReadLayout(layout) => Op::IsReadLayout(checked, layout),
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
        let otherwise = if let Op::IsPropertyCached(object, site, _) = &test {
            let prime = graph.blocks.push(BlockData::default());
            graph.cold.insert(prime);
            graph.append(prime, span, Op::PrimePropertyCache(*object, *site), false);
            graph.terminate(prime, span, TerminatorKind::Jump(original.clone()));
            Target::new(prime)
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
        let check = fact.test(checked).unwrap();
        let check = graph.append_value(current, span, check);
        let primitive = super::super::observations::primitive_site(graph, checked)
            .or_else(|| super::super::observations::primitive_site(graph, value));
        let site = primitive.or_else(|| super::super::observations::numeric_site(graph, value));
        let failure = type_failure(
            graph,
            checked,
            site,
            primitive.is_some(),
            original.clone(),
            span,
        );
        current = continue_if(graph, current, check, &failure);
    }
    let mut numbers = FxHashMap::default();
    for (value, fact) in numeric() {
        let checked = values.get(&value).copied().unwrap_or(value);
        let number = *numbers.entry(checked).or_insert_with(|| {
            graph.append_value(
                current,
                span,
                if fact == Fact::Type(ValueType::STRING) {
                    Op::AsString(checked)
                } else if fact == Fact::Type(ValueType::INTEGER) {
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
    let good = Target { block: head, args };
    graph.terminate(current, span, TerminatorKind::Jump(good.clone()));
    match entry {
        Some((preheader, _)) => {
            graph.terminate(preheader, span, TerminatorKind::Jump(Target::new(start)));
        }
        None => graph.entry = start,
    }
    AssumeCfg {
        start,
        exit: current,
        fallback: original.block,
        recovery: head,
        before: entry.map(|(before, _)| before),
        blocks: graph.blocks.keys().skip(first).collect(),
        scope: region.iter().copied().collect(),
        dependencies: Vec::new(),
    }
}

impl AssumeCfg {
    pub(super) fn atoms(self, graph: &mut Graph) -> Vec<Self> {
        let mut segments = Vec::new();
        let mut blocks = FxHashSet::default();
        let mut current = self.start;
        let mut start = current;
        let mut visited = FxHashSet::default();
        while current != self.exit {
            if !visited.insert(current) {
                return vec![self];
            }
            blocks.insert(current);
            match &graph.blocks[current].terminator.as_ref().unwrap().kind {
                TerminatorKind::Jump(target) if self.blocks.contains(&target.block) => {
                    current = target.block
                }
                TerminatorKind::Branch(_, yes, no) => {
                    let yes_failure = self.failure_path(graph, yes.block);
                    let no_failure = self.failure_path(graph, no.block);
                    let (success, failure, yes) = match (yes_failure, no_failure) {
                        (Some(failure), None) => (no.clone(), failure, false),
                        (None, Some(failure)) => (yes.clone(), failure, true),
                        _ => return vec![self],
                    };
                    blocks.extend(failure);
                    segments.push((
                        start,
                        current,
                        yes,
                        success.clone(),
                        std::mem::take(&mut blocks),
                    ));
                    current = success.block;
                    start = current;
                }
                _ => return vec![self],
            }
        }
        blocks.insert(current);
        let mut groups = Vec::new();
        let mut before = self.before;
        for (start, branch, yes, good, mut blocks) in segments {
            let exit = graph.blocks.push(BlockData::default());
            graph.terminate(exit, graph.span, TerminatorKind::Jump(good.clone()));
            let TerminatorKind::Branch(_, taken, otherwise) =
                &mut graph.blocks[branch].terminator.as_mut().unwrap().kind
            else {
                unreachable!()
            };
            *if yes { taken } else { otherwise } = Target::new(exit);
            let aliases: Vec<_> = graph.blocks[good.block]
                .insts
                .iter()
                .copied()
                .take_while(|&inst| matches!(graph.insts[inst].op, Op::AsObject(_)))
                .collect();
            for inst in aliases {
                graph.move_inst(inst, exit);
            }
            blocks.insert(exit);
            groups.push(Self {
                start,
                exit,
                fallback: self.fallback,
                recovery: self.recovery,
                before,
                blocks,
                scope: self.scope.clone(),
                dependencies: self.dependencies.clone(),
            });
            before = Some(exit);
        }
        groups.push(Self {
            start,
            before,
            blocks,
            ..self
        });
        Self::bind_outputs(graph, &groups);
        groups
    }

    fn bind_outputs(graph: &mut Graph, groups: &[Self]) {
        // Narrowing must move with its proof, rather than the next independent check.
        let cfg = Cfg::new(graph);
        let proofs: Vec<_> = groups
            .iter()
            .flat_map(|group| {
                group.blocks.iter().filter_map(|&block| {
                    let TerminatorKind::Branch(condition, good, _) =
                        &graph.blocks[block].terminator.as_ref()?.kind
                    else {
                        return None;
                    };
                    if good.block != group.exit {
                        return None;
                    }
                    let (value, fact) = implies(graph.defining_op(*condition)?)?;
                    Some((group.exit, value, fact))
                })
            })
            .collect();
        let insts: Vec<_> = groups
            .iter()
            .flat_map(|group| &group.blocks)
            .flat_map(|&block| graph.blocks[block].insts.iter().copied())
            .collect();
        for inst in insts {
            let data = &graph.insts[inst];
            let (value, required) = match data.op {
                Op::AsString(value) => (value, Fact::Type(ValueType::STRING)),
                Op::AsNumber(value) => (value, Fact::Type(ValueType::NUMBER)),
                Op::AsInteger(value) => (value, Fact::Type(ValueType::INTEGER)),
                _ => continue,
            };
            let block = data.block;
            let Some(&(exit, _, _)) = proofs.iter().find(|&&(exit, checked, fact)| {
                exit != block
                    && checked == value
                    && fact.implies(required)
                    && cfg.dominates(exit, block)
            }) else {
                continue;
            };
            let result = data.result().unwrap();
            let observations: Vec<_> = graph.blocks[block].insts.iter().copied()
                .filter(|&inst| matches!(graph.insts[inst].op, Op::Observe(value, Observation::Read(ReadKind::Number), _) if value == result)).collect();
            graph.move_inst(inst, exit);
            for observation in observations {
                graph.move_inst(observation, exit);
            }
        }
    }

    fn failure_path(&self, graph: &Graph, mut block: Block) -> Option<Vec<Block>> {
        let mut blocks = Vec::new();
        while block != self.fallback {
            if !self.blocks.contains(&block)
                || !graph.cold.contains(&block)
                || blocks.contains(&block)
            {
                return None;
            }
            blocks.push(block);
            let TerminatorKind::Jump(target) = &graph.blocks[block].terminator.as_ref()?.kind
            else {
                return None;
            };
            block = target.block;
        }
        Some(blocks)
    }
}

fn type_failure(
    graph: &mut Graph,
    value: Value,
    site: Option<crate::jit::profile::Site>,
    primitive: bool,
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
        Op::Observe(
            value,
            if primitive {
                Observation::PrimitiveFailure
            } else {
                Observation::NumberFailure
            },
            site,
        ),
        false,
    );
    graph.terminate(failure, span, TerminatorKind::Jump(fallback));
    Target::new(failure)
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

pub(super) struct Bailouts<'a> {
    generic: &'a Graph,
    live: cranelift_codegen::entity::SecondaryMap<Block, FxHashSet<Value>>,
}

impl<'a> Bailouts<'a> {
    pub(super) fn new(generic: &'a Graph) -> Self {
        let cfg = Cfg::new(generic);
        Self {
            generic,
            live: super::super::super::boundary::liveness(generic, &cfg.rpo),
        }
    }

    pub(super) fn available(
        &self,
        graph: &Graph,
        target: &Target,
        values: &FxHashMap<Value, Value>,
        cfg: &Cfg,
        before: Option<Block>,
    ) -> Option<FxHashMap<Value, Value>> {
        let insertion = before.unwrap_or(graph.entry);
        super::super::super::boundary::inputs(target, &self.live)
            .into_iter()
            .map(|original| {
                let current = values.get(&original).copied().unwrap_or(original);
                let available = |value| {
                    let block = graph.def_block(value);
                    let defined = match graph.values[value] {
                        ValueDef::Result(inst, _) => graph.blocks[block].insts.contains(&inst),
                        ValueDef::Param(_, _) => graph.blocks[block].params.contains(&value),
                    };
                    defined && cfg.rpo.contains(&block) && cfg.dominates(block, insertion)
                };
                if available(current) {
                    Some((original, current))
                } else if available(original) {
                    Some((original, original))
                } else {
                    None
                }
            })
            .collect()
    }

    pub(super) fn target(
        &self,
        graph: &mut Graph,
        target: Target,
        values: &FxHashMap<Value, Value>,
    ) -> Target {
        let inputs = super::super::super::boundary::inputs(&target, &self.live)
            .into_iter()
            .map(|value| values[&value])
            .collect();
        let block = graph.blocks.push(BlockData::default());
        graph.cold.insert(block);
        graph.terminate(
            block,
            graph.span,
            TerminatorKind::Bailout { target, inputs },
        );
        Target::new(block)
    }
}

pub(super) fn materialize(
    graph: &mut Graph,
    region: &[Block],
    entry: Option<(Block, Block)>,
    recovery: Target,
    speculation: &Speculation<'_>,
    requirements: &[Requirement],
    canonical_values: &mut FxHashMap<Value, Value>,
    bailouts: &mut Bailouts<'_>,
    notes: &mut Vec<String>,
) -> Result<Vec<AssumeCfg>> {
    let generic = bailouts.generic;
    let head = entry.map_or(graph.entry, |(_, header)| header);
    let recovery_block = recovery.block;
    let failure = bailouts.target(graph, recovery, canonical_values);
    let mut values = region
        .iter()
        .flat_map(|&block| {
            graph.blocks[block].params.iter().copied().chain(
                graph.blocks[block]
                    .insts
                    .iter()
                    .flat_map(|&inst| graph.insts[inst].results.iter().copied()),
            )
        })
        .map(|value| (value, value))
        .collect();
    let scope = region.iter().copied().collect();
    for &(inst, read) in speculation.typed_reads {
        if let Op::GetIndex(object, key) | Op::TypedIndex(object, key, _) = graph.insts[inst].op {
            graph.set_op(inst, Op::TypedIndex(object, key, read));
        }
    }
    let native: Vec<_> = requirements
        .iter()
        .map(|requirement| NativeCheck {
            global: requirement.global.clone(),
            property: requirement.property.clone(),
            native: requirement.native,
            captured: requirement.captured,
            object: requirement.object,
        })
        .collect();
    let hints = super::super::observations::type_hints(graph);
    let tested = EntryGuard::at_entry(graph, entry.map_or(graph.entry, |(before, _)| before));
    let mut group = entry_checks(
        graph,
        region,
        entry,
        speculation,
        &native,
        &hints,
        head,
        tested,
        &mut values,
        &scope,
        failure,
    );
    group.recovery = recovery_block;
    group.dependencies = guarded_accesses(requirements, speculation.code_guards);
    group
        .scope
        .retain(|block| (block.as_u32() as usize) < generic.blocks.len());
    let mut groups = vec![group];
    for current in canonical_values.values_mut() {
        *current = values.get(current).copied().unwrap_or(*current);
    }
    for &inst in speculation.locals {
        if !graph.blocks[graph.insts[inst].block].insts.contains(&inst) {
            continue;
        }
        let data = graph.insts[inst].clone();
        let Op::Assume(value, fact, origin) = data.op else {
            continue;
        };
        let facts = Facts::compute(graph);
        if fact
            .mask()
            .is_some_and(|types| facts.of(value).within(types))
            || super::super::super::guards::holds(graph, &facts.cfg, data.block, value, fact)
        {
            continue;
        }
        let [before] = facts.cfg.preds[data.block][..] else {
            return Err(format!("local assumption {inst} has no unique predecessor"));
        };
        let target = Target {
            block: data.block,
            args: generic.blocks[data.block].params.clone(),
        };
        let fallback = bailouts.target(graph, target, canonical_values);
        let first = graph.blocks.len();
        let start = graph.blocks.push(BlockData::default());
        let exit = graph.blocks.push(BlockData::default());
        let failure = type_failure(
            graph,
            value,
            origin.site,
            origin.kind == guards::Kind::Primitive,
            fallback.clone(),
            data.span,
        );
        let check = graph.append_value(
            start,
            data.span,
            fact.test(value).ok_or("unsupported local assumption")?,
        );
        graph.terminate(
            start,
            data.span,
            TerminatorKind::Branch(check, Target::new(exit), failure.clone()),
        );
        let good = Target::new(data.block);
        graph.terminate(exit, data.span, TerminatorKind::Jump(good.clone()));
        graph.terminate(before, data.span, TerminatorKind::Jump(Target::new(start)));
        groups.push(AssumeCfg {
            start,
            exit,
            fallback: fallback.block,
            recovery: data.block,
            before: Some(before),
            blocks: graph.blocks.keys().skip(first).collect(),
            scope: [data.block].into_iter().collect(),
            dependencies: Vec::new(),
        });
        if let Some(types) = fact.mask() {
            let (_, number) = graph.insert(
                data.block,
                0,
                data.span,
                if types == ValueType::STRING {
                    Op::AsString(value)
                } else if types.within(ValueType::INTEGER) {
                    Op::AsInteger(value)
                } else {
                    Op::AsNumber(value)
                },
                true,
            );
            let cfg = Cfg::new(graph);
            let dominated = cfg
                .rpo
                .iter()
                .copied()
                .filter(|&block| cfg.dominates(data.block, block))
                .collect();
            retarget(graph, &dominated, value, number.unwrap());
        }
        graph.detach_inst(inst);
        notes.push(format!("local assumption {inst}: bailout to generic code"));
    }
    for (block, taken) in speculation
        .exits
        .iter()
        .map(|&block| (block, false))
        .chain(speculation.exits_true.iter().map(|&block| (block, true)))
    {
        if speculation.guards.contains(&block) || block.as_u32() as usize >= generic.blocks.len() {
            continue;
        }
        let TerminatorKind::Branch(condition, yes, no) = graph.blocks[block]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .clone()
        else {
            continue;
        };
        let current = if taken { yes } else { no };
        if matches!(
            graph.blocks[current.block]
                .terminator
                .as_ref()
                .map(|term| &term.kind),
            Some(TerminatorKind::Bailout { .. })
        ) {
            continue;
        }
        let TerminatorKind::Branch(_, yes, no) =
            &generic.blocks[block].terminator.as_ref().unwrap().kind
        else {
            continue;
        };
        let failure = bailouts.target(
            graph,
            if taken { yes } else { no }.clone(),
            canonical_values,
        );
        if crate::jit::bailouts::enabled()
            && let Some(op) = graph.defining_op(condition)
        {
            let span = graph.blocks[block].terminator.as_ref().unwrap().span;
            crate::jit::bailouts::note_guard(span, graph.op_text(op));
        }
        if let Some(Op::IsPropertyCached(object, site, _)) = graph.defining_op(condition) {
            graph.append(
                failure.block,
                graph.span,
                Op::PrimePropertyCache(*object, *site),
                false,
            );
        }
        let TerminatorKind::Branch(_, yes, no) =
            &mut graph.blocks[block].terminator.as_mut().unwrap().kind
        else {
            unreachable!();
        };
        *if taken { yes } else { no } = failure;
    }
    consume_assumes(graph, region, entry, speculation.assumes);
    let groups = groups
        .into_iter()
        .flat_map(|group| group.atoms(graph))
        .collect();
    graph.verify()?;
    Ok(groups)
}

fn append_generic(graph: &mut Graph, generic: &Graph) -> Copy {
    let cfg = Cfg::new(generic);
    let mut copy = Copy {
        blocks: Default::default(),
        insts: Default::default(),
        values: Default::default(),
        copies: Default::default(),
    };
    for &block in &cfg.rpo {
        let new = graph.blocks.push(BlockData::default());
        copy.blocks.insert(block, new);
        copy.copies.insert(new);
        graph.cold.insert(new);
        for &param in &generic.blocks[block].params {
            let value = graph.add_param(new);
            graph.reprs[value] = generic.reprs[param];
            copy.values.insert(param, value);
        }
        for &inst in &generic.blocks[block].insts {
            let data = &generic.insts[inst];
            let (new_inst, results) =
                graph.append_results(new, data.span, data.op.clone(), data.results.len());
            graph.insts[new_inst].source_name = data.source_name;
            graph.insts[new_inst].inline_context = data.inline_context;
            graph.insts[new_inst].emission_location = data.emission_location;
            copy.insts.insert(inst, new_inst);
            for (&old, &value) in data.results.iter().zip(&results) {
                graph.reprs[value] = generic.reprs[old];
                copy.values.insert(old, value);
            }
        }
    }
    for &block in &cfg.rpo {
        for &inst in &generic.blocks[block].insts {
            let new = copy.insts[&inst];
            for value in graph.insts[new].op.operands_mut() {
                *value = copy.values[value];
            }
        }
        let mut term = generic.blocks[block].terminator.clone().unwrap();
        for value in term.kind.operands_mut() {
            *value = copy.values[value];
        }
        for target in term.kind.targets_mut() {
            target.block = copy.blocks[&target.block];
        }
        graph.blocks[copy.blocks[&block]].terminator = Some(term);
    }
    copy
}

pub(super) fn finish(
    graph: &mut Graph,
    generic: &Graph,
    fallback: Option<usize>,
    continuations: bool,
    regions: &[RegionId],
    notes: &mut Vec<String>,
) -> Result<()> {
    let bailouts = Bailouts::new(generic);
    let mut pieces = super::super::super::continuation::Pieces::default();
    let mut copied = None;
    let mut previous = None;
    let mut definitions: FxHashMap<Value, Vec<Value>> = FxHashMap::default();
    let scopes = super::super::super::boundary::scopes(graph);
    for block in Cfg::new(graph).rpo {
        let term = graph.blocks[block].terminator.clone().unwrap();
        let TerminatorKind::Bailout { target, inputs } = term.kind else {
            continue;
        };
        if target.block == generic.entry
            && let Some(address) = fallback
        {
            previous = Some(address);
            graph.terminate(block, term.span, TerminatorKind::Fallback(address));
            continue;
        }
        let originals = super::super::super::boundary::inputs(&target, &bailouts.live);
        let actual: FxHashMap<_, _> = originals.into_iter().zip(inputs).collect();
        if continuations
            && target.block != generic.entry
            && let Ok(piece) = pieces.resume(generic, &target)
            && let Some(inputs) = piece
                .inputs
                .iter()
                .map(|input| actual.get(input).copied())
                .collect::<Option<Vec<_>>>()
        {
            let result = graph.append_value(
                block,
                term.span,
                Op::Continuation(piece.body.clone(), inputs),
            );
            for &scope in scopes[block].iter().rev() {
                graph.append(block, term.span, Op::LeaveScope(scope), false);
            }
            graph.terminate(block, term.span, TerminatorKind::Return(result));
            notes.push(format!(
                "bailout at {block}: resumes in {}",
                piece.body.graph.name
            ));
            continue;
        }
        let copy = copied.get_or_insert_with(|| append_generic(graph, generic));
        for (&original, &value) in &actual {
            definitions
                .entry(copy.values[&original])
                .or_default()
                .push(value);
        }
        let args = target.args.iter().map(|value| actual[value]).collect();
        graph.terminate(
            block,
            term.span,
            TerminatorKind::Jump(Target {
                block: copy.blocks[&target.block],
                args,
            }),
        );
    }
    if let Some(copy) = copied {
        for (&old, &new) in &copy.values {
            let defined = match graph.values[old] {
                ValueDef::Result(inst, _) => {
                    graph.blocks[graph.insts[inst].block].insts.contains(&inst)
                }
                ValueDef::Param(block, _) => graph.blocks[block].params.contains(&old),
            };
            if defined {
                definitions.entry(new).or_default().push(old);
            }
        }
        repair(graph, &definitions);
        if continuations {
            for header in pieces.headers(generic) {
                let cfg = Cfg::new(graph);
                let current = copy.blocks[&header];
                if !cfg.rpo.contains(&current) {
                    continue;
                }
                let Ok(piece) = pieces.driver(generic, header) else {
                    continue;
                };
                let inputs = piece
                    .inputs
                    .iter()
                    .map(|value| copy.values[value])
                    .collect();
                let scopes = if current == graph.entry {
                    Vec::new()
                } else {
                    super::super::super::boundary::scopes(graph)[cfg.preds[current][0]].clone()
                };
                graph.blocks[current].insts.clear();
                let result =
                    graph.append_value(current, generic.span, Op::Continuation(piece.body, inputs));
                for scope in scopes.into_iter().rev() {
                    graph.append(current, generic.span, Op::LeaveScope(scope), false);
                }
                graph.terminate(current, generic.span, TerminatorKind::Return(result));
            }
            repair(graph, &definitions);
        }
        let source = graph.regions.push(Region {
            blocks: copy.copies,
            fallback: None,
        });
        for &region in regions {
            graph.regions[region].fallback = Some(RegionFallback::Region(source));
        }
    } else if let Some(address) = previous {
        for &region in regions {
            graph.regions[region].fallback = Some(RegionFallback::CompiledEntry(address));
        }
    }
    graph.verify()
}
