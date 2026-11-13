use crate::jit::effects::Locations;
use crate::jit::profile::{PropertyAccess, ShapeAccess, Site};

use crate::{FxHashMap, FxHashSet};

use super::super::{
    Block, BlockData, Cfg, Facts, Graph, Inst, Op, Target, Terminator, TerminatorKind, Value,
};

#[derive(Clone, Default, PartialEq, Eq)]
pub(in crate::jit) struct Plan {
    source: usize,
    accesses: Vec<(Site, PropertyAccess, u32, u32)>,
    polymorphic: Vec<(Site, Vec<(u32, u32)>)>,
    megamorphic: Vec<Site>,
}

impl Plan {
    pub(in crate::jit) fn polymorphic_reads(&self) -> &[(Site, Vec<(u32, u32)>)] {
        &self.polymorphic
    }

    pub(in crate::jit::semantic) fn alternatives(
        &self,
        graph: &Graph,
        inst: Inst,
    ) -> Option<&[(u32, u32)]> {
        let site = site(graph, inst, self.source);
        self.polymorphic
            .iter()
            .find_map(|(origin, entries)| (*origin == site).then_some(entries.as_slice()))
    }

    pub(in crate::jit) fn megamorphic_sites(&self) -> &[Site] {
        &self.megamorphic
    }

    pub(in crate::jit::semantic) fn megamorphic(&self, graph: &Graph, inst: Inst) -> bool {
        self.megamorphic.contains(&site(graph, inst, self.source))
    }

    pub(in crate::jit::semantic) fn accesses(&self) -> &[(Site, PropertyAccess, u32, u32)] {
        &self.accesses
    }

    pub(in crate::jit::semantic) fn read(&self, graph: &Graph, inst: Inst) -> Option<(u32, u32)> {
        let site = site(graph, inst, self.source);
        self.accesses
            .iter()
            .find_map(|&(origin, access, shape, index)| {
                (origin == site && access == PropertyAccess::Get).then_some((shape, index))
            })
    }
}

fn property_access(op: &Op) -> Option<PropertyAccess> {
    match op {
        Op::Get(_, name) | Op::ProbeProperty(_, name) if name != "length" => {
            Some(PropertyAccess::Get)
        }
        Op::Set(..) => Some(PropertyAccess::Set),
        _ => None,
    }
}

fn site(graph: &Graph, inst: Inst, source: usize) -> Site {
    let data = &graph.insts[inst];
    Site {
        source: data.source_name.map_or(source, |name| name.0),
        start: data.span.start,
        end: data.span.end,
    }
}

pub(in crate::jit::semantic) fn plan(
    graph: &Graph,
    bodies: &super::calls::Repository,
    source: usize,
) -> Plan {
    let callees = graph.callees();
    let assumed: FxHashMap<_, _> = graph.blocks[graph.entry]
        .insts
        .iter()
        .filter_map(|&inst| match graph.insts[inst].op {
            Op::EntryAssume(value, super::super::guards::Fact::Shape(shape)) => {
                Some((value, shape))
            }
            _ => None,
        })
        .collect();
    let mut accesses = Vec::new();
    let mut polymorphic = Vec::new();
    let mut megamorphic = Vec::new();
    for (inst, data) in graph.insts.iter() {
        if property_access(&data.op) != Some(PropertyAccess::Get) {
            continue;
        }
        let site = site(graph, inst, source);
        if bodies.shape_feedback.get(&(site, PropertyAccess::Get))
            == Some(&ShapeAccess::Megamorphic)
            && !megamorphic.contains(&site)
        {
            megamorphic.push(site);
        }
    }
    for block in Cfg::new(graph).rpo {
        if graph.cold.contains(&block) {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let Some(property) = property_access(&data.op) else {
                continue;
            };
            let site = site(graph, inst, source);
            let callee = data.result().is_some_and(|value| callees.contains(&value));
            let object = match data.op {
                Op::Get(object, _) | Op::ProbeProperty(object, _) | Op::Set(object, ..) => object,
                _ => continue,
            };
            if let Some(&ShapeAccess::PolymorphicRead { entries, len }) =
                bodies.shape_feedback.get(&(site, property))
                && matches!(data.op, Op::Get(..))
                && (callee
                    || !assumed.get(&object).is_some_and(|shape| {
                        entries[..usize::from(len)]
                            .iter()
                            .any(|entry| entry.0 == *shape)
                    }))
            {
                if !polymorphic.iter().any(|(origin, _)| *origin == site) {
                    polymorphic.push((site, entries[..usize::from(len)].to_vec()));
                }
                continue;
            }
            if callee {
                continue;
            }
            let access = match (property, bodies.shape_feedback.get(&(site, property))) {
                (PropertyAccess::Get, Some(&access)) => {
                    let Some((shape, index)) = access.read(assumed.get(&object).copied()) else {
                        continue;
                    };
                    (site, property, shape, index)
                }
                (PropertyAccess::Set, Some(&ShapeAccess::Overwrite { shape, index })) => {
                    (site, property, shape, index)
                }
                _ => continue,
            };
            if !accesses.contains(&access) {
                accesses.push(access);
            }
        }
    }
    Plan {
        source,
        accesses,
        polymorphic,
        megamorphic,
    }
}

fn planned(graph: &Graph, plan: &Plan) -> FxHashMap<Inst, (Value, u32, u32)> {
    let callees = graph.callees();
    let cfg = Cfg::new(graph);
    let loops = super::super::transforms::loops::natural_loops(graph, &cfg);
    let invariant = |block: Block, object: Value| {
        loops
            .iter()
            .any(|(_, body)| body.contains(&block) && !body.contains(&graph.def_block(object)))
    };
    let mut planned = FxHashMap::default();
    for &block in &cfg.rpo {
        if graph.cold.contains(&block) {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let Some(property) = property_access(&data.op) else {
                continue;
            };
            let (Op::Get(object, _) | Op::Set(object, ..)) = data.op else {
                continue;
            };
            if data.result().is_some_and(|value| callees.contains(&value))
                || invariant(block, object)
            {
                continue;
            }
            let site = site(graph, inst, plan.source);
            if let Some(&(.., shape, index)) = plan
                .accesses
                .iter()
                .find(|access| access.0 == site && access.1 == property)
            {
                planned.insert(inst, (object, shape, index));
            }
        }
    }
    planned
}

const SEGMENT_LIMIT: usize = 64;

struct Segment {
    insts: Vec<Inst>,
    guards: Vec<(Value, u32)>,
}

fn shape_preserving(graph: &Graph, facts: &Facts, op: &Op) -> bool {
    if matches!(
        op,
        Op::EnterScope(_)
            | Op::EnterCaptures(..)
            | Op::LeaveScope(_)
            | Op::NextIteration(_)
            | Op::FreshIteration(_)
            | Op::Continuation(..)
    ) {
        return false;
    }
    !graph
        .contract(op, facts)
        .effects
        .memory
        .writes
        .intersects(Locations::CODE)
}

fn segments(graph: &Graph, planned: &FxHashMap<Inst, (Value, u32, u32)>) -> Vec<Segment> {
    let facts = Facts::compute(graph);
    let blocks: FxHashSet<Block> = planned
        .keys()
        .map(|&inst| graph.insts[inst].block)
        .collect();
    let mut segments = Vec::new();
    for block in blocks {
        let mut current: Option<Segment> = None;
        let mut pending = Vec::new();
        let mut defined = FxHashSet::default();
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            if let Some(&(object, shape, _)) = planned.get(&inst) {
                let joins = current.as_ref().is_some_and(|segment| {
                    !defined.contains(&object)
                        && segment.insts.len() + pending.len() < SEGMENT_LIMIT
                        && segment
                            .guards
                            .iter()
                            .all(|&(guarded, expected)| guarded != object || expected == shape)
                });
                if joins {
                    let segment = current.as_mut().unwrap();
                    segment.insts.append(&mut pending);
                    segment.insts.push(inst);
                    if !segment.guards.iter().any(|&(guarded, _)| guarded == object) {
                        segment.guards.push((object, shape));
                    }
                } else {
                    segments.extend(current.take());
                    pending.clear();
                    defined.clear();
                    current = Some(Segment {
                        insts: vec![inst],
                        guards: vec![(object, shape)],
                    });
                }
                defined.extend(data.result());
                continue;
            }
            if current.is_some() && shape_preserving(graph, &facts, &data.op) {
                pending.push(inst);
                defined.extend(data.result());
                continue;
            }
            segments.extend(current.take());
            pending.clear();
            defined.clear();
        }
        segments.extend(current);
    }
    segments
}

pub(in crate::jit::semantic) fn apply(
    graph: &mut Graph,
    plan: &Plan,
    notes: &mut Vec<String>,
) -> super::super::Result<()> {
    if plan.accesses.is_empty() && plan.polymorphic.is_empty() {
        return Ok(());
    }
    let planned = planned(graph, plan);
    let accesses: FxHashMap<Inst, (u32, u32)> = planned
        .iter()
        .map(|(&inst, &(_, shape, index))| (inst, (shape, index)))
        .collect();
    let segments = segments(graph, &planned);
    let mut guards = 0;
    let mut aliases = FxHashMap::default();
    for segment in &segments {
        guard_segment(graph, segment, &accesses, &mut aliases);
        guards += segment.guards.len();
    }
    if !segments.is_empty() {
        notes.push(format!(
            "shape guards: {} property accesses, {} segments, {guards} guards",
            accesses.len(),
            segments.len()
        ));
    }
    let reads: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter(|&inst| {
            matches!(graph.insts[inst].op, Op::Get(..)) && plan.alternatives(graph, inst).is_some()
        })
        .collect();
    for inst in &reads {
        let data = graph.insts[*inst].clone();
        let Op::Get(object, name) = data.op else {
            unreachable!()
        };
        let alternatives = plan
            .alternatives(graph, *inst)
            .unwrap()
            .iter()
            .map(|&(shape, index)| super::super::dispatch::Alternative {
                requirements: vec![(object, super::super::guards::Fact::Shape(shape))],
                implementation: index,
            })
            .collect();
        super::super::dispatch::apply(graph, *inst, alternatives, |graph, block, index| {
            let (read, result) = graph.append(
                block,
                data.span,
                Op::ShapedGet(object, name.clone(), index),
                true,
            );
            graph.insts[read].source_name = data.source_name;
            result
        });
    }
    if !reads.is_empty() {
        notes.push(format!("polymorphic gets: {} accesses", reads.len()));
    }
    graph.verify()
}

fn guard_segment(
    graph: &mut Graph,
    segment: &Segment,
    accesses: &FxHashMap<Inst, (u32, u32)>,
    aliases: &mut FxHashMap<Value, Value>,
) {
    let first = graph.insts[segment.insts[0]].clone();
    let block = first.block;
    let at_entry = block == graph.entry && {
        let facts = Facts::compute(graph);
        graph.blocks[block]
            .insts
            .iter()
            .take_while(|&&inst| inst != segment.insts[0])
            .all(|&inst| shape_preserving(graph, &facts, &graph.insts[inst].op))
    };
    let last = *segment.insts.last().unwrap();
    let position = graph.blocks[block]
        .insts
        .iter()
        .position(|&inst| inst == last)
        .unwrap();
    let continuation = graph.blocks.push(BlockData::default());
    let tail = graph.blocks[block].insts.split_off(position + 1);
    for &inst in &tail {
        graph.insts[inst].block = continuation;
    }
    graph.blocks[continuation].insts = tail;
    graph.blocks[continuation].terminator = graph.blocks[block].terminator.take();

    let fast = graph.blocks.push(BlockData::default());
    let fallback = graph.blocks.push(BlockData::default());
    graph.cold.insert(fallback);
    let mut clones = FxHashMap::default();
    for &inst in &segment.insts {
        let data = graph.insts[inst].clone();
        let mut op = data.op.clone();
        for value in op.operands_mut() {
            *value = clones.get(value).copied().unwrap_or(*value);
        }
        let (clone, result) = graph.append(fallback, data.span, op, data.result().is_some());
        graph.insts[clone].source_name = data.source_name;
        if let (Some(original), Some(result)) = (data.result(), result) {
            clones.insert(original, result);
        }
        graph.move_inst(inst, fast);
        let Some(&(_, index)) = accesses.get(&inst) else {
            continue;
        };
        graph.insts[inst].op = match data.op {
            Op::Get(object, name) => Op::ShapedGet(object, name, index),
            Op::Set(object, name, value) => Op::ShapedSet(object, name, index, value),
            op => op,
        };
    }

    let counts = graph.use_counts();
    let mut inner: FxHashMap<Value, u32> = FxHashMap::default();
    for &inst in &segment.insts {
        for value in graph.insts[inst].op.operands_iter() {
            *inner.entry(value).or_default() += 1;
        }
    }
    let outputs: Vec<Value> = segment
        .insts
        .iter()
        .filter_map(|&inst| graph.insts[inst].result())
        .filter(|value| counts[*value] > inner.get(value).copied().unwrap_or(0))
        .collect();
    let params: FxHashMap<Value, Value> = outputs
        .iter()
        .map(|&value| (value, graph.add_param(continuation)))
        .collect();
    graph.replace_uses(&params);
    aliases.extend(params.iter().map(|(&value, &param)| (value, param)));
    let originals: FxHashMap<Value, Value> = params.iter().map(|(&k, &v)| (v, k)).collect();
    for &inst in &segment.insts {
        for value in graph.insts[inst].op.operands_mut() {
            *value = originals.get(value).copied().unwrap_or(*value);
        }
    }
    let jump = |args| Terminator {
        source_name: first.source_name,
        emission_location: None,
        span: first.span,
        kind: TerminatorKind::Jump(Target {
            block: continuation,
            args,
        }),
    };
    graph.blocks[fast].terminator = Some(jump(outputs.clone()));
    graph.blocks[fallback].terminator =
        Some(jump(outputs.iter().map(|value| clones[value]).collect()));

    let mut check_block = block;
    for (position, &(object, shape)) in segment.guards.iter().enumerate() {
        let object = super::super::resolve(aliases, object);
        let next = if position + 1 == segment.guards.len() {
            fast
        } else {
            graph.blocks.push(BlockData::default())
        };
        if at_entry {
            super::super::guards::record_entry(
                graph,
                object,
                super::super::guards::Fact::Shape(shape),
            );
        }
        let check = graph.append_value(check_block, first.span, Op::IsShape(object, shape));
        graph.blocks[check_block].terminator = Some(Terminator {
            source_name: first.source_name,
            emission_location: None,
            span: first.span,
            kind: TerminatorKind::Branch(check, Target::new(next), Target::new(fallback)),
        });
        check_block = next;
    }
}
