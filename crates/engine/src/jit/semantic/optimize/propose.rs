use super::super::guards::{Fact, Kind, Origin};
use super::super::specialize::ConditionalPlan;
use super::placement::{Reach, place_assumes};
use super::*;
use crate::jit::profile::ReadType;
use crate::{FxHashMap, FxHashSet};

pub(super) fn numbers(graph: &mut Graph) -> Vec<(Value, Fact)> {
    let cfg = Cfg::new(graph);
    let facts = Facts::compute(graph);
    let hints = super::super::observations::numeric_hints(graph);
    let native_calls: FxHashSet<Inst> = requirements(graph, &cfg.rpo)
        .into_iter()
        .map(|requirement| requirement.call)
        .collect();
    let mut proposals: Vec<(Inst, Value, Fact, Option<crate::jit::profile::Site>)> = Vec::new();
    let mut seen = FxHashSet::default();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            for (index, value) in op.operands_iter().enumerate() {
                let integer = hints.get(&value) == Some(&ReadType::Integer);
                let desired = if integer { ValueType::INTEGER } else { NUMERIC };
                if hints.get(&value) == Some(&ReadType::Unknown)
                    || !(numeric_use(op, index)
                        || native_calls.contains(&inst)
                            && matches!(op, Op::Call { args, .. } if args.contains(&value)))
                    || facts.of(value).within(desired)
                    || matches!(graph.defining_op(value), Some(Op::Constant(_)))
                {
                    continue;
                }
                let fact = Fact::Type(if integer {
                    ValueType::INTEGER
                } else {
                    ValueType::NUMBER
                });
                if !seen.insert((block, value, fact)) {
                    continue;
                }
                let site = super::super::observations::numeric_site(graph, value);
                proposals.push((inst, value, fact, site));
            }
        }
    }
    for &(before, value, fact, site) in &proposals {
        let block = graph.insts[before].block;
        let at = graph.blocks[block]
            .insts
            .iter()
            .position(|&inst| inst == before)
            .unwrap();
        let origin = Origin {
            site,
            kind: Kind::Number,
            gain: 1,
        };
        graph.insert(
            block,
            at,
            graph.insts[before].span,
            Op::Assume(value, fact, origin),
            false,
        );
    }
    proposals
        .into_iter()
        .map(|(_, value, fact, _)| (value, fact))
        .collect()
}

pub(super) fn typed(graph: &mut Graph, numbers: &[(Value, Fact)]) {
    let cfg = Cfg::new(graph);
    let under = Facts::assuming(graph, numbers);
    let mut proposals: Vec<(Block, Value, Fact, Kind)> = Vec::new();
    let mut seen = FxHashSet::default();
    let mut objects = FxHashSet::default();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            if let Op::GetIndex(object, key) | Op::TypedIndex(object, key, _) = graph.insts[inst].op
                && under.of(key).within(NUMERIC)
                && let Some(element) = element_guess(graph, object)
            {
                objects.insert(object);
                if seen.insert((block, object)) {
                    proposals.push((block, object, Fact::Typed(Some(element)), Kind::Receiver));
                }
            }
        }
    }
    let mut objects: Vec<Value> = objects.into_iter().collect();
    objects.sort();
    let everything: FxHashSet<Value> = graph.values.keys().collect();
    let body: FxHashSet<Block> = cfg.rpo.iter().copied().collect();
    let mut bounds_seen = FxHashSet::default();
    let mut reads: Vec<_> = bounded_reads(graph, &cfg, &body, &objects, &everything, &under)
        .into_iter()
        .collect();
    reads.sort_by_key(|&(inst, _)| inst);
    for (inst, (object, bound)) in reads {
        let block = graph.insts[inst].block;
        if bounds_seen.insert((block, object, bound)) {
            proposals.push((block, object, Fact::Bound(bound), Kind::Bound));
        }
    }
    let span = graph.span;
    for (block, value, fact, kind) in proposals {
        let origin = Origin {
            site: None,
            kind,
            gain: 1,
        };
        graph.append(block, span, Op::Assume(value, fact, origin), false);
    }
}

pub(super) fn natives(graph: &mut Graph) {
    let blocks = Cfg::new(graph).rpo;
    let mut proposals = Vec::new();
    let mut seen = FxHashSet::default();
    for requirement in requirements(graph, &blocks) {
        let Op::Call { callee, .. } = graph.insts[requirement.call].op else {
            continue;
        };
        let block = graph.insts[requirement.call].block;
        if seen.insert((block, callee, requirement.native)) {
            proposals.push((block, callee, requirement.native));
        }
    }
    let span = graph.span;
    for (block, callee, native) in proposals {
        let origin = Origin {
            site: None,
            kind: Kind::Native,
            gain: 1,
        };
        graph.append(
            block,
            span,
            Op::Assume(callee, Fact::Native(native), origin),
            false,
        );
    }
}

pub(super) fn identities(graph: &mut Graph, conditional: &ConditionalPlan) -> Vec<Inst> {
    let mut proposals = Vec::new();
    let mut seen = FxHashSet::default();
    for call in &conditional.calls {
        let block = graph.insts[call.inst].block;
        if graph.cold.contains(&block) {
            continue;
        }
        for (name, code) in &call.identities {
            if seen.insert((block, name.clone(), *code)) {
                proposals.push((block, name.clone(), *code));
            }
        }
    }
    let span = graph.span;
    let mut added = Vec::new();
    for (block, name, code) in proposals {
        let (probe_inst, probe) = graph.append(block, span, Op::ProbeGlobal(name), true);
        let origin = Origin {
            site: None,
            kind: Kind::Identity,
            gain: 1,
        };
        graph.append(
            block,
            span,
            Op::Assume(probe.unwrap(), Fact::Code(code), origin),
            false,
        );
        added.push(probe_inst);
    }
    added
}

pub(super) fn assumed_calls(
    graph: &Graph,
    conditional: &ConditionalPlan,
) -> FxHashMap<Value, cranelift_module::FuncId> {
    conditional
        .calls
        .iter()
        .filter(|call| !graph.cold.contains(&graph.insts[call.inst].block))
        .map(|call| (call.callee, call.target))
        .collect()
}

pub(super) fn assume(graph: &mut Graph, conditional: &ConditionalPlan) -> Vec<Reach> {
    let numbers = numbers(graph);
    typed(graph, &numbers);
    natives(graph);
    let probes = identities(graph, conditional);
    let assumed = assumed_calls(graph, conditional);
    let mut reached = place_assumes(graph, &assumed);
    let cfg = Cfg::new(graph);
    let loops = natural_loops(graph, &cfg);
    let hints = super::super::observations::numeric_hints(graph);
    let local: Vec<_> = cfg
        .rpo
        .iter()
        .flat_map(|&block| graph.blocks[block].insts.iter().copied())
        .filter(|&inst| {
            (match graph.insts[inst].op {
                Op::Assume(value, Fact::Type(_), _) => matches!(
                    hints.get(&value),
                    Some(ReadType::Integer | ReadType::Number)
                ),
                Op::Assume(_, Fact::Typed(_) | Fact::Indexed, _) => true,
                _ => false,
            }) && !reached.iter().any(|reach| reach.inst == inst)
                && loops
                    .iter()
                    .any(|(_, body)| body.contains(&graph.insts[inst].block))
        })
        .collect();
    for inst in local {
        let data = graph.insts[inst].clone();
        let Op::Assume(value, fact, origin) = data.op else {
            unreachable!()
        };
        let block = data.block;
        let at = graph.blocks[block]
            .insts
            .iter()
            .position(|&current| current == inst)
            .unwrap();
        let next = graph.blocks.push(BlockData::default());
        graph.blocks[next].insts = graph.blocks[block].insts.split_off(at);
        for &moved in &graph.blocks[next].insts {
            graph.insts[moved].block = next;
        }
        graph.blocks[next].terminator = graph.blocks[block].terminator.take();
        graph.terminate(block, data.span, TerminatorKind::Jump(Target::new(next)));
        for region in graph.regions.values_mut() {
            if region.blocks.contains(&block) {
                region.blocks.insert(next);
            }
        }
        reached.push(Reach {
            inst,
            local: true,
            value,
            fact,
            use_block: next,
            header: None,
            global: None,
            origin,
        });
    }
    for inst in probes {
        graph.remove_inst(inst);
    }
    reached
}
