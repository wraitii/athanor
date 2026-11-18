use super::super::super::guards::{Fact, Kind, Origin};
use super::super::calls::ConditionalPlan;
use super::*;
use crate::FxHashSet;
use crate::jit::profile::{ReadKind, ReadType};
use crate::jit::semantic::JsOp;

fn numeric_demand(graph: &Graph, cfg: &Cfg, native_calls: &FxHashSet<Inst>) -> FxHashSet<Value> {
    let mut pending = Vec::new();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            for (index, value) in op.operands_iter().enumerate() {
                if matches!(graph.values[value], ValueDef::Param(..))
                    && (numeric_use(op, index)
                        || native_calls.contains(&inst)
                            && matches!(op, Op::Javascript(JsOp::Call { args, .. }) if args.contains(&value)))
                {
                    pending.push(value);
                }
            }
        }
    }
    let mut demanded = FxHashSet::default();
    while let Some(value) = pending.pop() {
        if !demanded.insert(value) {
            continue;
        }
        let ValueDef::Param(block, index) = graph.values[value] else {
            continue;
        };
        for &pred in &cfg.preds[block] {
            let terminator = graph.blocks[pred].terminator.as_ref().unwrap();
            for target in terminator.kind.targets() {
                if target.block == block {
                    pending.push(target.args[index]);
                }
            }
        }
    }
    demanded
}

pub(in crate::jit::semantic) fn numbers(graph: &mut Graph) -> Vec<(Value, Fact)> {
    let cfg = Cfg::new(graph);
    let facts = Facts::compute(graph);
    let hints = super::super::observations::type_hints(graph);
    let native_calls: FxHashSet<Inst> = requirements(graph, &cfg.rpo)
        .into_iter()
        .map(|requirement| requirement.call)
        .collect();
    let demanded = numeric_demand(graph, &cfg, &native_calls);
    let mut proposals: Vec<(Inst, Value, Fact, Option<crate::jit::profile::Site>)> = Vec::new();
    let mut seen = FxHashSet::default();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            for (index, value) in op.operands_iter().enumerate() {
                let integer = hints.get(&value) == Some(&ReadType::Integer);
                let string = hints.get(&value) == Some(&ReadType::String);
                let desired = if string {
                    ValueType::STRING
                } else if integer {
                    ValueType::INTEGER
                } else {
                    NUMERIC
                };
                let observed = matches!(
                    op,
                    Op::ReadHint(
                        _,
                        ReadKind::Number | ReadKind::Primitive,
                        ReadType::Integer | ReadType::Number | ReadType::String,
                        _
                    )
                ) && demanded.contains(&value);
                if hints.get(&value) == Some(&ReadType::Unknown)
                    || !(numeric_use(op, index)
                        || native_calls.contains(&inst)
                            && matches!(op, Op::Javascript(JsOp::Call { args, .. }) if args.contains(&value))
                        || observed)
                    || !facts.of(value).intersects(desired)
                    || facts.of(value).within(desired)
                    || matches!(
                        graph.defining_op(value),
                        Some(Op::Javascript(JsOp::Constant(_)))
                    )
                {
                    continue;
                }
                let fact = Fact::Type(if string {
                    ValueType::STRING
                } else if integer {
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
    let observed: Vec<_> = proposals
        .iter()
        .filter(|(_, value, _, _)| hints.contains_key(value))
        .map(|&(_, value, fact, _)| (value, fact))
        .collect();
    if !observed.is_empty() {
        // Infer from observed guards without letting derived assumptions prove themselves.
        let inferred = Facts::assuming_values(graph, cfg, &observed);
        proposals.retain(|&(_, value, fact, _)| {
            hints.contains_key(&value) || !inferred.of(value).within(fact.mask().unwrap())
        });
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
            kind: if super::super::observations::primitive_site(graph, value).is_some() {
                Kind::Primitive
            } else {
                Kind::Number
            },
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

pub(in crate::jit::semantic) fn typed(graph: &mut Graph, numbers: &[(Value, Fact)]) {
    let cfg = Cfg::new(graph);
    let under = Facts::assuming(graph, numbers);
    let mut proposals: Vec<(Block, Value, Fact, Kind)> = Vec::new();
    let mut seen = FxHashSet::default();
    let mut objects = FxHashSet::default();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            if let Op::Javascript(JsOp::GetIndex(object, key) | JsOp::TypedIndex(object, key, _)) =
                graph.insts[inst].op
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
        let (probe_inst, probe) =
            graph.append(block, span, Op::Javascript(JsOp::ProbeGlobal(name)), true);
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

pub(super) fn integer_ranges(
    graph: &Graph,
    facts: &Facts,
    preheader: Block,
    header: Block,
    body: &FxHashSet<Block>,
    numbers: &[(Value, Fact)],
) -> Vec<(Value, Fact)> {
    use super::super::super::ranges::{IntRange, Ranges};

    const LIMIT: i32 = 1 << 15;
    const GUARD_RANGE: IntRange = IntRange::new(0, LIMIT as i64);
    if !numbers
        .iter()
        .any(|&(_, fact)| fact == Fact::Type(ValueType::INTEGER))
    {
        return Vec::new();
    }
    let hints = super::super::observations::type_hints(graph);
    let mut inputs: Vec<_> = numbers
        .iter()
        .filter_map(|&(value, fact)| {
            (fact == Fact::Type(ValueType::INTEGER)
                && hints.get(&value) == Some(&ReadType::Integer)
                && !body.contains(&graph.def_block(value))
                && facts.cfg.dominates(graph.def_block(value), preheader))
            .then_some((value, GUARD_RANGE))
        })
        .collect();
    if inputs.is_empty() || inputs.len() > 3 {
        return Vec::new();
    }

    let size: usize = body
        .iter()
        .map(|&block| graph.blocks[block].insts.len())
        .sum();
    if size > SPECIALIZATION_REGION_LIMIT {
        return Vec::new();
    }
    let loops = natural_loops(graph, &facts.cfg);
    let nested: FxHashSet<_> = loops
        .into_iter()
        .filter(|(inner, blocks)| *inner != header && blocks.is_subset(body))
        .flat_map(|(_, blocks)| blocks)
        .collect();
    if nested.is_empty() {
        return Vec::new();
    }

    let receivers: FxHashSet<_> = graph
        .insts
        .values()
        .filter_map(|data| match data.op {
            Op::ReadHint(value, ReadKind::Receiver, ReadType::Array | ReadType::Typed(_), _) => {
                Some(value)
            }
            _ => None,
        })
        .collect();
    let mut known = Ranges::new(graph, facts, &[]);
    let mut assumed = Ranges::new(graph, facts, &inputs);
    for &block in &facts.cfg.rpo {
        if !nested.contains(&block) || graph.cold.contains(&block) {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            let index = match graph.insts[inst].op {
                Op::Javascript(JsOp::GetIndex(receiver, index))
                    if receivers.contains(&receiver) =>
                {
                    index
                }
                Op::Javascript(JsOp::ArrayGet(_, index) | JsOp::TypedIndex(_, index, _)) => index,
                _ => continue,
            };
            if facts.of(index).within(ValueType::INTEGER)
                || !benefits_from_ranges(graph, &mut assumed, index)
                || known.prove_arithmetic(index, &mut FxHashSet::default())
            {
                continue;
            }
            for i in (0..inputs.len()).rev() {
                let mut fewer = inputs.clone();
                fewer.remove(i);
                if benefits_from_ranges(graph, &mut Ranges::new(graph, facts, &fewer), index) {
                    inputs = fewer;
                }
            }
            if inputs.len() > 2 {
                return Vec::new();
            }
            return inputs
                .into_iter()
                .map(|(value, _)| (value, Fact::IntegerRange(0, LIMIT)))
                .collect();
        }
    }
    Vec::new()
}

fn benefits_from_ranges(
    graph: &Graph,
    ranges: &mut super::super::super::ranges::Ranges<'_>,
    index: Value,
) -> bool {
    use super::super::super::ranges::IntRange;

    let mut proved = FxHashSet::default();
    ranges.prove_arithmetic(index, &mut proved)
        && proved.iter().any(|&inst| {
            matches!(
                graph.insts[inst].op,
                Op::Javascript(JsOp::Binary(BinaryOperator::Multiplication, ..))
            )
        })
        && ranges
            .integer_at(index, graph.def_block(index))
            .is_some_and(|range| range.within(IntRange::I32))
}
