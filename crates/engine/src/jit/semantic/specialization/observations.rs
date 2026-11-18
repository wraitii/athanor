use crate::jit::semantic::JsOp;
use crate::{FxHashMap, FxHashSet};
use oxc_syntax::operator::BinaryOperator;

use crate::jit::profile::{Observation, ReadKind, ReadType, Site};

use super::super::{Cfg, Facts, Graph, Op, TerminatorKind, UnaryOp, Value, ValueType};
use super::calls::Repository;

pub(in crate::jit::semantic) fn plan(
    graph: &mut Graph,
    specialization: Option<(&Repository, usize)>,
) -> Vec<(Site, ReadType)> {
    let number_first = specialization
        .is_some_and(|(repository, source)| repository.number_first(source, graph.span.start));
    let mut carried: FxHashMap<_, Vec<_>> = FxHashMap::default();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            if let Op::Observe(value, Observation::Read(kind), site)
            | Op::ReadHint(value, kind, _, site) = graph.insts[inst].op
            {
                let origins = carried.entry(value).or_default();
                if !origins.contains(&(kind, site)) {
                    origins.push((kind, site));
                }
            }
        }
    }
    for (_, block) in graph.blocks.iter_mut() {
        block.insts.retain(|&inst| {
            !matches!(
                graph.insts[inst].op,
                Op::Observe(..) | Op::ObserveMethod(..) | Op::ReadHint(..)
            )
        });
    }
    let cfg = Cfg::new(graph);
    let facts = Facts::compute(graph);
    let receivers: FxHashSet<_> = cfg
        .rpo
        .iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|&block| graph.blocks[block].insts.iter())
        .filter_map(|&inst| match graph.insts[inst].op {
            Op::Javascript(
                JsOp::GetIndex(object, _)
                | JsOp::SetIndex(object, _, _)
                | JsOp::TypedIndex(object, _, super::super::TypedRead { element: None, .. }),
            ) => Some(object),
            _ => None,
        })
        .collect();
    let primitive_uses: FxHashSet<_> = cfg
        .rpo
        .iter()
        .flat_map(|&block| graph.blocks[block].insts.iter())
        .filter_map(|&inst| match graph.insts[inst].op {
            Op::Javascript(JsOp::Binary(BinaryOperator::Addition, left, right)) => {
                Some([left, right])
            }
            _ => None,
        })
        .flatten()
        .collect();
    let numeric_uses = numeric_uses(graph, &cfg);
    let mut numbers = Vec::new();
    for block in cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let data = &graph.insts[inst];
            if graph.cold.contains(&block)
                && !matches!(data.op, Op::Javascript(JsOp::Call { target: None, .. }))
            {
                continue;
            }
            let site = Site {
                inline_context: 0,
                source: data
                    .source_name
                    .map_or(specialization.map_or(0, |(_, source)| source), |name| {
                        name.0
                    }),
                start: data.span.start,
                end: data.span.end,
            };
            let mut hints = Vec::new();
            if let Some(value) = data.result() {
                let mut origins = carried.get(&value).cloned().unwrap_or_default();
                for kind in [ReadKind::Number, ReadKind::Primitive] {
                    if !origins.iter().any(|(known, _)| *known == kind) {
                        origins.push((kind, site));
                    }
                }
                for (kind, mut origin) in origins {
                    if origin.source == 0 {
                        origin.source = specialization.map_or(0, |(_, source)| source);
                    }
                    if let Some(&observed) = specialization
                        .and_then(|(repository, _)| repository.read_feedback.get(&(origin, kind)))
                    {
                        let observed = if number_first
                            && matches!(kind, ReadKind::Number | ReadKind::Primitive)
                            && observed == ReadType::Integer
                        {
                            ReadType::Number
                        } else {
                            observed
                        };
                        hints.push((value, kind, origin, observed));
                    }
                }
            }
            let span = data.span;
            let source_name = data.source_name;
            let mut before = None;
            let method = match data.op {
                Op::Javascript(JsOp::Call {
                    callee,
                    receiver: Some(receiver),
                    target,
                    ..
                }) if !graph.method_shapes.get(&inst).is_some_and(Option::is_some)
                    && target.is_none_or(|target| {
                        matches!(target.variant, crate::jit::entries::VariantId::Ordinary(_))
                    })
                    && super::calls::is_method(graph, callee, receiver) =>
                {
                    Some((callee, receiver))
                }
                _ => None,
            };
            match data.op {
                Op::Javascript(
                    JsOp::GetIndex(object, _)
                    | JsOp::SetIndex(object, _, _)
                    | JsOp::TypedIndex(object, _, super::super::TypedRead { element: None, .. }),
                ) => {
                    // OSR inputs lose constructor-based guesses, so keep receiver feedback.
                    before = Some((object, Observation::Read(ReadKind::Receiver)));
                }
                Op::Javascript(
                    JsOp::Call {
                        callee,
                        target: None,
                        ..
                    }
                    | JsOp::CallPack(callee, ..)
                    | JsOp::CallForward(callee, _)
                    | JsOp::Construct { callee, .. }
                    | JsOp::ConstructorCall { callee, .. },
                ) => {
                    before = Some((callee, Observation::Call));
                }
                _ => {}
            }
            let pattern = match &data.op {
                Op::Javascript(JsOp::Call {
                    callee,
                    receiver: Some(receiver),
                    target: None,
                    ..
                }) if matches!(
                    graph.defining_op(*callee),
                    Some(Op::Javascript(JsOp::Get(object, name)))
                        if object == receiver && (name == "exec" || name == "test")
                ) =>
                {
                    Some((*receiver, Observation::Read(ReadKind::Pattern)))
                }
                _ => None,
            };
            let after = if matches!(
                data.op,
                Op::Javascript(
                    JsOp::Get(..) | JsOp::GetIndex(..) | JsOp::ArrayGet(..) | JsOp::ArrayLoad(..)
                )
            ) {
                data.result()
                    .filter(|value| {
                        receivers.contains(value) && facts.of(*value).intersects(ValueType::OBJECT)
                    })
                    .map(|value| (value, Observation::Read(ReadKind::Result)))
            } else {
                None
            };
            let primitive = data
                .result()
                .filter(|value| {
                    primitive_uses.contains(value)
                        && !facts.of(*value).within(ValueType::STRING)
                        && !facts.of(*value).within(ValueType::NUMBER)
                })
                .map(|value| (value, Observation::Read(ReadKind::Primitive)));
            let number = if primitive.is_none()
                && matches!(
                    data.op,
                    Op::Javascript(
                        JsOp::Get(..)
                            | JsOp::GetIndex(..)
                            | JsOp::ArrayGet(..)
                            | JsOp::ArrayLoad(..)
                    )
                ) {
                data.result()
                    .filter(|value| {
                        numeric_uses.contains(value)
                            && facts.of(*value).intersects(ValueType::NUMBER)
                            && !facts.of(*value).within(ValueType::INTEGER)
                    })
                    .map(|value| (value, Observation::Read(ReadKind::Number)))
            } else {
                None
            };
            for (value, kind, origin, observed) in hints {
                let position = graph.blocks[block]
                    .insts
                    .iter()
                    .position(|&candidate| candidate == inst)
                    .unwrap()
                    + 1;
                let (hint, _) = graph.insert(
                    block,
                    position,
                    span,
                    Op::ReadHint(value, kind, observed, origin),
                    false,
                );
                graph.insts[hint].source_name = source_name;
                if matches!(kind, ReadKind::Number | ReadKind::Primitive)
                    && observed != ReadType::Unknown
                {
                    numbers.push((origin, observed));
                }
            }
            if let Some((callee, receiver)) = method {
                before = None;
                if !specialization.is_some_and(|(repository, _)| {
                    repository
                        .method_pairs
                        .get(&site)
                        .is_some_and(|pairs| pairs.len() == crate::jit::profile::CALL_TARGET_LIMIT)
                }) {
                    let position = graph.blocks[block]
                        .insts
                        .iter()
                        .position(|&candidate| candidate == inst)
                        .unwrap();
                    let (observer, _) = graph.insert(
                        block,
                        position,
                        span,
                        Op::ObserveMethod(callee, receiver, site),
                        false,
                    );
                    graph.insts[observer].source_name = source_name;
                }
            }
            for (after, observation) in [
                (false, pattern),
                (false, before),
                (true, after),
                (true, number),
                (true, primitive),
            ] {
                let Some((value, kind)) = observation else {
                    continue;
                };
                let observed = specialization.and_then(|(repository, _)| {
                    let Observation::Read(kind) = kind else {
                        return None;
                    };
                    repository.read_feedback.get(&(site, kind)).copied()
                });
                if observed.is_none()
                    && specialization.is_some_and(|(repository, _)| {
                        if kind == Observation::Call {
                            repository.feedback.get(&site).is_some_and(|targets| {
                                targets.len() == crate::jit::profile::CALL_TARGET_LIMIT
                            })
                        } else {
                            repository.observed(site, kind)
                        }
                    })
                {
                    continue;
                }
                if observed.is_some()
                    && matches!(
                        kind,
                        Observation::Read(ReadKind::Number | ReadKind::Primitive)
                    )
                {
                    continue;
                }
                let position = graph.blocks[block]
                    .insts
                    .iter()
                    .position(|&candidate| candidate == inst)
                    .unwrap()
                    + usize::from(after);
                let (observer, _) = graph.insert(
                    block,
                    position,
                    span,
                    match (kind, observed) {
                        (Observation::Read(kind), Some(observed)) => {
                            Op::ReadHint(value, kind, observed, site)
                        }
                        _ => Op::Observe(value, kind, site),
                    },
                    false,
                );
                graph.insts[observer].source_name = source_name;
            }
        }
    }
    numbers.sort_by_key(|(site, _)| (site.source, site.start, site.end));
    numbers.dedup();
    numbers
}

fn numeric_uses(graph: &Graph, cfg: &Cfg) -> FxHashSet<Value> {
    let mut uses = FxHashSet::default();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let op = &graph.insts[inst].op;
            for (index, value) in op.operands_iter().enumerate() {
                let non_numeric = match op {
                    Op::Javascript(
                        JsOp::Truthy(_)
                        | JsOp::IsNullish(_)
                        | JsOp::IsUndefined(_)
                        | JsOp::Unary(UnaryOp::Not, _),
                    ) => true,
                    Op::Javascript(
                        JsOp::Get(..)
                        | JsOp::Set(..)
                        | JsOp::Define(..)
                        | JsOp::GetIndex(..)
                        | JsOp::SetIndex(..)
                        | JsOp::TypedIndex(..),
                    ) => index == 0,
                    _ => false,
                };
                if !non_numeric {
                    uses.insert(value);
                }
            }
        }
        if let Some(terminator) = &graph.blocks[block].terminator {
            for (index, value) in terminator.kind.operands_iter().enumerate() {
                if !matches!(terminator.kind, TerminatorKind::Branch(..)) || index != 0 {
                    uses.insert(value);
                }
            }
        }
    }
    uses
}

pub(in crate::jit::semantic) fn type_hints(
    graph: &Graph,
) -> FxHashMap<super::super::Value, ReadType> {
    graph
        .blocks
        .iter()
        .flat_map(|(_, block)| block.insts.iter())
        .filter_map(|&inst| match graph.insts[inst].op {
            Op::ReadHint(value, ReadKind::Number | ReadKind::Primitive, observed, _) => {
                Some((value, observed))
            }
            _ => None,
        })
        .collect()
}

pub(in crate::jit::semantic) fn numeric_site(
    graph: &Graph,
    value: super::super::Value,
) -> Option<Site> {
    if let Some(site) = primitive_site(graph, value) {
        return Some(site);
    }
    for (_, block) in graph.blocks.iter() {
        for &inst in &block.insts {
            if let Op::ReadHint(candidate, ReadKind::Number, _, site)
            | Op::Observe(candidate, Observation::Read(ReadKind::Number), site) =
                graph.insts[inst].op
                && candidate == value
            {
                return Some(site);
            }
        }
    }
    let super::super::ValueDef::Result(inst, _) = graph.values[value] else {
        return None;
    };
    let data = &graph.insts[inst];
    if let Op::Javascript(JsOp::AsNumber(input) | JsOp::AsInteger(input) | JsOp::AsString(input)) =
        data.op
    {
        return numeric_site(graph, input);
    }
    Some(Site {
        inline_context: 0,
        source: data.source_name.map_or(0, |name| name.0),
        start: data.span.start,
        end: data.span.end,
    })
}

pub(in crate::jit::semantic) fn read_hints(
    graph: &Graph,
) -> FxHashMap<super::super::Value, ReadType> {
    graph
        .blocks
        .iter()
        .flat_map(|(_, block)| block.insts.iter())
        .filter_map(|&inst| match graph.insts[inst].op {
            Op::ReadHint(value, _, observed @ (ReadType::Array | ReadType::Typed(_)), _) => {
                Some((value, observed))
            }
            _ => None,
        })
        .collect()
}

pub(in crate::jit::semantic) fn primitive_site(graph: &Graph, value: Value) -> Option<Site> {
    for (_, block) in graph.blocks.iter() {
        for &inst in &block.insts {
            if let Op::ReadHint(candidate, ReadKind::Primitive, _, site)
            | Op::Observe(candidate, Observation::Read(ReadKind::Primitive), site) =
                graph.insts[inst].op
                && candidate == value
            {
                return Some(site);
            }
        }
    }
    None
}
