use crate::{FxHashMap, FxHashSet};

use crate::jit::profile::{Observation, ReadKind, ReadType, Site};

use super::{Cfg, Facts, Graph, Op, ValueType, specialize::Repository};

pub(super) fn plan(
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
        block
            .insts
            .retain(|&inst| !matches!(graph.insts[inst].op, Op::Observe(..) | Op::ReadHint(..)));
    }
    let cfg = Cfg::new(graph);
    let facts = Facts::compute(graph);
    let receivers: FxHashSet<_> = cfg
        .rpo
        .iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|&block| graph.blocks[block].insts.iter())
        .filter_map(|&inst| match graph.insts[inst].op {
            Op::GetIndex(object, _)
            | Op::SetIndex(object, _, _)
            | Op::TypedIndex(object, _, super::TypedRead { element: None, .. }) => Some(object),
            _ => None,
        })
        .collect();
    let mut numbers = Vec::new();
    for block in cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let data = &graph.insts[inst];
            if graph.cold.contains(&block) && !matches!(data.op, Op::Call { target: None, .. }) {
                continue;
            }
            let site = Site {
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
                if !origins.iter().any(|(kind, _)| *kind == ReadKind::Number) {
                    origins.push((ReadKind::Number, site));
                }
                for (kind, mut origin) in origins {
                    if origin.source == 0 {
                        origin.source = specialization.map_or(0, |(_, source)| source);
                    }
                    if let Some(&observed) = specialization
                        .and_then(|(repository, _)| repository.read_feedback.get(&(origin, kind)))
                    {
                        let observed = if number_first
                            && kind == ReadKind::Number
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
            match data.op {
                Op::GetIndex(object, _)
                | Op::SetIndex(object, _, _)
                | Op::TypedIndex(object, _, super::TypedRead { element: None, .. })
                    if super::optimize::element_guess(graph, object).is_none() =>
                {
                    before = Some((object, Observation::Read(ReadKind::Receiver)));
                }
                Op::Call {
                    callee,
                    target: None,
                    ..
                }
                | Op::Construct { callee, .. }
                | Op::ConstructorCall { callee, .. } => {
                    before = Some((callee, Observation::Call));
                }
                _ => {}
            }
            let after = if matches!(
                data.op,
                Op::Get(..) | Op::GetIndex(..) | Op::ArrayGet(..) | Op::ArrayLoad(..)
            ) {
                data.result()
                    .filter(|value| {
                        receivers.contains(value) && facts.of(*value).intersects(ValueType::OBJECT)
                    })
                    .map(|value| (value, Observation::Read(ReadKind::Result)))
            } else {
                None
            };
            let number = if matches!(
                data.op,
                Op::Get(..) | Op::GetIndex(..) | Op::ArrayGet(..) | Op::ArrayLoad(..)
            ) {
                data.result()
                    .filter(|value| {
                        facts.of(*value).intersects(ValueType::NUMBER)
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
                if kind == ReadKind::Number && observed != ReadType::Unknown {
                    numbers.push((origin, observed));
                }
            }
            for (after, observation) in [(false, before), (true, after), (true, number)] {
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
                if observed.is_some() && kind == Observation::Read(ReadKind::Number) {
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

pub(super) fn numeric_hints(graph: &Graph) -> FxHashMap<super::Value, ReadType> {
    graph
        .blocks
        .iter()
        .flat_map(|(_, block)| block.insts.iter())
        .filter_map(|&inst| match graph.insts[inst].op {
            Op::ReadHint(value, ReadKind::Number, observed, _) => Some((value, observed)),
            _ => None,
        })
        .collect()
}

pub(super) fn numeric_site(graph: &Graph, value: super::Value) -> Option<Site> {
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
    let super::ValueDef::Result(inst, _) = graph.values[value] else {
        return None;
    };
    let data = &graph.insts[inst];
    if let Op::AsNumber(input) | Op::AsInteger(input) = data.op {
        return numeric_site(graph, input);
    }
    Some(Site {
        source: data.source_name.map_or(0, |name| name.0),
        start: data.span.start,
        end: data.span.end,
    })
}

pub(super) fn read_hints(graph: &Graph) -> FxHashMap<super::Value, ReadType> {
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
