use crate::FxHashMap;
use std::collections::BTreeMap;

use super::super::dispatch::{self, Alternative};
use super::super::guards::Fact;
use super::super::{Cfg, DirectTarget, Graph, Op, Value, ValueType};
use super::calls::Repository;
use crate::jit::profile::Site;
use crate::jit::variants::{Fact as InputFact, Input, Requirements};

fn input(graph: &Graph, value: Value) -> Option<Input> {
    match graph.defining_op(value)? {
        Op::This => Some(Input::This),
        Op::Argument(index) => Some(Input::Argument(*index)),
        _ => None,
    }
}

pub(in crate::jit::semantic) fn requirements(
    function: super::super::FunctionState,
    bodies: &Repository,
    source: usize,
) -> super::super::Result<Requirements> {
    contextual_requirements(function, bodies, source, &Vec::new())
}

pub(in crate::jit::semantic) fn contextual_requirements(
    mut function: super::super::FunctionState,
    bodies: &Repository,
    source: usize,
    offered: &Requirements,
) -> super::super::Result<Requirements> {
    assume(&mut function.graph, offered);
    let planned = super::super::pipeline::inline_and_plan_inner(
        function,
        true,
        Some((bodies, source)),
        0,
        false,
    )?;
    let mut optimized = super::super::specialize(
        planned,
        super::super::Versioning::Speculate {
            fallback: None,
            continuations: true,
        },
        false,
        None,
    )?;
    optimized.rewrite_properties()?;
    Ok(optimized
        .graph
        .entry_guards
        .into_iter()
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .take(4)
        .collect())
}

fn shape_hints(graph: &Graph, bodies: &Repository, source: usize) -> FxHashMap<Value, u32> {
    // These are candidates, not live facts; dispatch guards them at the call.
    let mut hints = FxHashMap::default();
    for block in Cfg::new(graph).rpo {
        if graph.cold.contains(&block) {
            continue;
        }
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let (object, property) = match data.op {
                Op::EntryAssume(value, Fact::Shape(shape)) | Op::IsShape(value, shape) => {
                    hints.entry(value).or_insert(shape);
                    continue;
                }
                Op::Get(object, _) | Op::ProbeProperty(object, _) => {
                    (object, crate::jit::profile::PropertyAccess::Get)
                }
                Op::Set(object, ..) => (object, crate::jit::profile::PropertyAccess::Set),
                _ => continue,
            };
            let site = Site {
                inline_context: data.inline_context,
                source: data.source_name.map_or(source, |name| name.0),
                start: data.span.start,
                end: data.span.end,
            };
            let shape = match bodies.shape_feedback(site, property) {
                Some(crate::jit::profile::ShapeAccess::Read(layout)) => layout.shape,
                Some(crate::jit::profile::ShapeAccess::Overwrite { shape, .. }) => shape,
                _ => continue,
            };
            hints.entry(object).or_insert(shape);
        }
    }
    hints
}

pub(in crate::jit) fn assume(graph: &mut Graph, requirements: &Requirements) {
    for (index, &(input, fact)) in requirements.iter().enumerate() {
        let op = match input {
            Input::This => Op::This,
            Input::Argument(index) => Op::Argument(index),
        };
        let (_, value) = graph.insert(graph.entry, index * 2, graph.span, op, true);
        let value = value.unwrap();
        let aliases: FxHashMap<_, _> = graph
            .insts
            .values()
            .filter_map(|data| {
                let result = data.result()?;
                (result != value && self::input(graph, result) == Some(input))
                    .then_some((result, value))
            })
            .collect();
        graph.replace_uses(&aliases);
        let fact = match fact {
            InputFact::Number => Fact::Type(ValueType::NUMBER),
            InputFact::Integer => Fact::Type(ValueType::INTEGER),
            InputFact::Shape(shape) => Fact::Shape(shape),
        };
        graph.insert(
            graph.entry,
            index * 2 + 1,
            graph.span,
            Op::EntryAssume(value, fact),
            true,
        );
    }
}

pub(in crate::jit::semantic) fn apply(
    graph: &mut Graph,
    bodies: &Repository,
    source: usize,
) -> super::super::Result<Vec<(Site, crate::jit::entries::VariantId)>> {
    if !bodies.call_variants {
        return Ok(Vec::new());
    }
    let mut selected = Vec::new();
    let mut contracts: FxHashMap<_, Requirements> = FxHashMap::default();
    let hints = if bodies.call_specialization {
        shape_hints(graph, bodies, source)
    } else {
        FxHashMap::default()
    };
    let calls: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .collect();
    for inst in calls {
        let data = graph.insts[inst].clone();
        let Op::Call {
            callee,
            receiver,
            ref args,
            target,
        } = data.op
        else {
            continue;
        };
        let site = Site {
            inline_context: 0,
            source: data.source_name.map_or(source, |name| name.0),
            start: data.span.start,
            end: data.span.end,
        };
        let mut targets = bodies.feedback.get(&site).cloned().unwrap_or_default();
        if let Some(target) = target
            && !targets.contains(&target.code)
        {
            targets.insert(0, target.code);
        }
        if graph.cold.contains(&data.block) && targets.len() < 2 {
            continue;
        }
        let mut alternatives = Vec::new();
        let offered: Requirements = receiver
            .map(|value| (Input::This, value))
            .into_iter()
            .chain(
                args.iter()
                    .enumerate()
                    .map(|(index, &value)| (Input::Argument(index), value)),
            )
            .filter_map(|(input, value)| {
                hints
                    .get(&value)
                    .map(|&shape| (input, InputFact::Shape(shape)))
            })
            .take(4)
            .collect();
        for code in targets {
            let Some((callee_source, body)) = bodies.emission(code) else {
                continue;
            };
            let Some(arity) = body.graph.arity else {
                continue;
            };
            if body.graph.has_handlers || body.graph.generator {
                continue;
            }
            let lexical_this = bodies.lexical_this(code);
            if lexical_this || args.len() != usize::from(arity) {
                continue;
            }
            let mut target = DirectTarget {
                code,
                variant: crate::jit::entries::VariantId::ordinary(code.as_u32() as u64),
                arity,
                lexical_this,
            };
            let requirements = if !bodies.call_specialization {
                Vec::new()
            } else if let Some(requirements) = contracts.get(&(code, offered.clone())) {
                requirements.clone()
            } else {
                let requirements = if offered.is_empty() {
                    requirements(body, bodies, callee_source)?
                } else {
                    contextual_requirements(body, bodies, callee_source, &offered)?
                };
                contracts.insert((code, offered.clone()), requirements.clone());
                requirements
            };
            let applicable = requirements.iter().all(|(input, _)| match input {
                Input::This => receiver.is_some(),
                Input::Argument(index) => *index < args.len(),
            });
            if applicable
                && let Some(variant) = bodies.variants.intern(code.as_u32() as u64, requirements)
            {
                target.variant = variant.id();
                let mut requirements = vec![(callee, Fact::Code(code))];
                for &(input, fact) in &variant.requirements {
                    let value = match input {
                        Input::This => receiver.unwrap(),
                        Input::Argument(index) => args[index],
                    };
                    let fact = match fact {
                        InputFact::Number => Fact::Type(ValueType::NUMBER),
                        InputFact::Integer => Fact::Type(ValueType::INTEGER),
                        InputFact::Shape(shape) => Fact::Shape(shape),
                    };
                    requirements.push((value, fact));
                }
                selected.push((site, variant.id()));
                alternatives.push(Alternative {
                    requirements,
                    implementation: target,
                });
            }
            target.variant = crate::jit::entries::VariantId::ordinary(code.as_u32() as u64);
            selected.push((site, target.variant));
            alternatives.push(Alternative {
                requirements: vec![(callee, Fact::Code(code))],
                implementation: target,
            });
        }
        if alternatives.is_empty() {
            continue;
        }
        if let Op::Call { target, .. } = &mut graph.insts[inst].op {
            *target = None;
        }
        dispatch::apply(graph, inst, alternatives, |graph, block, target| {
            let (call, result) = graph.append(
                block,
                data.span,
                Op::Call {
                    callee,
                    receiver,
                    args: args.clone(),
                    target: Some(target),
                },
                true,
            );
            graph.insts[call].source_name = data.source_name;
            graph.insts[call].inline_context = data.inline_context;
            result
        });
    }
    graph.verify()?;
    Ok(selected)
}
