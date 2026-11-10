use std::collections::HashSet;

use oxc_syntax::operator::BinaryOperator;

use super::{
    ArrayMethod, BlockData, Cfg, Constant, Facts, Graph, Inst, Op, Result, Target, Terminator,
    TerminatorKind, Types,
};

impl ArrayMethod {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Map => "map",
            Self::Some => "some",
            Self::Every => "every",
            Self::Filter => "filter",
            Self::ForEach => "forEach",
            Self::Find => "find",
            Self::Push => "push",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "map" => Some(Self::Map),
            "some" => Some(Self::Some),
            "every" => Some(Self::Every),
            "filter" => Some(Self::Filter),
            "forEach" => Some(Self::ForEach),
            "find" => Some(Self::Find),
            _ => None,
        }
    }
}

pub(super) fn expand(
    graph: &mut Graph,
    _source: usize,
    notes: &mut Vec<String>,
    expanded: &mut HashSet<Inst>,
) -> Result<()> {
    let facts = Facts::compute(graph);
    let mut sites = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            if expanded.contains(&inst) {
                continue;
            }
            let Op::Call {
                callee,
                receiver: Some(_),
                args,
                ..
            } = &graph.insts[inst].op
            else {
                continue;
            };
            let Some(Op::Get(_, name)) = graph.defining_op(*callee) else {
                continue;
            };
            let Some(method) = ArrayMethod::from_name(name) else {
                continue;
            };
            let Some(&callback) = args.first() else {
                continue;
            };
            if args
                .get(1)
                .is_some_and(|&value| !facts.of(value).within(Types::UNDEFINED))
            {
                continue;
            }
            if !matches!(
                graph.defining_op(callback),
                Some(Op::Closure(_, _, _, false, _, _, _))
            ) {
                continue;
            }
            sites.push((inst, method));
        }
    }
    for (inst, method) in sites {
        expanded.insert(inst);
        iteration(graph, inst, method);
        graph.verify()?;
        let span = graph.insts[inst].span;
        notes.push(format!(
            "array {} at {}..{}: expose callback loop behind native and array checks",
            method.name(),
            span.start,
            span.end
        ));
    }
    Ok(())
}

fn iteration(graph: &mut Graph, inst: Inst, method: ArrayMethod) {
    let call = graph.insts[inst].clone();
    let first_new = graph.insts.len();
    let Op::Call {
        callee,
        receiver: Some(receiver),
        args,
        ..
    } = call.op
    else {
        unreachable!();
    };
    let callback = args[0];
    let (block, continuation, fallback) = super::inlining::split_call(graph, inst);
    let start = graph.blocks.push(BlockData::default());
    let header = graph.blocks.push(BlockData::default());
    let read = graph.blocks.push(BlockData::default());
    let invoke = graph.blocks.push(BlockData::default());
    let advance = graph.blocks.push(BlockData::default());
    let exit = graph.blocks.push(BlockData::default());
    let span = call.span;
    let check = graph.append_value(block, span, Op::IsArrayIteration(callee, receiver, method));
    graph.blocks[block].terminator = Some(Terminator {
        source_name: call.source_name,
        span,
        kind: TerminatorKind::Branch(check, Target::new(start), Target::new(fallback)),
    });
    let length = graph.append_value(start, span, Op::IndexedLength(receiver));
    let result_op = match method {
        ArrayMethod::Map => Op::NewHoleyArray(length),
        ArrayMethod::Filter => Op::NewArray(Vec::new()),
        ArrayMethod::Some | ArrayMethod::Every => {
            Op::Constant(Constant::Boolean(method == ArrayMethod::Every))
        }
        ArrayMethod::ForEach | ArrayMethod::Find => Op::Constant(Constant::Undefined),
        ArrayMethod::Push => unreachable!(),
    };
    let result = graph.append_value(start, span, result_op);
    let zero = graph.append_value(start, span, Op::Constant(Constant::Number(0.0)));
    let undefined = graph.append_value(start, span, Op::Constant(Constant::Undefined));
    let index = graph.add_param(header);
    graph.blocks[start].terminator = Some(Terminator {
        source_name: call.source_name,
        span,
        kind: TerminatorKind::Jump(Target {
            block: header,
            args: vec![zero],
        }),
    });
    let more = graph.append_value(
        header,
        span,
        Op::Binary(BinaryOperator::LessThan, index, length),
    );
    graph.blocks[header].terminator = Some(Terminator {
        source_name: call.source_name,
        span,
        kind: TerminatorKind::Branch(more, Target::new(read), Target::new(exit)),
    });
    let element = graph.append_value(read, span, Op::MapElement(receiver, index));
    let absent = graph.append_value(read, span, Op::IsUninitialized(element));
    let value = graph.add_param(invoke);
    let hole = match method {
        ArrayMethod::Find => Target {
            block: invoke,
            args: vec![undefined],
        },
        _ => Target::new(advance),
    };
    let present = Target {
        block: invoke,
        args: vec![element],
    };
    graph.blocks[read].terminator = Some(Terminator {
        source_name: call.source_name,
        span,
        kind: TerminatorKind::Branch(absent, hole, present),
    });
    let mapped = graph.append_value(
        invoke,
        span,
        Op::Call {
            callee: callback,
            receiver: Some(undefined),
            args: vec![value, index, receiver],
            target: None,
        },
    );
    match method {
        ArrayMethod::Push => unreachable!(),
        ArrayMethod::ForEach => {
            graph.blocks[invoke].terminator = Some(Terminator {
                source_name: call.source_name,
                span,
                kind: TerminatorKind::Jump(Target::new(advance)),
            });
        }
        ArrayMethod::Find => {
            let found = graph.blocks.push(BlockData::default());
            graph.blocks[invoke].terminator = Some(Terminator {
                source_name: call.source_name,
                span,
                kind: TerminatorKind::Branch(mapped, Target::new(found), Target::new(advance)),
            });
            graph.blocks[found].terminator = Some(Terminator {
                source_name: call.source_name,
                span,
                kind: TerminatorKind::Jump(Target {
                    block: continuation,
                    args: vec![value],
                }),
            });
        }
        ArrayMethod::Map => {
            graph.append(invoke, span, Op::SetIndex(result, index, mapped), false);
            graph.blocks[invoke].terminator = Some(Terminator {
                source_name: call.source_name,
                span,
                kind: TerminatorKind::Jump(Target::new(advance)),
            });
        }
        ArrayMethod::Filter => {
            let keep = graph.blocks.push(BlockData::default());
            graph.blocks[invoke].terminator = Some(Terminator {
                source_name: call.source_name,
                span,
                kind: TerminatorKind::Branch(mapped, Target::new(keep), Target::new(advance)),
            });
            graph.append(keep, span, Op::ArrayAppend(result, value, false), false);
            graph.blocks[keep].terminator = Some(Terminator {
                source_name: call.source_name,
                span,
                kind: TerminatorKind::Jump(Target::new(advance)),
            });
        }
        ArrayMethod::Some | ArrayMethod::Every => {
            let stop = graph.blocks.push(BlockData::default());
            let targets = if method == ArrayMethod::Some {
                (Target::new(stop), Target::new(advance))
            } else {
                (Target::new(advance), Target::new(stop))
            };
            graph.blocks[invoke].terminator = Some(Terminator {
                source_name: call.source_name,
                span,
                kind: TerminatorKind::Branch(mapped, targets.0, targets.1),
            });
            let found = graph.append_value(
                stop,
                span,
                Op::Constant(Constant::Boolean(method == ArrayMethod::Some)),
            );
            graph.blocks[stop].terminator = Some(Terminator {
                source_name: call.source_name,
                span,
                kind: TerminatorKind::Jump(Target {
                    block: continuation,
                    args: vec![found],
                }),
            });
        }
    }
    let one = graph.append_value(advance, span, Op::Constant(Constant::Number(1.0)));
    let next = graph.append_value(
        advance,
        span,
        Op::Binary(BinaryOperator::Addition, index, one),
    );
    graph.blocks[advance].terminator = Some(Terminator {
        source_name: call.source_name,
        span,
        kind: TerminatorKind::Jump(Target {
            block: header,
            args: vec![next],
        }),
    });
    graph.blocks[exit].terminator = Some(Terminator {
        source_name: call.source_name,
        span,
        kind: TerminatorKind::Jump(Target {
            block: continuation,
            args: vec![result],
        }),
    });
    for data in graph.insts.values_mut().skip(first_new) {
        data.source_name = call.source_name;
    }
}

pub(super) fn push_calls(
    graph: &mut Graph,
    bodies: &super::specialize::Repository,
    source: usize,
    notes: &mut Vec<String>,
) -> Result<Vec<crate::jit::profile::Site>> {
    let mut sites = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let Op::Call {
                callee,
                receiver: Some(receiver),
                ..
            } = data.op
            else {
                continue;
            };
            let site = crate::jit::profile::Site {
                source: data.source_name.map_or(source, |name| name.0),
                start: data.span.start,
                end: data.span.end,
            };
            // Array named methods are fixed by Heap::property, including after escape.
            let known_array = matches!(
                graph.defining_op(receiver),
                Some(Op::NewArray(_) | Op::NewHoleyArray(_))
            ) && matches!(graph.defining_op(callee), Some(Op::Get(object, name)) if *object == receiver && name == "push");
            if known_array
                || bodies.native_feedback.get(&site)
                    == Some(&crate::heap::NativeFunction::ArrayPush)
            {
                sites.push((inst, site, known_array));
            }
        }
    }
    for &(inst, site, known_array) in &sites {
        let call = graph.insts[inst].clone();
        let first_new = graph.insts.len();
        let Op::Call {
            callee,
            receiver: Some(receiver),
            args,
            ..
        } = call.op
        else {
            unreachable!()
        };
        let (block, continuation, fallback) = super::inlining::split_call(graph, inst);
        let fast = graph.blocks.push(BlockData::default());
        let kind = if known_array {
            TerminatorKind::Jump(Target::new(fast))
        } else {
            let check = graph.append_value(
                block,
                call.span,
                Op::IsArrayIteration(callee, receiver, ArrayMethod::Push),
            );
            TerminatorKind::Branch(check, Target::new(fast), Target::new(fallback))
        };
        graph.blocks[block].terminator = Some(Terminator {
            source_name: call.source_name,
            span: call.span,
            kind,
        });
        for value in args {
            graph.append(
                fast,
                call.span,
                Op::ArrayAppend(receiver, value, false),
                false,
            );
        }
        let length = graph.append_value(fast, call.span, Op::IndexedLength(receiver));
        graph.blocks[fast].terminator = Some(Terminator {
            source_name: call.source_name,
            span: call.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![length],
            }),
        });
        for data in graph.insts.values_mut().skip(first_new) {
            data.source_name = call.source_name;
        }
        notes.push(if known_array {
            format!("known array at {site}: lower Array.push without guards")
        } else {
            format!(
                "native feedback at {site}: specialize Array.push behind native and array checks"
            )
        });
        graph.verify()?;
    }
    Ok(sites
        .into_iter()
        .filter_map(|(_, site, known)| (!known).then_some(site))
        .collect())
}
