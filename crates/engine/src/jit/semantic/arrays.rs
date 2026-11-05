use oxc_syntax::operator::BinaryOperator;

use super::specialize::{self, Repository};
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
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "map" => Some(Self::Map),
            "some" => Some(Self::Some),
            "every" => Some(Self::Every),
            "filter" => Some(Self::Filter),
            _ => None,
        }
    }
}

pub(super) fn expand(
    graph: &mut Graph,
    bodies: &Repository,
    source: usize,
    notes: &mut Vec<String>,
) -> Result<()> {
    let facts = Facts::compute(graph);
    let mut sites = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            let Op::Call {
                callee,
                receiver: Some(_),
                args,
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
            let Some(Op::Closure(span, _, false, _)) = graph.defining_op(callback) else {
                continue;
            };
            let Some(code) = bodies.closure(source, span.start) else {
                continue;
            };
            if !bodies.body(code).is_some_and(specialize::eligible) {
                continue;
            }
            sites.push((inst, method));
        }
    }
    for (inst, method) in sites {
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
    let Op::Call {
        callee,
        receiver: Some(receiver),
        args,
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
    let check = graph
        .append(
            block,
            span,
            Op::IsArrayIteration(callee, receiver, method),
            true,
        )
        .1
        .unwrap();
    graph.blocks[block].terminator = Some(Terminator {
        span,
        kind: TerminatorKind::Branch(check, Target::new(start), Target::new(fallback)),
    });
    let length = graph
        .append(start, span, Op::IndexedLength(receiver), true)
        .1
        .unwrap();
    let result_op = match method {
        ArrayMethod::Map => Op::NewHoleyArray(length),
        ArrayMethod::Filter => Op::NewArray(Vec::new()),
        ArrayMethod::Some | ArrayMethod::Every => {
            Op::Constant(Constant::Boolean(method == ArrayMethod::Every))
        }
    };
    let result = graph.append(start, span, result_op, true).1.unwrap();
    let zero = graph
        .append(start, span, Op::Constant(Constant::Number(0.0)), true)
        .1
        .unwrap();
    let undefined = graph
        .append(start, span, Op::Constant(Constant::Undefined), true)
        .1
        .unwrap();
    let index = graph.add_param(header);
    graph.blocks[start].terminator = Some(Terminator {
        span,
        kind: TerminatorKind::Jump(Target {
            block: header,
            args: vec![zero],
        }),
    });
    let more = graph
        .append(
            header,
            span,
            Op::Binary(BinaryOperator::LessThan, index, length),
            true,
        )
        .1
        .unwrap();
    graph.blocks[header].terminator = Some(Terminator {
        span,
        kind: TerminatorKind::Branch(more, Target::new(read), Target::new(exit)),
    });
    let element = graph
        .append(read, span, Op::MapElement(receiver, index), true)
        .1
        .unwrap();
    let absent = graph
        .append(read, span, Op::IsUninitialized(element), true)
        .1
        .unwrap();
    graph.blocks[read].terminator = Some(Terminator {
        span,
        kind: TerminatorKind::Branch(absent, Target::new(advance), Target::new(invoke)),
    });
    let mapped = graph
        .append(
            invoke,
            span,
            Op::Call {
                callee: callback,
                receiver: Some(undefined),
                args: vec![element, index, receiver],
            },
            true,
        )
        .1
        .unwrap();
    match method {
        ArrayMethod::Map => {
            graph.append(invoke, span, Op::SetIndex(result, index, mapped), false);
            graph.blocks[invoke].terminator = Some(Terminator {
                span,
                kind: TerminatorKind::Jump(Target::new(advance)),
            });
        }
        ArrayMethod::Filter => {
            let keep = graph.blocks.push(BlockData::default());
            graph.blocks[invoke].terminator = Some(Terminator {
                span,
                kind: TerminatorKind::Branch(mapped, Target::new(keep), Target::new(advance)),
            });
            graph.append(keep, span, Op::ArrayAppend(result, element, false), false);
            graph.blocks[keep].terminator = Some(Terminator {
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
                span,
                kind: TerminatorKind::Branch(mapped, targets.0, targets.1),
            });
            let found = graph
                .append(
                    stop,
                    span,
                    Op::Constant(Constant::Boolean(method == ArrayMethod::Some)),
                    true,
                )
                .1
                .unwrap();
            graph.blocks[stop].terminator = Some(Terminator {
                span,
                kind: TerminatorKind::Jump(Target {
                    block: continuation,
                    args: vec![found],
                }),
            });
        }
    }
    let one = graph
        .append(advance, span, Op::Constant(Constant::Number(1.0)), true)
        .1
        .unwrap();
    let next = graph
        .append(
            advance,
            span,
            Op::Binary(BinaryOperator::Addition, index, one),
            true,
        )
        .1
        .unwrap();
    graph.blocks[advance].terminator = Some(Terminator {
        span,
        kind: TerminatorKind::Jump(Target {
            block: header,
            args: vec![next],
        }),
    });
    graph.blocks[exit].terminator = Some(Terminator {
        span,
        kind: TerminatorKind::Jump(Target {
            block: continuation,
            args: vec![result],
        }),
    });
}
