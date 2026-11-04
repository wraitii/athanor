use std::collections::HashMap;

use super::specialize::Inline;
use super::{BlockData, Cfg, Constant, Graph, Op, Target, Terminator, TerminatorKind};

pub(super) fn apply(graph: &mut Graph, body: &Graph, plan: &Inline) {
    let call = graph.insts[plan.call].clone();
    let Op::Call { callee, args, .. } = &call.op else {
        unreachable!();
    };
    let block = call.block;
    let position = graph.blocks[block]
        .insts
        .iter()
        .position(|&inst| inst == plan.call)
        .unwrap();
    let continuation = graph.blocks.push(BlockData::default());
    let result = graph.add_param(continuation);
    let original = call.result.unwrap();
    graph.replace_uses(&HashMap::from([(original, result)]));
    let tail = graph.blocks[block].insts.split_off(position + 1);
    graph.blocks[continuation].insts = tail;
    for &inst in &graph.blocks[continuation].insts {
        graph.insts[inst].block = continuation;
    }
    graph.blocks[continuation].terminator = graph.blocks[block].terminator.take();
    let fallback = graph.blocks.push(BlockData::default());
    graph.move_inst(plan.call, fallback);
    graph.blocks[fallback].terminator = Some(Terminator {
        span: call.span,
        kind: TerminatorKind::Jump(Target {
            block: continuation,
            args: vec![original],
        }),
    });
    graph.cold.insert(fallback);

    let check = graph
        .append(block, call.span, Op::IsCode(*callee, plan.target), true)
        .1
        .unwrap();
    let mut current = graph.blocks.push(BlockData::default());
    graph.blocks[block].terminator = Some(Terminator {
        span: call.span,
        kind: TerminatorKind::Branch(check, Target::new(current), Target::new(fallback)),
    });
    for &index in &plan.numbers {
        let value = args[index];
        let check = graph
            .append(current, call.span, Op::IsNumber(value), true)
            .1
            .unwrap();
        let next = graph.blocks.push(BlockData::default());
        graph.blocks[current].terminator = Some(Terminator {
            span: call.span,
            kind: TerminatorKind::Branch(check, Target::new(next), Target::new(fallback)),
        });
        current = next;
    }
    let mut narrowed = HashMap::new();
    for &index in &plan.numbers {
        let value = args[index];
        let number = graph
            .append(current, call.span, Op::AsNumber(value), true)
            .1
            .unwrap();
        narrowed.insert(value, number);
    }
    let undefined = graph
        .append(current, call.span, Op::Constant(Constant::Undefined), true)
        .1
        .unwrap();
    let mut blocks = HashMap::new();
    let mut values = HashMap::new();
    let mut insts = Vec::new();
    let reachable = Cfg::new(body).rpo;
    for &block in &reachable {
        let copy = graph.blocks.push(BlockData::default());
        blocks.insert(block, copy);
        for &param in &body.blocks[block].params {
            values.insert(param, graph.add_param(copy));
        }
    }
    for &block in &reachable {
        for &inst in &body.blocks[block].insts {
            let data = &body.insts[inst];
            if let Op::Argument(index) = data.op {
                let value = args.get(index).copied().unwrap_or(undefined);
                values.insert(
                    data.result.unwrap(),
                    narrowed.get(&value).copied().unwrap_or(value),
                );
                continue;
            }
            let (copy, result) = graph.append(
                blocks[&block],
                data.span,
                data.op.clone(),
                data.result.is_some(),
            );
            if let Some(value) = data.result {
                values.insert(value, result.unwrap());
            }
            insts.push(copy);
        }
    }
    for inst in insts {
        for value in graph.insts[inst].op.operands_mut() {
            *value = values[value];
        }
    }
    for &block in &reachable {
        let mut terminator = body.blocks[block].terminator.clone().unwrap();
        for value in terminator.kind.operands_mut() {
            *value = values[value];
        }
        if let TerminatorKind::Return(value) = terminator.kind {
            terminator.kind = TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![value],
            });
        } else {
            for target in terminator.kind.targets_mut() {
                target.block = blocks[&target.block];
            }
        }
        graph.blocks[blocks[&block]].terminator = Some(terminator);
    }
    graph.blocks[current].terminator = Some(Terminator {
        span: call.span,
        kind: TerminatorKind::Jump(Target::new(blocks[&body.entry])),
    });
}
