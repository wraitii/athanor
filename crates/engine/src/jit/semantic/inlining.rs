use std::collections::{HashMap, HashSet};

use super::specialize::Inline;
use super::{
    Block, BlockData, Cfg, Constant, Graph, Inst, Op, Scope, ScopeKind, Target, Terminator,
    TerminatorKind,
};

pub(super) fn split_call(graph: &mut Graph, inst: Inst) -> (Block, Block, Block) {
    let call = graph.insts[inst].clone();
    let block = call.block;
    let position = graph.blocks[block]
        .insts
        .iter()
        .position(|&current| current == inst)
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
    graph.move_inst(inst, fallback);
    graph.blocks[fallback].terminator = Some(Terminator {
        span: call.span,
        kind: TerminatorKind::Jump(Target {
            block: continuation,
            args: vec![original],
        }),
    });
    graph.cold.insert(fallback);
    (block, continuation, fallback)
}

fn scope_at(graph: &Graph, inst: Inst) -> Option<Scope> {
    let mut pending = vec![(graph.entry, Vec::new())];
    let mut visited = HashSet::new();
    while let Some((block, mut scopes)) = pending.pop() {
        if !visited.insert(block) {
            continue;
        }
        for &current in &graph.blocks[block].insts {
            if current == inst {
                return scopes.last().copied();
            }
            match graph.insts[current].op {
                Op::EnterScope(scope) | Op::EnterCaptures(scope, _) => scopes.push(scope),
                Op::LeaveScope(_) => {
                    scopes.pop();
                }
                _ => {}
            }
        }
        for successor in graph.successors(block) {
            pending.push((successor, scopes.clone()));
        }
    }
    unreachable!()
}

pub(super) fn apply(graph: &mut Graph, body: &Graph, plan: &Inline) {
    let call = graph.insts[plan.call].clone();
    let (callee, receiver, args) = match &call.op {
        Op::Call {
            callee,
            receiver,
            args,
        } => (callee, *receiver, args),
        Op::ConstructorCall {
            callee,
            receiver,
            args,
        } => (callee, Some(*receiver), args),
        _ => unreachable!(),
    };
    let parent = scope_at(graph, plan.call);
    let (block, continuation, fallback) = split_call(graph, plan.call);
    let mut current = graph.blocks.push(BlockData::default());
    graph.blocks[block].terminator = Some(Terminator {
        span: call.span,
        kind: if plan.guard_code {
            let check = graph
                .append(block, call.span, Op::IsCode(*callee, plan.target), true)
                .1
                .unwrap();
            TerminatorKind::Branch(check, Target::new(current), Target::new(fallback))
        } else {
            TerminatorKind::Jump(Target::new(current))
        },
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
    let uses_this = Cfg::new(body).rpo.iter().any(|&block| {
        body.blocks[block]
            .insts
            .iter()
            .any(|&inst| matches!(body.insts[inst].op, Op::This))
    });
    let receiver = if plan.arrow && uses_this {
        let op = if plan.guard_code {
            Op::ClosureThis(*callee)
        } else {
            Op::This
        };
        graph.append(current, call.span, op, true).1.unwrap()
    } else {
        receiver.unwrap_or(undefined)
    };
    let mut scopes = HashMap::new();
    let mut bindings = HashMap::new();
    for (scope, data) in &body.scopes {
        scopes.insert(scope, graph.scopes.push(data.clone()));
    }
    for (binding, data) in &body.bindings {
        let mut data = data.clone();
        data.scope = scopes[&data.scope];
        bindings.insert(binding, graph.bindings.push(data));
    }
    for (scope, data) in &body.scopes {
        let copy = &mut graph.scopes[scopes[&scope]];
        copy.parent = data.parent.map(|parent| scopes[&parent]).or(parent);
        copy.bindings = data
            .bindings
            .iter()
            .map(|binding| bindings[binding])
            .collect();
    }
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
            if matches!(data.op, Op::This) {
                values.insert(data.result.unwrap(), receiver);
                continue;
            }
            let mut op = data.op.clone();
            match &mut op {
                Op::Read(binding) | Op::Write(binding, _) | Op::Initialize(binding, _) => {
                    *binding = bindings[binding];
                }
                Op::EnterScope(scope) if body.scopes[*scope].kind == ScopeKind::Captures => {
                    op = Op::EnterCaptures(scopes[scope], *callee);
                }
                Op::EnterScope(scope)
                | Op::LeaveScope(scope)
                | Op::NextIteration(scope)
                | Op::FreshIteration(scope) => {
                    *scope = scopes[scope];
                }
                _ => {}
            }
            let (copy, result) = graph.append(blocks[&block], data.span, op, data.result.is_some());
            graph.insts[copy].source_name = data.source_name.or(plan.source_name);
            if let Some(value) = data.result {
                values.insert(value, result.unwrap());
            }
            insts.push(copy);
        }
    }
    for inst in insts {
        if matches!(graph.insts[inst].op, Op::EnterCaptures(..)) {
            continue;
        }
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
