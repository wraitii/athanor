use crate::{FxHashMap, FxHashSet};

use super::super::{
    BlockData, Cfg, Constant, Graph, Inst, Op, Scope, ScopeKind, Target, Terminator, TerminatorKind,
};
use super::calls::Inline;

fn scope_at(graph: &Graph, inst: Inst) -> Option<Scope> {
    let mut pending = vec![(graph.entry, Vec::new())];
    let mut visited = FxHashSet::default();
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

pub(in crate::jit::semantic) fn direct_captures(
    graph: &Graph,
    call: Inst,
    callee: super::super::Value,
) -> Option<Vec<super::super::Binding>> {
    let Op::Closure(_, _, _, false, _, captures, _) = graph.defining_op(callee)? else {
        return None;
    };
    let origin = graph.def_block(callee);
    let target = graph.insts[call].block;
    let cfg = Cfg::new(graph);
    let mut pending = vec![target];
    let mut blocks = FxHashSet::default();
    while let Some(block) = pending.pop() {
        if !blocks.insert(block) || block == origin {
            continue;
        }
        pending.extend(cfg.preds[block].iter().copied());
    }
    // A closure can retain an earlier iteration even while the binding name stays in scope.
    let scopes: FxHashSet<_> = captures.iter().map(|&b| graph.bindings[b].scope).collect();
    for block in blocks {
        let mut after = block != origin;
        for &inst in &graph.blocks[block].insts {
            if inst == call {
                break;
            }
            if graph.insts[inst].results.contains(&callee) {
                after = true;
                continue;
            }
            if after
                && matches!(graph.insts[inst].op,
                Op::EnterScope(scope) | Op::LeaveScope(scope) | Op::NextIteration(scope)
                | Op::FreshIteration(scope) if scopes.contains(&scope))
            {
                return None;
            }
        }
    }
    Some(captures.clone())
}

pub(in crate::jit::semantic) fn uses_this(body: &Graph) -> bool {
    Cfg::new(body).rpo.iter().any(|&block| {
        body.blocks[block].insts.iter().any(|&inst| {
            matches!(
                body.insts[inst].op,
                Op::This | Op::Closure(_, _, true, _, true, _, None)
            )
        })
    })
}

pub(in crate::jit::semantic) fn apply(graph: &mut Graph, body: &Graph, plan: &Inline) {
    let call = graph.insts[plan.call].clone();
    let (callee, receiver, args, spread) = match &call.op {
        Op::Call {
            callee,
            receiver,
            args,
            ..
        } => (callee, *receiver, args.as_slice(), None),
        Op::ConstructorCall {
            callee,
            receiver,
            args,
        } => (callee, Some(*receiver), args.as_slice(), None),
        Op::CallPack(callee, receiver, args) | Op::ConstructorCallPack(callee, receiver, args) => {
            (callee, Some(*receiver), &[][..], Some(*args))
        }
        _ => unreachable!(),
    };
    let captures = (!plan.guard_code)
        .then(|| direct_captures(graph, plan.call, *callee))
        .flatten();
    let parent = scope_at(graph, plan.call);
    let (block, continuation, fallback) = super::super::dispatch::split(graph, plan.call);
    let current = graph.blocks.push(BlockData::default());
    graph.blocks[block].terminator = Some(Terminator {
        source_name: call.source_name,
        emission_location: None,
        span: call.span,
        kind: if plan.guard_code {
            let check = graph.append_value(block, call.span, Op::IsCode(*callee, plan.target));
            TerminatorKind::Branch(check, Target::new(current), Target::new(fallback))
        } else {
            TerminatorKind::Jump(Target::new(current))
        },
    });
    let undefined = graph.append_value(current, call.span, Op::Constant(Constant::Undefined));
    let receiver = if plan.arrow && uses_this(body) {
        let op = Op::ClosureThis(*callee);
        graph.append_value(current, call.span, op)
    } else {
        receiver.unwrap_or(undefined)
    };
    let mut scopes = FxHashMap::default();
    let mut bindings = FxHashMap::default();
    for (scope, data) in &body.scopes {
        scopes.insert(scope, graph.scopes.push(data.clone()));
    }
    let aliases: FxHashMap<_, _> = captures
        .as_ref()
        .map(|captures| {
            body.scopes
                .values()
                .filter(|scope| scope.kind == ScopeKind::Captures)
                .flat_map(|scope| scope.bindings.iter().copied())
                .zip(captures.iter().copied())
                .collect()
        })
        .unwrap_or_default();
    for (binding, data) in &body.bindings {
        if let Some(&alias) = aliases.get(&binding) {
            bindings.insert(binding, alias);
            continue;
        }
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
            .filter(|binding| !aliases.contains_key(binding))
            .map(|binding| bindings[binding])
            .collect();
    }
    let slot_base = graph.low_level_slots.len();
    graph
        .low_level_slots
        .extend_from_slice(&body.low_level_slots);
    let mut blocks = FxHashMap::default();
    let mut values = FxHashMap::default();
    let mut insts = Vec::new();
    let mut spread_args = FxHashMap::default();
    let mut lexical_closures = Vec::new();
    let reachable = Cfg::new(body).rpo;
    for &block in &reachable {
        let copy = graph.blocks.push(BlockData::default());
        blocks.insert(block, copy);
        for &param in &body.blocks[block].params {
            let new = graph.add_param(copy);
            graph.reprs[new] = body.reprs[param];
            values.insert(param, new);
        }
    }
    for &block in &reachable {
        for &inst in &body.blocks[block].insts {
            let data = &body.insts[inst];
            if let Op::Argument(index) = data.op {
                let value = if let Some(array) = spread {
                    *spread_args.entry(index).or_insert_with(|| {
                        let key = graph.append_value(
                            current,
                            call.span,
                            Op::Constant(Constant::Number(index as f64)),
                        );
                        graph.append_value(current, call.span, Op::PackTake(array, key))
                    })
                } else {
                    args.get(index).copied().unwrap_or(undefined)
                };
                values.insert(data.result().unwrap(), value);
                continue;
            }
            if matches!(data.op, Op::This) {
                values.insert(data.result().unwrap(), receiver);
                continue;
            }
            if matches!(data.op, Op::Arguments) {
                let (copy, value) =
                    graph.append(blocks[&block], data.span, Op::NewArray(args.to_vec()), true);
                graph.insts[copy].source_name = data.source_name.or(plan.source_name);
                let value = value.unwrap();
                values.insert(data.result().unwrap(), value);
                continue;
            }
            if matches!(data.op, Op::ArgumentCount) {
                let value = graph.append_value(
                    current,
                    call.span,
                    Op::Constant(Constant::Number(args.len() as f64)),
                );
                values.insert(data.result().unwrap(), value);
                continue;
            }
            let mut op = data.op.clone();
            match &mut op {
                Op::LowLevel(super::super::low_level::Op::StackAddress { slot, .. }) => {
                    slot.0 += slot_base
                }
                Op::Closure(_, _, _, _, _, captures, _) => {
                    for binding in captures {
                        *binding = bindings[binding];
                    }
                }
                Op::Read(binding)
                | Op::Write(binding, _)
                | Op::Initialize(binding, _)
                | Op::BindingLoad(binding)
                | Op::BindingRestore(binding, _) => {
                    *binding = bindings[binding];
                }
                Op::EnterScope(scope) if body.scopes[*scope].kind == ScopeKind::Captures => {
                    op = if captures.is_some() {
                        Op::EnterScope(scopes[scope])
                    } else {
                        Op::EnterCaptures(scopes[scope], *callee)
                    };
                }
                Op::EnterScope(scope)
                | Op::LeaveScope(scope)
                | Op::NextIteration(scope)
                | Op::FreshIteration(scope) => {
                    *scope = scopes[scope];
                }
                _ => {}
            }
            let count = data.results.len();
            let (copy, results) = graph.append_results(blocks[&block], data.span, op, count);
            graph.insts[copy].source_name = data.source_name.or(plan.source_name);
            graph.insts[copy].emission_location = data.emission_location;
            if matches!(data.op, Op::Closure(_, _, true, _, true, _, None)) {
                lexical_closures.push(copy);
            }
            for (&old, &new) in data.results.iter().zip(&results) {
                graph.reprs[new] = body.reprs[old];
                values.insert(old, new);
            }
            insts.push(copy);
        }
    }
    if let Some(pack) = spread {
        graph.append(current, call.span, Op::PackRelease(pack), false);
    }
    for inst in insts {
        if matches!(graph.insts[inst].op, Op::EnterCaptures(..)) {
            continue;
        }
        for value in graph.insts[inst].op.operands_mut() {
            *value = values[value];
        }
        if let Op::CallForward(callee, receiver) = graph.insts[inst].op {
            graph.set_op(
                inst,
                Op::Call {
                    callee,
                    receiver: Some(receiver),
                    args: args.to_vec(),
                    target: None,
                },
            );
        }
    }
    for inst in lexical_closures {
        if let Op::Closure(_, _, _, _, _, _, lexical_receiver) = &mut graph.insts[inst].op {
            *lexical_receiver = Some(receiver);
        }
    }
    for &block in &reachable {
        let mut terminator = body.blocks[block].terminator.clone().unwrap();
        terminator.source_name = terminator.source_name.or(plan.source_name);
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
        source_name: call.source_name,
        emission_location: None,
        span: call.span,
        kind: TerminatorKind::Jump(Target::new(blocks[&body.entry])),
    });
}
