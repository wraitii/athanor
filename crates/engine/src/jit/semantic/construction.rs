use crate::FxHashSet;

use super::{BlockData, Cfg, Graph, Inst, Op, Target, Terminator, TerminatorKind};

pub(super) fn resolve(graph: &mut Graph) {
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let callee = match graph.insts[inst].op {
                Op::Construct { callee, .. } | Op::ConstructPack(callee, _) => callee,
                _ => continue,
            };
            let position = graph.blocks[block]
                .insts
                .iter()
                .position(|&current| current == inst)
                .unwrap();
            let span = graph.insts[inst].span;
            let (probe, prototype) = graph.insert(
                block,
                position,
                span,
                Op::ConstructorPrototype(callee),
                true,
            );
            graph.insts[probe].source_name = graph.insts[inst].source_name;
            let prototype = prototype.unwrap();
            graph.insts[inst].op = match graph.insts[inst].op.clone() {
                Op::Construct { callee, args } => Op::ConstructResolved {
                    callee,
                    prototype,
                    args,
                },
                Op::ConstructPack(callee, args) => {
                    Op::ConstructPackResolved(callee, prototype, args)
                }
                _ => unreachable!(),
            };
        }
    }
}

pub(super) fn expand(graph: &mut Graph, expanded: &mut FxHashSet<Inst>) {
    let sites: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter(|&inst| {
            !expanded.contains(&inst)
                && matches!(
                    graph.insts[inst].op,
                    Op::ConstructResolved { .. } | Op::ConstructPackResolved(..)
                )
        })
        .collect();
    for inst in sites {
        expanded.insert(inst);
        let data = graph.insts[inst].clone();
        let (callee, prototype) = match data.op {
            Op::ConstructResolved {
                callee, prototype, ..
            }
            | Op::ConstructPackResolved(callee, prototype, _) => (callee, prototype),
            _ => unreachable!(),
        };
        let (block, continuation, fallback) = super::dispatch::split(graph, inst);
        let fast = graph.blocks.push(BlockData::default());
        let (check, unresolved) =
            graph.append(block, data.span, Op::IsUninitialized(prototype), true);
        let unresolved = unresolved.unwrap();
        graph.insts[check].source_name = data.source_name;
        graph.blocks[block].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Branch(unresolved, Target::new(fallback), Target::new(fast)),
        });
        let instance = graph.append_value(fast, data.span, Op::NewInstance(prototype));
        let call = match data.op {
            Op::ConstructResolved { args, .. } => Op::ConstructorCall {
                callee,
                receiver: instance,
                args,
            },
            Op::ConstructPackResolved(_, _, args) => {
                Op::ConstructorCallPack(callee, instance, args)
            }
            _ => unreachable!(),
        };
        let result = graph.append_value(fast, data.span, call);
        let result = graph.append_value(fast, data.span, Op::ConstructorResult(result, instance));
        for &inst in &graph.blocks[fast].insts {
            graph.insts[inst].source_name = data.source_name;
        }
        graph.blocks[fast].terminator = Some(Terminator {
            source_name: data.source_name,
            emission_location: None,
            span: data.span,
            kind: TerminatorKind::Jump(Target {
                block: continuation,
                args: vec![result],
            }),
        });
    }
}
