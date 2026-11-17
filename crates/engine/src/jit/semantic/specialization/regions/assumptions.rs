use super::super::super::guards::{Fact, Origin};
use super::super::calls::ConditionalPlan;
use super::*;
use crate::jit::profile::ReadType;
use crate::jit::semantic::JsOp;

pub(super) struct Proposal {
    pub(super) inst: Inst,
    pub(super) value: Value,
    pub(super) fact: Fact,
    pub(super) origin: Origin,
    pub(super) point: Block,
    pub(super) local: bool,
    pub(super) global: Option<String>,
}

pub(super) fn prepare(graph: &mut Graph, conditional: &ConditionalPlan) -> Vec<Proposal> {
    let numbers = super::propose::numbers(graph);
    super::propose::typed(graph, &numbers);
    let probes = super::propose::identities(graph, conditional);
    let hints = super::super::observations::type_hints(graph);
    let cfg = Cfg::new(graph);
    let loops = natural_loops(graph, &cfg);
    let insts: Vec<_> = cfg
        .rpo
        .iter()
        .flat_map(|&block| graph.blocks[block].insts.iter().copied())
        .filter(|&inst| matches!(graph.insts[inst].op, Op::Assume(..)))
        .map(|inst| {
            (
                inst,
                loops
                    .iter()
                    .any(|(_, body)| body.contains(&graph.insts[inst].block)),
            )
        })
        .collect();
    let mut proposals = Vec::new();
    for (inst, in_loop) in insts {
        let data = graph.insts[inst].clone();
        let Op::Assume(value, fact, origin) = data.op else {
            unreachable!()
        };
        let global = match (fact, graph.defining_op(value)) {
            (
                Fact::Code(_),
                Some(Op::Javascript(JsOp::ProbeGlobal(name) | JsOp::GlobalRead(name))),
            ) => Some(name.clone()),
            _ => None,
        };
        let local = match fact {
            Fact::Type(ValueType::STRING) => true,
            Fact::Type(_) => {
                matches!(
                    hints.get(&value),
                    Some(ReadType::Integer | ReadType::Number | ReadType::String)
                ) && in_loop
            }
            Fact::Typed(_) | Fact::Array => in_loop,
            _ => false,
        };
        let point = if matches!(fact, Fact::Code(_)) {
            data.block
        } else {
            split(graph, inst)
        };
        proposals.push(Proposal {
            inst,
            value,
            fact,
            origin,
            point,
            local,
            global,
        });
    }
    for inst in probes {
        graph.remove_inst(inst);
    }
    proposals
}

fn split(graph: &mut Graph, inst: Inst) -> Block {
    let data = &graph.insts[inst];
    let block = data.block;
    let span = data.span;
    let at = graph.blocks[block]
        .insts
        .iter()
        .position(|&other| other == inst)
        .unwrap();
    let next = graph.blocks.push(BlockData::default());
    graph.blocks[next].insts = graph.blocks[block].insts.split_off(at);
    for &moved in &graph.blocks[next].insts {
        graph.insts[moved].block = next;
    }
    graph.blocks[next].terminator = graph.blocks[block].terminator.take();
    graph.terminate(block, span, TerminatorKind::Jump(Target::new(next)));
    if graph.cold.contains(&block) {
        graph.cold.insert(next);
    }
    for region in graph.regions.values_mut() {
        if region.blocks.contains(&block) {
            region.blocks.insert(next);
        }
    }
    next
}
