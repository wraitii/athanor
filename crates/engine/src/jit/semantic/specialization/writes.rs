use super::super::dispatch::{branch, extend_regions};
use super::super::{Cfg, Graph, Inst, Op, Target, TerminatorKind};
use crate::jit::semantic::JsOp;

pub(in crate::jit::semantic) fn lower(
    graph: &mut Graph,
    planned: &[super::reads::Read],
) -> super::super::Result<()> {
    if graph.has_handlers || graph.generator {
        let typed: Vec<_> = graph
            .insts
            .iter()
            .filter_map(|(inst, data)| match data.op {
                Op::Javascript(JsOp::TypedSet(object, key, value, _)) => {
                    Some((inst, JsOp::SetIndex(object, key, value)))
                }
                _ => None,
            })
            .collect();
        for (inst, op) in typed {
            graph.set_op(inst, Op::Javascript(op));
        }
        return Ok(());
    }
    let stores: Vec<_> = Cfg::new(graph)
        .rpo
        .into_iter()
        .filter(|block| !graph.cold.contains(block))
        .flat_map(|block| graph.blocks[block].insts.clone())
        .filter_map(|inst| {
            let data = &graph.insts[inst];
            match data.op {
                Op::Javascript(JsOp::TypedSet(.., element)) => return Some((inst, Some(element))),
                Op::Javascript(JsOp::SetIndex(..)) => {}
                _ => return None,
            }
            let access = planned.iter().find(|access| {
                let original = &graph.insts[access.inst];
                original.span == data.span && original.source_name == data.source_name
            })?;
            Some((inst, access.typed.then(|| access.element.unwrap())))
        })
        .collect();
    for (inst, element) in stores {
        match element {
            Some(element) => super::super::runtime::typed::lower_store(graph, inst, element),
            None => lower_array(graph, inst),
        }
    }
    graph.verify()
}

fn lower_array(graph: &mut Graph, inst: Inst) {
    let data = graph.insts[inst].clone();
    let Op::Javascript(JsOp::SetIndex(object, key, value)) = data.op else {
        unreachable!()
    };
    let numeric = super::super::Facts::compute(graph)
        .of(key)
        .within(super::super::ValueType::NUMBER);
    let first = graph.blocks.len();
    let (block, done, fallback) = super::super::dispatch::split(graph, inst);
    let fast = if numeric {
        block
    } else {
        let number = graph.append_value(block, data.span, Op::Javascript(JsOp::IsNumber(key)));
        branch(graph, inst, block, number, fallback)
    };
    let key_number = graph.append_value(fast, data.span, Op::Javascript(JsOp::AsNumber(key)));
    let position = graph.append_value(
        fast,
        data.span,
        Op::Javascript(JsOp::ArrayPosition(key_number)),
    );
    let valid = graph.append_value(
        fast,
        data.span,
        Op::Javascript(JsOp::IsArrayIndex(
            object,
            key_number,
            position,
            super::super::ArrayLayout::Packed,
        )),
    );
    let fast = branch(graph, inst, fast, valid, fallback);
    let writable = graph.append_value(
        fast,
        data.span,
        Op::Javascript(JsOp::CanOverwrite(
            object,
            position,
            super::super::js::memory::Field::Elements,
        )),
    );
    let fast = branch(graph, inst, fast, writable, fallback);
    graph.append(
        fast,
        data.span,
        Op::Javascript(JsOp::ArrayStore(object, position, value)),
        false,
    );
    graph.terminate(fast, data.span, TerminatorKind::Jump(Target::new(done)));
    extend_regions(graph, block, first);
}
