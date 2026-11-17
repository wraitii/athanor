use crate::jit::semantic::JsOp;
use oxc_syntax::operator::BinaryOperator;

use super::super::{FunctionState, Op, ValueDef, ValueType};

pub(in crate::jit::semantic) fn concatenate(function: &mut FunctionState) {
    function.ensure_facts();
    let graph = &mut function.graph;
    let facts = &function.facts;
    let uses = graph.use_counts();
    let mut changed = false;
    for &block in &facts.cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Op::Javascript(JsOp::Binary(BinaryOperator::Addition, left, right)) =
                graph.insts[inst].op
            else {
                continue;
            };
            if uses[left] != 1
                || !facts.of(left).within(ValueType::STRING)
                || !facts.of(right).within(ValueType::STRING)
            {
                continue;
            }
            let ValueDef::Result(inner, _) = graph.values[left] else {
                continue;
            };
            if graph.insts[inner].block != block {
                continue;
            }
            let mut parts = match &graph.insts[inner].op {
                Op::Javascript(JsOp::Binary(BinaryOperator::Addition, first, second))
                    if facts.of(*first).within(ValueType::STRING)
                        && facts.of(*second).within(ValueType::STRING) =>
                {
                    vec![*first, *second]
                }
                Op::Javascript(JsOp::StringConcat(parts)) => parts.clone(),
                _ => continue,
            };
            let instructions = &graph.blocks[block].insts;
            let start = instructions
                .iter()
                .position(|&candidate| candidate == inner)
                .unwrap();
            let end = instructions
                .iter()
                .position(|&candidate| candidate == inst)
                .unwrap();
            if !instructions[start + 1..end]
                .iter()
                .all(|&between| facts.is_pure(between))
            {
                continue;
            }
            parts.push(right);
            graph.set_op(inst, Op::Javascript(JsOp::StringConcat(parts)));
            graph.remove_inst(inner);
            changed = true;
        }
    }
    if changed {
        function.invalidate_values();
    }
}
