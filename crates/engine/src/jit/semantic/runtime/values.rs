use cranelift_codegen::ir::condcodes::IntCC;

use super::super::ownership::Plan;
use super::super::selection::{Implementation, Lowering};
use super::super::{Cfg, FunctionState, Graph, Inst, Op, Repr, Value, low_level};
use super::helpers::{insert_before, integer};

const TAG_MASK: u64 = 0xffff_0000_0000_0000;

fn insert(graph: &mut Graph, plan: &mut Plan, at: Inst, op: low_level::Op, repr: Repr) -> Value {
    let (inst, value) = insert_before(graph, at, op, Some(repr));
    plan.implementations[inst] = Implementation {
        lowering: Lowering::Typed(Some(repr)),
        ..Default::default()
    };
    value.unwrap()
}

pub(in crate::jit::semantic) fn lower(function: &mut FunctionState) {
    let graph = &mut function.graph;
    let mut plan = graph
        .ownership
        .take()
        .expect("value lowering requires ownership planning");
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Op::ConstructorResult(result, instance) = graph.insts[inst].op else {
                continue;
            };
            let mask = insert(graph, &mut plan, inst, integer(TAG_MASK), Repr::I64);
            let tag = low_level::Op::Binary {
                op: low_level::Binary::And,
                left: result,
                right: mask,
            };
            let tag = insert(graph, &mut plan, inst, tag, Repr::I64);
            let object = integer(crate::value::TAG_OBJECT);
            let object = insert(graph, &mut plan, inst, object, Repr::I64);
            let condition = low_level::Op::IntCompare {
                condition: IntCC::Equal,
                left: tag,
                right: object,
            };
            let condition = insert(graph, &mut plan, inst, condition, Repr::I8);
            graph.insts[inst].op = Op::LowLevel(low_level::Op::Select {
                condition,
                yes: result,
                no: instance,
            });
            plan.implementations[inst].lowering = Lowering::Typed(Some(Repr::Boxed));
        }
    }
    graph.ownership = Some(plan);
}
