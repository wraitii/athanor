use super::super::{Cfg, Constant, Graph, Op, Result};
use crate::jit::semantic::JsOp;

pub(in crate::jit::semantic) fn lower(graph: &mut Graph) -> Result<()> {
    for block in Cfg::new(graph).rpo {
        for inst in graph.blocks[block].insts.clone() {
            let Op::Javascript(JsOp::Call { callee, args, .. }) = &graph.insts[inst].op else {
                continue;
            };
            let Some(Op::Javascript(JsOp::GlobalRead(name))) = graph.defining_op(*callee) else {
                continue;
            };
            let op = match name.as_str() {
                "$defineHidden" if args.len() == 3 => {
                    let Some(Op::Javascript(JsOp::Constant(Constant::String(name)))) =
                        graph.defining_op(args[1])
                    else {
                        return Err("library property name must be a string literal".into());
                    };
                    let define = Op::Javascript(JsOp::DefineHidden(args[0], name.clone(), args[2]));
                    let position = graph.blocks[block]
                        .insts
                        .iter()
                        .position(|&i| i == inst)
                        .unwrap();
                    graph.insert(block, position, graph.insts[inst].span, define, false);
                    Op::Javascript(JsOp::Constant(Constant::Undefined))
                }
                "$call" if args.len() >= 2 => Op::Javascript(JsOp::Call {
                    callee: args[0],
                    receiver: Some(args[1]),
                    args: args[2..].to_vec(),
                    target: None,
                }),
                "$callSpread" if args.len() == 3 => {
                    let (callee, receiver, arguments) = (args[0], args[1], args[2]);
                    let position = graph.blocks[block]
                        .insts
                        .iter()
                        .position(|&i| i == inst)
                        .unwrap();
                    let (_, pack) = graph.insert(
                        block,
                        position,
                        graph.insts[inst].span,
                        Op::Javascript(JsOp::Spread(arguments)),
                        true,
                    );
                    Op::Javascript(JsOp::CallPack(callee, receiver, pack.unwrap()))
                }
                "$isOrdinaryArray" if args.len() == 1 => {
                    Op::Javascript(JsOp::IsOrdinaryArray(args[0]))
                }
                "$call" | "$callSpread" | "$defineHidden" | "$isOrdinaryArray" => {
                    return Err("invalid library intrinsic arguments".into());
                }
                _ => continue,
            };
            graph.set_op(inst, op);
        }
    }
    for (_, data) in graph.insts.iter_mut() {
        if matches!(&data.op, Op::Javascript(JsOp::GlobalRead(name)) if name == "$call" || name == "$callSpread" || name == "$defineHidden" || name == "$isOrdinaryArray")
        {
            data.op = Op::Javascript(JsOp::Constant(Constant::Undefined));
        }
    }
    graph.verify()
}
