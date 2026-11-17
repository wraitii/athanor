use crate::FxHashSet;
use crate::jit::semantic::JsOp;

use super::super::{Block, Cfg, Constant, Graph, Inst, Op, Value, ValueDef};

fn users(graph: &Graph, value: Value) -> Option<Vec<Inst>> {
    let mut users = Vec::new();
    for block in Cfg::new(graph).rpo {
        if graph.blocks[block]
            .terminator
            .as_ref()
            .is_some_and(|term| term.kind.operands_iter().any(|operand| operand == value))
        {
            return None;
        }
        for &inst in &graph.blocks[block].insts {
            if graph.insts[inst]
                .op
                .operands_iter()
                .any(|operand| operand == value)
            {
                users.push(inst);
            }
        }
    }
    Some(users)
}

fn reaches(graph: &Graph, from: Block, to: Block, allocation: Block) -> bool {
    let mut visited = FxHashSet::default();
    let mut pending = vec![from];
    while let Some(block) = pending.pop() {
        if block == to {
            return true;
        }
        if !visited.insert(block) {
            continue;
        }
        // Passing the allocation starts another array lifetime.
        pending.extend(
            graph
                .successors(block)
                .into_iter()
                .filter(|&next| next != allocation),
        );
    }
    false
}

pub(in crate::jit::semantic) fn lower(graph: &mut Graph) -> bool {
    let spreads: Vec<_> = Cfg::new(graph)
        .rpo
        .iter()
        .flat_map(|&block| graph.blocks[block].insts.iter().copied())
        .filter(|&inst| matches!(graph.insts[inst].op, Op::Javascript(JsOp::Spread(_))))
        .collect();
    let mut changed = false;
    for spread in spreads {
        let Op::Javascript(JsOp::Spread(array)) = graph.insts[spread].op else {
            continue;
        };
        let ValueDef::Result(allocation, _) = graph.values[array] else {
            continue;
        };
        if !matches!(
            graph.insts[allocation].op,
            Op::Javascript(JsOp::Arguments | JsOp::NewArray(_))
        ) {
            continue;
        }
        let pack = graph.insts[spread].result().unwrap();
        let Some(pack_users) = users(graph, pack) else {
            continue;
        };
        let [call] = pack_users.as_slice() else {
            continue;
        };
        let Op::Javascript(JsOp::CallPack(callee, receiver, arguments)) = graph.insts[*call].op
        else {
            continue;
        };
        if arguments != pack || callee == pack || receiver == pack {
            continue;
        }
        let Some(array_users) = users(graph, array) else {
            continue;
        };
        let lengths: Vec<_> = array_users.iter().copied().filter(|&inst| {
            matches!(&graph.insts[inst].op, Op::Javascript(JsOp::Get(value, name)) if *value == array && name == "length")
        }).collect();
        let remaining: Vec<_> = array_users
            .into_iter()
            .filter(|&inst| inst != spread && !lengths.contains(&inst))
            .collect();
        if remaining.len() > 1 {
            continue;
        }
        if let Some(&other) = remaining.first() {
            let origin = graph.insts[allocation].block;
            let forward = graph.insts[spread].block;
            let materialize = graph.insts[other].block;
            if reaches(graph, forward, materialize, origin)
                || reaches(graph, materialize, forward, origin)
                || graph
                    .successors(materialize)
                    .into_iter()
                    .filter(|&next| next != origin)
                    .any(|next| reaches(graph, next, materialize, origin))
                || lengths
                    .iter()
                    .any(|&inst| reaches(graph, materialize, graph.insts[inst].block, origin))
            {
                continue;
            }
        }
        let op = match &graph.insts[allocation].op {
            Op::Javascript(JsOp::Arguments) => Op::Javascript(JsOp::CallForward(callee, receiver)),
            Op::Javascript(JsOp::NewArray(args)) => Op::Javascript(JsOp::Call {
                callee,
                receiver: Some(receiver),
                args: args.clone(),
                target: None,
            }),
            _ => unreachable!(),
        };
        for length in lengths {
            let op = match &graph.insts[allocation].op {
                Op::Javascript(JsOp::Arguments) => Op::ArgumentCount,
                Op::Javascript(JsOp::NewArray(args)) => {
                    Op::Javascript(JsOp::Constant(Constant::Number(args.len() as f64)))
                }
                _ => unreachable!(),
            };
            graph.set_op(length, op);
        }
        graph.set_op(*call, op);
        graph.remove_inst(spread);
        if let Some(&other) = remaining.first() {
            let block = graph.insts[other].block;
            graph.move_inst(allocation, block);
            let insts = &mut graph.blocks[block].insts;
            insts.pop();
            let position = insts.iter().position(|&inst| inst == other).unwrap();
            insts.insert(position, allocation);
        } else {
            graph.remove_inst(allocation);
        }
        changed = true;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxc_allocator::Allocator;
    use oxc_ast::ast::Statement;
    use oxc_parser::Parser;
    use oxc_span::SourceType;

    fn graph(body: &str) -> Graph {
        let source = format!(
            "function main(target,flag){{const args=[];{}}}",
            body.replace("arguments", "args")
        );
        let allocator = Allocator::default();
        let parsed = Parser::new(&allocator, &source, SourceType::script()).parse();
        assert!(!parsed.diagnostics.has_errors());
        let Statement::FunctionDeclaration(function) = &parsed.program.body[0] else {
            panic!()
        };
        let semantic = oxc_semantic::SemanticBuilder::new_compiler()
            .build(&parsed.program)
            .semantic;
        let graph =
            super::super::super::build::lower(function, semantic.scoping(), &FxHashSet::default())
                .unwrap();
        let mut graph = super::super::super::canonicalize_graph(graph, true)
            .unwrap()
            .graph;
        for data in graph.insts.values_mut() {
            if matches!(&data.op, Op::Javascript(JsOp::NewArray(values)) if values.is_empty()) {
                data.op = Op::Javascript(JsOp::Arguments);
            }
        }
        graph.arity = None;
        super::super::intrinsics::lower(&mut graph).unwrap();
        graph
    }

    fn operations(graph: &Graph) -> Vec<&Op> {
        Cfg::new(graph)
            .rpo
            .iter()
            .flat_map(|&block| {
                graph.blocks[block]
                    .insts
                    .iter()
                    .map(|&inst| &graph.insts[inst].op)
            })
            .collect()
    }

    #[test]
    fn forwards_dynamic_arguments_without_materializing_an_array() {
        let mut graph = graph("return $callSpread(target,this,arguments);");
        assert!(lower(&mut graph));
        graph.verify().unwrap();
        let ops = operations(&graph);
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::Javascript(JsOp::CallForward(..))))
        );
        assert!(
            !ops.iter()
                .any(|op| matches!(op, Op::Javascript(JsOp::Arguments | JsOp::Spread(_))))
        );
    }

    #[test]
    fn forwards_static_arguments_and_folds_their_length() {
        let mut graph = graph(
            "const values=[1,2];if(values.length>3)return 0;return $callSpread(target,this,values);",
        );
        assert!(lower(&mut graph));
        graph.verify().unwrap();
        let ops = operations(&graph);
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::Javascript(JsOp::Call { args, .. }) if args.len() == 2))
        );
        assert!(!ops.iter().any(|op| matches!(
            op,
            Op::Javascript(JsOp::NewArray(_) | JsOp::Spread(_) | JsOp::Get(_, _))
        )));
    }

    #[test]
    fn materializes_arguments_only_on_the_observable_branch() {
        let mut graph =
            graph("if(flag)return target(arguments);return $callSpread(target,this,arguments);");
        assert!(lower(&mut graph));
        graph.verify().unwrap();
        assert!(
            !graph.blocks[graph.entry]
                .insts
                .iter()
                .any(|&inst| matches!(graph.insts[inst].op, Op::Javascript(JsOp::Arguments)))
        );
        assert!(
            operations(&graph)
                .iter()
                .any(|op| matches!(op, Op::Javascript(JsOp::Arguments)))
        );
    }

    #[test]
    fn keeps_materialized_arguments_when_a_call_can_mutate_them() {
        let mut graph = graph("target(arguments);return $callSpread(target,this,arguments);");
        assert!(!lower(&mut graph));
        graph.verify().unwrap();
    }

    #[test]
    fn keeps_materialized_arguments_when_the_fallback_repeats() {
        let mut graph = graph(
            "if(flag){for(let i=0;i<3;++i)target(arguments);return 0;}return $callSpread(target,this,arguments);",
        );
        assert!(!lower(&mut graph));
        graph.verify().unwrap();
    }
}
