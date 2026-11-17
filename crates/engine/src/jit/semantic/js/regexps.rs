use crate::FxHashSet;
use crate::jit::profile::{ReadKind, ReadType, Site};
use crate::jit::semantic::JsOp;
use crate::regex::PatternRef;

use super::super::specialization::calls::Repository;
use super::super::{
    BlockData, Cfg, Constant, Graph, Inst, Op, Target, Terminator, TerminatorKind, regex,
};

#[derive(Clone, Copy)]
enum Method {
    Test,
    Exec,
}

pub(in crate::jit::semantic) fn expand(
    graph: &mut Graph,
    bodies: &Repository,
    source: usize,
    expanded: &mut FxHashSet<Inst>,
) {
    if !bodies.regex_inline {
        return;
    }
    let mut sites = Vec::new();
    for block in Cfg::new(graph).rpo {
        for &inst in &graph.blocks[block].insts {
            if expanded.contains(&inst) {
                continue;
            }
            let data = &graph.insts[inst];
            let Op::Javascript(JsOp::Call {
                callee,
                receiver: Some(receiver),
                args,
                target: None,
            }) = &data.op
            else {
                continue;
            };
            let Some(Op::Javascript(JsOp::Get(object, name))) = graph.defining_op(*callee) else {
                continue;
            };
            let method = match name.as_str() {
                "test" => Method::Test,
                "exec" => Method::Exec,
                _ => continue,
            };
            if object != receiver || args.is_empty() {
                continue;
            }
            let site = Site {
                inline_context: 0,
                source: data.source_name.map_or(source, |name| name.0),
                start: data.span.start,
                end: data.span.end,
            };
            let Some(&ReadType::Pattern(pattern)) =
                bodies.read_feedback.get(&(site, ReadKind::Pattern))
            else {
                continue;
            };
            if pattern.get().global() || pattern.get().program().is_err() {
                continue;
            }
            sites.push((inst, method, pattern));
        }
    }
    let uses = graph.use_counts();
    for (inst, method, pattern) in sites {
        expanded.insert(inst);
        let unused = graph.insts[inst]
            .result()
            .is_none_or(|result| uses[result] == 0);
        expand_site(
            graph,
            inst,
            method,
            pattern,
            unused && bodies.regex_drop_unused_exec,
        );
    }
}

fn expand_site(
    graph: &mut Graph,
    inst: Inst,
    method: Method,
    pattern: PatternRef,
    drop_unused: bool,
) {
    let call = graph.insts[inst].clone();
    let Op::Javascript(JsOp::Call {
        receiver: Some(receiver),
        ref args,
        ..
    }) = call.op
    else {
        unreachable!()
    };
    let subject = args[0];
    let (block, continuation, fallback) = super::super::dispatch::split(graph, inst);
    let append = |graph: &mut Graph, block, op| {
        let (inst, result) = graph.append(block, call.span, op, true);
        graph.insts[inst].source_name = call.source_name;
        graph.insts[inst].inline_context = call.inline_context;
        result.unwrap()
    };
    let terminate = |graph: &mut Graph, block, kind| {
        graph.blocks[block].terminator = Some(Terminator {
            source_name: call.source_name,
            emission_location: None,
            span: call.span,
            kind,
        });
    };
    // RegExp objects reject property writes and there is no RegExp prototype to patch, so the
    // pattern guard also proves the callee is the builtin method.
    let string = graph.blocks.push(BlockData::default());
    let fast = graph.blocks.push(BlockData::default());
    let is_pattern = append(
        graph,
        block,
        Op::Regex(regex::Op::IsPattern(receiver, pattern)),
    );
    terminate(
        graph,
        block,
        TerminatorKind::Branch(is_pattern, Target::new(string), Target::new(fallback)),
    );
    let is_string = append(graph, string, Op::Javascript(JsOp::IsString(subject)));
    terminate(
        graph,
        string,
        TerminatorKind::Branch(is_string, Target::new(fast), Target::new(fallback)),
    );
    let op = match method {
        Method::Test => Op::Regex(regex::Op::Test(subject, pattern)),
        Method::Exec if drop_unused => Op::Javascript(JsOp::Constant(Constant::Null)),
        Method::Exec => Op::Regex(regex::Op::Exec(receiver, subject, pattern)),
    };
    let result = append(graph, fast, op);
    let args = if graph.blocks[continuation].params.is_empty() {
        Vec::new()
    } else {
        vec![result]
    };
    terminate(
        graph,
        fast,
        TerminatorKind::Jump(Target {
            block: continuation,
            args,
        }),
    );
}
