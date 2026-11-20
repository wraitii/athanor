use super::boundary::{inputs, liveness, unbox};
use super::transforms::remove_trivial_params;
use super::{
    Block, BlockData, Cfg, Constant, Facts, FunctionState, Graph, Op, Target, TerminatorKind,
    Value, natural_loops,
};
use crate::jit::semantic::JsOp;
use crate::{FxHashMap, FxHashSet};

pub(in crate::jit) trait Offload {
    fn compile(&mut self, kernel: &Graph) -> Option<usize>;
    fn launch(&mut self, graph: &mut Graph, block: Block, kernel: usize, inputs: &[Value])
    -> Value;
}

struct Loop {
    header: Block,
    body: FxHashSet<Block>,
    preheader: Block,
    exit: Block,
}

pub(super) fn outline(
    function: &mut FunctionState,
    target: &mut dyn Offload,
    after: &mut dyn FnMut(&str, &Graph),
) {
    if function.graph.generator || function.graph.has_handlers {
        return;
    }
    function.ensure_facts();
    let mut graph = function.graph.clone();
    remove_trivial_params(&mut graph);
    let mut tried = FxHashSet::default();
    let mut changed = false;
    while let Some(candidate) = next_loop(&graph, &tried) {
        tried.insert(candidate.header);
        let Some((kernel, inputs)) = extract(&graph, &function.facts, &candidate) else {
            continue;
        };
        after("versioning/offload", &kernel);
        let Some(id) = target.compile(&kernel) else {
            continue;
        };
        let block = graph.blocks.push(BlockData::default());
        let done = target.launch(&mut graph, block, id, &inputs);
        install(&mut graph, &candidate, block, done);
        tried.extend(candidate.body.iter().copied());
        changed = true;
    }
    if changed {
        function.graph = graph;
        function.invalidate_cfg();
    }
}

fn next_loop(graph: &Graph, tried: &FxHashSet<Block>) -> Option<Loop> {
    let cfg = Cfg::new(graph);
    let live = liveness(graph, &cfg.rpo);
    natural_loops(graph, &cfg)
        .into_iter()
        .filter(|(header, _)| !tried.contains(header) && !graph.cold.contains(header))
        .filter_map(|(header, body)| qualify(graph, &cfg, &live, header, body))
        .max_by_key(|candidate| candidate.body.len())
}

fn qualify(
    graph: &Graph,
    cfg: &Cfg,
    live: &cranelift_codegen::entity::SecondaryMap<Block, FxHashSet<Value>>,
    header: Block,
    body: FxHashSet<Block>,
) -> Option<Loop> {
    let (inside, outside): (Vec<Block>, Vec<Block>) = cfg.preds[header]
        .iter()
        .partition(|pred| body.contains(pred));
    let ([_latch], [preheader]) = (&inside[..], &outside[..]) else {
        return None;
    };
    let preheader = *preheader;
    match &graph.blocks[preheader].terminator.as_ref()?.kind {
        TerminatorKind::Jump(target) if target.block == header => {}
        _ => return None,
    }
    let TerminatorKind::Branch(_, then, otherwise) =
        &graph.blocks[header].terminator.as_ref()?.kind
    else {
        return None;
    };
    let exit = match (body.contains(&then.block), body.contains(&otherwise.block)) {
        (true, false) => otherwise,
        (false, true) => then,
        _ => return None,
    };
    if !exit.args.is_empty() || !graph.blocks[exit.block].params.is_empty() {
        return None;
    }
    let escapes = live[exit.block]
        .iter()
        .any(|&value| body.contains(&graph.def_block(value)));
    if escapes {
        return None;
    }
    Some(Loop {
        header,
        exit: exit.block,
        body,
        preheader,
    })
}

fn extract(graph: &Graph, facts: &Facts, candidate: &Loop) -> Option<(Graph, Vec<Value>)> {
    let mut kernel = graph.clone();
    let span = kernel.span;
    let done = kernel.blocks.push(BlockData::default());
    let undefined = kernel.append_value(
        done,
        span,
        Op::Javascript(JsOp::Constant(Constant::Undefined)),
    );
    kernel.terminate(done, span, TerminatorKind::Return(undefined));
    for &block in &candidate.body {
        let exits: Vec<Target> = kernel.blocks[block]
            .terminator
            .as_ref()?
            .kind
            .targets()
            .filter(|target| !candidate.body.contains(&target.block))
            .cloned()
            .collect();
        let mut replacements = Vec::with_capacity(exits.len());
        for exit in exits {
            if block == candidate.header && exit.block == candidate.exit {
                replacements.push(done);
                continue;
            }
            let bail = kernel.blocks.push(BlockData::default());
            kernel.terminate(
                bail,
                span,
                TerminatorKind::Bailout {
                    target: exit,
                    inputs: Vec::new(),
                },
            );
            replacements.push(bail);
        }
        let terminator = kernel.blocks[block].terminator.as_mut()?;
        let outside = terminator
            .kind
            .targets_mut()
            .filter(|target| !candidate.body.contains(&target.block));
        for (target, replacement) in outside.zip(replacements) {
            *target = Target::new(replacement);
        }
    }
    let TerminatorKind::Jump(entry) = &graph.blocks[candidate.preheader].terminator.as_ref()?.kind
    else {
        return None;
    };
    let cfg = Cfg::from_entry(&kernel, candidate.header);
    let live = liveness(&kernel, &cfg.rpo);
    let values = inputs(entry, &live);
    let start = kernel.blocks.push(BlockData::default());
    let mut aliases = FxHashMap::default();
    let mut passed = Vec::new();
    for value in values {
        let definition = match kernel.defining_op(value) {
            Some(op @ Op::Javascript(JsOp::Constant(_))) => {
                let op = op.clone();
                kernel.append_value(start, span, op)
            }
            _ => {
                let argument = kernel.append_value(start, span, Op::Argument(passed.len()));
                passed.push(value);
                unbox(&mut kernel, start, argument, facts.of(value))
            }
        };
        aliases.insert(value, definition);
    }
    kernel.replace_uses(&aliases);
    let args = entry.args.iter().map(|arg| aliases[arg]).collect();
    kernel.terminate(
        start,
        span,
        TerminatorKind::Jump(Target {
            block: candidate.header,
            args,
        }),
    );
    kernel.entry = start;
    kernel.arity = None;
    kernel.name = format!("{}.kernel.{}", graph.name, candidate.header);
    let reachable: FxHashSet<_> = Cfg::new(&kernel).rpo.into_iter().collect();
    for (block, data) in kernel.blocks.iter_mut() {
        if !reachable.contains(&block) {
            *data = BlockData::default();
        }
    }
    kernel.cold.retain(|block| reachable.contains(block));
    Some((kernel, passed))
}

fn install(graph: &mut Graph, candidate: &Loop, block: Block, done: Value) {
    let terminator = graph.blocks[candidate.preheader]
        .terminator
        .as_mut()
        .unwrap();
    let span = terminator.span;
    let TerminatorKind::Jump(entry) = std::mem::replace(
        &mut terminator.kind,
        TerminatorKind::Jump(Target::new(block)),
    ) else {
        unreachable!()
    };
    graph.terminate(
        block,
        span,
        TerminatorKind::Branch(done, Target::new(candidate.exit), entry),
    );
    for region in graph.regions.values_mut() {
        if region.blocks.contains(&candidate.preheader) {
            region.blocks.insert(block);
        }
    }
}
