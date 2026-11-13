use super::super::transforms::loops::natural_loops;
use super::super::{Block, FunctionState, Op, Value};
use crate::{FxHashMap, FxHashSet};

pub(in crate::jit::semantic) fn sink_closures(function: &mut FunctionState) {
    function.ensure_cfg();
    let FunctionState {
        graph,
        facts,
        notes,
        ..
    } = function;
    let cfg = &facts.cfg;
    let loops = natural_loops(graph, cfg);
    let mut changed = false;
    let mut uses: FxHashMap<Value, FxHashSet<Block>> = FxHashMap::default();
    let mut closures = Vec::new();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            if matches!(&data.op, Op::Closure(_, _, _, false, _, _, _)) {
                closures.push(inst);
            }
            for value in data.op.operands_iter() {
                uses.entry(value).or_default().insert(block);
            }
        }
        for value in graph.blocks[block]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .operands_iter()
        {
            uses.entry(value).or_default().insert(block);
        }
    }
    for inst in closures {
        let value = graph.insts[inst].result().unwrap();
        let Some(users) = uses.get(&value) else {
            graph.remove_inst(inst);
            changed = true;
            notes.push(format!("unused closure {value} removed after inlining"));
            continue;
        };
        if users.len() != 1 {
            continue;
        }
        let target = *users.iter().next().unwrap();
        let origin = graph.insts[inst].block;
        if target == origin
            || !graph.cold.contains(&target)
            || loops
                .iter()
                .any(|(_, body)| body.contains(&origin) != body.contains(&target))
        {
            continue;
        }
        if matches!(&graph.insts[inst].op, Op::Closure(_, _, _, _, _, captures, _) if !captures.is_empty())
        {
            let user = graph.blocks[target].insts.iter().copied().find(|&user| {
                graph.insts[user]
                    .op
                    .operands_iter()
                    .any(|operand| operand == value)
            });
            if !user.is_some_and(|user| {
                super::super::specialization::inlining::direct_captures(graph, user, value)
                    .is_some()
            }) {
                continue;
            }
        }
        graph.remove_inst(inst);
        changed = true;
        let position = graph.blocks[target]
            .insts
            .iter()
            .position(|&user| {
                graph.insts[user]
                    .op
                    .operands_iter()
                    .any(|operand| operand == value)
            })
            .unwrap_or(graph.blocks[target].insts.len());
        graph.blocks[target].insts.insert(position, inst);
        graph.insts[inst].block = target;
        notes.push(format!(
            "closure {value} materialized only in fallback {target}"
        ));
    }
    if changed {
        function.invalidate_values();
    }
}

pub(in crate::jit::semantic) fn forward_captured_parameters(function: &mut FunctionState) {
    function.ensure_cfg();
    let FunctionState {
        graph,
        facts,
        notes,
        ..
    } = function;
    let cfg = &facts.cfg;
    let mut aliases = FxHashMap::default();
    let mut removed = Vec::new();
    for &binding in &graph.lazy_captures {
        if graph.scopes[graph.bindings[binding].scope].kind != super::super::ScopeKind::Parameters {
            continue;
        }
        let mut initializers = Vec::new();
        let mut writes = false;
        let mut escaped = FxHashSet::default();
        let mut pending = Vec::new();
        for &block in &cfg.rpo {
            for &inst in &graph.blocks[block].insts {
                match &graph.insts[inst].op {
                    Op::Initialize(target, value) if *target == binding => {
                        initializers.push((inst, *value))
                    }
                    Op::Write(target, _) if *target == binding => writes = true,
                    Op::Closure(_, _, _, _, _, captures, _) if captures.contains(&binding) => {
                        pending.push(block)
                    }
                    _ => {}
                }
            }
        }
        if writes || initializers.len() != 1 {
            continue;
        }
        let (initializer, value) = initializers[0];
        let origin = graph.insts[initializer].block;
        while let Some(block) = pending.pop() {
            if escaped.insert(block) {
                pending.extend(graph.successors(block));
            }
        }
        for &block in &cfg.rpo {
            if escaped.contains(&block) || !cfg.dominates(origin, block) {
                continue;
            }
            let mut initialized = block != origin;
            for &inst in &graph.blocks[block].insts {
                if inst == initializer {
                    initialized = true;
                }
                if initialized
                    && matches!(graph.insts[inst].op, Op::Read(target) if target == binding)
                {
                    aliases.insert(graph.insts[inst].result().unwrap(), value);
                    removed.push(inst);
                }
            }
        }
    }
    graph.replace_uses(&aliases);
    for inst in removed {
        notes.push(format!(
            "forwarded captured parameter read at {}",
            graph.note_inst(inst)
        ));
        graph.remove_inst(inst);
    }
    if !aliases.is_empty() {
        function.invalidate_values();
    }
}

pub(in crate::jit::semantic) fn lazy_captures(function: &mut FunctionState) -> bool {
    function.ensure_cfg();
    let FunctionState {
        graph,
        facts,
        notes,
        ..
    } = function;
    let mut demoted = false;
    let mut changed = false;
    let mut candidates = FxHashSet::default();
    let mut eager = FxHashSet::default();
    for &block in &facts.cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            match &graph.insts[inst].op {
                Op::Closure(_, _, _, false, _, captures, _) if graph.cold.contains(&block) => {
                    candidates.extend(captures.iter().copied());
                }
                Op::Closure(_, _, _, _, _, captures, _) => eager.extend(captures.iter().copied()),
                _ => {}
            }
        }
    }
    for (binding, data) in &mut graph.bindings {
        if data.captured
            && !data.persistent
            && graph.scopes[data.scope].kind != super::super::ScopeKind::Captures
            && !eager.contains(&binding)
        {
            if candidates.contains(&binding) {
                if graph.lazy_captures.insert(binding) {
                    changed = true;
                    notes.push(format!(
                        "binding {binding} materialized as a cell only on closure creation"
                    ));
                }
            } else {
                data.captured = false;
                graph.lazy_captures.remove(&binding);
                demoted = true;
                changed = true;
                notes.push(format!("binding {binding} no longer needs a capture cell"));
            }
        }
    }
    if changed {
        function.invalidate_values();
    }
    demoted
}
