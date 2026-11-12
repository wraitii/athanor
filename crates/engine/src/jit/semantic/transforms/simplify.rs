use std::collections::{HashMap, HashSet};

use cranelift_codegen::entity::SecondaryMap;

use super::super::{
    Block, Cfg, Constant, Facts, FunctionState, Graph, Inst, Op, Target, Terminator,
    TerminatorKind, UnaryOp, Value,
};

pub(in crate::jit::semantic) enum Fold {
    Op(Op),
    Alias(Value),
}

pub(in crate::jit::semantic) trait Folder {
    fn refine(&mut self, _graph: &mut Graph, _cfg: &Cfg) -> bool {
        false
    }

    fn fold(&mut self, graph: &Graph, facts: &Facts, inst: Inst) -> Option<Fold>;
}

pub(in crate::jit::semantic) fn simplify(function: &mut FunctionState, folder: &mut impl Folder) {
    loop {
        function.ensure_facts();
        let FunctionState { graph, facts, .. } = function;
        let cfg = &facts.cfg;
        let uses = graph.use_counts();
        let mut aliases = HashMap::new();
        let mut dead = Vec::new();
        let mut cfg_changed = folder.refine(graph, cfg);
        let mut changed = cfg_changed;
        for &block in &cfg.rpo {
            for &inst in &graph.blocks[block].insts {
                match folder.fold(graph, facts, inst) {
                    Some(Fold::Op(op)) => {
                        graph.insts[inst].op = op;
                        changed = true;
                        continue;
                    }
                    Some(Fold::Alias(value)) => {
                        aliases.insert(graph.insts[inst].result().unwrap(), value);
                        dead.push(inst);
                        continue;
                    }
                    None => {}
                }
                if let Op::Assume(value, ..) | Op::EntryAssume(value, _) = graph.insts[inst].op {
                    if uses[value] == 0 {
                        dead.push(inst);
                    }
                    continue;
                }
                if facts.removable(inst)
                    && graph.insts[inst]
                        .results
                        .iter()
                        .all(|&result| uses[result] == 0)
                {
                    dead.push(inst);
                }
            }
        }
        graph.replace_uses(&aliases);
        for inst in &dead {
            graph.remove_inst(*inst);
        }
        cfg_changed |= fold_constant_branches(graph, cfg);
        cfg_changed |= fold_dominated_branches(graph, cfg);
        cfg_changed |= thread_branches(graph);
        if cfg_changed {
            function.invalidate_cfg();
        } else if changed || !dead.is_empty() {
            function.invalidate_values();
        } else {
            return;
        }
    }
}

fn fold_constant_branches(graph: &mut Graph, cfg: &Cfg) -> bool {
    let mut changed = false;
    for &block in &cfg.rpo {
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, yes, no),
            ..
        }) = &graph.blocks[block].terminator
        else {
            continue;
        };
        let taken = match graph.defining_op(*condition) {
            Some(Op::Constant(Constant::Boolean(taken))) => *taken,
            Some(Op::EntryAssume(..)) => true,
            _ => continue,
        };
        let target = if taken { yes } else { no }.clone();
        graph.blocks[block].terminator.as_mut().unwrap().kind = TerminatorKind::Jump(target);
        changed = true;
    }
    changed
}

fn branch_condition(graph: &Graph, condition: Value) -> (Value, bool) {
    match graph.defining_op(condition) {
        Some(Op::Unary(UnaryOp::Not, value)) => (*value, true),
        _ => (condition, false),
    }
}

fn fold_dominated_branches(graph: &mut Graph, cfg: &Cfg) -> bool {
    let mut decided: HashMap<Value, Vec<(Block, bool)>> = HashMap::new();
    for &block in &cfg.rpo {
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, yes, no),
            ..
        }) = &graph.blocks[block].terminator
        else {
            continue;
        };
        if yes.block == no.block {
            continue;
        }
        let (condition, inverted) = branch_condition(graph, *condition);
        for (target, taken) in [(yes.block, !inverted), (no.block, inverted)] {
            if target != graph.entry && cfg.preds[target].len() == 1 {
                decided.entry(condition).or_default().push((target, taken));
            }
        }
    }
    if decided.is_empty() {
        return false;
    }
    let mut changed = false;
    for &block in &cfg.rpo {
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, yes, no),
            ..
        }) = &graph.blocks[block].terminator
        else {
            continue;
        };
        let (condition, inverted) = branch_condition(graph, *condition);
        let Some(&(_, value)) = decided.get(&condition).and_then(|facts| {
            facts
                .iter()
                .find(|(target, _)| cfg.dominates(*target, block))
        }) else {
            continue;
        };
        let target = if value != inverted { yes } else { no }.clone();
        graph.blocks[block].terminator.as_mut().unwrap().kind = TerminatorKind::Jump(target);
        changed = true;
    }
    changed
}

fn thread_branches(graph: &mut Graph) -> bool {
    let cfg = Cfg::new(graph);
    let uses = graph.use_counts();
    let header = |block: Block| {
        cfg.preds[block]
            .iter()
            .any(|&pred| cfg.dominates(block, pred))
    };
    let mut touched = HashSet::new();
    for &block in &cfg.rpo {
        let Some(Terminator {
            kind: TerminatorKind::Branch(condition, yes, no),
            ..
        }) = &graph.blocks[block].terminator
        else {
            continue;
        };
        let Some(Op::Unary(UnaryOp::Not, operand)) = graph.defining_op(*condition) else {
            continue;
        };
        let kind = TerminatorKind::Branch(*operand, no.clone(), yes.clone());
        graph.blocks[block].terminator.as_mut().unwrap().kind = kind;
        touched.insert(block);
    }
    if !touched.is_empty() {
        return true;
    }
    for &join in &cfg.rpo {
        if join == graph.entry || header(join) {
            continue;
        }
        let Some((index, yes, no)) = trivial_branch(graph, join, &uses) else {
            continue;
        };
        if header(yes.block) || header(no.block) {
            continue;
        }
        let params = graph.blocks[join].params.clone();
        for &pred in &cfg.preds[join] {
            if pred == join || touched.contains(&join) {
                continue;
            }
            let kind = &mut graph.blocks[pred].terminator.as_mut().unwrap().kind;
            let threaded = match kind {
                TerminatorKind::Jump(target) if target.block == join => {
                    let condition = target.args[index];
                    let yes = substitute(&yes, &params, &target.args);
                    let no = substitute(&no, &params, &target.args);
                    *kind = TerminatorKind::Branch(condition, yes, no);
                    true
                }
                TerminatorKind::Branch(condition, then, otherwise) => {
                    let mut threaded = false;
                    for (target, taken) in [(then, true), (otherwise, false)] {
                        if target.block != join || target.args[index] != *condition {
                            continue;
                        }
                        let next = if taken { &yes } else { &no };
                        *target = substitute(next, &params, &target.args);
                        threaded = true;
                    }
                    threaded
                }
                _ => false,
            };
            if threaded {
                touched.insert(pred);
            }
        }
    }
    !touched.is_empty()
}

fn trivial_branch(
    graph: &Graph,
    block: Block,
    uses: &SecondaryMap<Value, u32>,
) -> Option<(usize, Target, Target)> {
    let data = &graph.blocks[block];
    if !data.insts.is_empty() {
        return None;
    }
    let Terminator {
        kind: TerminatorKind::Branch(condition, yes, no),
        ..
    } = data.terminator.as_ref()?
    else {
        return None;
    };
    let index = data.params.iter().position(|param| param == condition)?;
    let operands = data.terminator.as_ref()?.kind.operands();
    let local = data.params.iter().all(|param| {
        uses[*param] as usize == operands.iter().filter(|value| *value == param).count()
    });
    local.then(|| (index, yes.clone(), no.clone()))
}

fn substitute(target: &Target, params: &[Value], args: &[Value]) -> Target {
    let args = target
        .args
        .iter()
        .map(|value| {
            params
                .iter()
                .position(|param| param == value)
                .map_or(*value, |index| args[index])
        })
        .collect();
    Target {
        block: target.block,
        args,
    }
}
