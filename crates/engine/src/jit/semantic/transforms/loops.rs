use super::super::{Block, Cfg, Graph, Terminator, TerminatorKind};
use crate::{FxHashMap, FxHashSet};

pub(in crate::jit::semantic) fn preheader(
    graph: &Graph,
    cfg: &Cfg,
    header: Block,
    body: &FxHashSet<Block>,
) -> Option<Block> {
    let outside: Vec<Block> = cfg.preds[header]
        .iter()
        .copied()
        .filter(|pred| !body.contains(pred))
        .collect();
    let [preheader] = outside[..] else {
        return None;
    };
    matches!(
        graph.blocks[preheader].terminator,
        Some(Terminator {
            kind: TerminatorKind::Jump(_),
            ..
        })
    )
    .then_some(preheader)
}

pub(in crate::jit::semantic) fn natural_loop(cfg: &Cfg, header: Block) -> Option<FxHashSet<Block>> {
    let mut body: Option<FxHashSet<Block>> = None;
    for &latch in cfg.preds[header].iter() {
        if !cfg.dominates(header, latch) {
            continue;
        }
        let body = body.get_or_insert_with(|| FxHashSet::from_iter([header]));
        let mut stack = vec![latch];
        while let Some(node) = stack.pop() {
            if body.insert(node) {
                stack.extend(cfg.preds[node].iter().copied());
            }
        }
    }
    body
}

pub(in crate::jit::semantic) fn natural_loops(
    graph: &Graph,
    cfg: &Cfg,
) -> Vec<(Block, FxHashSet<Block>)> {
    let mut loops: FxHashMap<Block, FxHashSet<Block>> = FxHashMap::default();
    for &block in &cfg.rpo {
        for header in graph.successors(block) {
            if !cfg.dominates(header, block) {
                continue;
            }
            let body = loops
                .entry(header)
                .or_insert_with(|| FxHashSet::from_iter([header]));
            let mut stack = vec![block];
            while let Some(node) = stack.pop() {
                if body.insert(node) {
                    stack.extend(cfg.preds[node].iter().copied());
                }
            }
        }
    }
    let mut loops: Vec<_> = loops.into_iter().collect();
    loops.sort_by_key(|(header, body)| (body.len(), *header));
    loops
}
