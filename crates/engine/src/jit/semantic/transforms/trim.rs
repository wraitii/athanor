use super::super::{Cfg, Graph, TerminatorKind, Value};
use std::collections::HashMap;

pub(in crate::jit::semantic) fn trim(graph: &mut Graph) {
    if graph.generator || graph.has_handlers {
        return;
    }
    let cfg = Cfg::new(graph);
    let mut preds = cfg.preds;
    let mut aliases: HashMap<Value, Value> = HashMap::new();
    for &block in &cfg.rpo {
        if graph.blocks[block].terminator.is_none() {
            continue;
        }
        while let TerminatorKind::Jump(target) =
            &graph.blocks[block].terminator.as_ref().unwrap().kind
        {
            let next = target.block;
            if next == block || next == graph.entry || preds[next][..] != [block] {
                break;
            }
            let target = target.clone();
            let mut data = std::mem::take(&mut graph.blocks[next]);
            for (&param, &arg) in data.params.iter().zip(&target.args) {
                aliases.insert(param, arg);
            }
            for &inst in &data.insts {
                graph.insts[inst].block = block;
            }
            let terminator = data.terminator.take().unwrap();
            for successor in terminator.kind.targets() {
                for pred in &mut preds[successor.block] {
                    if *pred == next {
                        *pred = block;
                    }
                }
            }
            let merged = &mut graph.blocks[block];
            merged.insts.append(&mut data.insts);
            merged.terminator = Some(terminator);
            graph.cold.remove(&next);
        }
    }
    graph.replace_uses(&aliases);
}
