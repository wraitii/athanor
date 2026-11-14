use super::super::{Block, Cfg, Graph, Value};
use crate::FxHashMap;

pub(in crate::jit::semantic) fn repair(graph: &mut Graph, values: &FxHashMap<Value, Vec<Value>>) {
    let cfg = Cfg::new(graph);
    let mut repair = Repair {
        cfg: &cfg,
        values,
        reaching: FxHashMap::default(),
    };
    for &block in &cfg.rpo {
        for inst in graph.blocks[block].insts.clone() {
            let mut op = graph.insts[inst].op.clone();
            for value in op.operands_mut() {
                if values.contains_key(value) && !cfg.dominates(graph.def_block(*value), block) {
                    *value = repair.entering(graph, block, *value);
                }
            }
            graph.set_op(inst, op);
            if let Some(cache) = graph
                .property_accesses
                .get(&inst)
                .and_then(|read| read.cache)
                && values.contains_key(&cache)
                && !cfg.dominates(graph.def_block(cache), block)
            {
                let cache = repair.entering(graph, block, cache);
                graph.property_accesses.get_mut(&inst).unwrap().cache = Some(cache);
            }
        }
        let mut reaching = FxHashMap::default();
        for value in graph.blocks[block]
            .terminator
            .as_ref()
            .unwrap()
            .kind
            .operands()
        {
            if values.contains_key(&value) && !cfg.dominates(graph.def_block(value), block) {
                let replacement = repair.entering(graph, block, value);
                reaching.insert(value, replacement);
            }
        }
        if let Some(terminator) = &mut graph.blocks[block].terminator {
            for value in terminator.kind.operands_mut() {
                if let Some(&replacement) = reaching.get(value) {
                    *value = replacement;
                }
            }
        }
    }
}

struct Repair<'a> {
    cfg: &'a Cfg,
    values: &'a FxHashMap<Value, Vec<Value>>,
    reaching: FxHashMap<(Block, Value), Value>,
}

impl Repair<'_> {
    fn leaving(&mut self, graph: &mut Graph, block: Block, value: Value) -> Value {
        if self.cfg.dominates(graph.def_block(value), block) {
            return value;
        }
        // A dominating copy may precede a newer original definition on another path.
        // Walk predecessors to merge the definitions reaching this block.
        let copy = self.values[&value]
            .iter()
            .copied()
            .find(|&copy| graph.def_block(copy) == block);
        match copy {
            Some(copy) => copy,
            None => self.entering(graph, block, value),
        }
    }

    fn entering(&mut self, graph: &mut Graph, block: Block, value: Value) -> Value {
        if let Some(&reaching) = self.reaching.get(&(block, value)) {
            return reaching;
        }
        let preds = self.cfg.preds[block].to_vec();
        if let [pred] = preds[..] {
            let reaching = self.leaving(graph, pred, value);
            self.reaching.insert((block, value), reaching);
            return reaching;
        }
        let param = graph.add_param(block);
        self.reaching.insert((block, value), param);
        for pred in preds {
            let arg = self.leaving(graph, pred, value);
            graph.push_edge_arg(pred, block, arg);
        }
        param
    }
}
