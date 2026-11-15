use crate::FxHashMap;
use cranelift_codegen::entity::SecondaryMap;
use oxc_span::Span;

use super::low_level::Op;
use super::{Block, Cfg, Graph, Inst, Op as GraphOp, Repr, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit) struct Variable(usize);

#[derive(Default)]
pub(in crate::jit) struct Variables {
    reprs: Vec<Repr>,
    definitions: FxHashMap<(Block, Variable), Value>,
    reads: Vec<(Block, Variable, Inst, Value)>,
}

impl Variables {
    pub(in crate::jit) fn declare(
        &mut self,
        graph: &mut Graph,
        entry: Value,
        repr: Repr,
    ) -> Variable {
        let variable = Variable(self.reprs.len());
        self.reprs.push(repr);
        self.define(graph, graph.entry, variable, entry);
        variable
    }

    pub(in crate::jit) fn define(
        &mut self,
        graph: &mut Graph,
        block: Block,
        variable: Variable,
        value: Value,
    ) {
        let repr = self.reprs[variable.0];
        match graph.reprs[value] {
            Some(existing) => assert_eq!(existing, repr, "{variable:?} written with {value}"),
            None => graph.reprs[value] = Some(repr),
        }
        self.definitions.insert((block, variable), value);
    }

    pub(in crate::jit) fn read(
        &mut self,
        graph: &mut Graph,
        block: Block,
        span: Span,
        variable: Variable,
    ) -> Value {
        if let Some(&value) = self.definitions.get(&(block, variable)) {
            return value;
        }
        let (inst, result) = graph.append(block, span, GraphOp::LowLevel(Op::VariableRead), true);
        let value = result.unwrap();
        graph.reprs[value] = Some(self.reprs[variable.0]);
        self.reads.push((block, variable, inst, value));
        self.definitions.insert((block, variable), value);
        value
    }

    pub(in crate::jit) fn finish(mut self, graph: &mut Graph) -> Result<(), String> {
        let cfg = Cfg::new(graph);
        let mut reachable = SecondaryMap::<Block, bool>::new();
        for &block in &cfg.rpo {
            reachable[block] = true;
        }
        let mut entries = FxHashMap::default();
        let mut aliases = SecondaryMap::new();
        let mut work = Vec::new();
        let mut changed = false;
        let mut removed = Vec::with_capacity(self.reads.len());
        for (block, variable, inst, value) in std::mem::take(&mut self.reads) {
            removed.push(inst);
            if !reachable[block] {
                continue;
            }
            let definition =
                self.entry_value(graph, &cfg, &mut entries, &mut work, block, variable)?;
            aliases[value] = Some(definition);
            changed = true;
        }
        graph.remove_insts(removed);
        super::transforms::simplify_parameters(graph, &cfg.rpo, &mut aliases, changed);
        Ok(())
    }

    fn entry_value(
        &self,
        graph: &mut Graph,
        cfg: &Cfg,
        entries: &mut FxHashMap<(Block, Variable), Value>,
        work: &mut Vec<(Block, Option<usize>)>,
        block: Block,
        variable: Variable,
    ) -> Result<Value, String> {
        if let Some(&value) = entries.get(&(block, variable)) {
            return Ok(value);
        }
        work.push((block, None));
        while let Some(&(current, next)) = work.last() {
            let next = match next {
                Some(next) => next,
                None => {
                    if current == graph.entry || cfg.preds[current].is_empty() {
                        return Err(format!(
                            "{variable:?} has no definition at entry to {current}"
                        ));
                    }
                    if cfg.preds[current].len() > 1 {
                        // Backedges must find this phi before its inputs are resolved.
                        let param = graph.add_param(current);
                        entries.insert((current, variable), param);
                        graph.reprs[param] = Some(self.reprs[variable.0]);
                    }
                    work.last_mut().unwrap().1 = Some(0);
                    0
                }
            };
            let pred = cfg.preds[current][next];
            let value = self
                .definitions
                .get(&(pred, variable))
                .or_else(|| entries.get(&(pred, variable)))
                .copied();
            let Some(value) = value else {
                work.push((pred, None));
                continue;
            };
            if cfg.preds[current].len() == 1 {
                entries.insert((current, variable), value);
            } else {
                graph.push_edge_arg(pred, current, value);
            }
            if next + 1 == cfg.preds[current].len() {
                work.pop();
            } else {
                work.last_mut().unwrap().1 = Some(next + 1);
            }
        }
        Ok(entries[&(block, variable)])
    }
}
