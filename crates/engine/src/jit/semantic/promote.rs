use std::collections::{HashMap, HashSet};

use cranelift_codegen::entity::SecondaryMap;

use super::{Binding, Block, Cfg, Constant, Graph, Op, Value};

struct Promoter<'g> {
    graph: &'g mut Graph,
    preds: SecondaryMap<Block, Vec<Block>>,
    entry_defs: HashMap<(Block, Binding), Value>,
    end_defs: HashMap<(Block, Binding), Value>,
    uninitialized: Value,
}

impl Promoter<'_> {
    fn at_entry(&mut self, block: Block, binding: Binding) -> Value {
        if let Some(&value) = self.entry_defs.get(&(block, binding)) {
            return value;
        }
        let mut chain = Vec::new();
        let mut block = block;
        while block != self.graph.entry && self.preds[block].len() == 1 {
            if let Some(&value) = self.entry_defs.get(&(block, binding)) {
                for block in chain {
                    self.entry_defs.insert((block, binding), value);
                }
                return value;
            }
            chain.push(block);
            let pred = self.preds[block][0];
            if let Some(&value) = self.end_defs.get(&(pred, binding)) {
                for block in chain {
                    self.entry_defs.insert((block, binding), value);
                }
                return value;
            }
            block = pred;
        }
        if let Some(&value) = self.entry_defs.get(&(block, binding)) {
            for block in chain {
                self.entry_defs.insert((block, binding), value);
            }
            return value;
        }
        let preds = self.preds[block].clone();
        let value = if block == self.graph.entry {
            self.uninitialized
        } else if let [pred] = preds[..] {
            self.at_end(pred, binding)
        } else {
            let param = self.graph.add_param(block);
            self.entry_defs.insert((block, binding), param);
            for pred in preds {
                let arg = self.at_end(pred, binding);
                self.graph.push_edge_arg(pred, block, arg);
            }
            param
        };
        self.entry_defs.insert((block, binding), value);
        for block in chain {
            self.entry_defs.insert((block, binding), value);
        }
        value
    }

    fn at_end(&mut self, block: Block, binding: Binding) -> Value {
        match self.end_defs.get(&(block, binding)) {
            Some(&value) => value,
            None => self.at_entry(block, binding),
        }
    }
}

enum Access {
    Read,
    Write(Value),
    Initialize(Value),
}

pub(super) fn promote(graph: &mut Graph, notes: &mut Vec<String>) {
    let promoted: HashSet<Binding> = graph
        .bindings
        .iter()
        .filter(|(_, data)| !data.captured)
        .map(|(binding, _)| binding)
        .collect();
    if promoted.is_empty() {
        return;
    }
    let cfg = Cfg::new(graph);
    let entry = graph.entry;
    let span = graph.span;
    let (_, uninitialized) =
        graph.insert(entry, 0, span, Op::Constant(Constant::Uninitialized), true);
    let uninitialized = uninitialized.unwrap();
    let mut end_defs = HashMap::new();
    for &block in &cfg.rpo {
        for &inst in &graph.blocks[block].insts {
            match graph.insts[inst].op {
                Op::EnterScope(scope) => {
                    for &binding in &graph.scopes[scope].bindings {
                        if promoted.contains(&binding) {
                            end_defs.insert((block, binding), uninitialized);
                        }
                    }
                }
                Op::Initialize(binding, value) | Op::Write(binding, value)
                    if promoted.contains(&binding) =>
                {
                    end_defs.insert((block, binding), value);
                }
                _ => {}
            }
        }
    }
    let mut promoter = Promoter {
        graph,
        preds: cfg.preds.clone(),
        entry_defs: HashMap::new(),
        end_defs,
        uninitialized,
    };
    let mut aliases = HashMap::new();
    for &block in &cfg.rpo {
        let mut current: HashMap<Binding, Value> = HashMap::new();
        for inst in promoter.graph.blocks[block].insts.clone() {
            let (binding, access) = match promoter.graph.insts[inst].op {
                Op::EnterScope(scope) => {
                    for &binding in &promoter.graph.scopes[scope].bindings {
                        if promoted.contains(&binding) {
                            current.insert(binding, uninitialized);
                        }
                    }
                    continue;
                }
                Op::Read(binding) => (binding, Access::Read),
                Op::Write(binding, value) => (binding, Access::Write(value)),
                Op::Initialize(binding, value) => (binding, Access::Initialize(value)),
                _ => continue,
            };
            if !promoted.contains(&binding) {
                continue;
            }
            if let Access::Initialize(value) = access {
                current.insert(binding, value);
                promoter.graph.remove_inst(inst);
                continue;
            }
            let old = match current.get(&binding) {
                Some(&value) => value,
                None => promoter.at_entry(block, binding),
            };
            let data = &mut promoter.graph.insts[inst];
            match access {
                Access::Read => {
                    aliases.insert(data.result.take().unwrap(), old);
                }
                Access::Write(value) => {
                    current.insert(binding, value);
                }
                Access::Initialize(_) => unreachable!(),
            }
            data.op = Op::CheckInitialized(old);
        }
    }
    promoter.graph.replace_uses(&aliases);
    remove_trivial_params(promoter.graph);
    notes.push(format!(
        "promoted {} binding(s) to SSA values",
        promoted.len()
    ));
}

fn remove_trivial_params(graph: &mut Graph) {
    let blocks = Cfg::new(graph).rpo;
    loop {
        let mut incoming: HashMap<Value, Vec<Value>> = HashMap::new();
        for &block in &blocks {
            let data = &graph.blocks[block];
            if let Some(terminator) = &data.terminator {
                for target in terminator.kind.targets() {
                    for (&param, &arg) in graph.blocks[target.block].params.iter().zip(&target.args)
                    {
                        incoming.entry(param).or_default().push(arg);
                    }
                }
            }
        }
        let mut aliases = HashMap::new();
        for (param, args) in incoming {
            let mut unique = None;
            let mut trivial = true;
            for value in args {
                let value = super::resolve(&aliases, value);
                if value == param || unique == Some(value) {
                    continue;
                }
                if unique.is_some() {
                    trivial = false;
                    break;
                }
                unique = Some(value);
            }
            if trivial && let Some(value) = unique {
                aliases.insert(param, value);
            }
        }
        if aliases.is_empty() {
            return;
        }
        graph.replace_uses(&aliases);
        let removed: HashMap<Block, Vec<usize>> = graph
            .blocks
            .iter()
            .map(|(block, data)| {
                (
                    block,
                    data.params
                        .iter()
                        .enumerate()
                        .filter_map(|(index, param)| aliases.contains_key(param).then_some(index))
                        .collect(),
                )
            })
            .collect();
        for &block in &blocks {
            let data = &mut graph.blocks[block];
            if let Some(terminator) = &mut data.terminator {
                for target in terminator.kind.targets_mut() {
                    for &index in removed[&target.block].iter().rev() {
                        target.args.remove(index);
                    }
                }
            }
        }
        for (block, data) in graph.blocks.iter_mut() {
            data.params.retain(|param| !aliases.contains_key(param));
            for (index, &param) in data.params.iter().enumerate() {
                graph.values[param] = super::ValueDef::Param(block, index);
            }
        }
    }
}
