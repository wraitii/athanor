use std::collections::{HashMap, HashSet};

use cranelift_codegen::entity::SecondaryMap;

use super::super::effects::{HEAP_WRITE, Reentry};
use super::{Block, Cfg, Facts, Graph, Inst, Op, Types, Value};

const HEAP: Types = Types::STRING.union(Types::OBJECT).union(Types::OTHER);

fn receiver(op: &Op) -> Option<Value> {
    match op {
        Op::Get(object, _) | Op::GetIndex(object, _) | Op::IndexedGet(object, _) => Some(*object),
        _ => None,
    }
}

fn stores(op: &Op) -> bool {
    matches!(
        op,
        Op::Set(..)
            | Op::SetIndex(..)
            | Op::Define(..)
            | Op::ArrayAppend(..)
            | Op::Initialize(..)
            | Op::Write(..)
    )
}

struct Borrows<'g> {
    graph: &'g Graph,
    facts: &'g Facts,
    chains: HashMap<Value, Vec<Value>>,
    rejected: HashSet<Value>,
}

impl Borrows<'_> {
    fn effects(&self, inst: Inst) -> (bool, bool) {
        let contract = self.graph.contract(&self.graph.insts[inst].op, self.facts);
        (
            contract.reentry != Reentry::Closed,
            contract.flags & HEAP_WRITE != 0,
        )
    }

    fn redefined(&mut self, live: &HashSet<Value>, defined: Value) {
        for &value in live {
            if value != defined && self.chains[&value].contains(&defined) {
                self.rejected.insert(value);
            }
        }
    }

    fn uses(&self, inst: Inst) -> Vec<Value> {
        self.graph.insts[inst]
            .op
            .operands()
            .into_iter()
            .filter(|value| self.chains.contains_key(value))
            .collect()
    }
}

// A borrowed read skips frame rooting and the inline temporary retain. While it is live, no
// write, reentry, or redefinition of a receiver in its chain may run, so a receiver that holds
// the value keeps holding it. When a receiver might not hold it (string indexing allocates),
// the value must die within its block: slow-path results stay owned by a temporary until the
// block's clear, and the inline path only runs on arrays, which own their elements.
pub(super) fn borrows(graph: &mut Graph, facts: &Facts, notes: &mut Vec<String>) {
    let mut reads = HashMap::new();
    for (_, data) in graph.insts.iter() {
        if let (Some(object), Some(result)) = (receiver(&data.op), data.result)
            && facts.of(result).intersects(HEAP)
        {
            reads.insert(result, object);
        }
    }
    let mut chains = HashMap::new();
    let mut objects = HashSet::new();
    for (&value, &object) in &reads {
        let mut chain = vec![object];
        while let Some(&next) = reads.get(chain.last().unwrap()) {
            chain.push(next);
        }
        if chain
            .iter()
            .all(|object| facts.of(*object).within(Types::OBJECT))
        {
            objects.insert(value);
        }
        chains.insert(value, chain);
    }
    if chains.is_empty() {
        return;
    }
    let mut local: HashSet<Value> = chains.keys().copied().collect();
    for (block, data) in graph.blocks.iter() {
        for &inst in &data.insts {
            for value in graph.insts[inst].op.operands() {
                if local.contains(&value) && graph.def_block(value) != block {
                    local.remove(&value);
                }
            }
        }
        if let Some(terminator) = &data.terminator {
            for value in terminator.kind.operands() {
                local.remove(&value);
            }
        }
    }
    let cfg = Cfg::new(graph);
    let mut borrows = Borrows {
        graph,
        facts,
        chains,
        rejected: HashSet::new(),
    };
    let mut live_in: SecondaryMap<Block, HashSet<Value>> = SecondaryMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for &block in cfg.rpo.iter().rev() {
            let live = scan(&mut borrows, block, &live_in, false);
            if live != live_in[block] {
                live_in[block] = live;
                changed = true;
            }
        }
    }
    for &block in &cfg.rpo {
        scan(&mut borrows, block, &live_in, true);
    }
    let mut accepted: Vec<Value> = borrows
        .chains
        .keys()
        .filter(|value| {
            !borrows.rejected.contains(value) && (objects.contains(value) || local.contains(value))
        })
        .copied()
        .collect();
    if accepted.is_empty() {
        return;
    }
    accepted.sort();
    let list: Vec<String> = accepted.iter().map(ToString::to_string).collect();
    notes.push(format!("borrowed reads: [{}]", list.join(", ")));
    graph.borrowed = accepted.into_iter().collect();
}

fn scan(
    borrows: &mut Borrows<'_>,
    block: Block,
    live_in: &SecondaryMap<Block, HashSet<Value>>,
    check: bool,
) -> HashSet<Value> {
    let graph = borrows.graph;
    let data = &graph.blocks[block];
    let mut live: HashSet<Value> = graph
        .successors(block)
        .into_iter()
        .flat_map(|successor| live_in[successor].iter().copied())
        .collect();
    if let Some(terminator) = &data.terminator {
        live.extend(
            terminator
                .kind
                .operands()
                .into_iter()
                .filter(|value| borrows.chains.contains_key(value)),
        );
    }
    for &inst in data.insts.iter().rev() {
        let result = graph.insts[inst].result;
        if check {
            if let Some(result) = result {
                borrows.redefined(&live, result);
            }
            let (reenters, writes) = borrows.effects(inst);
            if reenters || writes {
                for &value in &live {
                    if Some(value) != result {
                        borrows.rejected.insert(value);
                    }
                }
            }
            if reenters || (writes && !stores(&graph.insts[inst].op)) {
                for value in borrows.uses(inst) {
                    borrows.rejected.insert(value);
                }
            }
        }
        if let Some(result) = result {
            live.remove(&result);
        }
        live.extend(borrows.uses(inst));
    }
    for &param in &data.params {
        if check {
            borrows.redefined(&live, param);
        }
        live.remove(&param);
    }
    live
}
