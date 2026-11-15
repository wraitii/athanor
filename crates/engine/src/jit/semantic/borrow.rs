use crate::{FxHashMap, FxHashSet};

use cranelift_codegen::entity::SecondaryMap;

use super::liveness::{self, Set};
use super::{Block, Facts, Graph, Inst, Op, Value, ValueType};

const HEAP: ValueType = ValueType::STRING
    .union(ValueType::OBJECT)
    .union(ValueType::OTHER);

pub(in crate::jit) fn receiver(op: &Op) -> Option<Value> {
    match op {
        Op::HelperCall {
            helper:
                crate::jit::helpers::Helper::PropertyGetCached
                | crate::jit::helpers::Helper::SemanticPropertyProbeCached,
            args,
            ..
        } => args.first().copied(),
        Op::Get(object, _)
        | Op::PropertyLoad(object, _)
        | Op::CachedProperty(object, _, _)
        | Op::ShapedGet(object, ..)
        | Op::ProbeProperty(object, _)
        | Op::ProbeShapeProperty(object, ..)
        | Op::GetIndex(object, _)
        | Op::ArrayGet(object, _)
        | Op::ArrayLoad(object, _, _) => Some(*object),
        _ => None,
    }
}

fn stores(op: &Op) -> bool {
    matches!(
        op,
        Op::Set(..)
            | Op::ShapedSet(..)
            | Op::ArrayStore(..)
            | Op::PropertyStore(..)
            | Op::PropertyAdd(..)
            | Op::FillPropertyStore(..)
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
    chains: FxHashMap<Value, Vec<Value>>,
    rejected: FxHashSet<Value>,
    uses: SecondaryMap<Inst, Vec<Value>>,
}

impl Borrows<'_> {
    fn effects(&self, inst: Inst) -> (bool, bool) {
        let memory = self.facts.memory(inst);
        (memory.reenters(), memory.writes_data())
    }

    fn redefined(&mut self, live: &Set, defined: Value) {
        for value in live.iter() {
            if value == defined {
                continue;
            }
            let mut pending = self.chains[&value].clone();
            let mut visited = FxHashSet::default();
            while let Some(receiver) = pending.pop() {
                if receiver == defined {
                    self.rejected.insert(value);
                    break;
                }
                if visited.insert(receiver)
                    && let Some(chain) = self.chains.get(&receiver)
                {
                    pending.extend(chain.iter().copied());
                }
            }
        }
    }
}

// A borrowed read skips frame rooting and the inline temporary retain. While it is live, no
// write, reentry, or redefinition of a receiver in its chain may run, so a receiver that holds
// the value keeps holding it. Ownership proofs stop at receivers that remain rooted; borrowed
// receivers must themselves be held along the rest of the chain. When a receiver might not
// hold it (string indexing allocates), the value must die within its block: slow-path results
// stay owned by a temporary until the block's clear, and the inline path only runs on arrays,
// which own their elements.
// Indexed iteration reads only run on arrays and typed arrays, so their receiver always holds them.
pub(super) fn borrows(graph: &mut Graph, facts: &Facts, notes: &mut Vec<String>) {
    graph.borrowed.clear();
    let mut reads = FxHashMap::default();
    let mut held = FxHashSet::default();
    for (_, data) in graph.insts.iter() {
        if let (Some(object), Some(result)) = (receiver(&data.op), data.result())
            && facts.of(result).intersects(HEAP)
        {
            reads.insert(result, object);
            if matches!(
                data.op,
                Op::ArrayGet(..)
                    | Op::ArrayLoad(..)
                    | Op::ProbeProperty(..)
                    | Op::ProbeShapeProperty(..)
                    | Op::PropertyLoad(..)
                    | Op::CachedProperty(..)
                    | Op::ShapedGet(..)
            ) {
                held.insert(result);
            }
        }
    }
    let mut chains = FxHashMap::default();
    let mut objects = FxHashSet::default();
    for (&value, &object) in &reads {
        let mut chain = vec![object];
        while let Some(&next) = reads.get(chain.last().unwrap()) {
            chain.push(next);
        }
        if held.contains(&value) || facts.of(object).within(ValueType::OBJECT) {
            objects.insert(value);
        }
        chains.insert(value, chain);
    }
    let cfg = &facts.cfg;
    let mut merged = FxHashMap::default();
    loop {
        let mut changed = false;
        for &block in &cfg.rpo {
            for (index, &param) in graph.blocks[block].params.iter().enumerate() {
                if chains.contains_key(&param) || !facts.of(param).intersects(HEAP) {
                    continue;
                }
                let incoming: Vec<_> = cfg.preds[block]
                    .iter()
                    .flat_map(|&pred| {
                        graph.blocks[pred]
                            .terminator
                            .as_ref()
                            .unwrap()
                            .kind
                            .targets()
                    })
                    .filter(|target| target.block == block)
                    .map(|target| target.args[index])
                    .collect();
                if incoming.is_empty() || incoming.iter().any(|value| !objects.contains(value)) {
                    continue;
                }
                let mut chain: Vec<_> = incoming
                    .iter()
                    .flat_map(|value| chains[value].iter().copied())
                    .collect();
                chain.sort();
                chain.dedup();
                if chain
                    .iter()
                    .any(|&owner| !cfg.dominates(graph.def_block(owner), block))
                {
                    continue;
                }
                chains.insert(param, chain);
                objects.insert(param);
                held.insert(param);
                merged.insert(param, incoming);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // The environment owns these values until a write or reentry invalidates the borrow.
    for block in graph.blocks.values() {
        for &inst in &block.insts {
            let data = &graph.insts[inst];
            if let (Op::ProbeGlobal(_), Some(value)) = (&data.op, data.result()) {
                chains.insert(value, Vec::new());
                objects.insert(value);
            }
        }
    }
    if chains.is_empty() {
        return;
    }
    let mut local: FxHashSet<Value> = chains.keys().copied().collect();
    for (block, data) in graph.blocks.iter() {
        for &inst in &data.insts {
            for value in graph.insts[inst].op.operands_iter() {
                if local.contains(&value) && graph.def_block(value) != block {
                    local.remove(&value);
                }
            }
        }
        if let Some(terminator) = &data.terminator {
            for value in terminator.kind.operands_iter() {
                local.remove(&value);
            }
        }
    }
    let mut uses = SecondaryMap::<Inst, Vec<Value>>::new();
    let mut block_uses = SecondaryMap::<Block, Set>::new();
    let mut defs = SecondaryMap::<Block, Set>::new();
    for &block in &cfg.rpo {
        let data = &graph.blocks[block];
        let mut live = Set::default();
        if let Some(terminator) = &data.terminator {
            live.extend(
                terminator
                    .kind
                    .operands_iter()
                    .filter(|value| chains.contains_key(value)),
            );
        }
        for &inst in data.insts.iter().rev() {
            let data = &graph.insts[inst];
            if let Some(result) = data.result() {
                live.remove(&result);
                if chains.contains_key(&result) {
                    defs[block].insert(result);
                }
            }
            uses[inst] = data
                .op
                .operands_iter()
                .filter(|value| chains.contains_key(value))
                .collect();
            live.extend(uses[inst].iter().copied());
        }
        for &param in &data.params {
            live.remove(&param);
            if chains.contains_key(&param) {
                defs[block].insert(param);
            }
        }
        block_uses[block] = live;
    }
    let live = liveness::solve(graph, cfg, &block_uses, &defs);
    let mut borrows = Borrows {
        graph,
        facts,
        chains,
        rejected: FxHashSet::default(),
        uses,
    };
    for &block in &cfg.rpo {
        scan(&mut borrows, block, live.output[block].clone());
    }
    let mut accepted: FxHashSet<Value> = borrows
        .chains
        .keys()
        .filter(|value| {
            !borrows.rejected.contains(value) && (objects.contains(value) || local.contains(value))
        })
        .copied()
        .collect();
    loop {
        let rejected: Vec<Value> = accepted
            .iter()
            .filter(|value| {
                if let Some(incoming) = merged.get(value) {
                    return incoming.iter().any(|value| !accepted.contains(value));
                }
                if local.contains(value) {
                    return false;
                }
                let chain = &borrows.chains[value];
                let results = std::iter::once(**value).chain(chain.iter().copied());
                for (result, &object) in results.zip(chain) {
                    if !held.contains(&result) && !facts.of(object).within(ValueType::OBJECT) {
                        return true;
                    }
                    if !accepted.contains(&object) {
                        break;
                    }
                }
                false
            })
            .copied()
            .collect();
        if rejected.is_empty() {
            break;
        }
        for value in rejected {
            accepted.remove(&value);
        }
    }
    if accepted.is_empty() {
        return;
    }
    let mut accepted: Vec<Value> = accepted.into_iter().collect();
    accepted.sort();
    let list: Vec<String> = accepted.iter().map(ToString::to_string).collect();
    notes.push(format!("borrowed reads: [{}]", list.join(", ")));
    graph.borrowed = accepted.into_iter().collect();
}

fn scan(borrows: &mut Borrows<'_>, block: Block, mut live: Set) {
    let graph = borrows.graph;
    let data = &graph.blocks[block];
    if let Some(terminator) = &data.terminator {
        live.extend(
            terminator
                .kind
                .operands_iter()
                .filter(|value| borrows.chains.contains_key(value)),
        );
    }
    for &inst in data.insts.iter().rev() {
        let result = graph.insts[inst].result();
        if let Some(result) = result {
            borrows.redefined(&live, result);
        }
        let (reenters, writes) = borrows.effects(inst);
        if reenters || writes {
            for value in live.iter() {
                if Some(value) != result {
                    borrows.rejected.insert(value);
                }
            }
        }
        if (reenters || (writes && !stores(&graph.insts[inst].op)))
            && !matches!(graph.insts[inst].op, Op::Continuation(..))
        {
            borrows.rejected.extend(borrows.uses[inst].iter().copied());
        }
        if let Some(result) = result {
            live.remove(&result);
        }
        live.extend(borrows.uses[inst].iter().copied());
    }
    for &param in &data.params {
        borrows.redefined(&live, param);
        live.remove(&param);
    }
}
