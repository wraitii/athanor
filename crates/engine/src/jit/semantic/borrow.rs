use super::super::effects::Locations;
use crate::jit::semantic::JsOp;
use crate::{FxHashMap, FxHashSet};

use cranelift_codegen::entity::SecondaryMap;

use super::liveness::{self, Set};
use super::memory::Access;
use super::{Block, Facts, Graph, Inst, Op, Value, ValueType};

const HEAP: ValueType = ValueType::STRING
    .union(ValueType::OBJECT)
    .union(ValueType::OTHER);

pub(in crate::jit) fn receiver(op: &Op) -> Option<Value> {
    match op {
        Op::HelperCall {
            helper:
                crate::jit::helpers::Helper::PropertyGetCached
                | crate::jit::helpers::Helper::PropertyGetMegamorphic
                | crate::jit::helpers::Helper::SemanticPropertyProbeCached,
            args,
            ..
        } => args.first().copied(),
        Op::Javascript(
            JsOp::Get(object, _)
            | JsOp::PropertyLoad(object, ..)
            | JsOp::RelatedPropertyLoad(object, ..)
            | JsOp::CachedProperty(object, _, _)
            | JsOp::ShapedGet(object, ..)
            | JsOp::LayoutGet(object, ..)
            | JsOp::ProbeProperty(object, _)
            | JsOp::ProbeShapeProperty(object, ..)
            | JsOp::GetIndex(object, _)
            | JsOp::ArrayGet(object, _)
            | JsOp::ArrayLoad(object, _, _),
        ) => Some(*object),
        _ => None,
    }
}

struct BorrowedRead {
    receiver: Value,
}

fn anchored_read(op: &Op, facts: &Facts) -> Option<BorrowedRead> {
    let receiver = receiver(op)?;
    if matches!(
        op,
        Op::Javascript(
            JsOp::ArrayGet(..)
                | JsOp::ArrayLoad(..)
                | JsOp::ProbeProperty(..)
                | JsOp::ProbeShapeProperty(..)
                | JsOp::PropertyLoad(..)
                | JsOp::RelatedPropertyLoad(..)
                | JsOp::LayoutGet(..)
                | JsOp::CachedProperty(..)
                | JsOp::ShapedGet(..)
        ) | Op::HelperCall {
            helper: crate::jit::helpers::Helper::PropertyGetCached
                | crate::jit::helpers::Helper::PropertyGetMegamorphic
                | crate::jit::helpers::Helper::SemanticPropertyProbeCached,
            ..
        }
    ) || facts.of(receiver).within(ValueType::OBJECT)
    {
        return Some(BorrowedRead { receiver });
    }
    None
}

#[derive(Clone, Default)]
pub(super) struct Plan {
    anchors: FxHashMap<Value, Vec<Value>>,
    aliases: FxHashMap<Value, Vec<Value>>,
}

impl Plan {
    fn new(graph: &Graph, facts: &Facts) -> Self {
        let mut dependencies = FxHashMap::default();
        for &block in &facts.cfg.rpo {
            for &inst in &graph.blocks[block].insts {
                let data = &graph.insts[inst];
                if let Some(result) = data.result()
                    && facts.of(result).intersects(HEAP)
                    && graph.contract(&data.op, facts).result == super::Ownership::Borrowed
                {
                    let owners = match receiver(&data.op) {
                        Some(owner) => vec![owner],
                        None => data.op.operands_iter().collect(),
                    };
                    dependencies.insert(result, owners);
                }
            }
            for (index, &param) in graph.blocks[block].params.iter().enumerate() {
                if graph.borrowed.contains(&param) {
                    let incoming = facts.cfg.preds[block]
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
                    dependencies.insert(param, incoming);
                }
            }
        }
        let anchors = graph
            .borrowed
            .iter()
            .map(|&value| {
                let mut pending = vec![value];
                let mut visited = FxHashSet::default();
                let mut roots = Vec::new();
                while let Some(current) = pending.pop() {
                    if !visited.insert(current) {
                        continue;
                    }
                    if graph.borrowed.contains(&current) {
                        if let Some(owners) = dependencies.get(&current) {
                            pending.extend(owners.iter().copied());
                        }
                    } else {
                        roots.push(current);
                    }
                }
                roots.sort();
                (value, roots)
            })
            .collect();
        let aliases = dependencies
            .keys()
            .map(|&value| (value, closure(&dependencies, value)))
            .collect();
        Self { anchors, aliases }
    }

    pub(super) fn anchors<'a>(&'a self, value: &'a Value) -> &'a [Value] {
        self.anchors
            .get(value)
            .map_or_else(|| std::slice::from_ref(value), Vec::as_slice)
    }

    pub(super) fn aliases<'a>(&'a self, value: &'a Value) -> &'a [Value] {
        self.aliases
            .get(value)
            .map_or_else(|| std::slice::from_ref(value), Vec::as_slice)
    }
}

fn closure(dependencies: &FxHashMap<Value, Vec<Value>>, value: Value) -> Vec<Value> {
    let mut pending = vec![value];
    let mut visited = FxHashSet::default();
    while let Some(current) = pending.pop() {
        if visited.insert(current)
            && let Some(owners) = dependencies.get(&current)
        {
            pending.extend(owners.iter().copied());
        }
    }
    let mut values: Vec<_> = visited.into_iter().collect();
    values.sort();
    values
}

fn stores(op: &Op) -> bool {
    matches!(
        op,
        Op::Javascript(
            JsOp::Set(..)
                | JsOp::ShapedSet(..)
                | JsOp::ArrayStore(..)
                | JsOp::PropertyStore(..)
                | JsOp::PropertyAdd(..)
                | JsOp::FillPropertyStore(..)
                | JsOp::SetIndex(..)
                | JsOp::TypedSet(..)
                | JsOp::Define(..)
                | JsOp::ArrayAppend(..)
                | JsOp::Initialize(..)
                | JsOp::Write(..)
        ) | Op::HelperCall {
            helper: crate::jit::helpers::Helper::PropertySetCached,
            ..
        }
    )
}

struct Borrows<'g> {
    graph: &'g Graph,
    facts: &'g Facts,
    chains: FxHashMap<Value, Vec<Value>>,
    dependents: FxHashMap<Value, Vec<Value>>,
    accesses: FxHashMap<Value, Vec<Access>>,
    rejected: FxHashSet<Value>,
    uses: SecondaryMap<Inst, Vec<Value>>,
}

impl Borrows<'_> {
    fn invalidates(&self, inst: Inst, write: &Access, value: Value) -> bool {
        self.accesses[&value]
            .iter()
            .any(|read| self.facts.kills(self.graph, inst, write, read))
    }

    fn redefined(&mut self, live: &Set, defined: Value) {
        if let Some(values) = self.dependents.get(&defined) {
            self.rejected.extend(
                values
                    .iter()
                    .copied()
                    .filter(|&value| value != defined && live.contains(&value)),
            );
        }
    }
}

// A borrowed read skips frame rooting and the inline temporary retain. While it is live, no
// conflicting write, reentry, or redefinition of a receiver in its chain may run, so its
// owning property slots keep holding it. Ownership proofs stop at receivers that remain rooted; borrowed
// receivers must themselves be held along the rest of the chain. When a receiver might not
// hold it (string indexing allocates), the value must die within its block: slow-path results
// stay owned by a temporary until the block's clear, and the inline path only runs on arrays,
// which own their elements.
// Indexed iteration reads only run on arrays and typed arrays, so their receiver always holds them.
pub(super) fn borrows(graph: &mut Graph, facts: &Facts, notes: &mut Vec<String>) -> Plan {
    graph.borrowed.clear();
    let mut reads = FxHashMap::default();
    let mut anchored = FxHashSet::default();
    for (_, data) in graph.insts.iter() {
        if let (Some(object), Some(result)) = (receiver(&data.op), data.result())
            && facts.of(result).intersects(HEAP)
        {
            let object = if let Some(read) = anchored_read(&data.op, facts) {
                anchored.insert(result);
                read.receiver
            } else {
                object
            };
            reads.insert(result, object);
        }
    }
    let mut chains = FxHashMap::default();
    let mut objects = FxHashSet::default();
    for (&value, &object) in &reads {
        let mut chain = vec![object];
        while let Some(&next) = reads.get(chain.last().unwrap()) {
            chain.push(next);
        }
        if anchored.contains(&value) {
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
                    .filter(|&value| facts.of(value).intersects(HEAP))
                    .collect();
                if incoming.is_empty()
                    || incoming.iter().any(|&value| {
                        !objects.contains(&value)
                            && (chains.contains_key(&value)
                                || graph.def_block(value) == block
                                || !cfg.dominates(graph.def_block(value), block))
                    })
                {
                    continue;
                }
                if !incoming.iter().any(|value| objects.contains(value)) {
                    continue;
                }
                let mut chain: Vec<_> = incoming
                    .iter()
                    .flat_map(|value| match chains.get(value) {
                        Some(owners) => owners.as_slice(),
                        None => std::slice::from_ref(value),
                    })
                    .copied()
                    .filter(|&owner| owner != param)
                    .collect();
                chain.sort();
                chain.dedup();
                if chain.is_empty() && incoming.iter().any(|value| !objects.contains(value)) {
                    continue;
                }
                if chain
                    .iter()
                    .any(|&owner| !cfg.dominates(graph.def_block(owner), block))
                {
                    continue;
                }
                chains.insert(param, chain);
                objects.insert(param);
                anchored.insert(param);
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
            if let (Op::Javascript(JsOp::ProbeGlobal(_)), Some(value)) = (&data.op, data.result()) {
                chains.insert(value, Vec::new());
                objects.insert(value);
            }
        }
    }
    if chains.is_empty() {
        return Plan::new(graph, facts);
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
    let mut dependents = FxHashMap::<Value, Vec<Value>>::default();
    for &value in chains.keys() {
        for owner in closure(&chains, value) {
            dependents.entry(owner).or_default().push(value);
        }
    }
    let mut dependencies: FxHashMap<Value, Vec<Value>> = reads
        .into_iter()
        .map(|(value, owner)| (value, vec![owner]))
        .collect();
    dependencies.extend(
        merged
            .iter()
            .map(|(&value, incoming)| (value, incoming.clone())),
    );
    let accesses = chains
        .keys()
        .map(|&value| {
            let accesses = closure(&dependencies, value)
                .into_iter()
                .filter_map(|source| {
                    graph
                        .defining_op(source)
                        .and_then(|op| Access::read(graph, op, facts))
                })
                .collect::<Vec<_>>();
            (
                value,
                if accesses.is_empty() {
                    vec![Access::broad(Locations::HEAP.union(Locations::GLOBAL))]
                } else {
                    accesses
                },
            )
        })
        .collect();
    let mut borrows = Borrows {
        graph,
        facts,
        chains,
        dependents,
        accesses,
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
                    return incoming.iter().any(|value| {
                        borrows.chains.contains_key(value) && !accepted.contains(value)
                    });
                }
                if local.contains(value) {
                    return false;
                }
                let chain = &borrows.chains[value];
                let results = std::iter::once(**value).chain(chain.iter().copied());
                for (result, &object) in results.zip(chain) {
                    if !anchored.contains(&result) {
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
        return Plan::new(graph, facts);
    }
    let mut accepted: Vec<Value> = accepted.into_iter().collect();
    accepted.sort();
    let list: Vec<String> = accepted.iter().map(ToString::to_string).collect();
    notes.push(format!("borrowed reads: [{}]", list.join(", ")));
    graph.borrowed = accepted.into_iter().collect();
    Plan::new(graph, facts)
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
        let memory = borrows.facts.memory(inst);
        let (reenters, writes) = (memory.reenters(), memory.writes_data());
        if reenters || writes {
            let write = Access::written(graph, &graph.insts[inst].op, borrows.facts, memory.writes);
            for value in live.iter() {
                if Some(value) != result && (reenters || borrows.invalidates(inst, &write, value)) {
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
