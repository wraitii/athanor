use crate::jit::layout::ReadLayout;
use crate::{FxHashMap, FxHashSet};
use cranelift_codegen::entity::SecondaryMap;
use smallvec::SmallVec;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::rc::Rc;

use super::guards::{Fact, implies};
use super::memory::Access;
use super::{Block, Cfg, Facts, Graph, Inst, Op, TerminatorKind, Value, resolve};

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum LoadKey {
    Load(
        cranelift_codegen::ir::Type,
        Value,
        i32,
        cranelift_codegen::ir::MemFlagsData,
    ),
    Length(Value),
    ArrayKind(Value),
    Array(Value, super::ArrayLayout),
    Constructor(Value),
    Cell(Value),
    Property(Value, String),
    Index(Value, Value),
    Global(String),
    Probe(String),
    ShapeProbe(Value, String, ReadLayout),
    PropertyCache(Inst),
    This,
    Argument(usize),
    Check(Value, Fact),
}

#[derive(Clone, PartialEq)]
pub(super) struct Available {
    pub(super) value: Value,
    read: Option<Access>,
    truth: Option<bool>,
}

pub(super) type AvailableLoads = FxHashMap<LoadKey, Available>;

pub(super) struct Solved {
    exits: SecondaryMap<Block, Option<Rc<AvailableLoads>>>,
    aliases: FxHashMap<Value, Value>,
    pub(super) reused: Vec<(Inst, Value)>,
    pub(super) folded: Vec<(Inst, bool)>,
    cold: FxHashSet<Block>,
    probes: bool,
}

enum Reuse {
    Value(Value),
    Boolean(bool),
}

fn load_key(op: &Op, probes: bool, aliases: &FxHashMap<Value, Value>) -> Option<LoadKey> {
    let at = |value: &Value| resolve(aliases, *value);
    Some(match op {
        Op::Get(object, name) | Op::ShapedGet(object, name, _) | Op::LayoutGet(object, name, _) => {
            LoadKey::Property(at(object), name.clone())
        }
        Op::GetIndex(object, key)
        | Op::TypedIndex(object, key, _)
        | Op::ArrayGet(object, key)
        | Op::ArrayLoad(object, key, _) => LoadKey::Index(at(object), at(key)),
        Op::ConstructorPrototype(callee) => LoadKey::Constructor(at(callee)),
        Op::CellGet(cell) => LoadKey::Cell(at(cell)),
        Op::GlobalRead(name) => LoadKey::Global(name.clone()),
        Op::ProbeGlobal(name) if probes => LoadKey::Probe(name.clone()),
        Op::ProbeShapeProperty(object, name, layout) => {
            LoadKey::ShapeProbe(at(object), name.clone(), *layout)
        }
        Op::LowLevel(super::low_level::Op::Load {
            ty,
            address,
            offset,
            flags,
        }) => LoadKey::Load(*ty, at(address), *offset, *flags),
        Op::IndexedLength(object) => LoadKey::Length(at(object)),
        Op::ArrayKind(object) => LoadKey::ArrayKind(at(object)),
        Op::IsArray(object, layout) => LoadKey::Array(at(object), *layout),
        _ => match implies(op)? {
            (
                value,
                fact @ (Fact::Type(_)
                | Fact::Code(_)
                | Fact::Typed(_)
                | Fact::Array
                | Fact::Shape(_)
                | Fact::ReadLayout(_)
                | Fact::Cached(..)),
            ) => LoadKey::Check(at(&value), fact),
            _ => return None,
        },
    })
}

fn read_access(graph: &Graph, op: &Op, facts: &Facts) -> Option<Access> {
    match op {
        Op::LowLevel(super::low_level::Op::Load { .. }) => {
            Some(Access::broad(super::super::effects::Locations::HEAP))
        }
        Op::ProbeGlobal(name) => Access::read(graph, &Op::GlobalRead(name.clone()), facts),
        _ => Access::read(graph, op, facts),
    }
}

fn step(
    graph: &Graph,
    facts: &Facts,
    inst: Inst,
    cold: bool,
    probes: bool,
    aliases: &FxHashMap<Value, Value>,
    state: &mut Rc<AvailableLoads>,
) -> Option<Reuse> {
    let data = &graph.insts[inst];
    let memory = facts.memory(inst);
    if memory.reenters() {
        if state.values().any(|entry| entry.read.is_some()) {
            Rc::make_mut(state).retain(|_, entry| entry.read.is_none());
        }
        return None;
    }
    if memory.writes_state() {
        if state.is_empty() {
            return None;
        }
        let write = Access::written(graph, &data.op, facts, memory.writes);
        Rc::make_mut(state).retain(|_, entry| {
            entry
                .read
                .as_ref()
                .is_none_or(|read| !facts.kills(graph, inst, &write, read))
        });
        return None;
    }
    if cold {
        return None;
    }
    if let Some((object, wanted)) = match data.op {
        Op::IsShape(object, shape) => Some((object, Fact::Shape(shape))),
        Op::IsReadLayout(object, layout) => Some((object, Fact::ReadLayout(layout))),
        _ => None,
    } && state.iter().any(|(key, entry)| {
        matches!(key, LoadKey::ShapeProbe(receiver, _, layout)
            if *receiver == resolve(aliases, object)
                && (layout.requirement() == wanted || Fact::Shape(layout.shape) == wanted)
                && entry.truth == Some(true))
    }) {
        return Some(Reuse::Boolean(true));
    }
    let key = load_key(&data.op, probes, aliases)?;
    let [result] = data.results.as_slice() else {
        return None;
    };
    let result = *result;
    if let Some(entry) = state.get(&key)
        && (!matches!(key, LoadKey::Load(..)) || graph.repr(entry.value) == graph.repr(result))
    {
        return Some(Reuse::Value(entry.value));
    }
    let excluded = match &key {
        LoadKey::Check(object, Fact::Shape(shape)) => state.iter().any(|(known, entry)| {
            let LoadKey::Check(other, Fact::Shape(known_shape)) = known else {
                return false;
            };
            entry.truth == Some(true) && object == other && shape != known_shape
        }),
        _ => false,
    };
    let read = read_access(graph, &data.op, facts).map(|mut read| {
        if let Some(js) = &mut read.js {
            js.base = js.base.map(|base| resolve(aliases, base));
        }
        read
    });
    Rc::make_mut(state).insert(
        key,
        Available {
            value: result,
            read,
            truth: if excluded {
                Some(false)
            } else if matches!(data.op, Op::EntryAssume(..)) {
                Some(true)
            } else {
                None
            },
        },
    );
    excluded.then_some(Reuse::Boolean(false))
}

fn entering(
    graph: &Graph,
    cfg: &Cfg,
    exits: &SecondaryMap<Block, Option<Rc<AvailableLoads>>>,
    block: Block,
    aliases: &FxHashMap<Value, Value>,
) -> Rc<AvailableLoads> {
    if block == graph.entry {
        return Rc::default();
    }
    let mut incoming = cfg.preds[block]
        .iter()
        .filter_map(|pred| exits[*pred].as_ref().map(|state| (*pred, state)));
    let Some((first, mut state)) = incoming
        .next()
        .map(|(pred, state)| (pred, Rc::clone(state)))
    else {
        return Rc::default();
    };
    let edge_truth = |pred: Block| {
        let terminator = graph.blocks[pred].terminator.as_ref()?;
        let TerminatorKind::Branch(condition, yes, no) = &terminator.kind else {
            return None;
        };
        if yes.block == no.block {
            return None;
        }
        let (condition, inverted) = match super::guards::negated(graph, *condition) {
            Some(value) => (value, true),
            _ => (*condition, false),
        };
        Some((
            resolve(aliases, condition),
            (yes.block == block) != inverted,
        ))
    };
    let truth_on_edge = |key: &LoadKey, entry: &Available, edge: Option<(Value, bool)>| {
        let Some((condition, truth)) = edge else {
            return entry.truth;
        };
        if matches!(key, LoadKey::ShapeProbe(..)) {
            // Probe truth records success, independently of the property's value.
            if let Some(Op::IsUninitialized(value)) = graph.defining_op(condition)
                && resolve(aliases, *value) == resolve(aliases, entry.value)
            {
                return Some(!truth);
            }
            return entry.truth;
        }
        if resolve(aliases, entry.value) == condition {
            Some(truth)
        } else {
            entry.truth
        }
    };
    let first_edge = edge_truth(first);
    if state
        .iter()
        .any(|(key, entry)| entry.truth != truth_on_edge(key, entry, first_edge))
    {
        for (key, entry) in Rc::make_mut(&mut state) {
            entry.truth = truth_on_edge(key, entry, first_edge);
        }
    }
    for (pred, other) in incoming {
        let edge = edge_truth(pred);
        let merged = |key: &LoadKey, entry: &Available| {
            let other = other.get(key)?;
            if entry.value != other.value || entry.read != other.read {
                return None;
            }
            Some(if entry.truth == truth_on_edge(key, other, edge) {
                entry.truth
            } else {
                None
            })
        };
        if state
            .iter()
            .all(|(key, entry)| merged(key, entry) == Some(entry.truth))
        {
            continue;
        }
        Rc::make_mut(&mut state).retain(|key, entry| {
            let Some(truth) = merged(key, entry) else {
                return false;
            };
            entry.truth = truth;
            true
        });
    }
    state
}

pub(super) fn solve(
    graph: &Graph,
    facts: &Facts,
    cfg: &Cfg,
    cold: FxHashSet<Block>,
    probes: bool,
) -> Option<Solved> {
    let mut scan = SecondaryMap::<Block, SmallVec<[Inst; 4]>>::new();
    let mut users = SecondaryMap::<Value, SmallVec<[Block; 2]>>::new();
    let mut ranks = SecondaryMap::<Block, usize>::new();
    let mut aliases = FxHashMap::default();
    for (rank, &block) in cfg.rpo.iter().enumerate() {
        ranks[block] = rank;
        for &inst in &graph.blocks[block].insts {
            let data = &graph.insts[inst];
            let memory = facts.memory(inst);
            if !memory.reenters()
                && !memory.writes_state()
                && (cold.contains(&block)
                    || data.results.len() != 1
                    || load_key(&data.op, probes, &aliases).is_none())
            {
                continue;
            }
            scan[block].push(inst);
            for value in data.op.operands_iter() {
                if graph.def_block(value) != block && users[value].last() != Some(&block) {
                    users[value].push(block);
                }
            }
        }
        if let Some(terminator) = &graph.blocks[block].terminator {
            for value in terminator.kind.operands_iter() {
                if graph.def_block(value) != block && users[value].last() != Some(&block) {
                    users[value].push(block);
                }
            }
            if let TerminatorKind::Branch(condition, yes, no) = &terminator.kind
                && yes.block != no.block
            {
                let condition = super::guards::negated(graph, *condition).unwrap_or(*condition);
                users[condition].extend([yes.block, no.block]);
            }
        }
    }
    let mut exits = SecondaryMap::<Block, Option<Rc<AvailableLoads>>>::new();
    let mut folded = SecondaryMap::<Inst, Option<bool>>::new();
    let mut dependents = SecondaryMap::<Value, SmallVec<[Value; 2]>>::new();
    let mut pending: BinaryHeap<_> = (0..cfg.rpo.len()).map(Reverse).collect();
    let mut queued = vec![true; cfg.rpo.len()];
    let mut visits = cfg.rpo.len().saturating_mul(16);
    let mut changed_aliases = Vec::new();
    let mut seen = SecondaryMap::<Value, usize>::new();
    let mut generation = 0;
    while let Some(Reverse(rank)) = pending.pop() {
        if visits == 0 {
            return None;
        }
        visits -= 1;
        queued[rank] = false;
        let block = cfg.rpo[rank];
        let mut state = entering(graph, cfg, &exits, block, &aliases);
        // Backedge states start optimistically; keep aliases dominating while they are revised.
        let dominating = |entry: &Available| {
            let owner = graph.def_block(entry.value);
            owner != block && cfg.dominates(owner, block)
        };
        if !state.values().all(dominating) {
            Rc::make_mut(&mut state).retain(|_, entry| dominating(entry));
        }
        let is_cold = cold.contains(&block);
        for &inst in &scan[block] {
            let reuse = step(graph, facts, inst, is_cold, probes, &aliases, &mut state);
            let replacement = match reuse {
                Some(Reuse::Value(value)) => Some(value),
                _ => None,
            };
            folded[inst] = match reuse {
                Some(Reuse::Boolean(value)) => Some(value),
                _ => None,
            };
            let [result] = graph.insts[inst].results.as_slice() else {
                continue;
            };
            let result = *result;
            if aliases.get(&result).copied() == replacement {
                continue;
            }
            match replacement {
                Some(value) => {
                    aliases.insert(result, value);
                    if !dependents[value].contains(&result) {
                        dependents[value].push(result);
                    }
                }
                None => {
                    aliases.remove(&result);
                }
            }
            changed_aliases.push(result);
        }
        let mut enqueue = |block: Block| {
            let rank = ranks[block];
            if !queued[rank] {
                queued[rank] = true;
                pending.push(Reverse(rank));
            }
        };
        if exits[block]
            .as_ref()
            .is_none_or(|exit| !Rc::ptr_eq(exit, &state) && **exit != *state)
        {
            exits[block] = Some(state);
            for successor in graph.successors(block) {
                enqueue(successor);
            }
        }
        // Alias changes can alter load keys without changing this block's exit state.
        generation += 1;
        while let Some(value) = changed_aliases.pop() {
            if seen[value] == generation {
                continue;
            }
            seen[value] = generation;
            for &user in &users[value] {
                enqueue(user);
            }
            for &dependent in &dependents[value] {
                if aliases.get(&dependent) == Some(&value) {
                    changed_aliases.push(dependent);
                }
            }
        }
    }
    let mut reused = Vec::new();
    let mut folded_values = Vec::new();
    for &block in &cfg.rpo {
        for &inst in &scan[block] {
            if let [result] = graph.insts[inst].results.as_slice()
                && let Some(&value) = aliases.get(result)
            {
                reused.push((inst, value));
            }
            if let Some(value) = folded[inst] {
                folded_values.push((inst, value));
            }
        }
    }
    Some(Solved {
        exits,
        aliases,
        reused,
        folded: folded_values,
        cold,
        probes,
    })
}

impl Solved {
    pub(super) fn before(
        &self,
        graph: &Graph,
        facts: &Facts,
        cfg: &Cfg,
        inst: Inst,
    ) -> AvailableLoads {
        let block = graph.insts[inst].block;
        let mut state = entering(graph, cfg, &self.exits, block, &self.aliases);
        let cold = self.cold.contains(&block);
        for &earlier in &graph.blocks[block].insts {
            if earlier == inst {
                break;
            }
            step(
                graph,
                facts,
                earlier,
                cold,
                self.probes,
                &self.aliases,
                &mut state,
            );
        }
        Rc::unwrap_or_clone(state)
    }

    pub(super) fn canonical(&self, value: Value) -> Value {
        resolve(&self.aliases, value)
    }

    pub(super) fn holds(
        &self,
        graph: &Graph,
        state: &AvailableLoads,
        value: Value,
        fact: Fact,
    ) -> bool {
        let original = |mut value| {
            value = self.canonical(value);
            while let Some(Op::AsObject(source)) = graph.defining_op(value) {
                value = self.canonical(*source);
            }
            value
        };
        let value = original(value);
        state.iter().any(|(key, entry)| {
            if let LoadKey::ShapeProbe(known, _, layout) = key {
                return original(*known) == value
                    && (fact == layout.requirement() || fact == Fact::Shape(layout.shape))
                    && entry.truth == Some(true);
            }
            matches!(key, LoadKey::Check(known, established)
            if original(*known) == value && established.implies(fact) && entry.truth == Some(true))
        })
    }
}
