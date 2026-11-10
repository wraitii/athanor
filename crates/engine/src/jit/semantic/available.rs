use std::collections::{HashMap, HashSet};

use oxc_syntax::operator::BinaryOperator;

use super::access::Access;
use super::guards::{Fact, implies};
use super::{Block, Cfg, Facts, Graph, Inst, Op, Value, resolve};

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum LoadKey {
    ToInt32(Value),
    Position(Value),
    ExactPosition(Value, Value),
    InBounds(Value, Value),
    BooleanAnd(Value, Value),
    Length(Value),
    Array(Value, bool),
    Constructor(Value),
    Property(Value, String),
    Index(Value, Value),
    Global(String),
    Probe(String),
    This,
    Argument(usize),
    Binary(BinaryOperator, Value, Value),
    Check(Value, Fact),
}

#[derive(Clone, PartialEq)]
pub(super) struct Available {
    pub(super) value: Value,
    read: Option<Access>,
}

pub(super) type AvailableLoads = HashMap<LoadKey, Available>;

pub(super) struct Solved {
    exits: HashMap<Block, AvailableLoads>,
    aliases: HashMap<Value, Value>,
    pub(super) reused: Vec<(Inst, Value)>,
    cold: HashSet<Block>,
    probes: bool,
}

fn load_key(op: &Op, pure: bool, probes: bool, aliases: &HashMap<Value, Value>) -> Option<LoadKey> {
    let at = |value: &Value| resolve(aliases, *value);
    Some(match op {
        Op::Get(object, name) | Op::ShapedGet(object, name, _) => {
            LoadKey::Property(at(object), name.clone())
        }
        Op::GetIndex(object, key)
        | Op::TypedIndex(object, key, _)
        | Op::IndexedGet(object, key)
        | Op::ArrayLoad(object, key, _) => LoadKey::Index(at(object), at(key)),
        Op::ConstructorPrototype(callee) => LoadKey::Constructor(at(callee)),
        Op::GlobalRead(name) => LoadKey::Global(name.clone()),
        Op::ProbeGlobal(name) if probes => LoadKey::Probe(name.clone()),
        Op::ToInt32(value) => LoadKey::ToInt32(at(value)),
        Op::ArrayPosition(value) => LoadKey::Position(at(value)),
        Op::IsExactPosition(key, position) => LoadKey::ExactPosition(at(key), at(position)),
        Op::BooleanAnd(left, right) => LoadKey::BooleanAnd(at(left), at(right)),
        Op::IsInBounds(position, length) => LoadKey::InBounds(at(position), at(length)),
        Op::IndexedLength(object) => LoadKey::Length(at(object)),
        Op::IsArray(object, holey) => LoadKey::Array(at(object), *holey),
        Op::This => LoadKey::This,
        Op::Argument(index) => LoadKey::Argument(*index),
        Op::Binary(operator, left, right)
        | Op::GuardedCompare {
            operator,
            left,
            right,
            ..
        } if pure => LoadKey::Binary(*operator, at(left), at(right)),
        _ => match implies(op)? {
            (value, fact @ (Fact::Type(_) | Fact::Code(_) | Fact::Typed(_) | Fact::Shape(_))) => {
                LoadKey::Check(at(&value), fact)
            }
            _ => return None,
        },
    })
}

fn read_access(op: &Op, facts: &Facts) -> Option<Access> {
    match op {
        Op::ProbeGlobal(name) => Access::read(&Op::GlobalRead(name.clone()), facts),
        _ => Access::read(op, facts),
    }
}

fn step(
    graph: &Graph,
    facts: &Facts,
    inst: Inst,
    cold: bool,
    probes: bool,
    aliases: &HashMap<Value, Value>,
    state: &mut AvailableLoads,
) -> Option<Value> {
    let data = &graph.insts[inst];
    let memory = facts.memory(inst);
    if memory.reenters() {
        state.clear();
        return None;
    }
    if memory.writes_state() {
        let write = Access::written(&data.op, facts, memory.writes);
        state.retain(|_, entry| {
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
    let result = data.result?;
    let key = load_key(&data.op, facts.is_pure(inst), probes, aliases)?;
    if let Some(entry) = state.get(&key) {
        return Some(entry.value);
    }
    let read = read_access(&data.op, facts).map(|mut read| {
        read.base = read.base.map(|base| resolve(aliases, base));
        read
    });
    state.insert(
        key,
        Available {
            value: result,
            read,
        },
    );
    None
}

fn entering(
    graph: &Graph,
    cfg: &Cfg,
    exits: &HashMap<Block, AvailableLoads>,
    block: Block,
) -> AvailableLoads {
    if block == graph.entry {
        return AvailableLoads::new();
    }
    let mut incoming = cfg.preds[block].iter().filter_map(|pred| exits.get(pred));
    let Some(first) = incoming.next() else {
        return AvailableLoads::new();
    };
    let mut state = first.clone();
    for other in incoming {
        state.retain(|key, entry| other.get(key) == Some(entry));
    }
    state
}

pub(super) fn solve(
    graph: &Graph,
    facts: &Facts,
    cfg: &Cfg,
    cold: HashSet<Block>,
    probes: bool,
) -> Option<Solved> {
    let mut exits: HashMap<Block, AvailableLoads> = HashMap::new();
    let mut previous = HashMap::new();
    for _ in 0..16 {
        let mut changed = false;
        let mut aliases = HashMap::new();
        let mut reused = Vec::new();
        for &block in &cfg.rpo {
            let mut state = entering(graph, cfg, &exits, block);
            let is_cold = cold.contains(&block);
            for &inst in &graph.blocks[block].insts {
                if let Some(value) = step(graph, facts, inst, is_cold, probes, &aliases, &mut state)
                {
                    aliases.insert(graph.insts[inst].result.unwrap(), value);
                    reused.push((inst, value));
                }
            }
            if exits.get(&block) != Some(&state) {
                exits.insert(block, state);
                changed = true;
            }
        }
        if !changed && aliases == previous {
            return Some(Solved {
                exits,
                aliases,
                reused,
                cold,
                probes,
            });
        }
        previous = aliases;
    }
    None
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
        let mut state = entering(graph, cfg, &self.exits, block);
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
        state
    }

    pub(super) fn canonical(&self, value: Value) -> Value {
        resolve(&self.aliases, value)
    }
}
