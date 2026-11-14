use crate::{FxHashMap, FxHashSet};

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use oxc_syntax::operator::BinaryOperator;

use super::access::Access;
use super::guards::{Fact, implies};
use super::{Block, Cfg, Facts, Graph, Inst, Op, TerminatorKind, Value, resolve};

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum LoadKey {
    ToInt32(Value),
    NumericFloat(Value),
    IntCompare(IntCC, Value, Value),
    FloatCompare(FloatCC, Value, Value),
    SaturatingInteger(Value),
    Position(Value),
    ExactPosition(Value, Value),
    InBounds(Value, Value),
    BooleanAnd(Value, Value),
    Length(Value),
    ArrayKind(Value),
    Array(Value, super::ArrayLayout),
    Constructor(Value),
    Property(Value, String),
    Index(Value, Value),
    Global(String),
    Probe(String),
    PropertyCache(Inst),
    This,
    Argument(usize),
    Binary(BinaryOperator, Value, Value),
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
    exits: FxHashMap<Block, AvailableLoads>,
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

fn load_key(
    graph: &Graph,
    op: &Op,
    pure: bool,
    probes: bool,
    aliases: &FxHashMap<Value, Value>,
) -> Option<LoadKey> {
    let at = |value: &Value| resolve(aliases, *value);
    Some(match op {
        Op::Get(object, name) | Op::ShapedGet(object, name, _) => {
            LoadKey::Property(at(object), name.clone())
        }
        Op::GetIndex(object, key)
        | Op::TypedIndex(object, key, _)
        | Op::ArrayGet(object, key)
        | Op::ArrayLoad(object, key, _) => LoadKey::Index(at(object), at(key)),
        Op::ConstructorPrototype(callee) => LoadKey::Constructor(at(callee)),
        Op::GlobalRead(name) => LoadKey::Global(name.clone()),
        Op::ProbeGlobal(name) if probes => LoadKey::Probe(name.clone()),
        Op::NumericFloat(value) => {
            let value = at(value);
            let value = match graph.defining_op(value) {
                Some(Op::AsInteger(input))
                    if matches!(graph.defining_op(at(input)), Some(Op::SaturatingInteger(_))) =>
                {
                    at(input)
                }
                _ => value,
            };
            LoadKey::NumericFloat(value)
        }
        Op::SaturatingInteger(value) => LoadKey::SaturatingInteger(at(value)),
        Op::ToInt32(value) => LoadKey::ToInt32(at(value)),
        Op::LowLevel(super::low_level::Op::IntCompare {
            condition,
            left,
            right,
        }) if pure => LoadKey::IntCompare(*condition, at(left), at(right)),
        Op::LowLevel(super::low_level::Op::FloatCompare {
            condition,
            left,
            right,
        }) if pure => LoadKey::FloatCompare(*condition, at(left), at(right)),
        Op::ArrayPosition(value) => LoadKey::Position(at(value)),
        Op::IsExactPosition(key, position) => LoadKey::ExactPosition(at(key), at(position)),
        Op::BooleanAnd(left, right) => LoadKey::BooleanAnd(at(left), at(right)),
        Op::IsInBounds(position, length) => LoadKey::InBounds(at(position), at(length)),
        Op::IndexedLength(object) => LoadKey::Length(at(object)),
        Op::ArrayKind(object) => LoadKey::ArrayKind(at(object)),
        Op::IsArray(object, layout) => LoadKey::Array(at(object), *layout),
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
            (
                value,
                fact @ (Fact::Type(_)
                | Fact::Code(_)
                | Fact::Typed(_)
                | Fact::Shape(_)
                | Fact::Cached(..)),
            ) => LoadKey::Check(at(&value), fact),
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
    aliases: &FxHashMap<Value, Value>,
    state: &mut AvailableLoads,
) -> Option<Reuse> {
    let data = &graph.insts[inst];
    let memory = facts.memory(inst);
    if memory.reenters() {
        state.retain(|key, _| {
            matches!(
                key,
                LoadKey::NumericFloat(_) | LoadKey::SaturatingInteger(_)
            )
        });
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
    let key = load_key(graph, &data.op, facts.is_pure(inst), probes, aliases)?;
    let [result] = data.results.as_slice() else {
        return None;
    };
    let result = *result;
    if let Some(entry) = state.get(&key) {
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
    let read = read_access(&data.op, facts).map(|mut read| {
        read.base = read.base.map(|base| resolve(aliases, base));
        read
    });
    state.insert(
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
    exits: &FxHashMap<Block, AvailableLoads>,
    block: Block,
    aliases: &FxHashMap<Value, Value>,
) -> AvailableLoads {
    if block == graph.entry {
        return AvailableLoads::default();
    }
    let mut incoming = cfg.preds[block].iter().filter_map(|pred| {
        let mut state = exits.get(pred)?.clone();
        if let Some(terminator) = &graph.blocks[*pred].terminator
            && let TerminatorKind::Branch(condition, yes, no) = &terminator.kind
            && yes.block != no.block
        {
            let (condition, inverted) = match super::guards::negated(graph, *condition) {
                Some(value) => (value, true),
                _ => (*condition, false),
            };
            let condition = resolve(aliases, condition);
            for entry in state.values_mut() {
                if resolve(aliases, entry.value) == condition {
                    entry.truth = Some((yes.block == block) != inverted);
                }
            }
        }
        Some(state)
    });
    let Some(first) = incoming.next() else {
        return AvailableLoads::default();
    };
    let mut state = first;
    for other in incoming {
        state.retain(|key, entry| {
            let Some(other) = other.get(key) else {
                return false;
            };
            if entry.value != other.value || entry.read != other.read {
                return false;
            }
            if entry.truth != other.truth {
                entry.truth = None;
            }
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
    let mut exits: FxHashMap<Block, AvailableLoads> = FxHashMap::default();
    let mut previous = FxHashMap::default();
    for _ in 0..16 {
        let mut changed = false;
        let mut aliases = FxHashMap::default();
        let mut reused = Vec::new();
        let mut folded = Vec::new();
        for &block in &cfg.rpo {
            let mut state = entering(graph, cfg, &exits, block, &aliases);
            let is_cold = cold.contains(&block);
            for &inst in &graph.blocks[block].insts {
                if let Some(value) = step(graph, facts, inst, is_cold, probes, &aliases, &mut state)
                {
                    match value {
                        Reuse::Value(value) => {
                            aliases.insert(graph.insts[inst].result().unwrap(), value);
                            reused.push((inst, value));
                        }
                        Reuse::Boolean(value) => folded.push((inst, value)),
                    }
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
                folded,
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
        state
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
            matches!(key, LoadKey::Check(known, established)
            if original(*known) == value && established.implies(fact) && entry.truth == Some(true))
        })
    }
}
