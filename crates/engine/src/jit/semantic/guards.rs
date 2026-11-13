use crate::FxHashMap;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_module::FuncId;

use super::facts::ValueType;
use super::low_level;
use super::{Block, Graph, Native, Op, TerminatorKind, UnaryOp, Value};
use crate::heap::TypedKind;
use crate::jit::profile::Site;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit) enum Fact {
    Type(ValueType),
    Code(FuncId),
    Shape(u32),
    Typed(Option<TypedKind>),
    Indexed,
    // Planning resolves this IC proposal into a shape or cache-hit requirement.
    Property(Value),
    Cached(Value, bool),
    Native(Native),
    Bound(Value),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit) enum Kind {
    Number,
    Receiver,
    Bound,
    Identity,
    Native,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit) struct Origin {
    pub(in crate::jit) site: Option<Site>,
    pub(in crate::jit) kind: Kind,
    pub(in crate::jit) gain: u16,
}

pub(super) fn assumed(
    types: &FxHashMap<Value, ValueType>,
    code: &FxHashMap<Value, FuncId>,
) -> Vec<(Value, Fact)> {
    let mut facts: Vec<_> = types
        .iter()
        .map(|(&value, &types)| (value, Fact::Type(types)))
        .chain(code.iter().map(|(&value, &code)| (value, Fact::Code(code))))
        .collect();
    facts.sort_by_key(|&(value, _)| value);
    facts
}

impl Fact {
    pub(super) fn test(self, value: Value) -> Option<Op> {
        Some(match self {
            Self::Type(types) if types == ValueType::NUMBER => Op::IsNumber(value),
            Self::Type(types) if types == ValueType::INTEGER => Op::IsInteger(value),
            Self::Type(types) if types == ValueType::STRING => Op::IsString(value),
            Self::Type(types) if types == ValueType::UNDEFINED => Op::IsUndefined(value),
            Self::Code(code) => Op::IsCode(value, code),
            Self::Cached(cache, builtins) => Op::IsPropertyCached(value, cache, builtins),
            Self::Shape(shape) => Op::IsShape(value, shape),
            Self::Typed(element) => Op::IsTyped(value, element),
            Self::Indexed => Op::IsIndexed(value),
            Self::Native(native) => Op::IsNative(value, native),
            _ => return None,
        })
    }

    pub(super) fn mask(self) -> Option<ValueType> {
        match self {
            Self::Type(mask) => Some(mask),
            _ => None,
        }
    }

    pub(super) fn implies(self, other: Self) -> bool {
        match (self, other) {
            (Self::Type(known), Self::Type(wanted)) => known.within(wanted),
            (Self::Typed(Some(_)), Self::Typed(None)) => true,
            _ => self == other,
        }
    }

    pub(super) fn code(self) -> Option<FuncId> {
        match self {
            Self::Code(code) => Some(code),
            _ => None,
        }
    }
}

pub(super) struct Edge {
    pub(super) yes: Block,
    pub(super) value: Value,
    pub(super) fact: Fact,
}

pub(super) fn implies(op: &Op) -> Option<(Value, Fact)> {
    Some(match *op {
        Op::IsUninitialized(value) => (value, Fact::Type(ValueType::UNINITIALIZED)),
        Op::IsUndefined(value) => (value, Fact::Type(ValueType::UNDEFINED)),
        Op::IsNullish(value) => (
            value,
            Fact::Type(ValueType::UNDEFINED.union(ValueType::NULL)),
        ),
        Op::IsString(value) => (value, Fact::Type(ValueType::STRING)),
        Op::IsNumber(value) => (value, Fact::Type(ValueType::NUMBER)),
        Op::IsInteger(value) | Op::IntegerCheck(value, ..) => {
            (value, Fact::Type(ValueType::INTEGER))
        }
        Op::IsCode(value, code) => (value, Fact::Code(code)),
        Op::EntryAssume(value, fact) => (value, fact),
        Op::IsShape(value, shape) => (value, Fact::Shape(shape)),
        Op::IsTyped(value, element) => (value, Fact::Typed(element)),
        Op::IsIndexed(value) => (value, Fact::Indexed),
        Op::IsPropertyCached(value, cache, builtins) => (value, Fact::Cached(cache, builtins)),
        _ => return None,
    })
}

pub(super) fn negated(graph: &Graph, value: Value) -> Option<Value> {
    match *graph.defining_op(value)? {
        Op::Unary(UnaryOp::Not, value) => Some(value),
        Op::LowLevel(low_level::Op::IntCompare {
            condition: IntCC::Equal,
            left,
            right,
        }) => {
            let is_false = |value| {
                matches!(
                    graph.defining_op(value),
                    Some(Op::Constant(super::Constant::Boolean(false)))
                )
            };
            if is_false(right) {
                Some(left)
            } else if is_false(left) {
                Some(right)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn edge_of(graph: &Graph, kind: &TerminatorKind) -> Option<Edge> {
    let TerminatorKind::Branch(condition, yes, no) = kind else {
        return None;
    };
    if yes.block == no.block {
        return None;
    }
    let (condition, yes) = match negated(graph, *condition) {
        Some(value) => (value, no.block),
        _ => (*condition, yes.block),
    };
    let (value, fact) = implies(graph.defining_op(condition)?)?;
    Some(Edge { yes, value, fact })
}

pub(super) fn record_entry(graph: &mut Graph, value: Value, fact: Fact) {
    use crate::jit::variants::{Fact as InputFact, Input};
    let input = match graph.defining_op(value) {
        Some(Op::This) => Input::This,
        Some(Op::Argument(index)) => Input::Argument(*index),
        _ => return,
    };
    let fact = match fact {
        Fact::Type(types) if types == ValueType::NUMBER => InputFact::Number,
        Fact::Type(types) if types == ValueType::INTEGER => InputFact::Integer,
        Fact::Shape(shape) => InputFact::Shape(shape),
        _ => return,
    };
    if !graph.entry_guards.contains(&(input, fact)) {
        graph.entry_guards.push((input, fact));
    }
}

pub(super) fn holds(
    graph: &Graph,
    cfg: &super::Cfg,
    block: Block,
    value: Value,
    fact: Fact,
) -> bool {
    let original = |mut value| {
        while let Some(Op::AsObject(source)) = graph.defining_op(value) {
            value = *source;
        }
        value
    };
    let value = original(value);
    cfg.rpo.iter().any(|&guard| {
        graph.blocks[guard]
            .terminator
            .as_ref()
            .and_then(|terminator| edge_of(graph, &terminator.kind))
            .is_some_and(|edge| {
                original(edge.value) == value
                    && edge.fact.implies(fact)
                    && edge.yes != graph.entry
                    && cfg.preds[edge.yes].len() == 1
                    && cfg.dominates(edge.yes, block)
            })
    })
}
