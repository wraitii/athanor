use crate::FxHashMap;
use cranelift_module::FuncId;

use super::facts::Types;
use super::{Block, Graph, Native, Op, TerminatorKind, UnaryOp, Value};
use crate::heap::TypedKind;
use crate::jit::profile::Site;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit) enum Fact {
    Type(Types),
    Code(FuncId),
    Shape(u32),
    Typed(Option<TypedKind>),
    Indexed,
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
    types: &FxHashMap<Value, Types>,
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
            Self::Type(types) if types == Types::NUMBER => Op::IsNumber(value),
            Self::Type(types) if types == Types::INTEGER => Op::IsInteger(value),
            Self::Type(types) if types == Types::STRING => Op::IsString(value),
            Self::Type(types) if types == Types::UNDEFINED => Op::IsUndefined(value),
            Self::Code(code) => Op::IsCode(value, code),
            Self::Shape(shape) => Op::IsShape(value, shape),
            Self::Typed(element) => Op::IsTyped(value, element),
            Self::Indexed => Op::IsIndexed(value),
            Self::Native(native) => Op::IsNative(value, native),
            _ => return None,
        })
    }

    pub(super) fn mask(self) -> Option<Types> {
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
        Op::IsUninitialized(value) => (value, Fact::Type(Types::UNINITIALIZED)),
        Op::IsUndefined(value) => (value, Fact::Type(Types::UNDEFINED)),
        Op::IsNullish(value) => (value, Fact::Type(Types::UNDEFINED.union(Types::NULL))),
        Op::IsString(value) => (value, Fact::Type(Types::STRING)),
        Op::IsNumber(value) => (value, Fact::Type(Types::NUMBER)),
        Op::IsInteger(value) | Op::IntegerCheck(value, ..) => (value, Fact::Type(Types::INTEGER)),
        Op::IsCode(value, code) => (value, Fact::Code(code)),
        Op::EntryAssume(value, fact) => (value, fact),
        Op::IsShape(value, shape) => (value, Fact::Shape(shape)),
        Op::IsTyped(value, element) => (value, Fact::Typed(element)),
        Op::IsIndexed(value) => (value, Fact::Indexed),
        _ => return None,
    })
}

pub(super) fn edge_of(graph: &Graph, kind: &TerminatorKind) -> Option<Edge> {
    let TerminatorKind::Branch(condition, yes, no) = kind else {
        return None;
    };
    if yes.block == no.block {
        return None;
    }
    let (condition, yes) = match graph.defining_op(*condition)? {
        Op::Unary(UnaryOp::Not, value) => (*value, no.block),
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
        Fact::Type(types) if types == Types::NUMBER => InputFact::Number,
        Fact::Type(types) if types == Types::INTEGER => InputFact::Integer,
        Fact::Shape(shape) => InputFact::Shape(shape),
        _ => return,
    };
    if !graph.entry_guards.contains(&(input, fact)) {
        graph.entry_guards.push((input, fact));
    }
}
