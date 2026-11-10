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
    types: &std::collections::HashMap<Value, Types>,
    code: &std::collections::HashMap<Value, FuncId>,
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
        Op::IsInteger(value) => (value, Fact::Type(Types::INTEGER)),
        Op::IsCode(value, code) => (value, Fact::Code(code)),
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
