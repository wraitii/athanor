use cranelift_module::FuncId;

use super::facts::Types;
use super::{Block, Graph, Op, TerminatorKind, UnaryOp, Value};
use crate::heap::TypedKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Fact {
    Type(Types),
    Code(FuncId),
    Shape(u32),
    Typed(Option<TypedKind>),
    Indexed,
}

impl Fact {
    pub(super) fn mask(self) -> Option<Types> {
        match self {
            Self::Type(mask) => Some(mask),
            _ => None,
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
