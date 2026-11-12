use crate::heap::MathFunction;
use crate::jit::helpers::Helper;
use oxc_syntax::operator::BinaryOperator;

use super::{
    Constant, Facts, Graph, Inst, MemoryFormat, MemoryOp, Op, Ownership, Repr, TypedRead, UnaryOp,
    Value, ValueType,
};
use crate::jit::math::METHODS;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Use {
    Any,
    Exact(Repr),
}

pub(in crate::jit) struct Representations {
    pub(in crate::jit) uses: Vec<Use>,
    pub(in crate::jit) produces: Option<Repr>,
}

const NUMERIC: ValueType = ValueType::NUMBER.union(ValueType::UNDEFINED);

fn integer(repr: Repr) -> bool {
    matches!(repr, Repr::I32 | Repr::I64)
}

fn binary(operator: BinaryOperator, left: Repr, right: Repr) -> (Repr, Repr) {
    use BinaryOperator as B;
    match operator {
        B::Addition | B::Subtraction if integer(left) && integer(right) => (Repr::I64, Repr::I64),
        B::Addition | B::Subtraction | B::Multiplication | B::Division => (Repr::F64, Repr::F64),
        B::LessThan
        | B::LessEqualThan
        | B::GreaterThan
        | B::GreaterEqualThan
        | B::Equality
        | B::StrictEquality
        | B::Inequality
        | B::StrictInequality => {
            if !integer(left) || !integer(right) {
                (Repr::F64, Repr::I8)
            } else if left == Repr::I32 && right == Repr::I32 {
                (Repr::I32, Repr::I8)
            } else {
                (Repr::I64, Repr::I8)
            }
        }
        B::ShiftRightZeroFill => (Repr::I32, Repr::F64),
        _ => (Repr::I32, Repr::I32),
    }
}

fn typed_binary(
    graph: &Graph,
    facts: &Facts,
    operator: BinaryOperator,
    left: Value,
    right: Value,
) -> Option<Representations> {
    use BinaryOperator as B;
    let (left_repr, right_repr) = (graph.repr(left), graph.repr(right));
    if left_repr == Repr::I8
        && right_repr == Repr::I8
        && matches!(operator, B::StrictEquality | B::StrictInequality)
    {
        return Some(Representations {
            uses: vec![Use::Any, Use::Any],
            produces: Some(Repr::I8),
        });
    }
    let applicable = match operator {
        B::Equality | B::StrictEquality | B::Inequality | B::StrictInequality => {
            facts.of(left).within(ValueType::NUMBER) && facts.of(right).within(ValueType::NUMBER)
        }
        B::Instanceof | B::In | B::Remainder | B::Exponential => false,
        _ => facts.of(left).within(NUMERIC) && facts.of(right).within(NUMERIC),
    };
    if !applicable {
        return None;
    }
    let (operand, result) = binary(operator, left_repr, right_repr);
    Some(Representations {
        uses: vec![Use::Exact(operand), Use::Exact(operand)],
        produces: Some(result),
    })
}

fn view_object(graph: &Graph, inst: Inst) -> Use {
    if graph.views.contains_key(&inst) {
        Use::Any
    } else {
        Use::Exact(Repr::Boxed)
    }
}

fn key(graph: &Graph, value: Value) -> Use {
    if graph.repr(value) == Repr::I8 {
        Use::Exact(Repr::Boxed)
    } else {
        Use::Any
    }
}

fn position(graph: &Graph, value: Value) -> Use {
    if integer(graph.repr(value)) {
        Use::Exact(Repr::I64)
    } else {
        Use::Exact(Repr::F64)
    }
}

fn element(kind: crate::heap::TypedKind) -> Repr {
    use crate::heap::TypedKind as K;
    match kind {
        K::Float64 | K::Float32 | K::Uint32 => Repr::F64,
        _ => Repr::I32,
    }
}

fn typed(graph: &Graph, facts: &Facts, inst: Inst) -> Option<Representations> {
    let data = &graph.insts[inst];
    let target = data.result().map(|value| graph.repr(value));
    let numeric = |value: Value| facts.of(value).within(NUMERIC);
    let exact = |reprs: &[Repr], produces: Repr| Representations {
        uses: reprs.iter().map(|&repr| Use::Exact(repr)).collect(),
        produces: Some(produces),
    };
    let any = |count: usize, produces: Option<Repr>| Representations {
        uses: vec![Use::Any; count],
        produces,
    };
    Some(match &data.op {
        Op::Convert(_, repr) => any(1, Some(*repr)),
        Op::Constant(Constant::Number(_)) => match target {
            Some(repr @ (Repr::F64 | Repr::I64 | Repr::I32)) => any(0, Some(repr)),
            _ => return None,
        },
        Op::Constant(Constant::Boolean(_)) if target == Some(Repr::I8) => any(0, Some(Repr::I8)),
        Op::Binary(BinaryOperator::Multiplication, ..) if target == Some(Repr::I64) => {
            exact(&[Repr::I64, Repr::I64], Repr::I64)
        }
        Op::Binary(operator, left, right) => typed_binary(graph, facts, *operator, *left, *right)?,
        Op::GuardedCompare {
            operator,
            left,
            right,
            numbers,
        } => {
            let mut contract = typed_binary(graph, facts, *operator, *left, *right)?;
            contract.uses.extend(numbers.iter().map(|_| Use::Any));
            contract
        }
        Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, value) if numeric(*value) => {
            match target {
                Some(repr @ (Repr::I32 | Repr::I64)) => exact(&[repr], repr),
                _ => exact(&[Repr::F64], Repr::F64),
            }
        }
        Op::Unary(UnaryOp::Negate, value) if numeric(*value) => exact(&[Repr::F64], Repr::F64),
        Op::TypedData(_) => any(1, Some(Repr::I64)),
        Op::Memory(MemoryOp::Address { .. }) => any(2, Some(Repr::I64)),
        Op::Memory(MemoryOp::Load { format, .. }) => any(1, Some(format.repr())),
        Op::Memory(MemoryOp::IndexedLoad { format, .. }) => any(2, Some(format.repr())),
        Op::MemoryNumber(_, format) => any(
            1,
            Some(match format {
                MemoryFormat::U32 | MemoryFormat::F32 | MemoryFormat::F64 => Repr::F64,
                _ => Repr::I32,
            }),
        ),
        Op::KeepAlive(_) => any(1, None),
        Op::NumericFloat(_) => exact(&[Repr::F64], Repr::F64),
        Op::SaturatingInteger(_) | Op::AsInteger(_) => exact(&[Repr::I64], Repr::I64),
        Op::IntegerCheck(..) => Representations {
            uses: vec![Use::Exact(Repr::Boxed), Use::Any, Use::Any],
            produces: Some(Repr::I8),
        },
        Op::IsInteger(_) => exact(&[Repr::Boxed], Repr::I8),
        Op::ToInt32(_) => exact(&[Repr::I32], Repr::I32),
        Op::Unary(UnaryOp::BitwiseNot, value) if numeric(*value) => exact(&[Repr::I32], Repr::I32),
        Op::AsNumber(value) => any(1, Some(graph.repr(*value))),
        Op::Unary(UnaryOp::Plus, value) | Op::ToNumeric(value) if numeric(*value) => {
            match graph.repr(*value) {
                repr @ (Repr::I32 | Repr::I64 | Repr::F64) => any(1, Some(repr)),
                _ => exact(&[Repr::F64], Repr::F64),
            }
        }
        Op::Unary(UnaryOp::Not, value) if numeric(*value) || graph.repr(*value) != Repr::Boxed => {
            if graph.repr(*value) == Repr::Boxed {
                exact(&[Repr::F64], Repr::I8)
            } else {
                any(1, Some(Repr::I8))
            }
        }
        Op::IsCode(..) | Op::IsShape(..) | Op::IsPropertyCached(..) | Op::IsTyped(..) => {
            exact(&[Repr::Boxed], Repr::I8)
        }
        Op::IsIndexed(_) => exact(&[Repr::Boxed], Repr::I8),
        Op::IsNumber(_) | Op::IsUninitialized(_) | Op::IsUndefined(_) | Op::IsNullish(_) => {
            any(1, Some(Repr::I8))
        }
        Op::CallNative(native, args) if args.iter().all(|&arg| numeric(arg)) => {
            let inline = matches!(
                (METHODS[native.index()].1, args.len()),
                (
                    MathFunction::Floor | MathFunction::Ceil | MathFunction::Sqrt,
                    1
                ) | (MathFunction::Min | MathFunction::Max, 1..)
            );
            if !inline {
                return None;
            }
            exact(&vec![Repr::F64; args.len()], Repr::F64)
        }
        Op::IsExactPosition(..) => exact(&[Repr::F64, Repr::F64], Repr::I8),
        Op::IsInBounds(..) => exact(&[Repr::I64, Repr::I64], Repr::I8),
        Op::ArrayPosition(_) => exact(&[Repr::I64], Repr::I64),
        Op::BooleanAnd(..) => exact(&[Repr::I8, Repr::I8], Repr::I8),
        Op::ArrayKind(_) => Representations {
            uses: vec![view_object(graph, inst)],
            produces: Some(Repr::I32),
        },
        Op::IsArray(..) => Representations {
            uses: vec![view_object(graph, inst)],
            produces: Some(Repr::I8),
        },
        Op::ArrayLoad(..) => Representations {
            uses: vec![view_object(graph, inst), Use::Exact(Repr::I64)],
            produces: Some(Repr::Boxed),
        },
        Op::IndexedLength(_) => Representations {
            uses: vec![view_object(graph, inst)],
            produces: Some(if target == Some(Repr::I64) {
                Repr::I64
            } else {
                Repr::F64
            }),
        },
        Op::TypedIndex(
            _,
            index,
            TypedRead {
                element: Some(kind),
                bounded,
            },
        ) => Representations {
            uses: vec![view_object(graph, inst), position(graph, *index)],
            produces: Some(if *bounded {
                element(*kind)
            } else {
                Repr::Boxed
            }),
        },
        Op::GetIndex(_, index) | Op::TypedIndex(_, index, _) | Op::IndexedGet(_, index) => {
            Representations {
                uses: vec![Use::Exact(Repr::Boxed), key(graph, *index)],
                produces: Some(Repr::Boxed),
            }
        }
        Op::SetIndex(_, index, _) => Representations {
            uses: vec![Use::Exact(Repr::Boxed), key(graph, *index), Use::Any],
            produces: None,
        },
        _ => return None,
    })
}

fn generic(graph: &Graph, inst: Inst) -> Representations {
    let data = &graph.insts[inst];
    let count = data.op.operands_iter().count();
    let uses = match &data.op {
        Op::EntryAssume(..)
        | Op::ReadHint(..)
        | Op::Assume(..)
        | Op::ContinuationOutput(..)
        | Op::OsrStart(..)
        | Op::LowLevel(_) => vec![Use::Any; count],
        Op::MapElement(..) => vec![Use::Exact(Repr::Boxed), Use::Exact(Repr::I64)],
        Op::GuardedCompare { numbers, .. } => [Use::Exact(Repr::Boxed); 2]
            .into_iter()
            .chain(numbers.iter().map(|_| Use::Exact(Repr::I8)))
            .collect(),
        _ => vec![Use::Exact(Repr::Boxed); count],
    };
    Representations {
        uses,
        produces: data.result().map(|_| Repr::Boxed),
    }
}

pub(in crate::jit) fn representations(graph: &Graph, facts: &Facts, inst: Inst) -> Representations {
    typed(graph, facts, inst).unwrap_or_else(|| generic(graph, inst))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::jit) enum ResultReference {
    #[default]
    Managed,
    Borrowed,
    Slot,
    Move,
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::jit) enum Lowering {
    #[default]
    Generic,
    Typed(Option<Repr>),
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::jit) struct Implementation {
    pub(in crate::jit) result: ResultReference,
    pub(in crate::jit) helper: Option<Helper>,
    pub(in crate::jit) lowering: Lowering,
}

fn reference_implementation(
    graph: &Graph,
    facts: &Facts,
    inst: Inst,
    slot: bool,
) -> Implementation {
    let op = &graph.insts[inst].op;
    let helpers = match op {
        Op::NewHoleyArray(_) => Some((Helper::ArrayMapNew, Helper::ArrayMapNewOwned)),
        Op::NewObject => Some((Helper::ObjectNew, Helper::ObjectNewOwned)),
        Op::NewArray(_) => Some((Helper::ArrayNew, Helper::ArrayNewOwned)),
        Op::PackTake(..) => Some((Helper::PackTake, Helper::PackTakeOwned)),
        Op::NewInstance(_) => Some((Helper::NewInstance, Helper::NewInstanceOwned)),
        Op::StringSplit(..) => Some((Helper::StringSplit, Helper::StringSplitOwned)),
        Op::StringSplitParts(..) => Some((Helper::StringSplitParts, Helper::StringSplitPartsOwned)),
        Op::Call {
            target: Some(_), ..
        } => None,
        _ => {
            return Implementation {
                result: match graph.contract(op, facts).result {
                    Ownership::Borrowed => ResultReference::Borrowed,
                    Ownership::Owned if slot => ResultReference::Move,
                    _ => ResultReference::Managed,
                },
                helper: None,
                ..Default::default()
            };
        }
    };
    debug_assert_eq!(graph.contract(op, facts).result, Ownership::Owned);
    Implementation {
        result: if slot {
            ResultReference::Slot
        } else {
            ResultReference::Managed
        },
        helper: helpers.map(|(temporary, owned)| {
            debug_assert!(temporary.returns_reference());
            debug_assert!(!owned.returns_reference());
            if slot { owned } else { temporary }
        }),
        ..Default::default()
    }
}

pub(in crate::jit) fn implementation(
    graph: &Graph,
    facts: &Facts,
    inst: Inst,
    slot: bool,
) -> Implementation {
    let mut implementation = reference_implementation(graph, facts, inst, slot);
    if let Some(contract) = typed(graph, facts, inst) {
        implementation.lowering = Lowering::Typed(contract.produces);
    }
    implementation
}
