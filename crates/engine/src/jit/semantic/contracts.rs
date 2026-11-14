use crate::jit::helpers::Helper;

use super::{
    Constant, Facts, Graph, Inst, MemoryFormat, MemoryOp, Op, Ownership, Repr, TypedRead, Value,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Use {
    Any,
    Exact(Repr),
}

pub(in crate::jit) struct Representations {
    pub(in crate::jit) uses: Vec<Use>,
    pub(in crate::jit) produces: Option<Repr>,
}

fn integer(repr: Repr) -> bool {
    matches!(repr, Repr::I32 | Repr::I64)
}

fn view_object(graph: &Graph, inst: Inst) -> Use {
    if graph.views.contains_key(&inst) {
        Use::Any
    } else {
        Use::Exact(Repr::Boxed)
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

fn specific(graph: &Graph, inst: Inst) -> Option<Representations> {
    let data = &graph.insts[inst];
    let target = data.result().map(|value| graph.repr(value));
    let exact = |reprs: &[Repr], produces: Repr| Representations {
        uses: reprs.iter().map(|&repr| Use::Exact(repr)).collect(),
        produces: Some(produces),
    };
    let any = |count: usize, produces: Option<Repr>| Representations {
        uses: vec![Use::Any; count],
        produces,
    };
    Some(match &data.op {
        Op::HelperCall { uses, produces, .. } => Representations {
            uses: uses.iter().copied().map(Use::Exact).collect(),
            produces: *produces,
        },
        Op::PropertyLoad(..) => exact(&[Repr::Boxed, Repr::Word], Repr::Boxed),
        Op::LowLevel(super::low_level::Op::Load { ty, .. }) => {
            any(1, Some(Repr::from_machine(*ty)))
        }
        Op::LowLevel(super::low_level::Op::Integer { .. }) => any(0, target),
        Op::LowLevel(super::low_level::Op::StackAddress { ty, .. }) => {
            any(0, Some(Repr::from_machine(*ty)))
        }
        Op::LowLevel(super::low_level::Op::Select { .. }) => any(3, target),
        Op::LowLevel(super::low_level::Op::Binary { op, .. }) if op.is_float() => {
            exact(&[Repr::F64, Repr::F64], Repr::F64)
        }
        Op::LowLevel(super::low_level::Op::Binary { op, .. }) if !op.is_float() => {
            let repr = target.expect("integer arithmetic result representation");
            exact(&[repr, repr], repr)
        }
        Op::LowLevel(super::low_level::Op::IntCompare { left, right, .. }) => {
            let (left, right) = (graph.repr(*left), graph.repr(*right));
            if left.machine() == right.machine() {
                exact(&[left, right], Repr::Boolean)
            } else {
                exact(&[Repr::I64, Repr::I64], Repr::Boolean)
            }
        }
        Op::LowLevel(super::low_level::Op::FloatCompare { .. }) => {
            exact(&[Repr::F64, Repr::F64], Repr::Boolean)
        }
        Op::LowLevel(super::low_level::Op::Unary {
            op: super::low_level::Unary::FloatNeg | super::low_level::Unary::BitNot,
            value,
            ..
        }) => {
            let repr = graph.repr(*value);
            exact(&[repr], repr)
        }
        Op::Convert(_, repr) | Op::Unbox(_, repr) => any(1, Some(*repr)),
        Op::Box(_) | Op::AsBoxed(_) => any(1, Some(Repr::Boxed)),
        Op::AsWord(_) => exact(&[Repr::Boxed], Repr::Word),
        Op::Constant(Constant::Number(_)) => match target {
            Some(repr @ (Repr::F64 | Repr::I64 | Repr::I32)) => any(0, Some(repr)),
            _ => return None,
        },
        Op::Constant(Constant::Boolean(_)) if target == Some(Repr::Boolean) => {
            any(0, Some(Repr::Boolean))
        }
        Op::TypedData(_) => any(1, Some(Repr::Word)),
        Op::Memory(MemoryOp::Address { .. }) => any(2, Some(Repr::Word)),
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
        Op::PropertyCache(..) => any(0, Some(Repr::Word)),
        Op::CanOverwrite(..) => exact(&[Repr::Boxed, Repr::I64], Repr::Boolean),
        Op::ArrayStore(..) | Op::PropertyStore(..) | Op::PropertyAdd(..) => Representations {
            uses: vec![
                Use::Exact(Repr::Boxed),
                Use::Exact(Repr::I64),
                Use::Exact(Repr::Boxed),
            ],
            produces: None,
        },
        Op::LowLevel(super::low_level::Op::Store { .. }) => any(2, None),
        Op::LowLevel(super::low_level::Op::ExtractBits { value, .. }) => {
            any(1, Some(graph.repr(*value)))
        }
        Op::LowLevel(
            super::low_level::Op::InRange { .. } | super::low_level::Op::TestMask { .. },
        ) => any(1, Some(Repr::Boolean)),
        Op::LowLevel(super::low_level::Op::Reference { .. }) => Representations {
            uses: vec![Use::Exact(Repr::Boxed)],
            produces: None,
        },
        Op::LowLevel(super::low_level::Op::Unary { op, ty, value }) => Representations {
            uses: vec![match op {
                super::low_level::Unary::FloatDemote => Use::Exact(Repr::F64),
                super::low_level::Unary::Reduce => Use::Exact(Repr::I64),
                _ => Use::Any,
            }],
            produces: Some(match op {
                super::low_level::Unary::Bitcast => Repr::from_machine(*ty),
                super::low_level::Unary::Trunc
                | super::low_level::Unary::Floor
                | super::low_level::Unary::Ceil
                | super::low_level::Unary::Sqrt
                | super::low_level::Unary::FloatAbs => graph.repr(*value),
                _ => Repr::numeric(*ty),
            }),
        },
        Op::NumericFloat(_) => exact(&[Repr::F64], Repr::F64),
        Op::SaturatingInteger(_) | Op::AsInteger(_) => exact(&[Repr::I64], Repr::I64),
        Op::IntegerCheck(..) => Representations {
            uses: vec![Use::Exact(Repr::Boxed), Use::Any, Use::Any],
            produces: Some(Repr::Boolean),
        },
        Op::IsInteger(_) => exact(&[Repr::Boxed], Repr::Boolean),
        Op::ToInt32(_) => exact(&[Repr::I32], Repr::I32),
        Op::AsNumber(value) => any(1, Some(graph.repr(*value))),
        Op::IsCode(..) | Op::IsShape(..) | Op::IsTyped(..) => {
            exact(&[Repr::Boxed], Repr::Boolean)
        }
        Op::IsPropertyCached(..) => exact(&[Repr::Boxed, Repr::Word], Repr::Boolean),
        Op::CachedProperty(..) | Op::FillPropertyCache(..) => {
            exact(&[Repr::Boxed, Repr::Word], Repr::Boxed)
        }
        Op::PrimePropertyCache(..) => Representations {
            uses: vec![Use::Exact(Repr::Boxed), Use::Exact(Repr::Word)],
            produces: None,
        },
        Op::IsIndexed(_) | Op::IsString(_) | Op::Truthy(_) => {
            exact(&[Repr::Boxed], Repr::Boolean)
        }
        Op::IsNumber(_) | Op::IsUninitialized(_) | Op::IsUndefined(_) | Op::IsNullish(_) => {
            any(1, Some(Repr::Boolean))
        }
        Op::ArrayPosition(_) => exact(&[Repr::I64], Repr::I64),
        Op::ArrayKind(_) => Representations {
            uses: vec![view_object(graph, inst)],
            produces: Some(Repr::I32),
        },
        Op::IsArray(..) => Representations {
            uses: vec![view_object(graph, inst)],
            produces: Some(Repr::Boolean),
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
        | Op::LowLevel(_) => vec![Use::Any; count],
        Op::MapElement(..) => vec![Use::Exact(Repr::Boxed), Use::Exact(Repr::I64)],
        Op::GuardedCompare { numbers, .. } => [Use::Exact(Repr::Boxed); 2]
            .into_iter()
            .chain(numbers.iter().map(|_| Use::Exact(Repr::Boolean)))
            .collect(),
        _ => vec![Use::Exact(Repr::Boxed); count],
    };
    let produces = match &data.op {
        Op::LowLevel(_) => data.result().map(|value| graph.repr(value)),
        _ => data.result().map(|_| Repr::Boxed),
    };
    Representations { uses, produces }
}

pub(in crate::jit) fn repr_contract(graph: &Graph, inst: Inst) -> Representations {
    specific(graph, inst).unwrap_or_else(|| generic(graph, inst))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::jit) enum ResultReference {
    #[default]
    Managed,
    Borrowed,
    Slot,
    Move,
}

pub(super) fn fresh_reference(op: &Op) -> bool {
    matches!(
        op,
        Op::Call {
            target: Some(_),
            ..
        } | Op::HelperCall {
            helper: Helper::ArrayMapNew
                | Helper::ObjectNew
                | Helper::ArrayNew
                | Helper::PackTake
                | Helper::NewInstance
                | Helper::StringSplit,
            ..
        }
    )
}

pub(in crate::jit) fn result_reference(
    graph: &Graph,
    facts: &Facts,
    inst: Inst,
    slot: bool,
) -> ResultReference {
    let op = &graph.insts[inst].op;
    let allocation = fresh_reference(op) || matches!(op, Op::StringSplitParts(..));
    match graph.contract(op, facts).result {
        Ownership::Owned if allocation && slot => ResultReference::Slot,
        Ownership::Owned if allocation => ResultReference::Managed,
        Ownership::Borrowed => ResultReference::Borrowed,
        Ownership::Owned if slot => ResultReference::Move,
        _ => ResultReference::Managed,
    }
}
