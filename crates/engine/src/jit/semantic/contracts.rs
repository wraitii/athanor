use crate::jit::helpers::Helper;
use crate::jit::semantic::JsOp;
use smallvec::{SmallVec, smallvec};

use super::repr::{FloatWidth, IntegerWidth};
use super::{Facts, Graph, Inst, MemoryOp, Op, Ownership, Repr};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::jit) enum Use {
    Any,
    Exact(Repr),
}

pub(in crate::jit) struct Representations {
    pub(in crate::jit) uses: SmallVec<[Use; 4]>,
    pub(in crate::jit) produces: Option<Repr>,
}

fn specific(graph: &Graph, inst: Inst) -> Option<Representations> {
    let data = &graph.insts[inst];
    let target = data.results.first().map(|&value| graph.repr(value));
    let exact = |reprs: &[Repr], produces: Repr| Representations {
        uses: reprs.iter().map(|&repr| Use::Exact(repr)).collect(),
        produces: Some(produces),
    };
    let any = |count: usize, produces: Option<Repr>| Representations {
        uses: smallvec![Use::Any; count],
        produces,
    };
    Some(match &data.op {
        Op::Javascript(op) => return Some(super::js::contracts::representations(graph, inst, op)),
        Op::Aarch64(op) => any(op.operands().count(), target),
        Op::HelperCall { uses, produces, .. } => Representations {
            uses: uses.iter().copied().map(Use::Exact).collect(),
            produces: *produces,
        },
        Op::LowLevel(super::low_level::Op::Call {
            target,
            signature,
            args,
            ..
        }) => {
            let mut uses = SmallVec::with_capacity(args.len() + 1);
            if matches!(target, super::low_level::CallTarget::Indirect(_)) {
                uses.push(Use::Exact(Repr::Word));
            }
            uses.extend(args.iter().zip(&signature.params).map(|(&value, param)| {
                let repr = graph.repr(value);
                Use::Exact(if repr.fits(param.value_type) {
                    repr
                } else {
                    Repr::from_machine(param.value_type)
                })
            }));
            Representations {
                uses,
                produces: data.results.first().map(|&value| graph.repr(value)),
            }
        }
        Op::LowLevel(super::low_level::Op::Load { ty, .. }) => {
            any(1, Some(Repr::from_machine(*ty)))
        }
        Op::LowLevel(super::low_level::Op::Integer { .. }) => any(0, target),
        Op::LowLevel(super::low_level::Op::StackAddress { ty, .. }) => {
            any(0, Some(Repr::from_machine(*ty)))
        }
        Op::LowLevel(super::low_level::Op::Select { .. }) => any(3, target),
        Op::LowLevel(super::low_level::Op::Binary {
            op: super::low_level::Binary::FloatMin | super::low_level::Binary::FloatMax,
            ..
        }) => exact(&[Repr::FLOAT, Repr::FLOAT], Repr::FLOAT),
        Op::LowLevel(super::low_level::Op::Binary { op, .. }) if op.is_float() => {
            exact(&[Repr::F64, Repr::F64], Repr::F64)
        }
        Op::LowLevel(super::low_level::Op::Binary { op, .. }) if op.is_boolean() => {
            exact(&[Repr::Boolean, Repr::Boolean], Repr::Boolean)
        }
        Op::LowLevel(super::low_level::Op::Binary { .. }) => {
            exact(&[Repr::INTEGER, Repr::INTEGER], Repr::INTEGER)
        }
        Op::LowLevel(super::low_level::Op::IntCompare { .. }) => {
            exact(&[Repr::INTEGER, Repr::INTEGER], Repr::Boolean)
        }
        Op::LowLevel(super::low_level::Op::FloatCompare { .. }) => {
            exact(&[Repr::FLOAT, Repr::FLOAT], Repr::Boolean)
        }
        Op::LowLevel(super::low_level::Op::Unary {
            op:
                super::low_level::Unary::FloatNeg
                | super::low_level::Unary::FloatAbs
                | super::low_level::Unary::Trunc
                | super::low_level::Unary::Floor
                | super::low_level::Unary::Ceil,
            ..
        }) => exact(&[Repr::FLOAT], Repr::FLOAT),
        Op::LowLevel(super::low_level::Op::Unary {
            op: super::low_level::Unary::BitNot,
            ..
        }) => exact(&[Repr::INTEGER], Repr::INTEGER),
        Op::Convert(_, repr) | Op::Unbox(_, repr) => any(1, Some(*repr)),
        Op::Box(_) | Op::AsBoxed(_) => any(1, Some(Repr::Boxed)),
        Op::AsWord(_) => exact(&[Repr::Boxed], Repr::Word),
        Op::Memory(MemoryOp::Address { .. }) => any(2, Some(Repr::Word)),
        Op::Memory(MemoryOp::Load { format, .. }) => any(1, Some(format.repr())),
        Op::Memory(MemoryOp::IndexedLoad { format, .. }) => any(2, Some(format.repr())),
        Op::KeepAlive(_) => any(1, None),
        Op::LowLevel(super::low_level::Op::Store { .. }) => any(2, None),
        Op::LowLevel(super::low_level::Op::ExtractBits { value, .. }) => {
            any(1, Some(graph.repr(*value)))
        }
        Op::LowLevel(
            super::low_level::Op::InRange { .. } | super::low_level::Op::TestMask { .. },
        ) => any(1, Some(Repr::Boolean)),
        Op::LowLevel(super::low_level::Op::Reference { .. }) => Representations {
            uses: smallvec![Use::Exact(Repr::Boxed)],
            produces: None,
        },
        Op::LowLevel(super::low_level::Op::Unary { op, ty, value }) => Representations {
            uses: smallvec![match op {
                super::low_level::Unary::FloatDemote => Use::Exact(Repr::F64),
                super::low_level::Unary::Reduce => Use::Exact(Repr::I64),
                _ => Use::Any,
            }],
            produces: Some(match op {
                super::low_level::Unary::Bitcast => Repr::from_machine(*ty),
                super::low_level::Unary::Sqrt => graph.repr(*value),
                _ => Repr::numeric(*ty),
            }),
        },
        Op::Regex(super::regex::Op::IsPattern(..) | super::regex::Op::Test(..)) => {
            exact(&[Repr::Boxed], Repr::Boolean)
        }
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
        | Op::LowLevel(_) => smallvec![Use::Any; count],
        _ => smallvec![Use::Exact(Repr::Boxed); count],
    };
    let produces = match &data.op {
        Op::LowLevel(_) => data.results.first().map(|&value| graph.repr(value)),
        _ => data.result().map(|_| Repr::Boxed),
    };
    Representations { uses, produces }
}

pub(in crate::jit) fn repr_contract(graph: &Graph, inst: Inst) -> Representations {
    let contract = specific(graph, inst).unwrap_or_else(|| generic(graph, inst));
    instantiate(graph, inst, contract)
}

fn open(repr: Repr) -> bool {
    matches!(repr, Repr::INTEGER | Repr::FLOAT)
}

fn within(open: Repr, repr: Repr) -> bool {
    match (open, repr) {
        (Repr::Integer(_), Repr::Integer(width)) => width != IntegerWidth::Any,
        (Repr::Integer(_), Repr::Word) => true,
        (Repr::Float(_), Repr::Float(width)) => width != FloatWidth::Any,
        _ => false,
    }
}

/// Resolves the open widths of a contract against the representations currently assigned,
/// so that representation passes keep seeing concrete widths. Each family binds one width
/// per instruction: the result's when it is open, the widest operand's otherwise.
fn instantiate(graph: &Graph, inst: Inst, mut contract: Representations) -> Representations {
    let opens = |use_: &Use| matches!(use_, Use::Exact(repr) if open(*repr));
    if !contract.produces.is_some_and(open) && !contract.uses.iter().any(opens) {
        return contract;
    }
    let data = &graph.insts[inst];
    let bind = |family: Repr, default: Repr| {
        let target = data
            .result()
            .map(|value| graph.repr(value))
            .filter(|&repr| contract.produces == Some(family) && within(family, repr));
        target
            .or_else(|| {
                data.op
                    .operands_iter()
                    .zip(&contract.uses)
                    .filter(|&(_, use_)| *use_ == Use::Exact(family))
                    .map(|(value, _)| graph.repr(value))
                    .filter(|&repr| within(family, repr))
                    .max_by_key(|repr| repr.machine().bits())
            })
            .map(|repr| if repr == Repr::Word { Repr::I64 } else { repr })
            .unwrap_or(default)
    };
    let integer = bind(Repr::INTEGER, Repr::I64);
    let float = bind(Repr::FLOAT, Repr::F64);
    let resolve = |repr: Repr| match repr {
        Repr::INTEGER => integer,
        Repr::FLOAT => float,
        repr => repr,
    };
    for use_ in &mut contract.uses {
        if let Use::Exact(repr) = use_ {
            *repr = resolve(*repr);
        }
    }
    contract.produces = contract.produces.map(resolve);
    contract
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
        Op::Javascript(JsOp::Call {
            target: Some(_),
            ..
        }) | Op::HelperCall {
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
    let allocation =
        fresh_reference(op) || matches!(op, Op::Javascript(JsOp::StringSplitParts(..)));
    match graph.contract(op, facts).result {
        Ownership::Owned if allocation && slot => ResultReference::Slot,
        Ownership::Owned if allocation => ResultReference::Managed,
        Ownership::Borrowed => ResultReference::Borrowed,
        Ownership::Owned if slot => ResultReference::Move,
        _ => ResultReference::Managed,
    }
}
