use super::super::contracts::{Representations, Use};
use super::super::repr::element;
use super::super::{Contract, Facts, Graph, Inst, MemoryFormat, Ownership, Repr, Value, ValueType};
use super::{Constant, Op, PropertyWrite, TypedRead, UnaryOp};
use crate::jit::effects::{
    ALLOCATION, CODE_READ, CODE_WRITE, ERROR, Effects, GLOBAL_READ, GLOBAL_WRITE, HEAP_READ,
    HEAP_WRITE, INTERNAL,
};
use crate::jit::math::METHODS;
use oxc_syntax::operator::BinaryOperator;
use smallvec::smallvec;

pub(in crate::jit::semantic) fn contract(graph: &Graph, op: &Op, facts: &Facts) -> Contract {
    use Ownership::*;
    let closed = |flags, result| Contract {
        effects: Effects::of(flags),
        result,
    };
    const COERCING: u16 = HEAP_READ | ALLOCATION | INTERNAL | ERROR;
    let operands = || {
        op.operands_iter()
            .fold(ValueType::default(), |types, value| {
                types.union(facts.of(value))
            })
    };
    let coercing = |result| {
        if operands().within(ValueType::NUMBER) {
            closed(0, Unowned)
        } else {
            closed(COERCING, result)
        }
    };
    match op {
        Op::NewInstance(_) => closed(ALLOCATION | INTERNAL | ERROR, Owned),
        Op::ConstructorResult(..) => closed(0, Borrowed),
        Op::Rest(_) | Op::Arguments => closed(ALLOCATION | INTERNAL | ERROR, Owned),
        Op::Throw(_) => closed(INTERNAL | ERROR, None),
        Op::Catch => closed(INTERNAL, Owned),
        Op::This | Op::Callee => closed(0, Borrowed),
        Op::ClosureThis(_) => closed(HEAP_READ, Borrowed),
        Op::NewHoleyArray(_) => closed(ALLOCATION | INTERNAL | ERROR, Owned),
        Op::MapElement(..) => closed(HEAP_READ | INTERNAL, Owned),
        Op::IsArrayIteration(..) | Op::IsStringSplit(_) => closed(HEAP_READ, Unowned),
        Op::StringSplit(..) | Op::StringSplitParts(..) => {
            closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, Owned)
        }
        Op::Constant(_) => closed(0, Unowned),
        Op::GlobalRead(_) | Op::GlobalReadOptional(_) => {
            closed(GLOBAL_READ | INTERNAL | ERROR, Borrowed)
        }
        Op::RegExp(..) => closed(ALLOCATION | INTERNAL | ERROR, Owned),
        Op::StringConcat(_) => closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, Owned),
        Op::PropertyKey(_) => closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, Owned),
        Op::GlobalWrite(..) | Op::GlobalInitialize(..) => {
            closed(GLOBAL_WRITE | INTERNAL | ERROR, None)
        }
        Op::BindingLoad(_) => closed(INTERNAL, Borrowed),
        Op::BindingRestore(..) => closed(INTERNAL, None),
        Op::Read(binding) => closed(
            if graph.bindings[*binding].captured {
                HEAP_READ | INTERNAL | ERROR
            } else {
                ERROR
            },
            if graph.bindings[*binding].captured || graph.lazy_captures.contains(binding) {
                Owned
            } else {
                Borrowed
            },
        ),
        Op::Initialize(binding, _) | Op::Write(binding, _) if graph.bindings[*binding].captured => {
            closed(HEAP_WRITE | INTERNAL | ERROR, None)
        }
        Op::Initialize(..) => closed(INTERNAL, None),
        Op::Write(..) => closed(INTERNAL | ERROR, None),
        Op::CellGet(_) | Op::CaptureGet(..) => closed(HEAP_READ | INTERNAL | ERROR, Owned),
        Op::CellSet(..) => closed(HEAP_WRITE | INTERNAL | ERROR, None),
        Op::CheckInitialized(_) => {
            if operands().intersects(ValueType::UNINITIALIZED) {
                closed(ERROR, None)
            } else {
                closed(0, None)
            }
        }
        Op::Closure(..)
        | Op::ClosureCells(..)
        | Op::CellNew(_)
        | Op::NewPack
        | Op::NewObject
        | Op::NewArray(_)
        | Op::IteratorNew(_)
        | Op::EnumerateNew(_) => closed(ALLOCATION | INTERNAL | ERROR, Owned),
        Op::PackRelease(_) => closed(HEAP_WRITE | INTERNAL, None),
        Op::Spread(_) => closed(HEAP_READ | ALLOCATION | INTERNAL | ERROR, Owned),
        Op::IteratorNext(_) | Op::EnumerateNext(_) => closed(
            HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL | ERROR,
            Owned,
        ),
        Op::IsUninitialized(_) => closed(0, Unowned),
        // Defines only target the literal being built, which nothing has read yet, so the
        // HEAP write is left out to keep borrows and loads alive across object construction.
        Op::Define(..) => closed(HEAP_READ | CODE_WRITE | ALLOCATION | INTERNAL | ERROR, None),
        Op::DefineHidden(..) | Op::MarkClass(_) | Op::DisableConstructor(_) => closed(
            HEAP_WRITE | CODE_WRITE | ALLOCATION | INTERNAL | ERROR,
            None,
        ),
        Op::PackAppend(..) | Op::ArrayAppend(..) => {
            closed(HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL | ERROR, None)
        }
        Op::Get(..) | Op::GetIndex(..) | Op::FillPropertyCache(..) => closed(
            HEAP_READ | GLOBAL_READ | ALLOCATION | INTERNAL | ERROR,
            Borrowed,
        ),
        Op::PropertyLoad(..) | Op::RelatedPropertyLoad(..) => closed(HEAP_READ, Borrowed),
        Op::PropertyCache(..) | Op::GlobalCache(..) => closed(0, Unowned),
        Op::IsPropertyCached(..) | Op::IsShape(..) | Op::IsReadLayout(..) => {
            closed(CODE_READ, Unowned)
        }
        Op::ShapedGet(..) | Op::LayoutGet(..) => closed(HEAP_READ | INTERNAL | ERROR, Borrowed),
        Op::ShapedSet(..) => closed(HEAP_READ | HEAP_WRITE | ALLOCATION | INTERNAL | ERROR, None),
        Op::PrimePropertyCache(..) => closed(HEAP_READ | CODE_READ | INTERNAL, None),
        Op::CachedProperty(..) => closed(HEAP_READ | INTERNAL | ERROR, Borrowed),
        Op::FillPropertyStore(.., PropertyWrite::Define) => {
            closed(HEAP_READ | CODE_WRITE | ALLOCATION | INTERNAL | ERROR, None)
        }
        Op::PropertyStore(.., PropertyWrite::Define)
        | Op::PropertyAdd(.., PropertyWrite::Define) => {
            closed(HEAP_READ | CODE_WRITE | INTERNAL, None)
        }
        Op::Delete(..) => closed(
            HEAP_READ | HEAP_WRITE | CODE_WRITE | ALLOCATION | INTERNAL | ERROR,
            Unowned,
        ),
        Op::Set(..) | Op::SetIndex(..) | Op::TypedSet(..) | Op::FillPropertyStore(..) => closed(
            HEAP_READ | HEAP_WRITE | CODE_WRITE | ALLOCATION | INTERNAL | ERROR,
            None,
        ),
        Op::Unary(UnaryOp::Not, _) => closed(HEAP_READ, Unowned),
        Op::Unary(UnaryOp::Typeof, _) => closed(HEAP_READ | ALLOCATION | ERROR, Owned),
        Op::Unary(UnaryOp::Increment | UnaryOp::Decrement, _) => {
            if operands().within(ValueType::NUMBER) {
                closed(0, Unowned)
            } else {
                closed(ALLOCATION | INTERNAL | ERROR, Owned)
            }
        }
        Op::ToInt32(_) => closed(0, Unowned),
        Op::ToNumeric(_) | Op::Unary(..) => coercing(Owned),
        Op::GuardedCompare {
            operator,
            left,
            right,
            ..
        } => contract(graph, &Op::Binary(*operator, *left, *right), facts),
        Op::Binary(operator, ..) => match operator {
            BinaryOperator::StrictEquality | BinaryOperator::StrictInequality => closed(
                if operands().within(ValueType::NUMBER) {
                    0
                } else {
                    HEAP_READ
                },
                Unowned,
            ),
            BinaryOperator::Equality
            | BinaryOperator::Inequality
            | BinaryOperator::LessThan
            | BinaryOperator::LessEqualThan
            | BinaryOperator::GreaterThan
            | BinaryOperator::GreaterEqualThan => coercing(Unowned),
            BinaryOperator::In | BinaryOperator::Instanceof => Contract {
                effects: Effects::UNKNOWN,
                result: Unowned,
            },
            _ => coercing(Owned),
        },
        Op::Call { .. }
        | Op::Construct { .. }
        | Op::CallPack(..)
        | Op::CallForward(..)
        | Op::ConstructPack(..)
        | Op::ConstructorCall { .. }
        | Op::ConstructorCallPack(..)
        | Op::ConstructResolved { .. }
        | Op::ConstructPackResolved(..) => Contract {
            effects: Effects::UNKNOWN,
            result: Owned,
        },
        Op::CallNative(native, _)
            if METHODS[native.index()].1 == crate::heap::MathFunction::Random =>
        {
            closed(INTERNAL, Unowned)
        }
        Op::CallNative(..) => coercing(Unowned),
        Op::IsUndefined(_) | Op::IsNullish(_) | Op::IsNative(..) => closed(0, Unowned),
        Op::ConstructorPrototype(_) => closed(HEAP_READ, Borrowed),
        Op::IsCode(..) => closed(HEAP_READ, Unowned),
        Op::ProbeGlobal(_) => closed(GLOBAL_READ, Borrowed),
        Op::ProbeProperty(..) | Op::ProbeIndex(..) => closed(HEAP_READ, Borrowed),
        Op::ProbeShapeProperty(..) => closed(HEAP_READ | CODE_READ, Borrowed),
        Op::GuardValid(_) => closed(GLOBAL_READ, Unowned),
        Op::GuardCommit(_) => closed(INTERNAL, None),
        Op::View(_) | Op::TypedData(_) => closed(HEAP_READ, Unowned),
        Op::MemoryNumber(..) => closed(0, Unowned),
        Op::IsTyped(..) | Op::IsIndexed(_) | Op::IsOrdinaryArray(_) => closed(0, Unowned),
        Op::IsArrayIndex(..) | Op::IsArray(..) => closed(HEAP_READ, Unowned),
        Op::ArrayPosition(_) => closed(INTERNAL, Unowned),
        Op::IsExactPosition(..) | Op::IsInBounds(..) => closed(0, Unowned),
        Op::ArrayKind(_) | Op::IndexedLength(_) => closed(HEAP_READ, Unowned),
        Op::ArrayGet(..) => closed(HEAP_READ | INTERNAL | ERROR, Borrowed),
        Op::ArrayLoad(..) => closed(HEAP_READ, Borrowed),
        Op::ArrayStore(..) | Op::PropertyStore(..) => {
            closed(HEAP_READ | HEAP_WRITE | INTERNAL, None)
        }
        Op::PropertyAdd(..) => closed(HEAP_READ | HEAP_WRITE | CODE_WRITE | INTERNAL, None),
        Op::CanOverwrite(..) => closed(HEAP_READ, Unowned),
        Op::PackTake(..) => closed(HEAP_READ | HEAP_WRITE | INTERNAL, Owned),
        Op::TypedIndex(..) => closed(HEAP_READ, Unowned),
        Op::IsNumber(_)
        | Op::IsInteger(_)
        | Op::IsIntegerInRange(..)
        | Op::AsInteger(_)
        | Op::NumericFloat(_)
        | Op::SaturatingInteger(_)
        | Op::IntegerCheck(..)
        | Op::IsString(_) => closed(0, Unowned),
        Op::IsTypeof(..) | Op::Truthy(_) => closed(HEAP_READ, Unowned),
        Op::AsNumber(_) => closed(INTERNAL, Unowned),
        Op::AsString(_) => closed(0, Borrowed),
        Op::AsObject(_) => closed(0, Unowned),
        Op::EnterCaptures(..) => closed(HEAP_READ | INTERNAL | ERROR, None),
        Op::EnterScope(scope) | Op::NextIteration(scope) | Op::FreshIteration(scope) => {
            if graph.materialized(*scope) {
                closed(ALLOCATION | INTERNAL | ERROR, None)
            } else {
                closed(0, None)
            }
        }
        Op::LeaveScope(scope) => {
            if graph.materialized(*scope) {
                closed(INTERNAL, None)
            } else {
                closed(0, None)
            }
        }
    }
}

fn integer(repr: Repr) -> bool {
    matches!(repr, Repr::Integer(_))
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

fn specific(graph: &Graph, inst: Inst, op: &Op) -> Option<Representations> {
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
    Some(match op {
        Op::PropertyLoad(..) | Op::RelatedPropertyLoad(..) => {
            exact(&[Repr::Boxed, Repr::Word], Repr::Boxed)
        }
        Op::Constant(Constant::Number(_)) => match target {
            Some(repr @ (Repr::Float(_) | Repr::Integer(_))) => any(0, Some(repr)),
            _ => return None,
        },
        Op::Constant(Constant::Boolean(_)) if target == Some(Repr::Boolean) => {
            any(0, Some(Repr::Boolean))
        }
        Op::TypedData(_) => any(1, Some(Repr::Word)),
        Op::MemoryNumber(_, format) => any(
            1,
            Some(match format {
                MemoryFormat::U32 | MemoryFormat::F32 | MemoryFormat::F64 => Repr::F64,
                _ => Repr::I32,
            }),
        ),
        Op::PropertyCache(..) | Op::GlobalCache(..) => any(0, Some(Repr::Word)),
        Op::CanOverwrite(..) => exact(&[Repr::Boxed, Repr::I64], Repr::Boolean),
        Op::ArrayStore(..) | Op::PropertyStore(..) | Op::PropertyAdd(..) => Representations {
            uses: smallvec![
                Use::Exact(Repr::Boxed),
                Use::Exact(Repr::I64),
                Use::Exact(Repr::Boxed),
            ],
            produces: None,
        },
        Op::NumericFloat(_) => exact(&[Repr::F64], Repr::F64),
        Op::SaturatingInteger(_) | Op::AsInteger(_) => exact(&[Repr::I64], Repr::I64),
        Op::IntegerCheck(..) => Representations {
            uses: smallvec![Use::Exact(Repr::Boxed), Use::Any, Use::Any],
            produces: Some(Repr::Boolean),
        },
        Op::IsInteger(_) | Op::IsIntegerInRange(..) => exact(&[Repr::Boxed], Repr::Boolean),
        Op::ToInt32(_) => exact(&[Repr::I32], Repr::I32),
        Op::AsNumber(_) => any(1, target),
        Op::IsCode(..) | Op::IsShape(..) | Op::IsReadLayout(..) | Op::IsTyped(..) => {
            exact(&[Repr::Boxed], Repr::Boolean)
        }
        Op::IsPropertyCached(..) => exact(&[Repr::Boxed, Repr::Word], Repr::Boolean),
        Op::CachedProperty(..) | Op::FillPropertyCache(..) => {
            exact(&[Repr::Boxed, Repr::Word], Repr::Boxed)
        }
        Op::PrimePropertyCache(..) => Representations {
            uses: smallvec![Use::Exact(Repr::Boxed), Use::Exact(Repr::Word)],
            produces: None,
        },
        Op::IsIndexed(_) | Op::IsOrdinaryArray(_) | Op::IsString(_) | Op::Truthy(_) => {
            exact(&[Repr::Boxed], Repr::Boolean)
        }
        Op::IsNumber(_) | Op::IsUninitialized(_) | Op::IsUndefined(_) | Op::IsNullish(_) => {
            any(1, Some(Repr::Boolean))
        }
        Op::ArrayPosition(_) => exact(&[Repr::I64], Repr::I64),
        Op::ArrayKind(_) => Representations {
            uses: smallvec![view_object(graph, inst)],
            produces: Some(Repr::I32),
        },
        Op::IsArray(..) => Representations {
            uses: smallvec![view_object(graph, inst)],
            produces: Some(Repr::Boolean),
        },
        Op::ArrayLoad(..) => Representations {
            uses: smallvec![view_object(graph, inst), Use::Exact(Repr::I64)],
            produces: Some(Repr::Boxed),
        },
        Op::IndexedLength(_) => Representations {
            uses: smallvec![view_object(graph, inst)],
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
            uses: smallvec![view_object(graph, inst), position(graph, *index)],
            produces: Some(if *bounded {
                element(*kind)
            } else {
                Repr::Boxed
            }),
        },
        _ => return None,
    })
}

pub(in crate::jit::semantic) fn representations(
    graph: &Graph,
    inst: Inst,
    op: &Op,
) -> Representations {
    specific(graph, inst, op).unwrap_or_else(|| {
        let count = op.operands_iter().count();
        let uses = match op {
            Op::MapElement(..) => smallvec![Use::Exact(Repr::Boxed), Use::Exact(Repr::I64)],
            Op::GuardedCompare { numbers, .. } => [Use::Exact(Repr::Boxed); 2]
                .into_iter()
                .chain(numbers.iter().map(|_| Use::Exact(Repr::Boolean)))
                .collect(),
            _ => smallvec![Use::Exact(Repr::Boxed); count],
        };
        Representations {
            uses,
            produces: graph.insts[inst].result().map(|_| Repr::Boxed),
        }
    })
}
