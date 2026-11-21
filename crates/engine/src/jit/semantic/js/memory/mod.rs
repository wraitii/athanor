use super::super::{Facts, Graph, Op, Value, ValueType};
use crate::jit::effects::Locations;
use crate::jit::helpers::Helper;
use crate::jit::semantic::JsOp;

mod fresh;

pub(in crate::jit::semantic) use fresh::Fresh;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit) enum Field {
    Named(String),
    // Cells can alias other captures, but cannot alias JavaScript properties.
    Cells,
    Packs,
    Length,
    Elements,
    Iteration,
    Any,
}

impl Field {
    pub(in crate::jit::semantic) fn named(name: &str) -> Self {
        match name {
            "length" => Self::Length,
            _ if name.starts_with(|c: char| c.is_ascii_digit() || c == '-')
                || matches!(name, "NaN" | "Infinity") =>
            {
                Self::Any
            }
            _ => Self::Named(name.to_owned()),
        }
    }

    pub(in crate::jit::semantic) fn cached(graph: &Graph, cache: Value) -> Self {
        match graph.defining_op(cache) {
            Some(Op::Javascript(JsOp::PropertyCache(_, name))) => Self::named(name),
            _ => Self::Any,
        }
    }

    fn of(graph: &Graph, op: &Op, facts: &Facts) -> Self {
        match op {
            Op::Javascript(
                JsOp::PropertyLoad(_, _, field)
                | JsOp::RelatedPropertyLoad(_, _, field)
                | JsOp::PropertyStore(_, _, _, field, _),
            ) => field.clone(),
            Op::Javascript(JsOp::CachedProperty(_, cache, _) | JsOp::PropertyAdd(_, cache, ..)) => {
                Self::cached(graph, *cache)
            }
            Op::HelperCall {
                helper:
                    Helper::PropertyGetCached
                    | Helper::PropertyGetMegamorphic
                    | Helper::SemanticPropertyProbeCached
                    | Helper::PropertySetCached,
                args,
                ..
            } => Self::cached(graph, args[1]),
            Op::Javascript(
                JsOp::Read(_)
                | JsOp::Initialize(..)
                | JsOp::Write(..)
                | JsOp::CellGet(_)
                | JsOp::CellSet(..),
            ) => Self::Cells,
            Op::Javascript(
                JsOp::Get(_, name)
                | JsOp::FillPropertyStore(_, _, name, _, _)
                | JsOp::Set(_, name, _)
                | JsOp::Define(_, name, _)
                | JsOp::ShapedGet(_, name, _)
                | JsOp::LayoutGet(_, name, _)
                | JsOp::ProbeProperty(_, name)
                | JsOp::ShapedSet(_, name, _, _)
                | JsOp::GlobalRead(name)
                | JsOp::ProbeGlobal(name)
                | JsOp::GlobalWrite(name, _)
                | JsOp::GlobalInitialize(name, _),
            ) => Self::named(name),
            Op::Javascript(
                JsOp::GetIndex(_, key)
                | JsOp::TypedIndex(_, key, _)
                | JsOp::ArrayGet(_, key)
                | JsOp::ArrayLoad(_, key, _)
                | JsOp::ArrayStore(_, key, _)
                | JsOp::SetIndex(_, key, _)
                | JsOp::TypedSet(_, key, ..),
            ) if facts.of(*key).within(ValueType::NUMBER) => Self::Elements,
            Op::Javascript(JsOp::IndexedLength(_)) => Self::Length,
            Op::Javascript(JsOp::ArrayAppend(..) | JsOp::MapElement(..)) => Self::Elements,
            Op::Javascript(JsOp::PackAppend(..) | JsOp::PackRelease(_)) => Self::Packs,
            Op::Javascript(JsOp::IteratorNext(_) | JsOp::EnumerateNext(_)) => Self::Iteration,
            _ => Self::Any,
        }
    }

    pub(in crate::jit::semantic) fn conflicts(&self, read: &Self) -> bool {
        match (self, read) {
            (Self::Iteration, _) | (_, Self::Iteration) => false,
            (Self::Any, _) | (_, Self::Any) => true,
            (Self::Cells, Self::Cells) | (Self::Packs, Self::Packs) => true,
            (Self::Named(write), Self::Named(read)) => write == read,
            (Self::Elements, Self::Elements | Self::Length) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit::semantic) enum Base {
    Known(Value),
    // External state cannot alias an allocation that has not escaped.
    External,
    // Indirect storage can alias even an otherwise fresh allocation.
    Unknown,
}

impl Base {
    pub(in crate::jit::semantic) fn map(self, f: impl FnOnce(Value) -> Value) -> Self {
        match self {
            Self::Known(value) => Self::Known(f(value)),
            _ => self,
        }
    }

    pub(in crate::jit::semantic) fn value(self) -> Option<Value> {
        match self {
            Self::Known(value) => Some(value),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit::semantic) struct Access {
    pub(in crate::jit::semantic) base: Base,
    pub(in crate::jit::semantic) field: Field,
}

impl Access {
    pub(in crate::jit::semantic) const UNKNOWN: Self = Self {
        base: Base::Unknown,
        field: Field::Any,
    };

    pub(in crate::jit::semantic) fn is_unknown(&self) -> bool {
        self.field == Field::Any
    }

    pub(in crate::jit::semantic) fn read(
        graph: &Graph,
        op: &Op,
        facts: &Facts,
    ) -> Option<(Locations, Self)> {
        let (location, base) = match op {
            Op::Javascript(JsOp::LayoutGet(..) | JsOp::RelatedPropertyLoad(..)) => {
                (Locations::HEAP, Base::Unknown)
            }
            Op::Javascript(JsOp::IsReadLayout(..)) => (Locations::CODE, Base::Unknown),
            Op::Javascript(JsOp::ProbeShapeProperty(_, _, layout)) if layout.storage.is_some() => {
                (Locations::HEAP.union(Locations::CODE), Base::Unknown)
            }
            Op::Javascript(
                JsOp::PropertyLoad(object, ..)
                | JsOp::CachedProperty(object, ..)
                | JsOp::Get(object, _)
                | JsOp::ProbeProperty(object, _)
                | JsOp::ShapedGet(object, ..)
                | JsOp::GetIndex(object, _)
                | JsOp::TypedIndex(object, ..)
                | JsOp::ArrayGet(object, _)
                | JsOp::ArrayLoad(object, _, _)
                | JsOp::IndexedLength(object)
                | JsOp::ArrayKind(object)
                | JsOp::IsArray(object, _)
                | JsOp::ConstructorPrototype(object),
            ) => (Locations::HEAP, Base::Known(*object)),
            Op::Javascript(JsOp::ProbeShapeProperty(object, ..)) => {
                (Locations::HEAP.union(Locations::CODE), Base::Known(*object))
            }
            Op::EntryAssume(object, super::super::guards::Fact::Shape(_))
            | Op::Javascript(JsOp::IsShape(object, _) | JsOp::IsPropertyCached(object, ..)) => {
                (Locations::CODE, Base::Known(*object))
            }
            Op::Javascript(JsOp::CellGet(cell)) => (Locations::HEAP, Base::Known(*cell)),
            Op::Javascript(JsOp::GlobalRead(_) | JsOp::ProbeGlobal(_)) => {
                (Locations::GLOBAL, Base::External)
            }
            Op::HelperCall {
                helper: Helper::PropertyGetMegamorphic,
                ..
            } => (Locations::HEAP, Base::Unknown),
            Op::HelperCall {
                helper: Helper::PropertyGetCached | Helper::SemanticPropertyProbeCached,
                args,
                ..
            } => (Locations::HEAP, Base::Known(args[0])),
            _ => return None,
        };
        Some((
            location,
            Self {
                base,
                field: Field::of(graph, op, facts),
            },
        ))
    }

    pub(in crate::jit::semantic) fn written(graph: &Graph, op: &Op, facts: &Facts) -> Option<Self> {
        let field = match Field::of(graph, op, facts) {
            Field::Length => Field::Any,
            field => field,
        };
        let base = match op {
            Op::HelperCall {
                helper: Helper::PropertySetCached,
                args,
                ..
            } => Some(args[0]),
            _ => write_target(op),
        };
        (base.is_some() || field != Field::Any).then_some(Self {
            base: base.map_or(Base::External, Base::Known),
            field,
        })
    }
}

fn write_target(op: &Op) -> Option<Value> {
    match op {
        Op::Javascript(
            JsOp::CellSet(object, _)
            | JsOp::Define(object, ..)
            | JsOp::Set(object, ..)
            | JsOp::ShapedSet(object, ..)
            | JsOp::ArrayStore(object, ..)
            | JsOp::PropertyStore(object, ..)
            | JsOp::PropertyAdd(object, ..)
            | JsOp::FillPropertyStore(object, ..)
            | JsOp::SetIndex(object, ..)
            | JsOp::TypedSet(object, ..)
            | JsOp::ArrayAppend(object, ..)
            | JsOp::IteratorNext(object)
            | JsOp::EnumerateNext(object),
        ) => Some(*object),
        _ => None,
    }
}

impl Fresh {
    pub(in crate::jit::semantic) fn may_clobber(
        &self,
        inst: super::super::Inst,
        write: Option<&Access>,
        read: Option<&Access>,
    ) -> bool {
        // Broad accesses retain the JS conflict rules for unknown fields.
        let write = write.unwrap_or(&Access::UNKNOWN);
        let read = read.unwrap_or(&Access::UNKNOWN);
        self.may_alias(inst, write, read) && write.field.conflicts(&read.field)
    }

    // An own data property of an unescaped allocation shadows its prototype chain,
    // so only writes into that allocation can change it.
    pub(in crate::jit::semantic) fn may_clobber_own(
        &self,
        inst: super::super::Inst,
        write: Option<&Access>,
        read: &Access,
    ) -> bool {
        let write = write.unwrap_or(&Access::UNKNOWN);
        if !write.field.conflicts(&read.field) {
            return false;
        }
        let Some(base) = read.base.value().filter(|&base| self.unescaped(base, inst)) else {
            return true;
        };
        match write.base {
            Base::Unknown => true,
            Base::External => false,
            Base::Known(target) => self.same_class(target, base),
        }
    }
}
