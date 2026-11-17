use super::super::{Facts, Graph, Op, Value, ValueType};
use crate::jit::effects::Locations;
use crate::jit::helpers::Helper;

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
            Some(Op::PropertyCache(_, name)) => Self::named(name),
            _ => Self::Any,
        }
    }

    fn of(graph: &Graph, op: &Op, facts: &Facts) -> Self {
        match op {
            Op::PropertyLoad(_, _, field) | Op::PropertyStore(_, _, _, field, _) => field.clone(),
            Op::CachedProperty(_, cache, _) | Op::PropertyAdd(_, cache, ..) => {
                Self::cached(graph, *cache)
            }
            Op::HelperCall {
                helper:
                    Helper::PropertyGetCached
                    | Helper::SemanticPropertyProbeCached
                    | Helper::PropertySetCached,
                args,
                ..
            } => Self::cached(graph, args[1]),
            Op::Read(_) | Op::Initialize(..) | Op::Write(..) | Op::CellGet(_) | Op::CellSet(..) => {
                Self::Cells
            }
            Op::Get(_, name)
            | Op::FillPropertyStore(_, _, name, _, _)
            | Op::Set(_, name, _)
            | Op::Define(_, name, _)
            | Op::ShapedGet(_, name, _)
            | Op::ProbeProperty(_, name)
            | Op::ShapedSet(_, name, _, _)
            | Op::GlobalRead(name)
            | Op::ProbeGlobal(name)
            | Op::GlobalWrite(name, _)
            | Op::GlobalInitialize(name, _) => Self::named(name),
            Op::GetIndex(_, key)
            | Op::TypedIndex(_, key, _)
            | Op::ArrayGet(_, key)
            | Op::ArrayLoad(_, key, _)
            | Op::ArrayStore(_, key, _)
            | Op::SetIndex(_, key, _)
                if facts.of(*key).within(ValueType::NUMBER) =>
            {
                Self::Elements
            }
            Op::IndexedLength(_) => Self::Length,
            Op::ArrayAppend(..) | Op::MapElement(..) => Self::Elements,
            Op::PackAppend(..) | Op::PackRelease(_) => Self::Packs,
            Op::IteratorNext(_) | Op::EnumerateNext(_) => Self::Iteration,
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit::semantic) struct Access {
    pub(in crate::jit::semantic) base: Option<Value>,
    pub(in crate::jit::semantic) field: Field,
}

impl Access {
    pub(in crate::jit::semantic) const UNKNOWN: Self = Self {
        base: None,
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
            Op::PropertyLoad(object, ..)
            | Op::CachedProperty(object, ..)
            | Op::Get(object, _)
            | Op::ProbeProperty(object, _)
            | Op::ShapedGet(object, ..)
            | Op::GetIndex(object, _)
            | Op::TypedIndex(object, ..)
            | Op::ArrayGet(object, _)
            | Op::ArrayLoad(object, _, _)
            | Op::IndexedLength(object)
            | Op::ArrayKind(object)
            | Op::IsArray(object, _)
            | Op::ConstructorPrototype(object) => (Locations::HEAP, Some(*object)),
            Op::ProbeShapeProperty(object, ..) => {
                (Locations::HEAP.union(Locations::CODE), Some(*object))
            }
            Op::EntryAssume(object, super::super::guards::Fact::Shape(_))
            | Op::IsShape(object, _)
            | Op::IsPropertyCached(object, ..) => (Locations::CODE, Some(*object)),
            Op::CellGet(cell) => (Locations::HEAP, Some(*cell)),
            Op::GlobalRead(_) | Op::ProbeGlobal(_) => (Locations::GLOBAL, None),
            Op::HelperCall {
                helper: Helper::PropertyGetCached | Helper::SemanticPropertyProbeCached,
                args,
                ..
            } => (Locations::HEAP, Some(args[0])),
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
        (base.is_some() || field != Field::Any).then_some(Self { base, field })
    }
}

fn write_target(op: &Op) -> Option<Value> {
    match op {
        Op::CellSet(object, _)
        | Op::Define(object, ..)
        | Op::Set(object, ..)
        | Op::ShapedSet(object, ..)
        | Op::ArrayStore(object, ..)
        | Op::PropertyStore(object, ..)
        | Op::PropertyAdd(object, ..)
        | Op::FillPropertyStore(object, ..)
        | Op::SetIndex(object, ..)
        | Op::ArrayAppend(object, ..)
        | Op::IteratorNext(object)
        | Op::EnumerateNext(object) => Some(*object),
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
}
