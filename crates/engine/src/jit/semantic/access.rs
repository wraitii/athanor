use super::super::effects::Locations;
use super::{Facts, Op, Value, ValueType};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum Field {
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
    fn named(name: &str) -> Self {
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

    fn of(op: &Op, facts: &Facts) -> Self {
        match op {
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

    pub(super) fn conflicts(&self, read: &Self) -> bool {
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
pub(super) struct Access {
    pub(super) location: Locations,
    pub(super) base: Option<Value>,
    pub(super) field: Field,
}

impl Access {
    pub(super) fn overlaps(&self, read: &Self) -> bool {
        let data = Locations::HEAP.union(Locations::GLOBAL);
        let code = Locations::CODE;
        (self.location.intersects(data) && read.location.intersects(data))
            || (self.location.intersects(code) && read.location.intersects(code))
    }

    pub(super) fn read(op: &Op, facts: &Facts) -> Option<Self> {
        let (location, base) = match op {
            Op::Get(object, _)
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
            Op::EntryAssume(object, super::guards::Fact::Shape(_))
            | Op::IsShape(object, _)
            | Op::IsPropertyCached(object, ..) => (Locations::CODE, Some(*object)),
            Op::CellGet(cell) => (Locations::HEAP, Some(*cell)),
            Op::GlobalRead(_) => (Locations::GLOBAL, None),
            _ => return None,
        };
        Some(Self {
            location,
            base,
            field: Field::of(op, facts),
        })
    }

    pub(super) fn written(op: &Op, facts: &Facts, location: Locations) -> Self {
        let field = match Field::of(op, facts) {
            Field::Length => Field::Any,
            field => field,
        };
        Self {
            location,
            base: op.write_target(),
            field,
        }
    }
}
