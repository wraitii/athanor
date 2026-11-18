use super::super::repr::element;
use super::super::{MemoryFormat, Repr};
use super::{Op, TypedRead};

pub(in crate::jit) fn intrinsic(op: &Op) -> Option<Repr> {
    Some(match op {
        Op::TypedData(_) => Repr::Word,
        Op::PropertyCache(..) | Op::GlobalCache(..) => Repr::Word,
        Op::MemoryNumber(_, format) => match format {
            MemoryFormat::U32 | MemoryFormat::F32 | MemoryFormat::F64 => Repr::F64,
            _ => Repr::I32,
        },
        Op::NumericFloat(_) => Repr::F64,
        Op::AsNumber(_) => return None,
        Op::SaturatingInteger(_) | Op::AsInteger(_) | Op::ArrayPosition(_) => Repr::I64,
        Op::ToInt32(_) | Op::ArrayKind(_) => Repr::I32,
        Op::IntegerCheck(..)
        | Op::IsInteger(_)
        | Op::IsIntegerInRange(..)
        | Op::IsCode(..)
        | Op::IsShape(..)
        | Op::IsReadLayout(..)
        | Op::IsTyped(..)
        | Op::IsPropertyCached(..)
        | Op::IsIndexed(_)
        | Op::IsOrdinaryArray(_)
        | Op::IsString(_)
        | Op::Truthy(_)
        | Op::IsNumber(_)
        | Op::IsUninitialized(_)
        | Op::IsUndefined(_)
        | Op::IsNullish(_)
        | Op::CanOverwrite(..)
        | Op::IsArray(..) => Repr::Boolean,
        Op::TypedIndex(
            _,
            _,
            TypedRead {
                element: Some(kind),
                bounded: true,
            },
        ) => element(*kind),
        _ => Repr::Boxed,
    })
}
