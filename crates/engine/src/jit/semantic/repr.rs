use cranelift_codegen::ir::{Type, types};

use super::{Facts, Graph, MemoryFormat, MemoryOp, Op, ValueType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::jit) enum Repr {
    Boxed,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl Repr {
    pub(in crate::jit) fn of(types: ValueType) -> Self {
        if types == ValueType::default() {
            Self::Boxed
        } else if types.within(ValueType::BOOLEAN) {
            Self::I8
        } else if types.within(ValueType::INT32) {
            Self::I32
        } else if types.within(ValueType::INTEGER) {
            Self::I64
        } else if types.within(ValueType::NUMBER) {
            Self::F64
        } else {
            Self::Boxed
        }
    }

    pub(in crate::jit) fn admits(self, types: ValueType) -> bool {
        match self {
            Self::Boxed => true,
            Self::I8 => types.within(ValueType::BOOLEAN),
            Self::I32 => types.within(ValueType::INT32),
            Self::I64 => types.within(ValueType::INTEGER),
            Self::F64 => types.within(ValueType::NUMBER),
            Self::I16 | Self::F32 => false,
        }
    }

    pub(in crate::jit) fn coerce(self, types: ValueType) -> ValueType {
        if self.admits(types) {
            return types;
        }
        match self {
            Self::Boxed => types,
            Self::I8 => ValueType::BOOLEAN,
            Self::I16 | Self::I32 => ValueType::INT32,
            Self::I64 => ValueType::INT64,
            Self::F32 | Self::F64 => ValueType::NUMBER,
        }
    }

    pub(in crate::jit) fn machine(self) -> Type {
        match self {
            Self::Boxed | Self::I64 => types::I64,
            Self::I8 => types::I8,
            Self::I16 => types::I16,
            Self::I32 => types::I32,
            Self::F32 => types::F32,
            Self::F64 => types::F64,
        }
    }

    pub(in crate::jit) fn from_machine(ty: Type) -> Self {
        match ty {
            types::I8 => Self::I8,
            types::I16 => Self::I16,
            types::I32 => Self::I32,
            types::I64 => Self::I64,
            types::F32 => Self::F32,
            types::F64 => Self::F64,
            _ => panic!("unsupported machine type {ty}"),
        }
    }
}

impl MemoryFormat {
    pub(in crate::jit) fn repr(self) -> Repr {
        match self {
            Self::U8 | Self::I8 => Repr::I8,
            Self::U16 | Self::I16 => Repr::I16,
            Self::U32 | Self::I32 => Repr::I32,
            Self::F32 => Repr::F32,
            Self::F64 => Repr::F64,
        }
    }
}

pub(super) fn assign(graph: &mut Graph, facts: &Facts) {
    for value in graph.values.keys() {
        let repr = match graph.defining_op(value) {
            Some(Op::TypedData(_) | Op::Memory(MemoryOp::Address { .. })) => Repr::I64,
            Some(Op::Memory(
                MemoryOp::IndexedLoad { format, .. } | MemoryOp::Load { format, .. },
            )) => format.repr(),
            Some(Op::SaturatingInteger(_)) => Repr::I64,
            _ => Repr::of(facts.of(value)),
        };
        graph.reprs[value] = Some(repr);
    }
}
