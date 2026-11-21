pub(crate) mod graph;
mod liveness;
pub(crate) mod moves;

use cranelift_codegen::ir::{Type, types};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RegClass {
    Int,
    Float,
    Vector,
}

impl RegClass {
    pub(crate) fn of(ty: Type) -> Option<Self> {
        match ty {
            types::I8 | types::I16 | types::I32 | types::I64 => Some(Self::Int),
            types::F32 | types::F64 => Some(Self::Float),
            _ => None,
        }
    }
}

pub(crate) struct Registers {
    pub caller_saved: &'static [u8],
    pub callee_saved: &'static [u8],
}

pub(crate) struct Target {
    pub int: Registers,
    pub float: Registers,
    pub vector: Registers,
}

impl Target {
    fn registers(&self, class: RegClass) -> &Registers {
        match class {
            RegClass::Int => &self.int,
            RegClass::Float => &self.float,
            RegClass::Vector => &self.vector,
        }
    }
}
