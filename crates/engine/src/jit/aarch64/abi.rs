use cranelift_codegen::ir::{AbiParam, ArgumentPurpose, Type};
use cranelift_codegen::isa::CallConv;

use super::Unsupported;
use super::encode::Mem;
use crate::jit::alloc::RegClass;

#[derive(Clone, Copy, Debug)]
pub(super) enum Location {
    Reg { class: RegClass, reg: u8 },
    Stack { offset: u32, load: Mem, store: Mem },
}

#[derive(Clone, Debug)]
pub(super) struct Assignment {
    pub locations: Vec<Location>,
    pub stack_size: u32,
}

pub(super) fn assign(conv: CallConv, params: &[AbiParam]) -> Result<Assignment, Unsupported> {
    let mut next_int = match conv {
        CallConv::Tail => 2,
        CallConv::AppleAarch64 => 0,
        _ => return Err(Unsupported(format!("calling convention {conv}"))),
    };
    let mut next_float = 0;
    let mut stack_size: u32 = 0;
    let locations = params
        .iter()
        .map(|param| {
            if param.purpose != ArgumentPurpose::Normal {
                return Err(Unsupported(format!("argument purpose {}", param.purpose)));
            }
            let ty = param.value_type;
            let class =
                RegClass::of(ty).ok_or_else(|| Unsupported(format!("argument type {ty}")))?;
            let next = match class {
                RegClass::Int => &mut next_int,
                RegClass::Float => &mut next_float,
            };
            if *next < 8 {
                *next += 1;
                return Ok(Location::Reg {
                    class,
                    reg: *next - 1,
                });
            }
            let size = if conv == CallConv::AppleAarch64 {
                ty.bytes()
            } else {
                ty.bytes().max(8)
            };
            let offset = stack_size.next_multiple_of(size);
            stack_size = offset + size;
            Ok(Location::Stack {
                offset,
                load: memory(ty, true).unwrap(),
                store: memory(ty, false).unwrap(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Assignment {
        locations,
        stack_size: stack_size.next_multiple_of(16),
    })
}

pub(super) fn assign_returns(
    conv: CallConv,
    returns: &[AbiParam],
) -> Result<Vec<Location>, Unsupported> {
    let assignment = assign(conv, returns)?;
    if assignment.stack_size != 0 {
        return Err(Unsupported("stack returns".into()));
    }
    Ok(assignment.locations)
}

pub(super) fn memory(ty: Type, load: bool) -> Option<Mem> {
    use cranelift_codegen::ir::types;
    let (read, write) = match ty {
        types::I8 => (Mem::Ldrb, Mem::Strb),
        types::I16 => (Mem::Ldrh, Mem::Strh),
        types::I32 => (Mem::LdrW, Mem::StrW),
        types::I64 => (Mem::LdrX, Mem::StrX),
        types::F32 => (Mem::LdrS, Mem::StrS),
        types::F64 => (Mem::LdrD, Mem::StrD),
        _ => return None,
    };
    Some(if load { read } else { write })
}

pub(super) const T0: u8 = 9;
pub(super) const T1: u8 = 10;
pub(super) const T2: u8 = 11;
pub(super) const CALLEE: u8 = 16;
pub(super) const ADDRESS: u8 = 17;
pub(super) const F0: u8 = 16;
pub(super) const F1: u8 = 17;
