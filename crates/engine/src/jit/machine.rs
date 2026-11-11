use cranelift_codegen::Context;
use cranelift_codegen::control::ControlPlane;
use cranelift_codegen::ir::TrapCode;
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module, ModuleReloc};

use crate::error::Error;

#[derive(Clone, Copy)]
pub(super) struct SourceRange {
    pub start: u32,
    pub end: u32,
    pub location: Option<u32>,
}

pub(super) struct MachineCode {
    pub bytes: Vec<u8>,
    pub alignment: u64,
    pub relocs: Vec<ModuleReloc>,
    pub traps: Vec<(u32, TrapCode)>,
    pub locations: Vec<SourceRange>,
    pub block_starts: Vec<u32>,
}

impl MachineCode {
    pub(super) fn from_cranelift(
        context: &mut Context,
        module: &JITModule,
        id: FuncId,
    ) -> Result<Self, Error> {
        context
            .compile(module.isa(), &mut ControlPlane::default())
            .map_err(|error| Error::Compile(format!("{:?}", error.inner)))?;
        let compiled = context
            .compiled_code()
            .ok_or_else(|| Error::Compile("missing compiled code".into()))?;
        let buffer = &compiled.buffer;
        let locations = buffer
            .get_srclocs_sorted()
            .iter()
            .map(|range| SourceRange {
                start: range.start,
                end: range.end,
                location: Some(range.loc.as_abs())
                    .filter(|location| !location.is_default())
                    .map(|location| location.bits()),
            })
            .collect();
        let alignment = u64::from(buffer.min_alignment);
        let traps = buffer
            .traps()
            .iter()
            .map(|trap| (trap.offset, trap.code))
            .collect();
        let block_starts = compiled.bb_starts.clone();
        let bytes = compiled.code_buffer().to_vec();
        let relocs = buffer
            .relocs()
            .iter()
            .map(|reloc| ModuleReloc::from_mach_reloc(reloc, &context.func, id))
            .collect();
        Ok(Self {
            bytes,
            alignment,
            relocs,
            traps,
            locations,
            block_starts,
        })
    }

    pub(super) fn define(&self, module: &mut JITModule, id: FuncId) -> Result<(), Error> {
        module
            .define_function_bytes(id, self.alignment, &self.bytes, &self.relocs)
            .map_err(|error| Error::Compile(format!("{error:?}")))
    }

    pub(super) fn location_at(&self, offset: u32) -> Option<u32> {
        let index = self
            .locations
            .partition_point(|range| range.start <= offset);
        index
            .checked_sub(1)
            .and_then(|i| self.locations.get(i))
            .filter(|range| offset < range.end)
            .and_then(|range| range.location)
    }
}
