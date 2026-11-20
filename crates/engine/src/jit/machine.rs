use crate::jit::module::CodeModule;
use cranelift_codegen::Context;
use cranelift_codegen::control::ControlPlane;
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
    pub locations: Vec<SourceRange>,
}

impl MachineCode {
    pub(super) fn from_cranelift(
        context: &mut Context,
        module: &CodeModule,
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
            locations,
        })
    }

    pub(super) fn define(&self, module: &mut CodeModule, id: FuncId) -> Result<(), Error> {
        module
            .define_function_bytes(id, self.alignment, &self.bytes, &self.relocs)
            .map_err(|error| Error::Compile(format!("{error:?}")))
    }
}
