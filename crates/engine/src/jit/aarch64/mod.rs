mod abi;
mod asm;
mod encode;
mod lower;

use cranelift_codegen::ir::Function;

use super::machine::MachineCode;

#[derive(Debug)]
pub(super) struct Unsupported(pub String);

pub(super) fn compile(func: &Function) -> Result<MachineCode, Unsupported> {
    lower::lower(func)
}
