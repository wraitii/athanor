mod abi;
mod alloc;
mod asm;
mod baseline;
mod encode;
mod lower;
mod select;

use cranelift_codegen::ir::Function;

use super::machine::MachineCode;

#[derive(Debug)]
pub(super) struct Unsupported(pub String);

pub(super) fn compile(
    func: &Function,
    indexed_loads: &rustc_hash::FxHashMap<
        cranelift_codegen::ir::Inst,
        super::semantic::MemoryFormat,
    >,
) -> Result<MachineCode, Unsupported> {
    lower::lower(func, indexed_loads)
}

pub(super) fn compile_baseline(
    graph: &super::semantic::Graph,
    source_name: (usize, usize),
    semantic_locations: bool,
    names: &mut Vec<Box<str>>,
    data: &mut super::function_data::FunctionData,
) -> Result<MachineCode, Unsupported> {
    baseline::compile(graph, source_name, semantic_locations, names, data)
}
