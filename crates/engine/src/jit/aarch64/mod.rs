mod abi;
mod alloc;
mod asm;
mod baseline;
mod encode;
mod lower;
mod select;

use super::machine::MachineCode;

#[derive(Debug)]
pub(super) struct Unsupported(pub String);

pub(super) fn compile_graph(graph: &super::semantic::Graph) -> Result<MachineCode, Unsupported> {
    let mut graph = graph.clone();
    super::semantic::lower_machine(&mut graph).map_err(Unsupported)?;
    lower::lower(&graph)
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
