mod abi;
mod alloc;
mod asm;
mod baseline;
mod emit;
mod encode;
pub(in crate::jit) mod inst;
mod prepare;
mod select;

use super::machine::MachineCode;
pub(super) use select::run as select_graph;

#[derive(Debug)]
pub(super) struct Unsupported(pub String);

pub(super) fn compile_graph(graph: &super::semantic::Graph) -> Result<MachineCode, Unsupported> {
    let mut graph = graph.clone();
    select::run(&mut graph)?;
    let prepared = prepare::run(&graph)?;
    let allocation = crate::jit::alloc::graph::allocate(
        &graph,
        &prepared.cfg,
        &prepared.order,
        &alloc::TARGET,
        &prepared.input,
    )
    .map_err(Unsupported)?;
    emit::emit(&graph, &prepared, &allocation)
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
