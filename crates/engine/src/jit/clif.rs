use cranelift_codegen::ir::Function;

#[derive(Clone)]
pub(super) struct FunctionBody {
    pub function: Function,
    pub lowered: Option<super::semantic::Graph>,
    pub baseline: Option<std::sync::Arc<super::machine::MachineCode>>,
    pub semantic_locations: bool,
    pub listing: Option<super::jitdump::Listing>,
    pub inspection: super::inspect::Inspection,
}
