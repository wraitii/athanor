use crate::FxHashMap;
use cranelift_codegen::ir::{Function, Inst};

use super::helpers::Helper;

#[derive(Clone, Default)]
pub(super) struct Annotations {
    pub semantic_locations: bool,
    pub helpers: FxHashMap<Inst, Helper>,
}

#[derive(Clone)]
pub(super) struct FunctionBody {
    pub function: Function,
    pub lowered: Option<super::semantic::Graph>,
    pub baseline: Option<std::sync::Arc<super::machine::MachineCode>>,
    pub annotations: Annotations,
    pub listing: Option<super::jitdump::Listing>,
    pub inspection: super::inspect::Inspection,
}
