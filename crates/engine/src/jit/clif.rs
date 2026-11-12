use cranelift_codegen::ir::{Function, Inst};
use rustc_hash::FxHashMap as HashMap;

use super::helpers::Helper;

#[derive(Clone, Default)]
pub(super) struct Annotations {
    pub semantic_locations: bool,
    pub helpers: HashMap<Inst, Helper>,
}

#[derive(Clone)]
pub(super) struct FunctionBody {
    pub function: Function,
    pub baseline: Option<std::sync::Arc<super::machine::MachineCode>>,
    pub annotations: Annotations,
    pub listing: Option<Vec<super::semantic::ListingLine>>,
    pub inspection: super::inspect::Inspection,
}
