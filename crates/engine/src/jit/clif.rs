use cranelift_codegen::ir::{Function, Inst};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::helpers::Helper;

#[derive(Clone, Default)]
pub(super) struct Annotations {
    pub semantic_locations: bool,
    pub helpers: HashMap<Inst, Helper>,
    pub frame_stores: HashSet<Inst>,
    // Named call instructions carry graph loads through CLIF storage, never executable calls.
    pub indexed_loads: HashMap<Inst, super::semantic::MemoryFormat>,
}

#[derive(Clone)]
pub(super) struct FunctionBody {
    pub function: Function,
    pub cranelift: Option<Function>,
    pub baseline: Option<std::sync::Arc<super::machine::MachineCode>>,
    pub annotations: Annotations,
    pub listing: Option<Vec<super::semantic::ListingLine>>,
    pub inspection: super::inspect::Inspection,
}
