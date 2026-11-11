use cranelift_codegen::ir::{Function, Inst};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::helpers::Helper;

#[derive(Clone, Default)]
pub(super) struct Annotations {
    pub semantic_locations: bool,
    pub helpers: HashMap<Inst, Helper>,
    pub frame_stores: HashSet<Inst>,
}

#[derive(Clone)]
pub(super) struct FunctionBody {
    pub function: Function,
    pub annotations: Annotations,
    pub listing: Option<Vec<super::semantic::ListingLine>>,
    pub inspection: super::inspect::Inspection,
}
