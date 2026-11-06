use cranelift_codegen::ir::{Block, Function, Inst, Value};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::helpers::Helper;

#[derive(Clone, Copy)]
pub(super) struct IndexedRead {
    pub call: Inst,
    pub receiver: Value,
    pub key: Value,
    pub result: Value,
    pub success: Block,
    pub source: (u32, u32),
}

#[derive(Clone, Default)]
pub(super) struct Annotations {
    pub semantic_locations: bool,
    pub indexed_reads: Vec<IndexedRead>,
    pub helpers: HashMap<Inst, Helper>,
    pub temporary_pushes: HashSet<Inst>,
    pub temporary_clears: HashMap<Inst, Block>,
    pub temporary_resets: HashSet<Inst>,
    pub frame_stores: HashSet<Inst>,
}

#[derive(Clone)]
pub(super) struct FunctionBody {
    pub function: Function,
    pub annotations: Annotations,
    pub listing: Option<Vec<super::semantic::ListingLine>>,
}
