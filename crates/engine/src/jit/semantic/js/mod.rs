mod op;
pub(in crate::jit) use op::{
    ArrayLayout, ArrayMethod, Constant, DirectTarget, Native, Op, PropertyWrite, TypedRead, UnaryOp,
};
pub(in crate::jit::semantic) mod arguments;
pub(in crate::jit::semantic) mod arrays;
mod bit_tests;
pub(in crate::jit::semantic) mod captures;
mod coercions;
pub(in crate::jit::semantic) mod comparisons;
pub(in crate::jit::semantic) mod construction;
pub(in crate::jit::semantic) mod contracts;
pub(in crate::jit::semantic) mod conversions;
pub(in crate::jit::semantic) mod facts;
mod folding;
pub(super) mod intrinsics;
pub(in crate::jit::semantic) mod listing;
pub(in crate::jit::semantic) mod memory;
pub(in crate::jit::semantic) mod regexps;
pub(in crate::jit::semantic) mod repr;
pub(in crate::jit::semantic) mod string_split;
pub(in crate::jit::semantic) mod strings;
mod typeof_comparisons;

use super::ValueType;

pub(super) use bit_tests::bit_tests;
pub(super) use coercions::coercions;
pub(super) use folding::simplify;
pub(super) use typeof_comparisons::typeof_comparisons;

pub(super) const NUMERIC: ValueType = ValueType::NUMBER.union(ValueType::UNDEFINED);
