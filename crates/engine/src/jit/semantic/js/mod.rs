pub(in crate::jit::semantic) mod arrays;
mod bit_tests;
pub(in crate::jit::semantic) mod captures;
mod coercions;
pub(in crate::jit::semantic) mod comparisons;
pub(in crate::jit::semantic) mod construction;
pub(in crate::jit::semantic) mod conversions;
mod folding;
pub(in crate::jit::semantic) mod string_split;
mod typeof_comparisons;

use super::ValueType;

pub(super) use bit_tests::bit_tests;
pub(super) use coercions::coercions;
pub(super) use folding::simplify;
pub(super) use typeof_comparisons::typeof_comparisons;

pub(super) const NUMERIC: ValueType = ValueType::NUMBER.union(ValueType::UNDEFINED);
