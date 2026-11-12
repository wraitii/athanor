use super::ValueType;

mod bit_tests;
mod coercions;
mod folding;
mod typeof_comparisons;

pub(super) use bit_tests::bit_tests;
pub(super) use coercions::coercions;
pub(super) use folding::simplify;
pub(super) use typeof_comparisons::typeof_comparisons;

pub(super) const NUMERIC: ValueType = ValueType::NUMBER.union(ValueType::UNDEFINED);
