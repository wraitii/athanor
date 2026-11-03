use crate::heap::MathFunction;

pub(super) const METHODS: &[(&str, MathFunction)] = &[
    ("ceil", MathFunction::Ceil),
    ("floor", MathFunction::Floor),
    ("min", MathFunction::Min),
    ("max", MathFunction::Max),
    ("sqrt", MathFunction::Sqrt),
];
