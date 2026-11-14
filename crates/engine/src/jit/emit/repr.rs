use super::LoweredValue;

use crate::jit::semantic::Repr;
use crate::jit::semantic::repr::steps;

use super::FunctionCompiler;

#[derive(Clone, Copy)]
pub(super) struct Operand {
    pub(super) value: LoweredValue,
    pub(super) repr: Repr,
}

impl Operand {
    pub(super) fn new(value: LoweredValue, repr: Repr) -> Self {
        Self { value, repr }
    }
}

impl FunctionCompiler<'_, '_> {
    pub(super) fn convert(&mut self, operand: Operand, to: Repr) -> LoweredValue {
        steps(operand.repr, to)
            .into_iter()
            .fold(operand.value, |value, step| {
                let (op, repr) = step.op(value);
                self.b.semantic(op, repr)
            })
    }
}
