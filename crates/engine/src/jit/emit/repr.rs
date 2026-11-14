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
    pub(super) fn tagged(&mut self, operand: Operand) -> LoweredValue {
        let actual = self.b.graph.repr(operand.value);
        if actual == operand.repr {
            return operand.value;
        }
        assert_eq!(
            (actual, operand.repr),
            (Repr::I64, Repr::Boxed),
            "{} tagged as {:?}",
            operand.value,
            operand.repr
        );
        self.b.as_boxed(operand.value)
    }

    pub(super) fn convert(&mut self, operand: Operand, to: Repr) -> LoweredValue {
        let value = self.tagged(operand);
        steps(operand.repr, to)
            .into_iter()
            .fold(value, |value, step| {
                let (op, repr) = step.op(value);
                self.b.semantic(op, repr)
            })
    }

    pub(super) fn condition(&mut self, operand: Operand, numeric: bool) -> LoweredValue {
        match operand.repr {
            Repr::Boxed if numeric => {
                let float = self.convert(operand, Repr::F64);
                self.convert(Operand::new(float, Repr::F64), Repr::I8)
            }
            Repr::Boxed => self.truthy(operand.value),
            _ => self.convert(operand, Repr::I8),
        }
    }
}
