use super::LoweredValue;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{MemFlagsData, types};

use crate::heap::MathFunction;
use crate::jit::semantic::Repr;
use crate::value::Value;

use super::FunctionCompiler;
use super::ops::BinaryOp;

#[derive(Clone, Copy)]
pub(super) struct Operand {
    pub(super) value: LoweredValue,
    pub(super) repr: Repr,
    constant: Option<f64>,
}

impl Operand {
    pub(super) fn new(value: LoweredValue, repr: Repr) -> Self {
        Self {
            value,
            repr,
            constant: None,
        }
    }

    pub(super) fn with_constant(self, number: f64) -> Self {
        Self {
            constant: Some(number),
            ..self
        }
    }
}

impl FunctionCompiler<'_, '_> {
    fn fold(&mut self, operand: Operand, to: Repr) -> Option<LoweredValue> {
        let number = operand.constant?;
        let integer = match operand.repr {
            Repr::I32 => Some(i64::from(number as i32)),
            Repr::I64 => Some(number as i64),
            Repr::F64 => None,
            Repr::Boxed | Repr::I8 | Repr::I16 | Repr::F32 => return None,
        };
        Some(match (integer, to) {
            (Some(integer), Repr::I64) => self.b.ins().iconst(types::I64, integer),
            (Some(integer), Repr::I32) => {
                self.b.ins().iconst(types::I32, i64::from(integer as i32))
            }
            (Some(integer), Repr::F64) => self.b.ins().f64const(integer as f64),
            (None, Repr::I64) => self.b.ins().iconst(types::I64, number as i64),
            (None, Repr::I32) => self
                .b
                .ins()
                .iconst(types::I32, i64::from(crate::jit::helpers::to_int32(number))),
            _ => return None,
        })
    }

    pub(super) fn convert(&mut self, operand: Operand, to: Repr) -> LoweredValue {
        let value = operand.value;
        if operand.repr != to
            && let Some(folded) = self.fold(operand, to)
        {
            return folded;
        }
        match (operand.repr, to) {
            (from, to) if from == to => value,
            (Repr::Boxed, Repr::F64) => {
                self.b.ins().bitcast(types::F64, MemFlagsData::new(), value)
            }
            (Repr::Boxed, Repr::I64) => {
                let float = self.b.ins().bitcast(types::F64, MemFlagsData::new(), value);
                self.b.ins().fcvt_to_sint_sat(types::I64, float)
            }
            (Repr::Boxed, Repr::I32) => {
                let float = self.b.ins().bitcast(types::F64, MemFlagsData::new(), value);
                self.to_int32(float)
            }
            (Repr::Boxed, Repr::I8) => {
                self.b
                    .ins()
                    .icmp_imm_s(IntCC::Equal, value, Value::TRUE.bits() as i64)
            }
            (Repr::F64, Repr::Boxed) => {
                let unordered = self.b.ins().fcmp(FloatCC::Unordered, value, value);
                let bits = self.b.ins().bitcast(types::I64, MemFlagsData::new(), value);
                let nan = self.constant(Value::number(f64::NAN));

                self.b.ins().select(unordered, nan, bits)
            }
            (Repr::I64, Repr::Boxed) | (Repr::I32, Repr::Boxed) => {
                let float = self.b.ins().fcvt_from_sint(types::F64, value);

                self.b.ins().bitcast(types::I64, MemFlagsData::new(), float)
            }
            (Repr::I8, Repr::Boxed) => {
                let yes = self.constant(Value::TRUE);
                let no = self.constant(Value::FALSE);
                self.b.ins().select(value, yes, no)
            }
            (Repr::I64, Repr::F64) | (Repr::I32, Repr::F64) => {
                self.b.ins().fcvt_from_sint(types::F64, value)
            }
            (Repr::F64, Repr::I32) => self.to_int32(value),
            (Repr::I8, Repr::I32) => self.b.ins().uextend(types::I32, value),
            (Repr::I8, Repr::F64) => {
                let wide = self.b.ins().uextend(types::I32, value);
                self.b.ins().fcvt_from_uint(types::F64, wide)
            }
            (Repr::I64, Repr::I8) | (Repr::I32, Repr::I8) => {
                self.b.ins().icmp_imm_s(IntCC::NotEqual, value, 0)
            }
            (Repr::F64, Repr::I8) => {
                let zero = self.b.ins().f64const(0.0);
                let negative = self.b.ins().fcmp(FloatCC::LessThan, value, zero);
                let positive = self.b.ins().fcmp(FloatCC::GreaterThan, value, zero);
                self.b.ins().bor(negative, positive)
            }
            (Repr::I32, Repr::I64) => self.b.ins().sextend(types::I64, value),
            (Repr::F64, Repr::I64) => self.b.ins().fcvt_to_sint_sat(types::I64, value),
            (Repr::I64, Repr::I32) => self.b.ins().ireduce(types::I32, value),
            _ => unreachable!(),
        }
    }

    pub(super) fn to_int32(&mut self, float: LoweredValue) -> LoweredValue {
        let scale = self.b.ins().f64const(1.0 / 4294967296.0);
        let scaled = self.b.ins().fmul(float, scale);
        let wraps = self.b.ins().trunc(scaled);
        let modulus = self.b.ins().f64const(4294967296.0);
        let wrapped = self.b.ins().fmul(wraps, modulus);
        let remainder = self.b.ins().fsub(float, wrapped);
        let integer = self.b.ins().fcvt_to_sint_sat(types::I64, remainder);
        self.b.ins().ireduce(types::I32, integer)
    }

    pub(super) fn typed_binary(
        &mut self,
        op: BinaryOp,
        left: Operand,
        right: Operand,
        result: Repr,
    ) -> Operand {
        debug_assert_eq!(left.repr, right.repr);
        let repr = left.repr;
        let (left, right) = (left.value, right.value);
        let value = match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul if result == Repr::I64 => {
                debug_assert_eq!(repr, Repr::I64);
                match op {
                    BinaryOp::Add => self.b.ins().iadd(left, right),
                    BinaryOp::Sub => self.b.ins().isub(left, right),
                    _ => self.b.ins().imul(left, right),
                }
            }
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                debug_assert_eq!(repr, Repr::F64);
                match op {
                    BinaryOp::Add => self.b.ins().fadd(left, right),
                    BinaryOp::Sub => self.b.ins().fsub(left, right),
                    BinaryOp::Mul => self.b.ins().fmul(left, right),
                    _ => self.b.ins().fdiv(left, right),
                }
            }
            BinaryOp::Less
            | BinaryOp::LessEqual
            | BinaryOp::Greater
            | BinaryOp::GreaterEqual
            | BinaryOp::LooseEqual
            | BinaryOp::StrictEqual
            | BinaryOp::LooseNotEqual
            | BinaryOp::StrictNotEqual => {
                if matches!(repr, Repr::I8 | Repr::I32 | Repr::I64) {
                    let condition = match op {
                        BinaryOp::Less => IntCC::SignedLessThan,
                        BinaryOp::LessEqual => IntCC::SignedLessThanOrEqual,
                        BinaryOp::Greater => IntCC::SignedGreaterThan,
                        BinaryOp::GreaterEqual => IntCC::SignedGreaterThanOrEqual,
                        BinaryOp::LooseEqual | BinaryOp::StrictEqual => IntCC::Equal,
                        _ => IntCC::NotEqual,
                    };
                    self.b.ins().icmp(condition, left, right)
                } else {
                    debug_assert_eq!(repr, Repr::F64);
                    let condition = match op {
                        BinaryOp::Less => FloatCC::LessThan,
                        BinaryOp::LessEqual => FloatCC::LessThanOrEqual,
                        BinaryOp::Greater => FloatCC::GreaterThan,
                        BinaryOp::GreaterEqual => FloatCC::GreaterThanOrEqual,
                        BinaryOp::LooseEqual | BinaryOp::StrictEqual => FloatCC::Equal,
                        _ => FloatCC::NotEqual,
                    };
                    self.b.ins().fcmp(condition, left, right)
                }
            }
            BinaryOp::BitwiseAnd
            | BinaryOp::BitwiseOr
            | BinaryOp::BitwiseXor
            | BinaryOp::ShiftLeft
            | BinaryOp::ShiftRight
            | BinaryOp::ShiftRightUnsigned => {
                debug_assert_eq!(repr, Repr::I32);
                match op {
                    BinaryOp::BitwiseAnd => self.b.ins().band(left, right),
                    BinaryOp::BitwiseOr => self.b.ins().bor(left, right),
                    BinaryOp::BitwiseXor => self.b.ins().bxor(left, right),
                    BinaryOp::ShiftLeft => self.b.ins().ishl(left, right),
                    BinaryOp::ShiftRight => self.b.ins().sshr(left, right),
                    _ => {
                        let shifted = self.b.ins().ushr(left, right);
                        self.b.ins().fcvt_from_uint(types::F64, shifted)
                    }
                }
            }
            _ => unreachable!("unsupported selected binary operation"),
        };
        Operand::new(value, result)
    }

    pub(super) fn typed_math(&mut self, function: MathFunction, args: &[Operand]) -> Operand {
        debug_assert!(args.iter().all(|arg| arg.repr == Repr::F64));
        let mut values = args.iter().map(|arg| arg.value);
        let first = values
            .next()
            .expect("selected math operation has an argument");
        let result = match function {
            MathFunction::Floor => self.b.ins().floor(first),
            MathFunction::Ceil => self.b.ins().ceil(first),
            MathFunction::Sqrt => self.b.ins().sqrt(first),
            MathFunction::Min => values.fold(first, |left, right| self.b.ins().fmin(left, right)),
            MathFunction::Max => values.fold(first, |left, right| self.b.ins().fmax(left, right)),
            _ => unreachable!("unsupported selected math operation"),
        };
        Operand::new(result, Repr::F64)
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
