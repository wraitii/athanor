use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{InstBuilder, MemFlagsData, Type, Value as ClifValue, types};

use crate::heap::MathFunction;
use crate::jit::semantic::Types;
use crate::value::Value;

use super::FunctionCompiler;
use super::ops::BinaryOp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Repr {
    Boxed,
    Float,
    Int32,
    Bool,
}

impl Repr {
    pub(super) fn of(types: Types) -> Self {
        if types == Types::default() {
            Self::Boxed
        } else if types.within(Types::BOOLEAN) {
            Self::Bool
        } else if types.within(Types::INT32) {
            Self::Int32
        } else if types.within(Types::NUMBER) {
            Self::Float
        } else {
            Self::Boxed
        }
    }

    pub(super) fn clif(self) -> Type {
        match self {
            Self::Boxed => types::I64,
            Self::Float => types::F64,
            Self::Int32 => types::I32,
            Self::Bool => types::I8,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Operand {
    pub(super) value: ClifValue,
    pub(super) repr: Repr,
}

impl Operand {
    pub(super) fn new(value: ClifValue, repr: Repr) -> Self {
        Self { value, repr }
    }
}

impl FunctionCompiler<'_, '_> {
    pub(super) fn zero(&mut self, repr: Repr) -> ClifValue {
        match repr {
            Repr::Boxed => self.constant(Value::UNDEFINED),
            Repr::Float => self.b.ins().f64const(0.0),
            Repr::Int32 => self.b.ins().iconst(types::I32, 0),
            Repr::Bool => self.b.ins().iconst(types::I8, 0),
        }
    }

    pub(super) fn convert(&mut self, operand: Operand, to: Repr) -> ClifValue {
        let value = operand.value;
        match (operand.repr, to) {
            (from, to) if from == to => value,
            (Repr::Boxed, Repr::Float) => {
                self.b.ins().bitcast(types::F64, MemFlagsData::new(), value)
            }
            (Repr::Boxed, Repr::Int32) => {
                let float = self.b.ins().bitcast(types::F64, MemFlagsData::new(), value);
                self.to_int32(float)
            }
            (Repr::Boxed, Repr::Bool) => {
                self.b
                    .ins()
                    .icmp_imm_s(IntCC::Equal, value, Value::TRUE.bits() as i64)
            }
            (Repr::Float, Repr::Boxed) => {
                let unordered = self.b.ins().fcmp(FloatCC::Unordered, value, value);
                let bits = self.b.ins().bitcast(types::I64, MemFlagsData::new(), value);
                let nan = self.constant(Value::number(f64::NAN));
                let boxed = self.b.ins().select(unordered, nan, bits);
                boxed
            }
            (Repr::Int32, Repr::Boxed) => {
                let float = self.b.ins().fcvt_from_sint(types::F64, value);
                let boxed = self.b.ins().bitcast(types::I64, MemFlagsData::new(), float);
                boxed
            }
            (Repr::Bool, Repr::Boxed) => {
                let yes = self.constant(Value::TRUE);
                let no = self.constant(Value::FALSE);
                self.b.ins().select(value, yes, no)
            }
            (Repr::Int32, Repr::Float) => self.b.ins().fcvt_from_sint(types::F64, value),
            (Repr::Float, Repr::Int32) => self.to_int32(value),
            (Repr::Bool, Repr::Int32) => self.b.ins().uextend(types::I32, value),
            (Repr::Bool, Repr::Float) => {
                let wide = self.b.ins().uextend(types::I32, value);
                self.b.ins().fcvt_from_uint(types::F64, wide)
            }
            (Repr::Int32, Repr::Bool) => self.b.ins().icmp_imm_s(IntCC::NotEqual, value, 0),
            (Repr::Float, Repr::Bool) => {
                let zero = self.b.ins().f64const(0.0);
                let negative = self.b.ins().fcmp(FloatCC::LessThan, value, zero);
                let positive = self.b.ins().fcmp(FloatCC::GreaterThan, value, zero);
                self.b.ins().bor(negative, positive)
            }
            _ => unreachable!(),
        }
    }

    pub(super) fn to_int32(&mut self, float: ClifValue) -> ClifValue {
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
    ) -> Option<Operand> {
        Some(match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                let left = self.convert(left, Repr::Float);
                let right = self.convert(right, Repr::Float);
                let result = match op {
                    BinaryOp::Add => self.b.ins().fadd(left, right),
                    BinaryOp::Sub => self.b.ins().fsub(left, right),
                    BinaryOp::Mul => self.b.ins().fmul(left, right),
                    _ => self.b.ins().fdiv(left, right),
                };
                Operand::new(result, Repr::Float)
            }
            BinaryOp::Less
            | BinaryOp::LessEqual
            | BinaryOp::Greater
            | BinaryOp::GreaterEqual
            | BinaryOp::LooseEqual
            | BinaryOp::StrictEqual
            | BinaryOp::LooseNotEqual
            | BinaryOp::StrictNotEqual => {
                let result = if left.repr == Repr::Int32 && right.repr == Repr::Int32 {
                    let condition = match op {
                        BinaryOp::Less => IntCC::SignedLessThan,
                        BinaryOp::LessEqual => IntCC::SignedLessThanOrEqual,
                        BinaryOp::Greater => IntCC::SignedGreaterThan,
                        BinaryOp::GreaterEqual => IntCC::SignedGreaterThanOrEqual,
                        BinaryOp::LooseEqual | BinaryOp::StrictEqual => IntCC::Equal,
                        _ => IntCC::NotEqual,
                    };
                    self.b.ins().icmp(condition, left.value, right.value)
                } else {
                    let condition = match op {
                        BinaryOp::Less => FloatCC::LessThan,
                        BinaryOp::LessEqual => FloatCC::LessThanOrEqual,
                        BinaryOp::Greater => FloatCC::GreaterThan,
                        BinaryOp::GreaterEqual => FloatCC::GreaterThanOrEqual,
                        BinaryOp::LooseEqual | BinaryOp::StrictEqual => FloatCC::Equal,
                        _ => FloatCC::NotEqual,
                    };
                    let left = self.convert(left, Repr::Float);
                    let right = self.convert(right, Repr::Float);
                    self.b.ins().fcmp(condition, left, right)
                };
                Operand::new(result, Repr::Bool)
            }
            BinaryOp::BitwiseAnd
            | BinaryOp::BitwiseOr
            | BinaryOp::BitwiseXor
            | BinaryOp::ShiftLeft
            | BinaryOp::ShiftRight
            | BinaryOp::ShiftRightUnsigned => {
                let left = self.convert(left, Repr::Int32);
                let right = self.convert(right, Repr::Int32);
                let result = match op {
                    BinaryOp::BitwiseAnd => self.b.ins().band(left, right),
                    BinaryOp::BitwiseOr => self.b.ins().bor(left, right),
                    BinaryOp::BitwiseXor => self.b.ins().bxor(left, right),
                    BinaryOp::ShiftLeft => self.b.ins().ishl(left, right),
                    BinaryOp::ShiftRight => self.b.ins().sshr(left, right),
                    _ => {
                        let shifted = self.b.ins().ushr(left, right);
                        let float = self.b.ins().fcvt_from_uint(types::F64, shifted);
                        return Some(Operand::new(float, Repr::Float));
                    }
                };
                Operand::new(result, Repr::Int32)
            }
            _ => return None,
        })
    }

    pub(super) fn typed_math(
        &mut self,
        function: MathFunction,
        args: &[Operand],
    ) -> Option<Operand> {
        let inline = matches!(
            (function, args.len()),
            (
                MathFunction::Floor | MathFunction::Ceil | MathFunction::Sqrt,
                1
            ) | (MathFunction::Min | MathFunction::Max, 1..)
        );
        if !inline {
            return None;
        }
        let floats: Vec<_> = args
            .iter()
            .map(|&arg| self.convert(arg, Repr::Float))
            .collect();
        let result = match function {
            MathFunction::Floor => self.b.ins().floor(floats[0]),
            MathFunction::Ceil => self.b.ins().ceil(floats[0]),
            MathFunction::Sqrt => self.b.ins().sqrt(floats[0]),
            MathFunction::Min => floats[1..]
                .iter()
                .fold(floats[0], |left, &right| self.b.ins().fmin(left, right)),
            _ => floats[1..]
                .iter()
                .fold(floats[0], |left, &right| self.b.ins().fmax(left, right)),
        };
        Some(Operand::new(result, Repr::Float))
    }

    pub(super) fn condition(&mut self, operand: Operand, numeric: bool) -> ClifValue {
        match operand.repr {
            Repr::Boxed if numeric => {
                let float = self.convert(operand, Repr::Float);
                self.convert(Operand::new(float, Repr::Float), Repr::Bool)
            }
            Repr::Boxed => self.truthy(operand.value),
            _ => self.convert(operand, Repr::Bool),
        }
    }
}
