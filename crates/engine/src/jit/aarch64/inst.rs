use cranelift_codegen::ir::Signature;

use super::encode::{
    Alu, Cond, Extend, Float, FloatOp, FloatUnary, Index, Logical, Mem, Shift, VectorOp, Width,
};
use crate::jit::alloc::RegClass;
use crate::jit::effects::{Effects, HEAP_READ};
use crate::jit::semantic::{Contract, Operands, OperandsMut, Ownership, Value, low_level};

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum Immediate {
    Add(u32, bool),
    Sub(u32, bool),
    Logical(Logical, u64),
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum CompareOperand {
    Reg(Value),
    Subs(u32, bool),
    Adds(u32, bool),
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum Mask {
    Reg(Value),
    Immediate(u64),
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum Condition {
    Int {
        cond: Cond,
        width: Width,
        extend: Option<Extend>,
        left: Value,
        right: CompareOperand,
    },
    Float {
        ty: Float,
        first: Cond,
        second: Option<Cond>,
        left: Value,
        right: Value,
    },
    Mask {
        width: Width,
        value: Value,
        mask: Mask,
    },
    Zero {
        width: Width,
        value: Value,
        nonzero: bool,
    },
}

impl Condition {
    fn values(&self) -> [Option<Value>; 2] {
        match *self {
            Self::Int { left, right, .. } => [
                Some(left),
                match right {
                    CompareOperand::Reg(value) => Some(value),
                    _ => None,
                },
            ],
            Self::Float { left, right, .. } => [Some(left), Some(right)],
            Self::Mask { value, mask, .. } => [
                Some(value),
                match mask {
                    Mask::Reg(value) => Some(value),
                    _ => None,
                },
            ],
            Self::Zero { value, .. } => [Some(value), None],
        }
    }

    fn values_mut(&mut self) -> [Option<&mut Value>; 2] {
        match self {
            Self::Int { left, right, .. } => [
                Some(left),
                match right {
                    CompareOperand::Reg(value) => Some(value),
                    _ => None,
                },
            ],
            Self::Float { left, right, .. } => [Some(left), Some(right)],
            Self::Mask { value, mask, .. } => [
                Some(value),
                match mask {
                    Mask::Reg(value) => Some(value),
                    _ => None,
                },
            ],
            Self::Zero { value, .. } => [Some(value), None],
        }
    }

    pub(in crate::jit) fn operands(&self) -> impl Iterator<Item = Value> + '_ {
        self.values().into_iter().flatten()
    }

    pub(in crate::jit) fn operands_mut(&mut self) -> impl Iterator<Item = &mut Value> + '_ {
        self.values_mut().into_iter().flatten()
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum Address {
    Scaled {
        base: Value,
        offset: u32,
    },
    Unscaled {
        base: Value,
        offset: i32,
    },
    Indexed {
        base: Value,
        index: Value,
        kind: Index,
        scaled: bool,
    },
}

impl Address {
    fn values(&self) -> [Option<Value>; 2] {
        match *self {
            Self::Scaled { base, .. } | Self::Unscaled { base, .. } => [Some(base), None],
            Self::Indexed { base, index, .. } => [Some(base), Some(index)],
        }
    }

    fn values_mut(&mut self) -> [Option<&mut Value>; 2] {
        match self {
            Self::Scaled { base, .. } | Self::Unscaled { base, .. } => [Some(base), None],
            Self::Indexed { base, index, .. } => [Some(base), Some(index)],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum ShiftAmount {
    Reg(Value),
    Immediate(u32),
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum SelectType {
    Int(Width),
    Float(Float),
}

#[derive(Clone, Copy, Debug)]
pub(in crate::jit) enum Vector {
    Load {
        address: Address,
        bytes: u32,
        signed: bool,
    },
    Store {
        address: Address,
        bytes: u32,
        value: Value,
    },
    Splat {
        value: Value,
    },
    Binary {
        op: VectorOp,
        left: Value,
        right: Value,
    },
    Shift {
        op: Shift,
        value: Value,
        amount: u32,
    },
    MultiplyAdd {
        accumulator: Value,
        left: Value,
        right: Value,
    },
}

impl Vector {
    fn values(&self) -> [Option<Value>; 4] {
        match *self {
            Self::Load { address, .. } => {
                let [a, b] = address.values();
                [a, b, None, None]
            }
            Self::Store { address, value, .. } => {
                let [a, b] = address.values();
                [Some(value), a, b, None]
            }
            Self::Splat { value } | Self::Shift { value, .. } => [Some(value), None, None, None],
            Self::Binary { left, right, .. } => [Some(left), Some(right), None, None],
            Self::MultiplyAdd {
                accumulator,
                left,
                right,
            } => [Some(accumulator), Some(left), Some(right), None],
        }
    }

    fn values_mut(&mut self) -> [Option<&mut Value>; 4] {
        match self {
            Self::Load { address, .. } => {
                let [a, b] = address.values_mut();
                [a, b, None, None]
            }
            Self::Store { address, value, .. } => {
                let [a, b] = address.values_mut();
                [Some(value), a, b, None]
            }
            Self::Splat { value } | Self::Shift { value, .. } => [Some(value), None, None, None],
            Self::Binary { left, right, .. } => [Some(left), Some(right), None, None],
            Self::MultiplyAdd {
                accumulator,
                left,
                right,
            } => [Some(accumulator), Some(left), Some(right), None],
        }
    }
}

#[derive(Clone, Debug)]
pub(in crate::jit) enum Op {
    Constant {
        bits: u64,
        class: RegClass,
    },
    Alu {
        op: Alu,
        width: Width,
        left: Value,
        right: Value,
    },
    AluImmediate {
        width: Width,
        value: Value,
        immediate: Immediate,
    },
    AddShifted {
        sub: bool,
        width: Width,
        left: Value,
        right: Value,
        kind: Index,
        shift: u32,
    },
    Multiply {
        width: Width,
        left: Value,
        right: Value,
    },
    BitNot {
        width: Width,
        value: Value,
    },
    Shift {
        op: Shift,
        width: Width,
        value: Value,
        amount: ShiftAmount,
    },
    Compare(Condition),
    Select {
        condition: Condition,
        yes: Value,
        no: Value,
        ty: SelectType,
    },
    Copy {
        value: Value,
    },
    Bitcast {
        value: Value,
        width: Width,
        to: RegClass,
    },
    Extend {
        value: Value,
        width: Width,
        extend: Extend,
    },
    Load {
        op: Mem,
        address: Address,
    },
    Store {
        op: Mem,
        address: Address,
        value: Option<Value>,
    },
    LoadAcquire {
        address: Value,
        bytes: u32,
    },
    StackAddress {
        slot: low_level::StackSlot,
        offset: i32,
    },
    StackPointer,
    ExtractBits {
        value: Value,
        width: Width,
        shift: u8,
        bits: u8,
    },
    FloatBinary {
        op: FloatOp,
        ty: Float,
        left: Value,
        right: Value,
    },
    FloatUnary {
        op: FloatUnary,
        ty: Float,
        value: Value,
    },
    IntToFloat {
        signed: bool,
        width: Width,
        ty: Float,
        value: Value,
    },
    FloatToInt {
        signed: bool,
        width: Width,
        ty: Float,
        value: Value,
    },
    FloatConvert {
        promote: bool,
        value: Value,
    },
    Call {
        target: low_level::CallTarget,
        signature: Signature,
        args: Vec<Value>,
    },
    Vector(Vector),
}

impl Op {
    pub(in crate::jit) fn contract(&self) -> Contract {
        let effects = match self {
            Self::Load { .. } | Self::Vector(Vector::Load { .. }) => Effects::of(HEAP_READ),
            Self::Store { .. }
            | Self::Vector(Vector::Store { .. })
            | Self::Call { .. }
            | Self::LoadAcquire { .. } => Effects::UNKNOWN,
            _ => Effects::of(0),
        };
        Contract {
            effects,
            result: if matches!(
                self,
                Self::Store { .. } | Self::Vector(Vector::Store { .. })
            ) {
                Ownership::None
            } else {
                Ownership::Unowned
            },
        }
    }

    pub(in crate::jit) fn pure(&self) -> bool {
        !matches!(
            self,
            Self::Call { .. }
                | Self::Load { .. }
                | Self::LoadAcquire { .. }
                | Self::Store { .. }
                | Self::Vector(Vector::Load { .. } | Vector::Store { .. })
        )
    }

    pub(in crate::jit) fn operands(&self) -> Operands<'_> {
        let (fixed, rest): ([Option<Value>; 4], &[Value]) = match self {
            Self::Constant { .. } | Self::StackAddress { .. } | Self::StackPointer => {
                ([None; 4], &[])
            }
            Self::Alu { left, right, .. }
            | Self::AddShifted { left, right, .. }
            | Self::Multiply { left, right, .. }
            | Self::FloatBinary { left, right, .. } => {
                ([Some(*left), Some(*right), None, None], &[])
            }
            Self::AluImmediate { value, .. }
            | Self::BitNot { value, .. }
            | Self::Copy { value }
            | Self::Bitcast { value, .. }
            | Self::Extend { value, .. }
            | Self::ExtractBits { value, .. }
            | Self::FloatUnary { value, .. }
            | Self::IntToFloat { value, .. }
            | Self::FloatToInt { value, .. }
            | Self::FloatConvert { value, .. }
            | Self::LoadAcquire { address: value, .. } => ([Some(*value), None, None, None], &[]),
            Self::Shift { value, amount, .. } => (
                [
                    Some(*value),
                    match amount {
                        ShiftAmount::Reg(value) => Some(*value),
                        _ => None,
                    },
                    None,
                    None,
                ],
                &[],
            ),
            Self::Compare(condition) => {
                let [a, b] = condition.values();
                ([a, b, None, None], &[])
            }
            Self::Select {
                condition, yes, no, ..
            } => {
                let [a, b] = condition.values();
                ([a, b, Some(*yes), Some(*no)], &[])
            }
            Self::Load { address, .. } => {
                let [a, b] = address.values();
                ([a, b, None, None], &[])
            }
            Self::Store { address, value, .. } => {
                let [a, b] = address.values();
                ([*value, a, b, None], &[])
            }
            Self::Vector(vector) => (vector.values(), &[]),
            Self::Call { target, args, .. } => (
                [
                    match target {
                        low_level::CallTarget::Indirect(value) => Some(*value),
                        _ => None,
                    },
                    None,
                    None,
                    None,
                ],
                args,
            ),
        };
        Operands::new(fixed, rest)
    }

    pub(in crate::jit) fn operands_mut(&mut self) -> OperandsMut<'_> {
        let (fixed, rest): ([Option<&mut Value>; 4], &mut [Value]) = match self {
            Self::Constant { .. } | Self::StackAddress { .. } | Self::StackPointer => {
                ([None, None, None, None], &mut [])
            }
            Self::Alu { left, right, .. }
            | Self::AddShifted { left, right, .. }
            | Self::Multiply { left, right, .. }
            | Self::FloatBinary { left, right, .. } => {
                ([Some(left), Some(right), None, None], &mut [])
            }
            Self::AluImmediate { value, .. }
            | Self::BitNot { value, .. }
            | Self::Copy { value }
            | Self::Bitcast { value, .. }
            | Self::Extend { value, .. }
            | Self::ExtractBits { value, .. }
            | Self::FloatUnary { value, .. }
            | Self::IntToFloat { value, .. }
            | Self::FloatToInt { value, .. }
            | Self::FloatConvert { value, .. }
            | Self::LoadAcquire { address: value, .. } => {
                ([Some(value), None, None, None], &mut [])
            }
            Self::Shift { value, amount, .. } => (
                [
                    Some(value),
                    match amount {
                        ShiftAmount::Reg(value) => Some(value),
                        _ => None,
                    },
                    None,
                    None,
                ],
                &mut [],
            ),
            Self::Compare(condition) => {
                let [a, b] = condition.values_mut();
                ([a, b, None, None], &mut [])
            }
            Self::Select {
                condition, yes, no, ..
            } => {
                let [a, b] = condition.values_mut();
                ([a, b, Some(yes), Some(no)], &mut [])
            }
            Self::Load { address, .. } => {
                let [a, b] = address.values_mut();
                ([a, b, None, None], &mut [])
            }
            Self::Store { address, value, .. } => {
                let [a, b] = address.values_mut();
                ([value.as_mut(), a, b, None], &mut [])
            }
            Self::Vector(vector) => (vector.values_mut(), &mut []),
            Self::Call { target, args, .. } => (
                [
                    match target {
                        low_level::CallTarget::Indirect(value) => Some(value),
                        _ => None,
                    },
                    None,
                    None,
                    None,
                ],
                args,
            ),
        };
        OperandsMut::new(fixed, rest)
    }
}
