use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{BlockArg, InstBuilder, MemFlags, Value as ClifValue, types};
use cranelift_module::Module;
use oxc_syntax::operator::{AssignmentOperator, BinaryOperator};

use crate::jit::helpers::Helper;
use crate::value::Value;
use crate::value::{TAG_FIRST, TAG_SHIFT, TAG_SPAN};

use super::FunctionCompiler;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BinaryOp {
    Instanceof,
    In,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
    ShiftRightUnsigned,

    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    LooseEqual,
    LooseNotEqual,
    StrictEqual,
    StrictNotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UnaryOp {
    Typeof,
    Plus,
    Negation,
    Not,
    BitwiseNot,
}

impl BinaryOp {
    fn helper(self) -> Helper {
        match self {
            Self::Instanceof => Helper::Instanceof,
            Self::In => Helper::In,
            Self::BitwiseAnd => Helper::BitwiseAnd,
            Self::BitwiseOr => Helper::BitwiseOr,
            Self::BitwiseXor => Helper::BitwiseXor,
            Self::ShiftLeft => Helper::ShiftLeft,
            Self::ShiftRight => Helper::ShiftRight,
            Self::ShiftRightUnsigned => Helper::ShiftRightUnsigned,
            Self::Add => Helper::Add,
            Self::Sub => Helper::Sub,
            Self::Mul => Helper::Mul,
            Self::Div => Helper::Div,
            Self::Rem => Helper::Rem,
            Self::Pow => Helper::Pow,
            Self::LooseEqual => Helper::LooseEqual,
            Self::LooseNotEqual => Helper::LooseNotEqual,
            Self::StrictEqual => Helper::StrictEqual,
            Self::StrictNotEqual => Helper::StrictNotEqual,
            Self::Less => Helper::Less,
            Self::LessEqual => Helper::LessEqual,
            Self::Greater => Helper::Greater,
            Self::GreaterEqual => Helper::GreaterEqual,
        }
    }
}

impl UnaryOp {
    fn helper(self) -> Helper {
        match self {
            Self::Typeof => Helper::Typeof,
            Self::Plus => Helper::UnaryPlus,
            Self::Negation => Helper::UnaryNegation,
            Self::Not => Helper::LogicalNot,
            Self::BitwiseNot => Helper::BitwiseNot,
        }
    }
}

pub(super) fn binary_operator(operator: BinaryOperator) -> BinaryOp {
    match operator {
        BinaryOperator::Instanceof => BinaryOp::Instanceof,
        BinaryOperator::In => BinaryOp::In,
        BinaryOperator::Equality => BinaryOp::LooseEqual,
        BinaryOperator::Inequality => BinaryOp::LooseNotEqual,
        BinaryOperator::StrictEquality => BinaryOp::StrictEqual,
        BinaryOperator::StrictInequality => BinaryOp::StrictNotEqual,
        BinaryOperator::LessThan => BinaryOp::Less,
        BinaryOperator::LessEqualThan => BinaryOp::LessEqual,
        BinaryOperator::GreaterThan => BinaryOp::Greater,
        BinaryOperator::GreaterEqualThan => BinaryOp::GreaterEqual,
        BinaryOperator::Addition => BinaryOp::Add,
        BinaryOperator::Subtraction => BinaryOp::Sub,
        BinaryOperator::Multiplication => BinaryOp::Mul,
        BinaryOperator::Division => BinaryOp::Div,
        BinaryOperator::Remainder => BinaryOp::Rem,
        BinaryOperator::Exponential => BinaryOp::Pow,
        BinaryOperator::BitwiseAnd => BinaryOp::BitwiseAnd,
        BinaryOperator::BitwiseOR => BinaryOp::BitwiseOr,
        BinaryOperator::BitwiseXOR => BinaryOp::BitwiseXor,
        BinaryOperator::ShiftLeft => BinaryOp::ShiftLeft,
        BinaryOperator::ShiftRight => BinaryOp::ShiftRight,
        BinaryOperator::ShiftRightZeroFill => BinaryOp::ShiftRightUnsigned,
    }
}

pub(super) fn compound_operator(operator: AssignmentOperator) -> Option<BinaryOp> {
    match operator {
        AssignmentOperator::Addition => Some(BinaryOp::Add),
        AssignmentOperator::Subtraction => Some(BinaryOp::Sub),
        AssignmentOperator::Multiplication => Some(BinaryOp::Mul),
        AssignmentOperator::Division => Some(BinaryOp::Div),
        AssignmentOperator::Remainder => Some(BinaryOp::Rem),
        AssignmentOperator::Exponential => Some(BinaryOp::Pow),
        AssignmentOperator::BitwiseAnd => Some(BinaryOp::BitwiseAnd),
        AssignmentOperator::BitwiseOR => Some(BinaryOp::BitwiseOr),
        AssignmentOperator::BitwiseXOR => Some(BinaryOp::BitwiseXor),
        AssignmentOperator::ShiftLeft => Some(BinaryOp::ShiftLeft),
        AssignmentOperator::ShiftRight => Some(BinaryOp::ShiftRight),
        AssignmentOperator::ShiftRightZeroFill => Some(BinaryOp::ShiftRightUnsigned),
        _ => None,
    }
}

impl FunctionCompiler<'_, '_> {
    pub(super) fn constant(&mut self, value: Value) -> ClifValue {
        self.b.ins().iconst(types::I64, value.bits() as i64)
    }

    pub(super) fn is_number(&mut self, value: ClifValue) -> ClifValue {
        let top = self.b.ins().ushr_imm(value, TAG_SHIFT as i64);
        let offset = self.b.ins().iadd_imm(top, -(TAG_FIRST as i64));
        self.b
            .ins()
            .icmp_imm(IntCC::UnsignedGreaterThan, offset, TAG_SPAN as i64)
    }

    pub(super) fn truthy(&mut self, value: ClifValue) -> ClifValue {
        let truth = self.call_helper(Helper::Truthy, &[value]);
        self.b
            .ins()
            .icmp_imm(IntCC::Equal, truth, Value::TRUE.bits() as i64)
    }

    pub(super) fn is_nullish(&mut self, value: ClifValue) -> ClifValue {
        let undefined = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, value, Value::UNDEFINED.bits() as i64);
        let null = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, value, Value::NULL.bits() as i64);
        self.b.ins().bor(undefined, null)
    }

    pub(super) fn unary_helper(&mut self, op: UnaryOp, value: ClifValue) -> ClifValue {
        let result = self.call_helper(op.helper(), &[value]);
        self.check_error();
        result
    }

    pub(super) fn binary_helper(
        &mut self,
        op: BinaryOp,
        left: ClifValue,
        right: ClifValue,
    ) -> ClifValue {
        let result = self.call_helper(op.helper(), &[left, right]);
        self.check_error();
        result
    }

    pub(super) fn binary(&mut self, op: BinaryOp, left: ClifValue, right: ClifValue) -> ClifValue {
        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                self.numeric_binary(op, left, right)
            }
            BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual => {
                self.numeric_comparison(op, left, right)
            }
            _ => self.binary_helper(op, left, right),
        }
    }

    pub(super) fn call_helper(&mut self, helper: Helper, args: &[ClifValue]) -> ClifValue {
        let function = self.helpers[helper as usize];
        let callee = self.module.declare_func_in_func(function, self.b.func);
        let mut call_args = Vec::with_capacity(args.len() + 1);
        call_args.push(self.ctx);
        call_args.extend_from_slice(args);
        let call = self.b.ins().call(callee, &call_args);
        self.b.inst_results(call)[0]
    }

    fn numeric_fast_path(
        &mut self,
        left: ClifValue,
        right: ClifValue,
        fast: impl FnOnce(&mut Self, ClifValue, ClifValue) -> ClifValue,
        slow: impl FnOnce(&mut Self) -> ClifValue,
    ) -> ClifValue {
        let left_number = self.is_number(left);
        let right_number = self.is_number(right);
        let both = self.b.ins().band(left_number, right_number);
        let fast_block = self.b.create_block();
        let slow_block = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.brif(both, fast_block, &[], slow_block, &[]);

        self.switch(fast_block);
        let fast_result = fast(self, left, right);
        self.jump(done, &[BlockArg::Value(fast_result)]);

        self.switch(slow_block);
        let slow_result = slow(self);
        self.jump(done, &[BlockArg::Value(slow_result)]);

        self.switch(done);
        self.b.block_params(done)[0]
    }

    fn numeric_binary(&mut self, op: BinaryOp, left: ClifValue, right: ClifValue) -> ClifValue {
        self.numeric_fast_path(
            left,
            right,
            |compiler, left, right| {
                let left_float = compiler.b.ins().bitcast(types::F64, MemFlags::new(), left);
                let right_float = compiler.b.ins().bitcast(types::F64, MemFlags::new(), right);
                let result = match op {
                    BinaryOp::Add => compiler.b.ins().fadd(left_float, right_float),
                    BinaryOp::Sub => compiler.b.ins().fsub(left_float, right_float),
                    BinaryOp::Mul => compiler.b.ins().fmul(left_float, right_float),
                    BinaryOp::Div => compiler.b.ins().fdiv(left_float, right_float),
                    _ => unreachable!(),
                };
                compiler
                    .b
                    .ins()
                    .bitcast(types::I64, MemFlags::new(), result)
            },
            |compiler| compiler.binary_helper(op, left, right),
        )
    }

    fn numeric_comparison(&mut self, op: BinaryOp, left: ClifValue, right: ClifValue) -> ClifValue {
        self.numeric_fast_path(
            left,
            right,
            |compiler, left, right| {
                let left_float = compiler.b.ins().bitcast(types::F64, MemFlags::new(), left);
                let right_float = compiler.b.ins().bitcast(types::F64, MemFlags::new(), right);
                let condition = match op {
                    BinaryOp::Less => FloatCC::LessThan,
                    BinaryOp::LessEqual => FloatCC::LessThanOrEqual,
                    BinaryOp::Greater => FloatCC::GreaterThan,
                    BinaryOp::GreaterEqual => FloatCC::GreaterThanOrEqual,
                    _ => unreachable!(),
                };
                let condition = compiler.b.ins().fcmp(condition, left_float, right_float);
                let true_value = compiler.constant(Value::TRUE);
                let false_value = compiler.constant(Value::FALSE);
                compiler.b.ins().select(condition, true_value, false_value)
            },
            |compiler| compiler.binary_helper(op, left, right),
        )
    }
}

pub(super) fn to_number(compiler: &mut FunctionCompiler<'_, '_>, value: ClifValue) -> ClifValue {
    let is_number = compiler.is_number(value);
    let fast_block = compiler.b.create_block();
    let slow_block = compiler.b.create_block();
    let done = compiler.b.create_block();
    compiler.b.append_block_param(done, types::I64);
    compiler.brif(is_number, fast_block, &[], slow_block, &[]);

    compiler.switch(fast_block);
    compiler.jump(done, &[BlockArg::Value(value)]);

    compiler.switch(slow_block);
    let number = compiler.call_helper(Helper::UnaryPlus, &[value]);
    compiler.check_error();
    compiler.jump(done, &[BlockArg::Value(number)]);

    compiler.switch(done);
    compiler.b.block_params(done)[0]
}

impl FunctionCompiler<'_, '_> {
    pub(super) fn index_access(
        &mut self,
        object: ClifValue,
        index: ClifValue,
        value: Option<ClifValue>,
    ) -> ClifValue {
        let result = match value {
            Some(value) => self.call_helper(Helper::IndexSet, &[object, index, value]),
            None => self.call_helper(Helper::IndexGet, &[object, index]),
        };
        self.check_error();
        result
    }

    pub(super) fn global_access(&mut self, name: &str, value: Option<ClifValue>) -> ClifValue {
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        let result = match value {
            Some(value) => self.call_helper(Helper::GlobalSet, &[ptr, len, value]),
            None => self.call_helper(Helper::GlobalGet, &[ptr, len]),
        };
        self.check_error();
        result
    }

    pub(super) fn global_get_optional(&mut self, name: &str) -> ClifValue {
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        let value = self.call_helper(Helper::GlobalGetOptional, &[ptr, len]);
        self.check_error();
        value
    }

    pub(super) fn global_initialize(&mut self, name: &str, value: ClifValue) {
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        self.call_helper(Helper::GlobalInitialize, &[ptr, len, value]);
        self.check_error();
    }

    pub(super) fn property_access(
        &mut self,
        object: ClifValue,
        name: &str,
        value: Option<ClifValue>,
    ) -> ClifValue {
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        let result = match value {
            Some(value) => self.call_helper(Helper::PropertySet, &[object, ptr, len, value]),
            None => self.call_helper(Helper::PropertyGet, &[object, ptr, len]),
        };
        self.check_error();
        result
    }

    pub(super) fn define_hidden_property(
        &mut self,
        object: ClifValue,
        name: &str,
        value: ClifValue,
    ) {
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        self.call_helper(Helper::PropertyDefineHidden, &[object, ptr, len, value]);
        self.check_error();
    }

    pub(super) fn check_error(&mut self) {
        let dummy = self.constant(Value::UNDEFINED);
        let pending = self.call_helper(Helper::ErrorPending, &[dummy]);
        let condition = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, pending, Value::TRUE.bits() as i64);
        let done = self.b.create_block();
        self.brif(
            condition,
            self.return_block,
            &[BlockArg::Value(dummy)],
            done,
            &[],
        );
        self.switch(done);
        let pending = self.call_helper(Helper::ExceptionPending, &[dummy]);
        let condition = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, pending, Value::TRUE.bits() as i64);
        let done = self.b.create_block();
        if let Some(handler) = self.handlers.last().copied() {
            self.brif(condition, handler, &[], done, &[]);
        } else {
            self.brif(
                condition,
                self.return_block,
                &[BlockArg::Value(dummy)],
                done,
                &[],
            );
        }
        self.switch(done);
    }
}
