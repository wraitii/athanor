use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{BlockArg, InstBuilder, MemFlags, Value as ClifValue, types};
use cranelift_module::Module;
use oxc_syntax::operator::{AssignmentOperator, BinaryOperator};

use crate::jit::helpers::Helper;
use crate::jit::numeric::{IndexedRead, IndexedReadContract, Kind};
use crate::value::Value;
use crate::value::{TAG_FIRST, TAG_SHIFT, TAG_SPAN, TAG_STRING};

use super::{FunctionCompiler, ir_helpers};

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
        let result = self.b.ins().iconst(types::I64, value.bits() as i64);
        self.annotations
            .constants
            .insert(result, Kind::of_constant(value));
        result
    }

    pub(super) fn is_number(&mut self, value: ClifValue) -> ClifValue {
        let top = self.b.ins().ushr_imm(value, TAG_SHIFT as i64);
        let offset = self.b.ins().iadd_imm(top, -(TAG_FIRST as i64));
        let condition = self
            .b
            .ins()
            .icmp_imm(IntCC::UnsignedGreaterThan, offset, TAG_SPAN as i64);
        let inst = self.b.func.dfg.value_def(condition).unwrap_inst();
        self.annotations.guards.insert(inst, value);
        condition
    }

    pub(super) fn truthy(&mut self, value: ClifValue) -> ClifValue {
        let number = self.is_number(value);
        let number_block = self.b.create_block();
        let other_block = self.b.create_block();
        let string_block = self.b.create_block();
        let primitive_block = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I8);
        self.brif(number, number_block, &[], other_block, &[]);

        self.switch(number_block);
        let float = self.b.ins().bitcast(types::F64, MemFlags::new(), value);
        let zero = self.b.ins().f64const(0.0);
        let negative = self.b.ins().fcmp(FloatCC::LessThan, float, zero);
        let positive = self.b.ins().fcmp(FloatCC::GreaterThan, float, zero);
        let truth = self.b.ins().bor(negative, positive);
        self.jump(done, &[BlockArg::Value(truth)]);

        self.switch(other_block);
        let tag = self.b.ins().ushr_imm(value, TAG_SHIFT as i64);
        let is_string = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, tag, (TAG_STRING >> TAG_SHIFT) as i64);
        self.brif(is_string, string_block, &[], primitive_block, &[]);

        self.switch(string_block);
        let truth = self.call_helper(Helper::Truthy, &[value]);
        let truth = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, truth, Value::TRUE.bits() as i64);
        self.jump(done, &[BlockArg::Value(truth)]);

        self.switch(primitive_block);
        let undefined = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, value, Value::UNDEFINED.bits() as i64);
        let null = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, value, Value::NULL.bits() as i64);
        let false_value = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, value, Value::FALSE.bits() as i64);
        let falsy = self.b.ins().bor(undefined, null);
        let falsy = self.b.ins().bor(falsy, false_value);
        let truth = self.b.ins().icmp_imm(IntCC::Equal, falsy, 0);
        self.jump(done, &[BlockArg::Value(truth)]);

        self.switch(done);
        self.b.block_params(done)[0]
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
            BinaryOp::BitwiseAnd
            | BinaryOp::BitwiseOr
            | BinaryOp::BitwiseXor
            | BinaryOp::ShiftLeft
            | BinaryOp::ShiftRight
            | BinaryOp::ShiftRightUnsigned => self.integer_binary(op, left, right),
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
        self.annotations.helpers.insert(call, helper);
        self.b.inst_results(call)[0]
    }

    fn integer_binary(&mut self, op: BinaryOp, left: ClifValue, right: ClifValue) -> ClifValue {
        let (left_int, left_guard) = ir_helpers::unbox_int32(self.b, left);
        let (right_int, right_guard) = ir_helpers::unbox_int32(self.b, right);
        for (guard, value) in [(left_guard, left), (right_guard, right)] {
            let inst = self.b.func.dfg.value_def(guard).unwrap_inst();
            self.annotations.int32_guards.insert(inst, value);
        }
        let both = self.b.ins().band(left_guard, right_guard);
        let fast = self.b.create_block();
        let slow = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.brif(both, fast, &[], slow, &[]);

        self.switch(fast);
        let integer = match op {
            BinaryOp::BitwiseAnd => self.b.ins().band(left_int, right_int),
            BinaryOp::BitwiseOr => self.b.ins().bor(left_int, right_int),
            BinaryOp::BitwiseXor => self.b.ins().bxor(left_int, right_int),
            BinaryOp::ShiftLeft => self.b.ins().ishl(left_int, right_int),
            BinaryOp::ShiftRight => self.b.ins().sshr(left_int, right_int),
            BinaryOp::ShiftRightUnsigned => self.b.ins().ushr(left_int, right_int),
            _ => unreachable!(),
        };
        let (result, kind) = if matches!(op, BinaryOp::ShiftRightUnsigned) {
            let float = self.b.ins().fcvt_from_uint(types::F64, integer);
            (
                self.b.ins().bitcast(types::I64, MemFlags::new(), float),
                Kind::Number,
            )
        } else {
            (ir_helpers::box_int32(self.b, integer), Kind::Int32)
        };
        self.annotations.results.insert(result, kind);
        self.jump(done, &[BlockArg::Value(result)]);

        self.switch(slow);
        let result = self.binary_helper(op, left, right);
        self.jump(done, &[BlockArg::Value(result)]);

        self.switch(done);
        self.b.block_params(done)[0]
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
                let unordered = compiler.b.ins().fcmp(FloatCC::Unordered, result, result);
                let bits = compiler
                    .b
                    .ins()
                    .bitcast(types::I64, MemFlags::new(), result);
                let nan = compiler.constant(Value::number(f64::NAN));
                let boxed = compiler.b.ins().select(unordered, nan, bits);
                compiler.annotations.results.insert(boxed, Kind::Number);
                boxed
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
        if value.is_none() {
            self.annotations.indexed_reads.push(IndexedRead {
                call: self.b.func.dfg.value_def(result).unwrap_inst(),
                receiver: object,
                key: index,
                result,
                success: self.b.current_block().unwrap(),
                source: (self.source_span.start, self.source_span.end),
                contract: IndexedReadContract::RuntimeValue,
            });
        }
        result
    }

    pub(super) fn global_access(&mut self, name: &str, value: Option<ClifValue>) -> ClifValue {
        let annotation_name = name.to_owned();
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        let result = match value {
            Some(value) => self.call_helper(Helper::GlobalSet, &[ptr, len, value]),
            None => self.call_helper(Helper::GlobalGet, &[ptr, len]),
        };
        if value.is_none() {
            let call = self.b.func.dfg.value_def(result).unwrap_inst();
            self.annotations.names.insert(call, annotation_name);
        }
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
        let annotation_name = name.to_owned();
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        let result = match value {
            Some(value) => self.call_helper(Helper::PropertySet, &[object, ptr, len, value]),
            None => self.call_helper(Helper::PropertyGet, &[object, ptr, len]),
        };
        if value.is_none() {
            let call = self.b.func.dfg.value_def(result).unwrap_inst();
            self.annotations.names.insert(call, annotation_name);
        }
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
        // Execution reaches a helper with no pending exception; catch entry clears it.
        let helper_call = self
            .b
            .current_block()
            .and_then(|block| self.b.func.layout.last_inst(block))
            .and_then(|inst| {
                self.annotations
                    .helpers
                    .get(&inst)
                    .map(|&helper| (inst, helper))
            });
        let can_throw = helper_call.is_none_or(|(_, helper)| helper.can_throw());
        let numeric_result = helper_call.and_then(|(inst, helper)| {
            helper
                .result_kind_on_success()
                .map(|kind| (self.b.func.dfg.first_result(inst), kind))
        });
        let dummy = self.constant(Value::UNDEFINED);
        let pending = self.call_helper(Helper::ErrorPending, &[dummy]);
        let condition = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, pending, Value::TRUE.bits() as i64);
        let failed = self.b.create_block();
        let done = self.b.create_block();
        self.brif(condition, failed, &[], done, &[]);
        self.switch(failed);
        let ptr = self.b.ins().iconst(self.pointer, self.source_name.0 as i64);
        let len = self.b.ins().iconst(types::I64, self.source_name.1 as i64);
        let start = self
            .b
            .ins()
            .iconst(types::I64, self.source_span.start as i64);
        let end = self.b.ins().iconst(types::I64, self.source_span.end as i64);
        self.call_helper(Helper::ErrorOrigin, &[ptr, len, start, end]);
        self.jump(self.return_block, &[BlockArg::Value(dummy)]);
        self.switch(done);
        if !can_throw {
            if let Some((result, kind)) = numeric_result {
                self.annotations
                    .continuations
                    .entry(done)
                    .or_default()
                    .push((result, kind));
            }
            return;
        }
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
