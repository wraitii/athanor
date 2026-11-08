use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    BlockArg, InstBuilder, MemFlagsData, SigRef, Type, Value as ClifValue, types,
};
use cranelift_frontend::Switch;
use cranelift_module::Module;
use oxc_span::Span;
use oxc_syntax::operator::BinaryOperator;

use crate::error::Error;
use crate::heap::{
    CODE_OFFSET, DATA_OFFSET, HOLE, INLINE_PROPERTY_LENGTH_OFFSET, KIND_OFFSET, LENGTH_OFFSET,
    MathFunction, NO_ELEMENTS, SHAPE_OFFSET, TypedKind, kind,
};
use crate::jit::clif::IndexedRead;
use crate::jit::helpers::{
    self, CACHE_ADDITION_DESTINATION_OFFSET, CACHE_ADDITION_INDEX_OFFSET,
    CACHE_ADDITION_SOURCE_OFFSET, CACHE_BUILTIN_KIND_OFFSET, CACHE_BUILTIN_VALUE_OFFSET,
    CACHE_HOLDER_OFFSET, CACHE_HOLDER_SHAPE_OFFSET, CACHE_INDEX_OFFSET, CACHE_INITIALIZED_OFFSET,
    CACHE_SHAPE_OFFSET, HELPERS, Helper, PropertyCache,
};
use crate::jit::planner::{CallSite, Site};
use crate::jit::semantic::DirectTarget;
use crate::jit::session::CALL_SITE_OFFSET;
use crate::value::Value;
use crate::value::{TAG_FIRST, TAG_OBJECT, TAG_SHIFT, TAG_SPAN, TAG_STRING};

use super::repr::{Operand, Repr};
use super::{FunctionCompiler, ir_helpers};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Element {
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
    F32,
    F64,
}

use super::scope::ADDRESS_MASK;

const TYPED_KINDS: [Element; 8] = [
    Element::U8,
    Element::U16,
    Element::U32,
    Element::I8,
    Element::I16,
    Element::I32,
    Element::F32,
    Element::F64,
];

impl Element {
    fn bytes(self) -> u32 {
        Type::from(self).bytes()
    }

    fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    fn kind(self) -> u32 {
        kind::TYPED
            | match self {
                Self::U8 => 0,
                Self::U16 => 1,
                Self::U32 => 2,
                Self::I8 => kind::ELEMENT_SIGNED,
                Self::I16 => kind::ELEMENT_SIGNED | 1,
                Self::I32 => kind::ELEMENT_SIGNED | 2,
                Self::F32 => kind::ELEMENT_FLOAT | 2,
                Self::F64 => kind::ELEMENT_FLOAT | 3,
            }
    }
}

impl From<TypedKind> for Element {
    fn from(kind: TypedKind) -> Self {
        match kind {
            TypedKind::Uint8 => Self::U8,
            TypedKind::Uint16 => Self::U16,
            TypedKind::Uint32 => Self::U32,
            TypedKind::Int8 => Self::I8,
            TypedKind::Int16 => Self::I16,
            TypedKind::Int32 => Self::I32,
            TypedKind::Float32 => Self::F32,
            TypedKind::Float64 => Self::F64,
        }
    }
}

impl From<Element> for Type {
    fn from(element: Element) -> Self {
        match element {
            Element::U8 | Element::I8 => types::I8,
            Element::U16 | Element::I16 => types::I16,
            Element::U32 | Element::I32 => types::I32,
            Element::F32 => types::F32,
            Element::F64 => types::F64,
        }
    }
}

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

impl FunctionCompiler<'_, '_> {
    pub(super) fn constant(&mut self, value: Value) -> ClifValue {
        self.b.ins().iconst(types::I64, value.bits() as i64)
    }

    pub(super) fn is_number(&mut self, value: ClifValue) -> ClifValue {
        let top = self.b.ins().ushr_imm_s(value, TAG_SHIFT as i64);
        let offset = self.b.ins().iadd_imm_s(top, -(TAG_FIRST as i64));
        let condition =
            self.b
                .ins()
                .icmp_imm_s(IntCC::UnsignedGreaterThan, offset, TAG_SPAN as i64);
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
        let float = self.b.ins().bitcast(types::F64, MemFlagsData::new(), value);
        let zero = self.b.ins().f64const(0.0);
        let negative = self.b.ins().fcmp(FloatCC::LessThan, float, zero);
        let positive = self.b.ins().fcmp(FloatCC::GreaterThan, float, zero);
        let truth = self.b.ins().bor(negative, positive);
        self.jump(done, &[BlockArg::Value(truth)]);

        self.switch(other_block);
        let tag = self.b.ins().ushr_imm_s(value, TAG_SHIFT as i64);
        let is_string =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, tag, (TAG_STRING >> TAG_SHIFT) as i64);
        self.brif(is_string, string_block, &[], primitive_block, &[]);

        self.switch(string_block);
        let truth = self.call_helper(Helper::Truthy, &[value]);
        let truth = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, truth, Value::TRUE.bits() as i64);
        self.jump(done, &[BlockArg::Value(truth)]);

        self.switch(primitive_block);
        let undefined =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, value, Value::UNDEFINED.bits() as i64);
        let null = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, value, Value::NULL.bits() as i64);
        let false_value = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, value, Value::FALSE.bits() as i64);
        let falsy = self.b.ins().bor(undefined, null);
        let falsy = self.b.ins().bor(falsy, false_value);
        let truth = self.b.ins().icmp_imm_s(IntCC::Equal, falsy, 0);
        self.jump(done, &[BlockArg::Value(truth)]);

        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn is_nullish(&mut self, value: ClifValue) -> ClifValue {
        let undefined =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, value, Value::UNDEFINED.bits() as i64);
        let null = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, value, Value::NULL.bits() as i64);
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
            BinaryOp::Less
            | BinaryOp::LessEqual
            | BinaryOp::Greater
            | BinaryOp::GreaterEqual
            | BinaryOp::LooseEqual
            | BinaryOp::LooseNotEqual
            | BinaryOp::StrictEqual
            | BinaryOp::StrictNotEqual => self.numeric_comparison(op, left, right, None),
            _ => self.binary_helper(op, left, right),
        }
    }

    pub(super) fn function_data<T>(&mut self, value: T) -> ClifValue {
        let offset = self.data.push(value);
        self.b.ins().iadd_imm_s(self.data_base, offset)
    }

    pub(super) fn property_cache(
        &mut self,
        site: crate::jit::semantic::Inst,
        name: &str,
    ) -> ClifValue {
        let offset = match self.property_caches.get(&site) {
            Some(&offset) => offset,
            None => {
                let offset = self.data.push(PropertyCache::new(name));
                self.property_caches.insert(site, offset);
                offset
            }
        };
        self.b.ins().iadd_imm_s(self.data_base, offset)
    }

    pub(super) fn call_helper(&mut self, helper: Helper, args: &[ClifValue]) -> ClifValue {
        let target = self.module.target_config();
        let signature = *self.helper_signatures.entry(helper).or_insert_with(|| {
            self.b.import_signature(helpers::signature(
                HELPERS[helper as usize].shape,
                target.default_call_conv,
                self.pointer,
            ))
        });
        let callee = helpers::address(self.b.ins(), self.pointer, self.ctx, helper);
        let mut call_args = Vec::with_capacity(args.len() + 1);
        call_args.push(self.ctx);
        call_args.extend_from_slice(args);
        let call = self.b.ins().call_indirect(signature, callee, &call_args);
        self.annotations.helpers.insert(call, helper);
        self.b.inst_results(call)[0]
    }

    fn integer_binary(&mut self, op: BinaryOp, left: ClifValue, right: ClifValue) -> ClifValue {
        let (left_int, left_guard) = ir_helpers::unbox_int32(self.b, left);
        let (right_int, right_guard) = ir_helpers::unbox_int32(self.b, right);
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
        let result = if matches!(op, BinaryOp::ShiftRightUnsigned) {
            let float = self.b.ins().fcvt_from_uint(types::F64, integer);
            self.b.ins().bitcast(types::I64, MemFlagsData::new(), float)
        } else {
            ir_helpers::box_int32(self.b, integer)
        };
        self.jump(done, &[BlockArg::Value(result)]);

        self.switch(slow);
        let result = self.binary_helper(op, left, right);
        self.jump(done, &[BlockArg::Value(result)]);

        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn native_math(
        &mut self,
        function: MathFunction,
        args: &[ClifValue],
        slow: impl FnOnce(&mut Self) -> ClifValue,
    ) -> ClifValue {
        let inline = matches!(
            (function, args.len()),
            (
                MathFunction::Floor | MathFunction::Ceil | MathFunction::Sqrt,
                1
            ) | (MathFunction::Min | MathFunction::Max, 1..)
        );
        if !inline {
            return slow(self);
        }
        let mut numbers = self.b.ins().iconst(types::I8, 1);
        for &arg in args {
            let number = self.is_number(arg);
            numbers = self.b.ins().band(numbers, number);
        }
        let fast_block = self.b.create_block();
        let slow_block = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.brif(numbers, fast_block, &[], slow_block, &[]);

        self.switch(fast_block);
        let floats: Vec<_> = args
            .iter()
            .map(|&arg| self.b.ins().bitcast(types::F64, MemFlagsData::new(), arg))
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
        let unordered = self.b.ins().fcmp(FloatCC::Unordered, result, result);
        let bits = self
            .b
            .ins()
            .bitcast(types::I64, MemFlagsData::new(), result);
        let nan = self.constant(Value::number(f64::NAN));
        let boxed = self.b.ins().select(unordered, nan, bits);
        self.jump(done, &[BlockArg::Value(boxed)]);

        self.switch(slow_block);
        let slow_result = slow(self);
        self.jump(done, &[BlockArg::Value(slow_result)]);

        self.switch(done);
        self.b.block_params(done)[0]
    }

    fn numeric_fast_path(
        &mut self,
        left: ClifValue,
        right: ClifValue,
        guards: Option<(ClifValue, ClifValue)>,
        fast: impl FnOnce(&mut Self, ClifValue, ClifValue) -> ClifValue,
        slow: impl FnOnce(&mut Self) -> ClifValue,
    ) -> ClifValue {
        let (left_number, right_number) =
            guards.unwrap_or_else(|| (self.is_number(left), self.is_number(right)));
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
            None,
            |compiler, left, right| {
                let left_float = compiler
                    .b
                    .ins()
                    .bitcast(types::F64, MemFlagsData::new(), left);
                let right_float = compiler
                    .b
                    .ins()
                    .bitcast(types::F64, MemFlagsData::new(), right);
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
                    .bitcast(types::I64, MemFlagsData::new(), result);
                let nan = compiler.constant(Value::number(f64::NAN));
                let boxed = compiler.b.ins().select(unordered, nan, bits);
                boxed
            },
            |compiler| compiler.binary_helper(op, left, right),
        )
    }

    pub(super) fn numeric_comparison(
        &mut self,
        op: BinaryOp,
        left: ClifValue,
        right: ClifValue,
        guards: Option<(ClifValue, ClifValue)>,
    ) -> ClifValue {
        self.numeric_fast_path(
            left,
            right,
            guards,
            |compiler, left, right| {
                let left_float = compiler
                    .b
                    .ins()
                    .bitcast(types::F64, MemFlagsData::new(), left);
                let right_float = compiler
                    .b
                    .ins()
                    .bitcast(types::F64, MemFlagsData::new(), right);
                let condition = match op {
                    BinaryOp::Less => FloatCC::LessThan,
                    BinaryOp::LessEqual => FloatCC::LessThanOrEqual,
                    BinaryOp::Greater => FloatCC::GreaterThan,
                    BinaryOp::GreaterEqual => FloatCC::GreaterThanOrEqual,
                    BinaryOp::LooseEqual | BinaryOp::StrictEqual => FloatCC::Equal,
                    BinaryOp::LooseNotEqual | BinaryOp::StrictNotEqual => FloatCC::NotEqual,
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
        index: Operand,
        value: Option<ClifValue>,
    ) -> ClifValue {
        if let Some(value) = value {
            let index = self.convert(index, Repr::Boxed);
            let result = self.call_helper(Helper::IndexSet, &[object, index, value]);
            self.check_error();
            return result;
        }
        let (helper, index) = if matches!(index.repr, Repr::Float | Repr::Int32) {
            (Helper::IndexGetNumeric, self.convert(index, Repr::Float))
        } else {
            (Helper::IndexGet, index.value)
        };
        let result = self.call_helper(helper, &[object, index]);
        self.check_error();
        if helper == Helper::IndexGet {
            self.annotations.indexed_reads.push(IndexedRead {
                call: self.b.func.dfg.value_def(result).unwrap_inst(),
                receiver: object,
                key: index,
                result,
                success: self.b.current_block().unwrap(),
                source: (self.source_span.start, self.source_span.end),
            });
        }
        result
    }

    // Non-objects and the global object (address zero) see an all-zero slot, so every
    // kind check fails and the length reads as zero.
    pub(super) fn view(&mut self, object: ClifValue) -> ClifValue {
        let tag = self.b.ins().ushr_imm_s(object, TAG_SHIFT as i64);
        let is_object =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, tag, (TAG_OBJECT >> TAG_SHIFT) as i64);
        let address = self.b.ins().band_imm_s(object, ADDRESS_MASK);
        let present = self.b.ins().icmp_imm_s(IntCC::NotEqual, address, 0);
        let valid = self.b.ins().band(is_object, present);
        let empty = self
            .b
            .ins()
            .iconst(self.pointer, NO_ELEMENTS.as_ptr() as i64);
        self.b.ins().select(valid, address, empty)
    }

    pub(super) fn generic_call(
        &mut self,
        callee: ClifValue,
        receiver: ClifValue,
        args: &[ClifValue],
        span: Span,
    ) -> Result<ClifValue, Error> {
        if let Ok(arity) = u16::try_from(args.len())
            && let Some(miss) = self.stubs.miss(arity)
        {
            self.mark_call_site(span, false);
            let entry = self.b.ins().iconst(self.pointer, miss as i64);
            return Ok(self.checked_fixed_call(entry, arity, callee, receiver, args, false));
        }
        let result = self.unchecked_call(callee, receiver, args, span)?;
        self.check_error();
        Ok(result)
    }

    fn unchecked_call(
        &mut self,
        callee: ClifValue,
        receiver: ClifValue,
        args: &[ClifValue],
        span: Span,
    ) -> Result<ClifValue, Error> {
        let argv = self.stack_values(args, span)?;
        let argc = self.b.ins().iconst(types::I64, args.len() as i64);
        Ok(self.call_helper(Helper::Call, &[callee, receiver, argv, argc]))
    }

    /// With `owned`, the caller gets a reference of its own rather than a temporary. A direct
    /// call that reports no failure skips the pending-error checks.
    pub(super) fn direct_call(
        &mut self,
        target: DirectTarget,
        callee: ClifValue,
        receiver: ClifValue,
        args: &[ClifValue],
        span: Span,
        owned: bool,
    ) -> Result<ClifValue, Error> {
        if target.code.as_u32() as usize >= crate::jit::entries::DIRECT_ENTRIES {
            let result = self.generic_call(callee, receiver, args, span)?;
            if owned {
                self.increment(result);
            }
            return Ok(result);
        }
        if args.len() == usize::from(target.arity)
            && let Some(miss) = self.stubs.miss(target.arity)
        {
            return Ok(self.selected_call(target, callee, receiver, args, span, miss, owned));
        }
        let (entry, direct) = self.direct_entry(target, callee);
        let fast = self.b.create_block();
        let slow = self.b.create_block();
        let check = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(check, types::I64);
        self.b.append_block_param(done, types::I64);
        self.b.set_cold_block(slow);
        self.b.set_cold_block(check);
        self.brif(direct, fast, &[], slow, &[]);
        self.switch(fast);
        let (result, status) = self.fixed_call(entry, target.arity, callee, receiver, args);
        let succeeded = self.b.create_block();
        let undefined = self.constant(Value::UNDEFINED);
        self.brif(status, check, &[BlockArg::Value(undefined)], succeeded, &[]);
        self.switch(succeeded);
        let result = self.owned_result(result, owned);
        self.jump(done, &[BlockArg::Value(result)]);
        self.switch(slow);
        let result = self.unchecked_call(callee, receiver, args, span)?;
        self.jump(check, &[BlockArg::Value(result)]);
        self.switch(check);
        let result = self.b.block_params(check)[0];
        self.check_error();
        if owned {
            self.increment(result);
        }
        self.jump(done, &[BlockArg::Value(result)]);
        self.switch(done);
        Ok(self.b.block_params(done)[0])
    }

    /// Falls back to the miss stub rather than an inline generic call.
    fn selected_call(
        &mut self,
        target: DirectTarget,
        callee: ClifValue,
        receiver: ClifValue,
        args: &[ClifValue],
        span: Span,
        miss: usize,
        owned: bool,
    ) -> ClifValue {
        self.mark_call_site(span, true);
        let (entry, direct) = self.direct_entry(target, callee);
        let miss = self.b.ins().iconst(self.pointer, miss as i64);
        let entry = self.b.ins().select(direct, entry, miss);
        self.checked_fixed_call(entry, target.arity, callee, receiver, args, owned)
    }

    fn mark_call_site(&mut self, span: Span, guarded: bool) {
        let site = self.data.pin(CallSite {
            site: Site {
                source: self.source_name.0,
                start: span.start,
                end: span.end,
            },
            guarded,
        });
        let site = self.b.ins().iconst(self.pointer, site as i64);
        self.b
            .ins()
            .store(MemFlagsData::trusted(), site, self.ctx, CALL_SITE_OFFSET);
    }

    fn checked_fixed_call(
        &mut self,
        entry: ClifValue,
        arity: u16,
        callee: ClifValue,
        receiver: ClifValue,
        args: &[ClifValue],
        owned: bool,
    ) -> ClifValue {
        let (result, status) = self.fixed_call(entry, arity, callee, receiver, args);
        let failed = self.b.create_block();
        let succeeded = self.b.create_block();
        self.b.set_cold_block(failed);
        self.brif(status, failed, &[], succeeded, &[]);
        self.switch(failed);
        self.check_error();
        self.jump(succeeded, &[]);
        self.switch(succeeded);
        self.owned_result(result, owned)
    }

    fn fixed_call(
        &mut self,
        entry: ClifValue,
        arity: u16,
        callee: ClifValue,
        receiver: ClifValue,
        args: &[ClifValue],
    ) -> (ClifValue, ClifValue) {
        let arity = usize::from(arity);
        let signature = self.fixed_signature(arity as u16);
        let mut call_args = vec![self.ctx, callee, receiver];
        call_args.extend_from_slice(&args[..args.len().min(arity)]);
        if args.len() < arity {
            let undefined = self.constant(Value::UNDEFINED);
            call_args.resize(arity + 3, undefined);
        }
        let call = self.b.ins().call_indirect(signature, entry, &call_args);
        let [result, status] = self.b.inst_results(call).try_into().unwrap();
        (result, status)
    }

    fn owned_result(&mut self, result: ClifValue, owned: bool) -> ClifValue {
        if owned {
            return result;
        }
        self.own_temporary(result)
    }

    fn direct_entry(&mut self, target: DirectTarget, callee: ClifValue) -> (ClifValue, ClifValue) {
        if let Some(call) = super::stubs::poll(
            self.b,
            self.module,
            self.stubs.compilation_state,
            self.ctx,
            None,
        ) {
            self.annotations.helpers.insert(call, Helper::CompilePoll);
        }
        let slot = self.b.ins().iconst(
            self.pointer,
            self.stubs.direct_entries as i64
                + i64::from(target.code.as_u32()) * crate::jit::entries::DIRECT_ENTRY_SIZE as i64,
        );
        let entry = self
            .b
            .ins()
            .load(self.pointer, MemFlagsData::trusted(), slot, 0);
        let published = self.b.ins().icmp_imm_s(IntCC::NotEqual, entry, 0);
        let expected = self.is_code(callee, target.code);
        (entry, self.b.ins().band(expected, published))
    }

    fn fixed_signature(&mut self, arity: u16) -> SigRef {
        if let Some(&signature) = self.fixed_signatures.get(&arity) {
            return signature;
        }
        let signature = super::stubs::fixed_signature(self.pointer, arity);
        let signature = self.b.import_signature(signature);
        self.fixed_signatures.insert(arity, signature);
        signature
    }

    /// Whether a value is a closure of this code that may be called without `new`.
    pub(super) fn is_code(
        &mut self,
        value: ClifValue,
        code: cranelift_module::FuncId,
    ) -> ClifValue {
        let view = self.view(value);
        let word = self
            .b
            .ins()
            .load(types::I32, MemFlagsData::trusted(), view, CODE_OFFSET);
        self.b
            .ins()
            .icmp_imm_s(IntCC::Equal, word, i64::from(code.as_u32()) + 1)
    }

    pub(super) fn view_kind(&mut self, view: ClifValue) -> ClifValue {
        self.b
            .ins()
            .load(types::I32, MemFlagsData::trusted(), view, KIND_OFFSET)
    }

    pub(super) fn has_kind(&mut self, view: ClifValue, mask: u32) -> ClifValue {
        let kind = self.view_kind(view);
        let masked = self.b.ins().band_imm_s(kind, mask as i64);
        self.b.ins().icmp_imm_s(IntCC::NotEqual, masked, 0)
    }

    pub(super) fn view_length_integer(&mut self, view: ClifValue) -> ClifValue {
        self.b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), view, LENGTH_OFFSET)
    }

    pub(super) fn view_length(&mut self, view: ClifValue) -> ClifValue {
        let length = self.view_length_integer(view);
        self.b.ins().fcvt_from_uint(types::F64, length)
    }

    pub(super) fn length_access(&mut self, object: ClifValue) -> ClifValue {
        let view = self.view(object);
        let indexed = self.has_kind(view, kind::ARRAY | kind::TYPED);
        let inline = self.b.create_block();
        let slow = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.brif(indexed, inline, &[], slow, &[]);
        self.switch(inline);
        let length = self.view_length(view);
        let bits = self
            .b
            .ins()
            .bitcast(types::I64, MemFlagsData::new(), length);
        self.jump(done, &[BlockArg::Value(bits)]);
        self.switch(slow);
        let result = self.property_access(object, "length", None);
        self.jump(done, &[BlockArg::Value(result)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn element_read(
        &mut self,
        object: ClifValue,
        key: Operand,
        retain: bool,
    ) -> ClifValue {
        let view = self.view(object);
        self.viewed_index_access(view, object, key, None, true, retain, true)
    }

    pub(super) fn typed_kind_is(&mut self, view: ClifValue, kind: TypedKind) -> ClifValue {
        let actual = self.view_kind(view);
        let expected = Element::from(kind).kind();
        self.b
            .ins()
            .icmp_imm_s(IntCC::Equal, actual, expected as i64)
    }

    /// Reads an element of a typed array of known kind at an index known to be in bounds.
    pub(super) fn bounded_element_read(
        &mut self,
        view: ClifValue,
        key: Operand,
        kind: TypedKind,
    ) -> Operand {
        let element = Element::from(kind);
        let flags = MemFlagsData::trusted();
        let position = if key.repr == Repr::Int32 {
            self.b.ins().sextend(types::I64, key.value)
        } else {
            let key = self.convert(key, Repr::Float);
            self.b.ins().fcvt_to_sint_sat(types::I64, key)
        };
        let data = self.b.ins().load(self.pointer, flags, view, DATA_OFFSET);
        let offset = self.b.ins().imul_imm_s(position, element.bytes() as i64);
        let address = self.b.ins().iadd(data, offset);
        let raw = self.b.ins().load(element.into(), flags, address, 0);
        match element {
            Element::F64 => Operand::new(raw, Repr::Float),
            Element::F32 => Operand::new(self.b.ins().fpromote(types::F64, raw), Repr::Float),
            Element::U32 => {
                let wide = self.b.ins().uextend(types::I64, raw);
                Operand::new(self.b.ins().fcvt_from_sint(types::F64, wide), Repr::Float)
            }
            Element::U8 | Element::U16 => {
                Operand::new(self.b.ins().uextend(types::I32, raw), Repr::Int32)
            }
            Element::I8 | Element::I16 => {
                Operand::new(self.b.ins().sextend(types::I32, raw), Repr::Int32)
            }
            Element::I32 => Operand::new(raw, Repr::Int32),
        }
    }

    pub(super) fn checked_element_read(
        &mut self,
        view: ClifValue,
        key: Operand,
        kind: TypedKind,
    ) -> Operand {
        let length = self
            .b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), view, LENGTH_OFFSET);
        let inside = if key.repr == Repr::Int32 {
            let position = self.b.ins().sextend(types::I64, key.value);
            self.b.ins().icmp(IntCC::UnsignedLessThan, position, length)
        } else {
            let number = self.convert(key, Repr::Float);
            let position = self.b.ins().fcvt_to_sint_sat(types::I64, number);
            let exact = self.b.ins().fcvt_from_sint(types::F64, position);
            let integral = self.b.ins().fcmp(FloatCC::Equal, exact, number);
            let inside = self.b.ins().icmp(IntCC::UnsignedLessThan, position, length);
            self.b.ins().band(integral, inside)
        };
        let read = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        let undefined = self.constant(Value::UNDEFINED);
        self.brif(inside, read, &[], done, &[undefined.into()]);
        self.switch(read);
        let value = self.bounded_element_read(view, key, kind);
        let value = self.convert(value, Repr::Boxed);
        self.jump(done, &[value.into()]);
        self.switch(done);
        Operand::new(self.b.block_params(done)[0], Repr::Boxed)
    }

    pub(super) fn viewed_index_access(
        &mut self,
        view: ClifValue,
        object: ClifValue,
        key: Operand,
        value: Option<Operand>,
        fallback: bool,
        retain: bool,
        typed: bool,
    ) -> ClifValue {
        let flags = MemFlagsData::trusted();
        let length = self.b.ins().load(types::I64, flags, view, LENGTH_OFFSET);
        let (position, fast) = if key.repr == Repr::Int32 {
            let position = self.b.ins().sextend(types::I64, key.value);
            let inside = self.b.ins().icmp(IntCC::UnsignedLessThan, position, length);
            (position, inside)
        } else {
            let key = self.convert(key, Repr::Float);
            let position = self.b.ins().fcvt_to_sint_sat(types::I64, key);
            let exact = self.b.ins().fcvt_from_sint(types::F64, position);
            let integral = self.b.ins().fcmp(FloatCC::Equal, exact, key);
            let inside = self.b.ins().icmp(IntCC::UnsignedLessThan, position, length);
            (position, self.b.ins().band(integral, inside))
        };
        let inline = self.b.create_block();
        let slow = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.brif(fast, inline, &[], slow, &[]);

        self.switch(inline);
        let data = self.b.ins().load(self.pointer, flags, view, DATA_OFFSET);
        let kind = self.view_kind(view);
        let mut switch = Switch::new();
        let kinds: Vec<_> = TYPED_KINDS
            .iter()
            .filter(|_| value.is_some())
            .map(|&element| (element, self.b.create_block()))
            .collect();
        for (element, block) in &kinds {
            switch.set_entry(element.kind() as u128, *block);
        }
        let elements = value.is_none().then(|| {
            let packed = self.b.create_block();
            let holey = self.b.create_block();
            switch.set_entry(kind::ARRAY as u128, packed);
            switch.set_entry((kind::ARRAY | kind::HOLEY) as u128, holey);
            let offset = self.b.ins().ishl_imm_s(position, 3);
            let address = self.b.ins().iadd(data, offset);
            (packed, holey, address)
        });
        let typed_block = (value.is_none() && typed).then(|| self.b.create_block());
        switch.emit(self.b, kind, typed_block.unwrap_or(slow));
        if let Some(typed_block) = typed_block {
            self.switch(typed_block);
            let typed = self.has_kind(view, kind::TYPED);
            let read = self.b.create_block();
            self.brif(typed, read, &[], slow, &[]);
            self.switch(read);
            let bits = self.call_helper(Helper::TypedIndexGet, &[object, position]);
            self.jump(done, &[BlockArg::Value(bits)]);
        }
        if let Some((packed, holey, address)) = elements {
            let element = self.b.create_block();
            self.b.append_block_param(element, types::I64);
            self.switch(packed);
            let bits = self.b.ins().load(types::I64, flags, address, 0);
            self.jump(element, &[BlockArg::Value(bits)]);
            self.switch(holey);
            let bits = self.b.ins().load(types::I64, flags, address, 0);
            let hole = self
                .b
                .ins()
                .icmp_imm_s(IntCC::Equal, bits, HOLE.bits() as i64);
            let undefined = self.constant(Value::UNDEFINED);
            let bits = self.b.ins().select(hole, undefined, bits);
            self.jump(element, &[BlockArg::Value(bits)]);
            self.switch(element);
            let bits = self.b.block_params(element)[0];
            if retain {
                let heap = self.is_heap(bits);
                // Matches the IndexGet helper, which hands heap elements out as temporaries.
                let retain = self.b.create_block();
                self.brif(heap, retain, &[], done, &[BlockArg::Value(bits)]);
                self.switch(retain);
                let retained = self.retain_temporary(bits);
                self.jump(done, &[BlockArg::Value(retained)]);
            } else {
                self.jump(done, &[BlockArg::Value(bits)]);
            }
        }
        for (element, block) in kinds {
            self.switch(block);
            let offset = self.b.ins().imul_imm_s(position, element.bytes() as i64);
            let address = self.b.ins().iadd(data, offset);
            let value = value.unwrap();
            let (stored, valid) = if element.is_float() {
                let valid = match value.repr {
                    Repr::Boxed => self.is_number(value.value),
                    _ => self.b.ins().iconst(types::I8, 1),
                };
                let number = self.convert(value, Repr::Float);
                let stored = if element == Element::F32 {
                    self.b.ins().fdemote(types::F32, number)
                } else {
                    number
                };
                (stored, valid)
            } else if value.repr == Repr::Int32 {
                let stored = if Type::from(element) == types::I32 {
                    value.value
                } else {
                    self.b.ins().ireduce(element.into(), value.value)
                };
                (stored, self.b.ins().iconst(types::I8, 1))
            } else {
                let number = self.convert(value, Repr::Float);
                let integer = self.b.ins().fcvt_to_sint_sat(types::I64, number);
                let exact = self.b.ins().fcvt_from_sint(types::F64, integer);
                let integral = self.b.ins().fcmp(FloatCC::Equal, exact, number);
                let upper = self.b.ins().f64const(9223372036854775808.0);
                let lower = self.b.ins().f64const(-9223372036854775808.0);
                let below = self.b.ins().fcmp(FloatCC::LessThan, number, upper);
                let above = self
                    .b
                    .ins()
                    .fcmp(FloatCC::GreaterThanOrEqual, number, lower);
                let in_range = self.b.ins().band(below, above);
                let valid = self.b.ins().band(integral, in_range);
                (self.b.ins().ireduce(element.into(), integer), valid)
            };
            let store = self.b.create_block();
            self.brif(valid, store, &[], slow, &[]);
            self.switch(store);
            self.b.ins().store(flags, stored, address, 0);
            let undefined = self.constant(Value::UNDEFINED);
            self.jump(done, &[BlockArg::Value(undefined)]);
        }

        self.switch(slow);
        let result = if fallback {
            let value = value.map(|value| self.convert(value, Repr::Boxed));
            self.index_access(object, key, value)
        } else {
            self.constant(Value::UNDEFINED)
        };
        self.jump(done, &[BlockArg::Value(result)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn global_access(&mut self, name: &str, value: Option<ClifValue>) -> ClifValue {
        let Some(value) = value else {
            return self.cached_global_get(name, false, true);
        };
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        let result = self.call_helper(Helper::GlobalSet, &[ptr, len, value]);
        self.check_error();
        result
    }

    pub(super) fn global_get_optional(&mut self, name: &str, retain: bool) -> ClifValue {
        self.cached_global_get(name, true, retain)
    }

    pub(super) fn cached_global_get(
        &mut self,
        name: &str,
        optional: bool,
        retain: bool,
    ) -> ClifValue {
        use crate::jit::helpers::GlobalCache;
        use crate::jit::session::{GLOBAL_IDENTITY_OFFSET, GLOBAL_REVISION_OFFSET};

        let address = self.function_data(GlobalCache::new(name));
        let flags = MemFlagsData::trusted();
        let identity = self
            .b
            .ins()
            .load(self.pointer, flags, self.ctx, GLOBAL_IDENTITY_OFFSET);
        let expected_identity = self.b.ins().load(
            self.pointer,
            flags,
            address,
            std::mem::offset_of!(GlobalCache, identity) as i32,
        );
        let same = self.b.ins().icmp(IntCC::Equal, identity, expected_identity);
        let revision_ptr = self
            .b
            .ins()
            .load(self.pointer, flags, self.ctx, GLOBAL_REVISION_OFFSET);
        let revision = self.b.ins().load(types::I64, flags, revision_ptr, 0);
        let expected_revision = self.b.ins().load(
            types::I64,
            flags,
            address,
            std::mem::offset_of!(GlobalCache, revision) as i32,
        );
        let valid = self.b.ins().icmp(IntCC::Equal, revision, expected_revision);
        let valid = self.b.ins().band(same, valid);
        let slot = self.b.ins().load(
            self.pointer,
            flags,
            address,
            std::mem::offset_of!(GlobalCache, slot) as i32,
        );
        let present = self.b.ins().icmp_imm_s(IntCC::NotEqual, slot, 0);
        let valid = self.b.ins().band(valid, present);
        let hit = self.b.create_block();
        let ready = self.b.create_block();
        let slow = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.b.set_cold_block(slow);
        self.brif(valid, hit, &[], slow, &[]);
        self.switch(hit);
        // No reentry occurs between validating resolution and loading the current binding value.
        let value = self.b.ins().load(types::I64, flags, slot, 0);
        let uninitialized =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, value, Value::UNINITIALIZED.bits() as i64);
        self.brif(uninitialized, slow, &[], ready, &[]);
        self.switch(ready);
        if retain {
            let retain_block = self.b.create_block();
            let heap = self.is_heap(value);
            self.brif(heap, retain_block, &[], done, &[BlockArg::Value(value)]);
            self.switch(retain_block);
            let value = self.retain_temporary(value);
            self.jump(done, &[BlockArg::Value(value)]);
        } else {
            self.jump(done, &[BlockArg::Value(value)]);
        }
        self.switch(slow);
        let helper = match (optional, retain) {
            (false, true) => Helper::GlobalGet,
            (true, true) => Helper::GlobalGetOptional,
            (false, false) => Helper::GlobalGetBorrowed,
            (true, false) => Helper::GlobalGetOptionalBorrowed,
        };
        let value = self.call_helper(helper, &[address]);
        self.check_error();
        self.jump(done, &[BlockArg::Value(value)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn global_initialize(&mut self, name: &str, value: ClifValue) {
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        self.call_helper(Helper::GlobalInitialize, &[ptr, len, value]);
        self.check_error();
    }

    pub(super) fn cached_property_get(
        &mut self,
        object: ClifValue,
        name: &str,
        retain: bool,
    ) -> ClifValue {
        self.cached_property_read(object, name, retain, false)
    }

    pub(super) fn cached_property_probe(&mut self, object: ClifValue, name: &str) -> ClifValue {
        self.cached_property_read(object, name, false, true)
    }

    fn cached_property_read(
        &mut self,
        object: ClifValue,
        name: &str,
        retain: bool,
        probe: bool,
    ) -> ClifValue {
        let address = self.function_data(PropertyCache::new(name));
        let flags = MemFlagsData::trusted();
        let heap = self.is_heap(object);
        let address_bits = self.b.ins().band_imm_s(object, ADDRESS_MASK);
        let present = self.b.ins().icmp_imm_s(IntCC::NotEqual, address_bits, 0);
        let valid = self.b.ins().band(heap, present);
        let empty = self
            .b
            .ins()
            .iconst(self.pointer, NO_ELEMENTS.as_ptr() as i64);
        let view = self.b.ins().select(valid, address_bits, empty);
        let builtin = self.b.create_block();
        let shaped = self.b.create_block();
        let inline = self.b.create_block();
        let slow = self.b.create_block();
        let ready = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(ready, types::I64);
        self.b.append_block_param(done, types::I64);
        self.b.set_cold_block(slow);
        let kind = self.view_kind(view);
        let expected_kind =
            self.b
                .ins()
                .load(types::I32, flags, address, CACHE_BUILTIN_KIND_OFFSET);
        let cached = self.b.ins().icmp_imm_s(IntCC::NotEqual, expected_kind, 0);
        let matches = self.b.ins().icmp(IntCC::Equal, kind, expected_kind);
        let hit = self.b.ins().band(cached, matches);
        self.brif(hit, builtin, &[], shaped, &[]);
        self.switch(builtin);
        let bits = self
            .b
            .ins()
            .load(types::I64, flags, address, CACHE_BUILTIN_VALUE_OFFSET);
        self.jump(ready, &[BlockArg::Value(bits)]);
        self.switch(shaped);
        let shape = self.b.ins().load(types::I32, flags, view, SHAPE_OFFSET);
        let expected = self
            .b
            .ins()
            .load(types::I32, flags, address, CACHE_SHAPE_OFFSET);
        let holder = self
            .b
            .ins()
            .load(self.pointer, flags, address, CACHE_HOLDER_OFFSET);
        let own = self.b.ins().icmp_imm_s(IntCC::Equal, holder, 0);
        let holder = self.b.ins().select(own, view, holder);
        let holder_shape = self.b.ins().load(types::I32, flags, holder, SHAPE_OFFSET);
        let expected_holder =
            self.b
                .ins()
                .load(types::I32, flags, address, CACHE_HOLDER_SHAPE_OFFSET);
        let receiver_hit = self.b.ins().icmp(IntCC::Equal, shape, expected);
        let holder_hit = self
            .b
            .ins()
            .icmp(IntCC::Equal, holder_shape, expected_holder);
        let hit = self.b.ins().band(receiver_hit, holder_hit);
        self.brif(hit, inline, &[], slow, &[]);

        self.switch(inline);
        let index = self
            .b
            .ins()
            .load(types::I32, flags, address, CACHE_INDEX_OFFSET);
        let index = self.b.ins().uextend(types::I64, index);
        let offset = self.b.ins().ishl_imm_s(index, 3);
        let data = self.b.ins().load(self.pointer, flags, holder, DATA_OFFSET);
        let slot = self.b.ins().iadd(data, offset);
        let bits = self.b.ins().load(types::I64, flags, slot, 0);
        self.jump(ready, &[BlockArg::Value(bits)]);
        self.switch(ready);
        let bits = self.b.block_params(ready)[0];
        if retain {
            let heap = self.is_heap(bits);
            let retain = self.b.create_block();
            self.brif(heap, retain, &[], done, &[BlockArg::Value(bits)]);
            self.switch(retain);
            let retained = self.retain_temporary(bits);
            self.jump(done, &[BlockArg::Value(retained)]);
        } else {
            self.jump(done, &[BlockArg::Value(bits)]);
        }

        self.switch(slow);
        let result = self.call_helper(
            if probe {
                Helper::SemanticPropertyProbeCached
            } else {
                Helper::PropertyGetCached
            },
            &[object, address],
        );
        if !probe {
            self.check_error();
        }
        self.jump(done, &[BlockArg::Value(result)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn property_cache_hit(
        &mut self,
        object: ClifValue,
        address: ClifValue,
    ) -> ClifValue {
        let flags = MemFlagsData::trusted();
        let heap = self.is_heap(object);
        let address_bits = self.b.ins().band_imm_s(object, ADDRESS_MASK);
        let present = self.b.ins().icmp_imm_s(IntCC::NotEqual, address_bits, 0);
        let valid = self.b.ins().band(heap, present);
        let empty = self
            .b
            .ins()
            .iconst(self.pointer, NO_ELEMENTS.as_ptr() as i64);
        let view = self.b.ins().select(valid, address_bits, empty);
        let kind = self.view_kind(view);
        let expected_kind =
            self.b
                .ins()
                .load(types::I32, flags, address, CACHE_BUILTIN_KIND_OFFSET);
        let cached = self.b.ins().icmp_imm_s(IntCC::NotEqual, expected_kind, 0);
        let matches = self.b.ins().icmp(IntCC::Equal, kind, expected_kind);
        let builtin = self.b.ins().band(cached, matches);
        let shape = self.b.ins().load(types::I32, flags, view, SHAPE_OFFSET);
        let expected = self
            .b
            .ins()
            .load(types::I32, flags, address, CACHE_SHAPE_OFFSET);
        let holder = self
            .b
            .ins()
            .load(self.pointer, flags, address, CACHE_HOLDER_OFFSET);
        let own = self.b.ins().icmp_imm_s(IntCC::Equal, holder, 0);
        let holder = self.b.ins().select(own, view, holder);
        let holder_shape = self.b.ins().load(types::I32, flags, holder, SHAPE_OFFSET);
        let expected_holder =
            self.b
                .ins()
                .load(types::I32, flags, address, CACHE_HOLDER_SHAPE_OFFSET);
        let receiver_hit = self.b.ins().icmp(IntCC::Equal, shape, expected);
        let holder_hit = self
            .b
            .ins()
            .icmp(IntCC::Equal, holder_shape, expected_holder);
        let shaped = self.b.ins().band(receiver_hit, holder_hit);
        let hit = self.b.ins().bor(builtin, shaped);
        let initialized = self
            .b
            .ins()
            .load(types::I32, flags, address, CACHE_INITIALIZED_OFFSET);
        let initialized = self.b.ins().icmp_imm_s(IntCC::NotEqual, initialized, 0);
        self.b.ins().band(initialized, hit)
    }

    pub(super) fn cached_property_load(
        &mut self,
        object: ClifValue,
        address: ClifValue,
        retain: bool,
    ) -> ClifValue {
        let flags = MemFlagsData::trusted();
        let view = self.b.ins().band_imm_s(object, ADDRESS_MASK);
        let kind = self.view_kind(view);
        let expected_kind =
            self.b
                .ins()
                .load(types::I32, flags, address, CACHE_BUILTIN_KIND_OFFSET);
        let cached = self.b.ins().icmp_imm_s(IntCC::NotEqual, expected_kind, 0);
        let matches = self.b.ins().icmp(IntCC::Equal, kind, expected_kind);
        let builtin = self.b.ins().band(cached, matches);
        let builtin_block = self.b.create_block();
        let shaped_block = self.b.create_block();
        let ready = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(ready, types::I64);
        self.b.append_block_param(done, types::I64);
        self.brif(builtin, builtin_block, &[], shaped_block, &[]);
        self.switch(builtin_block);
        let bits = self
            .b
            .ins()
            .load(types::I64, flags, address, CACHE_BUILTIN_VALUE_OFFSET);
        self.jump(ready, &[BlockArg::Value(bits)]);
        self.switch(shaped_block);
        let holder = self
            .b
            .ins()
            .load(self.pointer, flags, address, CACHE_HOLDER_OFFSET);
        let own = self.b.ins().icmp_imm_s(IntCC::Equal, holder, 0);
        let holder = self.b.ins().select(own, view, holder);
        let index = self
            .b
            .ins()
            .load(types::I32, flags, address, CACHE_INDEX_OFFSET);
        let index = self.b.ins().uextend(types::I64, index);
        let offset = self.b.ins().ishl_imm_s(index, 3);
        let data = self.b.ins().load(self.pointer, flags, holder, DATA_OFFSET);
        let slot = self.b.ins().iadd(data, offset);
        let bits = self.b.ins().load(types::I64, flags, slot, 0);
        self.jump(ready, &[BlockArg::Value(bits)]);
        self.switch(ready);
        let bits = self.b.block_params(ready)[0];
        if retain {
            let heap = self.is_heap(bits);
            let retain = self.b.create_block();
            self.brif(heap, retain, &[], done, &[BlockArg::Value(bits)]);
            self.switch(retain);
            let retained = self.retain_temporary(bits);
            self.jump(done, &[BlockArg::Value(retained)]);
        } else {
            self.jump(done, &[BlockArg::Value(bits)]);
        }
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn fill_property_cache(
        &mut self,
        object: ClifValue,
        address: ClifValue,
    ) -> ClifValue {
        let result = self.call_helper(Helper::PropertyGetCached, &[object, address]);
        self.check_error();
        result
    }

    fn cached_property_set(
        &mut self,
        object: ClifValue,
        name: &str,
        value: ClifValue,
    ) -> ClifValue {
        let address = self.function_data(PropertyCache::new(name));
        let flags = MemFlagsData::trusted();
        let view = self.view(object);
        let slow = self.b.create_block();
        let inline = self.b.create_block();
        let retain = self.b.create_block();
        let store = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.b.set_cold_block(slow);
        let kind = self.view_kind(view);
        let inline_object =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, kind, (kind::OBJECT | kind::INLINE) as i64);
        let shape = self.b.ins().load(types::I32, flags, view, SHAPE_OFFSET);
        let expected = self
            .b
            .ins()
            .load(types::I32, flags, address, CACHE_ADDITION_SOURCE_OFFSET);
        let matches = self.b.ins().icmp(IntCC::Equal, shape, expected);
        let hit = self.b.ins().band(inline_object, matches);
        self.brif(hit, inline, &[], slow, &[]);
        self.switch(inline);
        let incoming = super::scope::heap_address(self.b, value);
        self.brif(incoming, retain, &[], store, &[]);
        self.switch(retain);
        super::scope::increment_references(self.b, incoming);
        self.jump(store, &[]);
        self.switch(store);
        let index = self
            .b
            .ins()
            .load(types::I32, flags, address, CACHE_ADDITION_INDEX_OFFSET);
        let index = self.b.ins().uextend(types::I64, index);
        let data = self.b.ins().load(self.pointer, flags, view, DATA_OFFSET);
        let offset = self.b.ins().ishl_imm_s(index, 3);
        let slot = self.b.ins().iadd(data, offset);
        self.b.ins().store(flags, value, slot, 0);
        let destination = self.b.ins().load(
            types::I32,
            flags,
            address,
            CACHE_ADDITION_DESTINATION_OFFSET,
        );
        self.b.ins().store(flags, destination, view, SHAPE_OFFSET);
        let length = self.b.ins().iadd_imm_s(index, 1);
        self.b.ins().store(flags, length, view, LENGTH_OFFSET);
        let small_length = self.b.ins().ireduce(types::I8, length);
        self.b
            .ins()
            .store(flags, small_length, data, INLINE_PROPERTY_LENGTH_OFFSET);
        self.jump(done, &[BlockArg::Value(value)]);
        self.switch(slow);
        let result = self.call_helper(Helper::PropertySetCached, &[object, address, value]);
        self.check_error();
        self.jump(done, &[BlockArg::Value(result)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn property_access(
        &mut self,
        object: ClifValue,
        name: &str,
        value: Option<ClifValue>,
    ) -> ClifValue {
        if let Some(value) = value {
            return self.cached_property_set(object, name, value);
        }
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        let result = self.call_helper(Helper::PropertyGet, &[object, ptr, len]);
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

    fn pending(&mut self, offset: i32, none: i64) -> ClifValue {
        let word = self
            .b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), self.ctx, offset);
        self.b.ins().icmp_imm_s(IntCC::NotEqual, word, none)
    }

    pub(super) fn check_error(&mut self) {
        // Execution reaches a helper with no pending exception; catch entry clears it.
        let can_throw = self
            .b
            .current_block()
            .and_then(|block| self.b.func.layout.last_inst(block))
            .and_then(|inst| self.annotations.helpers.get(&inst))
            .is_none_or(|helper| helper.can_throw());
        let condition = self.pending(crate::jit::session::ERROR_OFFSET, 0);
        let failed = self.b.create_block();
        let done = self.b.create_block();
        self.brif(condition, failed, &[], done, &[]);
        self.switch(failed);
        let source_name = self.source_name;
        let offset = *self
            .source_names
            .entry(source_name.0)
            .or_insert_with(|| self.data.push(source_name));
        let source = self.b.ins().iadd_imm_s(self.data_base, offset);
        let start = self
            .b
            .ins()
            .iconst(types::I64, self.source_span.start as i64);
        let end = self.b.ins().iconst(types::I64, self.source_span.end as i64);
        let origin = match self.error_origin {
            Some(block) => block,
            None => {
                let block = self.b.create_block();
                self.b.set_cold_block(block);
                for _ in 0..3 {
                    self.b.append_block_param(block, types::I64);
                }
                self.error_origin = Some(block);
                block
            }
        };
        self.b.set_cold_block(failed);
        self.jump(
            origin,
            &[
                BlockArg::Value(source),
                BlockArg::Value(start),
                BlockArg::Value(end),
            ],
        );
        self.switch(done);
        if !can_throw {
            return;
        }
        let condition = self.pending(
            crate::jit::session::EXCEPTION_OFFSET,
            Value::UNINITIALIZED.bits() as i64,
        );
        let done = self.b.create_block();
        if let Some(handler) = self.handlers.last().copied() {
            self.brif(condition, handler, &[], done, &[]);
        } else {
            let exit = self.failed_exit();
            self.brif(condition, exit, &[], done, &[]);
        }
        self.switch(done);
    }
}
