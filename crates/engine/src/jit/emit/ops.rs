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
use crate::jit::helpers::{
    self, CACHE_ADDITION_DESTINATION_OFFSET, CACHE_ADDITION_INDEX_OFFSET,
    CACHE_ADDITION_SOURCE_OFFSET, CACHE_BUILTIN_KIND_OFFSET, CACHE_BUILTIN_VALUE_OFFSET,
    CACHE_ENTRY_OFFSET, CACHE_ENTRY_SIZE, CACHE_HOLDER_OFFSET, CACHE_HOLDER_SHAPE_OFFSET,
    CACHE_INDEX_OFFSET, CACHE_LEN_OFFSET, CACHE_MEGAMORPHIC_OFFSET, CACHE_OVERFLOW_OFFSET,
    CACHE_OVERWRITE_INDEX_OFFSET, CACHE_OVERWRITE_SHAPE_OFFSET, CACHE_SHAPE_OFFSET, HELPERS,
    Helper, PROPERTY_CACHE_LIMIT, PropertyCache,
};
use crate::jit::profile::{CallSite, Site};
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

    pub(super) fn is_integer(&mut self, value: ClifValue) -> ClifValue {
        let numeric = self.is_number(value);
        let float = self.b.ins().bitcast(types::F64, MemFlagsData::new(), value);
        let integer = self.b.ins().fcvt_to_sint_sat(types::I64, float);
        let exact = self.b.ins().fcvt_from_sint(types::F64, integer);
        let integral = self.b.ins().fcmp(FloatCC::Equal, float, exact);
        let magnitude = self.b.ins().fabs(float);
        let limit = self.b.ins().f64const(9007199254740991.0);
        let safe = self
            .b
            .ins()
            .fcmp(FloatCC::LessThanOrEqual, magnitude, limit);
        let positive_zero = self.b.ins().icmp_imm_s(IntCC::NotEqual, value, i64::MIN);
        let valid = self.b.ins().band(numeric, integral);
        let valid = self.b.ins().band(valid, safe);
        self.b.ins().band(valid, positive_zero)
    }

    pub(super) fn is_number(&mut self, value: ClifValue) -> ClifValue {
        let top = self.b.ins().ushr_imm_s(value, TAG_SHIFT as i64);
        let offset = self.b.ins().iadd_imm_s(top, -(TAG_FIRST as i64));

        self.b
            .ins()
            .icmp_imm_s(IntCC::UnsignedGreaterThan, offset, TAG_SPAN as i64)
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
        observer: Option<(u64, crate::jit::profile::Site)>,
    ) -> ClifValue {
        let offset = match self.property_caches.get(&site) {
            Some(&offset) => offset,
            None => {
                let offset = self.data.push(PropertyCache::new(name, observer, false));
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
        let result = self.b.inst_results(call)[0];
        self.unchecked = None;
        if helper.returns_reference() {
            self.track(result);
            self.unchecked = self.owned.last().copied();
        }
        result
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

                compiler.b.ins().select(unordered, nan, bits)
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
        let (helper, index) = if matches!(index.repr, Repr::Float | Repr::Int32 | Repr::Int64) {
            (Helper::IndexGetNumeric, self.convert(index, Repr::Float))
        } else {
            (Helper::IndexGet, index.value)
        };
        let result = self.call_helper(helper, &[object, index]);
        self.check_error();
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
        known_code: bool,
    ) -> Result<ClifValue, Error> {
        if matches!(target.variant, crate::jit::entries::VariantId::Ordinary(_) if target.variant.index() >= crate::jit::entries::DIRECT_ENTRIES)
        {
            let result = self.generic_call(callee, receiver, args, span)?;
            if owned {
                self.increment(result);
            }
            return Ok(result);
        }
        if args.len() == usize::from(target.arity)
            && let Some(miss) = self.stubs.miss(target.arity)
        {
            return Ok(self.selected_call(
                target, callee, receiver, args, span, miss, owned, known_code,
            ));
        }
        let (entry, direct) = self.direct_entry(target, callee, known_code);
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
        known_code: bool,
    ) -> ClifValue {
        self.mark_call_site(span, true);
        let (entry, direct) = self.direct_entry(target, callee, known_code);
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

    fn direct_entry(
        &mut self,
        target: DirectTarget,
        callee: ClifValue,
        known_code: bool,
    ) -> (ClifValue, ClifValue) {
        if let Some(call) = super::stubs::poll(
            self.b,
            self.module,
            self.stubs.compilation_state,
            self.ctx,
            None,
        ) {
            self.annotations.helpers.insert(call, Helper::CompilePoll);
        }
        let entry = match target.variant {
            crate::jit::entries::VariantId::Ordinary(_) => {
                let slot = self.b.ins().iconst(
                    self.pointer,
                    self.stubs.direct_entries as i64
                        + target.variant.index() as i64
                            * crate::jit::entries::DIRECT_ENTRY_SIZE as i64,
                );
                self.b
                    .ins()
                    .load(self.pointer, MemFlagsData::trusted(), slot, 0)
            }
            crate::jit::entries::VariantId::Specialized(address) => {
                let slot = self.b.ins().iconst(self.pointer, address as i64);
                let entry = self
                    .b
                    .ins()
                    .atomic_load(self.pointer, MemFlagsData::trusted(), slot);
                let ready = self.b.ins().icmp_imm_s(IntCC::NotEqual, entry, 0);
                let request = self.b.create_block();
                let done = self.b.create_block();
                self.b.append_block_param(done, self.pointer);
                self.b.set_cold_block(request);
                self.brif(ready, done, &[BlockArg::Value(entry)], request, &[]);
                self.switch(request);
                let entry = self.call_helper(Helper::VariantEntry, &[slot]);
                let entry = if self.pointer == types::I64 {
                    entry
                } else {
                    self.b.ins().ireduce(self.pointer, entry)
                };
                self.jump(done, &[BlockArg::Value(entry)]);
                self.switch(done);
                self.b.block_params(done)[0]
            }
        };
        let published = self.b.ins().icmp_imm_s(IntCC::NotEqual, entry, 0);
        let direct = if known_code {
            published
        } else {
            let expected = self.is_code(callee, target.code);
            self.b.ins().band(expected, published)
        };
        (entry, direct)
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
        let result = self.property_access(object, "length", None, None);
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
        let position = if matches!(key.repr, Repr::Int32 | Repr::Int64) {
            self.convert(key, Repr::Int64)
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

    pub(super) fn array_load(
        &mut self,
        view: ClifValue,
        position: Operand,
        holey: bool,
        retain: bool,
    ) -> ClifValue {
        let position = self.convert(position, Repr::Int64);
        let flags = MemFlagsData::trusted();
        let data = self.b.ins().load(self.pointer, flags, view, DATA_OFFSET);
        let offset = self.b.ins().ishl_imm_s(position, 3);
        let address = self.b.ins().iadd(data, offset);
        let mut bits = self.b.ins().load(types::I64, flags, address, 0);
        if holey {
            let hole = self
                .b
                .ins()
                .icmp_imm_s(IntCC::Equal, bits, HOLE.bits() as i64);
            let undefined = self.constant(Value::UNDEFINED);
            bits = self.b.ins().select(hole, undefined, bits);
        }
        if retain {
            let heap = self.is_heap(bits);
            let retained = self.b.create_block();
            let done = self.b.create_block();
            self.b.append_block_param(done, types::I64);
            self.brif(heap, retained, &[], done, &[bits.into()]);
            self.switch(retained);
            let value = self.retain_temporary(bits);
            self.jump(done, &[value.into()]);
            self.switch(done);
            bits = self.b.block_params(done)[0];
        }
        bits
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
        let inside = if matches!(key.repr, Repr::Int32 | Repr::Int64) {
            let position = self.convert(key, Repr::Int64);
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
        let (position, fast) = if matches!(key.repr, Repr::Int32 | Repr::Int64) {
            let position = self.convert(key, Repr::Int64);
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
        let overwrite = value.map(|_| {
            let block = self.b.create_block();
            switch.set_entry(kind::ARRAY as u128, block);
            block
        });
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
        if let Some(overwrite) = overwrite {
            self.switch(overwrite);
            let value = self.convert(value.unwrap(), Repr::Boxed);
            self.overwrite_slot(view, position, value, slow, done);
        }
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
        let helper = match (optional, retain) {
            (false, true) => Helper::GlobalGet,
            (true, true) => Helper::GlobalGetOptional,
            (false, false) => Helper::GlobalGetBorrowed,
            (true, false) => Helper::GlobalGetOptionalBorrowed,
        };
        self.cached_global(name, helper, retain)
    }

    pub(super) fn cached_global_probe(&mut self, name: &str) -> ClifValue {
        self.cached_global(name, Helper::SemanticGlobalProbe, false)
    }

    fn cached_global(&mut self, name: &str, helper: Helper, retain: bool) -> ClifValue {
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
        let value = self.call_helper(helper, &[address]);
        if helper != Helper::SemanticGlobalProbe {
            self.check_error();
        }
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
        observer: Option<(u64, crate::jit::profile::Site)>,
        megamorphic: bool,
    ) -> ClifValue {
        self.cached_property_read(object, name, retain, false, observer, megamorphic)
    }

    pub(super) fn cached_property_probe(
        &mut self,
        object: ClifValue,
        name: &str,
        observer: Option<(u64, crate::jit::profile::Site)>,
        megamorphic: bool,
    ) -> ClifValue {
        self.cached_property_read(object, name, false, true, observer, megamorphic)
    }

    fn cached_property_read(
        &mut self,
        object: ClifValue,
        name: &str,
        retain: bool,
        probe: bool,
        observer: Option<(u64, crate::jit::profile::Site)>,
        megamorphic: bool,
    ) -> ClifValue {
        if megamorphic && !probe {
            return self.property_access(object, name, None, None);
        }
        let address = self.function_data(PropertyCache::new(name, observer, megamorphic));
        if megamorphic {
            return self.call_helper(Helper::SemanticPropertyProbeCached, &[object, address]);
        }
        let builtins = crate::heap::builtin_method(name);
        let entry = self.property_cache_entry(object, address, builtins, true);
        let hit = self.b.ins().icmp_imm_s(IntCC::NotEqual, entry, 0);
        let inline = self.b.create_block();
        let slow = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.brif(hit, inline, &[], slow, &[]);
        self.switch(inline);
        let value = self.cached_property_entry_load(object, entry, retain, builtins);
        self.jump(done, &[BlockArg::Value(value)]);
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

    fn property_cache_entry(
        &mut self,
        object: ClifValue,
        address: ClifValue,
        builtins: bool,
        validate: bool,
    ) -> ClifValue {
        let flags = MemFlagsData::trusted();
        let miss = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, self.pointer);
        if validate {
            let megamorphic =
                self.b
                    .ins()
                    .load(types::I32, flags, address, CACHE_MEGAMORPHIC_OFFSET);
            let lookup = self.b.create_block();
            self.brif(megamorphic, miss, &[], lookup, &[]);
            self.switch(lookup);
        }
        let heap = self.is_heap(object);
        let bits = self.b.ins().band_imm_s(object, ADDRESS_MASK);
        let present = self.b.ins().icmp_imm_s(IntCC::NotEqual, bits, 0);
        let valid = self.b.ins().band(heap, present);
        let empty = self
            .b
            .ins()
            .iconst(self.pointer, NO_ELEMENTS.as_ptr() as i64);
        let bits = if self.pointer == types::I64 {
            bits
        } else {
            self.b.ins().ireduce(self.pointer, bits)
        };
        let view = self.b.ins().select(valid, bits, empty);
        let shape = self.b.ins().load(types::I32, flags, view, SHAPE_OFFSET);
        let kind = builtins.then(|| self.view_kind(view));
        let len = self
            .b
            .ins()
            .load(types::I32, flags, address, CACHE_LEN_OFFSET);
        let mut overflow = None;
        for index in 0..PROPERTY_CACHE_LIMIT {
            let entry = if index == 0 {
                self.b
                    .ins()
                    .iadd_imm_s(address, i64::from(CACHE_ENTRY_OFFSET))
            } else {
                let overflow = *overflow.get_or_insert_with(|| {
                    self.b
                        .ins()
                        .load(self.pointer, flags, address, CACHE_OVERFLOW_OFFSET)
                });
                self.b
                    .ins()
                    .iadd_imm_s(overflow, (index - 1) as i64 * CACHE_ENTRY_SIZE)
            };
            let next = self.b.create_block();
            let mut shaped_kind = None;
            if let Some(kind) = kind {
                let expected =
                    self.b
                        .ins()
                        .load(types::I32, flags, entry, CACHE_BUILTIN_KIND_OFFSET);
                let cached = self.b.ins().icmp_imm_s(IntCC::NotEqual, expected, 0);
                shaped_kind = Some(self.b.ins().icmp_imm_s(IntCC::Equal, expected, 0));
                let matches = self.b.ins().icmp(IntCC::Equal, kind, expected);
                let hit = self.b.ins().band(cached, matches);
                let shaped = self.b.create_block();
                self.brif(hit, done, &[BlockArg::Value(entry)], shaped, &[]);
                self.switch(shaped);
            }
            let expected = self
                .b
                .ins()
                .load(types::I32, flags, entry, CACHE_SHAPE_OFFSET);
            let mut matches = self.b.ins().icmp(IntCC::Equal, shape, expected);
            if let Some(shaped) = shaped_kind {
                matches = self.b.ins().band(matches, shaped);
            }
            if validate {
                let holder_check = self.b.create_block();
                self.brif(matches, holder_check, &[], next, &[]);
                self.switch(holder_check);
                let holder = self
                    .b
                    .ins()
                    .load(self.pointer, flags, entry, CACHE_HOLDER_OFFSET);
                let own = self.b.ins().icmp_imm_s(IntCC::Equal, holder, 0);
                let holder = self.b.ins().select(own, view, holder);
                let actual = self.b.ins().load(types::I32, flags, holder, SHAPE_OFFSET);
                let expected =
                    self.b
                        .ins()
                        .load(types::I32, flags, entry, CACHE_HOLDER_SHAPE_OFFSET);
                let matches = self.b.ins().icmp(IntCC::Equal, actual, expected);
                self.brif(matches, done, &[BlockArg::Value(entry)], next, &[]);
            } else {
                self.brif(matches, done, &[BlockArg::Value(entry)], next, &[]);
            }
            self.switch(next);
            if index + 1 < PROPERTY_CACHE_LIMIT {
                let more =
                    self.b
                        .ins()
                        .icmp_imm_u(IntCC::UnsignedGreaterThan, len, (index + 1) as i64);
                let lookup = self.b.create_block();
                self.brif(more, lookup, &[], miss, &[]);
                self.switch(lookup);
            } else {
                self.jump(miss, &[]);
            }
        }
        self.switch(miss);
        let zero = self.b.ins().iconst(self.pointer, 0);
        self.jump(done, &[BlockArg::Value(zero)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn property_cache_hit(
        &mut self,
        object: ClifValue,
        address: ClifValue,
    ) -> ClifValue {
        let entry = self.property_cache_entry(object, address, true, true);
        self.b.ins().icmp_imm_s(IntCC::NotEqual, entry, 0)
    }

    pub(super) fn property_shape_hit(
        &mut self,
        object: ClifValue,
        address: ClifValue,
    ) -> ClifValue {
        let entry = self.property_cache_entry(object, address, false, true);
        self.b.ins().icmp_imm_s(IntCC::NotEqual, entry, 0)
    }

    pub(super) fn cached_property_load(
        &mut self,
        object: ClifValue,
        address: ClifValue,
        retain: bool,
        builtins: bool,
    ) -> ClifValue {
        // A dominating guard validated the holder; reselect without depending on mutable hit state.
        let entry = self.property_cache_entry(object, address, builtins, false);
        self.cached_property_entry_load(object, entry, retain, builtins)
    }

    fn cached_property_entry_load(
        &mut self,
        object: ClifValue,
        entry: ClifValue,
        retain: bool,
        builtins: bool,
    ) -> ClifValue {
        let flags = MemFlagsData::trusted();
        let ready = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(ready, types::I64);
        self.b.append_block_param(done, types::I64);
        if builtins {
            let kind = self
                .b
                .ins()
                .load(types::I32, flags, entry, CACHE_BUILTIN_KIND_OFFSET);
            let builtin = self.b.create_block();
            let shaped = self.b.create_block();
            self.brif(kind, builtin, &[], shaped, &[]);
            self.switch(builtin);
            let value = self
                .b
                .ins()
                .load(types::I64, flags, entry, CACHE_BUILTIN_VALUE_OFFSET);
            self.jump(ready, &[BlockArg::Value(value)]);
            self.switch(shaped);
        }
        let view = self.b.ins().band_imm_s(object, ADDRESS_MASK);
        let view = if self.pointer == types::I64 {
            view
        } else {
            self.b.ins().ireduce(self.pointer, view)
        };
        let holder = self
            .b
            .ins()
            .load(self.pointer, flags, entry, CACHE_HOLDER_OFFSET);
        let own = self.b.ins().icmp_imm_s(IntCC::Equal, holder, 0);
        let holder = self.b.ins().select(own, view, holder);
        let index = self
            .b
            .ins()
            .load(types::I32, flags, entry, CACHE_INDEX_OFFSET);
        let index = if self.pointer == types::I64 {
            self.b.ins().uextend(self.pointer, index)
        } else {
            index
        };
        let offset = self.b.ins().ishl_imm_s(index, 3);
        let data = self.b.ins().load(self.pointer, flags, holder, DATA_OFFSET);
        let slot = self.b.ins().iadd(data, offset);
        let value = self.b.ins().load(types::I64, flags, slot, 0);
        self.jump(ready, &[BlockArg::Value(value)]);
        self.switch(ready);
        let value = self.b.block_params(ready)[0];
        if retain {
            let heap = self.is_heap(value);
            let own = self.b.create_block();
            self.brif(heap, own, &[], done, &[BlockArg::Value(value)]);
            self.switch(own);
            let value = self.retain_temporary(value);
            self.jump(done, &[BlockArg::Value(value)]);
        } else {
            self.jump(done, &[BlockArg::Value(value)]);
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

    fn overwrite_slot(
        &mut self,
        view: ClifValue,
        overwrite_index: ClifValue,
        value: ClifValue,
        slow: cranelift_codegen::ir::Block,
        done: cranelift_codegen::ir::Block,
    ) {
        let flags = MemFlagsData::trusted();
        let overwrite_data = self.b.ins().load(self.pointer, flags, view, DATA_OFFSET);
        let overwrite_offset = self.b.ins().ishl_imm_s(overwrite_index, 3);
        let overwrite_slot = self.b.ins().iadd(overwrite_data, overwrite_offset);
        let old = self.b.ins().load(types::I64, flags, overwrite_slot, 0);
        let old_address = super::scope::heap_address(self.b, old);
        let old_native = self.b.create_block();
        let overwrite_retain = self.b.create_block();
        let overwrite_store = self.b.create_block();
        let overwrite_release = self.b.create_block();
        let overwrite_done = self.b.create_block();
        self.brif(old_address, old_native, &[], overwrite_retain, &[]);
        self.switch(old_native);
        let old_view_kind = self.view_kind(old_address);
        let native_bits = self.b.ins().band_imm_s(old_view_kind, kind::NATIVE as i64);
        let is_native = self.b.ins().icmp_imm_s(IntCC::NotEqual, native_bits, 0);
        self.brif(is_native, slow, &[], overwrite_retain, &[]);
        self.switch(overwrite_retain);
        let overwrite_incoming = super::scope::heap_address(self.b, value);
        let overwrite_increment = self.b.create_block();
        self.brif(
            overwrite_incoming,
            overwrite_increment,
            &[],
            overwrite_store,
            &[],
        );
        self.switch(overwrite_increment);
        super::scope::increment_references(self.b, overwrite_incoming);
        self.jump(overwrite_store, &[]);
        self.switch(overwrite_store);
        self.b.ins().store(flags, value, overwrite_slot, 0);
        self.brif(old_address, overwrite_release, &[], overwrite_done, &[]);
        self.switch(overwrite_release);
        self.release_inline(old, overwrite_done, false);
        self.switch(overwrite_done);
        self.jump(done, &[BlockArg::Value(value)]);
    }

    pub(super) fn shape_hit(&mut self, object: ClifValue, shape: u32) -> ClifValue {
        let view = self.view(object);
        let actual = self
            .b
            .ins()
            .load(types::I32, MemFlagsData::trusted(), view, SHAPE_OFFSET);
        self.b.ins().icmp_imm_s(IntCC::Equal, actual, shape as i64)
    }

    pub(super) fn shaped_property_probe(
        &mut self,
        object: ClifValue,
        shape: u32,
        index: u32,
    ) -> ClifValue {
        let hit = self.shape_hit(object, shape);
        let fast = self.b.create_block();
        let miss = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.brif(hit, fast, &[], miss, &[]);
        self.switch(fast);
        let value = self.shaped_property_get(object, index, false);
        self.jump(done, &[BlockArg::Value(value)]);
        self.switch(miss);
        let missing = self.constant(Value::UNINITIALIZED);
        self.jump(done, &[BlockArg::Value(missing)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn shaped_property_get(
        &mut self,
        object: ClifValue,
        index: u32,
        retain: bool,
    ) -> ClifValue {
        let flags = MemFlagsData::trusted();
        let view = self.b.ins().band_imm_s(object, ADDRESS_MASK);
        let data = self.b.ins().load(self.pointer, flags, view, DATA_OFFSET);
        let bits = self
            .b
            .ins()
            .load(types::I64, flags, data, (index as i32) * 8);
        if !retain {
            return bits;
        }
        let heap = self.is_heap(bits);
        let retain = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.brif(heap, retain, &[], done, &[BlockArg::Value(bits)]);
        self.switch(retain);
        let retained = self.retain_temporary(bits);
        self.jump(done, &[BlockArg::Value(retained)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn shaped_property_set(
        &mut self,
        object: ClifValue,
        name: &str,
        index: u32,
        value: ClifValue,
    ) {
        let view = self.b.ins().band_imm_s(object, ADDRESS_MASK);
        let index = self.b.ins().iconst(types::I64, index as i64);
        let slow = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        self.b.set_cold_block(slow);
        self.overwrite_slot(view, index, value, slow, done);
        self.switch(slow);
        let address = self.function_data(PropertyCache::new(name, None, false));
        self.call_helper(Helper::PropertySetCached, &[object, address, value]);
        self.check_error();
        self.jump(done, &[BlockArg::Value(value)]);
        self.switch(done);
    }

    fn cached_property_set(
        &mut self,
        object: ClifValue,
        name: &str,
        value: ClifValue,
        observer: Option<(u64, crate::jit::profile::Site)>,
    ) -> ClifValue {
        let address = self.function_data(PropertyCache::new(name, observer, false));
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
        let overwrite_check = self.b.create_block();
        self.brif(hit, inline, &[], overwrite_check, &[]);
        self.switch(overwrite_check);
        let object_bits = self.b.ins().band_imm_s(kind, kind::OBJECT as i64);
        let is_object = self.b.ins().icmp_imm_s(IntCC::NotEqual, object_bits, 0);
        let overwrite_expected =
            self.b
                .ins()
                .load(types::I32, flags, address, CACHE_OVERWRITE_SHAPE_OFFSET);
        let overwrite_matches = self.b.ins().icmp(IntCC::Equal, shape, overwrite_expected);
        let overwrite_hit = self.b.ins().band(is_object, overwrite_matches);
        let overwrite = self.b.create_block();
        self.brif(overwrite_hit, overwrite, &[], slow, &[]);
        self.switch(overwrite);
        let overwrite_index =
            self.b
                .ins()
                .load(types::I32, flags, address, CACHE_OVERWRITE_INDEX_OFFSET);
        let overwrite_index = self.b.ins().uextend(types::I64, overwrite_index);
        self.overwrite_slot(view, overwrite_index, value, slow, done);
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
        observer: Option<(u64, crate::jit::profile::Site)>,
    ) -> ClifValue {
        if let Some(value) = value {
            return self.cached_property_set(object, name, value, observer);
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
        self.b.set_cold_block(failed);
        self.brif(condition, failed, &[], done, &[]);
        self.switch(failed);
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
        self.unwind(
            origin,
            &[
                BlockArg::Value(source),
                BlockArg::Value(start),
                BlockArg::Value(end),
            ],
        );
        self.switch(done);
        if !can_throw {
            self.unchecked = None;
            return;
        }
        let condition = self.pending(
            crate::jit::session::EXCEPTION_OFFSET,
            Value::UNINITIALIZED.bits() as i64,
        );
        let done = self.b.create_block();
        let target = match self.handlers.last().copied() {
            Some(handler) => handler,
            None => self.failed_exit(),
        };
        if !self.unwind_releases() {
            self.brif(condition, target, &[], done, &[]);
        } else {
            let thrown = self.b.create_block();
            self.b.set_cold_block(thrown);
            self.brif(condition, thrown, &[], done, &[]);
            self.switch(thrown);
            self.unwind(target, &[]);
        }
        self.switch(done);
        self.unchecked = None;
    }
}
