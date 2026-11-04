use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{BlockArg, InstBuilder, MemFlags, Type, Value as ClifValue, types};
use cranelift_frontend::Switch;
use cranelift_module::Module;
use oxc_syntax::operator::BinaryOperator;

use crate::heap::{DATA_OFFSET, HOLE, KIND_OFFSET, LENGTH_OFFSET, MathFunction, NO_ELEMENTS, kind};
use crate::jit::helpers::Helper;
use crate::jit::numeric::{IndexedRead, IndexedReadContract, Kind};
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

const ADDRESS_MASK: i64 = (1 << TAG_SHIFT) - 1;

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
            BinaryOp::Less
            | BinaryOp::LessEqual
            | BinaryOp::Greater
            | BinaryOp::GreaterEqual
            | BinaryOp::LooseEqual
            | BinaryOp::LooseNotEqual
            | BinaryOp::StrictEqual
            | BinaryOp::StrictNotEqual => self.numeric_comparison(op, left, right),
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
            .map(|&arg| self.b.ins().bitcast(types::F64, MemFlags::new(), arg))
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
        let bits = self.b.ins().bitcast(types::I64, MemFlags::new(), result);
        let nan = self.constant(Value::number(f64::NAN));
        let boxed = self.b.ins().select(unordered, nan, bits);
        self.annotations.results.insert(boxed, Kind::Number);
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

    // Non-objects and the global object (address zero) see an all-zero slot, so every
    // kind check fails and the length reads as zero.
    pub(super) fn view(&mut self, object: ClifValue) -> ClifValue {
        let tag = self.b.ins().ushr_imm(object, TAG_SHIFT as i64);
        let is_object = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, tag, (TAG_OBJECT >> TAG_SHIFT) as i64);
        let address = self.b.ins().band_imm(object, ADDRESS_MASK);
        let present = self.b.ins().icmp_imm(IntCC::NotEqual, address, 0);
        let valid = self.b.ins().band(is_object, present);
        let empty = self
            .b
            .ins()
            .iconst(self.pointer, NO_ELEMENTS.as_ptr() as i64);
        self.b.ins().select(valid, address, empty)
    }

    pub(super) fn view_kind(&mut self, view: ClifValue) -> ClifValue {
        self.b
            .ins()
            .load(types::I32, MemFlags::trusted(), view, KIND_OFFSET)
    }

    pub(super) fn has_kind(&mut self, view: ClifValue, mask: u32) -> ClifValue {
        let kind = self.view_kind(view);
        let masked = self.b.ins().band_imm(kind, mask as i64);
        self.b.ins().icmp_imm(IntCC::NotEqual, masked, 0)
    }

    pub(super) fn view_length(&mut self, view: ClifValue) -> ClifValue {
        let length = self
            .b
            .ins()
            .load(types::I64, MemFlags::trusted(), view, LENGTH_OFFSET);
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
        let bits = self.b.ins().bitcast(types::I64, MemFlags::new(), length);
        self.jump(done, &[BlockArg::Value(bits)]);
        self.switch(slow);
        let result = self.property_access(object, "length", None);
        self.jump(done, &[BlockArg::Value(result)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn element_read(&mut self, object: ClifValue, key: Operand) -> ClifValue {
        let view = self.view(object);
        self.viewed_index_access(view, object, key, None, true)
    }

    pub(super) fn viewed_index_access(
        &mut self,
        view: ClifValue,
        object: ClifValue,
        key: Operand,
        value: Option<Operand>,
        fallback: bool,
    ) -> ClifValue {
        let flags = MemFlags::trusted();
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
            let offset = self.b.ins().ishl_imm(position, 3);
            let address = self.b.ins().iadd(data, offset);
            (packed, holey, address)
        });
        switch.emit(self.b, kind, slow);
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
                .icmp_imm(IntCC::Equal, bits, HOLE.bits() as i64);
            let undefined = self.constant(Value::UNDEFINED);
            let bits = self.b.ins().select(hole, undefined, bits);
            self.jump(element, &[BlockArg::Value(bits)]);
            self.switch(element);
            let bits = self.b.block_params(element)[0];
            let heap = self.is_heap(bits);
            // Matches the IndexGet helper, which hands heap elements out as temporaries.
            let retain = self.b.create_block();
            self.brif(heap, retain, &[], done, &[BlockArg::Value(bits)]);
            self.switch(retain);
            let retained = self.call_helper(Helper::TempRetain, &[bits]);
            self.check_error();
            self.jump(done, &[BlockArg::Value(retained)]);
        }
        for (element, block) in kinds {
            self.switch(block);
            let offset = self.b.ins().imul_imm(position, element.bytes() as i64);
            let address = self.b.ins().iadd(data, offset);
            match value {
                None => {
                    let raw = self.b.ins().load(element.into(), flags, address, 0);
                    let number = match element {
                        Element::F64 => raw,
                        Element::F32 => self.b.ins().fpromote(types::F64, raw),
                        Element::U32 => {
                            let wide = self.b.ins().uextend(types::I64, raw);
                            self.b.ins().fcvt_from_sint(types::F64, wide)
                        }
                        Element::U8 | Element::U16 => {
                            let wide = self.b.ins().uextend(types::I32, raw);
                            self.b.ins().fcvt_from_sint(types::F64, wide)
                        }
                        Element::I32 => self.b.ins().fcvt_from_sint(types::F64, raw),
                        Element::I8 | Element::I16 => {
                            let wide = self.b.ins().sextend(types::I32, raw);
                            self.b.ins().fcvt_from_sint(types::F64, wide)
                        }
                    };
                    let bits = self.b.ins().bitcast(types::I64, MemFlags::new(), number);
                    let bits = if element.is_float() {
                        let nan = self.b.ins().fcmp(FloatCC::Unordered, number, number);
                        let canonical = self.constant(Value::number(f64::NAN));
                        self.b.ins().select(nan, canonical, bits)
                    } else {
                        bits
                    };
                    self.jump(done, &[BlockArg::Value(bits)]);
                }
                Some(value) => {
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
            }
        }

        self.switch(slow);
        let result = if fallback {
            let index = self.convert(key, Repr::Boxed);
            let value = value.map(|value| self.convert(value, Repr::Boxed));
            self.index_access(object, index, value)
        } else {
            self.constant(Value::UNDEFINED)
        };
        self.jump(done, &[BlockArg::Value(result)]);
        self.switch(done);
        self.b.block_params(done)[0]
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

    fn pending(&mut self, offset: i32, none: i64) -> ClifValue {
        let word = self
            .b
            .ins()
            .load(types::I64, MemFlags::trusted(), self.ctx, offset);
        self.b.ins().icmp_imm(IntCC::NotEqual, word, none)
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
        let condition = self.pending(crate::jit::session::ERROR_OFFSET, 0);
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
        let condition = self.pending(
            crate::jit::session::EXCEPTION_OFFSET,
            Value::UNINITIALIZED.bits() as i64,
        );
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
