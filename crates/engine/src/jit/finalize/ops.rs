use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{MemFlagsData, StackSlotData, StackSlotKind, Type, types};
use cranelift_module::Module;
use oxc_span::Span;
use oxc_syntax::operator::BinaryOperator;

use crate::error::{Error, compile_at};
use crate::heap::{CODE_OFFSET, DATA_OFFSET, KIND_OFFSET, LENGTH_OFFSET, SHAPE_OFFSET, TypedKind};
use crate::jit::helpers::{
    self, CACHE_BUILTIN_KIND_OFFSET, CACHE_BUILTIN_VALUE_OFFSET, CACHE_ENTRY_OFFSET,
    CACHE_ENTRY_SIZE, CACHE_HOLDER_OFFSET, CACHE_HOLDER_SHAPE_OFFSET, CACHE_INDEX_OFFSET,
    CACHE_LEN_OFFSET, CACHE_MEGAMORPHIC_OFFSET, CACHE_OVERFLOW_OFFSET, CACHE_SHAPE_OFFSET, HELPERS,
    Helper, HelperKind, PROPERTY_CACHE_LIMIT, PropertyCache,
};
use crate::jit::profile::{CallSite, Site};
use crate::jit::semantic::DirectTarget;
use crate::jit::semantic::low_level::Binary;
use crate::jit::session::CALL_SITE_OFFSET;
use crate::value::Value;

use super::repr::Operand;
use super::{BlockArg, FunctionCompiler, LoweredValue, SigRef};
use crate::jit::semantic::Repr;
use crate::jit::semantic::runtime::layout;

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

impl Element {
    fn bytes(self) -> u32 {
        Type::from(self).bytes()
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
    pub(super) fn constant(&mut self, value: Value) -> LoweredValue {
        let word = self.b.ins().iconst(types::I64, value.bits() as i64);
        self.b.as_boxed(word)
    }

    pub(super) fn is_number(&mut self, value: LoweredValue) -> LoweredValue {
        let value = self.b.as_word(value);
        layout::is_number(self.b, value)
    }

    pub(super) fn truthy(&mut self, value: LoweredValue) -> LoweredValue {
        let number = self.is_number(value);
        let word = self.b.as_word(value);
        let number_block = self.b.create_block();
        let other_block = self.b.create_block();
        let string_block = self.b.create_block();
        let primitive_block = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param_repr(done, Repr::Boolean);
        self.brif(number, number_block, &[], other_block, &[]);

        self.switch(number_block);
        let float = self.b.ins().bitcast(types::F64, MemFlagsData::new(), word);
        let zero = self.b.ins().f64const(0.0);
        let negative = self.b.ins().fcmp(FloatCC::LessThan, float, zero);
        let positive = self.b.ins().fcmp(FloatCC::GreaterThan, float, zero);
        let truth = self.b.ins().boolean_or(negative, positive);
        self.jump(done, &[BlockArg::Value(truth)]);

        self.switch(other_block);
        let is_string = layout::is_string(self.b, word);
        self.brif(is_string, string_block, &[], primitive_block, &[]);

        self.switch(string_block);
        let truth = self.call_helper(Helper::Truthy, &[value]);
        let truth = self.b.as_word(truth);
        let truth = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, truth, Value::TRUE.bits() as i64);
        self.jump(done, &[BlockArg::Value(truth)]);

        self.switch(primitive_block);
        let undefined = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, word, Value::UNDEFINED.bits() as i64);
        let null = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, word, Value::NULL.bits() as i64);
        let false_value = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, word, Value::FALSE.bits() as i64);
        let falsy = self.b.ins().boolean_or(undefined, null);
        let falsy = self.b.ins().boolean_or(falsy, false_value);
        let no = self.b.ins().boolean(false);
        let truth = self.b.ins().boolean_equal(falsy, no);
        self.jump(done, &[BlockArg::Value(truth)]);

        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn unary_helper(&mut self, op: UnaryOp, value: LoweredValue) -> LoweredValue {
        let result = self.call_helper(op.helper(), &[value]);
        self.check_error();
        result
    }

    pub(super) fn binary_helper(
        &mut self,
        op: BinaryOp,
        left: LoweredValue,
        right: LoweredValue,
    ) -> LoweredValue {
        let result = self.call_helper(op.helper(), &[left, right]);
        self.check_error();
        result
    }

    pub(super) fn binary(
        &mut self,
        op: BinaryOp,
        left: LoweredValue,
        right: LoweredValue,
    ) -> LoweredValue {
        self.binary_helper(op, left, right)
    }

    pub(super) fn data_address(&mut self, offset: i64) -> LoweredValue {
        if offset == 0 {
            return self.data_base;
        }
        self.b.ins().iadd_imm_s(self.data_base, offset)
    }

    pub(super) fn function_data<T>(&mut self, value: T) -> LoweredValue {
        let offset = self.data.push(value);
        self.data_address(offset)
    }

    pub(super) fn property_cache(
        &mut self,
        site: crate::jit::semantic::Inst,
        name: &str,
        observer: Option<(u64, crate::jit::profile::Site)>,
    ) -> LoweredValue {
        let offset = match self.property_caches.get(&site) {
            Some(&offset) => offset,
            None => {
                let offset = self.data.push(PropertyCache::new(name, observer, false));
                self.property_caches.insert(site, offset);
                offset
            }
        };
        self.data_address(offset)
    }

    pub(super) fn call_helper(&mut self, helper: Helper, args: &[LoweredValue]) -> LoweredValue {
        self.call_helper_tracked(helper, args, helper.returns_reference())
    }

    pub(super) fn call_helper_tracked(
        &mut self,
        helper: Helper,
        args: &[LoweredValue],
        track: bool,
    ) -> LoweredValue {
        let target = self.module.target_config();
        let signature = *self.helper_signatures.entry(helper).or_insert_with(|| {
            self.b.import_signature(helpers::signature(
                helper,
                target.default_call_conv,
                self.pointer,
            ))
        });
        let offset = crate::jit::session::HELPERS_OFFSET
            + helper as i32 * std::mem::size_of::<usize>() as i32;
        let callee = self
            .b
            .ins()
            .load(self.pointer, MemFlagsData::trusted(), self.ctx, offset);
        let mut call_args = Vec::with_capacity(args.len() + 1);
        call_args.push(self.ctx);
        call_args.extend_from_slice(args);
        let call = self.b.ins().call_indirect(signature, callee, &call_args);
        self.b.mark_helper(call, helper);
        let mut result = self.b.inst_results(call)[0];
        if HELPERS[helper as usize].returns == HelperKind::Value {
            result = self.b.as_boxed(result);
        }
        self.unchecked = None;
        if track {
            self.unchecked = Some(
                self.b
                    .scratch(crate::jit::semantic::low_level::Scratch::Own, &[result]),
            );
        }
        result
    }
}

pub(super) fn to_number(
    compiler: &mut FunctionCompiler<'_, '_>,
    value: LoweredValue,
) -> LoweredValue {
    let is_number = compiler.is_number(value);
    let fast_block = compiler.b.create_block();
    let slow_block = compiler.b.create_block();
    let done = compiler.b.create_block();
    compiler.b.append_block_param_repr(done, Repr::Boxed);
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
    // Non-objects and the global object (address zero) see an all-zero slot, so every
    // kind check fails and the length reads as zero.
    pub(super) fn view(&mut self, object: LoweredValue) -> LoweredValue {
        let object = self.b.as_word(object);
        layout::object_view(self.b, object)
    }

    pub(super) fn generic_call(
        &mut self,
        callee: LoweredValue,
        receiver: LoweredValue,
        args: &[LoweredValue],
        span: Span,
    ) -> Result<LoweredValue, Error> {
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
        callee: LoweredValue,
        receiver: LoweredValue,
        args: &[LoweredValue],
        span: Span,
    ) -> Result<LoweredValue, Error> {
        let argv = self.stack_values(args, span)?;
        let argc = self.b.ins().iconst(types::I64, args.len() as i64);
        Ok(self.call_helper(Helper::Call, &[callee, receiver, argv, argc]))
    }

    /// With `owned`, the caller gets a reference of its own rather than a temporary. A direct
    /// call that reports no failure skips the pending-error checks.
    pub(super) fn direct_call(
        &mut self,
        target: DirectTarget,
        callee: LoweredValue,
        receiver: LoweredValue,
        args: &[LoweredValue],
        span: Span,
        owned: bool,
        known_code: bool,
    ) -> Result<LoweredValue, Error> {
        if matches!(target.variant, crate::jit::entries::VariantId::Ordinary(_) if target.variant.index() >= crate::jit::entries::DIRECT_ENTRIES)
        {
            let result = self.generic_call(callee, receiver, args, span)?;
            if owned {
                self.move_result_reference(result);
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
        self.b.append_block_param_repr(check, Repr::Boxed);
        self.b.append_block_param_repr(done, Repr::Boxed);
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
            self.move_result_reference(result);
        }
        self.jump(done, &[BlockArg::Value(result)]);
        self.switch(done);
        Ok(self.b.block_params(done)[0])
    }

    /// Falls back to the miss stub rather than an inline generic call.
    fn selected_call(
        &mut self,
        target: DirectTarget,
        callee: LoweredValue,
        receiver: LoweredValue,
        args: &[LoweredValue],
        span: Span,
        miss: usize,
        owned: bool,
        known_code: bool,
    ) -> LoweredValue {
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
        entry: LoweredValue,
        arity: u16,
        callee: LoweredValue,
        receiver: LoweredValue,
        args: &[LoweredValue],
        owned: bool,
    ) -> LoweredValue {
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
        entry: LoweredValue,
        arity: u16,
        callee: LoweredValue,
        receiver: LoweredValue,
        args: &[LoweredValue],
    ) -> (LoweredValue, LoweredValue) {
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
        (self.b.as_boxed(result), status)
    }

    fn owned_result(&mut self, result: LoweredValue, owned: bool) -> LoweredValue {
        if owned {
            return result;
        }
        self.own_temporary(result)
    }

    fn direct_entry(
        &mut self,
        target: DirectTarget,
        callee: LoweredValue,
        known_code: bool,
    ) -> (LoweredValue, LoweredValue) {
        if let Some(call) = super::poll::emit(
            self.b,
            self.module,
            self.stubs.compilation_state,
            self.ctx,
            None,
        ) {
            self.b.mark_helper(call, Helper::CompilePoll);
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
            self.b.ins().boolean_and(expected, published)
        };
        (entry, direct)
    }

    fn fixed_signature(&mut self, arity: u16) -> SigRef {
        if let Some(&signature) = self.fixed_signatures.get(&arity) {
            return signature;
        }
        let signature = crate::jit::cranelift::stubs::fixed_signature(self.pointer, arity);
        let signature = self.b.import_signature(signature);
        self.fixed_signatures.insert(arity, signature);
        signature
    }

    /// Whether a value is a closure of this code that may be called without `new`.
    pub(super) fn is_code(
        &mut self,
        value: LoweredValue,
        code: cranelift_module::FuncId,
    ) -> LoweredValue {
        let view = self.view(value);
        let word = self
            .b
            .ins()
            .load(types::I32, MemFlagsData::trusted(), view, CODE_OFFSET);
        self.b
            .ins()
            .icmp_imm_s(IntCC::Equal, word, i64::from(code.as_u32()) + 1)
    }

    pub(super) fn view_kind(&mut self, view: LoweredValue) -> LoweredValue {
        self.b
            .ins()
            .load(types::I32, MemFlagsData::trusted(), view, KIND_OFFSET)
    }

    /// Reads an element of a typed array of known kind at an index known to be in bounds.
    pub(super) fn bounded_element_read(
        &mut self,
        view: LoweredValue,
        key: Operand,
        kind: TypedKind,
    ) -> Operand {
        let element = Element::from(kind);
        let flags = MemFlagsData::trusted();
        let position = if key.repr == Repr::I64 {
            key.value
        } else {
            self.b.ins().fcvt_to_sint_sat(types::I64, key.value)
        };
        let data = self.b.ins().load(self.pointer, flags, view, DATA_OFFSET);
        let offset = self.b.ins().imul_imm_s(position, element.bytes() as i64);
        let address = self.b.ins().iadd(data, offset);
        let raw = self.b.ins().load(element.into(), flags, address, 0);
        match element {
            Element::F64 => Operand::new(raw, Repr::F64),
            Element::F32 => Operand::new(self.b.ins().fpromote(types::F64, raw), Repr::F64),
            Element::U32 => {
                let wide = self.b.ins().uextend(types::I64, raw);
                Operand::new(self.b.ins().fcvt_from_sint(types::F64, wide), Repr::F64)
            }
            Element::U8 | Element::U16 => {
                Operand::new(self.b.ins().uextend(types::I32, raw), Repr::I32)
            }
            Element::I8 | Element::I16 => {
                Operand::new(self.b.ins().sextend(types::I32, raw), Repr::I32)
            }
            Element::I32 => Operand::new(raw, Repr::I32),
        }
    }

    pub(super) fn checked_element_read(
        &mut self,
        view: LoweredValue,
        key: Operand,
        kind: TypedKind,
    ) -> Operand {
        let length = self
            .b
            .ins()
            .load(types::I64, MemFlagsData::trusted(), view, LENGTH_OFFSET);
        let inside = if key.repr == Repr::I64 {
            self.b
                .ins()
                .icmp(IntCC::UnsignedLessThan, key.value, length)
        } else {
            let number = key.value;
            let position = self.b.ins().fcvt_to_sint_sat(types::I64, number);
            let exact = self.b.ins().fcvt_from_sint(types::F64, position);
            let integral = self.b.ins().fcmp(FloatCC::Equal, exact, number);
            let inside = self.b.ins().icmp(IntCC::UnsignedLessThan, position, length);
            self.b.ins().boolean_and(integral, inside)
        };
        let read = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param_repr(done, Repr::Boxed);
        let undefined = self.constant(Value::UNDEFINED);
        self.brif(inside, read, &[], done, &[undefined.into()]);
        self.switch(read);
        let value = self.bounded_element_read(view, key, kind);
        let value = self.convert(value, Repr::Boxed);
        self.jump(done, &[value.into()]);
        self.switch(done);
        Operand::new(self.b.block_params(done)[0], Repr::Boxed)
    }

    pub(super) fn global_access(
        &mut self,
        name: &str,
        value: Option<LoweredValue>,
    ) -> LoweredValue {
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

    pub(super) fn global_get_optional(&mut self, name: &str) -> LoweredValue {
        self.cached_global_get(name, true, false)
    }

    pub(super) fn cached_global_get(
        &mut self,
        name: &str,
        optional: bool,
        retain: bool,
    ) -> LoweredValue {
        let helper = match (optional, retain) {
            (false, true) => Helper::GlobalGet,
            (true, true) => Helper::GlobalGetOptional,
            (false, false) => Helper::GlobalGetBorrowed,
            (true, false) => Helper::GlobalGetOptionalBorrowed,
        };
        self.cached_global(name, helper, retain)
    }

    pub(super) fn cached_global_probe(&mut self, name: &str) -> LoweredValue {
        self.cached_global(name, Helper::SemanticGlobalProbe, false)
    }

    fn cached_global(&mut self, name: &str, helper: Helper, retain: bool) -> LoweredValue {
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
        let valid = self.b.ins().boolean_and(same, valid);
        let slot = self.b.ins().load(
            self.pointer,
            flags,
            address,
            std::mem::offset_of!(GlobalCache, slot) as i32,
        );
        let present = self.b.ins().icmp_imm_s(IntCC::NotEqual, slot, 0);
        let valid = self.b.ins().boolean_and(valid, present);
        let hit = self.b.create_block();
        let ready = self.b.create_block();
        let slow = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param_repr(done, Repr::Boxed);
        self.b.set_cold_block(slow);
        self.brif(valid, hit, &[], slow, &[]);
        self.switch(hit);
        // No reentry occurs between validating resolution and loading the current binding value.
        let word = self.b.ins().load(types::I64, flags, slot, 0);
        let value = self.b.as_boxed(word);
        let uninitialized =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, word, Value::UNINITIALIZED.bits() as i64);
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

    pub(super) fn global_initialize(&mut self, name: &str, value: LoweredValue) {
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        self.call_helper(Helper::GlobalInitialize, &[ptr, len, value]);
        self.check_error();
    }

    pub(super) fn cached_property_get(
        &mut self,
        object: LoweredValue,
        name: &str,
        observer: Option<(u64, crate::jit::profile::Site)>,
        megamorphic: bool,
    ) -> LoweredValue {
        if megamorphic {
            return self.property_access(object, name, None, None);
        }
        let address = self.function_data(PropertyCache::new(name, observer, false));
        let result = self.call_helper(Helper::PropertyGetCached, &[object, address]);
        self.check_error();
        result
    }

    pub(super) fn cached_property_probe(
        &mut self,
        object: LoweredValue,
        name: &str,
        observer: Option<(u64, crate::jit::profile::Site)>,
        megamorphic: bool,
    ) -> LoweredValue {
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
        self.b.append_block_param_repr(done, Repr::Boxed);
        self.brif(hit, inline, &[], slow, &[]);
        self.switch(inline);
        let value = self.cached_property_entry_load(object, entry, builtins);
        self.jump(done, &[BlockArg::Value(value)]);
        self.switch(slow);
        let result = self.call_helper(Helper::SemanticPropertyProbeCached, &[object, address]);
        self.jump(done, &[BlockArg::Value(result)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    fn property_cache_entry(
        &mut self,
        object: LoweredValue,
        address: LoweredValue,
        builtins: bool,
        validate: bool,
    ) -> LoweredValue {
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
        let object = self.b.as_word(object);
        let view = layout::heap_view(self.b, object);
        let view = if self.pointer == types::I64 {
            view
        } else {
            self.b.ins().ireduce(self.pointer, view)
        };
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
                let hit = self.b.ins().boolean_and(cached, matches);
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
                matches = self.b.ins().boolean_and(matches, shaped);
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

    fn cached_property_entry_load(
        &mut self,
        object: LoweredValue,
        entry: LoweredValue,
        builtins: bool,
    ) -> LoweredValue {
        let flags = MemFlagsData::trusted();
        let ready = self.b.create_block();
        self.b.append_block_param(ready, types::I64);
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
        let object = self.b.as_word(object);
        let view = layout::address(self.b, object);
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
        let word = self.b.block_params(ready)[0];
        self.b.as_boxed(word)
    }

    pub(super) fn shape_hit(&mut self, object: LoweredValue, shape: u32) -> LoweredValue {
        let view = self.view(object);
        let actual = self
            .b
            .ins()
            .load(types::I32, MemFlagsData::trusted(), view, SHAPE_OFFSET);
        self.b.ins().icmp_imm_s(IntCC::Equal, actual, shape as i64)
    }

    pub(super) fn shaped_property_probe(
        &mut self,
        object: LoweredValue,
        shape: u32,
        index: u32,
    ) -> LoweredValue {
        let hit = self.shape_hit(object, shape);
        let fast = self.b.create_block();
        let miss = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param_repr(done, Repr::Boxed);
        self.brif(hit, fast, &[], miss, &[]);
        self.switch(fast);
        let value = self.shaped_property_get(object, index);
        self.jump(done, &[BlockArg::Value(value)]);
        self.switch(miss);
        let missing = self.constant(Value::UNINITIALIZED);
        self.jump(done, &[BlockArg::Value(missing)]);
        self.switch(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn shaped_property_get(&mut self, object: LoweredValue, index: u32) -> LoweredValue {
        let flags = MemFlagsData::trusted();
        let object = self.b.as_word(object);
        let view = layout::address(self.b, object);
        let data = self.b.ins().load(self.pointer, flags, view, DATA_OFFSET);
        let word = self
            .b
            .ins()
            .load(types::I64, flags, data, (index as i32) * 8);
        self.b.as_boxed(word)
    }

    pub(super) fn shaped_property_set(
        &mut self,
        object: LoweredValue,
        name: &str,
        _index: u32,
        value: LoweredValue,
    ) {
        self.cached_property_set(object, name, value, None);
    }

    fn cached_property_set(
        &mut self,
        object: LoweredValue,
        name: &str,
        value: LoweredValue,
        observer: Option<(u64, crate::jit::profile::Site)>,
    ) -> LoweredValue {
        let address = self.function_data(PropertyCache::new(name, observer, false));
        let result = self.call_helper(Helper::PropertySetCached, &[object, address, value]);
        self.check_error();
        result
    }

    pub(super) fn property_access(
        &mut self,
        object: LoweredValue,
        name: &str,
        value: Option<LoweredValue>,
        observer: Option<(u64, crate::jit::profile::Site)>,
    ) -> LoweredValue {
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
        object: LoweredValue,
        name: &str,
        value: LoweredValue,
    ) {
        let name: Box<str> = name.into();
        let ptr = self.b.ins().iconst(self.pointer, name.as_ptr() as i64);
        let len = self.b.ins().iconst(types::I64, name.len() as i64);
        self.names.push(name);
        self.call_helper(Helper::PropertyDefineHidden, &[object, ptr, len, value]);
        self.check_error();
    }

    pub(super) fn check_error(&mut self) {
        // Catch entry clears the exception before execution reaches another helper.
        let can_throw = self
            .b
            .current_block()
            .and_then(|_| self.b.last_helper())
            .is_none_or(|helper| helper.can_throw());
        let engine_error = self.load_ctx(crate::jit::session::ERROR_OFFSET);
        let condition = if can_throw {
            let exception = self.load_ctx(crate::jit::session::EXCEPTION_OFFSET);
            let empty = self
                .b
                .ins()
                .iconst(types::I64, Value::UNINITIALIZED.bits() as i64);
            let exception = self.b.ins().binary(Binary::Xor, exception, empty);
            self.b.ins().binary(Binary::Or, engine_error, exception)
        } else {
            engine_error
        };
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
                self.b.append_block_param(block, self.pointer);
                self.error_origin = Some(block);
                block
            }
        };
        let key = (
            self.source_name.0,
            self.source_span.start,
            self.source_span.end,
        );
        let source_name = self.source_name;
        let source = *self
            .source_names
            .entry(source_name.0)
            .or_insert_with(|| self.data.push(source_name));
        let offset = *self.error_sites.entry(key).or_insert_with(|| {
            self.data
                .push([source as u64, u64::from(key.1) | (u64::from(key.2) << 32)])
        });
        let site = self.data_address(offset);
        if can_throw {
            let target = match self.handlers.last().copied() {
                Some(handler) => handler,
                None => self.failed_exit(),
            };
            // Sharing the destination also lets cleanup share its release ladders.
            let dispatch = match self.error_dispatch.get(&target) {
                Some(&block) => block,
                None => {
                    let block = self.b.create_block();
                    self.b.set_cold_block(block);
                    self.b.append_block_param(block, self.pointer);
                    self.error_dispatch.insert(target, block);
                    block
                }
            };
            self.unwind(dispatch, &[site.into()]);
        } else {
            self.unwind(origin, &[site.into()]);
        }
        self.switch(done);
        self.unchecked = None;
    }
}

impl FunctionCompiler<'_, '_> {
    pub(super) fn stack_values(
        &mut self,
        values: &[LoweredValue],
        span: Span,
    ) -> Result<LoweredValue, Error> {
        if values.is_empty() {
            return Ok(self.b.ins().iconst(self.pointer, 0));
        }
        let size = values
            .len()
            .checked_mul(8)
            .filter(|size| *size <= i32::MAX as usize)
            .ok_or_else(|| compile_at(span, "too many stack values"))?;
        let slot = self.b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            size as u32,
            3,
        ));
        for (index, value) in values.iter().enumerate() {
            let address = self
                .b
                .ins()
                .stack_addr(self.pointer, slot, (index * 8) as i32);
            let word = self.b.as_word(*value);
            self.b
                .ins()
                .store(MemFlagsData::trusted(), word, address, 0);
        }
        Ok(self.b.ins().stack_addr(self.pointer, slot, 0))
    }
}
