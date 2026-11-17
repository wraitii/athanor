mod arithmetic;
mod calls;
mod generators;
mod property;

pub(crate) use arithmetic::*;
pub(crate) use calls::*;
pub(crate) use generators::generator_next;
use generators::*;
pub(in crate::jit) use property::*;

use cranelift_codegen::ir::{
    AbiParam, Inst, InstBuilder, MemFlagsData, Signature, Type, Value as ClifValue, types,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;

use crate::heap::{IndexRead, PropertyRead};
use crate::value::Value;

use super::osr::Site;
use super::session::{HELPERS_OFFSET, JitCtx};
use super::variants::Variant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HelperKind {
    Value,
    Integer,
    Float,
    Pointer,
}

pub(crate) trait HelperType {
    const KIND: HelperKind;
}

impl HelperType for Value {
    const KIND: HelperKind = HelperKind::Value;
}

impl HelperType for u64 {
    const KIND: HelperKind = HelperKind::Integer;
}

impl HelperType for usize {
    const KIND: HelperKind = HelperKind::Integer;
}

impl HelperType for f64 {
    const KIND: HelperKind = HelperKind::Float;
}

impl<T> HelperType for *const T {
    const KIND: HelperKind = HelperKind::Pointer;
}

impl<T> HelperType for *mut T {
    const KIND: HelperKind = HelperKind::Pointer;
}

pub(crate) struct HelperSpec {
    pub(crate) params: &'static [HelperKind],
    pub(crate) returns: HelperKind,
    pub(crate) pointer: *const u8,
    pub(super) ownership: super::semantic::Ownership,
}

macro_rules! define_helpers {
    (@ownership) => { super::semantic::Ownership::Unowned };
    (@ownership owned) => { super::semantic::Ownership::Owned };
    (@ownership borrowed) => { super::semantic::Ownership::Borrowed };
    ($($variant:ident => $function:ident($($param:ty),*) -> $result:ty $(, $ownership:ident)?;)*) => {
        #[repr(usize)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub(crate) enum Helper {
            $($variant),*
        }

        pub(crate) const HELPERS: &[HelperSpec] = &[
            $(HelperSpec {
                params: &[$(<$param as HelperType>::KIND),*],
                returns: <$result as HelperType>::KIND,
                pointer: {
                    const FUNCTION: extern "C" fn(*mut JitCtx $(, $param)*) -> $result = $function;
                    FUNCTION as *const () as *const u8
                },
                ownership: define_helpers!(@ownership $($ownership)?),
            }),*
        ];
    };
}

define_helpers! {
    PackInit => js_pack_init(Value) -> Value;
    PackRelease => js_pack_release(Value) -> Value;
    NewPack => js_new_pack(Value) -> Value, owned;
    Spread => js_spread(Value, Value) -> Value, owned;
    PackAppend => js_pack_append(Value, Value) -> Value;
    PackExtend => js_pack_extend(Value, Value) -> Value;
    PackTake => js_pack_take(Value, Value) -> Value, owned;
    CompilePoll => js_compile_poll() -> Value;
    VariantEntry => js_variant_entry(*const Variant) -> u64;
    OsrPoll => js_osr_poll(Value, *mut Value, *const Site) -> Value;
    CompileLeave => js_compile_leave(u64) -> Value;
    BaselineWarm => js_baseline_warm(u64) -> Value;
    Instanceof => js_instanceof(Value, Value) -> Value;
    In => js_in(Value, Value) -> Value;
    GeneratorNew => js_generator_new(Value, Value, Value, u64) -> Value, owned;
    CellNew => js_cell_new(Value) -> Value, owned;
    CellGet => js_cell_get(Value) -> Value, owned;
    CellSet => js_cell_set(Value, Value) -> Value;
    CaptureGet => js_capture_get(Value, u64) -> Value, owned;
    BindThis => js_bind_this(Value, Value) -> Value;
    DisableConstructor => js_disable_constructor(Value) -> Value;
    MarkClass => js_mark_class(Value) -> Value;
    ClosureNew => js_closure_new(u64, *const Value, u64) -> Value, owned;
    Construct => js_construct(Value, *const Value, u64) -> Value, owned;
    ConstructorPrototype => js_constructor_prototype(Value) -> Value, borrowed;
    NewInstance => js_new_instance(Value) -> Value, owned;
    ConstructorCall => js_constructor_call(Value, Value, *const Value, u64) -> Value, owned;
    ConstructorCallPack => js_constructor_call_pack(Value, Value, Value) -> Value, owned;
    ConstructResolved => js_construct_resolved(Value, Value, *const Value, u64) -> Value, owned;
    ConstructPackResolved => js_construct_pack_resolved(Value, Value, Value) -> Value, owned;
    ConstructPack => js_construct_pack(Value, Value) -> Value, owned;
    CallPack => js_call_pack(Value, Value, Value) -> Value, owned;
    ArrayAppend => js_array_append(Value, Value) -> Value;
    ArrayExtend => js_array_extend(Value, Value) -> Value;
    Call => js_call(Value, Value, *const Value, u64) -> Value, owned;
    ObserveCall => js_observe_call(Value, u64) -> Value;
    ObserveGet => js_observe_get(Value, u64) -> Value;
    ObserveIndex => js_observe_index(Value, u64) -> Value;
    ObservePattern => js_observe_pattern(Value, u64) -> Value;
    ObserveNumber => js_observe_number(Value, u64) -> Value;
    ObservePrimitive => js_observe_primitive(Value, u64) -> Value;
    ObservePrimitiveFailure => js_observe_primitive_failure(Value, u64) -> Value;
    ObserveNumberFailure => js_observe_number_failure(Value, u64) -> Value;
    SemanticGlobalProbe => js_semantic_global_probe(*mut GlobalCache) -> Value, borrowed;
    SemanticPropertyProbeCached => js_semantic_property_probe_cached(Value, *mut PropertyCache) -> Value, borrowed;
    SemanticIndexProbe => js_semantic_index_probe(Value, Value) -> Value, borrowed;
    SemanticMathNative => js_semantic_math_native(Value, u64) -> Value;
    ClosureThis => js_closure_this(Value) -> Value, borrowed;
    LexicalReceiver => js_lexical_receiver(Value, Value) -> Value, borrowed;
    ArrayIterationGuard => js_array_iteration_guard(Value, Value, u64) -> Value;
    IsStringSplit => js_is_string_split(Value) -> Value;
    StringSplitParts => js_string_split_parts(Value, Value, *const Value, u64) -> Value, owned;
    StringSplit => js_string_split(u64, Value, *const Value, u64) -> Value, owned;
    ArrayMapNew => js_array_map_new(Value) -> Value, owned;
    ArrayMapElement => js_array_map_element(Value, usize) -> Value, owned;
    MathDirect => js_math_direct(u64, Value, *const Value, u64) -> Value;
    FrameOverflow => js_frame_overflow() -> Value;
    FrameRelease => js_frame_release() -> Value;
    Release => js_release(Value) -> Value;
    Bailout => js_bailout(u64) -> Value;
    Retain => js_retain(Value) -> Value, owned;
    IterateNew => js_iterate_new(Value) -> Value, owned;
    IterateNext => js_iterate_next(Value) -> Value, owned;
    EnumerateNew => js_enumerate_new(Value) -> Value, owned;
    EnumerateNext => js_enumerate_next(Value) -> Value, owned;
    Typeof => js_typeof(Value) -> Value, owned;
    IsTypeof => js_is_typeof(Value, Value) -> Value;
    Truthy => js_truthy(Value) -> Value;
    ObjectNew => js_object_new(Value) -> Value, owned;
    PropertyKey => js_property_key(Value) -> Value, owned;
    StringConstant => js_string_constant(*mut u64, *const u8, u64) -> Value;
    RegExpNew => js_regexp_new(u64, *const u8, u64) -> Value, owned;
    RegExpExecResult => js_regexp_exec_result(Value, Value, u64, u64) -> Value, owned;
    ArrayNew => js_array_new(*const Value, u64) -> Value, owned;
    Rest => js_rest(*const Value, u64, u64) -> Value, owned;
    IndexGet => js_index_get(Value, Value) -> Value, owned;
    IndexGetNumeric => js_index_get_numeric(Value, f64) -> Value, owned;
    IndexSet => js_index_set(Value, Value, Value) -> Value;
    PropertySet => js_property_set(Value, *const u8, u64, Value) -> Value;
    PropertyGetCached => js_property_get_cached(Value, *mut PropertyCache) -> Value, borrowed;
    PropertyGetMegamorphic => js_property_get_megamorphic(Value, *mut PropertyCache) -> Value, borrowed;
    PropertyGetBaseline => js_property_get_baseline(Value, *mut PropertyCache) -> Value, owned;
    PropertyCachePrime => js_property_cache_prime(Value, *mut PropertyCache) -> Value;
    PropertySetCached => js_property_set_cached(Value, *mut PropertyCache, Value) -> Value;
    PropertyDefineHidden => js_property_define_hidden(Value, *const u8, u64, Value) -> Value;
    GlobalGetBorrowed => js_global_get_borrowed(*mut GlobalCache) -> Value, borrowed;
    GlobalGetOptionalBorrowed => js_global_get_optional_borrowed(*mut GlobalCache) -> Value, borrowed;
    GlobalSet => js_global_set(*const u8, u64, Value) -> Value;
    GlobalInitialize => js_global_initialize(*const u8, u64, Value) -> Value;
    Throw => js_throw(Value) -> Value;
    Catch => js_catch(Value) -> Value, owned;
    ErrorOrigin => js_error_origin(*const (usize, usize), u64, u64) -> Value;
    CallProfileFlush => js_call_profile_flush() -> Value;
    BitwiseAnd => js_bitwise_and(Value, Value) -> Value;
    BitwiseOr => js_bitwise_or(Value, Value) -> Value;
    BitwiseXor => js_bitwise_xor(Value, Value) -> Value;
    ShiftLeft => js_shift_left(Value, Value) -> Value;
    ShiftRight => js_shift_right(Value, Value) -> Value;
    ShiftRightUnsigned => js_shift_right_unsigned(Value, Value) -> Value;
    Add => js_add(Value, Value) -> Value, owned;
    StringConcat => js_string_concat(*const Value, u64) -> Value, owned;
    HeapAllocationSite => js_heap_allocation_site(u64) -> Value;
    Sub => js_sub(Value, Value) -> Value;
    Mul => js_mul(Value, Value) -> Value;
    Div => js_div(Value, Value) -> Value;
    Rem => js_rem(Value, Value) -> Value;
    Pow => js_pow(Value, Value) -> Value;
    LooseEqual => js_loose_equal(Value, Value) -> Value;
    LooseNotEqual => js_loose_not_equal(Value, Value) -> Value;
    StrictEqual => js_strict_equal(Value, Value) -> Value;
    StrictNotEqual => js_strict_not_equal(Value, Value) -> Value;
    Less => js_less(Value, Value) -> Value;
    LessEqual => js_less_equal(Value, Value) -> Value;
    Greater => js_greater(Value, Value) -> Value;
    GreaterEqual => js_greater_equal(Value, Value) -> Value;
    UnaryPlus => js_unary_plus(Value) -> Value;
    UnaryNegation => js_unary_negation(Value) -> Value;
    LogicalNot => js_logical_not(Value) -> Value;
    BitwiseNot => js_bitwise_not(Value) -> Value;
}

extern "C" fn js_variant_entry(ctx: *mut JitCtx, variant: *const super::variants::Variant) -> u64 {
    unsafe { &*ctx }
        .engine
        .tiering
        .variant_entry(unsafe { &*variant }) as u64
}

extern "C" fn js_compile_poll(ctx: *mut JitCtx) -> Value {
    unsafe { &*ctx }.engine.tiering.poll_compilation();
    Value::UNDEFINED
}

extern "C" fn js_osr_poll(
    ctx: *mut JitCtx,
    receiver: Value,
    buffer: *mut Value,
    site: *const super::osr::Site,
) -> Value {
    let site = unsafe { &*site };
    let entry = unsafe { &*ctx }.engine.tiering.osr_entry(site);
    if entry == 0 {
        return Value::number(-1.0);
    }
    unsafe { &*ctx }.engine.tiering.osr_entered(site);
    let transfer = site.plan.loops[site.candidate].transfer.as_ref().unwrap();
    let function: unsafe extern "C" fn(*mut JitCtx, Value, Value, *const Value, u64) -> Value =
        unsafe { std::mem::transmute(entry) };
    unsafe {
        function(
            ctx,
            Value::UNDEFINED,
            receiver,
            buffer,
            transfer.arguments() as u64,
        )
    }
}

extern "C" fn js_baseline_warm(ctx: *mut JitCtx, code: u64) -> Value {
    unsafe { &*ctx }.engine.tiering.baseline_warmed(code);
    Value::UNDEFINED
}

extern "C" fn js_compile_leave(ctx: *mut JitCtx, code: u64) -> Value {
    unsafe { &*ctx }.engine.tiering.leave_function(code);
    Value::UNDEFINED
}

impl Helper {
    pub(super) fn result_ownership(self) -> super::semantic::Ownership {
        HELPERS[self as usize].ownership
    }

    pub(crate) fn can_throw(self) -> bool {
        // These helpers may report engine errors, but never execute JavaScript or throw.
        !matches!(
            self,
            Self::Retain
                | Self::StringConstant
                | Self::CompilePoll
                | Self::CompileLeave
                | Self::BaselineWarm
                | Self::LexicalReceiver
                | Self::Release
                | Self::FrameOverflow
                | Self::FrameRelease
                | Self::IndexGet
                | Self::IndexGetNumeric
                | Self::PropertyGetCached
                | Self::PropertyGetMegamorphic
                | Self::GlobalGetBorrowed
                | Self::GlobalGetOptionalBorrowed
                | Self::BitwiseAnd
                | Self::BitwiseOr
                | Self::BitwiseXor
                | Self::ShiftLeft
                | Self::ShiftRight
                | Self::ShiftRightUnsigned
                | Self::Add
                | Self::Sub
                | Self::Mul
                | Self::Div
                | Self::Rem
                | Self::Pow
                | Self::LooseEqual
                | Self::LooseNotEqual
                | Self::StrictEqual
                | Self::StrictNotEqual
                | Self::Less
                | Self::LessEqual
                | Self::Greater
                | Self::GreaterEqual
                | Self::UnaryPlus
                | Self::UnaryNegation
                | Self::LogicalNot
                | Self::BitwiseNot
        )
    }
}

pub(crate) fn address<'f>(
    ins: impl InstBuilder<'f>,
    pointer: Type,
    ctx: ClifValue,
    helper: Helper,
) -> ClifValue {
    let offset = HELPERS_OFFSET + helper as i32 * std::mem::size_of::<usize>() as i32;
    ins.load(pointer, MemFlagsData::trusted(), ctx, offset)
}

pub(crate) fn call(
    b: &mut FunctionBuilder<'_>,
    call_conv: CallConv,
    pointer: Type,
    ctx: ClifValue,
    helper: Helper,
    args: &[ClifValue],
) -> Inst {
    let signature = b.import_signature(signature(helper, call_conv, pointer));
    let callee = address(b.ins(), pointer, ctx, helper);
    let mut call_args = Vec::with_capacity(args.len() + 1);
    call_args.push(ctx);
    call_args.extend_from_slice(args);
    b.ins().call_indirect(signature, callee, &call_args)
}

fn abi(kind: HelperKind, pointer: Type) -> AbiParam {
    AbiParam::new(match kind {
        HelperKind::Value | HelperKind::Integer => types::I64,
        HelperKind::Float => types::F64,
        HelperKind::Pointer => pointer,
    })
}

pub(crate) fn signature(helper: Helper, call_conv: CallConv, pointer: Type) -> Signature {
    let spec = &HELPERS[helper as usize];
    let mut signature = Signature::new(call_conv);
    signature.params.push(AbiParam::new(pointer));
    signature
        .params
        .extend(spec.params.iter().map(|&kind| abi(kind, pointer)));
    signature.returns.push(abi(spec.returns, pointer));
    signature
}

extern "C" fn js_error_origin(
    ctx: *mut JitCtx,
    source: *const (usize, usize),
    start: u64,
    end: u64,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    if ctx.error_origin.is_none() && matches!(ctx.error.as_deref(), Some(crate::Error::Runtime(_)))
    {
        let (ptr, len) = unsafe { *source };
        let filename = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr as *const u8, len))
        };
        ctx.error_origin = Some((filename.to_owned(), start as u32, end as u32));
    }
    Value::UNDEFINED
}

extern "C" fn js_call_profile_flush(ctx: *mut JitCtx) -> Value {
    unsafe { &mut *ctx }.flush_call_samples();
    Value::UNDEFINED
}

extern "C" fn js_frame_overflow(ctx: *mut JitCtx) -> Value {
    unsafe { &mut *ctx }.overflow();
    Value::UNDEFINED
}
extern "C" fn js_frame_release(ctx: *mut JitCtx) -> Value {
    unsafe { &mut *ctx }.release_frame_locals();
    Value::UNDEFINED
}
extern "C" fn js_bailout(_: *mut JitCtx, site: u64) -> Value {
    crate::jit::bailouts::hit(site);
    Value::UNDEFINED
}
extern "C" fn js_release(ctx: *mut JitCtx, value: Value) -> Value {
    unsafe { &*ctx }.global.heap().release(value);
    Value::UNDEFINED
}
extern "C" fn js_retain(ctx: *mut JitCtx, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if value == Value::UNINITIALIZED {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "binding accessed before initialization".into(),
        )));
        return Value::UNDEFINED;
    }
    ctx.retain(value)
}
extern "C" fn js_heap_allocation_site(ctx: *mut JitCtx, site: u64) -> Value {
    unsafe { &*ctx }.global.heap().allocation_site(site);
    Value::UNDEFINED
}

extern "C" fn js_array_new(ctx: *mut JitCtx, argv: *const Value, argc: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let values = if argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(argv, argc as usize) }
    };
    match ctx.global.heap().array(values) {
        Some(value) => value,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "invalid array elements or heap exhausted".into(),
            )));
            Value::UNDEFINED
        }
    }
}
extern "C" fn js_rest(ctx: *mut JitCtx, argv: *const Value, argc: u64, start: u64) -> Value {
    let start = start.min(argc);
    let ptr = if start == argc {
        std::ptr::null()
    } else {
        unsafe { argv.add(start as usize) }
    };
    js_array_new(ctx, ptr, argc - start)
}
extern "C" fn js_cell_new(ctx: *mut JitCtx, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().cell(value) {
        Some(cell) => cell,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "invalid cell value or heap exhausted".into(),
            )));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_cell_get(ctx: *mut JitCtx, cell: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().cell_get(cell) {
        Some(Value::UNINITIALIZED) => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "binding accessed before initialization".into(),
            )));
            Value::UNDEFINED
        }
        Some(value) => ctx.retain(value),
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "invalid capture cell".into(),
            )));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_cell_set(ctx: *mut JitCtx, cell: Value, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if !ctx.global.heap().cell_set(cell, value) {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "invalid capture cell or value".into(),
        )));
    }
    value
}

extern "C" fn js_capture_get(ctx: *mut JitCtx, closure: Value, index: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().capture(closure, index as usize) {
        Some(cell) => ctx.retain(cell),
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "invalid closure capture".into(),
            )));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_closure_new(
    ctx: *mut JitCtx,
    code: u64,
    captures: *const Value,
    count: u64,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    if !ctx.has_code(code) {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "unknown closure code".into(),
        )));
        return Value::UNDEFINED;
    }
    let captures = if count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(captures, count as usize) }
    };
    match ctx
        .global
        .heap()
        .closure(crate::heap::CodeId(code), captures)
    {
        Some(value) => value,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "invalid captures or heap exhausted".into(),
            )));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_array_append(ctx: *mut JitCtx, arguments: Value, value: Value) -> Value {
    let context = unsafe { &mut *ctx };
    let heap = context.global.heap();
    if !heap.set(arguments, heap.len(arguments).unwrap(), value) {
        context.error = Some(Box::new(crate::Error::Runtime(
            "cannot append array element".into(),
        )));
    }
    Value::UNDEFINED
}

extern "C" fn js_array_extend(ctx: *mut JitCtx, array: Value, pack: Value) -> Value {
    let values = unsafe { super::pack::take(ctx, pack) };
    let context = unsafe { &mut *ctx };
    if !context
        .global
        .heap()
        .extend_array_values(array, &values.values)
    {
        context.error = Some(Box::new(crate::Error::Runtime(
            "cannot extend array".into(),
        )));
    }
    Value::UNDEFINED
}

extern "C" fn js_semantic_math_native(ctx: *mut JitCtx, value: Value, operation: u64) -> Value {
    let expected = super::math::METHODS[operation as usize].1;
    Value::boolean(
        unsafe { &*ctx }.global.heap().native(value)
            == Some(crate::heap::NativeFunction::Math(expected)),
    )
}

extern "C" fn js_is_string_split(ctx: *mut JitCtx, value: Value) -> Value {
    Value::boolean(
        unsafe { &*ctx }.global.heap().native(value)
            == Some(crate::heap::NativeFunction::StringSplit),
    )
}

extern "C" fn js_string_split(
    ctx: *mut JitCtx,
    _: u64,
    receiver: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    let args = if argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(argv, argc as usize) }
    };
    crate::builtins::string::split(ctx, receiver, args)
}

extern "C" fn js_closure_this(ctx: *mut JitCtx, closure: Value) -> Value {
    unsafe { &*ctx }
        .global
        .heap()
        .lexical_this(closure)
        .unwrap_or(Value::UNDEFINED)
}

extern "C" fn js_lexical_receiver(ctx: *mut JitCtx, closure: Value, receiver: Value) -> Value {
    unsafe { &*ctx }
        .global
        .heap()
        .lexical_this(closure)
        .unwrap_or(receiver)
}

extern "C" fn js_array_iteration_guard(
    ctx: *mut JitCtx,
    callee: Value,
    receiver: Value,
    method: u64,
) -> Value {
    use super::semantic::ArrayMethod;
    use crate::heap::NativeFunction;
    let native = match method {
        x if x == ArrayMethod::Map as u64 => NativeFunction::ArrayMap,
        x if x == ArrayMethod::Some as u64 => NativeFunction::ArraySome,
        x if x == ArrayMethod::Every as u64 => NativeFunction::ArrayEvery,
        x if x == ArrayMethod::Filter as u64 => NativeFunction::ArrayFilter,
        x if x == ArrayMethod::ForEach as u64 => NativeFunction::ArrayForEach,
        x if x == ArrayMethod::Find as u64 => NativeFunction::ArrayFind,
        x if x == ArrayMethod::Push as u64 => NativeFunction::ArrayPush,
        _ => unreachable!(),
    };
    let heap = unsafe { &*ctx }.global.heap();
    Value::boolean(heap.native(callee) == Some(native) && heap.is_array(receiver))
}

extern "C" fn js_array_map_new(ctx: *mut JitCtx, length: Value) -> Value {
    let context = unsafe { &mut *ctx };
    match context
        .global
        .heap()
        .array_of_length(length.as_number().unwrap() as usize)
    {
        Some(value) => value,
        None => {
            context.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_array_map_element(ctx: *mut JitCtx, receiver: Value, index: usize) -> Value {
    let context = unsafe { &mut *ctx };
    if !context.global.heap().present(receiver, index) {
        return Value::UNINITIALIZED;
    }
    let value = context.global.heap().get(receiver, index).unwrap();
    context.retain(value)
}

extern "C" fn js_math_direct(
    ctx: *mut JitCtx,
    operation: u64,
    _: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    let operation = super::math::METHODS[operation as usize].1;
    let args = if argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(argv, argc as usize) }
    };
    crate::builtins::math::call(unsafe { &mut *ctx }, operation, args)
}

extern "C" fn js_observe_get(ctx: *mut JitCtx, value: Value, cache: u64) -> Value {
    observe_read(ctx, value, cache, super::profile::ReadKind::Result)
}

extern "C" fn js_observe_index(ctx: *mut JitCtx, value: Value, cache: u64) -> Value {
    observe_read(ctx, value, cache, super::profile::ReadKind::Receiver)
}

extern "C" fn js_observe_pattern(ctx: *mut JitCtx, value: Value, cache: u64) -> Value {
    observe_read(ctx, value, cache, super::profile::ReadKind::Pattern)
}

extern "C" fn js_observe_number(ctx: *mut JitCtx, value: Value, cache: u64) -> Value {
    observe_read(ctx, value, cache, super::profile::ReadKind::Number)
}

extern "C" fn js_observe_primitive(ctx: *mut JitCtx, value: Value, cache: u64) -> Value {
    observe_read(ctx, value, cache, super::profile::ReadKind::Primitive)
}

fn primitive_type(value: Value) -> super::profile::ReadType {
    use super::profile::ReadType;
    if value.is_string() {
        return ReadType::String;
    }
    let Some(number) = value.as_number() else {
        return ReadType::Unknown;
    };
    if number.abs() <= 9007199254740991.0
        && number.trunc() == number
        && !(number == 0.0 && number.is_sign_negative())
    {
        ReadType::Integer
    } else {
        ReadType::Number
    }
}

extern "C" fn js_observe_primitive_failure(ctx: *mut JitCtx, value: Value, cache: u64) -> Value {
    let context = unsafe { &*ctx };
    let cache = unsafe { &mut *(cache as *mut super::profile::ObservationCache) };
    if cache.complete == 0 {
        context
            .engine
            .tiering
            .observe_primitive_failure(cache, primitive_type(value));
    }
    Value::UNDEFINED
}

extern "C" fn js_observe_number_failure(ctx: *mut JitCtx, value: Value, cache: u64) -> Value {
    let context = unsafe { &*ctx };
    let cache = unsafe { &mut *(cache as *mut super::profile::ObservationCache) };
    if cache.complete != 0 {
        return Value::UNDEFINED;
    }
    let observed = if value.as_number().is_some() {
        super::profile::ReadType::Number
    } else {
        cache.failures += 1;
        if cache.failures < super::profile::NUMBER_FAILURE_LIMIT {
            return Value::UNDEFINED;
        }
        super::profile::ReadType::Unknown
    };
    context
        .engine
        .tiering
        .observe_number_failure(cache, observed);
    Value::UNDEFINED
}

fn observe_read(
    ctx: *mut JitCtx,
    value: Value,
    cache: u64,
    kind: super::profile::ReadKind,
) -> Value {
    let context = unsafe { &*ctx };
    let cache = unsafe { &mut *(cache as *mut super::profile::ObservationCache) };
    if cache.complete == 0 {
        let heap = context.global.heap();
        let observed = if kind == super::profile::ReadKind::Primitive {
            Some(primitive_type(value))
        } else if kind == super::profile::ReadKind::Number {
            value.as_number().map(|number| {
                if number.abs() <= 9007199254740991.0
                    && number.trunc() == number
                    && !(number == 0.0 && number.is_sign_negative())
                {
                    super::profile::ReadType::Integer
                } else {
                    super::profile::ReadType::Number
                }
            })
        } else if kind == super::profile::ReadKind::Pattern {
            heap.regexp_compiled(value)
                .map(|pattern| super::profile::ReadType::Pattern(pattern.reference()))
        } else if heap.is_array(value) {
            Some(super::profile::ReadType::Array)
        } else {
            heap.typed_kind(value).map(super::profile::ReadType::Typed)
        };
        if let Some(observed) = observed {
            context.engine.tiering.observe_get(cache, observed, kind);
        } else {
            context
                .engine
                .tiering
                .observe_get(cache, super::profile::ReadType::Unknown, kind);
        }
    }
    Value::UNDEFINED
}

extern "C" fn js_observe_call(ctx: *mut JitCtx, callee: Value, cache: u64) -> Value {
    let context = unsafe { &*ctx };
    let cache = unsafe { &mut *(cache as *mut super::profile::ObservationCache) };
    if cache.complete == 0 {
        cache.last_callee = callee.bits();
        if let Some(code) = context.global.heap().closure_code(callee) {
            context.engine.tiering.observe_call(cache, code.0);
        } else if let Some(native) = context.global.heap().native(callee) {
            context.engine.tiering.observe_native(cache, native);
        }
    }
    Value::UNDEFINED
}

extern "C" fn js_regexp_new(ctx: *mut JitCtx, flags: u64, ptr: *const u8, len: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let pattern =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    match ctx.global.heap().regexp(pattern, flags as u8) {
        Ok(value) => {
            if let Some(pattern) = ctx.global.heap().regexp_compiled(value) {
                ctx.engine.tiering.prepare_regex(&pattern);
            }
            value
        }
        Err(message) => {
            ctx.error = Some(Box::new(crate::Error::Runtime(format!(
                "invalid regular expression: {message}"
            ))));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_regexp_exec_result(
    ctx: *mut JitCtx,
    regexp: Value,
    input: Value,
    start: u64,
    end: u64,
) -> Value {
    crate::builtins::regexp_exec_result(
        unsafe { &mut *ctx },
        regexp,
        input,
        start as usize..end as usize,
    )
}

/// Creates a string literal's value on its first execution. The cache keeps the reference for
/// the lifetime of the compiled code, so the result is borrowed.
extern "C" fn js_string_constant(
    ctx: *mut JitCtx,
    cache: *mut u64,
    ptr: *const u8,
    len: u64,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let text =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    match ctx.global.heap().string(text) {
        Some(value) => {
            unsafe { *cache = value.bits() };
            value
        }
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "string heap exhausted".into(),
            )));
            Value::UNDEFINED
        }
    }
}

pub(crate) extern "C" fn js_truthy(ctx: *mut JitCtx, value: Value) -> Value {
    let ctx = unsafe { &*ctx };
    Value::boolean(match ctx.global.heap().string_is_empty(value) {
        Some(empty) => !empty,
        None => value.to_boolean(),
    })
}

extern "C" fn js_throw(ctx: *mut JitCtx, value: Value) -> Value {
    unsafe { &mut *ctx }.throw(value);
    Value::UNDEFINED
}

extern "C" fn js_catch(ctx: *mut JitCtx, _: Value) -> Value {
    unsafe { &mut *ctx }.catch()
}

extern "C" fn js_in(ctx: *mut JitCtx, key: Value, object: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if !object.is_object() {
        ctx.error = Some(Box::new(crate::Error::Runtime(format!(
            "right operand of in must be an object, got {object}"
        ))));
        return Value::UNDEFINED;
    }
    let Some(name) = property_key(ctx, key) else {
        return Value::UNDEFINED;
    };
    if object == Value::GLOBAL_OBJECT {
        return Value::boolean(ctx.global.get_object_raw(&name).is_some());
    }
    match ctx.global.heap().has_property(object, &name) {
        Some(present) => Value::boolean(present),
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "unsupported right operand of in".into(),
            )));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_instanceof(ctx: *mut JitCtx, value: Value, constructor: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let heap = ctx.global.heap();
    if let Some(crate::heap::NativeFunction::TypedConstructor(kind)) = heap.native(constructor) {
        return Value::boolean(heap.typed_kind(value) == Some(kind));
    }
    if heap.native(constructor) == Some(crate::heap::NativeFunction::Array) {
        return Value::boolean(heap.is_array(value));
    }
    if heap.closure_code(constructor).is_none()
        && heap.native(constructor) != Some(crate::heap::NativeFunction::Object)
    {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "instanceof requires a supported callable right operand".into(),
        )));
        return Value::UNDEFINED;
    }
    if !value.is_object() {
        return Value::FALSE;
    }
    let prototype = heap
        .property(constructor, "prototype")
        .unwrap_or(Value::UNDEFINED);
    if !prototype.is_object() {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "instanceof constructor prototype is not an object".into(),
        )));
        return Value::UNDEFINED;
    }
    Value::boolean(heap.inherits(value, prototype))
}

extern "C" fn js_is_typeof(ctx: *mut JitCtx, value: Value, kind: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let callable = ctx.global.heap().closure_code(value).is_some()
        || ctx.global.heap().native(value).is_some();
    let name = if callable {
        "function"
    } else {
        value.type_name()
    };
    let expected = match kind.as_number().unwrap() as u8 {
        2 => "boolean",
        3 => "undefined",
        4 => "object",
        5 => "function",
        _ => unreachable!(),
    };
    Value::boolean(name == expected)
}

extern "C" fn js_typeof(ctx: *mut JitCtx, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if value == Value::UNINITIALIZED {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "binding accessed before initialization".into(),
        )));
        return Value::UNDEFINED;
    }
    let name = if ctx.global.heap().closure_code(value).is_some()
        || ctx.global.heap().native(value).is_some()
    {
        "function"
    } else {
        value.type_name()
    };
    match ctx.global.heap().interned(name) {
        Some(value) => value,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_enumerate_new(ctx: *mut JitCtx, receiver: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().enumerate(receiver) {
        Some(state) => state,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "for-in receiver is unsupported".into(),
            )));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_enumerate_next(ctx: *mut JitCtx, state: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().enumerate_next(state) {
        Some(Some(value)) => value,
        Some(None) => Value::UNINITIALIZED,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "for-in enumeration failed".into(),
            )));
            Value::UNDEFINED
        }
    }
}

pub(crate) extern "C" fn js_iterate_new(ctx: *mut JitCtx, iterable: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if ctx.global.heap().len(iterable).is_none() {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "iteration requires an array, typed array, or string".into(),
        )));
        return Value::UNDEFINED;
    }
    match ctx.global.heap().array(&[iterable, Value::number(0.0)]) {
        Some(value) => value,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

pub(crate) extern "C" fn js_iterate_next(ctx: *mut JitCtx, state: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let heap = ctx.global.heap();
    let iterable = heap.get(state, 0).unwrap();
    let offset = heap.get(state, 1).unwrap().as_number().unwrap() as usize;
    if iterable.is_string() {
        let Some((character, next)) = heap.next_character(iterable, offset) else {
            return Value::UNINITIALIZED;
        };
        heap.set(state, 1, Value::number(next as f64));
        return match heap.string(character.encode_utf8(&mut [0; 4])) {
            Some(value) => value,
            None => {
                ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
                Value::UNDEFINED
            }
        };
    }
    if offset >= heap.len(iterable).unwrap() {
        return Value::UNINITIALIZED;
    }
    let value = heap.get(iterable, offset).unwrap();
    heap.set(state, 1, Value::number((offset + 1) as f64));
    ctx.retain(value)
}

extern "C" fn js_object_new(ctx: *mut JitCtx, _: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().object() {
        Some(value) => value,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_bind_this(ctx: *mut JitCtx, closure: Value, receiver: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if !ctx.global.heap().bind_this(closure, receiver) {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "invalid arrow receiver".into(),
        )));
    }
    closure
}

extern "C" fn js_disable_constructor(ctx: *mut JitCtx, closure: Value) -> Value {
    unsafe { &*ctx }.global.heap().disable_constructor(closure);
    closure
}

extern "C" fn js_mark_class(ctx: *mut JitCtx, closure: Value) -> Value {
    unsafe { &*ctx }.global.heap().mark_class(closure);
    closure
}

extern "C" fn js_new_pack(ctx: *mut JitCtx, destination: Value) -> Value {
    let mut values = unsafe { super::pack::buffer(ctx) };
    match unsafe { super::pack::store(ctx, destination, std::mem::take(&mut values.values)) } {
        Some(value) => value,
        None => {
            unsafe { &mut *ctx }.error =
                Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_spread(ctx: *mut JitCtx, destination: Value, value: Value) -> Value {
    let Some(mut values) = calls::argument_values(ctx, value) else {
        return Value::UNDEFINED;
    };
    match unsafe { super::pack::store(ctx, destination, std::mem::take(&mut values.values)) } {
        Some(value) => value,
        None => {
            unsafe { &mut *ctx }.error =
                Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_pack_append(ctx: *mut JitCtx, pack: Value, value: Value) -> Value {
    let heap = unsafe { &*ctx }.global.heap();
    heap.retain(value);
    unsafe {
        super::pack::push(heap, pack, value);
    }
    Value::UNDEFINED
}

extern "C" fn js_pack_extend(ctx: *mut JitCtx, pack: Value, value: Value) -> Value {
    let mut values = unsafe { super::pack::take(ctx, value) };
    unsafe {
        super::pack::append((&*ctx).global.heap(), pack, &mut values.values);
    }
    Value::UNDEFINED
}

extern "C" fn js_pack_take(ctx: *mut JitCtx, pack: Value, index: Value) -> Value {
    unsafe {
        super::pack::take_element(
            (&*ctx).global.heap(),
            pack,
            index.as_number().unwrap() as usize,
        )
    }
}

extern "C" fn js_string_split_parts(
    ctx: *mut JitCtx,
    destination: Value,
    receiver: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    let args = if argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(argv, argc as usize) }
    };
    crate::builtins::string::split_result(ctx, receiver, args, destination)
}

extern "C" fn js_pack_init(_: *mut JitCtx, destination: Value) -> Value {
    unsafe {
        super::pack::initialize(destination);
    }
    Value::UNDEFINED
}

extern "C" fn js_pack_release(ctx: *mut JitCtx, pack: Value) -> Value {
    drop(unsafe { super::pack::take(ctx, pack) });
    Value::UNDEFINED
}
