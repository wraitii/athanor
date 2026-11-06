mod arithmetic;
mod calls;
mod generators;
mod property;

pub(in crate::jit) use arithmetic::*;
pub(in crate::jit) use calls::*;
use generators::*;
pub(in crate::jit) use property::*;

use cranelift_codegen::ir::{AbiParam, Signature, Type, types};
use cranelift_codegen::isa::CallConv;

use crate::heap::{IndexRead, PropertyRead};
use crate::value::Value;

use super::session::JitCtx;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HelperShape {
    Nullary,
    Binary,
    NumericIndex,
    Unary,
    Pointer,
    GlobalGet,
    GlobalSet,
    PropertyGet,
    PropertySet,
    PropertySetCached,
    ArrayNew,
    Rest,
    IndexSet,
    Callable,
    MethodCallable,
    Quaternary,
    SourceLocation,
}

pub(crate) struct HelperSpec {
    pub(crate) name: &'static str,
    pub(crate) shape: HelperShape,
    pub(crate) pointer: *const u8,
}

macro_rules! define_helpers {
    ($($variant:ident => $shape:ident, $function:ident;)*) => {
        #[repr(usize)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum Helper {
            $($variant),*
        }

        pub(crate) const HELPERS: &[HelperSpec] = &[
            $(HelperSpec {
                name: stringify!($function),
                shape: HelperShape::$shape,
                pointer: $function as *const () as *const u8,
            }),*
        ];
    };
}

define_helpers! {
    Instanceof => Binary, js_instanceof;
    In => Binary, js_in;
    GeneratorNew => Quaternary, js_generator_new;
    GeneratorClosure => Unary, js_generator_closure;
    GeneratorState => Nullary, js_generator_state;
    GeneratorSuspend => IndexSet, js_generator_suspend;
    GeneratorComplete => Binary, js_generator_complete;
    LocalGet => Unary, js_local_get;
    CellNew => Unary, js_cell_new;
    CellGet => Unary, js_cell_get;
    CellSet => Binary, js_cell_set;
    CaptureGet => Binary, js_capture_get;
    BindThis => Binary, js_bind_this;
    DisableConstructor => Unary, js_disable_constructor;
    MarkClass => Unary, js_mark_class;
    ClosureNew => Callable, js_closure_new;
    Construct => Callable, js_construct;
    ConstructorPrototype => Unary, js_constructor_prototype;
    NewInstance => Unary, js_new_instance;
    ConstructorCall => MethodCallable, js_constructor_call;
    ConstructorCallSpread => IndexSet, js_constructor_call_spread;
    ConstructResolved => MethodCallable, js_construct_resolved;
    ConstructSpreadResolved => IndexSet, js_construct_spread_resolved;
    ConstructSpread => Binary, js_construct_spread;
    CallSpread => IndexSet, js_call_spread;
    ArrayAppend => Binary, js_array_append;
    ArrayExtend => Binary, js_array_extend;
    Call => MethodCallable, js_call;
    ObserveCall => Binary, js_observe_call;
    ObserveGet => Binary, js_observe_get;
    ObserveIndex => Binary, js_observe_index;
    SemanticGlobalProbe => GlobalGet, js_semantic_global_probe;
    SemanticPropertyProbeCached => Binary, js_semantic_property_probe_cached;
    SemanticIndexProbe => Binary, js_semantic_index_probe;
    SemanticMathNative => Binary, js_semantic_math_native;
    ClosureThis => Unary, js_closure_this;
    LexicalReceiver => Binary, js_lexical_receiver;
    ArrayIterationGuard => IndexSet, js_array_iteration_guard;
    IsStringSplit => Unary, js_is_string_split;
    StringSplit => MethodCallable, js_string_split;
    ArrayMapNew => Unary, js_array_map_new;
    ArrayMapElement => Binary, js_array_map_element;
    MathDirect => MethodCallable, js_math_direct;
    FrameOverflow => Nullary, js_frame_overflow;
    FrameRelease => Nullary, js_frame_release;
    IndexUnroot => Nullary, js_index_unroot;
    LocalSet => Binary, js_local_set;
    LocalReplace => IndexSet, js_local_replace;
    Release => Unary, js_release;
    TempClear => Unary, js_temp_clear;
    TempRetain => Unary, js_temp_retain;
    TempOwn => Unary, js_temp_own;
    IterateNew => Unary, js_iterate_new;
    IterateNext => Unary, js_iterate_next;
    EnumerateNew => Unary, js_enumerate_new;
    EnumerateNext => Unary, js_enumerate_next;
    Typeof => Unary, js_typeof;
    IsTypeof => Binary, js_is_typeof;
    Truthy => Unary, js_truthy;
    ObjectNew => Unary, js_object_new;
    PropertyKey => Unary, js_property_key;
    StringNew => ArrayNew, js_string_new;
    RegExpNew => PropertyGet, js_regexp_new;
    ArrayNew => ArrayNew, js_array_new;
    Rest => Rest, js_rest;
    IndexGet => Binary, js_index_get;
    IndexGetNumeric => NumericIndex, js_index_get_numeric;
    IndexRoot => IndexSet, js_index_root;
    IndexSet => IndexSet, js_index_set;
    PropertyGet => PropertyGet, js_property_get;
    PropertyGetCached => Binary, js_property_get_cached;
    PropertySetCached => PropertySetCached, js_property_set_cached;
    PropertyDefineHidden => PropertySet, js_property_define_hidden;
    GlobalGet => Pointer, js_global_get;
    GlobalGetOptional => Pointer, js_global_get_optional;
    GlobalSet => GlobalSet, js_global_set;
    GlobalInitialize => GlobalSet, js_global_initialize;
    Throw => Unary, js_throw;
    Catch => Unary, js_catch;
    ErrorOrigin => SourceLocation, js_error_origin;
    BitwiseAnd => Binary, js_bitwise_and;
    BitwiseOr => Binary, js_bitwise_or;
    BitwiseXor => Binary, js_bitwise_xor;
    ShiftLeft => Binary, js_shift_left;
    ShiftRight => Binary, js_shift_right;
    ShiftRightUnsigned => Binary, js_shift_right_unsigned;
    Add => Binary, js_add;
    Sub => Binary, js_sub;
    Mul => Binary, js_mul;
    Div => Binary, js_div;
    Rem => Binary, js_rem;
    Pow => Binary, js_pow;
    LooseEqual => Binary, js_loose_equal;
    LooseNotEqual => Binary, js_loose_not_equal;
    StrictEqual => Binary, js_strict_equal;
    StrictNotEqual => Binary, js_strict_not_equal;
    Less => Binary, js_less;
    LessEqual => Binary, js_less_equal;
    Greater => Binary, js_greater;
    GreaterEqual => Binary, js_greater_equal;
    UnaryPlus => Unary, js_unary_plus;
    UnaryNegation => Unary, js_unary_negation;
    LogicalNot => Unary, js_logical_not;
    BitwiseNot => Unary, js_bitwise_not;
}

impl Helper {
    pub(crate) fn can_throw(self) -> bool {
        // These helpers may report engine errors, but never execute JavaScript or throw.
        !matches!(
            self,
            Self::TempRetain
                | Self::TempOwn
                | Self::LexicalReceiver
                | Self::Release
                | Self::FrameOverflow
                | Self::FrameRelease
                | Self::IndexUnroot
                | Self::IndexGet
                | Self::IndexGetNumeric
                | Self::PropertyGet
                | Self::PropertyGetCached
                | Self::GlobalGet
                | Self::GlobalGetOptional
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

pub(crate) fn symbols() -> impl Iterator<Item = (&'static str, *const u8)> {
    HELPERS.iter().map(|spec| (spec.name, spec.pointer))
}

pub(crate) fn signature(shape: HelperShape, call_conv: CallConv, pointer: Type) -> Signature {
    let mut signature = Signature::new(call_conv);
    signature.params.push(AbiParam::new(pointer));
    match shape {
        HelperShape::MethodCallable => {
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(pointer));
            signature.params.push(AbiParam::new(types::I64));
        }
        HelperShape::Callable => {
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(pointer));
            signature.params.push(AbiParam::new(types::I64));
        }
        HelperShape::NumericIndex => {
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(types::F64));
        }
        HelperShape::Binary => {
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(types::I64));
        }
        HelperShape::Nullary => {}
        HelperShape::Unary => signature.params.push(AbiParam::new(types::I64)),
        HelperShape::ArrayNew | HelperShape::Rest => {
            signature.params.push(AbiParam::new(pointer));
            signature.params.push(AbiParam::new(types::I64));
            if shape == HelperShape::Rest {
                signature.params.push(AbiParam::new(types::I64));
            }
        }
        HelperShape::SourceLocation => {
            signature.params.push(AbiParam::new(pointer));
            for _ in 0..3 {
                signature.params.push(AbiParam::new(types::I64));
            }
        }
        HelperShape::Quaternary => {
            for _ in 0..4 {
                signature.params.push(AbiParam::new(types::I64));
            }
        }
        HelperShape::PropertySetCached => {
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(pointer));
            signature.params.push(AbiParam::new(types::I64));
        }
        HelperShape::IndexSet => {
            for _ in 0..3 {
                signature.params.push(AbiParam::new(types::I64));
            }
        }
        HelperShape::Pointer => signature.params.push(AbiParam::new(pointer)),
        HelperShape::GlobalGet
        | HelperShape::GlobalSet
        | HelperShape::PropertyGet
        | HelperShape::PropertySet => {
            if matches!(shape, HelperShape::PropertyGet | HelperShape::PropertySet) {
                signature.params.push(AbiParam::new(types::I64));
            }
            signature.params.push(AbiParam::new(pointer));
            signature.params.push(AbiParam::new(types::I64));
            if matches!(shape, HelperShape::GlobalSet | HelperShape::PropertySet) {
                signature.params.push(AbiParam::new(types::I64));
            }
        }
    }
    signature.returns.push(AbiParam::new(types::I64));
    signature
}

extern "C" fn js_error_origin(
    ctx: *mut JitCtx,
    ptr: *const u8,
    len: u64,
    start: u64,
    end: u64,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    if ctx.error_origin.is_none() && matches!(ctx.error.as_deref(), Some(crate::Error::Runtime(_)))
    {
        let filename =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
        ctx.error_origin = Some((filename.to_owned(), start as u32, end as u32));
    }
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
extern "C" fn js_index_unroot(ctx: *mut JitCtx) -> Value {
    unsafe { &mut *ctx }.release_index_root();
    Value::UNDEFINED
}
extern "C" fn js_local_set(ctx: *mut JitCtx, slot: u64, value: Value) -> Value {
    unsafe { &mut *ctx }.local_set(slot as usize, value)
}
extern "C" fn js_release(ctx: *mut JitCtx, value: Value) -> Value {
    unsafe { &*ctx }.global.heap().release(value);
    Value::UNDEFINED
}
extern "C" fn js_local_replace(ctx: *mut JitCtx, slot: u64, value: Value, old: Value) -> Value {
    unsafe { &mut *ctx }.local_replace(slot as usize, value, old)
}
extern "C" fn js_temp_clear(ctx: *mut JitCtx, _: Value) -> Value {
    unsafe { &mut *ctx }.clear();
    Value::UNDEFINED
}
extern "C" fn js_temp_own(ctx: *mut JitCtx, value: Value) -> Value {
    unsafe { &mut *ctx }.own(value)
}
extern "C" fn js_temp_retain(ctx: *mut JitCtx, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if value == Value::UNINITIALIZED {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "binding accessed before initialization".into(),
        )));
        return Value::UNDEFINED;
    }
    ctx.retain_temp(value)
}
extern "C" fn js_array_new(ctx: *mut JitCtx, argv: *const Value, argc: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let values = if argc == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(argv, argc as usize) }
    };
    match ctx.global.heap().array(values) {
        Some(value) => ctx.own(value),
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
        Some(cell) => ctx.own(cell),
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
        Some(value) => ctx.retain_temp(value),
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
        Some(cell) => ctx.retain_temp(cell),
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
    if ctx.entry(code).is_none() {
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
        Some(value) => ctx.own(value),
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

extern "C" fn js_array_extend(ctx: *mut JitCtx, arguments: Value, iterable: Value) -> Value {
    if let Some(extended) = unsafe { &*ctx }
        .global
        .heap()
        .extend_array(arguments, iterable)
    {
        if !extended {
            unsafe { &mut *ctx }.error = Some(Box::new(crate::Error::Runtime(
                "cannot extend array".into(),
            )));
        }
        return Value::UNDEFINED;
    }
    let state = js_iterate_new(ctx, iterable);
    if unsafe { &*ctx }.error.is_some() {
        return Value::UNDEFINED;
    }
    loop {
        let value = js_iterate_next(ctx, state);
        if unsafe { &*ctx }.error.is_some() || value == Value::UNINITIALIZED {
            break;
        }
        js_array_append(ctx, arguments, value);
        if unsafe { &*ctx }.error.is_some() {
            break;
        }
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
    super::builtins::string::call(
        ctx,
        crate::heap::NativeFunction::StringSplit,
        receiver,
        args,
    )
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
        Some(value) => context.own(value),
        None => {
            context.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_array_map_element(ctx: *mut JitCtx, receiver: Value, index: Value) -> Value {
    let context = unsafe { &mut *ctx };
    let index = index.as_number().unwrap() as usize;
    if !context.global.heap().present(receiver, index) {
        return Value::UNINITIALIZED;
    }
    let value = context.global.heap().get(receiver, index).unwrap();
    context.retain_temp(value)
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
    super::builtins::math::call(unsafe { &mut *ctx }, operation, args)
}

extern "C" fn js_observe_get(ctx: *mut JitCtx, value: Value, cache: u64) -> Value {
    observe_read(ctx, value, cache, super::planner::ReadKind::Result)
}

extern "C" fn js_observe_index(ctx: *mut JitCtx, value: Value, cache: u64) -> Value {
    observe_read(ctx, value, cache, super::planner::ReadKind::Receiver)
}

fn observe_read(
    ctx: *mut JitCtx,
    value: Value,
    cache: u64,
    kind: super::planner::ReadKind,
) -> Value {
    let context = unsafe { &*ctx };
    let cache = unsafe { &mut *(cache as *mut super::planner::ObservationCache) };
    if cache.recorded == 0 {
        let heap = context.global.heap();
        let observed = if heap.is_array(value) {
            Some(super::planner::ReadType::Array)
        } else {
            heap.typed_kind(value).map(super::planner::ReadType::Typed)
        };
        if let Some(observed) = observed {
            context.engine.observe_get(cache, observed, kind);
        } else {
            cache.recorded = 1;
        }
    }
    Value::UNDEFINED
}

extern "C" fn js_observe_call(ctx: *mut JitCtx, callee: Value, cache: u64) -> Value {
    let context = unsafe { &*ctx };
    let cache = unsafe { &mut *(cache as *mut super::planner::ObservationCache) };
    if cache.recorded == 0 {
        if let Some(code) = context.global.heap().closure_code(callee) {
            context.engine.observe_call(cache, code.0);
        } else {
            cache.recorded = 1;
        }
    }
    Value::UNDEFINED
}

extern "C" fn js_regexp_new(ctx: *mut JitCtx, flags: u64, ptr: *const u8, len: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let pattern =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    match ctx.global.heap().regexp(pattern, flags as u8) {
        Ok(value) => ctx.own(value),
        Err(message) => {
            ctx.error = Some(Box::new(crate::Error::Runtime(format!(
                "invalid regular expression: {message}"
            ))));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_string_new(ctx: *mut JitCtx, ptr: *const u8, len: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let text =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    match ctx.global.heap().string(text) {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "string heap exhausted".into(),
            )));
            Value::UNDEFINED
        }
    }
}

pub(super) extern "C" fn js_truthy(ctx: *mut JitCtx, value: Value) -> Value {
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
    if heap.closure_code(constructor).is_none() {
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
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_enumerate_new(ctx: *mut JitCtx, receiver: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().enumerate(receiver) {
        Some(state) => ctx.own(state),
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
        Some(Some(value)) => ctx.own(value),
        Some(None) => Value::UNINITIALIZED,
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "for-in enumeration failed".into(),
            )));
            Value::UNDEFINED
        }
    }
}

pub(super) extern "C" fn js_iterate_new(ctx: *mut JitCtx, iterable: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if ctx.global.heap().len(iterable).is_none() {
        ctx.error = Some(Box::new(crate::Error::Runtime(
            "iteration requires an array, typed array, or string".into(),
        )));
        return Value::UNDEFINED;
    }
    match ctx.global.heap().array(&[iterable, Value::number(0.0)]) {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(Box::new(crate::Error::Runtime("heap exhausted".into())));
            Value::UNDEFINED
        }
    }
}

pub(super) extern "C" fn js_iterate_next(ctx: *mut JitCtx, state: Value) -> Value {
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
            Some(value) => ctx.own(value),
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
    ctx.retain_temp(value)
}

extern "C" fn js_object_new(ctx: *mut JitCtx, _: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().object() {
        Some(value) => ctx.own(value),
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
