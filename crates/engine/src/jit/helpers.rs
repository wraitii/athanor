use cranelift_codegen::ir::{AbiParam, Signature, Type, types};
use cranelift_codegen::isa::CallConv;

use crate::heap::{IndexRead, PropertyRead};
use crate::value::Value;

use super::numeric::Kind;
use super::session::JitCtx;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HelperShape {
    Binary,
    NumericIndex,
    Unary,
    GlobalGet,
    GlobalSet,
    PropertyGet,
    PropertySet,
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
    GeneratorState => Unary, js_generator_state;
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
    ConstructSpread => Binary, js_construct_spread;
    CallSpread => IndexSet, js_call_spread;
    ArrayAppend => Binary, js_array_append;
    ArrayExtend => Binary, js_array_extend;
    Call => MethodCallable, js_call;
    SemanticGlobalProbe => GlobalGet, js_semantic_global_probe;
    SemanticPropertyProbe => PropertyGet, js_semantic_property_probe;
    SemanticMathNative => Binary, js_semantic_math_native;
    MathDirect => MethodCallable, js_math_direct;
    FrameEnter => Unary, js_frame_enter;
    FrameExit => Unary, js_frame_exit;
    LocalSet => Binary, js_local_set;
    TempClear => Unary, js_temp_clear;
    TempRetain => Unary, js_temp_retain;
    IterateNew => Unary, js_iterate_new;
    IterateNext => Unary, js_iterate_next;
    EnumerateNew => Unary, js_enumerate_new;
    EnumerateNext => Unary, js_enumerate_next;
    Typeof => Unary, js_typeof;
    Truthy => Unary, js_truthy;
    ObjectNew => Unary, js_object_new;
    PropertyKey => Unary, js_property_key;
    StringNew => ArrayNew, js_string_new;
    RegExpNew => PropertyGet, js_regexp_new;
    ArrayNew => ArrayNew, js_array_new;
    Rest => Rest, js_rest;
    IndexGet => Binary, js_index_get;
    IndexGetNumeric => NumericIndex, js_index_get_numeric;
    IndexRoot => Quaternary, js_index_root;
    IndexProbe => Binary, js_index_probe;
    IndexSet => IndexSet, js_index_set;
    PropertyGet => PropertyGet, js_property_get;
    PropertySet => PropertySet, js_property_set;
    PropertyDefineHidden => PropertySet, js_property_define_hidden;
    GlobalGet => GlobalGet, js_global_get;
    GlobalGetOptional => GlobalGet, js_global_get_optional;
    GlobalSet => GlobalSet, js_global_set;
    GlobalInitialize => GlobalSet, js_global_initialize;
    Throw => Unary, js_throw;
    Catch => Unary, js_catch;
    ExceptionPending => Unary, js_exception_pending;
    ErrorPending => Unary, js_error_pending;
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
    pub(super) fn result_kind_on_success(self) -> Option<Kind> {
        match self {
            Self::BitwiseAnd
            | Self::BitwiseOr
            | Self::BitwiseXor
            | Self::ShiftLeft
            | Self::ShiftRight
            | Self::BitwiseNot => Some(Kind::Int32),
            Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Rem
            | Self::Pow
            | Self::ShiftRightUnsigned
            | Self::UnaryPlus
            | Self::UnaryNegation => Some(Kind::Number),
            _ => None,
        }
    }

    pub(crate) fn can_throw(self) -> bool {
        // These helpers may report engine errors, but never execute JavaScript or throw.
        !matches!(
            self,
            Self::TempRetain
                | Self::IndexGet
                | Self::IndexGetNumeric
                | Self::PropertyGet
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
        HelperShape::IndexSet => {
            for _ in 0..3 {
                signature.params.push(AbiParam::new(types::I64));
            }
        }
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

fn primitive_number(ctx: *mut JitCtx, value: Value) -> f64 {
    if value.is_string() {
        let heap = unsafe { &*ctx }.global.heap();
        return heap
            .text(value)
            .map(|text| super::conversion::string_number(&text))
            .unwrap_or(f64::NAN);
    }
    value.to_number()
}

fn numeric_operands(ctx: *mut JitCtx, values: &[Value]) -> bool {
    if values.iter().any(|value| value.is_object()) {
        unsafe { &mut *ctx }.error = Some(crate::Error::Runtime(
            "object-to-primitive conversion is unsupported".into(),
        ));
        return false;
    }
    true
}

fn number_binary(
    ctx: *mut JitCtx,
    left: Value,
    right: Value,
    operation: impl FnOnce(f64, f64) -> f64,
) -> Value {
    if !numeric_operands(ctx, &[left, right]) {
        return Value::UNDEFINED;
    }
    Value::number(operation(
        primitive_number(ctx, left),
        primitive_number(ctx, right),
    ))
}

fn number_compare(
    ctx: *mut JitCtx,
    left: Value,
    right: Value,
    operation: impl FnOnce(f64, f64) -> bool,
) -> Value {
    if left.is_string() && right.is_string() {
        let heap = unsafe { &*ctx }.global.heap();
        let left = heap.text(left).unwrap();
        let right = heap.text(right).unwrap();
        let order = match left.cmp(&right) {
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
            std::cmp::Ordering::Greater => 1.0,
        };
        return Value::boolean(operation(order, 0.0));
    }
    if !numeric_operands(ctx, &[left, right]) {
        return Value::UNDEFINED;
    }
    Value::boolean(operation(
        primitive_number(ctx, left),
        primitive_number(ctx, right),
    ))
}

pub(super) fn to_int32(number: f64) -> i32 {
    if !number.is_finite() || number == 0.0 {
        return 0;
    }
    let number = number.trunc().rem_euclid(4_294_967_296.0);
    if number >= 2_147_483_648.0 {
        (number - 4_294_967_296.0) as i32
    } else {
        number as i32
    }
}

fn loose_equals(ctx: *mut JitCtx, left: Value, right: Value) -> bool {
    if left.is_string() && right.is_string() {
        return strict_equals(unsafe { &*ctx }.global.heap(), left, right);
    }
    if left.is_object() != right.is_object()
        && !left.is_nullish()
        && !right.is_nullish()
        && !numeric_operands(ctx, &[left, right])
    {
        return false;
    }
    if left.is_number() && right.is_number() {
        return left.as_number() == right.as_number();
    }
    if left.is_nullish() || right.is_nullish() {
        return left.is_nullish() && right.is_nullish();
    }
    if !left.is_number() && !right.is_number() {
        if left.bits() == right.bits() {
            return true;
        }
        if left.is_object() || right.is_object() {
            return false;
        }
    }
    primitive_number(ctx, left) == primitive_number(ctx, right)
}

pub(super) fn strict_equals(heap: &crate::heap::Heap, left: Value, right: Value) -> bool {
    if left.is_string() && right.is_string() {
        return heap.text(left) == heap.text(right);
    }
    if left.is_number() && right.is_number() {
        return left.as_number() == right.as_number();
    }
    if left.is_number() != right.is_number() {
        return false;
    }
    left.bits() == right.bits()
}

pub(crate) extern "C" fn js_add(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    if left.is_string() || right.is_string() {
        let ctx = unsafe { &mut *ctx };
        if (left.is_object() && ctx.global.heap().error_text(left).is_none())
            || (right.is_object() && ctx.global.heap().error_text(right).is_none())
        {
            ctx.error = Some(crate::Error::Runtime(
                "object-to-primitive conversion is unsupported".into(),
            ));
            return Value::UNDEFINED;
        }
        let heap = ctx.global.heap();
        let left = heap
            .text(left)
            .or_else(|| heap.error_text(left))
            .unwrap_or_else(|| left.to_string());
        let right = heap
            .text(right)
            .or_else(|| heap.error_text(right))
            .unwrap_or_else(|| right.to_string());
        return match heap.string(&(left + &right)) {
            Some(value) => ctx.own(value),
            None => {
                ctx.error = Some(crate::Error::Runtime("string heap exhausted".into()));
                Value::UNDEFINED
            }
        };
    }
    number_binary(ctx, left, right, |left, right| left + right)
}

pub(crate) extern "C" fn js_sub(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    number_binary(ctx, left, right, |left, right| left - right)
}

pub(crate) extern "C" fn js_mul(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    number_binary(ctx, left, right, |left, right| left * right)
}

pub(crate) extern "C" fn js_div(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    number_binary(ctx, left, right, |left, right| left / right)
}

pub(crate) extern "C" fn js_rem(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    number_binary(ctx, left, right, |left, right| left % right)
}

pub(crate) extern "C" fn js_pow(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    number_binary(ctx, left, right, f64::powf)
}

pub(crate) extern "C" fn js_loose_equal(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    Value::boolean(loose_equals(ctx, left, right))
}

pub(crate) extern "C" fn js_loose_not_equal(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    Value::boolean(!loose_equals(ctx, left, right))
}

pub(crate) extern "C" fn js_strict_equal(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    Value::boolean(strict_equals(unsafe { &*ctx }.global.heap(), left, right))
}

pub(crate) extern "C" fn js_strict_not_equal(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    Value::boolean(!strict_equals(unsafe { &*ctx }.global.heap(), left, right))
}

pub(crate) extern "C" fn js_less(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    number_compare(ctx, left, right, |left, right| left < right)
}

pub(crate) extern "C" fn js_less_equal(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    number_compare(ctx, left, right, |left, right| left <= right)
}

pub(crate) extern "C" fn js_greater(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    number_compare(ctx, left, right, |left, right| left > right)
}

pub(crate) extern "C" fn js_greater_equal(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    number_compare(ctx, left, right, |left, right| left >= right)
}

pub(crate) extern "C" fn js_unary_plus(ctx: *mut JitCtx, value: Value) -> Value {
    if !numeric_operands(ctx, &[value]) {
        return Value::UNDEFINED;
    }
    Value::number(primitive_number(ctx, value))
}

pub(crate) extern "C" fn js_unary_negation(ctx: *mut JitCtx, value: Value) -> Value {
    if !numeric_operands(ctx, &[value]) {
        return Value::UNDEFINED;
    }
    Value::number(-primitive_number(ctx, value))
}

pub(crate) extern "C" fn js_logical_not(ctx: *mut JitCtx, value: Value) -> Value {
    Value::boolean(js_truthy(ctx, value) == Value::FALSE)
}

pub(crate) extern "C" fn js_bitwise_not(ctx: *mut JitCtx, value: Value) -> Value {
    if !numeric_operands(ctx, &[value]) {
        return Value::UNDEFINED;
    }
    let value = primitive_number(ctx, value);
    Value::number(f64::from(!to_int32(value)))
}

extern "C" fn js_global_get(ctx: *mut JitCtx, ptr: *const u8, len: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    match ctx.global.get_raw(name) {
        Some(Value::UNINITIALIZED) => {
            ctx.error = Some(crate::Error::Runtime(format!(
                "binding `{name}` accessed before initialization"
            )));
            Value::UNDEFINED
        }
        Some(value) => ctx.retain_temp(value),
        None => {
            ctx.error = Some(crate::Error::Runtime(format!("unknown variable `{name}`")));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_global_get_optional(ctx: *mut JitCtx, ptr: *const u8, len: u64) -> Value {
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if unsafe { &*ctx }.global.get_raw(name).is_none() {
        return Value::UNDEFINED;
    }
    js_global_get(ctx, ptr, len)
}

extern "C" fn js_global_set(ctx: *mut JitCtx, ptr: *const u8, len: u64, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if let Err(reason) = ctx.global.set_binding(name, value) {
        ctx.error = Some(crate::Error::Runtime(format!("binding `{name}` {reason}")));
    }
    value
}

extern "C" fn js_global_initialize(
    ctx: *mut JitCtx,
    ptr: *const u8,
    len: u64,
    value: Value,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if !ctx.global.initialize_lexical(name, value) {
        ctx.error = Some(crate::Error::Runtime(format!(
            "cannot initialize binding `{name}`"
        )));
    }
    value
}

extern "C" fn js_error_origin(
    ctx: *mut JitCtx,
    ptr: *const u8,
    len: u64,
    start: u64,
    end: u64,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    if ctx.error_origin.is_none() && matches!(ctx.error, Some(crate::Error::Runtime(_))) {
        let filename =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
        ctx.error_origin = Some((filename.to_owned(), start as u32, end as u32));
    }
    Value::UNDEFINED
}

extern "C" fn js_error_pending(ctx: *mut JitCtx, _unused: Value) -> Value {
    Value::boolean(unsafe { &*ctx }.error.is_some())
}

extern "C" fn js_property_get(ctx: *mut JitCtx, object: Value, ptr: *const u8, len: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if object == Value::GLOBAL_OBJECT {
        let value = ctx.global.get_object_raw(name).unwrap_or(Value::UNDEFINED);
        return ctx.retain_temp(value);
    }
    match ctx.global.heap().read_property(object, name) {
        Some(PropertyRead::Value(value)) => return ctx.retain_temp(value),
        Some(PropertyRead::Native(function)) => {
            return match ctx.global.heap().native_function(function) {
                Some(value) => ctx.own(value),
                None => {
                    ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
                    Value::UNDEFINED
                }
            };
        }
        Some(PropertyRead::Prototype) => {
            if let Some(value) = ctx.global.heap().property(object, name) {
                return ctx.retain_temp(value);
            }
        }
        None => {}
    }
    ctx.error = Some(crate::Error::Runtime(format!(
        "cannot read property {name:?} of {}",
        ctx.global.heap().receiver_kind(object)
    )));
    Value::UNDEFINED
}

extern "C" fn js_property_set(
    ctx: *mut JitCtx,
    object: Value,
    ptr: *const u8,
    len: u64,
    value: Value,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if object != Value::GLOBAL_OBJECT {
        if name == "lastIndex" && ctx.global.heap().regexp_state(object).is_some() {
            if let Some(index) = value.as_number() {
                ctx.global.heap().regexp_set_last_index(object, index);
            } else {
                ctx.error = Some(crate::Error::Runtime(
                    "RegExp lastIndex requires a number".into(),
                ));
            }
            return value;
        }
        if ctx.global.heap().set_property(object, name, value) {
            return value;
        }
        if name == "length"
            && array_index(value).is_some_and(|length| ctx.global.heap().truncate(object, length))
        {
            return value;
        }
        ctx.error = Some(crate::Error::Runtime(format!(
            "cannot assign property {name:?} of {}",
            ctx.global.heap().receiver_kind(object)
        )));
        return Value::UNDEFINED;
    }
    if !ctx.global.set(name, value) {
        ctx.global.define(name, value);
    }
    value
}

extern "C" fn js_property_define_hidden(
    ctx: *mut JitCtx,
    object: Value,
    ptr: *const u8,
    len: u64,
    value: Value,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if !ctx
        .global
        .heap()
        .define_property(object, name, value, false)
    {
        ctx.error = Some(crate::Error::Runtime("cannot define property".into()));
    }
    value
}

extern "C" fn js_frame_enter(ctx: *mut JitCtx, _: Value) -> Value {
    unsafe { &mut *ctx }.enter();
    Value::UNDEFINED
}
extern "C" fn js_frame_exit(ctx: *mut JitCtx, value: Value) -> Value {
    unsafe { &mut *ctx }.exit(value)
}
extern "C" fn js_local_set(ctx: *mut JitCtx, slot: u64, value: Value) -> Value {
    unsafe { &mut *ctx }.local_set(slot as usize, value)
}
extern "C" fn js_temp_clear(ctx: *mut JitCtx, _: Value) -> Value {
    unsafe { &mut *ctx }.clear();
    Value::UNDEFINED
}
extern "C" fn js_temp_retain(ctx: *mut JitCtx, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if value == Value::UNINITIALIZED {
        ctx.error = Some(crate::Error::Runtime(
            "binding accessed before initialization".into(),
        ));
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
            ctx.error = Some(crate::Error::Runtime(
                "invalid array elements or heap exhausted".into(),
            ));
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
fn array_index(value: Value) -> Option<usize> {
    let number = value.as_number()?;
    (number.is_finite() && number >= 0.0 && number.fract() == 0.0 && number < u32::MAX as f64)
        .then_some(number as usize)
}
fn typed_numeric_index(ctx: &JitCtx, key: Value) -> Option<Option<usize>> {
    let number = if let Some(number) = key.as_number() {
        number
    } else {
        let text = ctx.global.heap().text(key)?;
        if text == "-0" {
            return Some(None);
        }
        let number = super::conversion::string_number(&text);
        let canonical = if number == 0.0 {
            "0".to_owned()
        } else {
            dragonbox_ecma::Buffer::new().format(number).to_owned()
        };
        if text != canonical {
            return None;
        }
        number
    };
    Some(
        if number.is_finite()
            && number >= 0.0
            && number.fract() == 0.0
            && number < usize::MAX as f64
        {
            Some(number as usize)
        } else {
            None
        },
    )
}

extern "C" fn js_index_root(
    ctx: *mut JitCtx,
    slot: u64,
    receiver: Value,
    key: Value,
    result: Value,
) -> Value {
    Value::boolean(unsafe { &mut *ctx }.root_index_read(slot as usize, receiver, key, result))
}

extern "C" fn js_index_probe(ctx: *mut JitCtx, receiver: Value, key: Value) -> Value {
    Value::boolean(unsafe { &*ctx }.global.heap().supports_indexed_storage(
        receiver,
        key,
        crate::heap::IndexedStorageContract::DirectNumericIndexStorage,
    ))
}

extern "C" fn js_index_get(ctx: *mut JitCtx, array: Value, index: Value) -> Value {
    if let Some(number) = index.as_number() {
        return js_index_get_numeric(ctx, array, number);
    }
    index_get_slow(unsafe { &mut *ctx }, array, index)
}

extern "C" fn js_index_get_numeric(ctx: *mut JitCtx, array: Value, index: f64) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().read_numeric_index(array, index) {
        Some(IndexRead::Primitive(value)) => value,
        Some(IndexRead::Borrowed(value)) => ctx.retain_temp(value),
        Some(IndexRead::Character(character)) => index_get_character(ctx, character),
        None => index_get_slow(ctx, array, Value::number(index)),
    }
}

#[cold]
#[inline(never)]
fn index_get_character(ctx: &mut JitCtx, character: char) -> Value {
    match ctx.global.heap().string(character.encode_utf8(&mut [0; 4])) {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(crate::Error::Runtime("string heap exhausted".into()));
            Value::UNDEFINED
        }
    }
}

#[cold]
#[inline(never)]
fn index_get_slow(ctx: &mut JitCtx, array: Value, index: Value) -> Value {
    if ctx.global.heap().typed_kind(array).is_some() {
        if let Some(index) = typed_numeric_index(ctx, index) {
            return index
                .and_then(|index| ctx.global.heap().get(array, index))
                .unwrap_or(Value::UNDEFINED);
        }
        let Some(name) = property_key(ctx, index) else {
            return Value::UNDEFINED;
        };
        return js_property_get(ctx, array, name.as_ptr(), name.len() as u64);
    }
    let index = if index.is_string()
        && (ctx.global.heap().is_array(array) || array.is_string() || array.is_number())
    {
        let name = ctx.global.heap().text(index).unwrap();
        match name
            .parse::<u32>()
            .ok()
            .filter(|index| *index != u32::MAX && index.to_string() == name)
        {
            Some(index) => Value::number(index as f64),
            None => return js_property_get(ctx, array, name.as_ptr(), name.len() as u64),
        }
    } else {
        index
    };
    if ctx.global.heap().regexp_state(array).is_some() {
        let Some(name) = property_key(ctx, index) else {
            return Value::UNDEFINED;
        };
        if name == "lastIndex" {
            return Value::number(ctx.global.heap().regexp_state(array).unwrap().1);
        }
        if let Some(value) = super::builtins::property_get(ctx, array, &name) {
            return value;
        }
        ctx.error = Some(crate::Error::Runtime("unsupported RegExp property".into()));
        return Value::UNDEFINED;
    }
    if ctx.global.heap().generator_state(array).is_some() {
        let Some(name) = property_key(ctx, index) else {
            return Value::UNDEFINED;
        };
        return generator_property(ctx, &name);
    }
    if ctx.global.heap().set_size(array).is_some() {
        let Some(name) = property_key(ctx, index) else {
            return Value::UNDEFINED;
        };
        return set_property_get(ctx, array, &name).unwrap();
    }
    if ctx.global.heap().has_properties(array) || array == Value::GLOBAL_OBJECT {
        let Some(name) = property_key(ctx, index) else {
            return Value::UNDEFINED;
        };
        let value = if array == Value::GLOBAL_OBJECT {
            ctx.global.get_object_raw(&name).unwrap_or(Value::UNDEFINED)
        } else {
            ctx.global.heap().property(array, &name).unwrap()
        };
        return ctx.retain_temp(value);
    }
    if let Some(text) = ctx.global.heap().text(array) {
        let Some(index) = array_index(index) else {
            return Value::UNDEFINED;
        };
        let Some(character) = text.chars().nth(index) else {
            return Value::UNDEFINED;
        };
        return match ctx.global.heap().string(character.encode_utf8(&mut [0; 4])) {
            Some(value) => ctx.own(value),
            None => {
                ctx.error = Some(crate::Error::Runtime("string heap exhausted".into()));
                Value::UNDEFINED
            }
        };
    }
    let result = array_index(index).and_then(|index| ctx.global.heap().get(array, index));
    match result {
        Some(value) => ctx.retain_temp(value),
        None => {
            ctx.error = Some(crate::Error::Runtime(format!(
                "unsupported index or receiver: {}",
                ctx.global.heap().receiver_kind(array)
            )));
            Value::UNDEFINED
        }
    }
}
extern "C" fn js_index_set(ctx: *mut JitCtx, array: Value, index: Value, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if ctx.global.heap().typed_kind(array).is_some() {
        if let Some(index) = typed_numeric_index(ctx, index) {
            let Some(number) = super::builtins::typed_array::numeric(ctx, value) else {
                return Value::UNDEFINED;
            };
            if let Some(index) = index {
                ctx.global.heap().typed_set(array, index, number);
            }
            return value;
        }
        let Some(name) = property_key(ctx, index) else {
            return Value::UNDEFINED;
        };
        return js_property_set(ctx, array, name.as_ptr(), name.len() as u64, value);
    }
    if ctx.global.heap().regexp_state(array).is_some() {
        let Some(name) = property_key(ctx, index) else {
            return Value::UNDEFINED;
        };
        if name == "lastIndex" {
            if let Some(index) = value.as_number() {
                ctx.global.heap().regexp_set_last_index(array, index);
            } else {
                ctx.error = Some(crate::Error::Runtime(
                    "RegExp lastIndex requires a number".into(),
                ));
            }
        } else {
            ctx.error = Some(crate::Error::Runtime("unsupported RegExp property".into()));
        }
        return value;
    }
    if ctx.global.heap().has_properties(array) || array == Value::GLOBAL_OBJECT {
        let Some(name) = property_key(ctx, index) else {
            return Value::UNDEFINED;
        };
        if array == Value::GLOBAL_OBJECT {
            if !ctx.global.set(&name, value) {
                ctx.global.define(name, value);
            }
        } else {
            ctx.global.heap().set_property(array, &name, value);
        }
        return value;
    }
    if !array_index(index).is_some_and(|index| ctx.global.heap().set(array, index, value)) {
        ctx.error = Some(crate::Error::Runtime(
            "unsupported array index or receiver".into(),
        ));
    }
    value
}

extern "C" fn js_cell_new(ctx: *mut JitCtx, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().cell(value) {
        Some(cell) => ctx.own(cell),
        None => {
            ctx.error = Some(crate::Error::Runtime(
                "invalid cell value or heap exhausted".into(),
            ));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_cell_get(ctx: *mut JitCtx, cell: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().cell_get(cell) {
        Some(Value::UNINITIALIZED) => {
            ctx.error = Some(crate::Error::Runtime(
                "binding accessed before initialization".into(),
            ));
            Value::UNDEFINED
        }
        Some(value) => ctx.retain_temp(value),
        None => {
            ctx.error = Some(crate::Error::Runtime("invalid capture cell".into()));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_cell_set(ctx: *mut JitCtx, cell: Value, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if !ctx.global.heap().cell_set(cell, value) {
        ctx.error = Some(crate::Error::Runtime(
            "invalid capture cell or value".into(),
        ));
    }
    value
}

extern "C" fn js_capture_get(ctx: *mut JitCtx, closure: Value, index: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().capture(closure, index as usize) {
        Some(cell) => ctx.retain_temp(cell),
        None => {
            ctx.error = Some(crate::Error::Runtime("invalid closure capture".into()));
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
        ctx.error = Some(crate::Error::Runtime("unknown closure code".into()));
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
            ctx.error = Some(crate::Error::Runtime(
                "invalid captures or heap exhausted".into(),
            ));
            Value::UNDEFINED
        }
    }
}

fn argument_values(ctx: *mut JitCtx, arguments: Value) -> Vec<Value> {
    let heap = unsafe { &*ctx }.global.heap();
    (0..heap.len(arguments).unwrap())
        .map(|index| heap.get(arguments, index).unwrap())
        .collect()
}

extern "C" fn js_call_spread(
    ctx: *mut JitCtx,
    callable: Value,
    receiver: Value,
    arguments: Value,
) -> Value {
    let values = argument_values(ctx, arguments);
    js_call(
        ctx,
        callable,
        receiver,
        values.as_ptr(),
        values.len() as u64,
    )
}

extern "C" fn js_construct_spread(ctx: *mut JitCtx, callable: Value, arguments: Value) -> Value {
    let values = argument_values(ctx, arguments);
    js_construct(ctx, callable, values.as_ptr(), values.len() as u64)
}

extern "C" fn js_array_append(ctx: *mut JitCtx, arguments: Value, value: Value) -> Value {
    let context = unsafe { &mut *ctx };
    let heap = context.global.heap();
    if !heap.set(arguments, heap.len(arguments).unwrap(), value) {
        context.error = Some(crate::Error::Runtime("cannot append array element".into()));
    }
    Value::UNDEFINED
}

extern "C" fn js_array_extend(ctx: *mut JitCtx, arguments: Value, iterable: Value) -> Value {
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

extern "C" fn js_semantic_global_probe(ctx: *mut JitCtx, ptr: *const u8, len: u64) -> Value {
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    match unsafe { &*ctx }.global.get_raw(name) {
        Some(value) if value != Value::UNINITIALIZED => value,
        _ => Value::UNDEFINED,
    }
}

extern "C" fn js_semantic_property_probe(
    ctx: *mut JitCtx,
    object: Value,
    ptr: *const u8,
    len: u64,
) -> Value {
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if object == Value::GLOBAL_OBJECT {
        return Value::UNDEFINED;
    }
    match unsafe { &*ctx }.global.heap().read_property(object, name) {
        Some(PropertyRead::Value(value)) => value,
        _ => Value::UNDEFINED,
    }
}

extern "C" fn js_semantic_math_native(ctx: *mut JitCtx, value: Value, operation: u64) -> Value {
    let expected = super::math::METHODS[operation as usize].1;
    Value::boolean(
        unsafe { &*ctx }.global.heap().native(value)
            == Some(crate::heap::NativeFunction::Math(expected)),
    )
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

pub(super) extern "C" fn js_call(
    ctx: *mut JitCtx,
    closure: Value,
    receiver: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    if unsafe { &*ctx }.global.heap().is_class(closure) {
        unsafe { &mut *ctx }.error = Some(crate::Error::Runtime(
            "class constructor requires new".into(),
        ));
        return Value::UNDEFINED;
    }
    invoke(ctx, closure, receiver, argv, argc)
}

fn invoke(
    ctx: *mut JitCtx,
    closure: Value,
    receiver: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    if let Some(function) = unsafe { &*ctx }.global.heap().native(closure) {
        return native_call(ctx, function, receiver, argv, argc);
    }
    let entry = {
        let context = unsafe { &mut *ctx };
        let entry = context
            .global
            .heap()
            .closure_code(closure)
            .and_then(|code| context.entry(code.0));
        let Some(entry) = entry else {
            context.error = Some(crate::Error::Runtime(
                "value is not callable in this engine".into(),
            ));
            return Value::UNDEFINED;
        };
        entry
    };
    let function: extern "C" fn(*mut JitCtx, Value, Value, *const Value, u64) -> Value =
        unsafe { std::mem::transmute(entry) };
    let receiver = unsafe { &*ctx }
        .global
        .heap()
        .lexical_this(closure)
        .unwrap_or(receiver);
    let value = function(ctx, closure, receiver, argv, argc);
    unsafe { &mut *ctx }.own(value)
}

extern "C" fn js_regexp_new(ctx: *mut JitCtx, flags: u64, ptr: *const u8, len: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let pattern =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    match ctx.global.heap().regexp(pattern, flags as u8) {
        Ok(value) => ctx.own(value),
        Err(message) => {
            ctx.error = Some(crate::Error::Runtime(format!(
                "invalid regular expression: {message}"
            )));
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
            ctx.error = Some(crate::Error::Runtime("string heap exhausted".into()));
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

extern "C" fn js_exception_pending(ctx: *mut JitCtx, _: Value) -> Value {
    Value::boolean(unsafe { &*ctx }.exception_pending())
}

extern "C" fn js_construct(
    ctx: *mut JitCtx,
    constructor: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    if ctx.global.heap().native(constructor) == Some(crate::heap::NativeFunction::Array) {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return super::builtins::array::construct(ctx, args);
    }
    if let Some(crate::heap::NativeFunction::TypedConstructor(kind)) =
        ctx.global.heap().native(constructor)
    {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return super::builtins::typed_array::construct(ctx, kind, args);
    }
    if ctx.global.heap().constructable(constructor) {
        let Some(instance) = ctx.global.heap().constructed_object(constructor) else {
            ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
            return Value::UNDEFINED;
        };
        ctx.own(instance);
        let result = invoke(ctx, constructor, instance, argv, argc);
        if ctx.error.is_some() || ctx.exception_pending() {
            return Value::UNDEFINED;
        }
        return if result.is_object() { result } else { instance };
    }
    if ctx.global.heap().native(constructor) == Some(crate::heap::NativeFunction::Set) {
        let argument = if argc == 0 {
            Value::UNDEFINED
        } else {
            unsafe { *argv }
        };
        if !argument.is_nullish() {
            ctx.error = Some(crate::Error::Runtime(
                "Set iterable construction is unsupported".into(),
            ));
            return Value::UNDEFINED;
        }
        return match ctx.global.heap().set_new() {
            Some(value) => ctx.own(value),
            None => {
                ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
                Value::UNDEFINED
            }
        };
    }
    if ctx.global.heap().native(constructor) != Some(crate::heap::NativeFunction::Error) {
        ctx.error = Some(crate::Error::Runtime("unsupported constructor".into()));
        return Value::UNDEFINED;
    }
    let argument = if argc == 0 {
        Value::UNDEFINED
    } else {
        unsafe { *argv }
    };
    let heap = ctx.global.heap();
    let text = if argument.is_undefined() {
        String::new()
    } else if argument.is_object() {
        let Some(text) = heap.error_text(argument) else {
            ctx.error = Some(crate::Error::Runtime(
                "object-to-primitive conversion is unsupported".into(),
            ));
            return Value::UNDEFINED;
        };
        text
    } else {
        heap.text(argument).unwrap_or_else(|| argument.to_string())
    };
    let Some(message) = heap.string(&text) else {
        ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
        return Value::UNDEFINED;
    };
    let Some(name) = heap.string("Error") else {
        heap.release(message);
        ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
        return Value::UNDEFINED;
    };
    let result = heap.error(message, name);
    heap.release(message);
    heap.release(name);
    match result {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_in(ctx: *mut JitCtx, key: Value, object: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if !object.is_object() {
        ctx.error = Some(crate::Error::Runtime(format!(
            "right operand of in must be an object, got {object}"
        )));
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
            ctx.error = Some(crate::Error::Runtime(
                "unsupported right operand of in".into(),
            ));
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
        ctx.error = Some(crate::Error::Runtime(
            "instanceof requires a supported callable right operand".into(),
        ));
        return Value::UNDEFINED;
    }
    if !value.is_object() {
        return Value::FALSE;
    }
    let prototype = heap
        .property(constructor, "prototype")
        .unwrap_or(Value::UNDEFINED);
    if !prototype.is_object() {
        ctx.error = Some(crate::Error::Runtime(
            "instanceof constructor prototype is not an object".into(),
        ));
        return Value::UNDEFINED;
    }
    Value::boolean(heap.inherits(value, prototype))
}

extern "C" fn js_typeof(ctx: *mut JitCtx, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if value == Value::UNINITIALIZED {
        ctx.error = Some(crate::Error::Runtime(
            "binding accessed before initialization".into(),
        ));
        return Value::UNDEFINED;
    }
    let name = if ctx.global.heap().closure_code(value).is_some()
        || ctx.global.heap().native(value).is_some()
    {
        "function"
    } else {
        value.type_name()
    };
    match ctx.global.heap().string(name) {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_enumerate_new(ctx: *mut JitCtx, receiver: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().enumerate(receiver) {
        Some(state) => ctx.own(state),
        None => {
            ctx.error = Some(crate::Error::Runtime(
                "for-in receiver is unsupported".into(),
            ));
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
            ctx.error = Some(crate::Error::Runtime("for-in enumeration failed".into()));
            Value::UNDEFINED
        }
    }
}

pub(super) extern "C" fn js_iterate_new(ctx: *mut JitCtx, iterable: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if ctx.global.heap().len(iterable).is_none() {
        ctx.error = Some(crate::Error::Runtime(
            "iteration requires an array, typed array, or string".into(),
        ));
        return Value::UNDEFINED;
    }
    match ctx.global.heap().array(&[iterable, Value::number(0.0)]) {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
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
                ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
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

fn bitwise_binary(
    ctx: *mut JitCtx,
    left: Value,
    right: Value,
    operation: impl FnOnce(i32, i32) -> i64,
) -> Value {
    if !numeric_operands(ctx, &[left, right]) {
        return Value::UNDEFINED;
    }
    Value::number(operation(
        to_int32(primitive_number(ctx, left)),
        to_int32(primitive_number(ctx, right)),
    ) as f64)
}

extern "C" fn js_bitwise_and(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    bitwise_binary(ctx, left, right, |left, right| (left & right) as i64)
}

extern "C" fn js_bitwise_or(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    bitwise_binary(ctx, left, right, |left, right| (left | right) as i64)
}

extern "C" fn js_bitwise_xor(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    bitwise_binary(ctx, left, right, |left, right| (left ^ right) as i64)
}

extern "C" fn js_shift_left(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    bitwise_binary(ctx, left, right, |left, right| {
        (left.wrapping_shl((right as u32) & 31)) as i64
    })
}

extern "C" fn js_shift_right(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    bitwise_binary(ctx, left, right, |left, right| {
        (left.wrapping_shr((right as u32) & 31)) as i64
    })
}

extern "C" fn js_shift_right_unsigned(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    bitwise_binary(ctx, left, right, |left, right| {
        (((left as u32) >> ((right as u32) & 31)) as i64) as i64
    })
}

extern "C" fn js_object_new(ctx: *mut JitCtx, _: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx.global.heap().object() {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
            Value::UNDEFINED
        }
    }
}

fn property_key(ctx: &mut JitCtx, value: Value) -> Option<String> {
    if value.is_object() {
        ctx.error = Some(crate::Error::Runtime(
            "object-to-property-key conversion is unsupported".into(),
        ));
        return None;
    }
    Some(ctx.global.heap().text(value).unwrap_or_else(|| {
        if value.as_number() == Some(0.0) {
            "0".into()
        } else {
            value.to_string()
        }
    }))
}

extern "C" fn js_property_key(ctx: *mut JitCtx, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let Some(name) = property_key(ctx, value) else {
        return Value::UNDEFINED;
    };
    match ctx.global.heap().string(&name) {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
            Value::UNDEFINED
        }
    }
}

fn set_property_get(ctx: &mut JitCtx, object: Value, name: &str) -> Option<Value> {
    use crate::heap::NativeFunction;
    let size = ctx.global.heap().set_size(object)?;
    if name == "size" {
        return Some(Value::number(size as f64));
    }
    let function = match name {
        "add" => NativeFunction::SetAdd,
        "has" => NativeFunction::SetHas,
        "delete" => NativeFunction::SetDelete,
        "clear" => NativeFunction::SetClear,
        _ => return Some(Value::UNDEFINED),
    };
    Some(match ctx.global.heap().native_function(function) {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
            Value::UNDEFINED
        }
    })
}

fn native_call(
    ctx: *mut JitCtx,
    function: crate::heap::NativeFunction,
    receiver: Value,
    argv: *const Value,
    argc: u64,
) -> Value {
    use crate::heap::NativeFunction;
    if matches!(
        function,
        NativeFunction::JsonParse | NativeFunction::JsonStringify
    ) {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return super::builtins::json::call(unsafe { &mut *ctx }, function, args);
    }
    if matches!(
        function,
        NativeFunction::ObjectKeys
            | NativeFunction::ObjectValues
            | NativeFunction::ObjectGetOwnPropertyNames
            | NativeFunction::ObjectFreeze
            | NativeFunction::ObjectIsFrozen
    ) {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return super::builtins::object::call(unsafe { &mut *ctx }, function, args);
    }
    if let NativeFunction::Math(operation) = function {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return super::builtins::math::call(unsafe { &mut *ctx }, operation, args);
    }
    if matches!(function, NativeFunction::TypedConstructor(_)) {
        unsafe { &mut *ctx }.error = Some(crate::Error::Runtime(
            "typed array constructor requires new".into(),
        ));
        return Value::UNDEFINED;
    }
    if function == NativeFunction::Array {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return super::builtins::array::construct(unsafe { &mut *ctx }, args);
    }
    if function == NativeFunction::HostSeedRandom {
        let argument = if argc == 0 {
            Value::UNDEFINED
        } else {
            unsafe { *argv }
        };
        if !numeric_operands(ctx, &[argument]) {
            return Value::UNDEFINED;
        }
        let seed = to_int32(primitive_number(ctx, argument)) as u32;
        unsafe { &*ctx }.engine.seed_random(seed);
        return Value::UNDEFINED;
    }
    if function == NativeFunction::Error {
        let constructor = unsafe { &*ctx }
            .global
            .heap()
            .native_function(function)
            .unwrap();
        let value = js_construct(ctx, constructor, argv, argc);
        unsafe { &*ctx }.global.heap().release(constructor);
        return value;
    }
    if function == NativeFunction::GeneratorNext {
        if argc != 0 && unsafe { *argv } != Value::UNDEFINED {
            unsafe { &mut *ctx }.error = Some(crate::Error::Runtime(
                "generator resume arguments are unsupported".into(),
            ));
            return Value::UNDEFINED;
        }
        return generator_next(ctx, receiver);
    }
    if matches!(
        function,
        NativeFunction::DateNow
            | NativeFunction::ArrayFrom
            | NativeFunction::ArrayIsArray
            | NativeFunction::TypedFrom(_)
            | NativeFunction::TypedFill
            | NativeFunction::HostReadFile
            | NativeFunction::HostListFiles
            | NativeFunction::HostEvaluateSource
            | NativeFunction::Print
            | NativeFunction::ConsoleLog
            | NativeFunction::ArrayFilter
            | NativeFunction::ArrayMap
            | NativeFunction::ArrayEvery
            | NativeFunction::ArraySome
            | NativeFunction::ArrayFind
            | NativeFunction::ArrayForEach
            | NativeFunction::ArrayAt
            | NativeFunction::ArraySort
            | NativeFunction::ArrayJoin
            | NativeFunction::ArrayPop
            | NativeFunction::ArrayIndexOf
            | NativeFunction::ArrayConcat
            | NativeFunction::ArrayPush
            | NativeFunction::ArrayIncludes
            | NativeFunction::ArrayReverse
            | NativeFunction::ArraySlice
            | NativeFunction::ArrayFlat
            | NativeFunction::ArrayFlatMap
            | NativeFunction::ArrayFill
            | NativeFunction::NumberToFixed
            | NativeFunction::NumberToString
            | NativeFunction::RegExpTest
            | NativeFunction::RegExpExec
            | NativeFunction::StringMatch
            | NativeFunction::StringSlice
            | NativeFunction::StringSplit
            | NativeFunction::StringStartsWith
            | NativeFunction::StringReplace
            | NativeFunction::StringPadStart
    ) {
        let args = if argc == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(argv, argc as usize) }
        };
        return super::builtins::call(ctx, function, receiver, args);
    }
    let ctx = unsafe { &mut *ctx };
    if function == NativeFunction::Set {
        ctx.error = Some(crate::Error::Runtime("Set requires new".into()));
        return Value::UNDEFINED;
    }
    if ctx.global.heap().set_size(receiver).is_none() {
        ctx.error = Some(crate::Error::Runtime(
            "Set method requires a Set receiver".into(),
        ));
        return Value::UNDEFINED;
    }
    let argument = if argc == 0 {
        Value::UNDEFINED
    } else {
        unsafe { *argv }
    };
    match function {
        NativeFunction::SetHas => {
            Value::boolean(ctx.global.heap().set_has(receiver, argument).unwrap())
        }
        NativeFunction::SetDelete => {
            Value::boolean(ctx.global.heap().set_delete(receiver, argument).unwrap())
        }
        NativeFunction::SetAdd => {
            if !ctx.global.heap().set_add(receiver, argument) {
                ctx.error = Some(crate::Error::Runtime("invalid Set entry".into()));
                return Value::UNDEFINED;
            }
            ctx.retain_temp(receiver)
        }
        NativeFunction::SetClear => {
            if !ctx.global.heap().set_clear(receiver) {
                ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
            }
            Value::UNDEFINED
        }
        _ => unreachable!(),
    }
}

fn generator_property(ctx: &mut JitCtx, name: &str) -> Value {
    if name != "next" {
        return Value::UNDEFINED;
    }
    match ctx
        .global
        .heap()
        .native_function(crate::heap::NativeFunction::GeneratorNext)
    {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_generator_new(
    ctx: *mut JitCtx,
    closure: Value,
    receiver: Value,
    arguments: Value,
    code: u64,
) -> Value {
    let ctx = unsafe { &mut *ctx };
    match ctx
        .global
        .heap()
        .generator(crate::heap::CodeId(code), closure, arguments, receiver)
    {
        Some(value) => ctx.own(value),
        None => {
            ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_generator_closure(ctx: *mut JitCtx, generator: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let closure = ctx.global.heap().generator_parts(generator).unwrap().1;
    ctx.retain_temp(closure)
}

extern "C" fn js_generator_state(ctx: *mut JitCtx, _: Value) -> u64 {
    unsafe { &*ctx }.resume_state() as u64
}

extern "C" fn js_local_get(ctx: *mut JitCtx, slot: u64) -> Value {
    unsafe { &mut *ctx }.local_get(slot as usize)
}

extern "C" fn js_generator_suspend(
    ctx: *mut JitCtx,
    generator: Value,
    value: Value,
    state: u64,
) -> Value {
    unsafe { &mut *ctx }.suspend(generator, value, state as usize)
}

extern "C" fn js_generator_complete(ctx: *mut JitCtx, generator: Value, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let value = ctx.exit(value);
    ctx.global.heap().generator_complete(generator);
    value
}

fn generator_result(ctx: &mut JitCtx, value: Value, done: bool) -> Value {
    let Some(result) = ctx.global.heap().object() else {
        ctx.global.heap().release(value);
        ctx.error = Some(crate::Error::Runtime("heap exhausted".into()));
        return Value::UNDEFINED;
    };
    ctx.global.heap().set_property(result, "value", value);
    ctx.global
        .heap()
        .set_property(result, "done", Value::boolean(done));
    ctx.global.heap().release(value);
    ctx.own(result)
}

fn generator_next(ctx: *mut JitCtx, generator: Value) -> Value {
    use crate::heap::{GENERATOR_DONE, GENERATOR_RUNNING};
    let context = unsafe { &mut *ctx };
    let Some(state) = context.global.heap().generator_state(generator) else {
        context.error = Some(crate::Error::Runtime(
            "next requires a generator receiver".into(),
        ));
        return Value::UNDEFINED;
    };
    if state == GENERATOR_DONE {
        return generator_result(context, Value::UNDEFINED, true);
    }
    if state == GENERATOR_RUNNING {
        context.error = Some(crate::Error::Runtime("generator is already running".into()));
        return Value::UNDEFINED;
    }
    let (code, _, arguments) = context.global.heap().generator_parts(generator).unwrap();
    let Some(entry) = context.entry(code.0) else {
        context.error = Some(crate::Error::Runtime(
            "generator code is unavailable".into(),
        ));
        return Value::UNDEFINED;
    };
    let args: Vec<_> = (0..context.global.heap().len(arguments).unwrap())
        .map(|index| context.global.heap().get(arguments, index).unwrap())
        .collect();
    let receiver = context.global.heap().generator_receiver(generator).unwrap();
    let (state, locals) = context.global.heap().generator_begin(generator).unwrap();
    context.resume(locals, state);
    let function: extern "C" fn(*mut JitCtx, Value, Value, *const Value, u64) -> Value =
        unsafe { std::mem::transmute(entry) };
    let value = function(ctx, generator, receiver, args.as_ptr(), args.len() as u64);
    let context = unsafe { &mut *ctx };
    if context.error.is_some() || context.exception_pending() {
        context.global.heap().release(value);
        return Value::UNDEFINED;
    }
    let done = context.global.heap().generator_state(generator) == Some(GENERATOR_DONE);
    generator_result(context, value, done)
}

extern "C" fn js_bind_this(ctx: *mut JitCtx, closure: Value, receiver: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    if !ctx.global.heap().bind_this(closure, receiver) {
        ctx.error = Some(crate::Error::Runtime("invalid arrow receiver".into()));
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
