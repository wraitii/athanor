use cranelift_codegen::ir::{AbiParam, Signature, Type, types};
use cranelift_codegen::isa::CallConv;

use crate::value::Value;

use super::session::JitCtx;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HelperShape {
    Binary,
    Unary,
    GlobalGet,
    GlobalSet,
    PropertyGet,
    PropertySet,
    ArrayNew,
    Rest,
    IndexSet,
    Callable,
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
    CellNew => Unary, js_cell_new;
    CellGet => Unary, js_cell_get;
    CellSet => Binary, js_cell_set;
    CaptureGet => Binary, js_capture_get;
    ClosureNew => Callable, js_closure_new;
    Call => Callable, js_call;
    FrameEnter => Unary, js_frame_enter;
    FrameExit => Unary, js_frame_exit;
    LocalSet => Binary, js_local_set;
    TempClear => Unary, js_temp_clear;
    TempRetain => Unary, js_temp_retain;
    ArrayNew => ArrayNew, js_array_new;
    Rest => Rest, js_rest;
    IndexGet => Binary, js_index_get;
    IndexSet => IndexSet, js_index_set;
    PropertyGet => PropertyGet, js_property_get;
    PropertySet => PropertySet, js_property_set;
    GlobalGet => GlobalGet, js_global_get;
    GlobalSet => GlobalSet, js_global_set;
    ErrorPending => Unary, js_error_pending;
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

pub(crate) fn symbols() -> impl Iterator<Item = (&'static str, *const u8)> {
    HELPERS.iter().map(|spec| (spec.name, spec.pointer))
}

pub(crate) fn signature(shape: HelperShape, call_conv: CallConv, pointer: Type) -> Signature {
    let mut signature = Signature::new(call_conv);
    signature.params.push(AbiParam::new(pointer));
    match shape {
        HelperShape::Callable => {
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(pointer));
            signature.params.push(AbiParam::new(types::I64));
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
    Value::number(operation(left.to_number(), right.to_number()))
}

fn number_compare(
    ctx: *mut JitCtx,
    left: Value,
    right: Value,
    operation: impl FnOnce(f64, f64) -> bool,
) -> Value {
    if !numeric_operands(ctx, &[left, right]) {
        return Value::UNDEFINED;
    }
    Value::boolean(operation(left.to_number(), right.to_number()))
}

fn to_int32(number: f64) -> i32 {
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
        if left.is_string() || right.is_string() || left.is_object() || right.is_object() {
            return false;
        }
    }
    left.to_number() == right.to_number()
}

fn strict_equals(left: Value, right: Value) -> bool {
    if left.is_number() && right.is_number() {
        return left.as_number() == right.as_number();
    }
    if left.is_number() != right.is_number() {
        return false;
    }
    left.bits() == right.bits()
}

pub(crate) extern "C" fn js_add(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
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

pub(crate) extern "C" fn js_strict_equal(_ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    Value::boolean(strict_equals(left, right))
}

pub(crate) extern "C" fn js_strict_not_equal(
    _ctx: *mut JitCtx,
    left: Value,
    right: Value,
) -> Value {
    Value::boolean(!strict_equals(left, right))
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
    Value::number(value.to_number())
}

pub(crate) extern "C" fn js_unary_negation(ctx: *mut JitCtx, value: Value) -> Value {
    if !numeric_operands(ctx, &[value]) {
        return Value::UNDEFINED;
    }
    Value::number(-value.to_number())
}

pub(crate) extern "C" fn js_logical_not(_ctx: *mut JitCtx, value: Value) -> Value {
    Value::boolean(!value.to_boolean())
}

pub(crate) extern "C" fn js_bitwise_not(ctx: *mut JitCtx, value: Value) -> Value {
    if !numeric_operands(ctx, &[value]) {
        return Value::UNDEFINED;
    }
    let value = value.to_number();
    Value::number(f64::from(!to_int32(value)))
}

extern "C" fn js_global_get(ctx: *mut JitCtx, ptr: *const u8, len: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    match ctx.global.get_raw(name) {
        Some(value) => ctx.retain_temp(value),
        None => {
            ctx.error = Some(crate::Error::Runtime(format!("unknown variable `{name}`")));
            Value::UNDEFINED
        }
    }
}

extern "C" fn js_global_set(ctx: *mut JitCtx, ptr: *const u8, len: u64, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if !ctx.global.set(name, value) {
        ctx.error = Some(crate::Error::Runtime(format!("unknown variable `{name}`")));
    }
    value
}

extern "C" fn js_error_pending(ctx: *mut JitCtx, _unused: Value) -> Value {
    Value::boolean(unsafe { &*ctx }.error.is_some())
}

extern "C" fn js_property_get(ctx: *mut JitCtx, object: Value, ptr: *const u8, len: u64) -> Value {
    let ctx = unsafe { &mut *ctx };
    let name =
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len as usize)) };
    if object != Value::GLOBAL_OBJECT {
        if name == "length" {
            if let Some(length) = ctx.global.heap().len(object) {
                return Value::number(length as f64);
            }
        }
        ctx.error = Some(crate::Error::Runtime(
            "property access on unsupported value".into(),
        ));
        return Value::UNDEFINED;
    }
    let value = ctx.global.get_raw(name).unwrap_or(Value::UNDEFINED);
    ctx.retain_temp(value)
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
        if name == "length"
            && array_index(value).is_some_and(|length| ctx.global.heap().truncate(object, length))
        {
            return value;
        }
        ctx.error = Some(crate::Error::Runtime(
            "property assignment on unsupported value".into(),
        ));
        return Value::UNDEFINED;
    }
    if !ctx.global.set(name, value) {
        ctx.global.define(name, value);
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
    unsafe { &mut *ctx }.retain_temp(value)
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
extern "C" fn js_index_get(ctx: *mut JitCtx, array: Value, index: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
    let result = array_index(index).and_then(|index| ctx.global.heap().get(array, index));
    match result {
        Some(value) => ctx.retain_temp(value),
        None => {
            ctx.error = Some(crate::Error::Runtime(
                "unsupported array index or receiver".into(),
            ));
            Value::UNDEFINED
        }
    }
}
extern "C" fn js_index_set(ctx: *mut JitCtx, array: Value, index: Value, value: Value) -> Value {
    let ctx = unsafe { &mut *ctx };
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

extern "C" fn js_call(ctx: *mut JitCtx, closure: Value, argv: *const Value, argc: u64) -> Value {
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
    let function: extern "C" fn(*mut JitCtx, Value, *const Value, u64) -> Value =
        unsafe { std::mem::transmute(entry) };
    let value = function(ctx, closure, argv, argc);
    unsafe { &mut *ctx }.own(value)
}
