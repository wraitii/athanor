use super::*;

pub(crate) fn primitive_number(ctx: *mut JitCtx, value: Value) -> f64 {
    if value.is_string() {
        let heap = unsafe { &*ctx }.global.heap();
        return heap
            .text(value)
            .map(|text| crate::jit::conversion::string_number(&text))
            .unwrap_or(f64::NAN);
    }
    value.to_number()
}

pub(super) fn numeric_operands(ctx: *mut JitCtx, values: &[Value]) -> bool {
    if values.iter().any(|value| value.is_object()) {
        unsafe { &mut *ctx }.error = Some(Box::new(crate::Error::Runtime(
            "object-to-primitive conversion is unsupported".into(),
        )));
        return false;
    }
    true
}

pub(super) fn number_binary(
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

pub(super) fn number_compare(
    ctx: *mut JitCtx,
    left: Value,
    right: Value,
    operation: impl FnOnce(f64, f64) -> bool,
) -> Value {
    if left.is_string() && right.is_string() {
        let heap = unsafe { &*ctx }.global.heap();
        let order = match heap.compare_strings(left, right) {
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

pub(crate) fn to_int32(number: f64) -> i32 {
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

pub(super) fn loose_equals(ctx: *mut JitCtx, left: Value, right: Value) -> bool {
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

pub(crate) fn strict_equals(heap: &crate::heap::Heap, left: Value, right: Value) -> bool {
    if left.is_string() && right.is_string() {
        return heap.strings_equal(left, right);
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
            ctx.error = Some(Box::new(crate::Error::Runtime(
                "object-to-primitive conversion is unsupported".into(),
            )));
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
            Some(value) => value,
            None => {
                ctx.error = Some(Box::new(crate::Error::Runtime(
                    "string heap exhausted".into(),
                )));
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
    number_binary(ctx, left, right, number_pow)
}

pub(crate) fn number_pow(base: f64, exponent: f64) -> f64 {
    if exponent.is_nan() || (base.abs() == 1.0 && exponent.is_infinite()) {
        f64::NAN
    } else {
        base.powf(exponent)
    }
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

pub(super) fn bitwise_binary(
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

pub(super) extern "C" fn js_bitwise_and(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    bitwise_binary(ctx, left, right, |left, right| (left & right) as i64)
}

pub(super) extern "C" fn js_bitwise_or(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    bitwise_binary(ctx, left, right, |left, right| (left | right) as i64)
}

pub(super) extern "C" fn js_bitwise_xor(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    bitwise_binary(ctx, left, right, |left, right| (left ^ right) as i64)
}

pub(super) extern "C" fn js_shift_left(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    bitwise_binary(ctx, left, right, |left, right| {
        (left.wrapping_shl((right as u32) & 31)) as i64
    })
}

pub(super) extern "C" fn js_shift_right(ctx: *mut JitCtx, left: Value, right: Value) -> Value {
    bitwise_binary(ctx, left, right, |left, right| {
        (left.wrapping_shr((right as u32) & 31)) as i64
    })
}

pub(super) extern "C" fn js_shift_right_unsigned(
    ctx: *mut JitCtx,
    left: Value,
    right: Value,
) -> Value {
    bitwise_binary(ctx, left, right, |left, right| {
        ((left as u32) >> ((right as u32) & 31)) as i64
    })
}
