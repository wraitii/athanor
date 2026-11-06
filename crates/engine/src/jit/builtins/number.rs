use num_bigint::BigUint;

use super::super::session::JitCtx;
use super::{arg, fail};
use crate::heap::NativeFunction;
use crate::value::{Value, number_text};

fn parts(value: f64) -> (u64, i32) {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent - 1075)
    }
}

fn fixed(value: f64, digits: u32) -> String {
    if !value.is_finite() || value.abs() >= 1e21 {
        return number_text(value);
    }
    let (mantissa, exponent) = parts(value.abs());
    let numerator = BigUint::from(mantissa) * BigUint::from(10_u8).pow(digits);
    let rounded = if exponent >= 0 {
        numerator << exponent as usize
    } else {
        let shift = (-exponent) as usize;
        let quotient = &numerator >> shift;
        let remainder = numerator - (&quotient << shift);
        if remainder >= (BigUint::from(1_u8) << (shift - 1)) {
            quotient + 1_u8
        } else {
            quotient
        }
    };
    let mut result = rounded.to_str_radix(10);
    let digits = digits as usize;
    if digits > 0 {
        if result.len() <= digits {
            result = format!("{}{}", "0".repeat(digits + 1 - result.len()), result);
        }
        result.insert(result.len() - digits, '.');
    }
    if value < 0.0 {
        result.insert(0, '-');
    }
    result
}

fn radix_integer(value: f64, radix: u32) -> String {
    if !value.is_finite() || value == 0.0 {
        return number_text(value);
    }
    let (mantissa, exponent) = parts(value.abs());
    let value_integer = if exponent >= 0 {
        BigUint::from(mantissa) << exponent as usize
    } else {
        BigUint::from(mantissa) >> (-exponent) as usize
    };
    let mut result = value_integer.to_str_radix(radix);
    if value < 0.0 {
        result.insert(0, '-');
    }
    result
}

pub(super) fn call(
    ctx: *mut JitCtx,
    function: NativeFunction,
    receiver: Value,
    args: &[Value],
) -> Value {
    let ctx = unsafe { &mut *ctx };
    let Some(value) = receiver.as_number() else {
        return fail(ctx, "number method requires a number receiver");
    };
    let argument = arg(args, 0);
    let parameter = if argument.is_undefined() {
        if function == NativeFunction::NumberToFixed {
            0.0
        } else {
            10.0
        }
    } else if let Some(number) = argument.as_number() {
        if number.is_nan() { 0.0 } else { number.trunc() }
    } else {
        return fail(
            ctx,
            "number formatting argument must be a number or undefined",
        );
    };
    let text = match function {
        NativeFunction::NumberToFixed => {
            if !(0.0..=100.0).contains(&parameter) {
                return fail(ctx, "toFixed digits must be between 0 and 100");
            }
            fixed(value, parameter as u32)
        }
        NativeFunction::NumberToString => {
            if !(2.0..=36.0).contains(&parameter) {
                return fail(ctx, "toString radix must be between 2 and 36");
            }
            let radix = parameter as u32;
            if radix == 10 {
                number_text(value)
            } else if value.is_finite() && value.fract() != 0.0 {
                return fail(
                    ctx,
                    "fractional nondecimal number formatting is unsupported",
                );
            } else {
                radix_integer(value, radix)
            }
        }
        _ => unreachable!(),
    };
    match ctx.global.heap().string(&text) {
        Some(value) => ctx.own(value),
        None => fail(ctx, "heap exhausted"),
    }
}
