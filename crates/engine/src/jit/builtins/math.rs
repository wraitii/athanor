use super::{arg, fail};
use crate::heap::MathFunction;
use crate::jit::helpers::to_int32;
use crate::jit::session::JitCtx;
use crate::value::Value;

pub(in crate::jit) fn call(ctx: &mut JitCtx, function: MathFunction, args: &[Value]) -> Value {
    use MathFunction::*;
    let count = match function {
        Min | Max => args.len(),
        Atan2 | Pow | Imul => 2,
        _ => 1,
    };
    let mut numbers = Vec::with_capacity(count);
    for index in 0..count {
        let value = arg(args, index);
        if value.is_string() || value.is_object() {
            return fail(ctx, "Math string and object coercion is unsupported");
        }
        numbers.push(value.to_number());
    }
    let x = numbers.first().copied().unwrap_or(f64::NAN);
    let y = numbers.get(1).copied().unwrap_or(f64::NAN);
    let result = match function {
        Abs => x.abs(),
        Ceil => x.ceil(),
        Floor => x.floor(),
        Trunc => x.trunc(),
        Round => {
            if !x.is_finite() || x == 0.0 {
                x
            } else if x >= -0.5 && x < 0.0 {
                -0.0
            } else {
                let floor = x.floor();
                if x - floor < 0.5 { floor } else { floor + 1.0 }
            }
        }
        Sqrt => x.sqrt(),
        Sin => x.sin(),
        Cos => x.cos(),
        Tan => x.tan(),
        Asin => x.asin(),
        Acos => x.acos(),
        Atan => x.atan(),
        Atan2 => x.atan2(y),
        Exp => x.exp(),
        Log => x.ln(),
        Pow => {
            if x.abs() == 1.0 && y.is_infinite() {
                f64::NAN
            } else {
                x.powf(y)
            }
        }
        Min | Max => {
            let mut result = if function == Min {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            for n in numbers {
                if n.is_nan() {
                    result = n;
                    break;
                }
                if (function == Min
                    && (n < result || (n == 0.0 && result == 0.0 && n.is_sign_negative())))
                    || (function == Max
                        && (n > result || (n == 0.0 && result == 0.0 && n.is_sign_positive())))
                {
                    result = n;
                }
            }
            result
        }
        Imul => f64::from(to_int32(x).wrapping_mul(to_int32(y))),
        Sign => {
            if x == 0.0 || x.is_nan() {
                x
            } else {
                x.signum()
            }
        }
    };
    Value::number(result)
}
