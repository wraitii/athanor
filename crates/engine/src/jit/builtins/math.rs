use super::{arg, fail};
use crate::heap::MathFunction;
use crate::jit::helpers::to_int32;
use crate::jit::session::JitCtx;
use crate::value::Value;

fn number(value: Value) -> Option<f64> {
    if value.is_string() || value.is_object() {
        return None;
    }
    Some(value.to_number())
}

fn min_max(ctx: &mut JitCtx, function: MathFunction, args: &[Value]) -> Value {
    let minimum = function == MathFunction::Min;
    let mut result = if minimum {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    for &value in args {
        let Some(n) = number(value) else {
            return fail(ctx, "Math string and object coercion is unsupported");
        };
        if result.is_nan() {
            continue;
        }
        if n.is_nan()
            || (minimum && (n < result || (n == 0.0 && result == 0.0 && n.is_sign_negative())))
            || (!minimum && (n > result || (n == 0.0 && result == 0.0 && n.is_sign_positive())))
        {
            result = n;
        }
    }
    Value::number(result)
}

pub(in crate::jit) fn call(ctx: &mut JitCtx, function: MathFunction, args: &[Value]) -> Value {
    use MathFunction::*;
    if function == Random {
        return Value::number(ctx.engine.random());
    }
    if matches!(function, Min | Max) {
        return min_max(ctx, function, args);
    }
    let Some(x) = number(arg(args, 0)) else {
        return fail(ctx, "Math string and object coercion is unsupported");
    };
    let y = if matches!(function, Atan2 | Pow | Imul) {
        let Some(y) = number(arg(args, 1)) else {
            return fail(ctx, "Math string and object coercion is unsupported");
        };
        y
    } else {
        f64::NAN
    };
    let result = match function {
        Random | Min | Max => unreachable!(),
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
