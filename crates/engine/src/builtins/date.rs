use std::time::{SystemTime, UNIX_EPOCH};

use super::NativeFunction;
use crate::jit::session::JitCtx;
use crate::value::Value;

fn unix_milliseconds(time: SystemTime) -> f64 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as f64,
        Err(error) => {
            let duration = error.duration();
            let whole = duration.as_millis();
            let remainder = duration.subsec_nanos() % 1_000_000;
            -((whole + u128::from(remainder != 0)) as f64)
        }
    }
}

pub(super) fn call(
    _ctx: *mut JitCtx,
    _function: NativeFunction,
    _receiver: Value,
    _args: &[Value],
) -> Value {
    Value::number(unix_milliseconds(SystemTime::now()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn unix_milliseconds_floors_before_epoch() {
        assert_eq!(unix_milliseconds(UNIX_EPOCH), 0.0);
        assert_eq!(
            unix_milliseconds(UNIX_EPOCH + Duration::from_micros(1999)),
            1.0
        );
        assert_eq!(
            unix_milliseconds(UNIX_EPOCH - Duration::from_micros(1)),
            -1.0
        );
        assert_eq!(
            unix_milliseconds(UNIX_EPOCH - Duration::from_micros(1001)),
            -2.0
        );
    }
}
