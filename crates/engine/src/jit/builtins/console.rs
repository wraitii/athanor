use super::super::session::JitCtx;
use crate::heap::NativeFunction;
use crate::value::Value;

fn printable(context: &mut JitCtx, value: Value) -> Option<String> {
    let heap = context.global.heap().clone();
    if let Some(text) = heap.text(value).or_else(|| heap.error_text(value)) {
        return Some(text);
    }
    if heap.is_array(value) {
        let length = heap.len(value).unwrap();
        let mut parts = Vec::with_capacity(length);
        for index in 0..length {
            let element = heap.get(value, index).unwrap();
            parts.push(if element == Value::UNDEFINED || element == Value::NULL {
                String::new()
            } else if let Some(text) = heap.text(element) {
                text
            } else if element.is_object() {
                "[object Object]".into()
            } else {
                element.to_string()
            });
        }
        return Some(format!("[{}]", parts.join(", ")));
    }
    if value.is_object() {
        return Some("[object Object]".into());
    }
    Some(value.to_string())
}

pub(super) fn call(ctx: *mut JitCtx, function: NativeFunction, args: &[Value]) -> Value {
    let context = unsafe { &mut *ctx };
    let mut output = String::new();
    for (index, value) in args.iter().copied().enumerate() {
        if function == NativeFunction::ConsoleLog && index > 0 {
            output.push(' ');
        }
        let Some(text) = printable(context, value) else {
            return Value::UNDEFINED;
        };
        output.push_str(&text);
    }
    if function == NativeFunction::ConsoleLog {
        output.push('\n');
    }
    if let Some(host) = context.engine.host.as_ref() {
        host.print(&output);
    }
    Value::UNDEFINED
}
