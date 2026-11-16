use crate::heap::{Heap, NativeFunction};
use crate::jit::session::JitCtx;
use crate::value::Value;

fn printable(heap: &Heap, value: Value) -> String {
    if let Some(text) = heap.text(value).or_else(|| heap.error_text(value)) {
        return text;
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
        return format!("[{}]", parts.join(", "));
    }
    if value.is_object() {
        return "[object Object]".into();
    }
    value.to_string()
}

pub(super) fn call(
    ctx: *mut JitCtx,
    function: NativeFunction,
    _receiver: Value,
    args: &[Value],
) -> Value {
    let context = unsafe { &mut *ctx };
    let mut output = String::new();
    for (index, value) in args.iter().copied().enumerate() {
        if function == NativeFunction::ConsoleLog && index > 0 {
            output.push(' ');
        }
        let text = printable(context.global.heap(), value);
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
