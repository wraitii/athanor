use crate::FxHashSet;

use super::{arg, fail};
use crate::heap::NativeFunction;
use crate::jit::session::{Held, JitCtx};
use crate::value::Value;

pub(in crate::jit) fn call(ctx: &mut JitCtx, function: NativeFunction, args: &[Value]) -> Value {
    match function {
        NativeFunction::JsonParse => {
            if !arg(args, 1).is_undefined() {
                return fail(ctx, "JSON.parse reviver is unsupported");
            }
            let Some(source) = ctx.global.heap().text(arg(args, 0)) else {
                return fail(ctx, "JSON.parse requires a string");
            };
            let Ok(json) = serde_json::from_str(&source) else {
                return fail(ctx, "invalid JSON");
            };
            match parse_value(ctx, json) {
                Some(value) => value,
                None => fail(ctx, "JSON allocation failed"),
            }
        }
        NativeFunction::JsonStringify => {
            if !arg(args, 1).is_undefined() || !arg(args, 2).is_undefined() {
                return fail(ctx, "JSON.stringify replacer and space are unsupported");
            }
            let mut active = FxHashSet::default();
            let result = stringify(ctx, arg(args, 0), &mut active, 0);
            match result {
                Ok(None) => Value::UNDEFINED,
                Ok(Some(text)) => match ctx.global.heap().string(&text) {
                    Some(value) => value,
                    None => fail(ctx, "JSON allocation failed"),
                },
                Err(message) => fail(ctx, message),
            }
        }
        _ => unreachable!(),
    }
}

fn parse_value(ctx: &mut JitCtx, json: serde_json::Value) -> Option<Value> {
    let heap = ctx.global.heap().clone();
    match json {
        serde_json::Value::Null => Some(Value::NULL),
        serde_json::Value::Bool(value) => Some(Value::boolean(value)),
        serde_json::Value::Number(value) => Some(Value::number(value.as_f64()?)),
        serde_json::Value::String(value) => heap.string(&value),
        serde_json::Value::Array(values) => {
            let mut elements = Held::new(&heap);
            for value in values {
                elements.push(parse_value(ctx, value)?);
            }
            heap.array_from_owned(elements.take())
        }
        serde_json::Value::Object(properties) => {
            let mut allocation = Held::new(&heap);
            let object = allocation.push(heap.object()?);
            for (name, value) in properties {
                let value = parse_value(ctx, value)?;
                let stored = heap.set_property(object, &name, value);
                heap.release(value);
                if !stored {
                    return None;
                }
            }
            allocation.take();
            Some(object)
        }
    }
}

fn stringify(
    ctx: &JitCtx,
    value: Value,
    active: &mut FxHashSet<u64>,
    depth: usize,
) -> Result<Option<String>, &'static str> {
    if value.is_undefined() {
        return Ok(None);
    }
    if value.is_null() {
        return Ok(Some("null".into()));
    }
    if value == Value::TRUE {
        return Ok(Some("true".into()));
    }
    if value == Value::FALSE {
        return Ok(Some("false".into()));
    }
    if let Some(number) = value.as_number() {
        if !number.is_finite() {
            return Ok(Some("null".into()));
        }
        if number == 0.0 {
            return Ok(Some("0".into()));
        }
        let mut text = serde_json::Number::from_f64(number).unwrap().to_string();
        if text.ends_with(".0") {
            text.truncate(text.len() - 2);
        }
        return Ok(Some(text));
    }
    let heap = ctx.global.heap();
    if let Some(text) = heap.text(value) {
        return Ok(Some(serde_json::to_string(&text).unwrap()));
    }
    if heap.native(value).is_some() || heap.closure_code(value).is_some() {
        return Ok(None);
    }
    if depth >= 128 {
        return Err("JSON nesting limit exceeded");
    }
    if !active.insert(value.bits()) {
        return Err("JSON cyclic object");
    }
    let output = if heap.is_array(value) {
        let mut elements = Vec::new();
        for index in 0..heap.len(value).unwrap() {
            let element = heap.get(value, index).unwrap();
            elements
                .push(stringify(ctx, element, active, depth + 1)?.unwrap_or_else(|| "null".into()));
        }
        format!("[{}]", elements.join(","))
    } else if let Some(properties) = heap.own_properties(value) {
        if heap
            .property(value, "toJSON")
            .is_some_and(|v| heap.native(v).is_some() || heap.closure_code(v).is_some())
        {
            return Err("JSON toJSON methods are unsupported");
        }
        let mut elements = Vec::new();
        for (name, value) in properties {
            if let Some(value) = stringify(ctx, value, active, depth + 1)? {
                elements.push(format!(
                    "{}:{}",
                    serde_json::to_string(&name).unwrap(),
                    value
                ));
            }
        }
        format!("{{{}}}", elements.join(","))
    } else {
        return Err("JSON unsupported object");
    };
    active.remove(&value.bits());
    Ok(Some(output))
}
