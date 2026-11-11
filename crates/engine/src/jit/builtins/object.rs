use super::{arg, fail};
use crate::heap::NativeFunction;
use crate::jit::session::JitCtx;
use crate::value::Value;

pub(in crate::jit) fn call(ctx: &mut JitCtx, function: NativeFunction, args: &[Value]) -> Value {
    let value = arg(args, 0);
    if function == NativeFunction::ObjectFreeze {
        return ctx.retain(value);
    }
    if function == NativeFunction::ObjectIsFrozen {
        return Value::boolean(!value.is_object());
    }
    let heap = ctx.global.heap().clone();
    let Some(names) =
        heap.own_property_names(value, function != NativeFunction::ObjectGetOwnPropertyNames)
    else {
        return fail(
            ctx,
            "Object enumeration requires a supported non-null value",
        );
    };
    let mut values = Vec::with_capacity(names.len());
    let text = if function == NativeFunction::ObjectValues {
        heap.text(value)
    } else {
        None
    };
    let characters: Vec<_> = text
        .as_ref()
        .map(|text| text.chars().collect())
        .unwrap_or_default();
    for name in names {
        let element = if function != NativeFunction::ObjectValues {
            heap.string(&name)
        } else if text.is_some() {
            let index: usize = name.parse().unwrap();
            heap.string(characters[index].encode_utf8(&mut [0; 4]))
        } else {
            let element = if heap.is_array(value) || heap.typed_kind(value).is_some() {
                heap.get(value, name.parse().unwrap())
            } else {
                heap.property(value, &name)
            };
            element.filter(|element| heap.retain(*element))
        };
        let Some(element) = element else {
            for value in values {
                heap.release(value);
            }
            return fail(ctx, "Object enumeration failed");
        };
        values.push(element);
    }
    let result = heap.array(&values);
    for value in values {
        heap.release(value);
    }
    match result {
        Some(value) => value,
        None => fail(ctx, "heap exhausted"),
    }
}
