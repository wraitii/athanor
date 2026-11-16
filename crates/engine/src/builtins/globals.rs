use super::NativeFunction;
use crate::environment::Environment;
use crate::{Error, Value};

pub(crate) fn install(global: &Environment) -> Result<(), Error> {
    use crate::heap::TypedKind;
    let heap = global.heap();
    let exhausted = || Error::Runtime("heap exhausted".into());
    let native = |function: NativeFunction| heap.native_function(function).ok_or_else(exhausted);
    let define = |name: &str, value: Value| {
        global.define(name, value);
        heap.release(value);
    };
    let add_methods = |object: Value, methods: &[(&str, NativeFunction)]| -> Result<(), Error> {
        for &(name, function) in methods {
            let value = native(function)?;
            heap.define_property(object, name, value, false);
            heap.release(value);
        }
        Ok(())
    };
    let namespace = |methods: &[(&str, NativeFunction)]| -> Result<Value, Error> {
        let object = heap.object().ok_or_else(exhausted)?;
        add_methods(object, methods)?;
        Ok(object)
    };

    global.define("globalThis", Value::GLOBAL_OBJECT);
    global.define_readonly("Infinity", Value::number(f64::INFINITY));
    global.define_readonly("NaN", Value::number(f64::NAN));
    for &(name, function) in super::GLOBALS {
        define(name, native(function)?);
    }
    for &(name, methods) in super::NAMESPACES {
        define(name, namespace(methods)?);
    }
    for (name, kind) in [
        ("Uint8Array", TypedKind::Uint8),
        ("Uint16Array", TypedKind::Uint16),
        ("Uint32Array", TypedKind::Uint32),
        ("Int8Array", TypedKind::Int8),
        ("Int16Array", TypedKind::Int16),
        ("Int32Array", TypedKind::Int32),
        ("Float32Array", TypedKind::Float32),
        ("Float64Array", TypedKind::Float64),
    ] {
        define(name, native(NativeFunction::TypedConstructor(kind))?);
    }
    let math = heap.object().ok_or_else(exhausted)?;
    for (name, value) in [("PI", std::f64::consts::PI), ("E", std::f64::consts::E)] {
        heap.define_property(math, name, Value::number(value), false);
    }
    add_methods(math, super::MATH_METHODS)?;
    define("Math", math);
    Ok(())
}
