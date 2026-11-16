mod array;
mod console;
mod date;
mod error;
mod generator;
mod globals;
mod host;
mod json;
pub(crate) mod math;
mod number;
mod object;
mod regexp;
mod set;
pub(crate) mod string;
pub(crate) mod typed_array;

use crate::heap::TypedKind;
use crate::jit::session::JitCtx;
use crate::value::Value;

pub(crate) use globals::install;

fn arg(args: &[Value], index: usize) -> Value {
    args.get(index).copied().unwrap_or(Value::UNDEFINED)
}

fn fail(context: &mut JitCtx, message: &str) -> Value {
    context.error = Some(Box::new(crate::Error::Runtime(message.into())));
    Value::UNDEFINED
}

macro_rules! define_builtins {
    (
        globals { $($global:ident: $global_name:literal => $global_call:path $(, new => $construct:path)?;)* }
        namespaces { $($namespace:literal { $($static:ident: $static_name:literal => $static_call:path;)* })* }
        methods { $($kind:ident { $($method:ident: $method_name:literal => $method_call:path;)* })* }
        statics { $($owner:ident { $($property:ident: $property_name:literal => $property_call:path;)* })* }
        internal { $($internal:ident => $internal_call:path;)* }
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum NativeFunction {
            $($global,)*
            $($($static,)*)*
            $($($method,)*)*
            $($($property,)*)*
            $($internal,)*
            Math(MathFunction),
            TypedConstructor(TypedKind),
            TypedFrom(TypedKind),
        }

        #[derive(Clone, Copy)]
        pub(crate) enum ReceiverKind { $($kind,)* }

        impl ReceiverKind {
            pub(crate) fn method(self, name: &str) -> Option<NativeFunction> {
                Some(match (self, name) {
                    $($( (Self::$kind, $method_name) => NativeFunction::$method,)*)*
                    _ => return None,
                })
            }
        }

        pub(crate) fn builtin_method(name: &str) -> bool {
            // Numeric receivers have no heap payload for the method cache.
            false $(|| (!matches!(ReceiverKind::$kind, ReceiverKind::Number)
                && matches!(name, $($method_name)|*)))*
        }

        impl NativeFunction {
            pub(crate) fn static_method(self, name: &str) -> Option<Self> {
                Some(match (self, name) {
                    $($( (Self::$owner, $property_name) => Self::$property,)*)*
                    (Self::TypedConstructor(kind), "from") => Self::TypedFrom(kind),
                    _ => return None,
                })
            }
        }

        const GLOBALS: &[(&str, NativeFunction)] = &[
            $(($global_name, NativeFunction::$global),)*
        ];

        const NAMESPACES: &[(&str, &[(&str, NativeFunction)])] = &[
            $(($namespace, &[$(($static_name, NativeFunction::$static),)*]),)*
        ];

        pub(crate) fn call(ctx: *mut JitCtx, function: NativeFunction, receiver: Value, args: &[Value]) -> Value {
            match function {
                $(NativeFunction::$global => $global_call(ctx, function, receiver, args),)*
                $($(NativeFunction::$static => $static_call(ctx, function, receiver, args),)*)*
                $($(NativeFunction::$method => $method_call(ctx, function, receiver, args),)*)*
                $($(NativeFunction::$property => $property_call(ctx, function, receiver, args),)*)*
                $(NativeFunction::$internal => $internal_call(ctx, function, receiver, args),)*
                NativeFunction::Math(operation) => math::call(unsafe { &mut *ctx }, operation, args),
                NativeFunction::TypedConstructor(_) | NativeFunction::TypedFrom(_) => typed_array::call(ctx, function, receiver, args),
            }
        }

        pub(crate) fn construct(ctx: *mut JitCtx, function: NativeFunction, args: &[Value]) -> Value {
            match function {
                $($(NativeFunction::$global => $construct(ctx, args),)?)*
                NativeFunction::TypedConstructor(kind) => typed_array::construct(unsafe { &mut *ctx }, kind, args),
                _ => fail(unsafe { &mut *ctx }, "unsupported constructor"),
            }
        }
    };
}

define_builtins! {
    globals {
        Error: "Error" => error::call, new => error::construct;
        Set: "Set" => set::call, new => set::construct;
        Array: "Array" => array::call, new => array::construct;
        Print: "print" => console::call;
        HostSeedRandom: "hostSeedRandom" => host::seed_random;
    }
    namespaces {
        "console" {
            ConsoleLog: "log" => console::call;
        }
        "JSON" {
            JsonParse: "parse" => json::call;
            JsonStringify: "stringify" => json::call;
        }
        "Object" {
            ObjectKeys: "keys" => object::call;
            ObjectValues: "values" => object::call;
            ObjectGetOwnPropertyNames: "getOwnPropertyNames" => object::call;
            ObjectFreeze: "freeze" => object::call;
            ObjectIsFrozen: "isFrozen" => object::call;
        }
        "Date" {
            DateNow: "now" => date::call;
        }
    }
    methods {
        Array {
            ArrayFilter: "filter" => array::call;
            ArrayMap: "map" => array::call;
            ArrayEvery: "every" => array::call;
            ArraySome: "some" => array::call;
            ArrayFind: "find" => array::call;
            ArrayForEach: "forEach" => array::call;
            ArrayAt: "at" => array::call;
            ArraySort: "sort" => array::call;
            ArrayJoin: "join" => array::call;
            ArrayPop: "pop" => array::call;
            ArrayIndexOf: "indexOf" => array::call;
            ArrayConcat: "concat" => array::call;
            ArrayPush: "push" => array::call;
            ArrayIncludes: "includes" => array::call;
            ArrayReverse: "reverse" => array::call;
            ArraySlice: "slice" => array::call;
            ArrayFlat: "flat" => array::call;
            ArrayFlatMap: "flatMap" => array::call;
            ArrayFill: "fill" => array::call;
        }
        TypedArray {
            TypedFill: "fill" => typed_array::call;
        }
        RegExp {
            RegExpTest: "test" => regexp::call;
            RegExpExec: "exec" => regexp::call;
        }
        String {
            StringMatch: "match" => regexp::call;
            StringSlice: "slice" => string::call;
            StringSplit: "split" => string::call;
            StringStartsWith: "startsWith" => string::call;
            StringReplace: "replace" => string::call;
            StringPadStart: "padStart" => string::call;
        }
        Number {
            NumberToFixed: "toFixed" => number::call;
            NumberToString: "toString" => number::call;
        }
        Set {
            SetAdd: "add" => set::call;
            SetHas: "has" => set::call;
            SetDelete: "delete" => set::call;
            SetClear: "clear" => set::call;
        }
        Generator {
            GeneratorNext: "next" => generator::call;
        }
    }
    statics {
        Array {
            ArrayFrom: "from" => array::call;
            ArrayIsArray: "isArray" => array::call;
        }
    }
    internal {
        HostReadFile => host::call;
        HostListFiles => host::call;
        HostEvaluateSource => host::call;
    }
}

macro_rules! define_math {
    ($($variant:ident: $name:literal;)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum MathFunction { $($variant,)* }

        const MATH_METHODS: &[(&str, NativeFunction)] = &[
            $(($name, NativeFunction::Math(MathFunction::$variant)),)*
        ];
    };
}

define_math! {
    Random: "random";
    Abs: "abs";
    Ceil: "ceil";
    Floor: "floor";
    Round: "round";
    Trunc: "trunc";
    Sqrt: "sqrt";
    Sin: "sin";
    Cos: "cos";
    Tan: "tan";
    Asin: "asin";
    Acos: "acos";
    Atan: "atan";
    Atan2: "atan2";
    Exp: "exp";
    Log: "log";
    Pow: "pow";
    Min: "min";
    Max: "max";
    Imul: "imul";
    Sign: "sign";
}
