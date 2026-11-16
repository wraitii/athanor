use super::*;
use crate::Engine;
use crate::value::Value;

fn new_engine() -> Result<Engine, Error> {
    Engine::with_options(EngineOptions {
        background_compile: false,
        ..Default::default()
    })
}

fn code(engine: &Engine, name: &str) -> u64 {
    engine
        .global()
        .heap()
        .closure_code(engine.global().get_raw(name).unwrap())
        .unwrap()
        .0
}

fn entry(engine: &Engine, code: u64) -> jit::entries::Entry {
    if !engine
        .tiering
        .variants
        .borrow()
        .get(VariantId::ordinary(code))
        .is_some()
    {
        engine.tiering.entry(code).unwrap().unwrap();
    }
    engine
        .tiering
        .variants
        .borrow()
        .get(VariantId::ordinary(code))
        .unwrap()
}

#[test]
fn ordinary_functions_compile_on_first_call_across_sources() {
    let mut engine = new_engine().unwrap();
    engine
        .compile(
            "function first(x) { return x + 2; } function unused() { return 7; }",
            "first.js",
        )
        .unwrap();
    let first = code(&engine, "first");
    let unused = code(&engine, "unused");
    assert!(
        engine
            .tiering
            .variants
            .borrow()
            .get(VariantId::ordinary(first))
            .is_none()
    );
    engine
        .compile("function second() { return 40; }", "second.js")
        .unwrap();
    assert_eq!(
        engine
            .call("first", &[Value::number(40.0)])
            .unwrap()
            .value(),
        Value::number(42.0)
    );
    assert!(
        engine
            .tiering
            .variants
            .borrow()
            .get(VariantId::ordinary(first))
            .is_some()
    );
    assert_eq!(
        engine.call("second", &[]).unwrap().value(),
        Value::number(40.0)
    );
    assert!(
        engine
            .tiering
            .variants
            .borrow()
            .get(VariantId::ordinary(unused))
            .is_none()
    );
}

#[test]
fn closure_creation_does_not_compile_its_body() {
    let mut engine = new_engine().unwrap();
    engine
        .compile("function make(x) { return () => x; }", "closure.js")
        .unwrap();
    let closure = engine.call("make", &[Value::number(42.0)]).unwrap();
    let id = engine
        .global()
        .heap()
        .closure_code(closure.value())
        .unwrap()
        .0;
    assert!(
        engine
            .tiering
            .variants
            .borrow()
            .get(VariantId::ordinary(id))
            .is_none()
    );
    assert!(engine.tiering.compiler.borrow().has_lazy(id));
}

#[test]
fn method_feedback_reemits_once_and_guards_replacements() {
    let mut engine = new_engine().unwrap();
    engine
        .compile("Math.square=function(x){return x*x;};", "math.js")
        .unwrap();
    engine
        .compile(
            r#"
            function distance(x, y) { return Math.square(x) + Math.square(y); }
            function replace() { Math.square=function(x){return x+10;}; }
        "#,
            "distance.js",
        )
        .unwrap();
    let code = code(&engine, "distance");
    let original = entry(&engine, code);
    assert_eq!(
        engine
            .call("distance", &[Value::number(3.0), Value::number(4.0)])
            .unwrap()
            .value(),
        Value::number(25.0)
    );
    assert_eq!(entry(&engine, code), original);
    assert_eq!(
        engine
            .call("distance", &[Value::number(3.0), Value::number(4.0)])
            .unwrap()
            .value(),
        Value::number(25.0)
    );
    let optimized = entry(&engine, code);
    assert_ne!(original, optimized);
    engine.call("replace", &[]).unwrap();
    let mut entries = Vec::new();
    for _ in 0..6 {
        assert_eq!(
            engine
                .call("distance", &[Value::number(3.0), Value::number(4.0)])
                .unwrap()
                .value(),
            Value::number(27.0)
        );
        entries.push(entry(&engine, code));
    }
    assert!(entries[3..].iter().all(|&settled| settled == entries[2]));
}

#[test]
fn feedback_keeps_captures_receivers_and_heap_returns() {
    let mut engine = new_engine().unwrap();
    engine
        .compile(
            r#"
            function make(n) { return function(x) { return {value:this.base+n+x}; }; }
            var first={base:10, run:make(2)};
            var second={base:20, run:make(3)};
            function invoke(o) { return o.run(4).value; }
        "#,
            "captures.js",
        )
        .unwrap();
    let code = code(&engine, "invoke");
    let original = entry(&engine, code);
    for _ in 0..4 {
        for (name, expected) in [("first", 16.0), ("second", 27.0)] {
            let receiver = engine.global().get_raw(name).unwrap();
            assert_eq!(
                engine.call("invoke", &[receiver]).unwrap().value(),
                Value::number(expected)
            );
            engine.drain_disposals(1000);
        }
    }
    assert_ne!(entry(&engine, code), original);
}

#[test]
fn previous_entry_fallback_preserves_arguments_heap_returns_and_errors() {
    for invoke in [
        "function invoke(x){return object.run(x);}",
        "function invoke(){return object.run(arguments[0]);}",
    ] {
        let mut engine = new_engine().unwrap();
        engine.compile(&format!(
            "var object={{run:function(x){{return x;}}}}; {invoke} function replace(){{object.run=function(x){{return {{value:x}};}};}} function fail(){{object.run=function(x){{throw x;}};}} function restore(){{object.run=function(x){{return x;}};}}"
        ), "previous-entry.js").unwrap();
        let code = code(&engine, "invoke");
        let original = entry(&engine, code);
        let input = engine.global().heap().object().unwrap();
        for _ in 0..3 {
            assert_eq!(engine.call("invoke", &[input]).unwrap().value(), input);
        }
        assert_ne!(entry(&engine, code), original);
        engine.call("replace", &[]).unwrap();
        for _ in 0..3 {
            let result = engine.call("invoke", &[input]).unwrap();
            engine.drain_disposals(1000);
            assert_eq!(
                engine.global().heap().property(result.value(), "value"),
                Some(input)
            );
        }
        engine.call("fail", &[]).unwrap();
        assert!(engine.call("invoke", &[input]).is_err());
        engine.call("restore", &[]).unwrap();
        assert_eq!(engine.call("invoke", &[input]).unwrap().value(), input);
        engine.global().heap().release(input);
    }
}

#[test]
fn feedback_preserves_foreign_source_errors_and_recovery() {
    let mut engine = new_engine().unwrap();
    engine
        .compile(
            "var object={run:function(x){return x.value;}};",
            "callee.js",
        )
        .unwrap();
    engine
        .compile("function invoke(x){return object.run(x);}", "caller.js")
        .unwrap();
    let object = engine.global().heap().object().unwrap();
    engine
        .global()
        .heap()
        .set_property(object, "value", Value::number(7.0));
    assert_eq!(
        engine.call("invoke", &[object]).unwrap().value(),
        Value::number(7.0)
    );
    let error = engine
        .call("invoke", &[Value::NULL])
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("callee.js:"), "{error}");
    assert_eq!(
        engine.call("invoke", &[object]).unwrap().value(),
        Value::number(7.0)
    );
    engine.global().heap().release(object);
}

#[test]
fn constructor_body_feedback_survives_prototype_replacement() {
    let mut engine = new_engine().unwrap();
    engine
        .compile(
            r#"
            function C(x){this.set(x);}
            C.prototype.set=function(x){this.value=x;};
            function make(K,x){return new K(x).value;}
            function replace(){C.prototype.set=function(x){this.value=x+10;};}
        "#,
            "constructors.js",
        )
        .unwrap();
    let code = code(&engine, "make");
    let original = entry(&engine, code);
    let constructor = engine.global().get("C").unwrap();
    let args = [constructor.value(), Value::number(3.0)];
    for _ in 0..2 {
        assert_eq!(
            engine.call("make", &args).unwrap().value(),
            Value::number(3.0)
        );
    }
    assert_ne!(entry(&engine, code), original);
    engine.call("replace", &[]).unwrap();
    assert_eq!(
        engine.call("make", &args).unwrap().value(),
        Value::number(13.0)
    );
}

#[test]
fn recursive_dispatch_can_replace_code_with_old_frames_active() {
    let mut engine = new_engine().unwrap();
    engine
        .compile(
            r#"
            var object={run:function(x){return x+1;}};
            function recurse(n){let x=object.run(n);if(n>0)return x+recurse(n-1);return x;}
        "#,
            "recursive.js",
        )
        .unwrap();
    let code = code(&engine, "recurse");
    let original = entry(&engine, code);
    assert_eq!(
        engine
            .call("recurse", &[Value::number(5.0)])
            .unwrap()
            .value(),
        Value::number(21.0)
    );
    assert_ne!(entry(&engine, code), original);
    assert_eq!(
        engine
            .call("recurse", &[Value::number(5.0)])
            .unwrap()
            .value(),
        Value::number(21.0)
    );
}

#[test]
fn property_integer_feedback_preserves_type_changes() {
    let mut engine = new_engine().unwrap();
    engine.compile(
        "var point={x:2}; function sum(n){let total=0;for(let i=0;i<n;++i)total+=point.x+i;return total;} function replace(){point.x='3';}",
        "property-numbers.js",
    ).unwrap();
    let code = code(&engine, "sum");
    let original = entry(&engine, code);
    for _ in 0..3 {
        assert_eq!(
            engine.call("sum", &[Value::number(3.0)]).unwrap().value(),
            Value::number(9.0)
        );
    }
    assert_ne!(entry(&engine, code), original);
    engine.call("replace", &[]).unwrap();
    let result = engine.call("sum", &[Value::number(3.0)]).unwrap();
    let expected = engine.string("0303132").unwrap();
    assert!(
        engine
            .global()
            .heap()
            .strings_equal(result.value(), expected.value())
    );
}

#[test]
fn ineligible_feedback_keeps_the_existing_entry() {
    let mut engine = new_engine().unwrap();
    engine
        .compile(
            "function run(...values){return values[0];}function invoke(x){return run(x);}",
            "ineligible.js",
        )
        .unwrap();
    let code = code(&engine, "invoke");
    let original = entry(&engine, code);
    for _ in 0..3 {
        assert_eq!(
            engine
                .call("invoke", &[Value::number(7.0)])
                .unwrap()
                .value(),
            Value::number(7.0)
        );
        assert_eq!(entry(&engine, code), original);
    }
}

#[test]
fn ineligible_method_feedback_keeps_the_existing_entry() {
    let mut engine = new_engine().unwrap();
    engine.compile(
        "var object={run:function(...values){return values[0];}};function invoke(x){return object.run(x);}",
        "ineligible_method.js",
    ).unwrap();
    let code = code(&engine, "invoke");
    let original = entry(&engine, code);
    for _ in 0..3 {
        assert_eq!(
            engine
                .call("invoke", &[Value::number(7.0)])
                .unwrap()
                .value(),
            Value::number(7.0)
        );
        assert_eq!(entry(&engine, code), original);
    }
}

#[test]
fn shape_feedback_respecializes_property_reads() {
    let mut engine = new_engine().unwrap();
    engine
        .compile(
            "var object={a:3,b:4};function read(){return object.a*object.b;}",
            "shape_feedback.js",
        )
        .unwrap();
    let code = code(&engine, "read");
    let original = entry(&engine, code);
    let mut entries = Vec::new();
    for _ in 0..4 {
        assert_eq!(
            engine.call("read", &[]).unwrap().value(),
            Value::number(12.0)
        );
        entries.push(entry(&engine, code));
    }
    assert_ne!(entries[1], original);
    assert!(entries[1..].windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn entry_guards_specialize_nested_reads_and_guard_replacements() {
    let mut engine = new_engine().unwrap();
    engine.compile(
        "var matrix=[Float64Array.from([1.5,2.5])]; function read(i,j){return matrix[i][j];} function replace(){matrix[0]=[7,8];}",
        "get_feedback.js",
    ).unwrap();
    let code = code(&engine, "read");
    let original = entry(&engine, code);
    let args = [Value::number(0.0), Value::number(1.0)];
    assert_eq!(
        engine.call("read", &args).unwrap().value(),
        Value::number(2.5)
    );
    assert_eq!(
        engine.call("read", &args).unwrap().value(),
        Value::number(2.5)
    );
    let optimized = entry(&engine, code);
    assert_ne!(optimized, original);
    assert_eq!(
        engine
            .call("read", &[Value::number(0.0), Value::number(-1.0)])
            .unwrap()
            .value(),
        Value::UNDEFINED
    );
    for index in [0.5, f64::NAN, f64::INFINITY, 2.0] {
        assert_eq!(
            engine
                .call("read", &[Value::number(0.0), Value::number(index)])
                .unwrap()
                .value(),
            Value::UNDEFINED
        );
    }
    let key = engine.string("0").unwrap();
    assert_eq!(
        engine
            .call("read", &[Value::number(0.0), key.value()])
            .unwrap()
            .value(),
        Value::number(1.5)
    );
    engine.call("replace", &[]).unwrap();
    assert_eq!(
        engine.call("read", &args).unwrap().value(),
        Value::number(8.0)
    );
}

#[test]
fn get_feedback_specializes_arrays_and_preserves_heap_results() {
    let mut engine = new_engine().unwrap();
    engine.compile(
        "var holder={values:new Array(3)};holder.values[0]={value:3};holder.values[2]=9; function read(i){return holder.values[i];} function replace(){holder.values={0:7};}",
        "array_feedback.js",
    ).unwrap();
    let code = code(&engine, "read");
    let original = entry(&engine, code);
    for _ in 0..2 {
        assert_eq!(
            engine.call("read", &[Value::number(2.0)]).unwrap().value(),
            Value::number(9.0)
        );
    }
    assert_ne!(entry(&engine, code), original);
    for index in [1.0, 3.0] {
        assert_eq!(
            engine
                .call("read", &[Value::number(index)])
                .unwrap()
                .value(),
            Value::UNDEFINED
        );
    }
    for index in [
        -1.0,
        0.5,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        f64::MIN,
    ] {
        assert!(engine.call("read", &[Value::number(index)]).is_err());
    }
    let key = engine.string("2").unwrap();
    assert_eq!(
        engine.call("read", &[key.value()]).unwrap().value(),
        Value::number(9.0)
    );
    let retained = engine.call("read", &[Value::number(0.0)]).unwrap();
    engine.call("replace", &[]).unwrap();
    engine.drain_disposals(1000);
    assert_eq!(
        engine.global().heap().property(retained.value(), "value"),
        Some(Value::number(3.0))
    );
    assert_eq!(
        engine.call("read", &[Value::number(0.0)]).unwrap().value(),
        Value::number(7.0)
    );
}

#[test]
fn array_read_guards_follow_layout_length_and_storage_changes() {
    let mut engine = new_engine().unwrap();
    engine
        .compile(
            "var values=new Array(3);values[0]=5;values[2]=9;
         function read(i){return values[i];}
         function fill(){values[1]=7;}
         function shrink(){values.length=1;}
         function grow(){for(let i=1;i<100;++i)values.push(i);}
         function replace(){values=new Array(3);values[2]=11;}",
            "array_layout_feedback.js",
        )
        .unwrap();
    let code = code(&engine, "read");
    let original = entry(&engine, code);
    for _ in 0..3 {
        assert_eq!(
            engine.call("read", &[Value::number(2.0)]).unwrap().value(),
            Value::number(9.0)
        );
    }
    assert_ne!(entry(&engine, code), original);
    assert_eq!(
        engine.call("read", &[Value::number(1.0)]).unwrap().value(),
        Value::UNDEFINED
    );
    engine.call("fill", &[]).unwrap();
    assert_eq!(
        engine.call("read", &[Value::number(1.0)]).unwrap().value(),
        Value::number(7.0)
    );
    assert_eq!(
        engine.call("read", &[Value::number(-0.0)]).unwrap().value(),
        Value::number(5.0)
    );
    engine.call("shrink", &[]).unwrap();
    assert_eq!(
        engine.call("read", &[Value::number(2.0)]).unwrap().value(),
        Value::UNDEFINED
    );
    engine.call("grow", &[]).unwrap();
    assert_eq!(
        engine.call("read", &[Value::number(99.0)]).unwrap().value(),
        Value::number(99.0)
    );
    engine.call("replace", &[]).unwrap();
    assert_eq!(
        engine.call("read", &[Value::number(0.0)]).unwrap().value(),
        Value::UNDEFINED
    );
    assert_eq!(
        engine.call("read", &[Value::number(2.0)]).unwrap().value(),
        Value::number(11.0)
    );
}

#[test]
fn get_feedback_specializes_typed_rows_inside_loops() {
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            background_compile: false,
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine.compile(
            "var height=[Float64Array.from([1.5,2.5]),Float64Array.from([3.5,4.5])]; function sum(){let result=0;for(let x=0;x<height.length;++x)for(let y=0;y<height[x].length;++y)result+=height[x][y];return result;} function replace(){height[1]=[5,6];}",
            "row_feedback.js",
        ).unwrap();
        let code = code(&engine, "sum");
        let original = entry(&engine, code);
        for _ in 0..3 {
            assert_eq!(
                engine.call("sum", &[]).unwrap().value(),
                Value::number(12.0)
            );
        }
        assert_eq!(entry(&engine, code) != original, ssa_opt);
        engine.call("replace", &[]).unwrap();
        assert_eq!(
            engine.call("sum", &[]).unwrap().value(),
            Value::number(15.0)
        );
    }
}

#[test]
fn entry_guards_specialize_named_typed_array_properties() {
    let mut engine = new_engine().unwrap();
    engine.compile(
        "var holder={values:Int32Array.from([3,4])}; function read(i){return holder.values[i];} function replace(){holder.values=Float64Array.from([5.5,6.5]);}",
        "property_feedback.js",
    ).unwrap();
    let code = code(&engine, "read");
    let original = entry(&engine, code);
    let args = [Value::number(1.0)];
    for _ in 0..3 {
        assert_eq!(
            engine.call("read", &args).unwrap().value(),
            Value::number(4.0)
        );
    }
    let optimized = entry(&engine, code);
    assert_ne!(optimized, original);
    engine.call("replace", &[]).unwrap();
    assert_eq!(
        engine.call("read", &args).unwrap().value(),
        Value::number(6.5)
    );
    assert_eq!(entry(&engine, code), optimized);
}

#[test]
fn disabling_get_refinement_keeps_call_feedback() {
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: false,
        get_refinement: false,
        ..Default::default()
    })
    .unwrap();
    engine.compile(
        "var holder={values:[1,2],run:function(){return 7;}}; function read(i){return holder.values[i];} function invoke(){return holder.run();}",
        "disabled_get_feedback.js",
    ).unwrap();
    let read_code = code(&engine, "read");
    let original = entry(&engine, read_code);
    for _ in 0..3 {
        assert_eq!(
            engine.call("read", &[Value::number(1.0)]).unwrap().value(),
            Value::number(2.0)
        );
        assert_eq!(entry(&engine, read_code), original);
        assert!(!engine.tiering.recompiles.borrow_mut().take(read_code));
    }
    let invoke_code = code(&engine, "invoke");
    let original = entry(&engine, invoke_code);
    for _ in 0..2 {
        assert_eq!(
            engine.call("invoke", &[]).unwrap().value(),
            Value::number(7.0)
        );
    }
    assert_ne!(entry(&engine, invoke_code), original);
}

#[test]
fn feedback_is_disabled_with_semantic_optimization() {
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: false,
        ssa_opt: false,
        ..Default::default()
    })
    .unwrap();
    engine
        .compile(
            "var o={f:function(x){return x*x;}};function run(){return o.f(3);}",
            "disabled.js",
        )
        .unwrap();
    let code = code(&engine, "run");
    let original = entry(&engine, code);
    for _ in 0..3 {
        assert_eq!(engine.call("run", &[]).unwrap().value(), Value::number(9.0));
    }
    assert_eq!(entry(&engine, code), original);
    assert!(!engine.tiering.recompiles.borrow_mut().take(code));
}

fn background_engine() -> Engine {
    Engine::with_options(EngineOptions {
        background_compile: true,
        ..Default::default()
    })
    .unwrap()
}

fn osr_engine(background_compile: bool) -> Engine {
    Engine::with_options(EngineOptions {
        osr: true,
        background_compile,
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn captured_writes_retain_replacements_and_release_old_values() {
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: false,
        inlining: false,
        ..Default::default()
    })
    .unwrap();
    engine
        .compile(
            "var writer;
            function setup() {
                let value = [17];
                writer = function(next) { value = next; return value; };
                return value;
            }
            function replacement() { return [29]; }
            function replace(next) { return writer(next); }",
            "capture-stores.js",
        )
        .unwrap();
    let original = engine.call("setup", &[]).unwrap();
    let old = original.value();
    for _ in 0..3 {
        assert_eq!(engine.call("replace", &[old]).unwrap().value(), old);
        engine.drain_disposals(1000);
        assert!(engine.global().heap().valid(old));
    }
    let replacement = engine.call("replacement", &[]).unwrap();
    let new = replacement.value();
    assert_eq!(engine.call("replace", &[new]).unwrap().value(), new);
    drop(original);
    drop(replacement);
    engine.drain_disposals(1000);
    assert!(!engine.global().heap().valid(old));
    assert!(engine.global().heap().valid(new));
    engine.call("replace", &[Value::UNDEFINED]).unwrap();
    engine.drain_disposals(1000);
    assert!(!engine.global().heap().valid(new));
}

#[test]
fn loop_replacement_preserves_shared_capture_cells() {
    let source = r#"
        function Counter() {
            var limit = 200;
            var total = 0;
            var values = [1, 2, 3];
            this.change = function() { limit = 220; values = [0.5, 1.5, 2.5]; };
            this.read = function() { return total; };
            this.run = function(callback) {
                for (let i = 0; i < limit; ++i) {
                    total += values[i % 3] / 1;
                    if (i === 80) callback();
                }
                return total;
            };
        }
        var first = new Counter();
        var second = new Counter();
        function main() {
            var result = first.run(function() { first.change(); });
            return result + first.read() + second.read();
        }
    "#;
    let run = |osr| {
        let mut engine = Engine::with_options(EngineOptions {
            osr,
            background_compile: false,
            inlining: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "capture-osr.js").unwrap();
        let result = engine.call("main", &[]).unwrap().value();
        let transfers = engine.tiering.compiler.borrow().osr_transfers();
        (result, transfers)
    };
    let (expected, _) = run(false);
    let (actual, transfers) = run(true);
    assert_eq!(expected, Value::number(739.0));
    assert_eq!(actual, expected);
    assert!(transfers > 0);
}

#[test]
fn numeric_feedback_recompiles_without_read_or_call_specialization() {
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    engine
        .compile(
            "function sum(x){let result=1/x;for(let i=0;i<2;++i)result+=x;return result;}",
            "integer-feedback.js",
        )
        .unwrap();
    let code = code(&engine, "sum");
    let original = entry(&engine, code);
    for _ in 0..3 {
        assert_eq!(
            engine.call("sum", &[Value::number(2.0)]).unwrap().value(),
            Value::number(4.5)
        );
    }
    assert_ne!(entry(&engine, code), original);
    for value in [
        0.5,
        -0.0,
        0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        9007199254740992.0,
    ] {
        let expected = (1.0 / value + value) + value;
        assert_eq!(
            engine.call("sum", &[Value::number(value)]).unwrap().value(),
            Value::number(expected)
        );
    }
}

#[test]
fn osr_does_not_add_observation_sites() {
    let source = "function main(){const holder={x:2};let result=0;for(let j=0;j<160;++j){if(j===100)holder.x=0.5;const x=holder.x+1;let sum=0;for(let i=0;i<3;++i)sum+=(x+i)*1;result+=holder.x*sum+(1-holder.x)*sum;}return result;}";
    let mut generic = Engine::with_options(EngineOptions {
        osr: false,
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    generic.compile(source, "osr-observations.js").unwrap();
    let expected = generic.call("main", &[]).unwrap().value();
    let mut engine = osr_engine(false);
    engine.compile(source, "osr-observations.js").unwrap();
    assert_eq!(engine.call("main", &[]).unwrap().value(), expected);
    assert!(engine.tiering.compiler.borrow().osr_transfers() > 0);
    let profile = engine.tiering.profile.borrow();
    let generic_profile = generic.tiering.profile.borrow();
    let numbers: Vec<_> = profile
        .reads()
        .filter(|((_, kind), _)| *kind == jit::profile::ReadKind::Number)
        .collect();
    assert!(!numbers.is_empty());
    for ((site, kind), _) in numbers {
        assert!(generic_profile.reads().any(|((generic, generic_kind), _)| {
            generic.start == site.start && generic.end == site.end && generic_kind == kind
        }));
    }
    let shapes: Vec<_> = profile.shapes().collect();
    assert!(!shapes.is_empty());
    for ((site, property), _) in shapes {
        assert!(
            generic_profile
                .shapes()
                .any(|((generic, generic_property), _)| {
                    generic.start == site.start
                        && generic.end == site.end
                        && generic_property == property
                })
        );
    }
}

#[test]
fn numeric_osr_feedback_preserves_fractional_guard_fallbacks() {
    let source = "function main(){const holder={x:2};let result=0;for(let j=0;j<160;++j){if(j===100)holder.x=0.5;const x=holder.x;let sum=0;for(let i=0;i<3;++i)sum+=(x+i)*1;result+=sum;}return result;}";
    let mut baseline = Engine::with_options(EngineOptions {
        ssa_opt: false,
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    baseline.compile(source, "numeric-osr.js").unwrap();
    let expected = baseline.call("main", &[]).unwrap().value();
    let mut engine = osr_engine(false);
    engine.compile(source, "numeric-osr.js").unwrap();
    assert_eq!(engine.call("main", &[]).unwrap().value(), expected);
    assert!(engine.tiering.compiler.borrow().osr_transfers() > 0);
}

#[test]
fn typed_read_feedback_preserves_kind_and_index_checks() {
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    engine.compile("var f32=Float32Array.from([1.5,2.5]);var f64=Float64Array.from([7.25]);function read(a,k){return a[k];}", "typed-read.js").unwrap();
    let f32 = engine.global().get_raw("f32").unwrap();
    assert_eq!(
        engine
            .call("read", &[f32, Value::number(0.0)])
            .unwrap()
            .value(),
        Value::number(1.5)
    );
    for index in [-1.0, 0.5, 2.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            engine
                .call("read", &[f32, Value::number(index)])
                .unwrap()
                .value(),
            Value::UNDEFINED
        );
    }
    assert_eq!(
        engine
            .call("read", &[f32, Value::number(-0.0)])
            .unwrap()
            .value(),
        Value::number(1.5)
    );
    let f64 = engine.global().get_raw("f64").unwrap();
    assert_eq!(
        engine
            .call("read", &[f64, Value::number(0.0)])
            .unwrap()
            .value(),
        Value::number(7.25)
    );
}

#[test]
fn loop_replacement_preserves_multiple_accumulators_and_iteration_order() {
    let source = "function main(){const rows=[Float32Array.from([1,2,3]),Float32Array.from([4,5,6])];let total=0;for(let y=0;y<200;++y){let weighted=0;let weights=0;for(let k=0;k<3;++k){if(y%2 && k===1)continue;weighted+=rows[y%2][k]/(k+1);weights+=1/(k+1);}total+=weighted/weights;}return total;}";
    let mut baseline = Engine::with_options(EngineOptions {
        osr: false,
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    baseline.compile(source, "smoothing.js").unwrap();
    let expected = baseline.call("main", &[]).unwrap().value();
    let mut engine = osr_engine(false);
    engine.compile(source, "smoothing.js").unwrap();
    assert_eq!(engine.call("main", &[]).unwrap().value(), expected);
    assert!(engine.tiering.compiler.borrow().osr_transfers() > 0);
}

#[test]
fn loop_replacement_keeps_effects_and_guard_failure_continuations() {
    let mut engine = osr_engine(false);
    engine.compile("function main(){const rows=[Float32Array.from([1,2,3])];const effects={n:0};let total=0;for(let y=0;y<200;++y){if(y===60)rows[0]={0:4,1:5,2:6};let sum=0;for(let k=0;k<3;++k){effects.n+=1;sum+=rows[0][k]/1;}total+=sum;}return total+effects.n;}", "effects.js").unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(3060.0)
    );
    assert!(engine.tiering.compiler.borrow().osr_transfers() > 0);
}

#[test]
fn loop_replacement_propagates_errors_and_releases_its_frame() {
    let mut engine = osr_engine(false);
    engine.compile("function main(){const rows=[Float32Array.from([1,2,3])];let total=0;for(let y=0;y<200;++y){if(y===60)rows[0]=null;let sum=0;for(let k=0;k<3;++k)sum+=rows[0][k]/1;total+=sum;}return total;} function good(){return 42;}", "loop-error.js").unwrap();
    let error = engine.call("main", &[]).err().unwrap();
    assert!(error.to_string().contains("loop-error.js"), "{error}");
    assert!(engine.tiering.compiler.borrow().osr_transfers() > 0);
    assert_eq!(
        engine.call("good", &[]).unwrap().value(),
        Value::number(42.0)
    );
}

#[test]
fn background_loop_replacement_runs_before_the_first_function_return() {
    let mut engine = osr_engine(true);
    engine.compile("function main(){const rows=[[1,2,3]];let total=0;for(let y=0;y<2000000;++y){let sum=0;for(let k=0;k<3;++k)sum+=rows[0][k]/1;total+=sum;}return total;}", "background-loop.js").unwrap();
    let code = code(&engine, "main");
    entry(&engine, code);
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(12000000.0)
    );
    assert!(engine.tiering.compiler.borrow().osr_transfers() > 0);
    finish_compilation(&engine);
}

#[test]
fn osr_request_queues_the_enclosing_function_before_return() {
    let mut engine = osr_engine(true);
    engine
        .compile(
        "var values=[1,2,3];function hot(xs){let total=0;for(let i=0;i<3;++i)total+=xs[i]/1;return total;}",
            "osr-warmup.js",
        )
        .unwrap();
    let code = code(&engine, "hot");
    entry(&engine, code);
    let (site, location, kind) = engine
        .tiering
        .compiler
        .borrow()
        .osr_read_site(code)
        .unwrap();
    let site = unsafe { &*site };
    let mut cache = jit::profile::ObservationCache {
        complete: 0,
        last_callee: 0,
        failures: 0,
        owner: code,
        site: location,
    };
    engine
        .tiering
        .observe_get(&mut cache, jit::profile::ReadType::Array, kind);
    assert!(!engine.tiering.direct.warmed(code));
    site.entries.set(jit::osr::TICK);
    engine.tiering.osr_entry(site);
    assert!(!engine.tiering.direct.warmed(code));
    {
        let worker = engine.tiering.worker.borrow();
        let worker = worker.as_ref().unwrap();
        assert!(
            worker
                .in_flight
                .contains_key(&jit::recompile::Key::Loop(site as *const _ as usize))
        );
        assert!(
            worker
                .in_flight
                .contains_key(&jit::recompile::Key::ordinary(code))
        );
    }
    finish_compilation(&engine);
    engine.tiering.request_compilation(code);
    let values = engine.global().get_raw("values").unwrap();
    for _ in 0..jit::entries::RECOMPILE_WARMUP {
        assert_eq!(
            engine.call("hot", &[values]).unwrap().value(),
            Value::number(6.0)
        );
    }
    assert!(
        engine
            .tiering
            .worker
            .borrow()
            .as_ref()
            .unwrap()
            .in_flight
            .contains_key(&jit::recompile::Key::ordinary(code))
    );
    finish_compilation(&engine);
}

fn wait_ready(engine: &Engine) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while engine.tiering.compilation_state.load(Ordering::Acquire) & jit::recompile::READY == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "compilation did not complete"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn finish_compilation(engine: &Engine) {
    loop {
        if engine
            .tiering
            .worker
            .borrow()
            .as_ref()
            .is_none_or(|worker| worker.in_flight.is_empty())
        {
            break;
        }
        wait_ready(engine);
        engine.tiering.poll_compilation();
    }
}

#[test]
fn background_pool_compiles_multiple_targets() {
    for compiler_threads in [1, 2, 3] {
        let mut engine = Engine::with_options(EngineOptions {
            background_compile: true,
            compiler_threads,
            ..Default::default()
        })
        .unwrap();
        let source = (0..4)
            .map(|i| format!("function read{i}(a){{return a.items[0];}}"))
            .collect::<String>();
        engine
            .compile(&format!("var holder={{items:[3]}};{source}"), "pool.js")
            .unwrap();
        let holder = engine.global().get_raw("holder").unwrap();
        let originals: Vec<_> = (0..4)
            .map(|i| {
                let name = format!("read{i}");
                let code = code(&engine, &name);
                (name, code, entry(&engine, code))
            })
            .collect();
        for (name, _, _) in &originals {
            for _ in 0..=jit::entries::RECOMPILE_WARMUP {
                assert_eq!(
                    engine.call(name, &[holder]).unwrap().value(),
                    Value::number(3.0)
                );
            }
        }
        finish_compilation(&engine);
        for (name, code, original) in originals {
            assert_ne!(entry(&engine, code), original);
            assert_eq!(
                engine.call(&name, &[holder]).unwrap().value(),
                Value::number(3.0)
            );
        }
    }
}

#[test]
fn baseline_bodies_upgrade_without_specialization_changes() {
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    engine
        .compile("function f(a){return a+1;}", "tier-upgrade.js")
        .unwrap();
    let code = code(&engine, "f");
    let mut job = engine
        .tiering
        .compiler
        .borrow()
        .lazy_compilation(code)
        .unwrap();
    job.tier = jit::CompilationTier::Baseline;
    let mut baseline = Compiler::new(
        engine.tiering.options.clone(),
        jit::CompilationTier::Baseline,
        engine.tiering.direct.address(),
        0,
        engine.tiering.recompile_log,
    )
    .unwrap();
    let replacement = baseline.recompile(job).unwrap().unwrap();
    let unpublished = baseline.take_unpublished(replacement);
    let jit::compiler::Installed {
        entry: original, ..
    } = engine
        .tiering
        .compiler
        .borrow_mut()
        .install_unpublished(unpublished)
        .unwrap();
    engine
        .tiering
        .variants
        .borrow_mut()
        .publish(VariantId::ordinary(code), original);
    assert_eq!(
        engine.call("f", &[Value::number(2.0)]).unwrap().value(),
        Value::number(3.0)
    );
    let job = engine
        .tiering
        .compiler
        .borrow()
        .recompilation(code, original)
        .unwrap();
    let replacement = engine
        .tiering
        .compiler
        .borrow_mut()
        .recompile(job)
        .unwrap()
        .unwrap();
    let jit::compiler::Installed {
        entry: optimized, ..
    } = engine.tiering.compiler.borrow_mut().install(replacement);
    engine
        .tiering
        .variants
        .borrow_mut()
        .publish(VariantId::ordinary(code), optimized);
    assert_ne!(original.code, optimized.code);
    assert_eq!(
        engine.call("f", &[Value::number(2.0)]).unwrap().value(),
        Value::number(3.0)
    );
    let job = engine
        .tiering
        .compiler
        .borrow()
        .recompilation(code, optimized)
        .unwrap();
    assert!(
        engine
            .tiering
            .compiler
            .borrow_mut()
            .recompile(job)
            .unwrap()
            .is_none()
    );
}

#[test]
fn workers_compile_independent_lazy_bodies_without_publication() {
    let engine = Engine::with_options(EngineOptions {
        background_compile: true,
        compiler_threads: 1,
        ..Default::default()
    })
    .unwrap();
    engine.compile_only(
        "function first(a){return a.x;} function second(a){return a.y;} function third(a){return a.z;}",
        "independent.js",
    ).unwrap();
    let count = engine.tiering.compiler.borrow().lazy_codes().len();
    assert_eq!(count, 3);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut service = engine.tiering.worker.borrow_mut();
    let service = service.as_mut().unwrap();
    for _ in 0..count {
        let completion = loop {
            assert!(
                std::time::Instant::now() < deadline,
                "compilation did not complete"
            );
            service.wait_for_completion().unwrap();
            if let Some(completion) = service.receive().unwrap() {
                break completion;
            }
        };
        assert!(completion.result.unwrap().unwrap().initial());
    }
    assert_eq!(engine.tiering.compiler.borrow().lazy_codes().len(), count);
}

#[test]
fn background_unlazification_drains_uncalled_bodies_across_sources() {
    let mut engine = background_engine();
    for source in [
        "function unused(a){return a.items[0];} function factory(a){return ()=>a;}",
        "function later(a){return a.items[1];}",
    ] {
        engine.compile(source, "unlazify.js").unwrap();
        finish_compilation(&engine);
        assert!(engine.tiering.compiler.borrow().lazy_codes().is_empty());
    }
}

#[test]
fn disabling_prefetch_compiles_only_demanded_functions() {
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: true,
        prefetch_compile: false,
        inlining: false,
        ..Default::default()
    })
    .unwrap();
    engine.compile("function caller(a){return callee(a);} function callee(a){return a.items[0];} function unused(a){return a.items[1];} var holder={items:[3,7]};", "demand.js").unwrap();
    let caller = code(&engine, "caller");
    let callee = code(&engine, "callee");
    let unused = code(&engine, "unused");
    assert!(engine.tiering.worker.borrow().is_none());
    assert_eq!(engine.tiering.compiler.borrow().lazy_codes().len(), 3);
    engine.tiering.entry(caller).unwrap().unwrap();
    finish_compilation(&engine);
    assert!(!engine.tiering.compiler.borrow().has_lazy(caller));
    assert!(engine.tiering.compiler.borrow().has_lazy(callee));
    assert!(engine.tiering.compiler.borrow().has_lazy(unused));
    let holder = engine.global().get_raw("holder").unwrap();
    assert_eq!(
        engine.call("caller", &[holder]).unwrap().value(),
        Value::number(3.0)
    );
    finish_compilation(&engine);
    assert!(!engine.tiering.compiler.borrow().has_lazy(callee));
    assert!(engine.tiering.compiler.borrow().has_lazy(unused));
}

#[test]
fn synchronous_compilation_prefetches_callees_exposed_by_inlining() {
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: true,
        recompile_log: true,
        ..Default::default()
    })
    .unwrap();
    engine.compile("function caller(a){return callee(a);} function callee(a){return leaf(a);} function leaf(a){return a.items[0];}", "prefetch.js").unwrap();
    let caller = code(&engine, "caller");
    let callee = code(&engine, "callee");
    let leaf = code(&engine, "leaf");
    engine.tiering.entry(caller).unwrap().unwrap();
    finish_compilation(&engine);
    assert!(!engine.tiering.compiler.borrow().has_lazy(callee));
    assert!(!engine.tiering.compiler.borrow().has_lazy(leaf));
    assert!((2..=3).contains(&engine.tiering.prefetch_stats.borrow().installed));
    assert_eq!(engine.tiering.prefetch_stats.borrow().used, 0);
    engine
        .compile("var holder={items:[3]};", "holder.js")
        .unwrap();
    let holder = engine.global().get_raw("holder").unwrap();
    let entry = engine
        .tiering
        .variants
        .borrow()
        .get(VariantId::ordinary(callee))
        .unwrap();
    let callable = engine.global().get_raw("callee").unwrap();
    let result =
        unsafe { jit::call(entry, &[holder], &engine, callable, Value::UNDEFINED) }.unwrap();
    assert_eq!(result, Value::number(3.0));
    engine.global().heap().release(result);
    finish_compilation(&engine);
    engine.tiering.log_prefetch_uses();
    assert_eq!(engine.tiering.prefetch_stats.borrow().used, 1);
    engine.tiering.log_prefetch_uses();
    assert_eq!(engine.tiering.prefetch_stats.borrow().used, 1);
}

#[test]
fn synchronous_compilation_supersedes_an_unpublished_initial_entry() {
    let mut engine = background_engine();
    engine
        .compile("function f(a){return a.items[0];}", "takeover.js")
        .unwrap();
    let code = code(&engine, "f");
    let job = engine
        .tiering
        .compiler
        .borrow()
        .lazy_compilation(code)
        .unwrap();
    let mut worker = Compiler::new(
        engine.tiering.options.clone(),
        jit::CompilationTier::Optimized,
        engine.tiering.direct.address(),
        Arc::as_ptr(&engine.tiering.compilation_state) as usize,
        engine.tiering.recompile_log,
    )
    .unwrap();
    let replacement = worker.recompile(job).unwrap().unwrap();
    let unpublished = worker.take_unpublished(replacement);
    let original = engine.tiering.entry(code).unwrap().unwrap();
    assert!(
        engine
            .tiering
            .compiler
            .borrow_mut()
            .install_unpublished(unpublished)
            .is_none()
    );
    assert_eq!(engine.tiering.entry(code).unwrap(), Some(original));
}

#[test]
fn prefetched_closures_preserve_captures_and_lexical_this() {
    let mut engine = background_engine();
    let body = "total += a;".repeat(80);
    engine.compile(&format!("function caller(a){{const f=()=>{{let total=this.base+a;{body}return total;}};return f();}} var obj={{base:10,run:caller}}; function run(){{return obj.run(2);}}"), "prefetch-closure.js").unwrap();
    let caller = code(&engine, "caller");
    let jobs = engine.tiering.compiler.borrow().lazy_callees(caller);
    assert_eq!(jobs.len(), 1);
    let closure = u64::from(jobs[0].code.as_u32());
    engine.tiering.entry(caller).unwrap().unwrap();
    finish_compilation(&engine);
    assert!(!engine.tiering.compiler.borrow().has_lazy(closure));
    assert!(
        engine
            .tiering
            .variants
            .borrow()
            .get(VariantId::ordinary(closure))
            .unwrap()
            .lexical_this
    );
    assert_eq!(
        engine.call("run", &[]).unwrap().value(),
        Value::number(172.0)
    );
    finish_compilation(&engine);
}

#[test]
fn background_recompilation_starts_after_warmup_and_guards_replacements() {
    let mut engine = background_engine();
    engine.compile(
        "var holder={items:[3]}; function read(a){return a.items[0];} function replace(){holder.items={0:9};}",
        "read.js",
    ).unwrap();
    finish_compilation(&engine);
    let code = code(&engine, "read");
    let original = entry(&engine, code);
    let holder = engine.global().get_raw("holder").unwrap();
    for _ in 0..jit::entries::RECOMPILE_WARMUP - 1 {
        assert_eq!(
            engine.call("read", &[holder]).unwrap().value(),
            Value::number(3.0)
        );
        assert_eq!(entry(&engine, code), original);
        assert!(
            engine
                .tiering
                .worker
                .borrow()
                .as_ref()
                .unwrap()
                .in_flight
                .is_empty()
        );
    }
    assert_eq!(
        engine.call("read", &[holder]).unwrap().value(),
        Value::number(3.0)
    );
    assert!(
        engine
            .tiering
            .worker
            .borrow()
            .as_ref()
            .unwrap()
            .in_flight
            .contains_key(&jit::recompile::Key::ordinary(code))
    );
    engine
        .compile("function unrelated(){return 42;}", "later.js")
        .unwrap();
    finish_compilation(&engine);
    let optimized = entry(&engine, code);
    assert_ne!(optimized, original);
    assert_eq!(
        engine.call("read", &[holder]).unwrap().value(),
        Value::number(3.0)
    );
    assert_eq!(
        engine.tiering.compilation_state.load(Ordering::Relaxed) & jit::recompile::REQUESTED,
        0
    );
    engine.call("replace", &[]).unwrap();
    assert_eq!(
        engine.call("read", &[holder]).unwrap().value(),
        Value::number(9.0)
    );
    assert_eq!(entry(&engine, code), optimized);
}

#[test]
fn generated_dispatch_installs_background_results() {
    let mut engine = background_engine();
    let body = "n += a.extra;".repeat(40);
    engine.compile(
        &format!("var holder={{items:[3],extra:0}}; function read(a){{let n=0;{body}return a.items[0]+n;}} function drive(a){{return read(a);}}"),
        "dispatch.js",
    ).unwrap();
    finish_compilation(&engine);
    let read = code(&engine, "read");
    let original = entry(&engine, read);
    let holder = engine.global().get_raw("holder").unwrap();
    for _ in 0..jit::entries::RECOMPILE_WARMUP {
        engine.call("read", &[holder]).unwrap();
    }
    wait_ready(&engine);
    let drive = entry(&engine, code(&engine, "drive"));
    let callable = engine.global().get_raw("drive").unwrap();
    let value =
        unsafe { jit::call(drive, &[holder], &engine, callable, Value::UNDEFINED) }.unwrap();
    assert_eq!(value, Value::number(3.0));
    engine.global().heap().release(value);
    assert_ne!(entry(&engine, read), original);
    finish_compilation(&engine);
}

#[test]
fn background_warmup_counts_direct_calls_before_feedback() {
    let mut engine = background_engine();
    engine.compile(
        "var holder={items:[3]}; function read(a){if(!a)return 0;return a.items[0];} function drive(f,a){return f(a);}",
        "warmup.js",
    ).unwrap();
    let code = code(&engine, "read");
    let callable = engine.global().get_raw("read").unwrap();
    for _ in 0..jit::entries::RECOMPILE_WARMUP - 1 {
        assert_eq!(
            engine
                .call("drive", &[callable, Value::NULL])
                .unwrap()
                .value(),
            Value::number(0.0)
        );
    }
    assert!(!engine.tiering.direct.warmed(code));
    let holder = engine.global().get_raw("holder").unwrap();
    assert_eq!(
        engine.call("drive", &[callable, holder]).unwrap().value(),
        Value::number(3.0)
    );
    assert!(!engine.tiering.direct.warmed(code));
    assert!(
        engine
            .tiering
            .worker
            .borrow()
            .as_ref()
            .unwrap()
            .in_flight
            .contains_key(&jit::recompile::Key::ordinary(code))
    );
    finish_compilation(&engine);
}

#[test]
fn requests_during_background_compilation_are_coalesced_and_retained() {
    let mut engine = background_engine();
    engine
        .compile(
            "var holder={items:[3]}; function read(a){return a.items[0];}",
            "requests.js",
        )
        .unwrap();
    let code = code(&engine, "read");
    let holder = engine.global().get_raw("holder").unwrap();
    for _ in 0..jit::entries::RECOMPILE_WARMUP {
        engine.call("read", &[holder]).unwrap();
    }
    for _ in 0..3 {
        engine.tiering.request_compilation(code);
        engine.tiering.submit_compilation(code);
    }
    assert_eq!(
        engine
            .tiering
            .worker
            .borrow()
            .as_ref()
            .unwrap()
            .in_flight
            .len(),
        1
    );
    assert!(engine.tiering.recompiles.borrow().has_pending());
    finish_compilation(&engine);
    assert!(engine.tiering.recompiles.borrow().has_pending());
    for _ in 0..jit::entries::RECOMPILE_WARMUP - 1 {
        engine.call("read", &[holder]).unwrap();
    }
    assert!(engine.tiering.recompiles.borrow().has_pending());
    assert!(!engine.tiering.direct.warmed(code));
    engine.call("read", &[holder]).unwrap();
    finish_compilation(&engine);
    assert!(!engine.tiering.recompiles.borrow().has_pending());
    assert_eq!(
        engine.call("read", &[holder]).unwrap().value(),
        Value::number(3.0)
    );
}

#[test]
fn background_versions_preserve_captures_heap_returns_and_foreign_errors() {
    let mut engine = background_engine();
    engine.compile(
        "function make(offset){return function(x){return {value:this.base+offset+x.value};};} var object={base:10,run:make(2)}; var input={value:3}; function replace(){object.run=function(x){return x;};}",
        "callee.js",
    ).unwrap();
    engine
        .compile("function invoke(x){return object.run(x);}", "caller.js")
        .unwrap();
    let input = engine.global().get_raw("input").unwrap();
    let code = code(&engine, "invoke");
    let original = entry(&engine, code);
    let first = engine.call("invoke", &[input]).unwrap();
    for _ in 1..jit::entries::RECOMPILE_WARMUP {
        engine.call("invoke", &[input]).unwrap();
    }
    finish_compilation(&engine);
    assert_ne!(entry(&engine, code), original);
    let second = engine.call("invoke", &[input]).unwrap();
    engine.drain_disposals(1000);
    for result in [first, second] {
        assert_eq!(
            engine.global().heap().property(result.value(), "value"),
            Some(Value::number(15.0))
        );
    }
    let error = engine
        .call("invoke", &[Value::NULL])
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("callee.js:"), "{error}");
    engine.call("replace", &[]).unwrap();
    assert_eq!(engine.call("invoke", &[input]).unwrap().value(), input);
    finish_compilation(&engine);
}

#[cfg(test)]
mod call_variant_tests {
    use super::*;
    use crate::jit::variants::{Fact, Input};
    use crate::{Engine, Value};
    use std::sync::atomic::Ordering;

    fn engine(background_compile: bool) -> Engine {
        Engine::with_options(EngineOptions {
            inlining: false,
            call_variants: true,
            background_compile,
            ..Default::default()
        })
        .unwrap()
    }

    fn variants(engine: &Engine, name: &str) -> Vec<Arc<crate::jit::variants::Variant>> {
        let code = engine
            .global()
            .heap()
            .closure_code(engine.global().get_raw(name).unwrap())
            .unwrap()
            .0;
        engine.tiering.compiler.borrow().variants_for_function(code)
    }

    fn selected(engine: &Engine, variant: VariantId) -> Vec<VariantId> {
        engine
            .tiering
            .compiler
            .borrow()
            .specialization_plan(variant)
            .unwrap()
            .call_variants
            .iter()
            .map(|(_, variant)| *variant)
            .collect()
    }

    #[test]
    fn disabling_call_specialization_preserves_polymorphic_dispatch() {
        let mut engine = Engine::with_options(EngineOptions {
            inlining: false,
            call_specialization: false,
            call_variants: true,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile("function a(x){return x+x;} function b(x){return x*3;} function invoke(f,x){return f(x);}", "unspecialized-dispatch.js").unwrap();
        for _ in 0..jit::entries::RECOMPILE_WARMUP {
            for (name, expected) in [("a", 4.0), ("b", 6.0)] {
                let callable = engine.global().get_raw(name).unwrap();
                assert_eq!(
                    engine
                        .call("invoke", &[callable, Value::number(2.0)])
                        .unwrap()
                        .value(),
                    Value::number(expected)
                );
            }
        }
        let invoke = code(&engine, "invoke");
        engine.tiering.entry(invoke).unwrap();
        let selected = selected(&engine, VariantId::ordinary(invoke));
        for name in ["a", "b"] {
            assert!(selected.contains(&VariantId::ordinary(code(&engine, name))));
            assert!(variants(&engine, name).is_empty());
        }
        assert!(
            selected
                .iter()
                .all(|variant| matches!(variant, VariantId::Ordinary(_)))
        );
    }

    #[test]
    fn caller_shapes_specialize_an_already_polymorphic_callee() {
        for background in [false, true] {
            let mut engine = engine(background);
            engine.compile(
                "function readX(o){return o.x;} function useA(o){return o.y+readX(o);} function useB(o){return o.y+readX(o);} function change(o){o.extra=1;} var a={x:1,y:10}; var b={y:20,x:2};",
                "contextual-shapes.js",
            ).unwrap();
            finish_compilation(&engine);
            let a = engine.global().get_raw("a").unwrap();
            let b = engine.global().get_raw("b").unwrap();
            for _ in 0..jit::entries::RECOMPILE_WARMUP {
                for (object, expected) in [(a, 1.0), (b, 2.0)] {
                    assert_eq!(
                        engine.call("readX", &[object]).unwrap().value(),
                        Value::number(expected)
                    );
                }
            }
            finish_compilation(&engine);
            for _ in 0..3 {
                for _ in 0..jit::entries::RECOMPILE_WARMUP {
                    for (name, object, expected) in [("useA", a, 11.0), ("useB", b, 22.0)] {
                        assert_eq!(
                            engine.call(name, &[object]).unwrap().value(),
                            Value::number(expected)
                        );
                    }
                }
                finish_compilation(&engine);
            }
            let specialized = variants(&engine, "readX");
            assert_eq!(specialized.len(), 2);
            for variant in &specialized {
                assert!(matches!(
                    variant.requirements.as_slice(),
                    [(Input::Argument(0), Fact::Shape(_))]
                ));
                assert_ne!(variant.entry.load(Ordering::Acquire), 0);
            }
            for (name, object, expected) in [("useA", b, 22.0), ("useB", a, 11.0)] {
                assert_eq!(
                    engine.call(name, &[object]).unwrap().value(),
                    Value::number(expected)
                );
            }
            engine.call("change", &[a]).unwrap();
            for _ in 0..jit::entries::RECOMPILE_WARMUP {
                assert_eq!(
                    engine.call("useA", &[a]).unwrap().value(),
                    Value::number(11.0)
                );
            }
            finish_compilation(&engine);
        }
    }

    #[test]
    fn polymorphic_get_dispatch_preserves_misses_and_owned_results() {
        for background_compile in [false, true] {
            for inlining in [false, true] {
                let mut engine = Engine::with_options(EngineOptions {
                    background_compile,
                    inlining,
                    call_specialization: false,
                    ..Default::default()
                })
                .unwrap();
                engine.compile(
                    "function read(o){return o.x;} function invoke(o){return read(o);} function value(o){return o.value;} function change(o){o.x=second;o.extra=1;} function updatePrototype(){C.prototype.x=second;} function C(){} var first={value:11};var second={value:22};var a={x:first,y:0};var b={y:0,x:second};var dictionary={0:0,x:first};C.prototype.x=first;var inherited=new C();",
                    "polymorphic-get-dispatch.js",
                ).unwrap();
                finish_compilation(&engine);
                let a = engine.global().get_raw("a").unwrap();
                let b = engine.global().get_raw("b").unwrap();
                for name in ["read", "invoke"] {
                    for _ in 0..jit::entries::RECOMPILE_WARMUP {
                        for object in [a, b] {
                            engine.call(name, &[object]).unwrap();
                        }
                    }
                    finish_compilation(&engine);
                }
                let read = code(&engine, "read");
                assert!(
                    !engine
                        .tiering
                        .compiler
                        .borrow()
                        .specialization_plan(VariantId::ordinary(read))
                        .unwrap()
                        .shapes
                        .polymorphic_reads()
                        .is_empty()
                );
                let held = engine.call("invoke", &[a]).unwrap();
                engine.call("change", &[a]).unwrap();
                assert_eq!(
                    engine.call("value", &[held.value()]).unwrap().value(),
                    Value::number(11.0)
                );
                for (name, expected) in [
                    ("a", 22.0),
                    ("b", 22.0),
                    ("dictionary", 11.0),
                    ("inherited", 11.0),
                ] {
                    let object = engine.global().get_raw(name).unwrap();
                    let result = engine.call("invoke", &[object]).unwrap();
                    assert_eq!(
                        engine.call("value", &[result.value()]).unwrap().value(),
                        Value::number(expected)
                    );
                }
                engine.call("updatePrototype", &[]).unwrap();
                let object = engine.global().get_raw("inherited").unwrap();
                for _ in 0..jit::entries::RECOMPILE_WARMUP {
                    let result = engine.call("invoke", &[object]).unwrap();
                    assert_eq!(
                        engine.call("value", &[result.value()]).unwrap().value(),
                        Value::number(22.0)
                    );
                }
                finish_compilation(&engine);
            }
        }
    }

    #[test]
    fn megamorphic_gets_stop_speculating_on_cache_hits() {
        for background in [false, true] {
            let mut engine = engine(background);
            let mut source = String::from(
                "function value(){return this.value;} function invoke(o){return o.run();}",
            );
            for index in 0..6 {
                source.push_str(&format!(
                    "var o{index}={{run:value,value:{index},pad{index}:0}};"
                ));
            }
            engine.compile(&source, "megamorphic-read.js").unwrap();
            finish_compilation(&engine);
            for _ in 0..jit::entries::RECOMPILE_WARMUP {
                for index in 0..6 {
                    let object = engine.global().get_raw(&format!("o{index}")).unwrap();
                    assert_eq!(
                        engine.call("invoke", &[object]).unwrap().value(),
                        Value::number(index as f64)
                    );
                }
                finish_compilation(&engine);
            }
            let invoke = code(&engine, "invoke");
            engine.tiering.entry(invoke).unwrap();
            let compiler = engine.tiering.compiler.borrow();
            let plan = compiler
                .specialization_plan(VariantId::ordinary(invoke))
                .unwrap();
            assert!(!plan.shapes.megamorphic_sites().is_empty());
            drop(compiler);
            let settled = entry(&engine, invoke);
            for _ in 0..4 {
                for index in 0..6 {
                    let object = engine.global().get_raw(&format!("o{index}")).unwrap();
                    assert_eq!(
                        engine.call("invoke", &[object]).unwrap().value(),
                        Value::number(index as f64)
                    );
                }
            }
            finish_compilation(&engine);
            assert_eq!(entry(&engine, invoke), settled);
        }
    }

    #[test]
    fn polymorphic_calls_learn_targets_and_bound_dispatch() {
        for background in [false, true] {
            for reverse in [false, true] {
                let mut engine = engine(background);
                engine.compile(
                    "function a(x){return x+1;} function b(x){return x*2;} function c(x){return x-1;} function d(x){return x*3;} function e(x){return x+4;} function invoke(f,x){return f(x);}",
                    "polymorphic.js",
                ).unwrap();
                finish_compilation(&engine);
                let invoke = code(&engine, "invoke");
                let mut targets = vec![("a", 11.0), ("b", 20.0), ("c", 9.0), ("d", 30.0)];
                if reverse {
                    targets.reverse();
                }
                for &(name, expected) in &targets {
                    let callable = engine.global().get_raw(name).unwrap();
                    for _ in 0..jit::entries::RECOMPILE_WARMUP {
                        assert_eq!(
                            engine
                                .call("invoke", &[callable, Value::number(10.0)])
                                .unwrap()
                                .value(),
                            Value::number(expected)
                        );
                    }
                    finish_compilation(&engine);
                    engine.tiering.entry(invoke).unwrap();
                }
                let selected = selected(&engine, VariantId::ordinary(invoke));
                for &(name, _) in &targets {
                    assert!(
                        selected.contains(&VariantId::ordinary(code(&engine, name))),
                        "missing {name}"
                    );
                    assert!(
                        !variants(&engine, name).is_empty(),
                        "no variant for {name}, background={background}, reverse={reverse}"
                    );
                }
                let settled = entry(&engine, invoke);
                let fifth = engine.global().get_raw("e").unwrap();
                for _ in 0..jit::entries::RECOMPILE_WARMUP {
                    assert_eq!(
                        engine
                            .call("invoke", &[fifth, Value::number(10.0)])
                            .unwrap()
                            .value(),
                        Value::number(14.0)
                    );
                }
                finish_compilation(&engine);
                assert_eq!(entry(&engine, invoke), settled);
            }
        }
    }

    #[test]
    fn specialized_callers_relearn_targets_and_widen_numeric_feedback() {
        for background in [false, true] {
            let mut engine = engine(background);
            engine.compile(
                "function a(x){return x+1;} function b(x){return x*2;} var task={run:a}; function dispatch(x){return this.task.run(x);} var holder={task:task,run:dispatch}; function outer(x){return holder.run(x);} function replace(){task.run=b;}",
                "specialized-dispatch.js",
            ).unwrap();
            finish_compilation(&engine);
            for _ in 0..jit::entries::RECOMPILE_WARMUP {
                engine.call("outer", &[Value::number(10.0)]).unwrap();
            }
            finish_compilation(&engine);
            let dispatch = code(&engine, "dispatch");
            let variants = variants(&engine, "dispatch");
            assert!(!variants.is_empty());
            engine.call("replace", &[]).unwrap();
            for _ in 0..jit::entries::RECOMPILE_WARMUP {
                assert_eq!(
                    engine
                        .call("outer", &[Value::number(10.5)])
                        .unwrap()
                        .value(),
                    Value::number(21.0)
                );
            }
            finish_compilation(&engine);
            engine.tiering.entry(dispatch).unwrap();
            finish_compilation(&engine);
            let b = VariantId::ordinary(code(&engine, "b"));
            assert!(selected(&engine, VariantId::ordinary(dispatch)).contains(&b));
            for variant in variants {
                assert!(selected(&engine, variant.id()).contains(&b));
                assert_ne!(variant.entry.load(Ordering::Acquire), 0);
            }
        }
    }

    #[test]
    fn polymorphic_dispatch_evaluates_operands_once_and_preserves_captures() {
        let mut engine = engine(false);
        engine.compile(
            "var evaluations=0; function make(n){return function(x){return this.base+n+x;};} var first={base:10,run:make(1)}; var second={base:20,run:make(2)}; function third(x){return this.base-x;} var other={base:30,run:third}; function receiver(o){evaluations=evaluations+1;return o;} function argument(){evaluations=evaluations+1;return 3;} function invoke(o){return receiver(o).run(argument());} function count(){return evaluations;} function flip(o){o.run=third;return 3;} function mutate(o){return o.run(flip(o));}",
            "polymorphic-order.js",
        ).unwrap();
        for _ in 0..8 {
            for (name, expected) in [("first", 14.0), ("second", 25.0), ("other", 27.0)] {
                let object = engine.global().get_raw(name).unwrap();
                assert_eq!(
                    engine.call("invoke", &[object]).unwrap().value(),
                    Value::number(expected)
                );
            }
        }
        assert_eq!(
            engine.call("count", &[]).unwrap().value(),
            Value::number(48.0)
        );
        let first = engine.global().get_raw("first").unwrap();
        assert_eq!(
            engine.call("mutate", &[first]).unwrap().value(),
            Value::number(14.0)
        );
        assert_eq!(
            engine.call("mutate", &[first]).unwrap().value(),
            Value::number(7.0)
        );
    }

    #[test]
    fn feedback_arriving_during_compilation_is_replanned() {
        let mut engine = engine(true);
        engine.compile(
            "function a(x){return x+1;} function b(x){return x*2;} function c(x){return x-1;} function invoke(f,x){return f(x);}",
            "pending-polymorphism.js",
        ).unwrap();
        finish_compilation(&engine);
        let invoke = code(&engine, "invoke");
        let a = engine.global().get_raw("a").unwrap();
        for _ in 0..jit::entries::RECOMPILE_WARMUP {
            engine.call("invoke", &[a, Value::number(10.0)]).unwrap();
        }
        finish_compilation(&engine);
        let site = engine.tiering.profile.borrow().targets().next().unwrap().0;
        let mut cache = jit::profile::ObservationCache {
            complete: 0,
            last_callee: 0,
            failures: 0,
            owner: invoke,
            site,
        };
        engine.tiering.observe_call(&mut cache, code(&engine, "b"));
        for _ in 0..jit::entries::RECOMPILE_WARMUP {
            engine.call("invoke", &[a, Value::number(10.0)]).unwrap();
        }
        engine.tiering.submit_compilation(invoke);
        assert!(
            engine
                .tiering
                .worker
                .borrow()
                .as_ref()
                .unwrap()
                .in_flight
                .contains_key(&jit::recompile::Key::ordinary(invoke))
        );
        engine.tiering.observe_call(&mut cache, code(&engine, "c"));
        for _ in 0..jit::entries::RECOMPILE_WARMUP {
            engine.call("invoke", &[a, Value::number(10.0)]).unwrap();
        }
        finish_compilation(&engine);
        assert!(
            selected(&engine, VariantId::ordinary(invoke))
                .contains(&VariantId::ordinary(code(&engine, "c")))
        );
    }

    #[test]
    fn integer_failures_replan_specialized_bodies() {
        for background in [false, true] {
            let mut engine = engine(background);
            engine.compile(
                "function read(x){return this.value+x;} var holder={value:10,run:read}; function outer(x){return holder.run(x);} function fraction(){holder.value=0.5;}",
                "variant-widening.js",
            ).unwrap();
            finish_compilation(&engine);
            for _ in 0..jit::entries::RECOMPILE_WARMUP {
                engine.call("outer", &[Value::number(2.0)]).unwrap();
            }
            finish_compilation(&engine);
            let variants = variants(&engine, "read");
            assert!(!variants.is_empty());
            engine.call("fraction", &[]).unwrap();
            for _ in 0..jit::entries::RECOMPILE_WARMUP {
                assert_eq!(
                    engine.call("outer", &[Value::number(2.0)]).unwrap().value(),
                    Value::number(2.5)
                );
            }
            finish_compilation(&engine);
            engine.tiering.entry(code(&engine, "read")).unwrap();
            finish_compilation(&engine);
            for variant in variants {
                let compiler = engine.tiering.compiler.borrow();
                let plan = compiler.specialization_plan(variant.id()).unwrap();
                assert!(
                    plan.numbers
                        .iter()
                        .any(|(_, kind)| *kind == jit::profile::ReadType::Number)
                );
            }
        }
    }

    #[test]
    fn numeric_call_variants_preserve_guard_misses_and_callee_identity() {
        let mut engine = engine(false);
        engine
            .compile(
                r#"
            function twice(x) { return x + x; }
            function call(x) { return twice(x); }
            function numeric() { return call(21); }
            function text() { return call("a") === "aa"; }
            function replaced() { twice = function(x) { return x + 1; }; return call(21); }
        "#,
                "numeric-variants.js",
            )
            .unwrap();
        assert_eq!(
            engine.call("numeric", &[]).unwrap().value(),
            Value::number(42.0)
        );
        assert_eq!(
            engine.call("call", &[Value::number(2.5)]).unwrap().value(),
            Value::number(5.0)
        );
        let variants = variants(&engine, "twice");
        assert_eq!(variants.len(), 1);
        assert_eq!(
            variants[0].requirements,
            vec![(Input::Argument(0), Fact::Number)]
        );
        let specialized = variants[0].entry.load(Ordering::Acquire);
        assert_ne!(specialized, 0);
        assert_ne!(
            specialized,
            engine
                .tiering
                .entry(variants[0].function)
                .unwrap()
                .unwrap()
                .code
        );
        engine.tiering.request_compilation(variants[0].function);
        engine.tiering.entry(variants[0].function).unwrap();
        assert_eq!(variants[0].entry.load(Ordering::Acquire), specialized);
        assert_eq!(
            engine.call("text", &[]).unwrap().value(),
            Value::boolean(true)
        );
        assert_eq!(
            engine.call("replaced", &[]).unwrap().value(),
            Value::number(22.0)
        );
    }

    #[test]
    fn recursive_variants_reuse_their_entry_and_preserve_closure_captures() {
        for background in [false, true] {
            let mut engine = engine(background);
            engine
                .compile(
                    r#"
                function fib(n) { return n < 2 ? n : fib(n-1) + fib(n-2); }
                function run() { return fib(12); }
                function make(offset) { return function(n) { return n + offset; }; }
                function call(f, n) { return f(n); }
                function captures() { return call(make(10), 2) + call(make(20), 3); }
            "#,
                    "recursive-variants.js",
                )
                .unwrap();
            assert_eq!(
                engine.call("run", &[]).unwrap().value(),
                Value::number(144.0)
            );
            assert_eq!(variants(&engine, "fib").len(), 1);
            assert_eq!(
                engine.call("captures", &[]).unwrap().value(),
                Value::number(35.0)
            );
        }
    }

    #[test]
    fn variants_preserve_owned_results_and_thrown_values() {
        for background in [false, true] {
            let mut engine = engine(background);
            engine.compile(r#"
                function pass(n, object) { if (n < 0) throw object; return object; }
                function call(n, object) { return pass(n, object); }
                function positive() { return call(1, {value:7}); }
                function negative() { const payload = {value:9}; try { call(-1, payload); } catch (error) { return error === payload; } return false; }
            "#, "variant-ownership.js").unwrap();
            let result = engine.call("positive", &[]).unwrap();
            finish_compilation(&engine);
            assert_eq!(
                engine.call("negative", &[]).unwrap().value(),
                Value::boolean(true)
            );
            engine.drain_disposals(1000);
            assert_eq!(
                engine.global().heap().property(result.value(), "value"),
                Some(Value::number(7.0))
            );
        }
    }

    #[test]
    fn background_variants_publish_separately_from_ordinary_entries() {
        let mut engine = engine(true);
        engine
            .compile(
                "function twice(x){return x+x;} function run(){return twice(21);}",
                "background-variant.js",
            )
            .unwrap();
        assert_eq!(
            engine.call("run", &[]).unwrap().value(),
            Value::number(42.0)
        );
        let variants = variants(&engine, "twice");
        assert_eq!(variants.len(), 1);
        let variant = &variants[0];
        assert_ne!(variant.entry.load(Ordering::Acquire), 0);
        finish_compilation(&engine);
        let specialized = engine.tiering.variants.borrow().get(variant.id()).unwrap();
        let ordinary = engine.tiering.entry(variant.function).unwrap().unwrap();
        assert_eq!(variant.entry.load(Ordering::Acquire), specialized.code);
        assert_ne!(specialized.code, ordinary.code);
        assert_eq!(
            engine.call("run", &[]).unwrap().value(),
            Value::number(42.0)
        );
    }

    #[test]
    fn receiver_shape_variants_keep_generic_calls_and_mutations_correct() {
        let mut engine = engine(false);
        engine
            .compile(
                r#"
            function read() { return this.x; }
            var same = {x:3, read:read};
            var other = {pad:0, x:7, read:read};
            function warmRead() { for (let i=0; i<32; ++i) same.read(); return same.read(); }
        "#,
                "shape-variants.js",
            )
            .unwrap();
        engine.call("warmRead", &[]).unwrap();
        engine
            .compile(
                r#"
            function call(object) { return object.read(); }
            function warm() { for (let i=0; i<32; ++i) call(same); return call(same); }
            function changed() { same.extra=9; same.x=11; return call(same)+call(other); }
        "#,
                "shape-caller.js",
            )
            .unwrap();
        assert_eq!(
            engine.call("warm", &[]).unwrap().value(),
            Value::number(3.0)
        );
        let variants = variants(&engine, "read");
        assert!(variants.iter().any(|variant| matches!(
            variant.requirements.as_slice(),
            [(Input::This, Fact::Shape(_))]
        ) && variant.entry.load(Ordering::Acquire) != 0));
        assert_eq!(
            engine.call("changed", &[]).unwrap().value(),
            Value::number(18.0)
        );
    }
}
