use js_engine::{Engine, Error, Value};

fn eval(source: &str, args: &[Value]) -> Value {
    let mut engine = Engine::new().unwrap();
    engine.compile(source, "semantics.js").unwrap();
    engine.call("main", args).unwrap().value()
}

fn check(source: &str, expected: Value) {
    let actual = eval(source, &[]);
    assert_eq!(actual, expected, "{source}");
}

#[test]
fn numeric_fast_path_checks_tags() {
    check("function main(){ return null + 1; }", Value::number(1.0));
    check("function main(){ return true + 1; }", Value::number(2.0));
    check(
        "function main(){ return undefined * 2; }",
        Value::number(f64::NAN),
    );
}

#[test]
fn hoisted_vars_start_undefined() {
    check("function main(){ return x; var x = 1; }", Value::UNDEFINED);
}

#[test]
fn var_without_initializer_does_not_reset() {
    let actual = eval(
        "function main(x){ var x; return x; }",
        &[Value::number(7.0)],
    );
    assert_eq!(actual, Value::number(7.0));
    check("function main(){ var x; return x; }", Value::UNDEFINED);
}

#[test]
fn let_and_const_are_block_scoped() {
    check(
        "function main(){ let x = 1; { let x = 2; } return x; }",
        Value::number(1.0),
    );
    check(
        "function main(){ const x = 3; { const x = 4; } return x; }",
        Value::number(3.0),
    );
    check(
        "function main(){ let x = 1; { let y = 2; x = x + y; } return x; }",
        Value::number(3.0),
    );
}

#[test]
fn compound_assignment_reads_before_rhs() {
    check(
        "function main(){ var x = 1; x += (x = 5); return x; }",
        Value::number(6.0),
    );
}

#[test]
fn update_expressions_convert_to_number() {
    check(
        "function main(){ var s = null; return s++; }",
        Value::number(0.0),
    );
    check(
        "function main(){ var s = null; s++; return s; }",
        Value::number(1.0),
    );
    check(
        "function main(){ var s = true; return --s; }",
        Value::number(0.0),
    );
}

#[test]
fn locals_shadow_undefined() {
    check(
        "function main(){ var undefined = 3; return undefined; }",
        Value::number(3.0),
    );
    check("function main(){ return undefined; }", Value::UNDEFINED);
}

#[test]
fn arithmetic_and_comparisons() {
    check(
        "function main(){ return 1 + 2 * 3 - 4 / 2; }",
        Value::number(5.0),
    );
    check("function main(){ return 7 % 4; }", Value::number(3.0));
    check("function main(){ return 2 ** 5; }", Value::number(32.0));
    check("function main(){ return 1 < 2; }", Value::TRUE);
    check("function main(){ return 2 <= 2; }", Value::TRUE);
    check("function main(){ return 3 > 4; }", Value::FALSE);
    check("function main(){ return 3 >= 3; }", Value::TRUE);
    check("function main(){ return null == undefined; }", Value::TRUE);
    check("function main(){ return 1 === 1; }", Value::TRUE);
    check("function main(){ return 1 !== 2; }", Value::TRUE);
}

#[test]
fn loops() {
    check(
        "function main(){ var sum = 0; for (let i = 0; i < 5; i = i + 1) { sum += i; } return sum; }",
        Value::number(10.0),
    );
    check(
        "function main(){ var i = 0; var sum = 0; while (i < 4) { sum += i; i++; } return sum; }",
        Value::number(6.0),
    );
    check(
        "function main(){ var i = 0; do { i++; } while (i < 3); return i; }",
        Value::number(3.0),
    );
    check(
        "function main(){ var sum = 0; for (var i = 0; i < 3; i++) { sum += i; } return sum; }",
        Value::number(3.0),
    );
}

#[test]
fn logical_and_conditional_expressions() {
    check("function main(){ return 0 || 5; }", Value::number(5.0));
    check("function main(){ return 1 && 7; }", Value::number(7.0));
    check("function main(){ return null ?? 3; }", Value::number(3.0));
    check("function main(){ return 0 ?? 3; }", Value::number(0.0));
    check(
        "function main(){ return true ? 11 : 22; }",
        Value::number(11.0),
    );
    check(
        "function main(){ return false ? 11 : 22; }",
        Value::number(22.0),
    );
}

#[test]
fn calls_handle_missing_arguments() {
    check(
        "function first(a, b){ return a; } function main(){ return first(9); }",
        Value::number(9.0),
    );
    check(
        "function add(a, b){ return a + b; } function main(){ return add(1); }",
        Value::number(f64::NAN),
    );
}

#[test]
fn recursion() {
    check(
        "function fib(n){ if (n < 2) { return n; } return fib(n - 1) + fib(n - 2); } function main(){ return fib(10); }",
        Value::number(55.0),
    );
}

#[test]
fn functions_can_call_across_compile_calls() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile("function base(x){ return x + 1; }", "base.js")
        .unwrap();
    engine
        .compile("function main(){ return base(4); }", "main.js")
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(5.0)
    );
}

#[test]
fn optimization_level_is_configurable() {
    let mut engine = Engine::with_options(js_engine::EngineOptions {
        opt_level: js_engine::OptLevel::None,
    })
    .unwrap();
    engine
        .compile("function main(){ return 2 + 3; }", "none.js")
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(5.0)
    );
}

#[test]
fn unsupported_errors_include_span_and_node_kind() {
    let mut engine = Engine::new().unwrap();
    let error = engine
        .compile("function main(){ return {}; }", "error.js")
        .unwrap_err()
        .to_string();
    assert!(error.contains("object expression"), "{error}");
    assert!(error.contains(" at "), "{error}");
}

#[test]
fn truthiness_of_edge_case_numbers_and_nullish_values() {
    check(
        "function main(){ if (0 / 0) { return 1; } return 2; }",
        Value::number(2.0),
    );
    check(
        "function main(){ if (-0) { return 1; } return 2; }",
        Value::number(2.0),
    );
    check(
        "function main(){ if (null) { return 1; } return 2; }",
        Value::number(2.0),
    );
    check(
        "function main(){ if (undefined) { return 1; } return 2; }",
        Value::number(2.0),
    );
    check(
        "function main(){ if (0.5) { return 1; } return 2; }",
        Value::number(1.0),
    );
}

#[test]
fn update_expressions_on_null_and_true() {
    check(
        "function main(){ var i = null; i++; return i; }",
        Value::number(1.0),
    );
    check(
        "function main(){ var i = true; i++; return i; }",
        Value::number(2.0),
    );
    check(
        "function main(){ var i = null; return i++; }",
        Value::number(0.0),
    );
    check(
        "function main(){ var i = true; return i++; }",
        Value::number(1.0),
    );
}

#[test]
fn environment_lookup_and_assignment_follow_parent_chain() {
    let engine = Engine::new().unwrap();
    let global = engine.global();
    assert!(global.define("x", Value::number(1.0)));
    assert!(!global.define("x", Value::number(9.0)));
    let child = global.child();
    let nested = child.child();
    assert_eq!(
        nested.get("x").map(|value| value.value()),
        Some(Value::number(1.0))
    );
    assert!(nested.set("x", Value::number(2.0)));
    assert_eq!(
        global.get("x").map(|value| value.value()),
        Some(Value::number(2.0))
    );
    assert!(child.define("x", Value::number(3.0)));
    assert!(nested.set("x", Value::number(4.0)));
    assert_eq!(
        child.get("x").map(|value| value.value()),
        Some(Value::number(4.0))
    );
    assert_eq!(
        global.get("x").map(|value| value.value()),
        Some(Value::number(2.0))
    );
    assert_eq!(nested.get("missing").map(|value| value.value()), None);
    assert!(!nested.set("missing", Value::UNDEFINED));
}

#[test]
fn compiled_functions_access_shared_globals() {
    let mut engine = Engine::new().unwrap();
    engine.global().define("x", Value::number(1.0));
    engine
        .compile(
            "function inc(){ return ++x; } function main(){ x += inc(); return x; }",
            "test.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(3.0)
    );
    assert_eq!(
        engine.global().get("x").map(|value| value.value()),
        Some(Value::number(3.0))
    );
    assert_eq!(engine.call("inc", &[]).unwrap().value(), Value::number(4.0));
}

#[test]
fn missing_global_stops_nested_execution() {
    let mut engine = Engine::new().unwrap();
    engine.global().define("x", Value::number(0.0));
    engine
        .compile(
            "function bad(){ return missing; } function main(){ bad(); x = 1; }",
            "test.js",
        )
        .unwrap();
    assert!(engine.call("main", &[]).is_err());
    assert_eq!(
        engine.global().get("x").map(|value| value.value()),
        Some(Value::number(0.0))
    );
}

#[test]
fn top_level_vars_are_hoisted_initialized_and_shared() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "var first = later, later = 2, result = get(); function get(){ return later; }",
            "first.js",
        )
        .unwrap();
    assert_eq!(
        engine.global().get("first").map(|value| value.value()),
        Some(Value::UNDEFINED)
    );
    assert_eq!(
        engine.global().get("result").map(|value| value.value()),
        Some(Value::number(2.0))
    );
    engine
        .compile(
            "var later; var next = later + 1; function main(){ return next; }",
            "second.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(3.0)
    );
}

#[test]
fn top_level_initializer_error_stops_execution() {
    let mut engine = Engine::new().unwrap();
    assert!(
        engine
            .compile("var first = missing, later = 2;", "test.js")
            .is_err()
    );
    assert_eq!(
        engine.global().get("later").map(|value| value.value()),
        Some(Value::UNDEFINED)
    );
}

#[test]
fn global_object_properties_share_global_bindings() {
    let mut engine = Engine::new().unwrap();
    engine.compile("var x = 2; globalThis.x += 3; globalThis.y = x + 1; function main(){ return globalThis.y; }", "test.js").unwrap();
    assert_eq!(
        engine.global().get("x").map(|value| value.value()),
        Some(Value::number(5.0))
    );
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(6.0)
    );
    engine.compile("globalThis.y = 9;", "next.js").unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(9.0)
    );
}

#[test]
fn property_access_on_non_object_reports_error() {
    let mut engine = Engine::new().unwrap();
    assert!(
        engine
            .compile("var x = null; x.foo = 2;", "test.js")
            .is_err()
    );
}

#[test]
fn semantic_binding_resolution() {
    check(
        "function main(){ var x = 1; { let x = 2; x++; } return x; }",
        Value::number(1.0),
    );
    check(
        "var undefined = 4; function main(){ return undefined; }",
        Value::number(4.0),
    );
    check(
        "function f(){ return 3; } function main(){ { let f = 2; } return f(); }",
        Value::number(3.0),
    );
    for source in [
        "function f(){ return 1; } function main(f){ return f(); }",
        "function f(){ return 1; } function main(){ let f = 2; return f(); }",
        "var f = 2; function main(){ return f(); }",
    ] {
        let error = Engine::new()
            .unwrap()
            .compile(source, "semantic.js")
            .unwrap_err();
        assert!(
            matches!(error, Error::Compile(ref message) if message.contains("non-callable function `f`")),
            "{source}: {error}"
        );
    }
    let error = Engine::new()
        .unwrap()
        .compile("function main(){ let x; let x; }", "semantic.js")
        .unwrap_err();
    assert!(matches!(error, Error::Parse(_)), "{error}");
}

#[test]
fn prior_function_cannot_override_current_binding() {
    for source in [
        "var f = 2; function main(){ return f(); }",
        "function main(f){ return f(); }",
    ] {
        let mut engine = Engine::new().unwrap();
        engine
            .compile("function f(){ return 1; }", "first.js")
            .unwrap();
        let error = engine.compile(source, "second.js").unwrap_err();
        assert!(
            matches!(error, Error::Compile(ref message) if message.contains("non-callable function `f`")),
            "{source}: {error}"
        );
    }
}

#[test]
fn dense_arrays_retain_aliases_and_rest_arguments() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            r#"
        function rest(first, ...tail) { return tail; }
        function main() {
            let inner = [7];
            let outer = [inner];
            let args = rest(0, inner, 9);
            inner = [2];
            outer[0] = outer[0];
            return args[0][0] + args[1] + outer.length;
        }
    "#,
            "arrays.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(17.0)
    );
}

#[test]
fn host_roots_and_globals_keep_arrays_alive() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            r#"
        var saved;
        function make() { return [13]; }
        function store(value) { saved = value; }
        function read() { return saved[0]; }
    "#,
            "roots.js",
        )
        .unwrap();
    let value = engine.call("make", &[]).unwrap();
    engine.call("store", &[value.value()]).unwrap();
    drop(value);
    engine.drain_disposals(100);
    assert_eq!(
        engine.call("read", &[]).unwrap().value(),
        Value::number(13.0)
    );
}

#[test]
fn arrays_reject_foreign_and_stale_host_values() {
    let mut first = Engine::new().unwrap();
    let mut second = Engine::new().unwrap();
    first
        .compile("function make() { return [1]; }", "first.js")
        .unwrap();
    second
        .compile("function id(x) { return x; }", "second.js")
        .unwrap();
    let root = first.call("make", &[]).unwrap();
    let raw = root.value();
    assert!(second.call("id", &[raw]).is_err());
    drop(root);
    assert!(first.call("make", &[raw]).is_err());
}

#[test]
fn dense_arrays_reject_holes_and_allow_length_shrink() {
    let mut engine = Engine::new().unwrap();
    assert!(
        engine
            .compile("function hole() { return [1,,3]; }", "hole.js")
            .is_err()
    );
    engine
        .compile(
            r#"
        function truncate() { let a = [1, 2, 3]; a.length = 1; return a.length + a[1]; }
        function grow() { let a = [1]; a.length = 3; }
        function sparse() { let a = [1]; a[3] = 4; }
    "#,
            "dense.js",
        )
        .unwrap();
    assert!(
        engine
            .call("truncate", &[])
            .unwrap()
            .value()
            .as_number()
            .unwrap()
            .is_nan()
    );
    assert!(engine.call("grow", &[]).is_err());
    assert!(engine.call("sparse", &[]).is_err());
}

#[test]
fn rest_copies_trailing_arguments_and_handles_missing_arguments() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            r#"
        function collect(first, ...rest) { return rest; }
        function empty() { return collect().length; }
        function copy() { let a = [8]; let r = collect(0, a, 2); a = [0]; return r[0][0] + r[1]; }
    "#,
            "rest.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("empty", &[]).unwrap().value(),
        Value::number(0.0)
    );
    assert_eq!(
        engine.call("copy", &[]).unwrap().value(),
        Value::number(10.0)
    );
}

#[test]
fn unsupported_object_coercion_stops_execution() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            r#"
        var side = 0;
        function arithmetic() { let x = [] + 1; side = 1; return x; }
        function equality() { let x = [] == true; side = 2; return x; }
        function unary() { return +[]; }
        function bitwise() { return ~[]; }
        function relational() { return [] < 1; }
        function update() { let x = []; return x++; }
        function identity() { let x = []; return x == x && x != null; }
        function read() { return side; }
    "#,
            "coercion.js",
        )
        .unwrap();
    assert!(engine.call("arithmetic", &[]).is_err());
    assert!(engine.call("equality", &[]).is_err());
    for name in ["unary", "bitwise", "relational", "update"] {
        assert!(engine.call(name, &[]).is_err(), "{name}");
    }
    assert_eq!(
        engine.call("read", &[]).unwrap().value(),
        Value::number(0.0)
    );
    assert_eq!(engine.call("identity", &[]).unwrap().value(), Value::TRUE);
}
