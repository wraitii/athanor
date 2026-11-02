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
        .compile("function main(){ return `text`; }", "error.js")
        .unwrap_err()
        .to_string();
    assert!(error.contains("template literal expression"), "{error}");
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
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "semantic.js").unwrap();
        let error = engine.call("main", &[Value::number(2.0)]).err().unwrap();
        assert!(matches!(error, Error::Runtime(_)), "{source}: {error}");
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
        engine.compile(source, "second.js").unwrap();
        let error = engine.call("main", &[Value::number(2.0)]).err().unwrap();
        assert!(matches!(error, Error::Runtime(_)), "{source}: {error}");
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

#[test]
fn member_calls_resolve_callable_properties() {
    check(
        "globalThis.f = function(x) { return x + 1; }; function main() { return globalThis.f(41); }",
        Value::number(42.0),
    );
    check(
        "function main() { return [function(x) { return x + 1; }][0](41); }",
        Value::number(42.0),
    );
}

#[test]
fn utf8_strings_use_scalar_indices_and_content_equality() {
    check(
        "function main() { return 'a😀é'.length; }",
        Value::number(3.0),
    );
    check(
        "function main() { return 'a😀é'[1] === '😀'; }",
        Value::TRUE,
    );
    check(
        "function main() { return ('é' + '😀') === 'é😀'; }",
        Value::TRUE,
    );
    check("function main() { return '😀'[1]; }", Value::UNDEFINED);
    check("function main() { return !'' && !!'😀'; }", Value::TRUE);
    check(
        "function main() { if ('') return false; return true; }",
        Value::TRUE,
    );
    check(
        "function main() { return '\\uD83D\\uDE00' === '😀'; }",
        Value::TRUE,
    );
    check(
        "function main() { return 'e\\u0301'.length; }",
        Value::number(2.0),
    );
    check(
        "function main() { let a = ['😀']; let f = () => a[0]; a = ['é']; return f() === 'é'; }",
        Value::TRUE,
    );
}

#[test]
fn strings_reject_lone_surrogates() {
    let mut engine = Engine::new().unwrap();
    assert!(
        engine
            .compile("function main() { return '\\uD800'; }", "surrogate.js")
            .is_err()
    );
}

#[test]
fn exceptions_propagate_and_catch_values() {
    check(
        "function main() { try { throw undefined; } catch (e) { return e; } }",
        Value::UNDEFINED,
    );
    check(
        "function main() { try { throw [42]; } catch (e) { return e[0]; } }",
        Value::number(42.0),
    );
    check(
        "function fail() { throw 'é😀'; } function main() { try { fail(); return false; } catch (e) { return e === 'é😀'; } }",
        Value::TRUE,
    );
    check(
        "function main() { try { try { throw 1; } catch (e) { throw e + 1; } } catch (e) { return e; } }",
        Value::number(2.0),
    );
    check(
        "function main() { let f; try { throw [42]; } catch (e) { f = () => e[0]; } return f(); }",
        Value::number(42.0),
    );
    check(
        "function main() { try { throw 1; } catch { } return 42; }",
        Value::number(42.0),
    );
    check(
        "function main() { try { return 42; } catch (e) { return 0; } }",
        Value::number(42.0),
    );
    check(
        "function main() { try { var x = 42; } catch { var y = 1; } return x; }",
        Value::number(42.0),
    );
}

#[test]
fn engine_errors_bypass_catch_and_uncaught_exceptions_do_not_persist() {
    let mut engine = Engine::new().unwrap();
    engine.compile("function bad() { try { undefined(); } catch { return 42; } } function fail() { throw 'oops'; } function main() { return 42; }", "exceptions.js").unwrap();
    assert!(matches!(engine.call("bad", &[]), Err(Error::Runtime(_))));
    assert!(
        engine
            .call("fail", &[])
            .err()
            .unwrap()
            .to_string()
            .contains("uncaught exception: oops")
    );
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(42.0)
    );
}

#[test]
fn exceptions_keep_thrown_closures_alive() {
    check(
        "function fail() { let a = [42]; throw () => a[0]; } function main() { let f; try { fail(); } catch (e) { f = e; } return f(); }",
        Value::number(42.0),
    );
    check(
        "function main() { let n = 0; for (let i = 0; i < 10; ++i) { try { throw [i]; } catch (e) { n += e[0]; } } return n; }",
        Value::number(45.0),
    );
}

#[test]
fn finally_is_explicitly_unsupported() {
    let mut engine = Engine::new().unwrap();
    assert!(matches!(
        engine.compile(
            "function main() { try { return 1; } finally { return 2; } }",
            "finally.js"
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn basic_error_construction_and_propagation() {
    check(
        "function main() { return new Error('é😀').message === 'é😀'; }",
        Value::TRUE,
    );
    check(
        "function main() { return new Error().message === ''; }",
        Value::TRUE,
    );
    check(
        "function main() { return new Error(undefined).message === ''; }",
        Value::TRUE,
    );
    check(
        "function main() { return new Error(42).message === '42'; }",
        Value::TRUE,
    );
    check(
        "function main() { return Error('oops').name === 'Error'; }",
        Value::TRUE,
    );
    check(
        "function main() { let C = Error; return new C('oops').message === 'oops'; }",
        Value::TRUE,
    );
    check(
        "function main() { return ('prefix: ' + new Error('oops')) === 'prefix: Error: oops'; }",
        Value::TRUE,
    );
    check(
        "function main() { return ('' + new Error()) === 'Error'; }",
        Value::TRUE,
    );
    check(
        "function main() { let e = new Error('oops'); e.name = 'Custom'; e.message = 'changed'; return ('' + e) === 'Custom: changed'; }",
        Value::TRUE,
    );
    check(
        "function fail() { throw new Error('oops'); } function main() { try { fail(); } catch (e) { return e.message === 'oops'; } }",
        Value::TRUE,
    );
    check(
        "function main() { let e = new Error(); return e === e && e !== new Error(); }",
        Value::TRUE,
    );
}

#[test]
fn uncaught_error_reports_its_message() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile("function main() { throw new Error('oops'); }", "error.js")
        .unwrap();
    assert!(
        engine
            .call("main", &[])
            .err()
            .unwrap()
            .to_string()
            .contains("Error: oops")
    );
}

#[test]
fn typeof_reports_supported_types() {
    for (expression, expected) in [
        ("undefined", "undefined"),
        ("neverDeclared", "undefined"),
        ("null", "object"),
        ("true", "boolean"),
        ("42", "number"),
        ("0 / 0", "number"),
        ("'é😀'", "string"),
        ("[]", "object"),
        ("globalThis", "object"),
        ("new Error('oops')", "object"),
        ("Error", "function"),
        ("main", "function"),
        ("() => 1", "function"),
        ("function() {}", "function"),
    ] {
        check(
            &format!("function main() {{ return typeof ({expression}) === '{expected}'; }}"),
            Value::TRUE,
        );
    }
    check(
        "function main() { let x = 0; typeof (x = x + 1); return x; }",
        Value::number(1.0),
    );
    check(
        "function main() { let f = () => { throw 42; }; try { return typeof f(); } catch (e) { return e; } }",
        Value::number(42.0),
    );
}

#[test]
fn typeof_rejects_uninitialized_lexical_bindings() {
    for source in [
        "function main() { return typeof x; let x = 1; }",
        "function main() { let f = () => typeof x; return f(); let x = 1; }",
    ] {
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "typeof.js").unwrap();
        assert!(matches!(engine.call("main", &[]), Err(Error::Runtime(_))));
    }
}

#[test]
fn for_of_iterates_arrays_and_utf8_strings() {
    check(
        "function main() { let sum = 0; for (const x of [1, 2, 3]) sum += x; return sum; }",
        Value::number(6.0),
    );
    check(
        "function main() { let text = ''; for (const x of 'a😀e\\u0301') text += x; return text === 'a😀e\\u0301'; }",
        Value::TRUE,
    );
    check(
        "function main() { let n = 0; for (const x of '😀é') ++n; return n; }",
        Value::number(2.0),
    );
    check(
        "function main() { let n = 0; for (var x of []) ++n; return n; }",
        Value::number(0.0),
    );
    check(
        "function main() { for (var x of [1, 2]) {} return x; }",
        Value::number(2.0),
    );
    check(
        "function main() { let a = [1, 2]; let n = 0; for (const x of a) { n += x; if (x === 1) a[a.length] = 3; } return n; }",
        Value::number(6.0),
    );
    check(
        "function main() { let a = [1, 2, 3]; let n = 0; for (const x of a) { n += x; a.length = 0; } return n; }",
        Value::number(1.0),
    );
    check(
        "function main() { let a = [1, 2]; let n = 0; for (const x of a) { a = []; n += x; } return n; }",
        Value::number(3.0),
    );
    check(
        "function main() { let n = 0; let f = () => { ++n; return [1, 2]; }; for (const x of f()) {} return n; }",
        Value::number(1.0),
    );
}

#[test]
fn for_of_captures_fresh_bindings() {
    check(
        "function main() { let fs = []; for (const x of [1, 2, 3]) fs[fs.length] = () => x; return fs[0]() * 100 + fs[1]() * 10 + fs[2](); }",
        Value::number(123.0),
    );
    check(
        "function main() { let fs = []; for (let x of [1, 2]) { fs[fs.length] = () => x; ++x; } return fs[0]() * 10 + fs[1](); }",
        Value::number(23.0),
    );
    check(
        "function main() { let fs = []; for (var x of [1, 2]) fs[fs.length] = () => x; return fs[0]() * 10 + fs[1](); }",
        Value::number(22.0),
    );
    check(
        "function main() { let fs = []; for (const x of [1, 2]) { fs[fs.length] = () => x; continue; } return fs[0]() * 10 + fs[1](); }",
        Value::number(12.0),
    );
}

#[test]
fn loop_break_and_continue_use_innermost_targets() {
    check(
        "function main() { let n = 0; for (const x of [1, 2, 3, 4]) { if (x === 2) continue; if (x === 4) break; n += x; } return n; }",
        Value::number(4.0),
    );
    check(
        "function main() { let n = 0; for (const x of [1, 2]) { for (const y of [3, 4]) { ++n; break; } } return n; }",
        Value::number(2.0),
    );
    check(
        "function main() { let n = 0; for (let i = 0; i < 5; ++i) { if (i === 1) continue; if (i === 4) break; n += i; } return n; }",
        Value::number(5.0),
    );
    check(
        "function main() { let i = 0; while (i < 5) { ++i; if (i === 1) continue; break; } return i; }",
        Value::number(2.0),
    );
    check(
        "function main() { let i = 0; do { ++i; if (i < 3) continue; break; } while (i < 5); return i; }",
        Value::number(3.0),
    );
    check(
        "function main() { try { for (const x of [[42]]) throw x; } catch (e) { return e[0]; } }",
        Value::number(42.0),
    );
}

#[test]
fn for_of_rejects_unsupported_iterables_and_binding_forms() {
    for expression in ["42", "null", "new Error()"] {
        let mut engine = Engine::new().unwrap();
        engine
            .compile(
                &format!("function main() {{ for (const x of {expression}) {{}} }}"),
                "iteration.js",
            )
            .unwrap();
        assert!(matches!(engine.call("main", &[]), Err(Error::Runtime(_))));
    }
    for source in [
        "function main() { for (const [x] of [[1]]) {} }",
        "function main() { let x; for (x of [1]) {} }",
        "function main() { outer: while (true) { break outer; } }",
    ] {
        let mut engine = Engine::new().unwrap();
        assert!(matches!(
            engine.compile(source, "iteration.js"),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn bitwise_operators_use_32_bit_values_and_mask_shift_counts() {
    for (expression, expected) in [
        ("0xf0 & 0x3c", 48.0),
        ("0xf0 | 0x3c", 252.0),
        ("0xf0 ^ 0x3c", 204.0),
        ("1 << 31", -2147483648.0),
        ("-8 >> 2", -2.0),
        ("-1 >>> 0", 4294967295.0),
        ("-1 >>> 1", 2147483647.0),
        ("1 << 32", 1.0),
        ("1 << -1", -2147483648.0),
        ("-8 >> 33", -4.0),
        ("4294967297 | 0", 1.0),
        ("-4294967297 | 0", -1.0),
        ("3.9 & 1", 1.0),
        ("(0 / 0) ^ 1", 1.0),
        ("(1 / 0) | 1", 1.0),
        ("true & 3", 1.0),
        ("null | 2", 2.0),
    ] {
        check(
            &format!("function main() {{ return {expression}; }}"),
            Value::number(expected),
        );
    }
}

#[test]
fn bitwise_compound_assignments_preserve_evaluation_order() {
    for (operator, expected) in [
        ("&=", 2.0),
        ("|=", 7.0),
        ("^=", 5.0),
        ("<<=", 48.0),
        (">>=", 0.0),
        (">>>=", 0.0),
    ] {
        check(
            &format!("function main() {{ let x = 6; return x {operator} 3; }}"),
            Value::number(expected),
        );
    }
    check(
        "function main() { let x = 6; x ^= (x = 3); return x; }",
        Value::number(5.0),
    );
    check(
        "function main() { let a = [6]; let n = 0; let get = () => { ++n; return a; }; let index = () => { ++n; return 0; }; get()[index()] ^= (a[0] = 3); return a[0] * 10 + n; }",
        Value::number(52.0),
    );
    check(
        "function main() { let e = new Error(); e.x = 6; e.x ^= (e.x = 3); return e.x; }",
        Value::number(5.0),
    );
    check(
        "globalThis.bits = 6; function main() { bits ^= (bits = 3); return bits; }",
        Value::number(5.0),
    );
    check(
        "function main() { let x = 6; let f = () => x ^= 3; f(); return x; }",
        Value::number(5.0),
    );
    check(
        "function main() { let x = 6; let fail = () => { throw 42; }; try { x ^= fail(); } catch {} return x; }",
        Value::number(6.0),
    );
}

#[test]
fn object_literals_support_named_computed_and_shorthand_properties() {
    check(
        "function main() { let x = 42; return ({ x }).x; }",
        Value::number(42.0),
    );
    check(
        "function main() { return ({ a: { b: 42 } }).a.b; }",
        Value::number(42.0),
    );
    check(
        "function main() { return ({ 'é😀': 42 })['é😀']; }",
        Value::number(42.0),
    );
    check(
        "function main() { return ({ 42: 7 })[42]; }",
        Value::number(7.0),
    );
    check(
        "function main() { let k = 'x'; return ({ [k]: 42 }).x; }",
        Value::number(42.0),
    );
    check(
        "function main() { return ({ [-0]: 42 })[0]; }",
        Value::number(42.0),
    );
    check(
        "function main() { return ({ [true]: 42 })['true']; }",
        Value::number(42.0),
    );
    check("function main() { return ({}).missing; }", Value::UNDEFINED);
    check(
        "function main() { let o = {}; o['x'] = 6; o['x'] ^= 3; return o.x; }",
        Value::number(5.0),
    );
    check(
        "function main() { let o = {}; let a = o; a.x = 42; return o === a && o !== {} && o.x === 42; }",
        Value::TRUE,
    );
    check(
        "function main() { return ({ f: function(x) { return x + 1; } }).f(41); }",
        Value::number(42.0),
    );
    check(
        "function main() { let o = { ['__proto__']: 42 }; return o['__proto__']; }",
        Value::number(42.0),
    );
}

#[test]
fn object_properties_evaluate_in_source_order_and_survive_exceptions() {
    check(
        "function main() { let n = 0; let o = { [++n]: ++n, [++n]: ++n }; return o[1] * 10 + o[3]; }",
        Value::number(24.0),
    );
    check(
        "function main() { let n = 0; let o = { x: ++n, x: ++n }; return o.x * 10 + n; }",
        Value::number(22.0),
    );
    check(
        "function main() { let o = { x: [42] }; o.x = o.x; return o.x[0]; }",
        Value::number(42.0),
    );
    check(
        "function fail() { throw { message: 'oops', data: [42] }; } function main() { try { fail(); } catch (e) { return e.data[0]; } }",
        Value::number(42.0),
    );
    check(
        "function main() { let n = 0; let fail = () => { throw 42; }; try { let o = { a: fail(), b: ++n }; } catch {} return n; }",
        Value::number(0.0),
    );
}

#[test]
fn object_literals_reject_unsupported_property_semantics() {
    for expression in [
        "{ get x() { return 1; } }",
        "{ f() {} }",
        "{ ...{} }",
        "{ __proto__: null }",
        "{ '\\uD800': 1 }",
    ] {
        let mut engine = Engine::new().unwrap();
        assert!(matches!(
            engine.compile(
                &format!("function main() {{ return ({expression}); }}"),
                "object.js"
            ),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn native_sets_support_membership_and_mutation() {
    check(
        "function main() { let s = new Set(); s.add(42).add(42).add('x'); return s.size === 2 && s.has(42) && s.has('x') && !s.has(1); }",
        Value::TRUE,
    );
    check(
        "function main() { let s = new Set(); s.add(1); return s.delete(1) && !s.delete(1) && s.size === 0; }",
        Value::TRUE,
    );
    check(
        "function main() { let s = new Set(); s.add(undefined); s.clear(); s.add(42); return !s.has(undefined) && s.size === 1; }",
        Value::TRUE,
    );
    check(
        "function main() { let s = new Set(); s.add(); return s.has() && s.size === 1; }",
        Value::TRUE,
    );
    check(
        "function main() { let s = new Set(); s.add(0 / 0); s.add(0 / 0); s.add(-0); s.add(0); return s.size === 2 && s.has(0 / 0) && s.has(-0); }",
        Value::TRUE,
    );
    check(
        "function main() { let s = new Set(); s.add('é' + '😀'); return s.has('é😀'); }",
        Value::TRUE,
    );
    check(
        "function main() { let a = {}; let s = new Set(); s.add(a); s.add({}); return s.size === 2 && s.has(a) && !s.has({}); }",
        Value::TRUE,
    );
    check(
        "function main() { let a = new Set(); let b = new Set(); a.add(1); b.add(2); return a.has(1) && !b.has(1) && a.add === b.add; }",
        Value::TRUE,
    );
    check(
        "function main() { let s = new Set(); s['add'](42); return (s.has)(42) && typeof s.add === 'function' && typeof Set === 'function'; }",
        Value::TRUE,
    );
    check(
        "function main() { let s = new Set(); let alias = s; s.add(s); return alias.has(s); }",
        Value::TRUE,
    );
}

#[test]
fn native_set_methods_require_receivers_and_constructors() {
    for source in [
        "function main() { return Set(); }",
        "function main() { return new Set([1]); }",
        "function main() { let s = new Set(); let f = s.add; return f(1); }",
        "function main() { let s = new Set(); return ({ f: s.has }).f(1); }",
        "function main() { let s = new Set(); return new s.add(); }",
    ] {
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "set.js").unwrap();
        assert!(matches!(engine.call("main", &[]), Err(Error::Runtime(_))));
    }
}
