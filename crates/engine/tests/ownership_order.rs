use js_engine::{Engine, Value};

fn check(source: &str, expected: f64) {
    let mut engine = Engine::new().unwrap();
    engine.compile(source, "ownership.js").unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(expected)
    );
}

#[test]
fn environment_writes_reject_foreign_heap_values() {
    let mut source = Engine::new().unwrap();
    source
        .compile("function make(){ return [5]; }", "foreign.js")
        .unwrap();
    let destination = Engine::new().unwrap();
    let environment = destination.global().child();
    assert!(destination.global().define("existing", Value::number(7.0)));
    let string = source.string("foreign").unwrap();
    let array = source.call("make", &[]).unwrap();
    for value in [string.value(), array.value()] {
        assert!(!environment.define("foreign", value));
        assert!(environment.get("foreign").is_none());
        assert!(!environment.set("existing", value));
        assert_eq!(
            destination.global().get("existing").unwrap().value(),
            Value::number(7.0)
        );
    }
}

#[test]
fn environment_writes_retain_local_heap_values_and_reject_disposed_values() {
    let engine = Engine::new().unwrap();
    let first = engine.string("first").unwrap();
    assert!(engine.global().define("value", first.value()));
    drop(first);
    assert_eq!(
        engine.global().get("value").unwrap().text().as_deref(),
        Some("first")
    );
    let second = engine.string("second").unwrap();
    assert!(engine.global().set("value", second.value()));
    let stale = second.value();
    drop(second);
    assert_eq!(
        engine.global().get("value").unwrap().text().as_deref(),
        Some("second")
    );
    assert!(engine.global().set("value", Value::UNDEFINED));
    engine.drain_disposals(1000);
    assert!(!engine.global().define("stale", stale));
    assert!(!engine.global().set("value", stale));
}

#[test]
fn arguments_survive_later_argument_side_effects() {
    check(
        "var saved; function first(a, b){ return a; }
         function clear(){ saved = undefined; return 0; }
         function main(){ saved = [11]; return first(saved, clear())[0]; }",
        11.0,
    );
    check(
        "var saved; function first(a, b){ return a; }
         function clear(){ saved[0] = undefined; return 0; }
         function main(){ saved = [[9]]; return first(saved[0], clear())[0]; }",
        9.0,
    );
    check(
        "function first(a, b){ return a; }
         function main(){ let a = [7]; return first(a, a = undefined)[0]; }",
        7.0,
    );
}

#[test]
fn assignment_keeps_original_receiver_alive() {
    check(
        "var saved; function replace(){ saved = []; return 9; }
         function main(){ saved = [1]; let original = saved;
             saved[0] = replace(); return original[0]; }",
        9.0,
    );
    check(
        "function main(){ let a = [1]; let original = a;
             a[(a = [], 0)] = 12; return original[0]; }",
        12.0,
    );
}

#[test]
fn returned_arrays_survive_branches_and_nested_frames() {
    check(
        "function choose(flag){ { let a = [8]; if(flag){ return a; } } return [2]; }
         function main(){ let a = choose(true); let b = choose(false);
             return a[0] + b[0]; }",
        10.0,
    );
    check(
        "function nested(n, a){ if(n == 0){ return a; } return nested(n - 1, [a]); }
         function main(){ let a = nested(3, [4]); return a[0][0][0][0]; }",
        4.0,
    );
}

#[test]
fn loops_release_replaced_arrays_and_preserve_results() {
    check(
        "function main(){ let result = []; for(let i = 0; i < 200; i++){
             let item = [i]; result[0] = item; item = undefined;
         } return result[0][0]; }",
        199.0,
    );
}

#[test]
fn nested_error_releases_frames_without_losing_host_roots() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "function make(){ return [5]; }
         function fail(a){ let b = [a]; return missing; }
         function main(a){ let b = [a]; return fail(b); }
         function read(a){ return a[0]; }",
            "errors.js",
        )
        .unwrap();
    let array = engine.call("make", &[]).unwrap();
    for _ in 0..100 {
        assert!(engine.call("main", &[array.value()]).is_err());
        engine.drain_disposals(100);
        assert_eq!(
            engine.call("read", &[array.value()]).unwrap().value(),
            Value::number(5.0)
        );
    }
    let stale = array.value();
    drop(array);
    engine.drain_disposals(1000);
    assert!(engine.call("read", &[stale]).is_err());
}

#[test]
fn compiler_recovers_after_unsupported_array_syntax() {
    let mut engine = Engine::new().unwrap();
    assert!(
        engine
            .compile(
                "function first(){ return second(); } function second(){ return [,]; }",
                "unsupported.js",
            )
            .is_err()
    );
    engine
        .compile("function main(){ return [6][0]; }", "valid.js")
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(6.0)
    );
}
