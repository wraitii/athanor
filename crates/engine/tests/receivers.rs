use js_engine::{Engine, EngineOptions, Error, OptLevel, Value};

fn check(source: &str, expected: Value) {
    for opt_level in [OptLevel::None, OptLevel::Speed] {
        let mut engine = Engine::with_options(EngineOptions { opt_level }).unwrap();
        engine.compile(source, "receivers.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), expected);
    }
}

#[test]
fn methods_receive_the_evaluated_base_and_detached_calls_receive_undefined() {
    check(
        "function f() { return this; } function main() { let o = { f }; return o.f() === o && o['f']() === o && (o.f)() === o && f() === undefined; }",
        Value::TRUE,
    );
    check(
        "function f() { return this.x; } function main() { let a = { x: 42, f }; let b = { x: 7, f: a.f }; return a.f() === 42 && b.f() === 7; }",
        Value::TRUE,
    );
    check(
        "function main() { let n = 0; let o = { f: function(x) { return this.x + x; }, x: 40 }; let get = () => { ++n; return o; }; return get().f((o = {}, 2)) === 42 && n === 1; }",
        Value::TRUE,
    );
    check(
        "var topThis = this; var topArrow = () => this; function main() { return topThis === globalThis && topArrow() === globalThis; }",
        Value::TRUE,
    );
}

#[test]
fn arrows_capture_lexical_this_transitively_and_survive_the_parent() {
    check(
        "function main() { let o = { x: 42, make: function() { return () => this.x; } }; let f = o.make(); o = undefined; return ({ x: 7, f }).f(); }",
        Value::number(42.0),
    );
    check(
        "function main() { let o = { x: 42, make: function() { return () => () => this.x; } }; let middle = o.make(); o = undefined; return middle()(); }",
        Value::number(42.0),
    );
    check(
        "function main() { let o = { x: 1, f: function() { return function() { return () => this.x; }; } }; let f = o.f(); let other = { x: 42, f }; let arrow = other.f(); return arrow(); }",
        Value::number(42.0),
    );
    check(
        "function main() { let f = () => this; return ({ f }).f() === undefined; }",
        Value::TRUE,
    );
}

#[test]
fn basic_constructors_bind_this_and_apply_return_rules() {
    check(
        "function C(x) { this.x = x; } function main() { let o = new C(42); return o.x; }",
        Value::number(42.0),
    );
    check(
        "function C() { this.x = 42; return 7; } function main() { return new C().x; }",
        Value::number(42.0),
    );
    check(
        "function C() { this.x = 1; return { x: 42 }; } function main() { return new C().x; }",
        Value::number(42.0),
    );
    check(
        "function C() { this.x = 42; return () => this.x; } function main() { let f = new C(); return f(); }",
        Value::number(42.0),
    );
    check(
        "function C() { this.x = 42; throw this; } function main() { try { new C(); } catch (e) { return e.x; } }",
        Value::number(42.0),
    );
}

#[test]
fn generators_retain_the_original_receiver_across_yields() {
    check(
        "function* gen() { yield this.x; this.x += 1; return () => this.x; } function main() { let o = { x: 41, gen }; let g = o.gen(); o = undefined; let a = g.next().value; let f = g.next().value; g = undefined; return a === 41 && f() === 42; }",
        Value::TRUE,
    );
    check(
        "function main() { let o = { x: 42, gen: function*() { yield this.x; return this; } }; let g = o.gen(); let other = { next: g.next }; g.next(); return g.next().value === o; }",
        Value::TRUE,
    );
}

#[test]
fn arrows_and_generators_are_not_constructors() {
    for source in [
        "function main() { let f = () => 1; return new f(); }",
        "function* gen() { yield 1; } function main() { return new gen(); }",
        "function main() { let gen = function*() { yield 1; }; return new gen(); }",
    ] {
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "constructors.js").unwrap();
        assert!(matches!(engine.call("main", &[]), Err(Error::Runtime(_))));
    }
}

#[test]
fn host_calls_respect_arrow_receivers_and_evaluated_scripts_use_global_this() {
    check("var main = () => this === globalThis;", Value::TRUE);
    check(
        "function main() { return this === undefined; }",
        Value::TRUE,
    );
}

#[test]
fn constructor_prototypes_preserve_identity_receivers_and_shadowing() {
    check(
        "function Area(x) { this.x = x; } function main() { Area.prototype.get = function() { return this.x; }; let first = new Area(42); let second = new Area(7); first.get = undefined; return first.get === undefined && second.get() === 7 && Area.prototype === Area.prototype; }",
        Value::TRUE,
    );
    check(
        "function Area() {} function main() { Area.prototype.x = 1; let first = new Area(); Area.prototype = { x: 2 }; let second = new Area(); Area.prototype.x = 3; return first.x === 1 && second.x === 3; }",
        Value::TRUE,
    );
    check(
        "function Area() {} function main() { Area.prototype = 42; let area = new Area(); return area.missing === undefined; }",
        Value::TRUE,
    );
}

#[test]
fn instanceof_checks_prototype_identity_and_arrays() {
    check(
        "function A() {} function B() {} function main() { B.prototype = new A(); let b = new B(); let Alias = Array; return b instanceof B && b instanceof A && [] instanceof Alias && !({} instanceof Array) && !(1 instanceof A); }",
        Value::TRUE,
    );
    check(
        "function A() {} function main() { let a = new A(); A.prototype = {}; return !(a instanceof A); }",
        Value::TRUE,
    );
}

#[test]
fn typeof_distinguishes_supported_value_kinds() {
    check(
        "function main() { return typeof 1 === 'number' && typeof true === 'boolean' && typeof null === 'object' && typeof undefined === 'undefined' && typeof 'x' === 'string' && typeof [] === 'object' && typeof {} === 'object' && typeof (() => 1) === 'function' && typeof Array === 'function' && typeof (new Set()) === 'object'; }",
        Value::TRUE,
    );
}

#[test]
fn unsupported_expressions_report_filename_and_span() {
    let mut engine = Engine::new().unwrap();
    let error = engine
        .compile(
            "function main() { return delete ({}).x; }",
            "operator-test.js",
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("operator-test.js"));
    assert!(error.contains("unary operator"));
    assert!(error.contains(" at "));
}

#[test]
fn instanceof_rejects_invalid_right_operands_and_prototypes() {
    for source in [
        "function main() { return {} instanceof 1; }",
        "function A() {} function main() { A.prototype = 1; return {} instanceof A; }",
    ] {
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "instanceof-errors.js").unwrap();
        assert!(matches!(engine.call("main", &[]), Err(Error::Runtime(_))));
    }
}

#[test]
fn default_parameters_are_lazy_and_use_earlier_parameters_and_captures() {
    check(
        "var calls = 0; function next() { ++calls; return 42; } function f(a = next(), b = a) { return b; } function main() { return f() === 42 && f(undefined) === 42 && f(null) === null && f(0) === 0 && calls === 2; }",
        Value::TRUE,
    );
    check(
        "function main() { let x = 42; let f = (a = x) => a; let g = function(a = [], b = a) { return a === b; }; return f() === 42 && g(); }",
        Value::TRUE,
    );
}

#[test]
fn in_checks_presence_including_undefined_and_inherited_properties() {
    check(
        "function A() {} function main() { A.prototype.x = undefined; let a = new A(); let o = { y: undefined, 1: 42, null: true }; return 'x' in a && !('y' in a) && 'y' in o && 1 in o && null in o && 'prototype' in A && 'Array' in globalThis; }",
        Value::TRUE,
    );
    check(
        "function main() { let a = [undefined, 2]; return 0 in a && '1' in a && !(2 in a) && !('01' in a) && !('-0' in a) && 'length' in a && 'map' in a; }",
        Value::TRUE,
    );
    check(
        "function main() { let n = 0; let key = () => { n = n * 10 + 1; return 'x'; }; let object = () => { n = n * 10 + 2; return { x: 1 }; }; return key() in object() && n === 12; }",
        Value::TRUE,
    );
}

#[test]
fn in_rejects_primitive_receivers_and_object_keys() {
    for source in [
        "function main() { return 'length' in 'abc'; }",
        "function main() { return 'x' in null; }",
        "function main() { return {} in {}; }",
    ] {
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "in-errors.js").unwrap();
        assert!(matches!(engine.call("main", &[]), Err(Error::Runtime(_))));
    }
}

#[test]
fn property_updates_preserve_results_and_evaluate_the_base_once() {
    check(
        "function main() { let o = { x: 1 }; let n = 0; let get = () => { ++n; return o; }; let a = get().x++; let b = ++get().x; let c = get().x--; let d = --get().x; return a === 1 && b === 3 && c === 3 && d === 1 && o.x === 1 && n === 4; }",
        Value::TRUE,
    );
    check(
        "function A() {} function main() { A.prototype.x = 4; let a = new A(); let old = a.x++; return old === 4 && a.x === 5 && A.prototype.x === 4; }",
        Value::TRUE,
    );
    check(
        "function main() { let o = { x: null }; return o.x++ === 0 && o.x === 1; }",
        Value::TRUE,
    );
}
