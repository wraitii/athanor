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
