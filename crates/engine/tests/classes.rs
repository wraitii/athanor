use js_engine::{Engine, EngineOptions, Error, OptLevel, Value};

fn check(source: &str) {
    for opt_level in [OptLevel::None, OptLevel::Speed] {
        let mut engine = Engine::with_options(EngineOptions {
            opt_level,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "classes.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    }
}

#[test]
fn constructors_and_methods_share_a_prototype_and_preserve_receivers() {
    check(
        "class Area { constructor(x) { this.x = x; } get() { return this.x; } self() { return Area; } } function main() { let a = new Area(42); let b = new Area(7); return a.get() === 42 && b.get() === 7 && a.get === b.get && a instanceof Area && a.constructor === Area && a.self() === Area && typeof Area === 'function'; }",
    );
}

#[test]
fn default_constructors_and_local_captures_work() {
    check(
        "class Empty {} function main() { let x = 42; class Local { constructor(y = x) { this.y = y; } get() { return this.y; } self() { return Local; } } let a = new Local(); return new Empty() instanceof Empty && a.get() === 42 && a.self() === Local; }",
    );
    check(
        "function main() { class Empty {} let a = new Empty(); return a instanceof Empty && a.constructor === Empty; }",
    );
}

#[test]
fn class_calls_method_construction_and_early_access_fail() {
    for source in [
        "class A {} function main() { return A(); }",
        "class A { method() {} } function main() { return new (A.prototype.method)(); }",
        "function main() { let a = A; class A {} return a; }",
        "var early = A; class A {} function main() {}",
    ] {
        let mut engine = Engine::new().unwrap();
        match engine.compile(source, "classes.js") {
            Err(Error::Runtime(_)) => {}
            Ok(()) => assert!(
                matches!(engine.call("main", &[]), Err(Error::Runtime(_))),
                "{source}"
            ),
            Err(error) => panic!("{source}: {error}"),
        }
    }
    let mut engine = Engine::new().unwrap();
    engine.compile("class A {}", "classes.js").unwrap();
    assert!(matches!(engine.call("A", &[]), Err(Error::Runtime(_))));
}

#[test]
fn unsupported_class_features_fail_explicitly() {
    for source in [
        "class A extends B {}",
        "class A { method() { A = 1; } }",
        "class A { x = 1; }",
        "class A { static method() {} }",
        "class A { get x() { return 1; } }",
        "class A { ['method']() {} }",
        "class A { #method() {} }",
        "class A { static {} }",
    ] {
        let mut engine = Engine::new().unwrap();
        assert!(
            matches!(
                engine.compile(source, "classes.js"),
                Err(Error::Unsupported(_))
            ),
            "{source}"
        );
    }
}

#[test]
fn class_name_references_keep_the_original_constructor_after_reassignment() {
    check(
        "class A { self() { return A; } } var Original = A; A = undefined; function main() { return new Original().self() === Original; }",
    );
    check(
        "function main() { class A { self() { return A; } } let Original = A; A = undefined; return new Original().self() === Original; }",
    );
}

#[test]
fn method_replacement_detached_receivers_and_constructor_returns_work() {
    check(
        "class A { constructor() { return { x: 42 }; } } class B { constructor() { this.x = 7; return 1; } value() { return 1; } value() { return this; } } function main() { let b = new B(); let detached = b.value; return new A().x === 42 && b.x === 7 && b.value() === b && detached() === undefined && !(new A() instanceof A); }",
    );
}
