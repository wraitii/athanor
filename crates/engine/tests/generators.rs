use js_engine::{Engine, EngineOptions, Error, OptLevel, Value};

fn check(source: &str, expected: Value) {
    for opt_level in [OptLevel::None, OptLevel::Speed] {
        let mut engine = Engine::with_options(EngineOptions { opt_level }).unwrap();
        engine.compile(source, "generators.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), expected);
    }
}

#[test]
fn generators_are_lazy_and_resume_to_completion() {
    check(
        "var n = 0; function* gen() { ++n; yield 10; ++n; yield 20; return 30; } function main() { let g = gen(); if (n !== 0) return false; let a = g.next(); if (a.value !== 10 || a.done || n !== 1) return false; let b = g.next(); if (b.value !== 20 || b.done || n !== 2) return false; let c = g.next(); let d = g.next(); return c.done && c.value === 30 && d.done && d.value === undefined && n === 2; }",
        Value::TRUE,
    );
    check(
        "function* gen() { yield; } function main() { let g = gen(); let a = g.next(); let b = g.next(); return a.value === undefined && !a.done && b.done; }",
        Value::TRUE,
    );
    check(
        "function* gen() { return 42; } function main() { let r = gen().next(); return r.done && r.value === 42; }",
        Value::TRUE,
    );
}

#[test]
fn generator_frames_preserve_locals_parameters_and_closures() {
    check(
        "function* gen(x, ...rest) { let a = [x]; yield a; a[0] += rest[0]; yield a[0]; return x; } function main() { let g = gen(40, 2); let a = g.next().value; let b = g.next().value; let c = g.next().value; return a[0] === 42 && b === 42 && c === 40; }",
        Value::TRUE,
    );
    check(
        "function* gen() { let x = 1; let f = () => x; yield f; ++x; yield x; } function main() { let g = gen(); let f = g.next().value; g.next(); return f(); }",
        Value::number(2.0),
    );
    check(
        "function main() { let x = 40; let gen = function*() { yield x; ++x; return x; }; let g = gen(); ++x; let a = g.next().value; let b = g.next().value; return a === 41 && b === 42 && x === 42; }",
        Value::TRUE,
    );
    check(
        "function* gen(x) { yield x; return x + 1; } function main() { let a = gen(1); let b = gen(10); a.next(); b.next(); return a.next().value * 10 + b.next().value; }",
        Value::number(31.0),
    );
    check(
        "function* gen() { yield 1; let x = 42; return x; } function main() { let g = gen(); g.next(); return g.next().value; }",
        Value::number(42.0),
    );
}

#[test]
fn generator_exceptions_propagate_and_close_the_generator() {
    check(
        "function* gen() { yield 1; throw new Error('oops'); } function main() { let g = gen(); g.next(); try { g.next(); } catch (e) { return e.message === 'oops' && g.next().done; } return false; }",
        Value::TRUE,
    );
    check(
        "function* gen() { try { throw 42; } catch (e) { return e; } } function main() { return gen().next().value; }",
        Value::number(42.0),
    );
}

#[test]
fn unsupported_generator_forms_are_rejected() {
    for source in [
        "function* gen() { let x = yield 1; }",
        "function* gen() { while (true) { yield 1; } }",
        "function* gen() { try { yield 1; } catch {} }",
        "function* gen() { yield* []; }",
        "async function* gen() { yield 1; }",
    ] {
        let mut engine = Engine::new().unwrap();
        assert!(matches!(
            engine.compile(source, "unsupported.js"),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn generator_runtime_restrictions_fail_without_corrupting_the_engine() {
    for source in [
        "function* gen() { yield 1; } function main() { let next = gen().next; next(); }",
        "function* gen() { yield 1; } function main() { gen().next(42); }",
        "var g; function* gen() { g.next(); } function main() { g = gen(); g.next(); }",
    ] {
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "invalid-generator.js").unwrap();
        assert!(matches!(engine.call("main", &[]), Err(Error::Runtime(_))));
        engine
            .compile("function good() { return 42; }", "good.js")
            .unwrap();
        assert_eq!(
            engine.call("good", &[]).unwrap().value(),
            Value::number(42.0)
        );
    }
}

#[test]
fn generators_handle_unreachable_yields_and_nested_resumption() {
    check(
        "function* gen() { return 42; yield 1; } function main() { return gen().next().value; }",
        Value::number(42.0),
    );
    check(
        "function* inner() { yield 10; return 20; } function* outer() { let g = inner(); yield g.next().value; return g.next().value; } function main() { let g = outer(); return g.next().value + g.next().value; }",
        Value::number(30.0),
    );
    check(
        "function* gen() { let n = 0; for (const x of [1, 2, 3]) n += x; yield n; for (const x of [4, 5]) n += x; return n; } function main() { let g = gen(); return g.next().value + g.next().value; }",
        Value::number(21.0),
    );
}

#[test]
fn yielded_closures_survive_generator_completion() {
    check(
        "function* gen() { let a = [42]; yield () => a[0]; return 0; } function main() { let g = gen(); let f = g.next().value; g.next(); g = undefined; return f(); }",
        Value::number(42.0),
    );
}
