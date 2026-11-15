use js_engine::{Engine, EngineOptions, Error, Value};

fn check(source: &str, expected: Value) {
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "generators.js").unwrap();
        assert_eq!(
            engine
                .call("main", &[])
                .unwrap_or_else(|error| panic!("ssa={ssa_opt}: {error}"))
                .value(),
            expected
        );
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
        for ssa_opt in [false, true] {
            let mut engine = Engine::with_options(EngineOptions {
                ssa_opt,
                ..Default::default()
            })
            .unwrap();
            assert!(matches!(
                engine.compile(source, "unsupported.js"),
                Err(Error::Unsupported(_))
            ));
        }
    }
}

#[test]
fn generator_runtime_restrictions_fail_without_corrupting_the_engine() {
    for source in [
        "function* gen() { yield 1; } function main() { let next = gen().next; next(); }",
        "function* gen() { yield 1; } function main() { gen().next(42); }",
        "var g; function* gen() { g.next(); } function main() { g = gen(); g.next(); }",
    ] {
        for ssa_opt in [false, true] {
            let mut engine = Engine::with_options(EngineOptions {
                ssa_opt,
                ..Default::default()
            })
            .unwrap();
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

#[test]
fn generator_resume_preserves_tdz_and_arguments() {
    check(
        "function* f(x){ yield 1; return x ? 42 : 0; } function main(){ let g=f(true); g.next(); return g.next().value; }",
        Value::number(42.0),
    );
    check(
        "function* f(){ yield arguments[0]; return arguments.length; } function main(){ let g=f(5,6); return g.next().value*10+g.next().value; }",
        Value::number(52.0),
    );
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine.compile("function* f(){ yield 1; return x; let x=2; } function main(){ let g=f(); g.next(); return g.next(); }", "generator-tdz.js").unwrap();
        assert!(engine.call("main", &[]).is_err(), "ssa={ssa_opt}");
    }
}

#[test]
fn spread_arguments_work_before_and_after_suspension() {
    check(
        "function f(a,b){return a.x+b;} function* gen(xs){let first=f(...xs,2);yield first;return f(...xs,2);} function main(){let xs=[{x:40}];let g=gen(xs);if(g.next().value!==42)return false;xs[0]={x:7};let r=g.next();return r.done && r.value===9;}",
        Value::TRUE,
    );
    check(
        "function* gen(){let a=[...'a,b'.split(',')];yield a;return [...a,'c'];} function main(){let g=gen();g.next();let r=g.next();return r.done && r.value.join('')==='abc';}",
        Value::TRUE,
    );
}

#[test]
fn generator_region_local_scopes_keep_initialization_checks() {
    for ssa_opt in [false, true] {
        for source in [
            "function* gen(){return x;let x=42;} function main(){return gen().next();}",
            "function* gen(){yield 1;{let x=x;return x;}} function main(){let g=gen();g.next();return g.next();}",
        ] {
            let mut engine = Engine::with_options(EngineOptions {
                ssa_opt,
                background_compile: false,
                ..Default::default()
            })
            .unwrap();
            engine.compile(source, "generator-local-tdz.js").unwrap();
            assert!(engine.call("main", &[]).is_err(), "ssa={ssa_opt}: {source}");
        }
    }
}
