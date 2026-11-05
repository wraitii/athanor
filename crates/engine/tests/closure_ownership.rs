use js_engine::{Engine, EngineOptions, Value};

#[test]
fn escaped_capture_survives_host_and_argument_replacement() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "var saved;
         function make(){ let value = [17]; return () => value[0]; }
         function invoke(f, ignored){ return f(); }
         function main(){ saved = make(); return invoke(saved, saved = undefined); }",
            "closure-ownership.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(17.0)
    );
    let closure = engine.call("make", &[]).unwrap();
    engine.drain_disposals(1000);
    assert_eq!(
        engine.call("invoke", &[closure.value()]).unwrap().value(),
        Value::number(17.0)
    );
    let stale = closure.value();
    drop(closure);
    engine.drain_disposals(1000);
    assert!(engine.call("invoke", &[stale]).is_err());
}

#[test]
fn transitive_captures_preserve_shadowed_binding_identity() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "function make(){ let value = 7;
             let outer = () => () => ++value;
             { let value = 100; return [outer(), () => ++value]; }
         }
         function invoke(f){ return f(); }
         function main(){ let pair = make();
             let a = pair[0]; let b = pair[1]; return a() + b() + a(); }",
            "closure-shadow.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(118.0)
    );
}

#[test]
fn closures_observe_origin_globals_across_compilation() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "var base = 2; var saved;
         function make(){ let offset = 3; return () => base + offset; }
         saved = make();",
            "origin.js",
        )
        .unwrap();
    engine
        .compile(
            "base = 10; function main(){ let base = 100; return saved(); }",
            "later.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(13.0)
    );
}

#[test]
fn sibling_closures_share_updates_and_independent_activations_do_not() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "function make(value){ return [() => ++value, () => value]; }
         function invoke(f){ return f(); }
         function main(){ let a = make(1); let b = make(10);
             let inc = a[0]; let read = a[1]; let other = b[1];
             inc(); inc(); return read() + other(); }",
            "siblings.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(13.0)
    );
}

#[test]
fn dynamic_calls_follow_reassignment_and_preserve_callee_evaluation_order() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "function first(){ return 1; } function second(){ return 2; }
         function main(){ let f = first; first = second;
             let old = f(f = second); return old + f() + first(); }",
            "dynamic.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(5.0)
    );
    assert_eq!(
        engine.call("first", &[]).unwrap().value(),
        Value::number(2.0)
    );
}

#[test]
fn failing_closures_release_frames_and_preserve_host_capture() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "function make(){ let value = [19]; return () => value[0]; }
         function invoke(f){ return f(); }
         function fail(f){ let local = [f]; let bad = () => missing + local[0]; return bad(); }",
            "closure-error.js",
        )
        .unwrap();
    let closure = engine.call("make", &[]).unwrap();
    for _ in 0..20 {
        assert!(engine.call("fail", &[closure.value()]).is_err());
        engine.drain_disposals(1000);
        assert_eq!(
            engine.call("invoke", &[closure.value()]).unwrap().value(),
            Value::number(19.0)
        );
    }
}

#[test]
fn recursive_closures_and_captured_rest_work_at_each_optimization_level() {
    for opt_level in [js_engine::OptLevel::None, js_engine::OptLevel::Speed] {
        let mut engine = Engine::with_options(js_engine::EngineOptions {
            opt_level,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "function make(...values){ return function(){ return values[0]; }; }
             function main(){ let f = n => n ? n * f(n - 1) : 1;
                 let read = make(7); return f(5) + read(); }",
                "closure-recursion.js",
            )
            .unwrap();
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::number(127.0)
        );
    }
}

#[test]
fn captured_lexicals_check_initialization_for_reads_and_writes() {
    for source in [
        "function main(){ let f = () => x; f(); let x = 1; }",
        "function main(){ let f = () => x = 2; f(); let x = 1; }",
    ] {
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "tdz.js").unwrap();
        let error = engine.call("main", &[]).err().unwrap().to_string();
        assert!(error.contains("before initialization"), "{error}");
    }
}

#[test]
fn block_captures_are_fresh_but_var_loop_captures_are_shared() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "function main(){ let blocks = []; let vars = [];
             for (var i = 0; i < 3; i++){ let x = i;
                 blocks[i] = () => x; vars[i] = () => i;
             }
             let first = blocks[0]; let last = blocks[2]; let shared = vars[0];
             return first() + last() + shared(); }",
            "loop-captures.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(5.0)
    );
}

#[test]
fn unsupported_closure_semantics_fail_explicitly() {
    for source in [
        "function main(){ const x = 1; return () => ++x; }",
        "function main(){ return function named(){ return 1; }; }",
    ] {
        let error = Engine::new()
            .unwrap()
            .compile(source, "unsupported-closure.js")
            .unwrap_err();
        assert!(
            matches!(error, js_engine::Error::Unsupported(_)),
            "{source}: {error}"
        );
    }
}

#[test]
fn lazily_materialized_captures_retain_replaced_heap_values() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "var saved;
         function make(xs){ let value=[17]; xs.some(x=>value); value=[23]; return value; }
         function main(){ return make({some:f=>{saved=f; return f(1);}}); }
         function invoke(){ return saved()[0]; }
         function clear(){ saved=undefined; }",
            "lazy-capture-ownership.js",
        )
        .unwrap();
    let returned = engine.call("main", &[]).unwrap();
    drop(returned);
    engine.drain_disposals(1000);
    assert_eq!(
        engine.call("invoke", &[]).unwrap().value(),
        Value::number(23.0)
    );
    engine.call("clear", &[]).unwrap();
    engine.drain_disposals(1000);
    assert!(engine.call("invoke", &[]).is_err());
}

#[test]
fn borrowed_probe_returns_and_mutating_fallbacks_preserve_heap_values() {
    for enabled in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt: enabled,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "var target;
             function square(x){return x*x;}
             function setup(){target={item:[7],n:2};return target;}
             function read(a){let kept=a.item; square(a.n);return kept;}
             function replacement(x){target.item=[99];return x;}
             function switcher(){square=replacement;}
             function clear(){target=undefined;}
             function peek(a){return a[0];}",
                "borrowed-probe-ownership.js",
            )
            .unwrap();
        let holder = engine.call("setup", &[]).unwrap();
        let fast = engine.call("read", &[holder.value()]).unwrap();
        engine.call("switcher", &[]).unwrap();
        let slow = engine.call("read", &[holder.value()]).unwrap();
        engine.call("clear", &[]).unwrap();
        drop(holder);
        engine.drain_disposals(1000);
        assert_eq!(
            engine.call("peek", &[fast.value()]).unwrap().value(),
            Value::number(7.0)
        );
        assert_eq!(
            engine.call("peek", &[slow.value()]).unwrap().value(),
            Value::number(7.0)
        );
        let stale = fast.value();
        drop(fast);
        drop(slow);
        engine.drain_disposals(1000);
        assert!(engine.call("peek", &[stale]).is_err());
    }
}
