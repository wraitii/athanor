use js_engine::{Engine, EngineOptions, Value};

fn check(source: &str, expected: f64) {
    let mut engine = Engine::new().unwrap();
    engine.compile(source, "ownership.js").unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(expected)
    );
}

#[test]
fn specialized_slot_stores_release_overwritten_values() {
    for property in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        let source = if property {
            "function make(){return {};}
             function wrap(value){return {slot:value};}
             function replace(object){object.slot=7;}"
        } else {
            "function make(){return {};}
             function wrap(value){return [value];}
             function replace(object){object[0]=7;}"
        };
        engine.compile(source, "slot_release.js").unwrap();
        let warm = engine.call("wrap", &[Value::number(0.0)]).unwrap();
        for _ in 0..16 {
            engine.call("replace", &[warm.value()]).unwrap();
        }
        let old = engine.call("make", &[]).unwrap();
        let stale = old.value();
        let object = engine.call("wrap", &[stale]).unwrap();
        drop(old);
        engine.call("replace", &[object.value()]).unwrap();
        engine.drain_disposals(1000);
        assert!(
            !engine.global().define("disposed", stale),
            "property={property}"
        );
    }
}

#[test]
fn allocation_results_are_released_after_forwarding_and_errors() {
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "var witness;
            function Item(texture, template = undefined) {
                this.texture = texture;
                this.template = template;
                return undefined;
            }
            function forward(value) { return value; }
            function make(flag) {
                const parts = 'grass|tree'.split('|');
                const value = new Item(...parts);
                return flag ? forward(value) : value;
            }
            function texture(value) { return value.texture; }
            function fail() { witness = make(true); return missing; }",
                "owned_allocations.js",
            )
            .unwrap();
        for flag in [Value::TRUE, Value::FALSE] {
            let result = engine.call("make", &[flag]).unwrap();
            let text = engine.call("texture", &[result.value()]).unwrap();
            assert_eq!(text.text().as_deref(), Some("grass"));
            let stale_result = result.value();
            let stale_text = text.value();
            drop(result);
            engine.drain_disposals(1000);
            assert_eq!(text.text().as_deref(), Some("grass"));
            drop(text);
            engine.drain_disposals(1000);
            assert!(!engine.global().define("disposed", stale_result));
            assert!(!engine.global().define("disposed", stale_text));
        }
        assert!(engine.call("fail", &[]).is_err());
        let witness = engine.global().get("witness").unwrap();
        let stale = witness.value();
        assert!(engine.global().set("witness", Value::UNDEFINED));
        drop(witness);
        engine.drain_disposals(1000);
        assert!(!engine.global().define("disposed", stale));
    }
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

#[test]
fn owned_helper_results_survive_slot_moves_and_are_disposed() {
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "function maker(){ return {answer:42}; }
             function factory(){ return maker; }
             function main(make, flag){
                 const value = make();
                 if(flag){ return value; }
                 return [value][0];
             }
             function captured(flag){
                 let value = {answer:42};
                 function read(){ return value; }
                 const result = read();
                 if(flag){ return result; }
                 return [result][0];
             }
             function mapped(flag){
                 let values = [{answer:42}].map(value => value);
                 if(flag){ return values[0]; }
                 return [values[0]][0];
             }
             function read(value){ return value.answer; }",
                "slot_moves.js",
            )
            .unwrap();
        let make = engine.call("factory", &[]).unwrap();
        for _ in 0..16 {
            for flag in [Value::TRUE, Value::FALSE] {
                for (name, args) in [
                    ("main", vec![make.value(), flag]),
                    ("captured", vec![flag]),
                    ("mapped", vec![flag]),
                ] {
                    let result = engine.call(name, &args).unwrap();
                    assert_eq!(
                        engine.call("read", &[result.value()]).unwrap().value(),
                        Value::number(42.0)
                    );
                    let stale = result.value();
                    drop(result);
                    engine.drain_disposals(1000);
                    assert!(!engine.global().define("disposed", stale));
                }
            }
        }
    }
}

#[test]
fn failed_calls_release_scratch_references_from_local_writes() {
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "var witness;
             function runtimeFailure(){ return missing; }
             function thrownFailure(){ throw 13; }
             function factory(flag){ return flag ? runtimeFailure : thrownFailure; }
             function main(raise, flag){
                 let value = [9];
                 witness = value;
                 value = flag ? [8] : [7];
                 return raise();
             }
             function caught(raise, flag){
                 try { return main(raise, flag); }
                 catch (error) { return error; }
             }",
                "scratch_errors.js",
            )
            .unwrap();
        for mode in [Value::TRUE, Value::FALSE] {
            let raise = engine.call("factory", &[mode]).unwrap();
            for _ in 0..16 {
                for (function, flag) in [
                    ("main", Value::TRUE),
                    ("main", Value::FALSE),
                    ("caught", Value::TRUE),
                    ("caught", Value::FALSE),
                ] {
                    let result = engine.call(function, &[raise.value(), flag]);
                    if function == "caught" && mode == Value::FALSE {
                        assert_eq!(result.unwrap().value(), Value::number(13.0));
                    } else {
                        assert!(result.is_err());
                    }
                    let witness = engine.global().get("witness").unwrap();
                    let stale = witness.value();
                    assert!(engine.global().set("witness", Value::UNDEFINED));
                    drop(witness);
                    engine.drain_disposals(1000);
                    assert!(!engine.global().define("disposed", stale));
                }
            }
        }
    }
}

#[test]
fn property_results_survive_receiver_replacement_without_leaking() {
    for (ssa_opt, background_compile) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            background_compile,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "function make(){ return {child:{value:7}}; }
             function take(parent){ const child=parent.child; parent.child=null; return child; }
             function read(child){ return child.value; }
             function method(){ return [].map; }
             function prototype(){ function C(){} return C.prototype; }",
                "borrowed_properties.js",
            )
            .unwrap();
        for _ in 0..8 {
            let parent = engine.call("make", &[]).unwrap();
            let child = engine.call("take", &[parent.value()]).unwrap();
            let stale = child.value();
            drop(parent);
            engine.drain_disposals(1000);
            assert_eq!(
                engine.call("read", &[stale]).unwrap().value(),
                Value::number(7.0)
            );
            drop(child);
            engine.drain_disposals(1000);
            assert!(!engine.global().define("disposed", stale));
            let method = engine.call("method", &[]).unwrap();
            assert_ne!(method.value(), Value::UNDEFINED);
            let prototype = engine.call("prototype", &[]).unwrap();
            let stale = prototype.value();
            drop(prototype);
            engine.drain_disposals(1000);
            assert!(!engine.global().define("disposed", stale));
        }
    }
}

#[test]
fn distinct_property_slots_preserve_aliased_values() {
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "function make(){return {x:{n:7},y:null};}
             function take(o){o.y=o.x;let value=o.x;o.y=null;return value;}
             function replace(o){let value=o.x;o.x=null;return value;}
             function chain(o){let value=o.x.child;o.x=null;return value;}
             function nested(){return {x:{child:{n:9}}};}
             function read(value){return value.n;}",
                "owning-slots.js",
            )
            .unwrap();
        for (make, take, expected) in [
            ("make", "take", 7.0),
            ("make", "replace", 7.0),
            ("nested", "chain", 9.0),
        ] {
            for _ in 0..16 {
                let owner = engine.call(make, &[]).unwrap();
                let value = engine.call(take, &[owner.value()]).unwrap();
                let stale = value.value();
                drop(owner);
                engine.drain_disposals(1000);
                assert_eq!(
                    engine.call("read", &[stale]).unwrap().value(),
                    Value::number(expected)
                );
                drop(value);
                engine.drain_disposals(1000);
                assert!(!engine.global().define("disposed", stale));
            }
        }
    }
}
