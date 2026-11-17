use super::*;

fn baseline_osr_engine(background_compile: bool) -> Engine {
    Engine::with_options(EngineOptions {
        optimization: crate::OptimizationLevel::Baseline,
        osr: true,
        background_compile,
        prefetch_compile: false,
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn baseline_osr_enters_observing_code_then_optimized_loop() {
    let mut engine = baseline_osr_engine(false);
    engine.compile("function main(){const a=[1,2,3];let sum=0;for(let i=0;i<2000;++i){if(i===1000)a[0]=0.5;sum+=a[i%3];}return sum+7;}", "baseline-osr.js").unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(3839.5)
    );
    let compiler = engine.tiering.compiler.borrow();
    assert!(compiler.baseline_osr_transfers() > 0);
    assert!(compiler.osr_transfers() > 0);
}

#[test]
fn baseline_osr_background_transfer_finishes_the_function() {
    let mut engine = baseline_osr_engine(true);
    engine.compile("function main(){const a=[1,2,3];let sum=0;for(let i=0;i<2000000;++i)sum+=a[i%3];return sum+7;}", "baseline-osr-background.js").unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(4000006.0)
    );
    let compiler = engine.tiering.compiler.borrow();
    assert!(compiler.baseline_osr_transfers() > 0);
    assert!(compiler.osr_transfers() > 0);
}

#[test]
fn baseline_osr_preserves_effects_and_owned_results() {
    let mut engine = baseline_osr_engine(false);
    engine.compile("let calls=0;function next(i){calls++;return [i];}function main(){let last=[-1];for(let i=0;i<200;++i)last=next(i);return [last,calls];}", "baseline-osr-owned.js").unwrap();
    let result = engine.call("main", &[]).unwrap();
    engine.drain_disposals(10000);
    let heap = engine.global().heap();
    let outer = result.value();
    let last = heap.get(outer, 0).unwrap();
    assert_eq!(heap.get(last, 0), Some(Value::number(199.0)));
    assert_eq!(heap.get(outer, 1), Some(Value::number(200.0)));
    assert!(engine.tiering.compiler.borrow().baseline_osr_transfers() > 0);
}

#[test]
fn baseline_osr_preserves_uninitialized_bindings_and_errors() {
    let mut engine = baseline_osr_engine(false);
    engine.compile("function main(){let sum=0;for(let i=0;i<200;++i){sum+=i;if(i===150)return later;}let later=7;return sum+later;}", "baseline-osr-tdz.js").unwrap();
    let error = engine.call("main", &[]).err().unwrap();
    assert!(
        error.to_string().contains("before initialization"),
        "{error}"
    );
    assert!(engine.tiering.compiler.borrow().baseline_osr_transfers() > 0);
    engine.drain_disposals(10000);
}

#[test]
fn baseline_osr_propagates_thrown_values_to_the_caller() {
    let mut engine = baseline_osr_engine(false);
    engine.compile("function fail(){const a=[7];let sum=0;for(let i=0;i<200;++i){sum+=a[0];if(i===150)throw a;}return sum;}function main(){try{return fail();}catch(e){return e[0];}}", "baseline-osr-throw.js").unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(7.0)
    );
    assert!(engine.tiering.compiler.borrow().baseline_osr_transfers() > 0);
    engine.drain_disposals(10000);
}

#[test]
fn baseline_call_warmup_upgrades_without_loop_feedback() {
    for warmup in [0, 64, 128, 1024] {
        for background_compile in [false, true] {
            for (eager, source) in [
                (false, "function f(a){return a+1;}"),
                (true, "function f(a){return a+1;}"),
                (false, "function f(){return arguments[0]+1;}"),
            ] {
                let mut engine = Engine::with_options(EngineOptions {
                    optimization: crate::OptimizationLevel::Baseline,
                    baseline_warmup: warmup,
                    osr: false,
                    background_compile,
                    prefetch_compile: false,
                    compile: eager.then(|| {
                        Arc::new(crate::CompileOptions {
                            semantic_ir: false,
                            ..Default::default()
                        })
                    }),
                    ..Default::default()
                })
                .unwrap();
                engine.compile(source, "baseline-calls.js").unwrap();
                let code = code(&engine, "f");
                let original = entry(&engine, code);
                if warmup == 0 {
                    for _ in 0..1025 {
                        assert_eq!(
                            engine.call("f", &[Value::number(41.0)]).unwrap().value(),
                            Value::number(42.0)
                        );
                    }
                    finish_compilation(&engine);
                    assert_eq!(entry(&engine, code), original);
                    continue;
                }
                for _ in 0..warmup - 1 {
                    assert_eq!(
                        engine.call("f", &[Value::number(41.0)]).unwrap().value(),
                        Value::number(42.0)
                    );
                }
                assert_eq!(entry(&engine, code), original);
                assert!(
                    engine
                        .tiering
                        .worker
                        .borrow()
                        .as_ref()
                        .is_none_or(|worker| worker.in_flight.is_empty())
                );
                assert_eq!(
                    engine.call("f", &[Value::number(41.0)]).unwrap().value(),
                    Value::number(42.0)
                );
                if background_compile {
                    assert!(
                        engine
                            .tiering
                            .worker
                            .borrow()
                            .as_ref()
                            .unwrap()
                            .in_flight
                            .contains_key(&jit::recompile::Key::ordinary(code))
                    );
                    finish_compilation(&engine);
                }
                assert_eq!(
                    engine.call("f", &[Value::number(41.0)]).unwrap().value(),
                    Value::number(42.0)
                );
                assert_ne!(entry(&engine, code), original);
                assert_eq!(
                    engine.call("f", &[Value::boolean(true)]).unwrap().value(),
                    Value::number(2.0)
                );
            }
        }
    }
}

#[test]
fn baseline_call_upgrade_preserves_captures_and_receivers() {
    for background_compile in [false, true] {
        let mut engine = baseline_osr_engine(background_compile);
        engine.compile("function make(){let value=[3];return function(n){value[0]+=n;this.last=value;return value;};}let holder={run:make()};function main(){return holder.run(1)[0];}function read(){return holder.last[0];}", "baseline-capture-calls.js").unwrap();
        for i in 0..160 {
            assert_eq!(
                engine.call("main", &[]).unwrap().value(),
                Value::number(4.0 + i as f64)
            );
        }
        finish_compilation(&engine);
        engine.drain_disposals(10000);
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::number(164.0)
        );
        assert_eq!(
            engine.call("read", &[]).unwrap().value(),
            Value::number(164.0)
        );
    }
}
