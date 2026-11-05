use js_engine::{Engine, EngineOptions, Host, Value};

struct TestHost;

impl Host for TestHost {
    fn read_file(&self, _: &str) -> Result<String, String> {
        Err("unused".into())
    }
    fn list_files(&self, _: &str, _: &str, _: bool) -> Result<String, String> {
        Err("unused".into())
    }
}

#[test]
fn random_vectors_reseeding_and_primitive_coercion() {
    let mut engine = Engine::new().unwrap();
    engine.compile(r#"function main(){
        hostSeedRandom(0);
        if (Math.random()*2147483648 !== 366850414 || Math.random()*2147483648 !== 1610402240) return false;
        hostSeedRandom(1);
        if (Math.random()*2147483648 !== 89400484 || Math.random()*2147483648 !== 976015093) return false;
        hostSeedRandom(-1);
        if (Math.random()*2147483648 !== 644300343 || Math.random()*2147483648 !== 97305740) return false;
        hostSeedRandom('1'); let a=Math.random(); hostSeedRandom(1); if (a!==Math.random()) return false;
        hostSeedRandom(null); let b=Math.random(); hostSeedRandom(); if (b!==Math.random()) return false;
        hostSeedRandom(0/0); let c=Math.random(); hostSeedRandom(0); if (c!==Math.random()) return false;
        hostSeedRandom(0); let f=Math.random; let d=f({}); hostSeedRandom(0);
        return d===Math.random() && f===Math.random && Object.keys(Math).indexOf('random')===-1;
    }"#, "random.js").unwrap();
    assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    engine
        .compile("function bad(){ hostSeedRandom({}); }", "bad.js")
        .unwrap();
    assert!(engine.call("bad", &[]).is_err());
}

#[test]
fn random_state_persists_and_is_isolated_per_engine() {
    let source = "function next(){ return Math.random()*2147483648; }";
    let mut first = Engine::new().unwrap();
    let mut second = Engine::new().unwrap();
    first.compile(source, "first.js").unwrap();
    second.compile(source, "second.js").unwrap();
    assert_eq!(
        first.call("next", &[]).unwrap().value(),
        Value::number(366850414.0)
    );
    assert_eq!(
        first.call("next", &[]).unwrap().value(),
        Value::number(1610402240.0)
    );
    assert_eq!(
        second.call("next", &[]).unwrap().value(),
        Value::number(366850414.0)
    );
    first.install_host(TestHost).unwrap();
    first
        .compile(
            "hostEvaluateSource('hostSeedRandom(1);', 'seed.js');",
            "eval.js",
        )
        .unwrap();
    assert_eq!(
        first.call("next", &[]).unwrap().value(),
        Value::number(89400484.0)
    );
    assert_eq!(
        second.call("next", &[]).unwrap().value(),
        Value::number(1610402240.0)
    );
}

#[test]
fn guarded_random_preserves_unused_draws_loop_counts_and_replacements() {
    for enabled in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt: enabled,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "function seed(){hostSeedRandom(0);}
             function next(){return Math.random()*2147483648;}
             function skip(n){for(let i=0;i<n;++i) Math.random();}
             function discarded(){Math.random();return Math.random()*2147483648;}
             function sum(n){let total=0;for(let i=0;i<n;++i) total+=Math.random();return total;}
             function replace(){Math.random=function(){Math.random=()=>0.75;return 0.25;};}",
                "guarded-random.js",
            )
            .unwrap();
        engine.call("seed", &[]).unwrap();
        engine.call("skip", &[Value::number(0.0)]).unwrap();
        assert_eq!(
            engine.call("next", &[]).unwrap().value(),
            Value::number(366850414.0)
        );
        engine.call("seed", &[]).unwrap();
        engine.call("skip", &[Value::number(1.0)]).unwrap();
        assert_eq!(
            engine.call("next", &[]).unwrap().value(),
            Value::number(1610402240.0)
        );
        engine.call("seed", &[]).unwrap();
        assert_eq!(
            engine.call("discarded", &[]).unwrap().value(),
            Value::number(1610402240.0)
        );
        engine.call("seed", &[]).unwrap();
        let total = engine.call("sum", &[Value::number(2.0)]).unwrap().value();
        assert_eq!(total, Value::number(1977252654.0 / 2147483648.0));
        engine.call("replace", &[]).unwrap();
        assert_eq!(
            engine.call("sum", &[Value::number(3.0)]).unwrap().value(),
            Value::number(1.75)
        );
    }
}
