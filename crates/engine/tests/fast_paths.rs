use js_engine::{Engine, EngineOptions, Error, OptLevel, Value};

#[test]
fn local_truthiness_and_indexing_preserve_semantics() {
    for opt_level in [OptLevel::None, OptLevel::Speed] {
        let mut engine = Engine::with_options(EngineOptions {
            opt_level,
            ..Default::default()
        })
        .unwrap();
        engine.compile(
            "function main() { let a = [7]; let z = -0; let n = 0/0; let empty = ''; let text = 'x'; return !z && !n && !empty && !!text && !!a && a[z] === 7 && a[5] === undefined && a['0'] === 7; }",
            "fast_paths.js",
        ).unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        engine
            .compile("function tdz() { return x; let x = 1; }", "tdz.js")
            .unwrap();
        assert!(matches!(engine.call("tdz", &[]), Err(Error::Runtime(_))));
    }
}
