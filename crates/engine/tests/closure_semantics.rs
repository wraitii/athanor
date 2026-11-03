use js_engine::{Engine, Error, Value};

#[test]
fn captured_binding_writes_respect_initialization() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "function main(){ let f = () => x; x = 2; let x; return f(); }",
            "tdz.js",
        )
        .unwrap();
    assert!(matches!(engine.call("main", &[]), Err(Error::Runtime(_))));
    assert!(matches!(
        Engine::new().unwrap().compile(
            "function main(){ const x = 1; return () => ++x; }",
            "const.js"
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn loop_block_bindings_are_distinct_and_loop_header_captures_are_rejected() {
    let mut engine = Engine::new().unwrap();
    engine.compile("function main(){ let fs = []; let i = 0; while (i < 2) { let x = i; fs[i] = () => x; i++; } let a = fs[0]; let b = fs[1]; return a() * 10 + b(); }", "loop.js").unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(1.0)
    );
}
