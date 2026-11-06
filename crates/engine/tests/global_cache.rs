use js_engine::{Engine, EngineOptions, Value};

#[test]
fn cached_globals_observe_assignments_and_binding_growth() {
    for enabled in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt: enabled,
            ..Default::default()
        })
        .unwrap();
        engine.compile("var value=1; function read(){return value;} function update(){globalThis.value=3;return read();}", "globals.js").unwrap();
        for _ in 0..3 {
            assert_eq!(
                engine.call("read", &[]).unwrap().value(),
                Value::number(1.0)
            );
        }
        assert!(engine.global().set("value", Value::number(2.0)));
        assert_eq!(
            engine.call("read", &[]).unwrap().value(),
            Value::number(2.0)
        );
        for i in 0..1000 {
            assert!(
                engine
                    .global()
                    .define(format!("extra{i}"), Value::number(i as f64))
            );
        }
        assert_eq!(
            engine.call("read", &[]).unwrap().value(),
            Value::number(2.0)
        );
        assert_eq!(
            engine.call("update", &[]).unwrap().value(),
            Value::number(3.0)
        );
    }
}

#[test]
fn cached_globals_retain_replaced_heap_values() {
    let mut engine = Engine::new().unwrap();
    engine.compile("var value; function read(){return value;} function save(){const previous=value;value={text:'new'};return previous.text;}", "heap.js").unwrap();
    for i in 0..20 {
        let text = format!("value{i}");
        let value = engine.string(&text).unwrap();
        assert!(engine.global().set("value", value.value()));
        drop(value);
        let saved = engine.call("read", &[]).unwrap();
        assert!(engine.global().set("value", Value::UNDEFINED));
        assert_eq!(saved.text().as_deref(), Some(text.as_str()));
    }
    engine.compile("value={text:'old'};", "object.js").unwrap();
    assert_eq!(
        engine.call("save", &[]).unwrap().text().as_deref(),
        Some("old")
    );
}

#[test]
fn cached_globals_handle_missing_names_lexical_shadowing_and_tdz() {
    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "function read(){return value;} function kind(){return typeof value;}",
            "lookup.js",
        )
        .unwrap();
    assert!(engine.call("read", &[]).is_err());
    assert_eq!(
        engine.call("kind", &[]).unwrap().text().as_deref(),
        Some("undefined")
    );
    assert!(engine.global().define("value", Value::number(1.0)));
    assert_eq!(
        engine.call("read", &[]).unwrap().value(),
        Value::number(1.0)
    );
    assert_eq!(
        engine.call("kind", &[]).unwrap().text().as_deref(),
        Some("number")
    );
    assert!(
        engine
            .compile("let value=read();", "redeclaration.js")
            .is_err()
    );
    assert_eq!(
        engine.call("read", &[]).unwrap().value(),
        Value::number(1.0)
    );

    let mut engine = Engine::new().unwrap();
    engine
        .compile(
            "function read(){return value;} function kind(){return typeof value;}",
            "lookup.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("kind", &[]).unwrap().text().as_deref(),
        Some("undefined")
    );
    assert!(engine.compile("let value=read();", "tdz.js").is_err());
    assert!(engine.call("read", &[]).is_err());
    assert!(engine.call("kind", &[]).is_err());

    let mut engine = Engine::new().unwrap();
    engine
        .compile("let value=2; function read(){return value;}", "lexical.js")
        .unwrap();
    for _ in 0..3 {
        assert_eq!(
            engine.call("read", &[]).unwrap().value(),
            Value::number(2.0)
        );
    }
    engine.compile("value=3;", "assign.js").unwrap();
    assert_eq!(
        engine.call("read", &[]).unwrap().value(),
        Value::number(3.0)
    );
}
