use js_engine::{Engine, Value};

fn run(source: &str) -> Result<Value, js_engine::Error> {
    let mut engine = Engine::new().unwrap();
    engine.compile(source, "builtins.js").unwrap();
    Ok(engine.call("main", &[])?.value())
}

#[test]
fn array_callbacks_observe_mutations_and_arguments() {
    let source = r#"function main() {
        let a = [1, 2, 3];
        let b = a.map((v, i, array) => {
            if (i === 0) { array[1] = 8; array.length = 2; }
            return v + i + array.length;
        });
        let count = 0;
        let c = b.filter((v, i, array) => { count++; return i === 1 && array === b; });
        return b.length === 3 && b[0] === 3 && b[1] === 11 && b[2] === undefined && count === 2 && c.join() === "11";
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn array_values_survive_callback_replacement_and_appends_are_ignored() {
    let source = r#"function main() {
        let a = [{v: 7}, {v: 9}];
        let b = a.filter((v, i, array) => {
            if (i === 0) { array[0] = null; array[2] = {v: 10}; }
            return true;
        });
        return b.length === 2 && b[0].v === 7 && b[1].v === 9;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn callback_errors_propagate() {
    assert!(run("function main() { return [].map(3); }").is_err());
    assert!(run("function main() { return [1].map(x => { throw 5; }); }").is_err());
    assert_eq!(
        run("function main() { try { [1].filter(x => { throw 5; }); } catch (e) { return e; } }")
            .unwrap(),
        Value::number(5.0)
    );
}

#[test]
fn strings_use_unicode_scalars() {
    let source = r#"function main() {
        let s = "a😀é/β";
        return s.slice(1, -2) === "😀é" && s.startsWith("é/", 2) &&
            s.split("").join("|") === "a|😀|é|/|β" &&
            s.split("/").join(":") === "a😀é:β" &&
            "abcabc".replace("bc", "x") === "axabc";
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn unsupported_forms_report_errors() {
    for source in [
        "function main() { return 'a'.split(',', 1); }",
        "function main() { return 'a'.replace('a', '$&'); }",
        "function main() { return [1].join({}); }",
        "function main() { return [1].map(x => x, {}); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}
