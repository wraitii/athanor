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

#[test]
fn math_functions_and_global_object() {
    let source = r#"function main() {
        let floor = Math.floor;
        let m = Math;
        Math.extra = 7;
        return floor === m.floor && floor(2.9) === 2 && Math.ceil(-1.2) === -1 &&
            Math.sqrt(9) === 3 && Math.abs(-4) === 4 && Math.imul(4294967295, 3) === -3 &&
            Math.sin(0) === 0 && Math.cos(0) === 1 && Math.PI > 3 && Math.PI < 4 && m.extra === 7;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    assert_eq!(
        run("function main(){ let m = Math; Math = 5; return m.floor(2.5) + Math; }").unwrap(),
        Value::number(7.0)
    );
}

#[test]
fn math_edge_cases() {
    let source = r#"function main() {
        let nan = 0/0;
        return 1/Math.round(-0.5) === -1/0 && Math.round(-1.5) === -1 &&
            Math.round(1.5) === 2 && 1/Math.ceil(-0.1) === -1/0 &&
            1/Math.min(0, -0) === -1/0 && 1/Math.max(-0, 0) === 1/0 &&
            Math.min() === 1/0 && Math.max() === -1/0 &&
            Math.min(1, nan) !== Math.min(1, nan) && Math.pow(1, 1/0) !== Math.pow(1, 1/0) &&
            Math.abs() !== Math.abs() && Math.abs(null) === 0 && Math.sign(true) === 1;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return Math.abs('1'); }",
        "function main(){ return Math.max(1, {}); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn json_parse_and_stringify() {
    let source = r#"function main() {
        let p = JSON.parse;
        let obj = p('{"z":[null,true,false,"é\\n",2],"a":{"x":3}}');
        let text = JSON.stringify(obj);
        let copy = JSON.parse(text);
        return p === JSON.parse && copy.z[0] === null && copy.z[1] === true &&
            copy.z[2] === false && copy.z[3] === 'é\n' && copy.z[4] === 2 &&
            copy.a.x === 3 && text === '{"z":[null,true,false,"é\\n",2],"a":{"x":3}}';
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    assert_eq!(run("function main(){ return JSON.stringify({a:undefined,b:[undefined,0/0,1/0,-0],c:()=>3}) === '{\"b\":[null,null,null,0]}'; }").unwrap(), Value::TRUE);
    assert_eq!(run("function main(){ return JSON.stringify(undefined) === undefined && JSON.stringify(() => 1) === undefined; }").unwrap(), Value::TRUE);
    assert_eq!(run("function main(){ let a = {x:1}; return JSON.stringify([a,a]) === '[{\"x\":1},{\"x\":1}]'; }").unwrap(), Value::TRUE);
}

#[test]
fn json_rejects_unsupported_or_invalid_inputs() {
    for source in [
        "function main(){ return JSON.parse('{'); }",
        "function main(){ return JSON.parse(1); }",
        "function main(){ return JSON.parse('1', () => 1); }",
        "function main(){ return JSON.stringify({}, null); }",
        "function main(){ return JSON.stringify({}, undefined, 2); }",
        "function main(){ let a = {}; a.a = a; return JSON.stringify(a); }",
        "function main(){ return JSON.stringify({toJSON: () => 1}); }",
        "function main(){ return JSON.stringify(new Set()); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}
