use js_engine::{Engine, Error, Host, Value};

struct TestHost;

impl Host for TestHost {
    fn read_file(&self, path: &str) -> Result<String, String> {
        match path {
            "outer" => Ok("hostEvaluateSource(hostReadFile('inner'), 'inner'); var result = innerValue + outerFunction(); function loaded() { return result; }".into()),
            "inner" => Ok("var innerValue = 2;".into()),
            "text" => Ok("é😀".into()),
            _ => Err(format!("missing file: {path}")),
        }
    }

    fn list_files(&self, path: &str, filter: &str, recursive: bool) -> Result<String, String> {
        Ok(format!("{path}|{filter}|{recursive}"))
    }
}

fn engine() -> Engine {
    let mut engine = Engine::new().unwrap();
    engine.install_host(TestHost).unwrap();
    engine
}

#[test]
fn evaluation_reenters_the_same_engine_and_globals() {
    let mut engine = engine();
    engine.compile("function outerFunction() { return 40; } function main() { let a = [42]; hostEvaluateSource(hostReadFile('outer'), 'outer'); return loaded() === a[0]; }", "main.js").unwrap();
    assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    assert!(engine.has_function("loaded"));
    assert_eq!(engine.arity("loaded"), Some(0));
    assert_eq!(
        engine.call("loaded", &[]).unwrap().value(),
        Value::number(42.0)
    );
}

#[test]
fn evaluation_during_initialization_and_callbacks_keeps_frames_alive() {
    let mut engine = engine();
    engine.compile("hostEvaluateSource('var x = 40;', 'first.js'); function main() { let a = [1, 2]; let b = a.map(v => { hostEvaluateSource('x += 1;', 'update.js'); return v + x; }); return b[0] === 42 && b[1] === 44; }", "main.js").unwrap();
    assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
}

#[test]
fn evaluated_js_exceptions_preserve_identity_and_propagate_to_the_caller() {
    let mut engine = engine();
    engine.compile("function main() { try { hostEvaluateSource('var payload = {x: 42}; function boom() { throw payload; } boom();', 'throw.js'); } catch (e) { return e === payload && e.x === 42; } return false; }", "main.js").unwrap();
    assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
}

#[test]
fn host_file_results_and_errors_use_runtime_values() {
    let mut engine = engine();
    engine.compile("function main() { let text = hostReadFile('text'); let list = hostListFiles('dir', '*.js', true); try { hostReadFile('missing'); } catch (e) { return text === 'é😀' && list === 'dir|*.js|true' && e.name === 'Error' && e.message === 'missing file: missing'; } return false; }", "main.js").unwrap();
    assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
}

#[test]
fn compilation_errors_during_evaluation_leave_the_engine_usable() {
    let mut engine = engine();
    engine.compile("function main() { hostEvaluateSource('class Unsupported {}', 'unsupported.js'); } function good() { return 42; }", "main.js").unwrap();
    assert!(matches!(
        engine.call("main", &[]),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(
        engine.call("good", &[]).unwrap().value(),
        Value::number(42.0)
    );
    engine
        .compile("function later() { return 43; }", "later.js")
        .unwrap();
    assert_eq!(
        engine.call("later", &[]).unwrap().value(),
        Value::number(43.0)
    );
}

#[test]
fn nested_source_evaluation_keeps_script_this_separate_from_method_this() {
    let mut engine = engine();
    engine.compile("function main() { let o = { f: function() { hostEvaluateSource('var scriptThis = this;', 'nested.js'); return this === o && scriptThis === globalThis; } }; return o.f(); }", "main.js").unwrap();
    assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
}
