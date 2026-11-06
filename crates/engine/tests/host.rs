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
fn cached_bindings_observe_reentrant_declarations_and_keep_live_values() {
    let mut engine = engine();
    engine
        .compile(
            r#"
        var shared='old';
        function read(){return shared;}
        function kind(){return typeof later;}
        function main(){
            read();
            const saved=read();
            const before=kind();
            hostEvaluateSource("shared='new'; let later=7;", 'nested.js');
            return saved==='old' && read()==='new' && before==='undefined' && kind()==='number';
        }
    "#,
            "cached.js",
        )
        .unwrap();
    assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
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
    engine.compile("function main() { hostEvaluateSource('class Unsupported extends Base {}', 'unsupported.js'); } function good() { return 42; }", "main.js").unwrap();
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

#[test]
fn runtime_errors_preserve_innermost_source_location() {
    let mut first = engine();
    let inner = "function inner(){ return [1].missing; }";
    first.compile(inner, "inner.js").unwrap();
    first
        .compile("function main(){ return inner(); }", "outer.js")
        .unwrap();
    let error = first.call("main", &[]).err().unwrap().to_string();
    let start = inner.find("[1].missing").unwrap();
    assert!(
        error.contains(&format!(
            "inner.js:{start}..{}",
            start + "[1].missing".len()
        )),
        "{error}"
    );
    assert!(error.contains("property \"missing\" of array"), "{error}");

    let mut engine = engine();
    engine
        .compile(
            "function main(){ hostEvaluateSource('var q = /x/.missing;', 'loaded.js'); }",
            "caller.js",
        )
        .unwrap();
    let error = engine.call("main", &[]).err().unwrap().to_string();
    assert!(error.contains("loaded.js:8..19"), "{error}");
    assert!(error.contains("property \"missing\" of regexp"), "{error}");
}

#[test]
fn print_and_console_log_write_to_the_host() {
    use std::cell::RefCell;
    use std::rc::Rc;

    struct PrintHost(Rc<RefCell<String>>);

    impl Host for PrintHost {
        fn read_file(&self, path: &str) -> Result<String, String> {
            Err(path.to_string())
        }

        fn list_files(&self, _: &str, _: &str, _: bool) -> Result<String, String> {
            Ok(String::new())
        }

        fn print(&self, message: &str) {
            self.0.borrow_mut().push_str(message);
        }
    }

    let output = Rc::new(RefCell::new(String::new()));
    let mut engine = Engine::new().unwrap();
    engine.install_host(PrintHost(output.clone())).unwrap();
    engine
        .compile(
            "function main() { print('a', 1, true); console.log(); console.log('x', 2, null, [1, 'y']); }",
            "print.js",
        )
        .unwrap();
    assert_eq!(engine.call("main", &[]).unwrap().value(), Value::UNDEFINED);
    assert_eq!(*output.borrow(), "a1true\nx 2 null [1, y]\n");
}

#[test]
fn evaluated_sources_call_older_deferred_functions() {
    let mut engine = engine();
    engine
        .compile("function older(x) { return () => x + 2; }", "older.js")
        .unwrap();
    engine.compile("function main() { hostEvaluateSource('function newer() { return older(40)(); }', 'newer.js'); return newer(); }", "main.js").unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(42.0)
    );
}
