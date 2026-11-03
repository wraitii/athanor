use js_engine::{Engine, EngineOptions, OptLevel, Value};

fn check(source: &str) {
    for opt_level in [OptLevel::None, OptLevel::Speed] {
        let mut engine = Engine::with_options(EngineOptions {
            opt_level,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "switch.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    }
}

#[test]
fn cases_use_strict_equality_ordered_tests_and_fallthrough() {
    check(
        r#"function main() {
        let n = 0; let result = '';
        switch ((n++, 'b')) {
        case (n++, 'a'): result += 'a'; break;
        default: result += 'd';
        case (n++, 'b'): result += 'b';
        case (n++, 'c'): result += 'c'; break;
        }
        switch (1) { case '1': return false; default: result += 'x'; }
        switch (0/0) { case 0/0: return false; }
        switch (-0) { case 0: result += 'z'; }
        switch (1) {}
        return n === 3 && result === 'bcxz';
    }"#,
    );
    check(
        r#"function main() {
        let s = '';
        switch (9) { case 1: s+='a'; default: s+='d'; case 2: s+='b'; }
        let a = {}; let b = {};
        switch (a) { case b: return false; case a: return s === 'db'; }
        return false;
    }"#,
    );
    check(
        r#"function main() {
        let saved = {x: 7}; let original = saved;
        switch (saved) {
        case (saved = null): return false;
        case original: return original.x === 7;
        }
        return false;
    }"#,
    );
}

#[test]
fn break_and_continue_resolve_switches_and_loops() {
    check(
        r#"function main() {
        let s = '';
        for (let i=0; i<4; i++) {
            switch (i) {
            case 0: continue;
            case 1: switch (2) { case 2: s+='n'; break; } break;
            case 2: for (let j=0; j<2; j++) { s+='l'; break; } break;
            default: s+='d';
            }
            s+='x';
        }
        return s === 'nxlxdx';
    }"#,
    );
}

#[test]
fn cases_share_lexical_scope_and_hoist_functions_and_vars() {
    check(
        r#"function main() {
        let x = 1; let f;
        switch (x) {
        case 1: let x = 7; f=()=>x;
        case 2: x++; var v = call(); break;
        default: function call() { return 9; }
        }
        return x === 1 && f() === 8 && v === 9;
    }"#,
    );
    check(
        r#"function main() {
        let f = [];
        for (let i=0; i<2; i++) {
            switch (i) { case 0: case 1: let x=i; f[i]=()=>x; break; }
        }
        return f[0]() === 0 && f[1]() === 1;
    }"#,
    );
    for source in [
        "function main(){ switch(1){ case x: break; default: let x=2; } }",
        "function main(){ switch(1){ case 1: return x; case 2: let x=2; } }",
    ] {
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "switch-tdz.js").unwrap();
        assert!(engine.call("main", &[]).is_err());
    }
}
