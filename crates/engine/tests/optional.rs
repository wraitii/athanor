use js_engine::{Engine, EngineOptions, OptLevel, Value};

fn check(source: &str) {
    for (opt_level, ssa_opt) in [OptLevel::None, OptLevel::Speed]
        .into_iter()
        .flat_map(|level| [false, true].map(|semantic| (level, semantic)))
    {
        let mut engine = Engine::with_options(EngineOptions {
            opt_level,
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "optional.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    }
}

#[test]
fn chains_short_circuit_keys_arguments_and_remaining_accesses() {
    check(
        r#"function main(){
        let n=0; let a=null; let f=undefined;
        let x=a?.[(n++, 'x')].y(n++);
        let y=f?.(n++, ...null);
        let z=a?.x?.().y;
        let b={x:null}; let w=b.x?.y;
        return n===0 && x===undefined && y===undefined && z===undefined && w===undefined;
    }"#,
    );
}

#[test]
fn optional_calls_preserve_receivers_and_evaluate_bases_once() {
    check(
        r#"function main(){
        let calls=0;
        let o={x:7, f:function(a){return this.x+a;}};
        function get(){calls++; return o;}
        let a=get()?.f(1);
        let b=get().f?.(2);
        let c=get()?.['f']?.(...[3]);
        let d=(get()?.f)(4);
        let e=(get()?.f)?.(5);
        let detached=function(){return this;};
        return calls===5 && a===8 && b===9 && c===10 && d===11 && e===12 &&
            detached?.()===undefined && o.missing?.()===undefined;
    }"#,
    );
}

#[test]
fn chain_boundaries_and_non_nullish_errors_are_preserved() {
    for source in [
        "function main(){ return missing?.x; }",
        "function main(){ let a=null; return (a?.x).y; }",
        "function main(){ let a=null; return (a?.f)(); }",
        "function main(){ return ({f:1}).f?.(); }",
        "function main(){ return false?.(); }",
        "function main(){ return (0)?.(); }",
        "function main(){ return ({x:null})?.x.y; }",
        "function main(){ return a?.x; let a; }",
    ] {
        for ssa_opt in [false, true] {
            let mut engine = Engine::with_options(EngineOptions {
                ssa_opt,
                ..Default::default()
            })
            .unwrap();
            engine.compile(source, "optional-errors.js").unwrap();
            assert!(engine.call("main", &[]).is_err(), "{source}");
        }
    }
    check(
        r#"function main(){
        let saved={x:9}; let o={f:function(a){return this.x+a;},x:7};
        let a=o?.f((o=null, 1));
        let b=saved?.x;
        return a===8 && b===9 && (null?.x ?? 4)===4;
    }"#,
    );
}

#[test]
fn spread_calls_snapshot_values_and_preserve_order() {
    check(
        r#"function main() {
        let a = [{x: 7}];
        let o = {x: 5, f: function(a,b) { return this.x + a.x + b; }};
        let saved = o;
        let result = o.f(...a, (a[0] = {x: 9}, o = null, 2));
        function C(a,b,c) { this.value = a*100+b*10+c; }
        return result === 14 && saved.x === 5 && new C(1,...[2,3]).value === 123;
    }"#,
    );
    for source in [
        "function f(){} function main(){ return f(...1); }",
        "function C(){} function main(){ return new C(...null); }",
    ] {
        for ssa_opt in [false, true] {
            let mut engine = Engine::with_options(EngineOptions {
                ssa_opt,
                ..Default::default()
            })
            .unwrap();
            engine.compile(source, "spread.js").unwrap();
            assert!(engine.call("main", &[]).is_err(), "{source}");
        }
    }
}

#[test]
fn remaining_expression_forms_preserve_results_and_order() {
    check(
        r#"function main() {
        let n = 0; let a = [{x:7}, 2];
        let b = [(n++, 0), ...a, (a[1] = 9), ...'é😀'];
        let o = {[(n++, 'key')]: (n++, 42)};
        let r = /ab+/gi;
        return typeof missing === 'undefined' && typeof Math === 'object' &&
            n === 3 && b.length === 6 && b[1].x === 7 && b[2] === 2 &&
            b[3] === 9 && b[4] === 'é' && b[5] === '😀' && o.key === 42 &&
            r.test('ABBB');
    }"#,
    );
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile("function main(){ return typeof x; let x = 1; }", "tdz.js")
            .unwrap();
        assert!(engine.call("main", &[]).is_err());
        for source in [
            "function main(){return [,...[]];}",
            "function main(){return /(?<name>a)/;}",
        ] {
            assert!(
                engine.compile(source, "unsupported.js").is_err(),
                "{source}"
            );
        }
        engine
            .compile("function fail(){return [...null];}", "spread-error.js")
            .unwrap();
        assert!(engine.call("fail", &[]).is_err());
    }
}
