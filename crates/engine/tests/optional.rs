use js_engine::{Engine, EngineOptions, OptLevel, Value};

fn check(source: &str) {
    for opt_level in [OptLevel::None, OptLevel::Speed] {
        let mut engine = Engine::with_options(EngineOptions {
            opt_level,
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
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "optional-errors.js").unwrap();
        assert!(engine.call("main", &[]).is_err(), "{source}");
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
