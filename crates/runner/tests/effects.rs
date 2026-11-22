use std::process::Command;

#[test]
fn string_operand_feedback_fuses_concatenation_and_preserves_fallbacks() {
    let source =
        "function join(tag){return 'key:'+tag+':leaf'+'!';} function add(a,b){return a+b;}";
    let execute = r#"
        for (let i=0;i<32;++i)
            if (join('é😀') !== 'key:é😀:leaf!') throw 'string';
        for (let i=0;i<32;++i)
            if (join(12) !== 'key:12:leaf!') throw 'number';
        if (join(null) !== 'key:null:leaf!') throw 'null';
        if (join(true) !== 'key:true:leaf!') throw 'boolean';
        if (join('again') !== 'key:again:leaf!') throw 'string again';
        for (let i=0;i<32;++i)
            if (add(1,2) !== 3) throw 'numeric addition';
        if (add(1.5,2) !== 3.5) throw 'fractional addition';
        for (let i=0;i<32;++i)
            if (add('x',2) !== 'x2') throw 'mixed addition';
        if (add(1,2) !== 3) throw 'numeric again';
    "#;
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_BACKGROUND", "0")
        .env("JS_JIT_INLINE", "0")
        .args([
            "--compile",
            "--after",
            "string-concat",
            "--filter",
            "join",
            "--eval",
            source,
            "--execute",
            execute,
        ])
        .output()
        .unwrap();
    let dump = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "{}\n{dump}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(dump.contains("observe Read(Primitive)"), "{dump}");
    assert!(dump.contains("as_string"), "{dump}");
    assert!(dump.contains("string_concat ["), "{dump}");
    assert!(dump.contains("observe PrimitiveFailure"), "{dump}");
}

#[test]
fn string_read_feedback_fuses_entire_concatenation_chains() {
    for (parameter, operand, argument) in [("o", "o.tag", "{tag:tag}"), ("a", "a[0]", "[tag]")] {
        let source = format!(
            "function join(tag){{return concatenate({argument});}} function concatenate({parameter}){{return 'key:'+{operand}+':leaf'+'!';}}"
        );
        let execute = r#"
            for (let i=0;i<32;++i)
                if (join('é😀') !== 'key:é😀:leaf!') throw 'string';
            for (let i=0;i<32;++i)
                if (join(12) !== 'key:12:leaf!') throw 'number';
            if (join(null) !== 'key:null:leaf!') throw 'null';
            if (join(true) !== 'key:true:leaf!') throw 'boolean';
            if (join('again') !== 'key:again:leaf!') throw 'string again';
        "#;
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_BACKGROUND", "0")
            .env("JS_JIT_INLINE", "0")
            .args([
                "--compile",
                "--after",
                "string-concat",
                "--filter",
                "concatenate",
                "--eval",
                &source,
                "--execute",
                execute,
            ])
            .output()
            .unwrap();
        let dump = String::from_utf8(output.stdout).unwrap();
        assert!(
            output.status.success(),
            "{operand}: {}\n{dump}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            dump.contains("observe Read(Primitive)"),
            "{operand}: {dump}"
        );
        assert!(dump.contains("as_string"), "{operand}: {dump}");
        assert!(
            dump.lines().any(|line| line
                .split_once("string_concat [")
                .and_then(|(_, parts)| parts.split_once(']'))
                .is_some_and(|(parts, _)| parts.split(',').count() == 4)),
            "{operand}: {dump}"
        );
        assert!(
            dump.contains("observe PrimitiveFailure"),
            "{operand}: {dump}"
        );
    }
}

#[test]
fn numeric_guard_failures_widen_feedback_and_then_stop_speculating() {
    let source =
        "function accumulate(a){let sum=0;for(let i=0;i<10&&!(sum>1e9);++i)sum+=a[i];return sum;}";
    let execute = r#"
        const runner = {call: accumulate};
        const values = Array(10).fill(5);
        for (let i = 0; i < 32; ++i)
            if (runner.call(values) !== 50) throw 'integer';
        values[5] = 5.2;
        for (let i = 0; i < 32; ++i)
            if (runner.call(values) !== 50.2) throw 'fractional';
        values[5] = 'x';
        for (let i = 0; i < 32; ++i)
            if (runner.call(values) !== '25x5555') throw 'string';
        values[5] = 5.2;
        for (let i = 0; i < 32; ++i)
            if (runner.call(values) !== 50.2) throw 'numeric again';
    "#;
    {
        let mut command = Command::new(env!("CARGO_BIN_EXE_js-runner"));
        command.env("JS_JIT_SPECIALIZATION_LOG", "accumulate");
        let output = command
            .args([
                "--compile",
                "--after",
                "planner",
                "--filter",
                "accumulate",
                "--eval",
                source,
                "--execute",
                execute,
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let log = String::from_utf8(output.stderr).unwrap();
        assert!(
            log.lines()
                .any(|line| line.contains("~ feedback Primitive ")
                    && line.ends_with("Integer -> Number")),
            "{log}"
        );
        assert!(
            log.lines()
                .any(|line| line.contains("~ feedback Primitive ")
                    && line.ends_with("Number -> Unknown")),
            "{log}"
        );
        assert!(
            log.lines()
                .filter(|line| line.starts_with("[js jit specialization]"))
                .all(|line| line.contains("function=accumulate")),
            "{log}"
        );
        let dump = String::from_utf8(output.stdout).unwrap();
        let plans: Vec<_> = dump
            .split("planner ")
            .skip(1)
            .filter(|part| part.starts_with("accumulate"))
            .collect();
        let numeric: Vec<_> = plans
            .iter()
            .filter(|plan| plan.contains("Primitive -> "))
            .collect();
        assert_eq!(numeric.len(), 3, "{dump}");
        assert!(numeric[0].contains("Primitive -> Integer"), "{dump}");
        assert!(numeric[1].contains("Primitive -> Number"), "{dump}");
        assert!(numeric[2].contains("Primitive -> Unknown"), "{dump}");
        let graph = dump
            .rsplit_once("semantic IR ")
            .unwrap()
            .1
            .split("planner ")
            .next()
            .unwrap();
        assert!(!graph.contains("observe PrimitiveFailure"), "{graph}");
        assert!(graph.contains(":number|string"), "{graph}");
    }
}

#[test]
fn integer_failure_replans_other_numeric_sites_as_numbers() {
    let source = "function accumulate(a,b,c){let sum=0;for(let i=0;i<10&&!(sum>1e9);++i){sum+=a[i];sum+=b[i];sum+=c[i];}return sum;}";
    let execute = r#"
        const runner = {call: accumulate};
        const a = Array(10).fill(1);
        const b = Array(10).fill(2);
        const c = Array(10).fill(3);
        for (let i = 0; i < 32; ++i)
            if (runner.call(a,b,c) !== 60) throw 'integer';
        a.fill(1.5);
        b.fill(2.5);
        c.fill(3.5);
        for (let i = 0; i < 32; ++i)
            if (runner.call(a,b,c) !== 75) throw 'fractional';
        a.fill(1);
        b.fill(2);
        c.fill(3);
        for (let i = 0; i < 32; ++i)
            if (runner.call(a,b,c) !== 60) throw 'integer again';
    "#;
    {
        let mut command = Command::new(env!("CARGO_BIN_EXE_js-runner"));
        command.env("JS_JIT_SPECIALIZATION_LOG", "accumulate");
        let output = command
            .args([
                "--compile",
                "--after",
                "ownership",
                "--after",
                "planner",
                "--filter",
                "accumulate",
                "--eval",
                source,
                "--execute",
                execute,
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let log = String::from_utf8(output.stderr).unwrap();
        assert!(
            log.contains("+ policy numeric speculation = number-first"),
            "{log}"
        );
        assert_eq!(
            log.lines()
                .filter(|line| line.starts_with("[js jit specialization]")
                    && line.ends_with("action=compile-specialized"))
                .count(),
            2,
            "{log}"
        );
        let dump = String::from_utf8(output.stdout).unwrap();
        let final_graph = dump
            .rsplit_once("semantic IR ")
            .unwrap()
            .1
            .split("planner ")
            .next()
            .unwrap();
        assert!(!final_graph.contains("is_integer"), "{final_graph}");
        assert!(final_graph.contains("is_number"), "{final_graph}");
        assert!(
            final_graph.contains(":int32 = low_level Binary { op: Add,"),
            "{final_graph}"
        );
    }
}

#[test]
fn property_probes_report_shapes_and_specialize_region_entry() {
    let source = "function f(){return this.x*this.y;}";
    let execute = "let o={x:3,y:4,f:f};
        for(let i=0;i<6;i++) if(o.f()!==12) throw 'initial shape';
        let changed={y:5,x:6,f:f};
        if(changed.f()!==30) throw 'changed shape';
        function Child(){this.f=f;}
        Child.prototype={x:7,y:8};
        if(new Child().f()!==56) throw 'prototype fallback';
        o.x=2.5;
        if(o.f()!==10) throw 'changed value';";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args([
            "--compile",
            "--filter",
            "f",
            "--execute",
            execute,
            "--eval",
            source,
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let (initial, feedback) = dump.split_once(".feedback").unwrap();
    assert!(initial.contains("helper_call SemanticPropertyProbeCached("));
    let feedback = feedback.split("\n[").next().unwrap();
    assert!(!feedback.contains("helper_call SemanticPropertyProbeCached("));
    assert!(feedback.contains("low_level Load"));
}

#[test]
fn semantic_graph_versions_math_calls_and_preserves_fallbacks() {
    let source = "function f(a,n){ let x=0; for(let i=0;i<n;i++) x+=Math.floor(a[i]); return x; } function main(){ let a=f([1.2,2.8],2); Math.floor=x=>x+1; return a*10+f([1,2],2); }";
    let dump = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--after", "planner", "--eval", source])
        .output()
        .unwrap();
    assert!(dump.status.success());
    let dump = String::from_utf8(dump.stdout).unwrap();
    assert!(dump.contains("global_cache Math"));
    assert!(dump.contains("helper_call SemanticGlobalProbe("));
    assert!(dump.contains("call_native Math.floor"));
    assert!(dump.contains("combined entry guards:"));
    let disabled = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_OPT", "none")
        .args(["--compile", "--after", "planner", "--eval", source])
        .output()
        .unwrap();
    assert!(disabled.status.success());
    assert!(
        !String::from_utf8(disabled.stdout)
            .unwrap()
            .contains("helper_call SemanticGlobalProbe(")
    );
    for enabled in [false, true] {
        let result = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_OPT", if enabled { "full" } else { "none" })
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(result.status.success());
        assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), "35");
    }
}

#[test]
fn semantic_index_reads_observe_writes_and_calls() {
    let source = "function main(){ let a=[1]; let sum=0; for(let i=0;i<3;i++){ sum+=a[0]; a[0]=i+2; } let f=()=>a[0]=9; for(let i=0;i<2;i++){ sum+=a[0]; f(); } return sum; }";
    for enabled in [false, true] {
        let result = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_OPT", if enabled { "full" } else { "none" })
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(result.status.success());
        assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), "19");
    }
}

#[test]
fn private_iteration_does_not_block_versioning() {
    let source = "function f(p,xs){ const [a,b]=[p.x,p.y]; let s=0; for(const v of xs) s+=Math.floor(v+a+b); return s; } function main(){ const xs=[0.5,1.5]; const r=f({x:1,y:2},xs); Math.floor=x=>x*10; return r+','+f({x:0,y:0},xs); }";
    let dump = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--after", "planner", "--eval", source])
        .output()
        .unwrap();
    assert!(dump.status.success());
    let dump = String::from_utf8(dump.stdout).unwrap();
    let f = dump.split("semantic IR f").nth(1).unwrap();
    assert!(
        f.split("semantic IR")
            .next()
            .unwrap()
            .contains("combined entry guards:")
    );
    for enabled in [false, true] {
        let result = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_OPT", if enabled { "full" } else { "none" })
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(result.status.success());
        assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), "7,20");
    }
}

#[test]
fn loops_version_independently_of_their_function() {
    let source = "function f(g,xs){ const t=g(); let s=0; let last=-1; for(let i=0;i<xs.length;++i){ last=Math.floor(xs[i]); s+=last; } return s+':'+last+':'+t; } function main(){ const xs=[0.5,2.5]; const r=f(()=>1,xs); Math.floor=x=>x*10; return r+','+f(()=>2,xs); }";
    let dump = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_OPT", "full")
        .args(["--compile", "--after", "planner", "--eval", source])
        .output()
        .unwrap();
    assert!(dump.status.success());
    let dump = String::from_utf8(dump.stdout).unwrap();
    let f = dump.split("semantic IR f").nth(1).unwrap();
    assert!(
        f.split("semantic IR")
            .next()
            .unwrap()
            .contains(": combined entry guards:")
    );
    for enabled in [false, true] {
        let result = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_OPT", if enabled { "full" } else { "none" })
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(result.status.success());
        assert_eq!(
            String::from_utf8(result.stdout).unwrap().trim(),
            "2:2:1,30:25:2"
        );
    }
}

#[test]
fn array_iteration_exposes_and_inlines_known_closures() {
    for (method, callback_args) in [
        ("map", "t=>createTerrain(t)"),
        ("some", "t=>createTerrain(t)"),
        ("every", "t=>createTerrain(t)"),
        ("filter", "t=>createTerrain(t)"),
        ("reduce", "(acc,t)=>acc + createTerrain(t), ''"),
    ] {
        let source = format!(
            "function createTerrain(terrain){{ return typeof terrain === 'string' ? terrain : terrain.{method}({callback_args}); }}"
        );
        for enabled in [false, true] {
            let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
                .env("JS_JIT_OPT", if enabled { "full" } else { "none" })
                .args(["--compile", "--after", "planner", "--eval", &source])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let dump = String::from_utf8(output.stdout).unwrap();
            let caller = dump
                .split("semantic IR createTerrain\n")
                .nth(1)
                .unwrap()
                .split("semantic IR")
                .next()
                .unwrap();
            assert_eq!(caller.contains("helper_call ArrayIterationGuard("), enabled);
            assert_eq!(caller.contains("helper_call ArrayMapElement("), enabled);
            if enabled {
                assert!(caller.contains("inline") && caller.contains("0 code check(s)"));
                assert!(caller.contains("materialized only in fallback"));
                assert!(
                    caller.find("helper_call ArrayIterationGuard(").unwrap()
                        < caller.find(" = closure_cells ").unwrap()
                );
            }
        }
    }
}

#[test]
fn inlined_local_calls_remove_unused_closure_allocations() {
    let source = "function main(){ const f=x=>x*x; return f(3); }";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_OPT", "full")
        .args(["--compile", "--after", "planner", "--eval", source])
        .output()
        .unwrap();
    assert!(output.status.success());
    let dump = String::from_utf8(output.stdout).unwrap();
    let caller = dump
        .split("semantic IR main\n")
        .nth(1)
        .unwrap()
        .split("semantic IR")
        .next()
        .unwrap();
    assert!(!caller.contains(" = closure_cells "));
    assert!(caller.contains("unused closure"));
}

#[test]
fn construction_exposes_prototype_resolution() {
    let source = "function make(C,p){return new C((C.prototype=p,1));} function spread(C,args){return new C(...args);}";
    for enabled in [false, true] {
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_OPT", if enabled { "full" } else { "none" })
            .args(["--compile", "--after", "planner", "--eval", source])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let dump = String::from_utf8(output.stdout).unwrap();
        assert_eq!(dump.contains("helper_call PropertyGetCached("), enabled);
        assert_eq!(dump.contains("helper_call ConstructResolved("), enabled);
        assert_eq!(dump.contains("helper_call ConstructPackResolved("), enabled);
        assert_eq!(dump.contains("helper_call NewInstance"), enabled);
        assert_eq!(dump.contains("helper_call ConstructorCall("), enabled);
        assert_eq!(dump.contains("helper_call ConstructorCallPack("), enabled);
        if enabled {
            assert!(
                dump.find("helper_call PropertySetCached").unwrap()
                    < dump.find("helper_call PropertyGetCached(").unwrap()
            );
        }
    }
}

#[test]
fn known_constructor_bodies_can_be_inlined() {
    let source = "function C(x){this.x=x;} function main(){C.prototype={};return new C(7).x;}";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--after", "planner", "--eval", source])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let main = dump.split("semantic IR main\n").nth(1).unwrap();
    assert!(main.contains("inline"));
    assert!(main.contains("helper_call NewInstance"));
}

#[test]
fn array_iteration_preserves_mutation_short_circuiting_and_fallbacks() {
    let source = r#"
function main() {
    let a = [1, 2, 3], seen = '';
    let some = a.some((v, i, xs) => { seen += v; xs.push(9); return i == 1 ? 'yes' : ''; });
    let b = [1, 2, 3];
    let every = b.every((v, i, xs) => { seen += v; xs[2] = 8; return i == 1 ? 0 : 1; });
    let c = [1, 2, 3, 4];
    let filtered = c.filter((v, i, xs) => { if (i == 0) { xs.length = 3; xs.push(7); xs.push(9); } return v % 2; });
    let sparse = new Array(4); sparse[2] = 5;
    let holes = sparse.filter(v => { seen += v; return v; });
    let emptySome = [].some(v => { seen += 'bad'; return true; });
    let emptyEvery = [].every(v => { seen += 'bad'; return false; });
    let custom = { some: f => f(9) };
    let fallback = custom.some(v => v + 1);
    let threshold = 2;
    let captured = [1, 3].some(v => v > threshold);
    return some + ':' + every + ':' + seen + ':' + filtered.join(',') + ':' + holes.join(',') + ':' + emptySome + ':' + emptyEvery + ':' + fallback + ':' + captured;
}
"#;
    for enabled in [false, true] {
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_OPT", if enabled { "full" } else { "none" })
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap().trim(),
            "true:false:12125:1,3,7:5:false:true:10:true"
        );
    }
}

#[test]
fn array_reduce_preserves_accumulator_and_observes_mutations() {
    let source = r#"
function main() {
    const values = new Array(5);
    values[1] = {n: 3};
    values[3] = {n: 5};
    const initial = {n: 1};
    const result = values.reduce((acc, item, index, array) => {
        if (index === 1) {
            delete array[1];
            array[2] = {n: 6};
            delete array[3];
            array.push({n: 100});
            return {n: acc.n + item.n};
        }
        acc.n += item.n + index;
        return acc;
    }, initial);
    const empty = new Array(2);
    let caught = false;
    try { empty.reduce((a, b) => a + b); } catch (e) { caught = e.name === 'TypeError'; }
    const custom = {reduce: (callback, seed) => callback(seed, 4)};
    const fallback = custom.reduce((a, b) => a + b, 3);
    const missing = [1].reduce((a, b) => a === undefined ? b : 0, undefined);
    Array.prototype[1] = 2;
    const inherited = new Array(3).reduce((a, b) => a + b, 5);
    delete Array.prototype[1];
    return result.n + ':' + initial.n + ':' + values.reduce((a, b) => a + b.n, 0)
        + ':' + empty.reduce(() => 0, initial).n + ':' + [2, 3].reduce((a, b) => a * b)
        + ':' + caught + ':' + fallback + ':' + missing + ':' + inherited;
}
"#;
    for opt in ["full", "none"] {
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_OPT", opt)
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap().trim(),
            "12:1:106:1:6:true:7:1:7"
        );
    }
}

#[test]
fn inlined_captures_materialize_cells_only_for_fallbacks() {
    let source = "function main(xs){ let limit=2; return xs.some(x=>x>limit); }";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_OPT", "full")
        .args([
            "--compile",
            "--after",
            "planner",
            "--after",
            "input",
            "--eval",
            source,
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let caller = dump
        .split("semantic IR main after input\n")
        .nth(1)
        .unwrap()
        .split("semantic IR")
        .next()
        .unwrap();
    assert!(!caller.contains("enter_captures"));
    assert!(caller.contains("cell_get"));
    assert!(!caller.contains("read limit#"));
    assert!(caller.contains("materialized only in fallback"));
    assert!(caller.contains("materialized as a cell only on closure creation"));
}

#[test]
fn lazy_captures_share_writes_and_preserve_escaped_iteration_environments() {
    let source = r#"
let saved;
function run(xs) {
    let n = 1;
    let result = xs.some(v => { n += v; return n > 3; });
    n += 10;
    return result + ':' + n;
}
function main() {
    let fast = run([1, 2, 3]);
    let slow = run({ some: f => { saved = f; return f(1); } });
    let escaped = saved(1);
    let callbacks = [];
    let custom = { some: f => { callbacks.push(f); return false; } };
    for (let i = 0; i < 3; ++i) custom.some(v => i + v);
    let iterations = callbacks[0](10) + ',' + callbacks[1](10) + ',' + callbacks[2](10);
    let snapshots = [];
    for (let i = 0, f = () => i; i < 3; ++i) snapshots.push(f());
    let value = 1;
    let other = () => value;
    [2].every(v => { value += v; return true; });
    return fast + ':' + slow + ':' + escaped + ':' + iterations + ':' + snapshots.join(',') + ':' + other();
}
"#;
    for enabled in [false, true] {
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_OPT", if enabled { "full" } else { "none" })
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap().trim(),
            "true:14:false:12:true:10,11,12:0,0,0:3"
        );
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_OPT", if enabled { "full" } else { "none" })
            .args(["--eval", "function main(){ [1].some(v=>v+n); let n=2; }"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("before initialization"));
    }
}

#[test]
fn whole_function_guards_expose_closed_inlined_arithmetic() {
    let source = "function square(x){return x*x;} function distance(x1,y1,x2,y2){return square(x2-x1)+square(y2-y1);}";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_OPT", "full")
        .args([
            "--compile",
            "--no-ir",
            "--after",
            "ownership",
            "--after",
            "planner",
            "--eval",
            source,
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let function = dump
        .split("semantic IR distance after ownership\n")
        .nth(1)
        .unwrap()
        .split("semantic IR")
        .next()
        .unwrap();
    assert!(function.contains("function: combined entry guards:"));
    let fast = function.split("\n  return ").next().unwrap();
    assert_eq!(fast.matches("global_cache square").count(), 1);
    assert_eq!(fast.matches(" = is_code ").count(), 1);
    assert_eq!(fast.matches(" = is_number ").count(), 4);
    assert!(!fast.contains(" = global "));
    assert!(!fast.contains(" = call "));
    assert_eq!(fast.matches("op: FloatMultiply").count(), 2);
    assert!(!function.contains("not optimized:"));
}

#[test]
fn whole_function_guard_fallbacks_preserve_conversion_and_method_replacement() {
    let source = r#"
function square(x) { return x * x; }
function distance(x1,y1,x2,y2) { return square(x2-x1) + square(y2-y1); }
function replacement(x) { square = y => 100 + y; return 10 + x; }
function mixed(a,b) { return square(a) + Math.floor(b); }
function replaceFloor(x) { Math.floor = y => 100 + y; return 10 + x; }
function main() {
    let first = distance(1,2,4,6);
    let converted = distance('1',null,4,5);
    square = replacement;
    let replaced = distance(1,2,4,6);
    square = replaceFloor;
    return first + ':' + converted + ':' + replaced + ':' + mixed(2,3);
}
"#;
    for enabled in [false, true] {
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_OPT", if enabled { "full" } else { "none" })
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap().trim(),
            "25:34:117:115"
        );
    }
}

#[test]
fn feedback_inlining_closes_nested_math_property_calls() {
    let source = r#"
Math.square = function(x) { return x*x; };
Math.euclidDistance2DSquared = function(x1,y1,x2,y2) {
    return Math.square(x2-x1) + Math.square(y2-y1);
};
function distance(a,b) {
    return Math.euclidDistance2DSquared(a.x,a.y,b.x,b.y);
}
function main() {
    let a={x:1,y:2}, b={x:4,y:6}, sum=0;
    try { for(let i=0;i<200;++i) sum+=distance(a,b); } catch(e) { throw e; }
    Math.square = x => x+1;
    return sum + ':' + distance(a,b);
}
"#;
    for enabled in [false, true] {
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_OPT", if enabled { "full" } else { "none" })
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8(output.stdout).unwrap().trim(), "5000:9");
    }
}

#[test]
fn spread_constructor_has_guarded_inline_and_generic_fallback() {
    let source =
        "function C(a,b=4){this.x=a+b;} function main(){C.prototype={};return new C(...[3,5]).x;}";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args([
            "--compile",
            "--no-ir",
            "--after",
            "ownership",
            "--after",
            "planner",
            "--eval",
            source,
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let caller = dump
        .split("semantic IR main after ownership\n")
        .nth(1)
        .unwrap();
    assert!(caller.contains("helper_call ConstructorCallPack("));
    assert!(caller.contains("inline funcid"));
    assert!(caller.contains("is_code"));
}

#[test]
fn recursively_inlined_bodies_expand_arrays_and_constructors_once() {
    let source = "function SimpleTerrain(a){this.x=a;} function createTerrain(t){return typeof t === 'string' ? new SimpleTerrain(...[t]) : t.map(x=>createTerrain(x));} function main(t){return createTerrain(t);}";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--after", "planner", "--eval", source])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let caller = dump
        .split("semantic IR createTerrain\n")
        .nth(1)
        .unwrap()
        .split("semantic IR")
        .next()
        .unwrap();
    let maps = caller
        .matches(" = helper_call ArrayIterationGuard(")
        .count();
    let constructors = caller
        .matches(" = helper_call ConstructPackResolved(")
        .count();
    assert!((2..=4).contains(&maps), "{maps}");
    assert!((2..=4).contains(&constructors), "{constructors}");
    assert!(!caller.contains(" = helper_call ConstructPack("));
    let start = source.find("new SimpleTerrain").unwrap();
    assert!(caller.lines().any(|line| {
        line.contains("note: call inst")
            && line.contains(&format!(":{start}.."))
            && line.contains("inline funcid")
    }));
    assert_eq!(caller.matches("note: array map at ").count(), maps);
}

#[test]
fn split_and_spread_inline_have_guarded_semantic_operations() {
    let source = "function C(a,b){this.a=a;this.b=b;} function main(s){C.prototype={};return new C(...s.split('|')).a;}";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--after", "planner", "--eval", source])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let caller = dump.split("semantic IR main\n").nth(1).unwrap();
    assert!(caller.contains("helper_call IsStringSplit("));
    assert!(caller.contains(" = helper_call StringSplit"));
    assert!(caller.contains(" = spread "));
    assert!(!caller.contains("is_indexed"));
    assert_eq!(caller.matches(" = helper_call PackTake").count(), 2);
    assert!(caller.contains("helper_call ConstructorCallPack("));
    assert!(caller.contains("reenters, throws"));
}

#[test]
fn string_split_guard_can_select_a_closed_version() {
    let source = "function f(s,n){let x=0;for(let i=0;i<n;i++){x+=s.split(',').length;}return x;} function main(){return f('a,b',3);}";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--after", "planner", "--eval", source])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let caller = dump
        .split("semantic IR f\n")
        .nth(1)
        .unwrap()
        .split("semantic IR")
        .next()
        .unwrap();
    assert!(caller.contains("probe") && caller.contains("helper_call IsStringSplit("));
    assert!(caller.contains("hoisted guards"));
}
