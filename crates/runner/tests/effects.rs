use std::process::Command;

#[test]
fn compile_reports_closed_and_unresolved_calls() {
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args([
            "--compile",
            "--no-ir",
            "--eval",
            "function closed(x) { return x + 1; } function open(f) { return f(); } function unknown(o) { o.x = 1; }",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let output = String::from_utf8(output.stdout).unwrap();
    let closed = output
        .split("[closed@")
        .nth(1)
        .unwrap()
        .split("[open@")
        .next()
        .unwrap();
    assert!(closed.contains("admission: closed against reentry"));
    assert!(closed.contains("js_local_set"));
    let open = output
        .split("[open@")
        .nth(1)
        .unwrap()
        .split("[unknown@")
        .next()
        .unwrap();
    assert!(open.contains("admission: unresolved"));
    assert!(open.contains("js_call: 1 site(s), unresolved: arbitrary call"));
    assert!(open.contains("@line 1"));
    let unknown = output.split("[unknown@").nth(1).unwrap();
    assert!(unknown.contains("admission: unresolved"));
    assert!(unknown.contains("unresolved: unclassified helper"));
}

#[test]
fn compile_recognizes_math_candidates_without_proving_identity() {
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_SSA", "0")
        .args([
            "--compile",
            "--no-ir",
            "--eval",
            "function f(x) { return Math.floor(x); } function g(Math) { return Math.floor(1); }",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let output = String::from_utf8(output.stdout).unwrap();
    let global = output
        .split("[f@")
        .nth(1)
        .unwrap()
        .split("[g@")
        .next()
        .unwrap();
    assert!(global.contains("candidate call targets (identity unproven)"));
    assert!(global.contains("Math.floor"));
    assert!(global.contains("admission: unresolved"));
    let local = output.split("[g@").nth(1).unwrap();
    assert!(!local.contains("candidate call targets"));
}

#[test]
fn constant_branch_cleanup_is_gated_by_ssa_optimization() {
    for enabled in [false, true] {
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
            .env("JS_JIT_SSA_DUMP", "1")
            .args([
                "--compile",
                "--no-ir",
                "--eval",
                "function f() { let x = 0; return x + 1; }",
            ])
            .output()
            .unwrap();
        assert!(output.status.success());
        let output = String::from_utf8(output.stderr).unwrap();
        let simplified = output
            .lines()
            .find(|line| line.contains("[f@") && line.contains("constant branches"));
        if enabled {
            let line = simplified.unwrap();
            let count: usize = line
                .split("simplified ")
                .nth(1)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .parse()
                .unwrap();
            assert!(count > 0);
        } else {
            assert!(simplified.is_none());
        }
    }
}

#[test]
fn semantic_graph_versions_math_calls_and_preserves_fallbacks() {
    let source = "function f(a,n){ let x=0; for(let i=0;i<n;i++) x+=Math.floor(a[i]); return x; } function main(){ let a=f([1.2,2.8],2); Math.floor=x=>x+1; return a*10+f([1,2],2); }";
    let dump = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
        .output()
        .unwrap();
    assert!(dump.status.success());
    let dump = String::from_utf8(dump.stdout).unwrap();
    assert!(dump.contains("probe_global Math"));
    assert!(dump.contains("call_native Math.floor"));
    assert!(dump.contains("note: versioned:"));
    let disabled = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_SSA", "0")
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
        .output()
        .unwrap();
    assert!(disabled.status.success());
    assert!(
        !String::from_utf8(disabled.stdout)
            .unwrap()
            .contains("probe_global Math")
    );
    for enabled in [false, true] {
        let result = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
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
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(result.status.success());
        assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), "19");
    }
}
