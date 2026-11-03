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
fn math_guard_deduplicates_targets_and_rejects_writes() {
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--no-ir", "--eval", "function f(x) { return Math.floor(x) + Math.floor(x); } function g(o) { o.x = 1; return Math.floor(2); }"])
        .output().unwrap();
    assert!(output.status.success());
    let output = String::from_utf8(output.stdout).unwrap();
    assert!(output.contains("Math guard: 1 distinct targets, 2 specialized calls"));
    assert!(!output.contains("[g@1#1] Math guard"));
}

#[test]
fn receiver_property_guard_is_selected_and_writes_are_rejected() {
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--no-ir", "--eval", "function f() { return this.items[0] + this.items[1] + Math.floor(0); } function g() { this.items = []; return this.items[0] + this.items[1] + Math.floor(0); }"])
        .output().unwrap();
    assert!(output.status.success());
    let output = String::from_utf8(output.stdout).unwrap();
    assert!(output.contains("receiver property guard: items, 2 cached reads"));
    assert!(!output.contains("[g@1#1] receiver property guard"));
}
