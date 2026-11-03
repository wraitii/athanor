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
fn indexed_read_scope_requires_dominance_and_closed_effects() {
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_SSA", "1")
        .env("JS_JIT_SSA_DUMP", "1")
        .args([
            "--compile",
            "--no-ir",
            "--eval",
            "function reuse(a, k, n) { let x = a[k]; for (let i = 0; i < n; ++i) x = a[k]; return x; } function write(a, k, n) { let x = a[k]; for (let i = 0; i < n; ++i) { a[k] = i; x = a[k]; } return x; } function call(a, k, n, f) { let x = a[k]; for (let i = 0; i < n; ++i) { f(); x = a[k]; } return x; }",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let output = String::from_utf8(output.stderr).unwrap();
    assert!(
        output
            .lines()
            .any(|line| line.contains("[reuse@") && line.contains("1 scoped indexed reads reused"))
    );
    for name in ["write", "call"] {
        assert!(
            output
                .lines()
                .any(|line| line.contains(&format!("[{name}@"))
                    && line.contains("0 scoped indexed reads reused"))
        );
    }
}

#[test]
fn indexed_scope_distinguishes_invocation_and_frame_lifetimes() {
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_SSA", "1")
        .env("JS_JIT_SSA_DUMP", "1")
        .args(["--compile", "--no-ir", "--eval", "function rooted(n) { if (!this.items[0]) return 0; let x = 0; for (let i = 0; i < Math.floor(n); ++i) x += this.items[0]; return x; } function unrooted(a, n) { let x = a[0]; for (let i = 0; i < n; ++i) x = a[0]; return x; }"])
        .output().unwrap();
    assert!(output.status.success());
    let output = String::from_utf8(output.stderr).unwrap();
    assert!(output.lines().any(|line| line.contains("[rooted@")
        && line.contains(".math-fast]")
        && line.contains("lifetime: Some(Invocation)")));
    assert!(
        output
            .lines()
            .any(|line| line.contains("[unrooted@") && line.contains("lifetime: Some(Frame)"))
    );
}

#[test]
fn semantic_bit_loop_has_a_call_free_numeric_body() {
    let source = "function count(start, end, value) { let result = value; let key = 0; if (this.data[key]) for (let i = start; i <= end; ++i) if (this.data[key] & (1 << i)) ++result; return result; } function local(data, start, end, value) { let result = value; let key = 0; if (data[key]) for (let i = start; i <= end; ++i) if (data[key] & (1 << i)) ++result; return result; }";
    for enabled in [false, true] {
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
            .args(["--compile", "--eval", source])
            .output()
            .unwrap();
        assert!(output.status.success());
        let output = String::from_utf8(output.stdout).unwrap();
        assert_eq!(
            output.contains("semantic loop (before CLIF optimization)"),
            enabled
        );
        if enabled {
            assert!(output.contains("induction: i = start..=end (inclusive, step 1)"));
            assert!(output.contains("fast-path guards: direct numeric index storage"));
            assert!(output.contains("fallback: original loop"));
        }
        for name in ["count", "local"] {
            let ir = output
                .split(&format!("[{name}@"))
                .find(|part| part.contains("] IR\n"))
                .unwrap();
            let fast = ir
                .split("block")
                .find(|block| block.contains(" = ishl ") && block.contains(" = fadd "));
            assert_eq!(fast.is_some(), enabled);
            if let Some(fast) = fast {
                assert!(!fast.contains(" = call "));
                assert!(fast.contains("brif"));
            }
        }
    }
}

#[test]
fn semantic_and_scoped_reads_share_the_storage_guard_only_when_rooted() {
    let source = r#"
        function rooted(start, end) {
            let count = 0; const key = 0;
            if (this.data[key])
                for (let i = start; i <= end; ++i)
                    if (this.data[key] & (1 << i)) ++count;
            return count + Math.floor(0);
        }
        function frame(start, end) {
            let count = 0; const key = 0;
            if (this.data[key])
                for (let i = start; i <= end; ++i)
                    if (this.data[key] & (1 << i)) ++count;
            return count;
        }
    "#;
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_SSA", "1")
        .env("JS_JIT_SSA_DUMP", "1")
        .args(["--compile", "--no-ir", "--eval", source])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.lines().any(|line| line.contains("[rooted@")
        && line.contains(".math-fast]")
        && line.contains("shared existing indexed storage guard")));
    assert!(
        stderr
            .lines()
            .any(|line| line.contains("[frame@") && line.contains("lifetime: Some(Frame)"))
    );
    assert!(
        !stderr.lines().any(|line| line.contains("[frame@")
            && line.contains("shared existing indexed storage guard"))
    );
    let fast = stdout
        .split(".math-fast] effects")
        .nth(1)
        .unwrap()
        .split("] effects")
        .next()
        .unwrap();
    assert!(fast.contains("js_index_probe: 1 site(s)"));
}
