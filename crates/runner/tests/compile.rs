use std::process::{Command, Output};

fn compile(arguments: &[&str], source: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_SSA", "1")
        .env("JS_JIT_OSR", "0")
        .env_remove("JS_JIT_JITDUMP")
        .args(["--compile"])
        .args(arguments)
        .args(["--eval", source])
        .output()
        .unwrap()
}

#[test]
fn after_views_span_specialization_and_lowering() {
    let source = "function helper(x){return x*2;} function selected(a,n){let sum=0;for(let i=0;i<n;++i)sum+=helper(a[i]);return sum;}";
    for pass in [
        "inline",
        "captures",
        "versioning",
        "osr",
        "input",
        "property-specialization",
        "indexed-accesses",
        "normalise",
        "load-reuse",
        "indexed-checks",
        "property-caches",
        "js-cleanup",
        "numeric-specialization",
        "js",
        "typed-storage",
        "ownership",
        "boxing",
        "scalar-cleanup",
    ] {
        let output = compile(&["--filter", "selected", "--after", pass], source);
        assert!(
            output.status.success(),
            "{pass}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(
            text.contains(&format!("semantic IR selected after {pass}\n")),
            "{text}"
        );
    }
    let output = compile(&["--after", "unknown"], source);
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains(
            "unknown pass unknown; available passes: inline, captures, versioning, osr, input"
        ),
        "{error}"
    );
    assert!(error.contains("scalar-cleanup"), "{error}");
}

#[test]
#[cfg(target_arch = "aarch64")]
fn native_selection_view_preserves_clif_inspection() {
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_BACKEND", "native-only")
        .env("JS_JIT_SSA", "1")
        .env("JS_JIT_OSR", "0")
        .env_remove("JS_JIT_JITDUMP")
        .args([
            "--compile",
            "--filter",
            "f",
            "--after",
            "aarch64-select",
            "--clif",
            "--asm",
            "--eval",
            "function f(a,b){return a*b+1;}",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("semantic IR f after aarch64-select"));
    assert!(text.contains("aarch64-branch"));
    assert!(text.contains("Cranelift IR before optimization"));
    assert!(text.contains("machine code ("));
}

#[test]
fn filter_selects_all_views_without_removing_inline_candidates() {
    let output = compile(
        &[
            "--filter",
            "selected",
            "--ast",
            "--clif",
            "--planner",
            "--asm",
        ],
        "function helper(x){return x+1;} function selected(x){return helper(x);} function unrelated(){return 42;}",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    for heading in [
        "analyzed AST selected",
        "semantic IR selected",
        "planner selected",
        "Cranelift IR before optimization",
        "Cranelift IR after optimization",
        "machine code (",
    ] {
        assert!(text.contains(heading), "{heading}\n{text}");
    }
    assert!(!text.contains("[helper@"));
    assert!(!text.contains("[unrelated@"));
    let semantic = text
        .split("semantic IR selected\n")
        .nth(1)
        .unwrap()
        .split("[selected@")
        .next()
        .unwrap();
    assert!(!semantic.contains("note:"));
    assert!(text.contains("note: call inst"));
    assert!(text.contains("inline funcid"));
    assert!(text.contains(" ; @"));
    assert!(text.contains(" ; unattributed"));
    let locations: Vec<_> = semantic
        .lines()
        .filter_map(|line| line.rsplit_once("  @").map(|(_, location)| location))
        .collect();
    let asm = text.split("machine code (").nth(1).unwrap();
    for line in asm.lines() {
        if let Some((_, location)) = line.split_once(" ; @") {
            assert!(locations.contains(&location), "{line}");
        }
    }
}

#[test]
fn filter_reports_no_matches_and_requires_compile_mode() {
    let output = compile(&["--filter", "missing"], "function f(){return 1;}");
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no functions match filter `missing`")
    );
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--filter", "f", "--eval", "function f(){return 1;}"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("require --compile"));
}

#[test]
fn semantic_dump_includes_emitted_osr_instrumentation() {
    for arguments in [vec![], vec!["--after", "osr"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_SSA", "1")
            .env("JS_JIT_OSR", "1")
            .env_remove("JS_JIT_JITDUMP")
            .args(arguments)
            .args([
                "--compile",
                "--no-ir",
                "--filter",
                "f",
                "--semantic-ir",
                "--eval",
                "function f(rows,n){let total=0;for(let y=0;y<n;++y){let sum=0;for(let x=0;x<n;++x)sum+=rows[y][x]/1;total+=sum;}return total;}",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.contains("osr_start"), "{text}");
        assert!(!text.contains("note:"));
    }
}

#[test]
fn planner_can_be_printed_without_ir_or_execution() {
    let output = compile(
        &["--no-ir", "--planner"],
        "function f(a){return a+1;} print('executed');",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("planner f"));
    assert!(text.contains("feedback: 0"));
    assert!(text.contains("inline budget:"));
    assert!(!text.contains("semantic IR"));
    assert!(!text.contains("Cranelift IR"));
    assert!(!text.contains("machine code"));
    assert!(!text.lines().any(|line| line == "executed"));
}

#[test]
fn semantic_ir_is_default_and_clif_is_explicit() {
    let source = "function f(a){return a+1;}";
    let output = compile(&[], source);
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("semantic IR f"));
    assert!(!text.contains("Cranelift IR"));
    assert!(!text.contains("planner f"));
    let output = compile(&["--no-semantic-ir", "--clif"], source);
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(!text.contains("semantic IR"));
    assert!(text.contains("Cranelift IR before optimization"));
    assert!(text.contains("Cranelift IR after optimization"));
}

#[test]
fn planner_reports_materialized_region_candidates() {
    let output = compile(
        &["--planner"],
        "function f(rows){let sum=0;for(let y=0;y<2;++y){const row=new Float64Array(2);for(let x=0;x<2;++x)sum+=row[x];}return sum;}",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let planner = text.split("planner f\n").nth(1).unwrap();
    for entry in [
        "regions:",
        "region candidates after specialization:",
        "region loop at block",
        "preheader=",
        "blocks:",
        "typed receiver ",
        "typed read ",
    ] {
        assert!(planner.contains(entry), "{entry}\n{planner}");
    }
    let semantic = text
        .split("semantic IR f\n")
        .nth(1)
        .unwrap()
        .split("planner f")
        .next()
        .unwrap();
    assert!(!semantic.contains("region candidates after specialization"));
    let plain = compile(
        &[],
        "function f(rows){let sum=0;for(let y=0;y<2;++y){const row=new Float64Array(2);for(let x=0;x<2;++x)sum+=row[x];}return sum;}",
    );
    assert!(plain.status.success());
    let plain = String::from_utf8(plain.stdout).unwrap();
    assert_eq!(
        without_constants(text.split("[f@1#0] planner").next().unwrap()),
        without_constants(&plain)
    );
}

fn without_constants(text: &str) -> String {
    text.lines()
        .map(|line| match line.split_once("bits: ") {
            Some((head, tail)) => {
                let rest = tail.trim_start_matches(|c: char| c.is_ascii_digit());
                format!("{head}bits: _{rest}")
            }
            None => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn execution_collects_object_feedback_and_prints_guarded_variants() {
    let output = compile(
        &[
            "--filter",
            "selected",
            "--planner",
            "--clif",
            "--asm",
            "--execute",
            "const object = {method: function(x){return x+1;}}; const runner = {call: selected}; for(let i=0;i<32;++i) runner.call(object); object.method = function(x){return x+2;}; print(runner.call(object));",
        ],
        "function selected(object){return object.method(4);}",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("feedback: 0"), "{text}");
    assert!(text.contains(".feedback@"), "{text}");
    assert!(text.contains("guard_code=true"), "{text}");
    assert!(
        text.matches("Cranelift IR after optimization").count() >= 2,
        "{text}"
    );
    assert!(text.matches("machine code (").count() >= 2, "{text}");
    assert!(text.lines().any(|line| line == "6"), "{text}");
}

#[test]
fn execution_requires_compile_and_reports_runtime_errors() {
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--execute", "main();", "--eval", "function main(){}"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("require --compile"));
    let output = compile(&["--execute", "missing();"], "function f(){}");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing"));
}

#[test]
fn filter_append_allows_loop_guards_and_preserves_method_replacement() {
    let output = compile(
        &[
            "--filter",
            "selected",
            "--planner",
            "--execute",
            "const values = [1,2,3]; const constraint = {allows: function(value){return true;}}; const runner = {call:selected}; for(let i=0;i<32;++i) runner.call(values,constraint); const before = runner.call(values,constraint).length; constraint.allows = function(value){return false;}; print(before === 3 && runner.call(values,constraint).length === 0);",
        ],
        "function selected(values,constraint){return values.filter(value => constraint.allows(value));}",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("combined entry guards:"), "{text}");
    assert!(text.contains("hoisted guard"), "{text}");
    assert!(text.lines().any(|line| line == "true"), "{text}");
}

#[test]
fn native_push_feedback_specializes_unknown_receivers_and_keeps_fallback() {
    let output = compile(
        &[
            "--filter",
            "selected",
            "--planner",
            "--execute",
            "const runner={call:selected}; const a=[]; for(let i=0;i<32;++i) runner.call(a,i); const custom={push:function(x){return x+100;}}; print(a.length===32 && runner.call(custom,7)===107);",
        ],
        "function selected(a,x){return a.push(x);}",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let (initial, specialized) = text.split_once(".feedback@").unwrap();
    assert!(!initial.contains("helper_call ArrayAppend"), "{text}");
    assert!(specialized.contains("native ArrayPush"), "{text}");
    assert!(
        specialized.contains("helper_call ArrayIterationGuard"),
        "{text}"
    );
    assert!(specialized.contains("helper_call ArrayAppend"), "{text}");
    assert!(text.lines().any(|line| line == "true"), "{text}");
}

#[test]
fn fresh_array_push_is_lowered_without_execution_or_identity_guards() {
    let source = "function selected(x){const a=[]; const first=a.push(x,2); const second=a.push(); return first===2 && second===2 && a[0]===x && a[1]===2;}";
    let output = compile(&["--filter", "selected", "--planner"], source);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("feedback: 0"), "{text}");
    assert!(text.contains("lower Array.push without guards"), "{text}");
    assert!(text.contains("helper_call ArrayAppend"), "{text}");
    assert!(!text.contains("helper_call ArrayIterationGuard"), "{text}");
    assert!(!text.contains(".feedback@"), "{text}");
    let output = compile(
        &["--filter", "selected", "--execute", "print(selected(7));"],
        source,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| line == "true")
    );
}

#[test]
fn generator_regions_are_inspectable_and_published_for_execution() {
    let output = compile(
        &[
            "--filter",
            ".resume.",
            "--clif",
            "--execute",
            "const g=gen(40);print(g.next().value+g.next().value);",
        ],
        "function* gen(x){yield x;return x+2;}",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        text.matches("Cranelift IR after optimization").count(),
        2,
        "{text}"
    );
    assert!(text.lines().any(|line| line == "82"), "{text}");
}

#[test]
fn truthiness_lowering_uses_type_facts() {
    for (source, helper, numeric) in [
        (
            "function selected(){if({})return 1;return 0;}",
            false,
            false,
        ),
        (
            "function selected(x){if(''+x)return 1;return 0;}",
            true,
            false,
        ),
        (
            "function selected(n,b){let value=n>0?+n:b>0;if(value)return 1;return 0;}",
            false,
            true,
        ),
        (
            "function selected(n){let value=n>0?{}:null;if(value)return 1;return 0;}",
            false,
            true,
        ),
        ("function selected(x){if(x)return 1;return 0;}", true, true),
    ] {
        let output = compile(
            &["--filter", "selected", "--after", "finalize-runtime"],
            source,
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        assert_eq!(text.contains("helper: Some(Truthy)"), helper, "{text}");
        assert_eq!(text.contains("FloatCompare"), numeric, "{text}");
    }
}
