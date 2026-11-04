use std::process::Command;

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

#[test]
fn private_iteration_does_not_block_versioning() {
    let source = "function f(p,xs){ const [a,b]=[p.x,p.y]; let s=0; for(const v of xs) s+=Math.floor(v+a+b); return s; } function main(){ const xs=[0.5,1.5]; const r=f({x:1,y:2},xs); Math.floor=x=>x*10; return r+','+f({x:0,y:0},xs); }";
    let dump = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
        .output()
        .unwrap();
    assert!(dump.status.success());
    let dump = String::from_utf8(dump.stdout).unwrap();
    let f = dump.split("semantic IR f").nth(1).unwrap();
    assert!(
        f.split("semantic IR")
            .next()
            .unwrap()
            .contains("note: versioned:")
    );
    for enabled in [false, true] {
        let result = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
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
        .env("JS_JIT_SSA", "1")
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
        .output()
        .unwrap();
    assert!(dump.status.success());
    let dump = String::from_utf8(dump.stdout).unwrap();
    let f = dump.split("semantic IR f").nth(1).unwrap();
    assert!(
        f.split("semantic IR")
            .next()
            .unwrap()
            .contains(": versioned:")
    );
    for enabled in [false, true] {
        let result = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
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
