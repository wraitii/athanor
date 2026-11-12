#![cfg(target_arch = "aarch64")]

use std::process::Command;

fn run(args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_BASELINE", "1")
        .env("JS_JIT_SSA", "0")
        .env("JS_JIT_BACKGROUND", "0")
        .env("JS_JIT_OSR", "0")
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn direct_baseline_skips_clif_and_executes_calls_and_locals() {
    let source = "function f(){return (21+3)*2;} function g(){var a=42; return a;} function main(){return f()+g();}";
    let inspection = run(&["--compile", "--clif", "--filter", "f@", "--eval", source]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "90");
}

#[test]
fn direct_baseline_branches_and_strings() {
    let source = "function f(){if (true) return 11; return 22;} function g(){if (false) return 11; return 22;} function h(){return 'x';} function main(){return f()+g()+h();}";
    let inspection = run(&["--compile", "--clif", "--filter", "h@", "--eval", source]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "33x");
}

#[test]
fn direct_baseline_retains_dynamic_arguments() {
    let source = "function id(a){return a;} function second(a,b){return b;} function fifth(a,b,c,d,e){return e;} function main(){var x=['ok']; var y=id(x); x=null; return fifth(1,2,3,4,second(1,y))[0];}";
    let inspection = run(&["--compile", "--clif", "--filter", "id@", "--eval", source]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "ok");
    let inspection = run(&[
        "--compile",
        "--clif",
        "--filter",
        "fifth@",
        "--eval",
        source,
    ]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
}

#[test]
fn direct_baseline_handles_dynamic_numeric_operations() {
    let source =
        "function f(a,b){if(a<b)return a*b; return a-b;} function main(){return f(3,4)+f('9',2);}";
    let inspection = run(&["--compile", "--clif", "--filter", "f@", "--eval", source]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "19");
}

#[test]
fn direct_baseline_propagates_helper_errors() {
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_BASELINE", "1")
        .env("JS_JIT_SSA", "0")
        .env("JS_JIT_BACKGROUND", "0")
        .env("JS_JIT_OSR", "0")
        .args([
            "--eval",
            "function f(a){return a-1;} function main(){return f({});}",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("object-to-primitive"));
}

#[test]
fn direct_baseline_keeps_indexed_values_alive_across_mutation() {
    let source = "function take(a,b){var x=a[0]; a[0]=b; return x;} function join(a,b){return a+b;} function main(){var a=['old']; var x=take(a,'new'); return join(x,a[0]);}";
    let inspection = run(&["--compile", "--clif", "--filter", "take@", "--eval", source]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "oldnew");
}

#[test]
fn direct_baseline_executes_nested_loops_with_five_arguments() {
    let source = "function blur(image,kernel,out,w,h){for(var y=1;y<h-1;++y) for(var x=1;x<w-1;++x){var sum=0; for(var k=0;k<3;++k) sum+=image[y*w+x+k-1]*kernel[k]; out[y*w+x]=sum;}} function main(){var image=[1,2,3,4,5,6,7,8,9]; var out=[0,0,0,0,0,0,0,0,0]; blur(image,[1,2,1],out,3,3); return out[4];}";
    let inspection = run(&["--compile", "--clif", "--filter", "blur@", "--eval", source]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "20");
}

#[test]
fn direct_baseline_assembly_has_semantic_locations() {
    let output = run(&["--compile", "--asm", "--eval", "function f(a){return a-1;}"]);
    assert!(output.contains("; @000"), "{output}");
}

#[test]
fn direct_baseline_handles_methods_properties_globals_and_block_arguments() {
    let source = "var mask=2; function work(a){a.n=a.n+1; return (a.n & mask) || a.next(a.n);} function main(){var o={n:0,next:function(x){return x+10;}}; return work(o)+work(o);}";
    let inspection = run(&["--compile", "--clif", "--filter", "work@", "--eval", source]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "13");
}

#[test]
fn direct_baseline_propagates_exceptions_from_calls() {
    let source = "function callit(f){return f();} function boom(){throw 'caught';} function main(){try {callit(boom);} catch(e) {return e;} return 'missed';}";
    let inspection = run(&[
        "--compile",
        "--clif",
        "--filter",
        "callit@",
        "--eval",
        source,
    ]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "caught");
}

#[test]
fn direct_baseline_checks_recursive_stack_limit() {
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_BASELINE", "1")
        .env("JS_JIT_SSA", "0")
        .env("JS_JIT_BACKGROUND", "0")
        .env("JS_JIT_OSR", "0")
        .args([
            "--eval",
            "function f(n){if(n)return f(n-1);return 0;} function main(){return f(1000000);}",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("too much recursion"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn direct_baseline_constructs_objects_and_propagates_throws() {
    let source = "function Item(a,b){this.value=a+b;} function build(a,b){return new Item(a,b);} function fail(){throw new Error('failure');} function main(){var x=build(4,5); try{fail();}catch(e){return x.value;} return 0;}";
    for name in ["build@", "fail@"] {
        let inspection = run(&["--compile", "--clif", "--filter", name, "--eval", source]);
        assert!(
            inspection.contains("Cranelift IR unavailable for direct baseline code"),
            "{inspection}"
        );
    }
    assert_eq!(run(&["--eval", source]).trim(), "9");
}

#[test]
fn direct_baseline_preserves_heap_values_across_slot_replacements() {
    let source = "function replace(a,b){var x=a; x=1; x=b; x=x; x=0; return b;} function main(){var a=['a']; var b=['b']; return replace(a,b)[0]+a[0]+b[0];}";
    let inspection = run(&[
        "--compile",
        "--clif",
        "--filter",
        "replace@",
        "--eval",
        source,
    ]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "bab");
}

#[test]
fn direct_baseline_property_cache_observes_mutation_and_shadowing() {
    let source = "function C(){} C.prototype.x=1; function read(o){return o.x;} function main(){var a=new C(); var b=new C(); var sum=read(a)+read(b); C.prototype.x=3; sum+=read(a); a.x=5; sum+=read(a)+read(b); return sum;}";
    let inspection = run(&["--compile", "--clif", "--filter", "read@", "--eval", source]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "13");
}

#[test]
fn direct_baseline_handles_lexical_scopes_and_loops() {
    let source = "function work(a){let result=''; const x=a; for(let i=0;i<4;++i){let y; {const x='inner'; y=x;} if(i===1)continue; result+=y; if(i===2)break;} {let x='shadow'; result+=x;} return result+x[0];} function main(){return work(['outer']);}";
    let inspection = run(&["--compile", "--clif", "--filter", "work@", "--eval", source]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "innerinnershadowouter");
    assert_eq!(
        run(&["--eval", "function main(){let x; return x;}"]).trim(),
        "undefined"
    );
}

#[test]
fn direct_baseline_checks_lexical_initialization_on_each_scope_entry() {
    for source in [
        "function work(){return x; let x=1;} function main(){return work();}",
        "function work(){x=1; let x;} function main(){return work();}",
        "function work(){const x=x; return x;} function main(){return work();}",
        "function work(){for(let i=0;i<2;++i){if(i)return x; let x='old';}} function main(){return work();}",
        "function work(){for(let i=0;i<2;++i){if(i)x='new'; let x='old';}} function main(){return work();}",
    ] {
        let inspection = run(&["--compile", "--clif", "--filter", "work@", "--eval", source]);
        assert!(
            inspection.contains("Cranelift IR unavailable for direct baseline code"),
            "{inspection}"
        );
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_BASELINE", "1")
            .env("JS_JIT_SSA", "0")
            .env("JS_JIT_BACKGROUND", "0")
            .env("JS_JIT_OSR", "0")
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{source}");
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("binding accessed before initialization"),
            "{source}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn direct_baseline_allocates_nested_literals_and_preserves_references() {
    let source = "function build(value){const item={value:value}; return [item,[],{},[item]];} function main(){let value=['old']; const result=build(value); value=null; result[0].value=['new']; return result[3][0].value[0]+result[1].length;}";
    let inspection = run(&[
        "--compile",
        "--clif",
        "--filter",
        "build@",
        "--eval",
        source,
    ]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "new0");
}

#[test]
fn direct_baseline_iterates_arrays_typed_arrays_and_strings() {
    let source = "function collect(input){let result=''; for(const value of input){result+=value;} return result;} function grow(input){let sum=0; for(const value of input){sum+=value; if(value===1)input[1]=2;} return sum;} function main(){const typed=new Uint8Array(2); typed[0]=3; typed[1]=4; return collect(['a','b'])+collect(typed)+collect('😀é')+grow([1]);}";
    for name in ["collect@", "grow@"] {
        let inspection = run(&["--compile", "--clif", "--filter", name, "--eval", source]);
        assert!(
            inspection.contains("Cranelift IR unavailable for direct baseline code"),
            "{inspection}"
        );
    }
    assert_eq!(run(&["--eval", source]).trim(), "ab34😀é3");
}

#[test]
fn direct_baseline_enumerates_objects_and_checks_types() {
    let source = "function Item(){} function inspect(value){let result=''; for(const key in value){result+=key+typeof value[key];} return result+('x' in value)+(value instanceof Item)+(2**3);} function main(){const value=new Item(); value.x=1; value.y='s'; return inspect(value);}";
    let inspection = run(&[
        "--compile",
        "--clif",
        "--filter",
        "inspect@",
        "--eval",
        source,
    ]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "xnumberystringtruetrue8");
}

#[test]
fn direct_baseline_preserves_captured_cells_and_loop_iterations() {
    let source = "function make(){let value=['old']; const read=()=>value[0]; const write=x=>value=x; const functions=[]; for(let i=0;i<3;++i)functions.push(()=>i); for(const x of ['a','b'])functions.push(()=>x); return {read:read,write:write,functions:functions};} function main(){const result=make(); result.write(['new']); let text=result.read(); for(const f of result.functions)text+=f(); return text;}";
    let inspection = run(&["--compile", "--clif", "--filter", "make@", "--eval", source]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "new012ab");
    assert_eq!(run(&["--eval", "function make(){return ()=>this.x;} function main(){const object={x:'receiver',make:make}; return object.make()();}"]).trim(), "receiver");
}

#[test]
fn direct_baseline_checks_captured_initialization() {
    for source in [
        "function main(){const read=()=>x; return read(); let x=1;}",
        "function main(){const write=()=>x=1; write(); let x;}",
    ] {
        let inspection = run(&["--compile", "--clif", "--filter", "main", "--eval", source]);
        assert!(
            inspection.contains("Cranelift IR unavailable for direct baseline code"),
            "{inspection}"
        );
        assert!(
            !inspection.contains("Cranelift IR before optimization"),
            "{inspection}"
        );
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_BASELINE", "1")
            .env("JS_JIT_SSA", "0")
            .env("JS_JIT_BACKGROUND", "0")
            .env("JS_JIT_OSR", "0")
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("binding accessed before initialization"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn direct_baseline_handles_large_frames_and_stack_arguments() {
    let source = format!(
        "function work(a,b,c,d,e){{let x=e; {} return x;}} function main(){{return work(2,3,4,5,6);}}",
        "x+=a;".repeat(250)
    );
    let inspection = run(&[
        "--compile",
        "--clif",
        "--filter",
        "work@",
        "--eval",
        &source,
    ]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", &source]).trim(), "506");
    let source = format!(
        "function work(n,a,b,c,d){{let x=d; {} if(n)return work(n-1,a,b,c,d); return x;}} function main(){{return work(1000000,2,3,4,5);}}",
        "x+=a;".repeat(250)
    );
    let inspection = run(&[
        "--compile",
        "--clif",
        "--filter",
        "work@",
        "--eval",
        &source,
    ]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_BASELINE", "1")
        .env("JS_JIT_SSA", "0")
        .env("JS_JIT_BACKGROUND", "0")
        .env("JS_JIT_OSR", "0")
        .args(["--eval", &source])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("too much recursion"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn direct_baseline_executes_script_initializers_and_global_writes() {
    let source = "var value=['old']; let current=value; const fixed='fixed'; var receiver=this; value=['new']; function main(){current=value; return current[0]+fixed+(receiver===globalThis);}";
    let inspection = run(&[
        "--compile",
        "--clif",
        "--filter",
        ".module",
        "--eval",
        source,
    ]);
    assert!(
        inspection.contains("Cranelift IR unavailable for direct baseline code"),
        "{inspection}"
    );
    assert_eq!(run(&["--eval", source]).trim(), "newfixedtrue");
}

#[test]
fn direct_baseline_propagates_script_initializer_errors() {
    for (source, message) in [
        (
            "let value=value; function main(){return 0;}",
            "before initialization",
        ),
        (
            "function fail(){throw 'initializer failed';} fail(); function main(){return 0;}",
            "initializer failed",
        ),
    ] {
        let inspection = run(&[
            "--compile",
            "--clif",
            "--filter",
            ".module",
            "--eval",
            source,
        ]);
        assert!(
            inspection.contains("Cranelift IR unavailable for direct baseline code"),
            "{inspection}"
        );
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_BASELINE", "1")
            .env("JS_JIT_SSA", "0")
            .env("JS_JIT_BACKGROUND", "0")
            .env("JS_JIT_OSR", "0")
            .args(["--eval", source])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(message),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
