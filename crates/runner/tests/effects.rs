use std::process::Command;

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
    assert!(dump.contains("combined entry guards:"));
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
            .contains("combined entry guards:")
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
            .contains(": combined entry guards:")
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

#[test]
fn array_iteration_exposes_and_inlines_known_closures() {
    for method in ["map", "some", "every", "filter"] {
        let source = format!(
            "function createTerrain(terrain){{ return typeof terrain === 'string' ? terrain : terrain.{method}(t=>createTerrain(t)); }}"
        );
        for enabled in [false, true] {
            let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
                .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
                .args(["--compile", "--no-ir", "--semantic-ir", "--eval", &source])
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
            assert_eq!(caller.contains(&format!("is_array_{method}")), enabled);
            assert_eq!(caller.contains("map_element"), enabled);
            if enabled {
                assert!(caller.contains("inline") && caller.contains("0 code check(s)"));
                assert!(caller.contains("materialized only in fallback"));
                assert!(
                    caller.find(&format!("is_array_{method}")).unwrap()
                        < caller.find(" = closure ").unwrap()
                );
            }
        }
    }
}

#[test]
fn inlined_local_calls_remove_unused_closure_allocations() {
    let source = "function main(){ const f=x=>x*x; return f(3); }";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_SSA", "1")
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
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
    assert!(!caller.contains(" = closure "));
    assert!(caller.contains("unused closure"));
}

#[test]
fn construction_exposes_prototype_resolution() {
    let source = "function make(C,p){return new C((C.prototype=p,1));} function spread(C,args){return new C(...args);}";
    for enabled in [false, true] {
        let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
            .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let dump = String::from_utf8(output.stdout).unwrap();
        assert_eq!(dump.contains("constructor_prototype"), enabled);
        assert_eq!(dump.contains("construct_resolved"), enabled);
        assert_eq!(dump.contains("construct_spread_resolved"), enabled);
        assert_eq!(dump.contains("new_instance"), enabled);
        assert_eq!(dump.contains("constructor_call "), enabled);
        assert_eq!(dump.contains("constructor_call_spread"), enabled);
        assert_eq!(dump.contains("constructor_result"), enabled);
        if enabled {
            assert!(dump.find("set ").unwrap() < dump.find("constructor_prototype").unwrap());
        }
    }
}

#[test]
fn known_constructor_bodies_can_be_inlined() {
    let source = "function C(x){this.x=x;} function main(){C.prototype={};return new C(7).x;}";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
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
    assert!(main.contains("constructor_result"));
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
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
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
fn inlined_captures_use_bindings_and_materialize_only_for_fallbacks() {
    let source = "function main(xs){ let limit=2; return xs.some(x=>x>limit); }";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .env("JS_JIT_SSA", "1")
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let caller = dump
        .split("semantic IR main\n")
        .nth(1)
        .unwrap()
        .split("semantic IR")
        .next()
        .unwrap();
    assert!(!caller.contains("enter_captures"));
    assert!(caller.contains("read limit#"));
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
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
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
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
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
        .env("JS_JIT_SSA", "1")
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let function = dump
        .split("semantic IR distance\n")
        .nth(1)
        .unwrap()
        .split("semantic IR")
        .next()
        .unwrap();
    assert!(function.contains("function: combined entry guards:"));
    let fast = function.split("\nblock0:").next().unwrap();
    assert_eq!(fast.matches("probe_global square").count(), 1);
    assert_eq!(fast.matches(" = is_code ").count(), 1);
    assert_eq!(fast.matches(" = is_number ").count(), 4);
    assert!(!fast.contains(" = global "));
    assert!(!fast.contains(" = call "));
    assert_eq!(fast.matches(" * ").count(), 2);
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
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
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
            .env("JS_JIT_SSA", if enabled { "1" } else { "0" })
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
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let caller = dump.split("semantic IR main\n").nth(1).unwrap();
    assert!(caller.contains("constructor_call_spread"));
    assert!(caller.contains("inline funcid"));
    assert!(caller.contains("is_code"));
}

#[test]
fn recursively_inlined_bodies_expand_arrays_and_constructors_once() {
    let source = "function SimpleTerrain(a){this.x=a;} function createTerrain(t){return typeof t === 'string' ? new SimpleTerrain(...[t]) : t.map(x=>createTerrain(x));} function main(t){return createTerrain(t);}";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
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
    let maps = caller.matches(" = is_array_map ").count();
    let constructors = caller.matches(" = construct_spread_resolved ").count();
    assert!(maps >= 2 && maps <= 4, "{maps}");
    assert!(constructors >= 2 && constructors <= 4, "{constructors}");
    assert!(!caller.contains(" = construct_spread "));
    let start = source.find("new SimpleTerrain").unwrap();
    assert!(
        caller
            .lines()
            .filter(|line| line.contains(&format!("note: call at {start}.."))
                && line.contains("inline funcid"))
            .count()
            >= 2
    );
    assert_eq!(caller.matches("note: array map at ").count(), maps);
}

#[test]
fn split_and_spread_inline_have_guarded_semantic_operations() {
    let source = "function C(a,b){this.a=a;this.b=b;} function main(s){C.prototype={};return new C(...s.split('|')).a;}";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dump = String::from_utf8(output.stdout).unwrap();
    let caller = dump.split("semantic IR main\n").nth(1).unwrap();
    assert!(caller.contains("is_string_split"));
    assert!(caller.contains(" = string_split "));
    assert!(caller.contains("is_indexed"));
    assert_eq!(caller.matches(" = spread_get ").count(), 2);
    assert!(caller.contains("constructor_call_spread"));
    assert!(caller.contains("reenters, throws"));
}

#[test]
fn string_split_guard_can_select_a_closed_version() {
    let source = "function f(s,n){let x=0;for(let i=0;i<n;i++){x+=s.split(',').length;}return x;} function main(){return f('a,b',3);}";
    let output = Command::new(env!("CARGO_BIN_EXE_js-runner"))
        .args(["--compile", "--no-ir", "--semantic-ir", "--eval", source])
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
    assert!(caller.contains("probe") && caller.contains("is_string_split"));
    assert!(caller.contains("hoisted guards"));
}
