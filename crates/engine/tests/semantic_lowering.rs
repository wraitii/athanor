use js_engine::{Engine, EngineOptions, Value};

fn run(source: &str, ssa_opt: bool) -> Result<Value, String> {
    let options = EngineOptions {
        ssa_opt,
        ..Default::default()
    };
    let mut engine = Engine::with_options(options).unwrap();
    engine
        .compile(source, "semantic-lowering.js")
        .map_err(|e| e.to_string())?;
    engine
        .call("main", &[])
        .map(|value| value.value())
        .map_err(|e| e.to_string())
}

#[test]
fn explicit_integer_coercions_preserve_numeric_edges() {
    let source = r#"
        function check(input, expected) {
            const value = +input;
            const shift = 33.75;
            return (value | 0) === expected && (value & -1) === expected &&
                (value ^ 0) === expected && ~value === ~expected &&
                (value << shift) === (expected << 1) &&
                (value >> shift) === (expected >> 1) &&
                (value >>> shift) === (expected >>> 1);
        }
        function count(grid, index, limit) {
            const word = grid[index];
            let members = 0;
            if (word)
                for (let i = 0; i < limit; ++i)
                    if (word & (1 << i))
                        ++members;
            return members;
        }
        function sum(grid) {
            let result = 0;
            for (let i = 0; i < 3; ++i)
                result += grid[i] | 0;
            return result;
        }
        function countAll(grid) {
            let members = 0;
            for (let x = 0; x < grid.length; ++x) {
                const word = grid[x];
                if (word)
                    for (let i = 0; i < 16; ++i)
                        if (word & (1 << i))
                            ++members;
            }
            return members;
        }
        function main() {
            const grid = new Uint16Array(2);
            grid[0] = 65535;
            grid[1] = 5;
            return check(undefined, 0) && check(NaN, 0) &&
                check(Infinity, 0) && check(-Infinity, 0) && check(-0, 0) &&
                check(3.75, 3) && check(-3.75, -3) &&
                check(4294967297.75, 1) && check(-4294967297.75, -1) &&
                check(2147483648, -2147483648) && check(-2147483649, 2147483647) &&
                count(grid, 0, 16) === 16 && count(grid, 1, 16) === 2 &&
                count(grid, 2, 16) === 0 && count(grid, 0, 0) === 0 &&
                sum(grid) === 65540 && countAll(grid) === 18 &&
                count([3.75], 0, 16) === 2 && count(["5"], 0, 16) === 2;
        }
    "#;
    for ssa_opt in [false, true] {
        assert_eq!(run(source, ssa_opt), Ok(Value::TRUE));
    }
}

#[test]
fn bit_test_branches_preserve_signed_bits_and_shift_counts() {
    let source = r#"
        function test(word, index) {
            if ((word | 0) & (1 << (index | 0)))
                return true;
            return false;
        }
        function reversed(word, index) {
            if ((1 << (index | 0)) & (word | 0))
                return true;
            return false;
        }
        function mask(word, index) {
            return (word | 0) & (1 << (index | 0));
        }
        function reused(word, index) {
            const value = (word | 0) & (1 << (index | 0));
            if (value)
                return value;
            return 0;
        }
        function main() {
            const words = [0, -1, 1, 65535, -2147483648, 2147483647, 4294967295];
            const shifts = [-65, -33, -32, -1, 0, 1, 15, 31, 32, 33, 63, NaN, Infinity, 3.75];
            for (const word of words)
                for (const shift of shifts) {
                    const expected = !!((word >> shift) & 1);
                    if (test(word, shift) !== expected || reversed(word, shift) !== expected)
                        return false;
                }
            return mask(-2147483648, 31) === -2147483648 && mask(-1, 32) === 1 &&
                reused(-2147483648, 31) === -2147483648 && reused(0, 31) === 0;
        }
    "#;
    for ssa_opt in [false, true] {
        assert_eq!(run(source, ssa_opt), Ok(Value::TRUE));
    }
}

#[test]
fn repeated_typed_reads_preserve_fallbacks_and_mutations() {
    let source = r#"
        function count(grid, width) {
            let members = 0;
            for (let y = 0; y < 2; ++y) {
                const indexY = y * width;
                for (let x = 0; x < 2; ++x)
                    if (grid[indexY + x])
                        for (let i = 0; i < 16; ++i)
                            if (grid[indexY + x] & (1 << i))
                                ++members;
            }
            return members;
        }
        function mutate(grid) {
            let members = 0;
            if (grid[0])
                for (let i = 0; i < 16; ++i) {
                    grid[0] = 1 << i;
                    if (grid[0] & (1 << i))
                        ++members;
                }
            return members;
        }
        function main() {
            const grid = new Uint16Array(2);
            grid[0] = 65535;
            grid[1] = 5;
            const floats = new Float64Array(4);
            floats[0] = NaN;
            floats[1] = Infinity;
            floats[2] = 3.75;
            for (let warmup = 0; warmup < 3; ++warmup) {
                const changing = new Uint16Array(1);
                changing[0] = 1;
                if (count(grid, 2) !== 18 || mutate(changing) !== 16)
                    return false;
            }
            return count(["5", 3.75], 2) === 4 && count(floats, 2) === 2 &&
                count(grid, 0.5) === 18 && count(grid, NaN) === 0 &&
                count(grid, Infinity) === 0 && count([], 2) === 0;
        }
    "#;
    for ssa_opt in [false, true] {
        assert_eq!(run(source, ssa_opt), Ok(Value::TRUE));
    }
}

#[test]
fn typeof_string_comparisons_preserve_values_and_evaluation() {
    let source = r#"
        function isString(value) { return typeof value === "string"; }
        function reversed(value) { return "string" === typeof value; }
        function isOther(value) { return typeof value !== "string"; }
        function reversedOther(value) { return "string" !== typeof value; }
        function shared(value) {
            let type = typeof value;
            return type === "string" && type === value;
        }
        function main() {
            let values = ["", "é😀", undefined, null, false, 0, NaN, [], {}, isString, Math.abs];
            for (let i = 0; i < values.length; ++i) {
                let expected = i < 2;
                if (isString(values[i]) !== expected || reversed(values[i]) !== expected ||
                    isOther(values[i]) === expected || reversedOther(values[i]) === expected)
                    return false;
            }
            let calls = 0;
            function next() { calls++; return "string"; }
            return typeof next() === "string" && "string" !== typeof (calls++, 7) &&
                calls === 2 && typeof missingGlobal !== "string" && shared("string");
        }
    "#;
    for ssa_opt in [false, true] {
        assert_eq!(run(source, ssa_opt), Ok(Value::TRUE));
    }
}

#[test]
fn typeof_string_comparisons_preserve_errors() {
    for source in [
        "function main(){ return typeof x === 'string'; let x; }",
        "function main(){ return 'string' !== typeof x; let x; }",
        "function main(){ return typeof null.x === 'string'; }",
        "function main(){ return typeof x === 'string'; } let result = main(); let x;",
        "function main(){ let type = typeof x; let x; return type === 'string'; }",
    ] {
        let expected = run(source, false);
        assert!(expected.is_err(), "{source}");
        assert_eq!(run(source, true), expected, "{source}");
    }
}

#[test]
fn string_equality_compares_contents_across_allocations() {
    let source = r#"
        function equal(a, b) { return a === b && a == b && !(a !== b) && !(a != b); }
        function main() {
            let text = "é😀";
            return equal(text, text) && equal(text, "é" + "😀") && equal("", "x".slice(1)) &&
                !equal(text, "e😀") && !equal("é", "é") && !equal("7", 7) &&
                !equal("", undefined) && !equal("", null);
        }
    "#;
    for ssa_opt in [false, true] {
        assert_eq!(run(source, ssa_opt), Ok(Value::TRUE));
    }
}

#[test]
fn heightmap_runs_through_semantic_graph() {
    let source = r#"
        function getMinAndMaxHeight(heightmap = g_Map.height) {
            const height = { "min": Infinity, "max": -Infinity };
            for (let x = 0; x < heightmap.length; ++x)
                for (let y = 0; y < heightmap[x].length; ++y) {
                    height.min = Math.min(height.min, heightmap[x][y]);
                    height.max = Math.max(height.max, heightmap[x][y]);
                }
            return height;
        }
        function main() {
            let result = getMinAndMaxHeight([[3, -4], [8, 2]]);
            return result.min * 100 + result.max;
        }
    "#;
    assert_eq!(run(source, true).unwrap(), Value::number(-392.0));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn graph_ownership_and_errors_match_existing_emitter() {
    for source in [
        "function main(){ let a = [7]; let b = [9]; for (let i = 0; i < 10; ++i) { let c = a; a = b; b = c; } return a[0] * 10 + b[0]; }",
        "function main(){ let result = 0; for (let i = 0; i < 3; ++i) { result += x; let x = i; } return result; }",
        "function main(){ return null.missing; }",
        "function main(){ let x = { value: 3 }; let y = x; x = [1, 2]; return y.value + x[1]; }",
        "var g_Map = { height: [[-2, 5], [9, 3]] }; function range(heightmap = g_Map.height){ return heightmap[0][0] + heightmap[1][0]; } function main(){ return range(); }",
        "function substitute(a, b){ return a + b; } function main(){ Math.min = substitute; return Math.min(3, 5); }",
    ] {
        assert_eq!(run(source, true), run(source, false), "{source}");
    }
}

#[test]
fn versioned_heightmap_matches_fallbacks_and_errors() {
    let function = r#"
        function getMinAndMaxHeight(heightmap = g_Map.height) {
            const height = { min: Infinity, max: -Infinity };
            for (let x = 0; x < heightmap.length; ++x)
                for (let y = 0; y < heightmap[x].length; ++y) {
                    height.min = Math.min(height.min, heightmap[x][y]);
                    height.max = Math.max(height.max, heightmap[x][y]);
                }
            return height;
        }
    "#;
    for suffix in [
        "function main(){ let h = getMinAndMaxHeight([]); return h.min; }",
        "function main(){ let h = getMinAndMaxHeight([[3, -4], [8, 2]]); return h.min * 100 + h.max; }",
        "function replace(a, b){ return a + b; } function main(){ Math.min = replace; let h = getMinAndMaxHeight([[1, 2]]); return h.min; }",
        "function replace(a, b){ return a + b; } function main(){ Math.max = replace; let h = getMinAndMaxHeight([[1, 2]]); return h.max; }",
        "function main(){ return getMinAndMaxHeight([null]); }",
        "function main(){ return getMinAndMaxHeight([[{}]]); }",
        "function main(){ return getMinAndMaxHeight(); }",
    ] {
        let source = format!("{function}\n{suffix}");
        assert_eq!(run(&source, true), run(&source, false), "{suffix}");
    }
}

#[test]
fn versioned_loop_preserves_heap_values_across_phi_swaps() {
    let source = r#"
        function run(a, b) {
            for (let i = 0; i < 11; ++i) {
                let previous = a;
                a = b;
                b = previous;
                Math.min(i, 9);
            }
            return a[0] * 10 + b[0];
        }
        function main(){ return run([7], [9]); }
    "#;
    assert_eq!(run(source, true), Ok(Value::number(97.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn tdz_is_checked_only_when_binding_is_accessed() {
    for source in [
        "function main(){ for(let i=0;i<2;i++){ if(i) return x; let x=1; } return 0; }",
        "function main(){ for(let i=0;i<2;i++){ if(i > 1) return x; let x=1; } return 0; }",
        "function main(){ let n = 0; for(let i=0;i<3;i++){ if(i == 3) n += x; let x = i; n += x; } return n; }",
        "function main(){ if(false) return x; if(true) return 8; let x = 1; return x; }",
    ] {
        assert_eq!(run(source, true), run(source, false), "{source}");
    }
}

#[test]
fn semantic_construction_and_receivers_match_existing_emitter() {
    for (source, expected) in [
        (
            "function Point(x, y){ this.x = x; this.y = y; } function main(){ let p = new Point(3, 5); return p.x * 10 + p.y; }",
            Value::number(35.0),
        ),
        (
            "function Point(x){ this.x = x; return 7; } function main(){ return (new Point(6)).x; }",
            Value::number(6.0),
        ),
        (
            "function Point(x){ this.x = x; return { x: 9 }; } function main(){ return (new Point(6)).x; }",
            Value::number(9.0),
        ),
        (
            "function receiver(){ return this === undefined; } function main(){ return receiver(); }",
            Value::TRUE,
        ),
        (
            "function receiver(){ return this.value; } function main(){ let o = {value: 7, method: receiver}; return o.method(); }",
            Value::number(7.0),
        ),
        (
            "function C(a, b){ this.value = a * 10 + b; } function main(){ let i = 0; let makers = { ctor: C }; let v = new makers[(i = 1, 'ctor')]((i = i + 2), i); return v.value * 10 + i; }",
            Value::number(333.0),
        ),
    ] {
        assert_eq!(run(source, true), Ok(expected), "{source}");
        assert_eq!(run(source, true), run(source, false), "{source}");
    }
    let source = "function main(){ return new 42(); }";
    assert!(run(source, true).is_err());
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn semantic_noise_constructor_executes_real_map_library_code() {
    let library = include_str!("../../../../binaries/data/mods/public/maps/random/rmgen/Noise.js");
    let start = library.find("function Noise2D(freq)").unwrap();
    let end = library[start..].find("\nNoise2D.prototype.get").unwrap() + start;
    let source = format!(
        r#"
        function Vector2D(x, y) {{ this.x = x; this.y = y; }}
        function randomAngle() {{ return 0; }}
        {}
        function main() {{ let noise = new Noise2D(2); return noise.freq * 10 + noise.grads[1][1].x; }}
    "#,
        &library[start..end]
    );
    assert_eq!(run(&source, true), Ok(Value::number(21.0)));
    assert_eq!(run(&source, true), run(&source, false));
}

#[test]
fn semantic_global_writes_match_existing_emitter() {
    for (source, expected) in [
        ("var x = 2; function main(){ return x = 9; }", 9.0),
        ("var x = 2; function main(){ x += (x = 5); return x; }", 7.0),
        (
            "var x = 2; function main(){ return x++ * 100 + ++x * 10 + x; }",
            244.0,
        ),
        (
            "var saved = [1]; function main(){ let old = saved; saved = [2]; return old[0] * 10 + saved[0]; }",
            12.0,
        ),
        (
            "var x = 1; function main(){ x = Math.min(5, 3); return x; }",
            3.0,
        ),
    ] {
        assert_eq!(run(source, true), Ok(Value::number(expected)), "{source}");
        assert_eq!(run(source, true), run(source, false), "{source}");
    }
    for source in [
        "function main(){ missing = 3; }",
        "function main(){ missing++; }",
        "function main(){ Infinity = 4; }",
        "function main(){ NaN++; }",
        "function main(){ undefined = 1; }",
        "const fixed = 1; function main(){ fixed = 2; }",
    ] {
        assert!(run(source, true).is_err(), "{source}");
        assert!(run(source, false).is_err(), "{source}");
    }
}

#[test]
fn semantic_global_write_evaluates_rhs_before_reporting_error() {
    let source = "var count = 0; function main(){ missing = (count = count + 1); } function read(){ return count; }";
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "global-write.js").unwrap();
        assert!(engine.call("main", &[]).is_err());
        assert_eq!(
            engine.call("read", &[]).unwrap().value(),
            Value::number(1.0)
        );
    }
}

#[test]
fn semantic_global_write_checks_lexical_initialization() {
    let source = "let x = initialize(); function initialize(){ x = 1; return 0; } function main(){ return x; }";
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        let error = engine
            .compile(source, "global-tdz.js")
            .unwrap_err()
            .to_string();
        assert!(error.contains("before initialization"), "{error}");
    }
}

#[test]
fn semantic_for_of_matches_existing_emitter() {
    for (source, expected) in [
        (
            "function main(){ let sum = 0; for (const x of [1, 2, 3]) sum += x; return sum; }",
            Value::number(6.0),
        ),
        (
            "function main(){ let sum = 5; for (const x of []) sum += x; return sum; }",
            Value::number(5.0),
        ),
        (
            "function main(){ let sum = 0; for (let x of [1, 2, 3, 4]) { if (x == 2) continue; if (x == 4) break; sum += x; } return sum; }",
            Value::number(4.0),
        ),
        (
            "function main(){ for (const x of [5, 6]) { let y = [x]; if (x == 5) continue; return y[0]; } return 0; }",
            Value::number(6.0),
        ),
        (
            "function main(){ let a = [[3], [7]]; let sum = 0; for (const x of a) { a = []; sum += x[0]; } return sum; }",
            Value::number(10.0),
        ),
        (
            "function main(){ let a = new Uint8Array(2); a[0] = 2; a[1] = 4; let sum = 0; for (const x of a) sum += x; return sum; }",
            Value::number(6.0),
        ),
        (
            "function main(){ let s = ''; for (const x of 'a🌍') s += x; return s === 'a🌍'; }",
            Value::TRUE,
        ),
    ] {
        assert_eq!(run(source, true), Ok(expected), "{source}");
        assert_eq!(run(source, true), run(source, false), "{source}");
    }
    for source in [
        "function main(){ for (let x of x) return x; return 0; }",
        "function main(){ for (const x of [1]) x = 3; return 0; }",
        "function main(){ for (const x of null) return x; return 0; }",
    ] {
        assert!(run(source, true).is_err(), "{source}");
        assert!(run(source, false).is_err(), "{source}");
    }
}

#[test]
fn semantic_noncapturing_nested_callables() {
    for (source, expected) in [
        (
            "function main(){ return add(2, 3); function add(a, b){ return a + b; } }",
            5.0,
        ),
        (
            "function main(){ let f = function(x){ return x * 2; }; return f(4); }",
            8.0,
        ),
        ("function main(){ let f = (x) => x + 3; return f(4); }", 7.0),
        (
            "function main(){ let o = { x: 9, f: function(){ let f = () => this.x; return f(); } }; return o.f(); }",
            9.0,
        ),
        (
            "function main(){ let f = (x = 4) => { return x * 2; }; return f(); }",
            8.0,
        ),
    ] {
        assert_eq!(run(source, true), Ok(Value::number(expected)), "{source}");
        assert_eq!(run(source, true), run(source, false), "{source}");
    }
}

#[test]
fn semantic_closure_cells_and_loop_environments() {
    for (source, expected) in [
        ("function main(){ let x = 3; return (() => x)(); }", 3.0),
        (
            "function main(){let a=[];for(let i=0;i<2;i++){ {let x=i+4; a[i]=()=>x;} } return a[0]()*10+a[1]();}",
            45.0,
        ),
        (
            "function main(){ function f(n){ if (n) return f(n - 1) + 1; return 0; } return f(3); }",
            3.0,
        ),
        (
            "function make(){ let x = 0; return [() => ++x, () => x]; } function main(){ let pair = make(); pair[0](); pair[0](); return pair[1](); }",
            2.0,
        ),
        (
            "function make(){ let x = 7; return () => () => ++x; } function main(){ let outer = make(); let inner = outer(); return inner() * 10 + inner(); }",
            89.0,
        ),
        (
            "function main(){ let a = []; for (let i = 0; i < 3; ++i) a[i] = () => i; return a[0]() * 100 + a[1]() * 10 + a[2](); }",
            12.0,
        ),
        (
            "function main(){ let a = []; let i = 0; for (const x of [4, 5, 6]) { a[i++] = () => x; if (x == 5) continue; } return a[0]() * 100 + a[1]() * 10 + a[2](); }",
            456.0,
        ),
        (
            "function f(a = () => b, b = 7){ return a(); } function main(){ return f(); }",
            7.0,
        ),
    ] {
        assert_eq!(run(source, true), Ok(Value::number(expected)), "{source}");
    }
    for source in [
        "function f(a = (() => b)(), b = 7){ return a; } function main(){ return f(); }",
        "function main(){ let f = () => x; return f(); let x = 1; }",
    ] {
        let error = run(source, true).unwrap_err();
        assert!(error.contains("before initialization"), "{source}: {error}");
    }
}

#[test]
fn semantic_var_rest_and_destructuring() {
    for (source, expected) in [
        (
            "function main(){ if (x !== undefined) return 0; if (true) { var x = 5; } return x; }",
            5.0,
        ),
        (
            "function f(x){ var x; var x; return x; } function main(){ return f(8); }",
            8.0,
        ),
        ("function main(){ var x = 3; { var x; } return x; }", 3.0),
        (
            "function main(){ let f = () => x; var x = 8; return f(); }",
            8.0,
        ),
        (
            "function f(a, ...rest){ return a * 100 + rest[0] * 10 + rest.length; } function main(){ return f(2, 4, 8); }",
            242.0,
        ),
        (
            "function f([a,,b]){ return a * 10 + b; } function main(){ return f([3,4,5]); }",
            35.0,
        ),
        (
            "function main(){ let [a,,b] = [3,4,5]; const {x:y} = {x:7}; return a*100 + b*10 + y; }",
            357.0,
        ),
        (
            "function main(){ let sum=0; for (const [x,y] of [[1,2],[3,4]]) sum += x*10+y; return sum; }",
            46.0,
        ),
        (
            "function main(){ var sum=0; for (var x of [1,2,3]) sum += x; return sum*10+x; }",
            63.0,
        ),
        (
            "function f([a,b] = [2,3]){ return a*10+b; } function main(){ return f(); }",
            23.0,
        ),
    ] {
        assert_eq!(run(source, true), Ok(Value::number(expected)), "{source}");
        assert_eq!(run(source, true), run(source, false), "{source}");
    }
    for source in [
        "function main(){ let [x] = null; return x; }",
        "function f([a] = b, b = [1]){ return a; } function main(){ return f(); }",
    ] {
        assert!(run(source, true).is_err(), "{source}");
        assert!(run(source, false).is_err(), "{source}");
    }
}

#[test]
fn exceptions_preserve_values_bindings_and_error_boundaries() {
    for (source, expected) in [
        (
            "function main(){ try { throw undefined; } catch(e) { return e; } }",
            Value::UNDEFINED,
        ),
        (
            "function main(){ try { throw [42]; } catch(e) { return e[0]; } }",
            Value::number(42.0),
        ),
        (
            "function fail(){ throw 'é'; } function main(){ try { fail(); } catch(e) { return e === 'é'; } }",
            Value::TRUE,
        ),
        (
            "function main(){ try { try { throw 1; } catch(e) { throw e+1; } } catch(e) { return e; } }",
            Value::number(2.0),
        ),
        (
            "function main(){ let f; try { throw [42]; } catch(e) { f=()=>e[0]; } return f(); }",
            Value::number(42.0),
        ),
        (
            "function main(){ let n=0; for(let i=0;i<10;++i) { try { throw [i]; } catch(e) { n+=e[0]; } } return n; }",
            Value::number(45.0),
        ),
        (
            "function main(){ let x=1; try { x=2; throw 3; } catch(e) { return x+e; } }",
            Value::number(5.0),
        ),
        (
            "function main(){ try { throw 1; } catch {} return 42; }",
            Value::number(42.0),
        ),
    ] {
        for semantic in [false, true] {
            assert_eq!(run(source, semantic).unwrap(), expected, "{source}");
        }
    }
    for semantic in [false, true] {
        assert!(
            run(
                "function main(){ try { undefined(); } catch { return 42; } }",
                semantic
            )
            .is_err()
        );
        assert!(
            run(
                "function main(){ try { return 1; } finally { return 2; } }",
                semantic
            )
            .is_err()
        );
    }
}

#[test]
fn semantic_arguments_snapshot_and_arrow_captures() {
    for (source, expected) in [
        (
            "function main(a){ a=8; return arguments.length*100+(arguments[0]===undefined)+(arguments[1]===undefined); }",
            2.0,
        ),
        (
            "function f(a){ a=8; arguments[0]=4; return a*10+arguments[0]; } function main(){ return f(3); }",
            84.0,
        ),
        (
            "function f(a){ return ()=>()=>arguments[0]; } function main(){ return f(7)()(); }",
            7.0,
        ),
        (
            "function f(a){ let g=(arguments)=>arguments; return g(9)+(()=>arguments[0])(); } function main(){ return f(3); }",
            12.0,
        ),
        (
            "var arguments=99; function f(){ let g=()=>arguments[0]; arguments=[8]; return g(); } function main(){ return f(4); }",
            8.0,
        ),
        (
            "function f(arguments){ return (()=>arguments)(); } function main(){ return f(11); }",
            11.0,
        ),
        (
            "function C(){ this.x=arguments[0]; } function main(){ return new C(6).x; }",
            6.0,
        ),
        (
            "function f(a=arguments[0]){ return a; } function main(){ return f(4); }",
            4.0,
        ),
    ] {
        for semantic in [false, true] {
            assert_eq!(
                run(source, semantic).unwrap(),
                Value::number(expected),
                "{source}"
            );
        }
    }
}

#[test]
fn semantic_script_initializers_match_global_order_and_tdz() {
    for source in [
        "var x=1; let y=x+2; const z=y+3; class A { value(){return z;} } function main(){return new A().value()+x;}",
        "var x; var x=3; let y; y=4; function main(){return x+y;}",
        "class A { self(){return A;} } var Original=A; A=undefined; function main(){return new Original().self()===Original;}",
    ] {
        assert_eq!(
            run(source, true).unwrap(),
            run(source, false).unwrap(),
            "{source}"
        );
    }
    for source in [
        "let x=x; function main(){}",
        "typeof x; let x; function main(){}",
        "var early=A; class A {} function main(){}",
    ] {
        assert!(run(source, false).is_err(), "{source}");
        assert!(run(source, true).is_err(), "{source}");
    }
}

#[test]
fn function_and_var_share_storage() {
    let source = "function main() { var f; function f() { return 1; } let read = () => f; f = 7; return read() === 7; }";
    for enabled in [false, true] {
        assert_eq!(run(source, enabled).unwrap(), Value::TRUE);
    }
}

#[test]
fn numeric_properties_and_global_aliases_observe_writes() {
    for body in [
        "const a=o.NaN; o[0/0]=2; return a+o.NaN;",
        "const a=o[0/0]; o.NaN=2; return a+o[0/0];",
        "const a=o.Infinity; o[1/0]=2; return a+o.Infinity;",
        "const a=o[-1]; o['-1']=2; return a+o[-1];",
    ] {
        let source = format!(
            "function f(o){{ {body} }} function main(){{ return f({{NaN:1,Infinity:1,'-1':1}}); }}"
        );
        for enabled in [false, true] {
            assert_eq!(run(&source, enabled).unwrap(), Value::number(3.0), "{body}");
        }
    }
    for body in [
        "const a=x; o.x=2; return a+x;",
        "const a=o.x; x=2; return a+o.x;",
    ] {
        let source = format!(
            "var x=1; function f(o){{ {body} }} function main(){{ return f(globalThis); }}"
        );
        for enabled in [false, true] {
            assert_eq!(run(&source, enabled).unwrap(), Value::number(3.0), "{body}");
        }
    }
    let source = "function f(o){ const a=Math.floor(1.9); o.Math={floor:x=>9}; return a+Math.floor(1.9); } function main(){ return f(globalThis); }";
    for enabled in [false, true] {
        assert_eq!(run(source, enabled).unwrap(), Value::number(10.0));
    }
}

#[test]
fn loop_guards_check_captured_math_callees() {
    for source in [
        "function f(reset){ const floor=Math.floor; reset(); let sum=0; for(let i=0;i<2;++i) sum+=floor(1.9); return sum; } function main(){ const original=Math.floor; Math.floor=x=>99; return f(()=>{Math.floor=original;}); }",
        "function f(reset){ const m=Math; reset(); let sum=0; for(let i=0;i<2;++i) sum+=m.floor(1.9); return sum; } function main(){ const original=Math.floor; Math.floor=x=>99; return f(()=>{Math={floor:original};}); }",
    ] {
        for enabled in [false, true] {
            assert_eq!(run(source, enabled).unwrap(), Value::number(198.0));
        }
    }
    let source = "function f(reset,key){ const m=Math; reset(); let sum=0; for(let i=0;i<2;++i){ sum+=m.floor(1.9); m[key]=x=>99; } return sum; } function main(){ return f(()=>0,'floor'); }";
    for enabled in [false, true] {
        assert_eq!(run(source, enabled).unwrap(), Value::number(100.0));
    }
}

#[test]
fn speculative_reads_preserve_skipped_paths_and_ownership() {
    for source in [
        "function f(n){ const a=[]; let sum=0; for(let i=0;i<n;++i) sum+=a.missing; return sum; } function main(){ return f(0); }",
        "function f(a,n){ let sum=0; for(let i=0;i<n;++i) sum+=a.missing; return sum; } function main(){ return f([],0); }",
        "function f(a,n){ const length=a.length; let sum=0; for(let i=0;i<n;++i) if(i>0) sum+=a.missing; return sum+length; } function main(){ return f([],1); }",
        "function f(a,n){ let sum=0; for(let i=0;i<n;++i) sum+=a.missing; return sum; } function main(){ return f([],1); }",
        "function f(a,n){ let sum=0; for(let i=0;i<n;++i) sum+=a[0]; return sum; } function main(){ return f(null,0); }",
        "function f(a,n){ let value; for(let i=0;i<n;++i) value=a.x; return value; } function main(){ return f({},2)===undefined; }",
        "function f(a,n){ let value; for(let i=0;i<n;++i) value=a.x; return value; } function main(){ const value=f({x:{v:7}},2); return value.v; }",
        "function f(a,k,n){ let value; for(let i=0;i<n;++i) value=a[k]; return value; } function main(){ return f({x:'kept'},'x',2)==='kept'; }",
        "function f(a,k,n){ let value; for(let i=0;i<n;++i) value=a[k]; return value; } function main(){ return f('abc',1,2)==='b'; }",
    ] {
        assert_eq!(run(source, true), run(source, false), "{source}");
    }
}

#[test]
fn typed_loop_stores_preserve_integer_conversion_boundaries() {
    let source = "function fill(a,value){ for(let i=0;i<a.length;++i) a[i]=value; return a[0]; } function main(){ const a=new Uint8Array(2); const b=new Int32Array(2); return fill(a,9223372036854775808)===0 && fill(b,9223372036854775808)===0 && fill(a,-9223372036854775808)===0 && fill(a,9223372036854774784)===0 && fill(a,255)===255; }";
    for enabled in [false, true] {
        assert_eq!(run(source, enabled).unwrap(), Value::TRUE);
    }
}

#[test]
fn loop_versions_keep_callee_reads_used_after_the_loop() {
    for source in [
        "function f(h,xs){ h(); let s=0; for(var i=0;;++i){ var g=Math.floor; s+=g(xs[i]); if(i>0) break; } return g===Math.floor && s===3; } function main(){ return f(()=>0,[1.5,2.5]); }",
        "function f(h,xs){ h(); let s=0; for(var i=0;;++i){ var m=Math; s+=m.floor(xs[i]); if(i>0) break; } return m===Math && s===3; } function main(){ return f(()=>0,[1.5,2.5]); }",
    ] {
        for enabled in [false, true] {
            assert_eq!(run(source, enabled).unwrap(), Value::TRUE, "{source}");
        }
    }
}

#[test]
fn bounded_typed_reads_match_generic_execution() {
    let kinds = [
        "Uint8Array",
        "Uint16Array",
        "Uint32Array",
        "Int8Array",
        "Int16Array",
        "Int32Array",
        "Float32Array",
        "Float64Array",
    ];
    let mut bodies: Vec<String> = kinds
        .iter()
        .map(|kind| {
            format!(
                "const a = {kind}.from([-1, 300, 2.5, 4294967295, -129, 70000, NaN]);
                 const b = new {kind}(7);
                 b[3] = 0.1;
                 for (let k = 0; k < a.length; ++k) {{ const v = a[k] + b[k]; out = out * 3 + (v === v ? v : 7); }}"
            )
        })
        .collect();
    bodies.extend(
        [
            "const a = Int32Array.from([1, 2, 3]); const b = Int32Array.from([4, 5]); for (let k = 0; k < a.length; ++k) { const v = b[k]; out = out * 3 + (v === v ? v : 7); }",
            "const a = Int32Array.from([1, 2, 3]); for (let k = -1; k < a.length; ++k) { const v = a[k]; out = out * 3 + (v === v ? v : 7); }",
            "const a = Int32Array.from([1, 2, 3]); for (let k = 0; k <= a.length; ++k) { const v = a[k]; out = out * 3 + (v === v ? v : 7); }",
            "const a = Int32Array.from([1, 2, 3]); for (let k = 0; k < a.length; k += 2) { const v = a[k]; out = out * 3 + (v === v ? v : 7); }",
            "const a = Int32Array.from([1, 2, 3]); for (let k = 0; k < 5; k += 1) { const v = a[k]; out = out * 3 + (v === v ? v : 7); }",
            "const a = Int32Array.from([1, 2, 3]); for (let k = 0; k < a.length; ++k) { { const v = a[k]; out = out * 3 + (v === v ? v : 7); } if (k == 0) k = 5; }",
            "const a = Int16Array.from([1.5, -2, 3]); for (let k = 0; k < a.length; ++k) { const v = a[k]; out = out * 3 + (v === v ? v : 7); }",
        ]
        .map(String::from),
    );
    for body in bodies {
        let source = format!(
            "Int16Array = Float64Array;
             function main() {{ let out = 0; {body} return out; }}"
        );
        let expected = run(&source, false);
        assert!(expected.is_ok(), "{body}: {expected:?}");
        assert_eq!(run(&source, true), expected, "{body}");
    }
}

#[test]
fn constructor_resolution_preserves_identity_prototypes_and_argument_order() {
    let source = r#"
        function make(C, value) { return new C(value); }
        function spread(C, values) { return new C(...values); }
        function change(C, prototype) { return new C((C.prototype = prototype, 7)); }
        function First(value) { this.value = value; }
        function Second(value) { this.value = value * 2; }
        function Override() { return {value: 99}; }
        function main() {
            let C = First;
            let sum = 0;
            for (let i = 0; i < 4; ++i) {
                C = i % 2 ? First : Second;
                C.prototype = {tag: i};
                let instance = make(C, 3);
                let expanded = spread(C, [3]);
                sum += instance.value + expanded.value + instance.tag + expanded.tag;
            }
            let original = {tag: 10};
            First.prototype = original;
            let before = make(First, 1);
            let after = change(First, {tag: 20});
            First.prototype = 0;
            let primitive = make(First, 2);
            return sum === 48 && before.tag === 10 && after.tag === 20 && after.value === 7 &&
                primitive.value === 2 && make(Override, 0).value === 99 &&
                make(Array, 3).length === 3 && spread(Uint8Array, [2]).length === 2;
        }
    "#;
    for ssa_opt in [false, true] {
        assert_eq!(run(source, ssa_opt).unwrap(), Value::TRUE);
    }
}

#[test]
fn constructor_resolution_keeps_failures_at_the_construction_site() {
    for source in [
        "function make(C){return new C();} function main(){return make(()=>7);}",
        "function make(C, args){return new C(...args);} function main(){return make(null, []);}",
        "function make(C, args){return new C(...args);} function main(){return make(Array, null);}",
    ] {
        for ssa_opt in [false, true] {
            let mut engine = Engine::with_options(EngineOptions {
                ssa_opt,
                ..Default::default()
            })
            .unwrap();
            engine.compile(source, "constructor-errors.js").unwrap();
            assert!(engine.call("main", &[]).is_err());
            engine
                .compile(
                    "function good(){return new Array(2).length;}",
                    "constructor-recovery.js",
                )
                .unwrap();
            assert_eq!(
                engine.call("good", &[]).unwrap().value(),
                Value::number(2.0)
            );
        }
    }
    let source = "function C(){throw 7;} function make(C){return new C();} function main(){try{make(C);}catch(e){return e === 7;} return false;}";
    for ssa_opt in [false, true] {
        assert_eq!(run(source, ssa_opt).unwrap(), Value::TRUE);
    }
}

#[test]
fn expanded_construction_preserves_return_values_and_classes() {
    let source = r#"
        function C(x) { this.value = x; return 42; }
        function Replacement() { return {value: 99}; }
        function NullReturn() { this.value = 8; return null; }
        class Klass { constructor(x) { this.value = x; } }
        function main() {
            C.prototype = {tag: 3};
            Replacement.prototype = {};
            NullReturn.prototype = {};
            let sum = 0;
            for(let i = 0; i < 20; ++i) {
                let instance = new C(i);
                sum += instance.value + instance.tag;
            }
            return sum === 250 && new Replacement().value === 99 &&
                new NullReturn().value === 8 && new Klass(7).value === 7 &&
                new Klass(...[9]).value === 9;
        }
    "#;
    for ssa_opt in [false, true] {
        assert_eq!(run(source, ssa_opt).unwrap(), Value::TRUE);
    }
}

#[test]
fn builtin_property_caches_preserve_kinds_identity_and_mutable_methods() {
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                r#"
            var receiver = [1, 2, 3];
            function read() { return receiver.slice; }
            function array() { receiver = [4, 5, 6]; }
            function string() { receiver = "abc"; }
            function object() { receiver = {slice: 7}; }
            function replace() { receiver.slice = 9; }
            function missing() { receiver = null; }
            function from(constructor) { return constructor.from; }
            function arrayConstructor() { return Array; }
            function typedConstructor() { return Uint8Array; }
            "#,
                "builtin_property_cache.js",
            )
            .unwrap();
        let array_method = engine.call("read", &[]).unwrap();
        for _ in 0..3 {
            assert_eq!(
                engine.call("read", &[]).unwrap().value(),
                array_method.value()
            );
        }
        engine.call("string", &[]).unwrap();
        let string_method = engine.call("read", &[]).unwrap();
        assert_ne!(string_method.value(), array_method.value());
        engine.call("array", &[]).unwrap();
        engine.drain_disposals(1000);
        assert_eq!(
            engine.call("read", &[]).unwrap().value(),
            array_method.value()
        );
        engine.call("string", &[]).unwrap();
        assert_eq!(
            engine.call("read", &[]).unwrap().value(),
            string_method.value()
        );
        engine.call("object", &[]).unwrap();
        assert_eq!(
            engine.call("read", &[]).unwrap().value(),
            Value::number(7.0)
        );
        engine.call("replace", &[]).unwrap();
        assert_eq!(
            engine.call("read", &[]).unwrap().value(),
            Value::number(9.0)
        );
        engine.call("missing", &[]).unwrap();
        assert!(engine.call("read", &[]).is_err());
        let array = engine.call("arrayConstructor", &[]).unwrap();
        let typed = engine.call("typedConstructor", &[]).unwrap();
        let array_from = engine.call("from", &[array.value()]).unwrap();
        let typed_from = engine.call("from", &[typed.value()]).unwrap();
        assert_ne!(array_from.value(), typed_from.value());
        assert_eq!(
            engine.call("from", &[array.value()]).unwrap().value(),
            array_from.value()
        );
    }
}

#[test]
fn cross_source_inlining_preserves_writes_and_global_rebinding() {
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine.compile("var count=0; function C(x){count+=1;this.value=x;} function touch(o){o.value+=2;return o;}", "definitions.js").unwrap();
        engine.compile("function main(){C.prototype={};let a=new C(3);return touch(a).value+count;} function replace(){C=function(x){this.value=x*10;};touch=function(o){o.value+=20;return o;};}", "caller.js").unwrap();
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::number(6.0)
        );
        engine.call("replace", &[]).unwrap();
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::number(51.0)
        );
        engine
            .compile(
                "C=function(x){this.value=x*100;}; touch=function(o){return o;};",
                "replacement.js",
            )
            .unwrap();
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::number(301.0)
        );
    }
}
