use js_engine::{Engine, EngineOptions, Value};

fn run(source: &str, enabled: bool) -> Result<Value, String> {
    let mut engine = Engine::with_options(EngineOptions {
        ssa_opt: enabled,
        ..Default::default()
    })
    .unwrap();
    engine
        .compile(source, "specialization.js")
        .map_err(|error| error.to_string())?;
    engine
        .call("main", &[])
        .map(|value| value.value())
        .map_err(|error| error.to_string())
}

#[test]
fn numeric_inline_and_generic_arguments_agree() {
    let source = r#"
        function sum(a, b) { return a + b; }
        function square(x) { return x * x; }
        function choose(x) { if (x < 0) return -x; return x + 1; }
        function apply(x) { return sum(square(x), choose(x)); }
        function main() {
            let values = [0, -0, 3, -4, 1.5, NaN, Infinity, -Infinity, "2", null, undefined, false];
            let result = 0;
            for (let i = 0; i < values.length; ++i) {
                let value = apply(values[i]);
                if (value === value && value !== Infinity && value !== -Infinity)
                    result += +value;
            }
            return result;
        }
    "#;
    assert_eq!(run(source, true), run(source, false));
    assert!(run(source, true).is_ok());
}

#[test]
fn callee_guards_preserve_lookup_and_argument_evaluation_order() {
    for body in [
        "return square((square = replacement, 3)) * 100 + square(4);",
        "globalThis.square = replacement; return square(4);",
        "let result = square(3, square = replacement); return result * 100 + square(4);",
        "let result = square(3); square = Math.abs; return result * 100 + square(-4);",
    ] {
        let source = format!(
            "function square(x) {{ return x * x; }}
             function replacement(x) {{ return this === undefined ? x + 10 : -100; }}
             function main() {{ {body} }}",
        );
        let expected = run(&source, false);
        assert!(expected.is_ok());
        assert_eq!(run(&source, true), expected, "{body}");
    }
}

#[test]
fn fallback_returns_keep_heap_values_alive() {
    let source = r#"
        function first(x) { return x; }
        function square(x) { return x * x; }
        function replacement(x) { return { value: x }; }
        function main() {
            let total = first(3);
            square = replacement;
            for (let i = 0; i < 40; ++i) {
                let value = square(i);
                total += first(value).value;
            }
            return total;
        }
    "#;
    assert_eq!(run(source, true), Ok(Value::number(783.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn missing_arguments_and_fallback_errors_agree() {
    for body in [
        "return square();",
        "return square({});",
        "square = 0; return square(3);",
        "square = function(x) { throw x; }; return square(3);",
        "return plus();",
        "return plus(3);",
    ] {
        let source = format!(
            "function square(x) {{ return x * x; }}
             function plus(x = 4) {{ return x + 1; }}
             function main() {{ {body} }}",
        );
        assert_eq!(run(&source, true), run(&source, false), "{body}");
    }
}

#[test]
fn loop_guard_exits_preserve_progress_captures_and_mutated_globals() {
    for enabled in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt: enabled,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                r#"
            var change;
            function method(x) { return Math.floor(x); }
            var points = [{method:method}, {method:method}, {method:method}];
            function sum() {
                let count = 0;
                change = x => { count = x; };
                let total = 0;
                for (let i = 0; i < 3; ++i) {
                    ++count;
                    total += points[i].method(i + 0.9);
                    total += Math.floor(i + 0.9);
                }
                return count * 1000 + total;
            }
            function replace() {
                points[1].method = x => {
                    change(20);
                    Math.floor = x => 90;
                    return 7;
                };
            }
            "#,
                "loop_guard_exit.js",
            )
            .unwrap();
        for _ in 0..3 {
            assert_eq!(
                engine.call("sum", &[]).unwrap().value(),
                Value::number(3006.0)
            );
        }
        engine.call("replace", &[]).unwrap();
        assert_eq!(
            engine.call("sum", &[]).unwrap().value(),
            Value::number(21277.0)
        );
    }
}

#[test]
fn loop_guard_exits_keep_heap_values_alive() {
    for enabled in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt: enabled,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                r#"
                function identity(x) { return x; }
                var points = [{method:identity}, {method:identity}, {method:identity}];
                var saved;
                function collect() {
                    let value = {v:-1};
                    const read = () => value;
                    let total = 0;
                    for (let i = 0; i < 3; ++i) {
                        const previous = value;
                        value = points[i].method({v:i});
                        total += previous.v + Math.floor(i + 0.9);
                    }
                    saved = read;
                    return total + value.v + read().v;
                }
                function replace() {
                    points[1].method = x => {
                        Math.floor = x => 10;
                        return {v:x.v + 10};
                    };
                }
                function retained() { return saved().v; }
                "#,
                "loop_guard_heap_exit.js",
            )
            .unwrap();
        for _ in 0..3 {
            assert_eq!(
                engine.call("collect", &[]).unwrap().value(),
                Value::number(7.0)
            );
        }
        engine.call("replace", &[]).unwrap();
        assert_eq!(
            engine.call("collect", &[]).unwrap().value(),
            Value::number(34.0)
        );
        engine.drain_disposals(1000);
        assert_eq!(
            engine.call("retained", &[]).unwrap().value(),
            Value::number(2.0)
        );
    }
}

#[test]
fn local_closures_share_mutable_captures_and_preserve_heap_returns() {
    for (source, expected) in [
        (
            "function main(){ let total=1; const add=x=>{ total+=x; return total; }; const set=function(x){ total=x; return total; }; let a=add(2); set(10); return a*100+add(4); }",
            314.0,
        ),
        (
            "function main(){ let saved={v:7}; const read=()=>saved; const kept=read(); saved=null; return kept.v; }",
            7.0,
        ),
        (
            "function main(){ let total=0; const f=x=>{ let y=x+1; { const z=y*2; total+=z; if(x) return total; } return -1; }; f(0); return f(2); }",
            8.0,
        ),
        (
            "function use(x){ const read=()=>arguments[0]; return read(); } function main(){ return use(7); }",
            7.0,
        ),
    ] {
        let optimized = run(source, true);
        let generic = run(source, false);
        assert_eq!(optimized, generic, "{source}");
        assert_eq!(optimized, Ok(Value::number(expected)));
    }
}

#[test]
fn closure_receivers_remain_lexical_for_arrows_and_dynamic_for_functions() {
    let source = r#"
        function C(value) { this.value = value; }
        C.prototype.run = function() {
            const f = x => this.value + x;
            const ordinary = function() { return this; };
            const mapped = [1, 2].map(f);
            return ordinary() === undefined && mapped[0] === 5 && f(3) === 7;
        };
        function main() { return new C(4).run(); }
    "#;
    assert_eq!(run(source, true), Ok(Value::TRUE));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn map_observes_mutation_and_preserves_snapshot_length_and_holes() {
    let source = r#"
        function main() {
            let calls = 0;
            const values = [1, 2, 3, 4];
            const mapped = values.map((value, index, array) => {
                ++calls;
                if (index === 0) {
                    array.length = 2;
                    array[1] = 8;
                }
                return value + index;
            });
            const growing = [1, 2];
            const second = growing.map((value, index, array) => {
                array.push(9);
                return value * 2;
            });
            return calls === 2 && mapped.length === 4 && mapped[0] === 1 && mapped[1] === 9 &&
                !(2 in mapped) && !(3 in mapped) && second.length === 2 && growing.length === 4;
        }
    "#;
    assert_eq!(run(source, true), Ok(Value::TRUE));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn map_keeps_callback_values_alive_across_source_replacement() {
    let source = r#"
        function main() {
            const values = [{v: 7}, {v: 9}];
            let saved;
            const result = values.map((value, index, array) => {
                array[index] = null;
                saved = value;
                return value;
            });
            return result[0].v * 10 + saved.v;
        }
    "#;
    assert_eq!(run(source, true), Ok(Value::number(79.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn map_fallbacks_and_callback_errors_match_generic_execution() {
    for body in [
        "const o={map:function(f){return f(4);}}; let n=2; return o.map(x=>x+n);",
        "const o={map:[].map}; return o.map(x=>x);",
        "let n=0; const a=[1,2].map(x=>x, undefined, n++); return a.length+n;",
        "return [1].map(x=>x, {});",
        "return [1,2].map(x=>{ throw x; });",
        "return [1].map(x=>null.missing);",
        "class C {} return [1].map(C);",
        "class C {} return [].map(C).length;",
        "const f=()=>{return x;}; const y=f(); let x; return y;",
    ] {
        let source = format!("function main(){{ {body} }}");
        assert_eq!(run(&source, true), run(&source, false), "{body}");
    }
}

#[test]
fn recursive_create_terrain_maps_strings_and_nested_arrays() {
    let library =
        include_str!("../../../../binaries/data/mods/public/maps/random/rmgen/library.js");
    let start = library.find("function createTerrain(terrain)").unwrap();
    let end = start + library[start..].find("\n}").unwrap() + 2;
    let source = format!(
        r#"
        var TERRAIN_SEPARATOR = "|";
        function SimpleTerrain(texture, template) {{ this.texture=texture; this.template=template; }}
        function RandomTerrain(terrains) {{ this.terrains=terrains; }}
        {}
        function main() {{
            const result=createTerrain(["grass", ["sand|tree", "rock"]]);
            return result.terrains[0].texture === "grass" &&
                result.terrains[1].terrains[0].template === "tree" &&
                result.terrains[1].terrains[1].texture === "rock";
        }}
    "#,
        &library[start..end]
    );
    assert_eq!(run(&source, true), Ok(Value::TRUE));
    assert_eq!(run(&source, true), run(&source, false));
}

#[test]
fn fallback_closure_materialization_preserves_identity_across_loop_iterations() {
    let source = r#"
        function main() {
            let previous;
            const object = {map: function(callback) {
                const same = previous === undefined || previous === callback;
                previous = callback;
                return same;
            }};
            const callback = value => value;
            for (let i = 0; i < 4; ++i)
                if (!object.map(callback)) return false;
            previous = undefined;
            for (let i = 0; i < 4; ++i) {
                const next = value => value;
                if (i && object.map(next)) return false;
                if (!i) object.map(next);
            }
            return true;
        }
    "#;
    assert_eq!(run(source, true), Ok(Value::TRUE));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn loop_hoisted_guards_match_generic_execution() {
    for body in [
        "let sum = 0; for (let i = 0; i < 20; ++i) sum += square(k) + square(i); return sum;",
        "let sum = 0; for (let i = 0; i < 20; ++i) { sum += square(i); if (i == 7) square = replacement; } return sum;",
        "let sum = 0; for (let i = 0; i < 20; ++i) sum += square(i) + square(missing); return sum;",
        "let sum = 0; for (let i = 0; i < 20; ++i) sum += square(i) + (missing === undefined ? 1 : 0); return sum;",
        "let sum = 0; for (let i = 0; i < 20; ++i) sum += square(text) + square(i); return sum;",
    ] {
        let source = format!(
            "function square(x) {{ return x * x; }}
             function replacement(x) {{ return x + 10; }}
             let k = 3; let text = \"4\";
             function main() {{ {body} }}",
        );
        let expected = run(&source, false);
        assert_eq!(run(&source, true), expected, "{body}");
    }
}
