mod common;

use js_engine::{Engine, EngineOptions, Value};

#[test]
fn array_feedback_preserves_holes_bounds_and_receiver_changes() {
    let source = r#"
        function read(array, index) { return array[index]; }
        function main() {
            const runner = {read};
            const object = {};
            const array = new Array(5);
            array[0] = object;
            array[1] = 2;
            for (let i = 0; i < 1100; ++i)
                runner.read(array, 0);
            if (runner.read(array, 0) !== object)
                return false;
            array[4] = 9;
            if (runner.read(array, 2) !== undefined || runner.read(array, 4) !== 9)
                return false;
            array[1] = 4;
            array[2] = 5;
            array[3] = 6;
            if (runner.read(array, 2) !== 5 || runner.read(array, 10) !== undefined)
                return false;
            array.length = 1;
            array[1] = 7;
            if (runner.read(array, 1) !== 7 || runner.read(array, 4) !== undefined)
                return false;
            const typed = new Float64Array(2);
            typed[0] = 17;
            return runner.read(array, -0) === object
                && runner.read(typed, 0) === 17
                && runner.read({0: 19}, 0) === 19;
        }
    "#;
    for osr in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            osr,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "array-feedback.js").unwrap();
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::boolean(true)
        );
    }
}

#[test]
fn array_feedback_preserves_non_index_numeric_properties() {
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    engine
        .compile(
            "function make(){return [7];} function read(array,index){return array[index];}",
            "array-numeric-properties.js",
        )
        .unwrap();
    let array = engine.call("make", &[]).unwrap();
    for _ in 0..1100 {
        assert_eq!(
            engine
                .call("read", &[array.value(), Value::number(0.0)])
                .unwrap()
                .value(),
            Value::number(7.0)
        );
    }
    for index in [-1.0, 0.5, f64::NAN, f64::INFINITY, 4294967295.0] {
        assert_eq!(
            engine
                .call("read", &[array.value(), Value::number(index)])
                .unwrap()
                .value(),
            Value::UNDEFINED
        );
    }
}

#[test]
fn capture_reads_preserve_changes_across_calls_and_cell_writes() {
    let source = r#"
        function main() {
            let width = 4;
            function change() { width = 2; }
            function run(array, callback) {
                let sum = 0;
                for (let i = 0; i < width; ++i) {
                    array[i] = i;
                    sum += width;
                    callback();
                }
                return sum;
            }
            function keep() {}
            const runner = {run};
            const array = [0, 0, 0, 0];
            for (let i = 0; i < 1100; ++i)
                runner.run(array, keep);
            const changed = runner.run(array, change);
            width = 3;
            return changed === 6 && runner.run(array, keep) === 9;
        }
    "#;
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    engine.compile(source, "cell-aliasing.js").unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::boolean(true)
    );
}

#[test]
fn loop_carried_numeric_feedback_preserves_initial_values() {
    let source = r#"
        function iterate(holder, n) {
            let value = holder.value;
            for (let i = 0; i < n; ++i)
                value = (value + 1) * 0.5;
            return value;
        }
        function main() {
            const runner = {call: iterate};
            const holder = {value: 2};
            for (let i = 0; i < 1100; ++i)
                runner.call(holder, 4);
            const numeric = runner.call(holder, 4);
            holder.value = "2";
            const empty = runner.call(holder, 0);
            const string = runner.call(holder, 1);
            holder.value = false;
            const boolean = runner.call(holder, 1);
            const object = {};
            holder.value = object;
            return numeric === 1.0625 && empty === "2" && string === 10.5
                && boolean === 0.5 && runner.call(holder, 0) === object;
        }
    "#;
    for osr in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            osr,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "loop-carried-feedback.js").unwrap();
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::boolean(true)
        );
    }
}

#[test]
fn floating_consumers_preserve_numbers_after_integer_feedback() {
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    engine
        .compile(
            "function make(value){return {value};}
         function scaled(object){return object.value*0.5;}
         function change(object,value){object.value=value;}",
            "floating_consumers.js",
        )
        .unwrap();
    let object = engine.call("make", &[Value::number(7.0)]).unwrap();
    for _ in 0..16 {
        assert_eq!(
            engine.call("scaled", &[object.value()]).unwrap().value(),
            Value::number(3.5)
        );
    }
    for number in [-0.0, 1.25, -3.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        engine
            .call("change", &[object.value(), Value::number(number)])
            .unwrap();
        let actual = engine.call("scaled", &[object.value()]).unwrap().value();
        assert_eq!(actual, Value::number(number * 0.5));
    }
}

#[test]
fn numeric_loop_continuations_preserve_accumulators_and_shared_bounds() {
    let source = r#"
        function accumulate(rows, offsets, bound) {
            if (bound < 0)
                offsets[0] = 0;
            let sum = 0;
            let weighted = 0;
            for (let k = 0; k < bound; ++k) {
                const x = offsets[k];
                sum += rows[x][0];
                weighted += rows[x][0] / (k + 1);
            }
            return sum + weighted;
        }
        function main() {
            const runner = {call: accumulate};
            const rows = [Float64Array.from([2]), Float64Array.from([4]), Float64Array.from([8])];
            const offsets = Int32Array.from([0, 1, 2, 0, 1, 2]);
            for (let i = 0; i < 40; ++i)
                runner.call(rows, offsets, 6);
            rows[2] = [8];
            return runner.call(rows, offsets, 6);
        }
    "#;
    let execute = |enabled| {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt: enabled,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "numeric_continuation.js").unwrap();
        engine.call("main", &[]).unwrap().value()
    };
    assert_eq!(execute(true), Value::number(37.3));
    assert_eq!(execute(true), execute(false));
}

#[test]
fn local_numeric_assumptions_resume_after_effects_and_loaded_values() {
    let source = r#"
        function accumulate(values, state) {
            let total = 0;
            for (let row = 0; row < 2; ++row)
                for (let i = 0; i < 4;) {
                    state.count++;
                    total += values[i++];
                }
            return total;
        }
        function main() {
            const runner = {call: accumulate};
            const state = {count: 0};
            for (let i = 0; i < 40; ++i)
                runner.call([1, 2, 3, 4], state);
            state.count = 0;
            const mixed = runner.call([1, 2, "x", 4], state);
            const effects = state.count;
            const fractional = runner.call([1, 2, 0.5, 4], state);
            const missing = runner.call([1, 2], state);
            return mixed === "3x412x4" && effects === 8
                && fractional === 15 && missing !== missing;
        }
    "#;
    for ssa_opt in [false, true] {
        for osr in [false, true] {
            let mut engine = Engine::with_options(EngineOptions {
                ssa_opt,
                osr,
                background_compile: false,
                ..Default::default()
            })
            .unwrap();
            engine.compile(source, "local-assumptions.js").unwrap();
            assert_eq!(
                engine.call("main", &[]).unwrap().value(),
                Value::boolean(true)
            );
        }
    }
}

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
fn polymorphic_calls_preserve_arguments_receivers_and_owned_returns() {
    let source = r#"
        function initial(a, b) { return {value: this.base + a + b}; }
        function fewer(a) { return {value: this.base + a}; }
        function more(a, b, c = 10) { return {value: this.base + a + b + c}; }
        function many(a, b, c, d, e, f, g, h, i, j = 20) {
            return {value: this.base + a + b + j};
        }
        function rest(...args) { return {value: this.base + args.length}; }
        function make(n) { return function(a, b) { return {value: this.base + n + a + b}; }; }
        function makeArrow() { return (a, b) => ({value: this.base + a + b}); }
        function invoke(o) { return o.run(2, 3); }
        function main() {
            let object = {base: 100, run: initial};
            let arrow = {base: 200, make: makeArrow}.make();
            let targets = [initial, fewer, more, many, rest, make(30), arrow];
            let total = 0;
            for (let i = 0; i < 40; ++i) {
                for (let j = 0; j < targets.length; ++j) {
                    object.run = targets[j];
                    let result = invoke(object);
                    total += result.value;
                }
            }
            return total;
        }
    "#;
    assert_eq!(run(source, true), Ok(Value::number(35560.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn polymorphic_calls_preserve_exceptions_and_recovery() {
    let source = r#"
        function initial(x) { return x + 1; }
        function throwing(x) { throw {value: x + 10}; }
        function invoke(o, x) { return o.run(x); }
        function main() {
            let object = {run: initial};
            let total = 0;
            for (let i = 0; i < 20; ++i) {
                object.run = initial;
                total += invoke(object, i);
                object.run = throwing;
                try { invoke(object, i); }
                catch (error) { total += error.value; }
                object.run = Math.abs;
                total += invoke(object, -i);
            }
            return total;
        }
    "#;
    assert_eq!(run(source, true), Ok(Value::number(790.0)));
    assert_eq!(run(source, true), run(source, false));
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
    let library = common::map_library("library.js");
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

#[test]
fn inlined_closures_preserve_cells_and_lexical_receivers() {
    let source = r#"
        function Make(x = 3) {
            this.value = x * 10;
            let result = [];
            for (let i = 0; i < 3; ++i)
                result.push(() => this.value + x + i);
            x += 2;
            return result;
        }
        function main() {
            Make.prototype = {};
            let a = new Make();
            let b = new Make(1);
            return a[0]() + a[1]() * 10 + a[2]() * 100 + b[2]() * 1000;
        }
    "#;
    assert_eq!(run(source, false), Ok(Value::number(19095.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn spread_constructor_inlining_preserves_arguments_and_fallbacks() {
    let source = r#"
        let log = 0;
        function C(a, b = 4) { this.x = a.value + b; a.value = 90; }
        function object(a) { return { x: a }; }
        function main() {
            C.prototype = {};
            let a = { value: 2 };
            let first = new C(...[a, 3, (log = 7)]);
            let second = new C(...[{ value: 5 }]);
            let native = new Array(...[2]);
            C = object;
            let third = new C(...[11]);
            return first.x * 10000 + second.x * 1000 + native.length * 100 + third.x * 10 + log;
        }
    "#;
    assert_eq!(run(source, false), Ok(Value::number(59317.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn cross_source_inline_closures_use_their_own_code_and_receiver() {
    for enabled in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt: enabled,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                "function C(x){this.value=x;this.get=()=>this.value;}",
                "first.js",
            )
            .unwrap();
        engine
            .compile(
                "function main(){C.prototype={};let o=new C(10);let f=o.get;return f();}",
                "second.js",
            )
            .unwrap();
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::number(10.0)
        );
    }
}

#[test]
fn recursively_inlined_terrain_preserves_constructor_replacements() {
    let source = r#"
        function SimpleTerrain(a) { this.x = a.length; }
        function createTerrain(t) {
            return typeof t === 'string' ? new SimpleTerrain(...[t]) : t.map(x => createTerrain(x));
        }
        function replacement(a) { return { x: a.length + 10 }; }
        function main() {
            SimpleTerrain.prototype = {};
            let a = createTerrain(['a', ['bc', ['def']]]);
            SimpleTerrain = replacement;
            let b = createTerrain(['abcd', ['ef']]);
            return a[0].x + a[1][0].x + a[1][1][0].x + b[0].x + b[1][0].x;
        }
    "#;
    assert_eq!(run(source, false), Ok(Value::number(32.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn inlined_spread_validates_and_snapshots_inputs() {
    for function in ["function C(a){this.a=a;}", "function C(){this.a=1;}"] {
        let source = format!(
            "{function} function main(){{C.prototype={{}};return new C(...{{0:42,length:1}}).a;}}"
        );
        for enabled in [false, true] {
            assert!(run(&source, enabled).is_err(), "{source}");
        }
    }
    for (source, expected) in [
        (
            "function C(a,b=7){this.a=(a===undefined?2:a)+b;} function main(){C.prototype={};return new C(...Array(1)).a;}",
            9.0,
        ),
        (
            "function C(a,b=7){this.a=a+b;} function main(){C.prototype={};return new C(...[3]).a;}",
            10.0,
        ),
        (
            "function C(a,b){this.a=a.length*10+b.length;} function main(){C.prototype={};return new C(...'é🌲').a;}",
            11.0,
        ),
        (
            "let source; function C(a=(source.length=0),b){this.a=b.x;} function main(){C.prototype={};source=[undefined,{x:19},42];return new C(...source).a;}",
            19.0,
        ),
        (
            "function C(a){this.a=a;} function R(a){return {a:a+5};} function main(){C.prototype={};let x=new C(...[3]); C=R;return x.a+new C(...[4]).a+new Array(...[2]).length;}",
            14.0,
        ),
    ] {
        for enabled in [false, true] {
            assert_eq!(
                run(source, enabled),
                Ok(Value::number(expected)),
                "{source}, ssa={enabled}"
            );
        }
    }
}

#[test]
fn string_split_specialization_preserves_replacements_and_errors() {
    for source in [
        "function main(){return 'a,b'.split(',').length;}",
        "function split(x){return [x,x];} function main(){let o={split:split};return o.split(3).length;}",
        "function main(){return 'x'.split(1).length;}",
        "function main(){return 'x'.split('x',1).length;}",
    ] {
        assert_eq!(run(source, true), run(source, false), "{source}");
    }
}

#[test]
fn array_read_feedback_preserves_layouts_bounds_and_fallbacks() {
    for enabled in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt: enabled,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine
            .compile(
                r#"
            function read(a, k) { return a[k]; }
            const runner = {call: read};
            function warm() {
                const values = [10, 20, 30];
                for (let i = 0; i < 32; ++i)
                    runner.call(values, i % 3);
            }
            function main() {
                const array = [10, 20, 30];
                const holey = [10, 20, 30].map((value, index, source) => {
                    if (index === 0) source.length = 1;
                    return value;
                });
                const object = {"0": 7, "0.5": 9};
                const owned = {value: 42};
                const references = [owned];
                if (runner.call(array, -0) !== 10 || runner.call(array, 2) !== 30 ||
                    runner.call(array, 3) !== undefined || runner.call(array, "1") !== 20 ||
                    runner.call(holey, 0) !== 10 || runner.call(holey, 1) !== undefined ||
                    runner.call(object, 0) !== 7 || runner.call(object, 0.5) !== 9 ||
                    runner.call(Uint8Array.from([4, 5]), 1) !== 5 ||
                    runner.call("abc", 1) !== "b" || runner.call(references, 0) !== owned)
                    return false;
                array.length = 1;
                if (runner.call(array, 2) !== undefined) return false;
                array.push(40);
                return runner.call(array, 1) === 40;
            }
            function fractional() { return runner.call([10], 0.5); }
            function negative() { return runner.call([10], -1); }
            function nan() { return runner.call([10], NaN); }
            function infinite() { return runner.call([10], Infinity); }
            function huge() { return runner.call([10], 1e30); }
            function nullReceiver() { return runner.call(null, 0); }
        "#,
                "array-read-feedback.js",
            )
            .unwrap();
        engine.call("warm", &[]).unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        for name in ["fractional", "negative", "nan", "infinite", "huge"] {
            assert_eq!(
                engine.call(name, &[]).unwrap().value(),
                Value::UNDEFINED,
                "{name}, optimized={enabled}"
            );
        }
        assert!(engine.call("nullReceiver", &[]).is_err());
    }
}

#[test]
fn integer_feedback_guards_preserve_numeric_boundaries() {
    let source = r#"
        function work(a) {
            let result = a;
            for (let i = 0; i < 20; ++i)
                result = a * 0.5;
            return result;
        }
        function bits(a) { return a | 0; }
        function warm() {
            const h = {call: work, bits: bits};
            for (let i = 0; i < 100; ++i) {
                h.call(4);
                h.bits(4);
            }
        }
        function check(which) {
            const h = {call: work, bits: bits};
            if (which === 0) return 1 / h.call(-0) === -Infinity;
            if (which === 1) { const x = h.call(0 / 0); return x !== x; }
            if (which === 2) return h.call(Infinity) === Infinity;
            if (which === 3) return h.call(-Infinity) === -Infinity;
            if (which === 4) return h.call(0.5) === 0.25;
            if (which === 5) return h.call(9007199254740991) === 4503599627370495.5;
            if (which === 6) return h.call(9007199254740992) === 4503599627370496;
            if (which === 7) return h.call('6') === 3;
            if (which === 8) return h.bits(4294967297) === 1;
            if (which === 9) return h.bits(-1.5) === -1;
            if (which === 10) return h.bits(Infinity) === 0;
            return h.bits(0 / 0) === 0;
        }
    "#;
    for case in 0..12 {
        let mut options = vec![EngineOptions {
            background_compile: false,
            ..Default::default()
        }];
        #[cfg(target_arch = "aarch64")]
        options.push(EngineOptions {
            backend: js_engine::Backend::NativeOnly,
            background_compile: false,
            ..Default::default()
        });
        for options in options {
            let mut engine = Engine::with_options(options).unwrap();
            engine.compile(source, "integer-feedback.js").unwrap();
            engine.call("warm", &[]).unwrap();
            assert_eq!(
                engine
                    .call("check", &[Value::number(case as f64)])
                    .unwrap()
                    .value(),
                Value::TRUE,
                "case {case}"
            );
        }
    }
}

#[test]
fn cold_array_fallback_preserves_heap_values_and_numeric_coercion() {
    let source = r#"
        function update(input, output) {
            let total = 0;
            for (let i = 0; i < 4; ++i) {
                output[i] = input[i];
                total += input[i];
            }
            return total;
        }
        function main() {
            const runner = {call: update};
            const output = [0, 0, 0, 0];
            for (let i = 0; i < 40; ++i)
                runner.call([1, 2, 3, 4], output);
            const input = [1, 2, "x", 4];
            const result = runner.call(input, output);
            return result === "3x4" && output[2] === "x" && output[3] === 4;
        }
    "#;
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            inlining: false,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "cold_array_fallback.js").unwrap();
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::boolean(true)
        );
    }
}
