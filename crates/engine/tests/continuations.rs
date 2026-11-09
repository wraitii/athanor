use js_engine::{Engine, EngineOptions, Value};

fn run(source: &str, enabled: bool) -> Result<Value, String> {
    let mut engine = Engine::with_options(EngineOptions {
        ssa_opt: enabled,
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    engine
        .compile(source, "continuations.js")
        .map_err(|error| error.to_string())?;
    engine
        .call("main", &[])
        .map(|value| value.value())
        .map_err(|error| error.to_string())
}

#[test]
fn changed_element_preserves_iteration_effects_and_accumulators() {
    let source = r#"
        let effects = 0;
        function sum(rows, offsets) {
            let total = 0;
            let weighted = 0;
            for (let i = 0; i < offsets.length; ++i) {
                ++effects;
                const value = rows[offsets[i]][0];
                total += value;
                weighted += value * (i + 1);
            }
            return total * 100 + weighted;
        }
        function main() {
            const runner = {call: sum};
            const rows = [Float64Array.from([2]), Float64Array.from([4]), Float64Array.from([8])];
            const offsets = Int32Array.from([0, 1, 2, 0, 1, 2]);
            for (let i = 0; i < 40; ++i)
                runner.call(rows, offsets);
            rows[2] = [8];
            effects = 0;
            const result = runner.call(rows, offsets);
            return result * 10 + effects;
        }
    "#;
    assert_eq!(run(source, false), Ok(Value::number(29106.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn resumed_nested_loops_and_suffix_use_the_current_receiver() {
    let source = r#"
        let effects = 0;
        function sum(rows, offsets) {
            let total = 0;
            for (let outer = 0; outer < 2; ++outer) {
                for (let i = 0; i < offsets.length; ++i) {
                    ++effects;
                    total += rows[offsets[i]][0];
                }
                total += this.bias;
            }
            this.result = total;
            return total + this.extra;
        }
        function main() {
            const runner = {call: sum, bias: 1, extra: 5, result: 0};
            const rows = [Float64Array.from([2]), Float64Array.from([4]), Float64Array.from([8])];
            const offsets = Int32Array.from([0, 1, 2]);
            for (let i = 0; i < 40; ++i)
                runner.call(rows, offsets);
            rows[2] = [8];
            effects = 0;
            const other = {call: sum, bias: 7, extra: 11, result: 0};
            const result = other.call(rows, offsets);
            return result * 10000 + other.result * 10 + effects;
        }
    "#;
    assert_eq!(run(source, false), Ok(Value::number(530426.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn nested_loops_preserve_progress_and_heap_intermediates() {
    let source = r#"
        let effects = 0;
        let saved;
        function collect(rows, offsets) {
            let total = 0;
            let previous = {value: 1};
            for (let outer = 0; outer < 2; ++outer) {
                for (let i = 0; i < offsets.length; ++i) {
                    ++effects;
                    const old = previous;
                    previous = {value: rows[offsets[i]][0] + outer};
                    total += old.value + previous.value;
                }
            }
            saved = previous;
            return total;
        }
        function main() {
            const runner = {call: collect};
            const rows = [Float64Array.from([2]), Float64Array.from([4]), Float64Array.from([8])];
            const offsets = Int32Array.from([0, 1, 2]);
            for (let i = 0; i < 40; ++i)
                runner.call(rows, offsets);
            rows[2] = [8];
            effects = 0;
            const result = runner.call(rows, offsets);
            return result * 100 + saved.value * 10 + effects;
        }
    "#;
    assert_eq!(run(source, false), Ok(Value::number(5496.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn error_after_changed_element_keeps_effects_once() {
    let source = r#"
        let effects = 0;
        function sum(rows, offsets) {
            let total = 0;
            for (let i = 0; i < offsets.length; ++i) {
                ++effects;
                total += rows[offsets[i]][0];
            }
            return total;
        }
        function main() {
            const runner = {call: sum};
            const rows = [Float64Array.from([2]), Float64Array.from([4]), Float64Array.from([8])];
            const offsets = Int32Array.from([0, 1, 2]);
            for (let i = 0; i < 40; ++i)
                runner.call(rows, offsets);
            rows[2] = null;
            effects = 0;
            return runner.call(rows, offsets);
        }
        function effectCount() { return effects; }
    "#;
    let execute = |enabled| {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt: enabled,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "continuations.js").unwrap();
        let error = engine.call("main", &[]).err().unwrap().to_string();
        let effects = engine.call("effectCount", &[]).unwrap().value();
        (error, effects)
    };
    let generic = execute(false);
    assert_eq!(generic.1, Value::number(3.0));
    assert_eq!(execute(true), generic);
}

#[test]
fn resumed_loop_keeps_borrowed_heap_input_alive_after_source_mutation() {
    let source = r#"
        let effects = 0;
        let change = false;
        function sum(rows, offsets) {
            const kept = rows[0];
            let total = 0;
            for (let i = 0; i < offsets.length; ++i) {
                ++effects;
                total += rows[offsets[i]][0];
                if (change && i === 2)
                    rows[0] = null;
                total += kept[0];
            }
            return total;
        }
        function main() {
            const runner = {call: sum};
            const rows = [Float64Array.from([2]), Float64Array.from([4]), Float64Array.from([8])];
            const offsets = Int32Array.from([0, 1, 2]);
            for (let i = 0; i < 40; ++i)
                runner.call(rows, offsets);
            rows[2] = [8];
            change = true;
            effects = 0;
            return runner.call(rows, offsets) * 10 + effects;
        }
    "#;
    assert_eq!(run(source, false), Ok(Value::number(203.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn borrowed_guard_inputs_may_be_immediate_values() {
    for replacement in ["3.25", "true", "undefined"] {
        let source = format!(
            r#"
            let effects = 0;
            function sum(rows, offsets) {{
                let total = 0;
                for (let i = 0; i < offsets.length; ++i) {{
                    ++effects;
                    const value = rows[offsets[i]][0];
                    total += value === undefined ? 7 : value;
                }}
                return total * 10 + effects;
            }}
            function main() {{
                const runner = {{call: sum}};
                const rows = [Float64Array.from([2]), Float64Array.from([4]), Float64Array.from([8])];
                const offsets = Int32Array.from([0, 1, 2]);
                for (let i = 0; i < 40; ++i)
                    runner.call(rows, offsets);
                rows[2] = {replacement};
                effects = 0;
                return runner.call(rows, offsets);
            }}
            "#
        );
        let generic = run(&source, false);
        assert!(generic.is_err(), "replacement: {replacement}");
        assert_eq!(run(&source, true), generic, "replacement: {replacement}");
    }
}

#[test]
fn loop_exits_carry_break_and_return_state() {
    let source = r#"
        let effects = 0;
        function scan(rows, offsets, stop, quit) {
            let total = 0;
            let last = {value: 0};
            for (let outer = 0; outer < 3; ++outer) {
                for (let i = 0; i < offsets.length; ++i) {
                    ++effects;
                    const value = rows[offsets[i]][0];
                    if (value === stop)
                        break;
                    if (value === quit)
                        return total * 1000 + last.value * 10 + outer;
                    last = {value: value + outer};
                    total += value;
                }
                total += last.value;
            }
            return total * 100 + last.value;
        }
        function main() {
            const runner = {call: scan};
            const rows = [Float64Array.from([2]), Float64Array.from([4]), Float64Array.from([8]), Float64Array.from([16])];
            const offsets = Int32Array.from([0, 1, 2, 3]);
            for (let i = 0; i < 40; ++i)
                runner.call(rows, offsets, -1, -1);
            rows[1] = [4];
            effects = 0;
            const stopped = runner.call(rows, offsets, 8, -1);
            const quit = runner.call(rows, offsets, -1, 16);
            return (stopped * 100000 + quit) * 100 + effects;
        }
    "#;
    assert_eq!(run(source, false), Ok(Value::number(33061408013.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn three_nested_loops_resume_at_each_level() {
    let source = r#"
        let effects = 0;
        function cube(grid, n) {
            let total = 0;
            for (let a = 0; a < n; ++a) {
                let plane = 0;
                for (let b = 0; b < n; ++b) {
                    let row = 0;
                    for (let c = 0; c < n; ++c) {
                        ++effects;
                        row += grid[(a * n + b) * n + c][0];
                    }
                    plane += row * (b + 1);
                }
                total += plane * (a + 1);
            }
            return total;
        }
        function main() {
            const runner = {call: cube};
            const grid = [];
            for (let i = 0; i < 27; ++i)
                grid.push(Float64Array.from([i % 5]));
            for (let i = 0; i < 40; ++i)
                runner.call(grid, 3);
            grid[13] = [7];
            effects = 0;
            return runner.call(grid, 3) * 100 + effects;
        }
    "#;
    assert_eq!(run(source, false), Ok(Value::number(21727.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn inner_entry_guard_hands_over_to_generic_inner_loop() {
    let source = r#"
        function dot(rows, w) {
            let t = 0;
            for (let y = 0; y < rows.length; ++y) {
                const row = rows[y];
                for (let i = 0; i < w.length; ++i)
                    t += row[i] * w[i];
            }
            return t;
        }
        function main() {
            const runner = {call: dot};
            const w = Float64Array.from([1, 2, 3]);
            const rows = [];
            for (let y = 0; y < 6; ++y)
                rows.push(Float64Array.from([y, y + 1, y + 2]));
            for (let i = 0; i < 40; ++i)
                runner.call(rows, w);
            rows[3] = [5, 6, 7];
            return runner.call(rows, w);
        }
    "#;
    assert_eq!(run(source, false), Ok(Value::number(150.0)));
    assert_eq!(run(source, true), run(source, false));
}

#[test]
fn early_bailout_runs_long_outer_loop_without_recursion() {
    let source = r#"
        function long(rows, n) {
            let total = 0;
            for (let i = 0; i < n; ++i)
                for (let j = 0; j < rows.length; ++j)
                    total += rows[j][0];
            return total;
        }
        function main() {
            const runner = {call: long};
            const rows = [Float64Array.from([1]), Float64Array.from([2]), Float64Array.from([3])];
            for (let i = 0; i < 40; ++i)
                runner.call(rows, 10);
            rows[0] = [1];
            return runner.call(rows, 200000);
        }
    "#;
    assert_eq!(run(source, false), Ok(Value::number(1200000.0)));
    assert_eq!(run(source, true), run(source, false));
}
