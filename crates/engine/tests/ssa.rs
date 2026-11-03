use js_engine::{Engine, EngineOptions, Error, OptLevel, Value};

fn engines() -> impl Iterator<Item = Engine> {
    [OptLevel::None, OptLevel::Speed]
        .into_iter()
        .flat_map(|opt_level| {
            [false, true].into_iter().map(move |ssa_opt| {
                Engine::with_options(EngineOptions {
                    opt_level,
                    ssa_opt,
                    ssa_dump: false,
                    ..Default::default()
                })
                .unwrap()
            })
        })
}

#[test]
fn numeric_analysis_preserves_mixed_paths_loops_and_calls() {
    let source = r#"
        function add(x) { return x + 1; }
        function* values(x) { yield x * 2; x = 's'; yield x + 1; }
        function main() {
            let mixed = 0;
            for (let i = 0; i < 4; ++i) {
                if (i === 2) mixed = 's';
                mixed += 1;
            }
            let captured = 1;
            function change() { captured = 's'; }
            const before = captured * 2;
            change();
            const generator = values(3);
            return mixed === 's11' && before === 2 && captured + 1 === 's1' &&
                add(2) === 3 && add('2') === '21' &&
                generator.next().value === 6 && generator.next().value === 's1';
        }
    "#;
    for mut engine in engines() {
        engine.compile(source, "ssa-paths.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    }
}

#[test]
fn numeric_analysis_preserves_nan_signed_zero_and_runtime_errors() {
    for mut engine in engines() {
        engine.compile(
            "function main() { let zero = -0; let nan = 0/0; let infinity = 1/0; return 1/zero === -infinity && nan !== nan && infinity * 0 !== infinity * 0; }",
            "ssa-numbers.js",
        ).unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        engine.compile(
            "function coercion() { return {} * 2; } function tdz() { let value = value * 2; return value; }",
            "ssa-errors.js",
        ).unwrap();
        for function in ["coercion", "tdz"] {
            assert!(matches!(engine.call(function, &[]), Err(Error::Runtime(_))));
        }
    }
}

#[test]
fn helper_effects_preserve_exception_order_and_catch_reentry() {
    let source = r#"
        function fail() { throw 7; }
        function main() {
            let state = 0;
            try {
                const value = fail();
                state = value + 1;
            } catch (error) {
                const array = new Uint16Array(1);
                array[0] = error;
                state = (array[0] & 3) + 1;
            }
            try {
                fail();
                state = 99;
            } catch (error) {
                state += error;
            }
            return state === 11;
        }
    "#;
    for mut engine in engines() {
        engine.compile(source, "ssa-exceptions.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    }
}

#[test]
fn numeric_results_preserve_input_types_and_error_exits() {
    let source = r#"
        function calculate(radius) {
            const square = radius * radius;
            let total = 0;
            for (let i = square >> 1; i <= (square & 7); ++i)
                total += (i & 3) + 1;
            return square === 9 && radius + 1 === '31' && total === 0;
        }
        function main() {
            const nan = 0 / 0;
            const infinity = 1 / 0;
            return calculate('3') && !(nan * 2) && !(infinity - infinity) &&
                (nan >> 1) === 0 && 1 / (-0 * 2) === -infinity;
        }
        function fail() { return ({} * 2) + 1; }
    "#;
    for mut engine in engines() {
        engine.compile(source, "ssa-result-facts.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        assert!(matches!(engine.call("fail", &[]), Err(Error::Runtime(_))));
    }
}

#[test]
fn bitwise_and_preserves_conversion_boundaries_and_fallbacks() {
    let cases = [
        (0.0, 0),
        (-0.0, 0),
        (i32::MIN as f64, i32::MIN),
        (i32::MAX as f64, i32::MAX),
        (2147483648.0, i32::MIN),
        (4294967295.0, -1),
        (4294967296.0, 0),
        (4294967297.0, 1),
        (-4294967297.0, -1),
        (4503599627370497.0, 1),
        (1e100, 0),
        (1.5, 1),
        (-1.5, -1),
        (f64::NAN, 0),
        (f64::INFINITY, 0),
        (f64::NEG_INFINITY, 0),
    ];
    for mut engine in engines() {
        engine.compile(
            "function bitAnd(a, b) { return a & b; } function chained(a, b) { return (a & b) & 15; } function fallback() { return ('31' & 15) === 15 && (true & 3) === 1 && (null & 3) === 0 && (undefined & 3) === 0; } function fail() { return {} & 15; }",
            "ssa-bitwise-and.js",
        ).unwrap();
        for (left, left_int) in cases {
            for (right, right_int) in cases {
                let args = [Value::number(left), Value::number(right)];
                let expected = left_int & right_int;
                assert_eq!(
                    engine.call("bitAnd", &args).unwrap().value(),
                    Value::number(expected as f64),
                    "{left} & {right}",
                );
                assert_eq!(
                    engine.call("chained", &args).unwrap().value(),
                    Value::number((expected & 15) as f64),
                );
            }
        }
        assert_eq!(engine.call("fallback", &[]).unwrap().value(), Value::TRUE);
        assert!(matches!(engine.call("fail", &[]), Err(Error::Runtime(_))));
    }
}

#[test]
fn numeric_index_getters_preserve_receivers_keys_and_ownership() {
    let source = r#"
        function read(receiver, index) { return receiver[index * 1]; }
        function main() {
            const array = [{ value: 7 }, 'text'];
            const typed = new Uint16Array(2);
            typed[0] = 19;
            const object = { '0': 23, '1.5': 29, '-1': 31 };
            object[0/0] = 37;
            object[1/0] = 41;
            let sum = 0;
            for (let i = 0; i < 20; ++i) sum += read(array, 0).value;
            return sum === 140 && read(array, 1) === 'text' &&
                read(array, 2) === undefined && read(typed, -0) === 19 &&
                read(typed, -1) === undefined && read(typed, 1.5) === undefined &&
                read(typed, 0/0) === undefined && read(typed, 1/0) === undefined &&
                read('abc', 1) === 'b' && read('abc', 9) === undefined &&
                read(object, -0) === 23 && read(object, 1.5) === 29 &&
                read(object, -1) === 31 && read(object, 0/0) === 37 &&
                read(object, 1/0) === 41;
        }
        function fail() { return read(null, 0); }
    "#;
    for mut engine in engines() {
        engine.compile(source, "ssa-numeric-index.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        assert!(matches!(engine.call("fail", &[]), Err(Error::Runtime(_))));
    }
}
