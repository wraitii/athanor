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
fn integer_binary_operations_preserve_conversion_and_shift_counts() {
    let cases = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        31.0,
        32.0,
        33.0,
        -33.0,
        i32::MIN as f64,
        i32::MAX as f64,
        4294967295.0,
        4294967296.0,
        1.5,
        -1.5,
        f64::NAN,
        f64::INFINITY,
    ];
    for mut engine in engines() {
        engine
            .compile(
                r#"
            function shl(a, b) { return a << b; }
            function shr(a, b) { return a >> b; }
            function ushr(a, b) { return a >>> b; }
            function or(a, b) { return a | b; }
            function xor(a, b) { return a ^ b; }
            function fallback() { return ('8' << '2') === 32 && (true >> null) === 1; }
            function fail() { return {} << 1; }
        "#,
                "ssa-integer-binary.js",
            )
            .unwrap();
        for left in cases {
            for right in cases {
                let to_int = |n: f64| -> i32 {
                    if !n.is_finite() || n == 0.0 {
                        return 0;
                    }
                    n.trunc().rem_euclid(4294967296.0) as u32 as i32
                };
                let (a, b) = (to_int(left), to_int(right));
                for (name, expected) in [
                    ("shl", a.wrapping_shl(b as u32) as f64),
                    ("shr", a.wrapping_shr(b as u32) as f64),
                    ("ushr", (a as u32).wrapping_shr(b as u32) as f64),
                    ("or", (a | b) as f64),
                    ("xor", (a ^ b) as f64),
                ] {
                    assert_eq!(
                        engine
                            .call(name, &[Value::number(left), Value::number(right)])
                            .unwrap()
                            .value(),
                        Value::number(expected),
                        "{name} {left} {right}"
                    );
                }
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

#[test]
fn guarded_math_preserves_mutation_and_fallback_semantics() {
    let source = r#"
        function calculate(x) { return Math.floor(x) + Math.ceil(x); }
        function conditional(x) { if (x) return Math.sqrt(1); return 42; }
        function ceilValue(x) { return Math.ceil(x); }
        function breakMath() { Math = 1; }
        function local(Math) { return Math.floor(2); }
        function main() {
            const originalMath = Math;
            const floor = Math.floor;
            const ceil = Math.ceil;
            if (calculate(2.2) !== 5) return false;
            Math.floor = function(x) { Math.ceil = function(x) { return 20; }; return 10; };
            if (calculate(2.2) !== 30) return false;
            Math.floor = floor;
            Math.ceil = ceil;
            if (calculate(2.2) !== 5) return false;
            if (1 / ceilValue(-0.1) !== -Infinity) return false;
            let nan = calculate(0 / 0);
            if (nan === nan) return false;
            Math = null;
            if (conditional(false) !== 42) return false;
            Math = originalMath;
            return local({floor: function(x) { return 7; }}) === 7;
        }
    "#;
    for mut engine in engines() {
        engine.compile(source, "math-guard.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        assert!(matches!(
            engine
                .call("ceilValue", &[Value::NULL])
                .unwrap()
                .value()
                .as_number(),
            Some(0.0)
        ));
        engine.call("breakMath", &[]).unwrap();
        assert_eq!(
            engine
                .call("conditional", &[Value::FALSE])
                .unwrap()
                .value()
                .as_number(),
            Some(42.0)
        );
        assert!(matches!(
            engine.call("conditional", &[Value::TRUE]),
            Err(Error::Runtime(_))
        ));
    }
}

#[test]
fn guarded_receiver_property_preserves_errors_and_lifetimes() {
    let source = r#"
        function sum(active) {
            if (!active) return 9;
            let total = 0;
            for (let i = 0; i < Math.floor(3); ++i)
                total += this.items[i] + this.items[i];
            return total;
        }
        function test() {
            const holder = {items: [1, 2, 3], sum: sum};
            const detached = holder.sum;
            return holder.sum(true) === 12 && detached(false) === 9;
        }
        function missing() { const holder = {sum: sum}; return holder.sum(true); }
        function invalid() { return sum(true); }
        function order() { return {} * 2 + this.items[0] + this.items[0] + Math.floor(1); }
        function errorOrder() { const holder = {items: [1], run: order}; return holder.run(); }
        function change() { this.items = [9]; return 0; }
        function mutate() { let x = this.items[0]; this.change(); return x + this.items[0] + Math.floor(0); }
        function testMutation() { const holder = {items: [1], change: change, run: mutate}; return holder.run() === 10; }
        function returnItems() { let first = this.items; if (Math.floor(1)) return this.items; return first; }
        function testReturned() { const result = {items: [4, 5], run: returnItems}.run(); return result[0] === 4 && result[1] === 5; }
    "#;
    let mut errors = None;
    for mut engine in engines() {
        engine.compile(source, "receiver-guard.js").unwrap();
        assert_eq!(engine.call("test", &[]).unwrap().value(), Value::TRUE);
        assert_eq!(
            engine.call("testMutation", &[]).unwrap().value(),
            Value::TRUE
        );
        assert_eq!(
            engine.call("testReturned", &[]).unwrap().value(),
            Value::TRUE
        );
        let actual: Vec<_> = ["missing", "invalid", "errorOrder"]
            .into_iter()
            .map(|name| format!("{:?}", engine.call(name, &[]).err().unwrap()))
            .collect();
        if let Some(expected) = &errors {
            assert_eq!(&actual, expected);
        } else {
            errors = Some(actual);
        }
    }
}
