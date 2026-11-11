use js_engine::{Engine, EngineOptions, Error, OptLevel, Value};

fn engines() -> impl Iterator<Item = Engine> {
    [OptLevel::None, OptLevel::Speed]
        .into_iter()
        .flat_map(|opt_level| {
            [false, true].into_iter().map(move |ssa_opt| {
                Engine::with_options(EngineOptions {
                    opt_level,
                    ssa_opt,
                    ..Default::default()
                })
                .unwrap()
            })
        })
}

#[test]
fn constant_integer_products_preserve_extremes_and_zero_signs() {
    for factor in [0.0, -1.0, 0.5, 3.0, 4_194_304.0, 4_194_305.0] {
        for expression in [format!("(x|0)*{factor}"), format!("{factor}*(x|0)")] {
            for mut engine in engines() {
                engine
                    .compile(
                        &format!("function main(x){{return {expression};}}"),
                        "integer-products.js",
                    )
                    .unwrap();
                for input in [i32::MIN, -1, 0, 1, i32::MAX] {
                    assert_eq!(
                        engine
                            .call("main", &[Value::number(input as f64)])
                            .unwrap()
                            .value(),
                        Value::number(input as f64 * factor),
                        "{expression}, x={input}"
                    );
                }
            }
        }
    }
}

#[test]
fn indexed_stores_preserve_values_and_ownership() {
    let source = r#"
        function store(a, k, v) { return a[k] = v; }
        function main() {
            const a = [{n: 1}, 'old', Math.abs];
            const saved = a[0];
            for (let i = 0; i < 300; ++i) {
                a[0] = a[0];
                a[0] = {n: i};
                a[1] = 'value' + i;
            }
            store(a, 2, {n: 7});
            const assigned = store(a, 0, {n: 1000});
            a[1] = assigned;
            a[0] = 5;
            return saved.n === 1 && assigned.n === 1000 &&
                a[0] === 5 && a[1].n === 1000 && a[2].n === 7;
        }
    "#;
    for mut engine in engines() {
        engine.compile(source, "indexed-stores.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    }
}

#[test]
fn indexed_stores_preserve_fallbacks_and_array_transitions() {
    let source = r#"
        function store(a, k, v) { a[k] = v; }
        function rejected(k) { const a = [1, 2]; store(a, k, 9); }
        function booleanKey() { const a = [1, 2]; a[true] = 9; }
        function main() {
            const a = [1, 2];
            for (let i = 0; i < 4; ++i)
                a[i] = i + 10;
            const holey = new Array(3);
            store(holey, 1, 20);
            store(holey, 0, 10);
            store(holey, 2, 3);
            store(holey, 2, 30);
            store(a, 1, 20);
            store(a, 2, 30);
            store(a, '0', 40);
            const o = {};
            store(o, 0, 90);
            return a.length === 4 && a[0] === 40 && a[1] === 20 &&
                a[2] === 30 && a[3] === 13 && o[0] === 90 &&
                holey[0] === 10 && holey[1] === 20 && holey[2] === 30;
        }
    "#;
    for mut engine in engines() {
        engine
            .compile(source, "indexed-store-fallbacks.js")
            .unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        for key in [
            Value::number(-1.0),
            Value::number(0.5),
            Value::number(f64::NAN),
            Value::number(f64::INFINITY),
            Value::number(5.0),
            Value::TRUE,
            Value::FALSE,
        ] {
            assert!(matches!(
                engine.call("rejected", &[key]),
                Err(Error::Runtime(_))
            ));
        }
        assert!(matches!(
            engine.call("booleanKey", &[]),
            Err(Error::Runtime(_))
        ));
    }
}

#[test]
fn numeric_lowering_preserves_mixed_paths_loops_and_calls() {
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
fn numeric_lowering_preserves_nan_signed_zero_and_runtime_errors() {
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

#[test]
fn temporary_clears_preserve_calls_branches_loops_and_exceptions() {
    let source = r#"
        function make(n) { return { value: n, text: 'value' + n }; }
        function testBranch(flag) {
            let item = flag ? make(7) : { value: 8, text: 'other' };
            1 + 2;
            return item.value + ':' + item.text;
        }
        function testLoop() {
            let item = make(0);
            for (let i = 0; i < 20; ++i) {
                if (i % 2) item = make(i);
                else 1 + i;
            }
            return item.value === 19 && item.text === 'value19';
        }
        function numberCall() { return [1, 2, 3].push(4); }
        function testCall() { let a = make(11); numberCall(); return a.value === 11 && a.text === 'value11'; }
        function testCatch() {
            try { throw make(23); }
            catch (value) { 1 + 2; return value.value === 23 && value.text === 'value23'; }
        }
        function* values() { yield make(31); yield make(32); }
        function testGenerator() {
            let g = values(); let a = g.next().value; let b = g.next().value;
            return a.value === 31 && a.text === 'value31' && b.value === 32 && b.text === 'value32';
        }
        function failAdd() { return 'text' + {}; }
        function main() {
            return testBranch(true) === '7:value7' && testBranch(false) === '8:other' &&
                testLoop() && testCall() && testCatch() && testGenerator();
        }
    "#;
    for mut engine in engines() {
        engine.compile(source, "ssa-temporary-clears.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        assert!(matches!(
            engine.call("failAdd", &[]),
            Err(Error::Runtime(_))
        ));
    }
}

#[test]
fn scoped_index_reads_preserve_values_guards_and_errors() {
    let source = r#"
        function repeat(a, key, count) {
            let first = a[key];
            let sum = 0;
            for (let i = 0; i < count; ++i) {
                if (a[key]) ++sum;
            }
            return first ? sum : -1;
        }
        function objects(a, count) {
            let first = a[0];
            let sum = 0;
            for (let i = 0; i < count; ++i) sum += a[0].value;
            return first.value + sum;
        }
        function methodReads(a) {
            let first = a['map'];
            let equal = true;
            for (let i = 0; i < 3; ++i) equal = equal && a['map'] === first;
            return equal;
        }
        function differing(a) {
            let first = a[0];
            let sum = 0;
            for (let i = 0; i < 3; ++i) sum += a[i];
            return first + sum;
        }
        function maybeFail(a, b, count) {
            let first = a[0];
            for (let i = 0; i < count; ++i) b[0];
            return first;
        }
        function mutate(a) {
            let first = a[0];
            for (let i = 0; i < 2; ++i) { a[0] = i + 2; }
            return first + a[0];
        }
        function callback(a, f) {
            let first = a[0];
            for (let i = 0; i < 2; ++i) f(a);
            return first + a[0];
        }
        function main() {
            let typed = new Uint16Array(1); typed[0] = 4;
            let a = [{value: 7}];
            return repeat(typed, 0, 5) === 5 && repeat([3], 0, 4) === 4 &&
                repeat('hello', 0, 3) === 3 && repeat({'0': 9}, 0, 2) === 2 &&
                repeat([0], 0, 3) === -1 && objects(a, 4) === 35 &&
                differing([2, 3, 4]) === 11 && methodReads([]) && maybeFail([9], null, 0) === 9 &&
                mutate([1]) === 4 && callback([1], function(a) { a[0] += 1; }) === 4;
        }
        function fail() { return maybeFail([9], null, 1); }
        function failFirst() { return repeat(null, 0, 0); }
    "#;
    for mut engine in engines() {
        engine.compile(source, "ssa-index-scope.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        assert!(matches!(engine.call("fail", &[]), Err(Error::Runtime(_))));
        assert!(matches!(
            engine.call("failFirst", &[]),
            Err(Error::Runtime(_))
        ));
    }
}

#[test]
fn rooted_receiver_scope_preserves_borrowed_values_and_errors() {
    let source = r#"
        function read(n) {
            if (!this.items[0]) return null;
            let total = 0;
            for (let i = 0; i < Math.floor(n); ++i) total += this.items[0].value;
            if (total < 0) return null;
            return this.items[0];
        }
        function fail(n) {
            if (!this.items[0]) return 0;
            for (let i = 0; i < Math.floor(n); ++i) this.items[0] * {};
            return 1;
        }
        function run() {
            const value = {items: [{value: 7}], read: read}.read(5);
            return value.value === 7;
        }
        function runFailure() { return {items: [{value: 7}], read: fail}.read(1); }
    "#;
    let mut expected_error = None;
    for mut engine in engines() {
        engine.compile(source, "rooted-index-scope.js").unwrap();
        assert_eq!(engine.call("run", &[]).unwrap().value(), Value::TRUE);
        let error = format!("{:?}", engine.call("runFailure", &[]).err().unwrap());
        if let Some(expected) = &expected_error {
            assert_eq!(&error, expected);
        } else {
            expected_error = Some(error);
        }
    }
}

#[test]
fn semantic_bit_loop_accepts_local_indexed_receivers() {
    let source = r#"
        function count(data, key, start, end, initial) {
            let result = initial;
            if (data[key])
                for (let i = start; i <= end; ++i)
                    if (data[key] & (1 << i)) ++result;
            return result;
        }
        function run() {
            let data = new Uint16Array(1); data[0] = 257;
            return count(data, 0, 0, 8, 0) === 2 &&
                count([257], 0, 0, 8, 0) === 2 &&
                count(['257'], 0, 0, 8, 0) === 2 &&
                count({'0': 257}, 0, 0, 8, 0) === 2 &&
                count(data, 0, 32, 32, 0) === 1 &&
                count(data, 0, 0, 8, '2') === 4;
        }
        function fail() { return count(null, 0, 0, 0, 0); }
    "#;
    let mut expected = None;
    for mut engine in engines() {
        engine
            .compile(source, "semantic-local-receiver.js")
            .unwrap();
        assert_eq!(engine.call("run", &[]).unwrap().value(), Value::TRUE);
        let error = format!("{:?}", engine.call("fail", &[]).err().unwrap());
        if let Some(expected) = &expected {
            assert_eq!(&error, expected);
        } else {
            expected = Some(error);
        }
    }
}

#[test]
fn semantic_bit_loop_preserves_numeric_boundaries_and_fallbacks() {
    let source = r#"
        function count(start, end, initial) {
            let result = initial;
            const key = 0;
            if (this.data[key])
                for (let i = start; i <= end; ++i)
                    if (this.data[key] & (1 << i)) ++result;
            return result;
        }
        function aliasKey(start, end) {
            let result = 0;
            if (this.data[result])
                for (let i = start; i <= end; ++i)
                    if (this.data[result] & (1 << i)) ++result;
            return result;
        }
        function aliasBound(start) {
            let result = 0;
            const key = 0;
            if (this.data[key])
                for (let i = start; i <= result; ++i)
                    if (this.data[key] & (1 << i)) ++result;
            return result;
        }
        function run() {
            let data = new Uint16Array(1); data[0] = 257;
            let unsigned = new Uint32Array(1); unsigned[0] = 2147483648;
            let full = new Uint32Array(1); full[0] = 4294967295;
            let holder = { data: data, count: count };
            let big = 9007199254740992;
            let nan = holder.count(0, 0, 0/0);
            return holder.count(0, 8, 0) === 2 && holder.count(0, 8, big) === big &&
                holder.count(0, 8, 9007199254740991) === big &&
                1 / holder.count(1, 7, -0) === -Infinity &&
                holder.count(8, 0, 5) === 5 && holder.count(-32, -32, 5) === 6 &&
                holder.count(32, 32, 5) === 6 && holder.count(0.5, 0.5, 5) === 6 &&
                holder.count(0, 0, '2') === 3 && holder.count(0, 0, Infinity) === Infinity &&
                nan !== nan && {data: [257], count: count}.count(0, 8, 0) === 2 &&
                {data: ['257'], count: count}.count(0, 8, 0) === 2 &&
                {data: {'0': 257}, count: count}.count(0, 8, 0) === 2 &&
                {data: [-2147483648], count: count}.count(31, 31, 0) === 1 &&
                {data: unsigned, count: count}.count(31, 31, 0) === 1 &&
                {data: full, count: count}.count(0, 31, 0) === 32 &&
                {data: [7, 0], count: aliasKey}.count(0, 2) === 1 &&
                {data: [1], count: aliasBound}.count(0) === 1;
        }
        function fail() { return {data: [1], count: count}.count(0, 0, {}); }
        function badSource() { return {data: null, count: count}.count(0, 0, 0); }
    "#;
    let mut expected = None;
    for mut engine in engines() {
        engine.compile(source, "semantic-bit-loop.js").unwrap();
        assert_eq!(engine.call("run", &[]).unwrap().value(), Value::TRUE);
        let errors: Vec<_> = ["fail", "badSource"]
            .into_iter()
            .map(|name| format!("{:?}", engine.call(name, &[]).err().unwrap()))
            .collect();
        if let Some(expected) = &expected {
            assert_eq!(&errors, expected);
        } else {
            expected = Some(errors);
        }
    }
}

#[test]
fn semantic_bit_loop_preserves_tdz_and_const_errors() {
    let source = r#"
        function tdzBound() {
            let result = 0; const key = 0; const end = 0;
            if (this.data[key])
                for (let i = start; i <= end; ++i)
                    if (this.data[key] & (1 << i)) ++result;
            let start = 0;
            return result;
        }
        function tdzCounter() {
            const key = 0; const start = 0; const end = 0;
            if (this.data[key])
                for (let i = start; i <= end; ++i)
                    if (this.data[key] & (1 << i)) ++result;
            let result = 9;
            return result;
        }
        function firstError() { return {data: null, run: tdzBound}.run(); }
        function boundError() { return {data: [1], run: tdzBound}.run(); }
        function counterError() { return {data: [1], run: tdzCounter}.run(); }
        function noCounterRead() { return {data: [2], run: tdzCounter}.run(); }
    "#;
    let invalid = r#"function f() { const result = 0; const key = 0; const start = 0; const end = 0; if (this.data[key]) for (let i = start; i <= end; ++i) if (this.data[key] & (1 << i)) ++result; }"#;
    let mut expected = None;
    for mut engine in engines() {
        engine.compile(source, "semantic-bit-tdz.js").unwrap();
        assert_eq!(
            engine.call("noCounterRead", &[]).unwrap().value(),
            Value::number(9.0)
        );
        let errors: Vec<_> = ["firstError", "boundError", "counterError"]
            .into_iter()
            .map(|name| format!("{:?}", engine.call(name, &[]).err().unwrap()))
            .collect();
        if let Some(expected) = &expected {
            assert_eq!(&errors, expected);
        } else {
            expected = Some(errors);
        }
        assert!(engine.compile(invalid, "semantic-bit-const.js").is_err());
    }
}

#[test]
fn mixed_relational_comparisons_preserve_coercions_and_string_ordering() {
    let source = r#"
        function less(value, number) { return value < number; }
        function main() {
            for (let i = 0; i < 4; ++i) {
                const mixed = i & 1 ? '12' : 12;
                if (!(mixed < 20) || mixed < 10 || mixed >= 20 || !(mixed > 10))
                    return false;
            }
            return less('2', 10) && !less('12', 10) && less('12', '2') &&
                !less('invalid', 10) && !less(0 / 0, 10) && less(null, 1) &&
                less(false, 1) && !less(undefined, 1);
        }
        function fail() { return {} < 1; }
    "#;
    for mut engine in engines() {
        engine.compile(source, "mixed-comparisons.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        assert!(matches!(engine.call("fail", &[]), Err(Error::Runtime(_))));
    }
}

#[test]
fn conditional_region_guards_preserve_recursive_calls_and_replacements() {
    let source = r#"
        let state = { value: 3 };
        function first(n) { if (n) return second(n - 1); return state.value; }
        function second(n) { if (n) return first(n - 1); return state.value; }
        function sum(xs, n) {
            let total = 0;
            for (let i = 0; i < n; ++i) total += Math.floor(xs[i]);
            return total + first(n);
        }
        function rows(xs) {
            let total = 0;
            for (let y = 0; y < 2; ++y) {
                const row = xs[y];
                for (let x = 0; x < 2; ++x) total += row[x] + first(x);
            }
            return total;
        }
        function change() {
            second = n => 20;
            Math.floor = n => n + 10;
            state.value = 7;
        }
    "#;
    for mut engine in engines() {
        engine.compile(source, "conditional-regions.js").unwrap();
        engine.compile("function runSum(){return sum([1.5,2.5],2);} function runRows(){return rows([[1,2],[3,4]]);}", "conditional-callers.js").unwrap();
        for _ in 0..3 {
            assert_eq!(
                engine.call("runSum", &[]).unwrap().value(),
                Value::number(6.0)
            );
            assert_eq!(
                engine.call("runRows", &[]).unwrap().value(),
                Value::number(22.0)
            );
        }
        engine.call("change", &[]).unwrap();
        for _ in 0..3 {
            assert_eq!(
                engine.call("runSum", &[]).unwrap().value(),
                Value::number(44.0)
            );
            assert_eq!(
                engine.call("runRows", &[]).unwrap().value(),
                Value::number(64.0)
            );
        }
    }
}

#[test]
fn array_iteration_indices_and_captured_parameters_survive_calls() {
    let source = r#"
        function select(values, threshold) {
            return values.filter((value, index) => value + index >= threshold);
        }
        function mutate(values, threshold) {
            return values.filter((value, index) => {
                threshold += index;
                return value >= threshold;
            });
        }
        function main() {
            const selected = select([1, 2, 3], 3);
            const changed = mutate([1, 2, 3], 2);
            const mapped = [10, 20, 30].map((value, index) => value + index);
            const custom = {};
            custom.filter = function(callback) {
                return [callback(1, 0), callback(4, 1)];
            };
            const fallback = select(custom, 3);
            return selected.length === 2 && selected[0] === 2 && selected[1] === 3 &&
                changed.length === 0 && mapped[0] === 10 && mapped[2] === 32 &&
                fallback[0] === false && fallback[1] === true;
        }
    "#;
    for mut engine in engines() {
        engine.compile(source, "array-indices-captures.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    }
}

#[test]
fn shared_error_origin_preserves_each_failing_span() {
    let source = "function fail(first) { if (first) return {} * 2; return {} / 3; }";
    for mut engine in engines() {
        engine.compile(source, "shared-error-origin.js").unwrap();
        for (first, expression) in [(true, "{} * 2"), (false, "{} / 3")] {
            let start = source.find(expression).unwrap();
            let end = start + expression.len();
            let error = engine.call("fail", &[Value::boolean(first)]).err().unwrap();
            assert!(
                error
                    .to_string()
                    .contains(&format!("shared-error-origin.js:{start}..{end}")),
                "{error}"
            );
        }
    }
}

#[test]
fn borrowed_rows_preserve_guards_and_owner_invalidation() {
    let source = r#"
        function selected(owner) {
            const height = owner.height;
            let sum = 0;
            for (let i = 0; i < 64; ++i) sum += height[i & 1][0];
            return sum;
        }
        function mutate(owner, callback) {
            const height = owner.height;
            let sum = 0;
            for (let i = 0; i < 2; ++i) {
                const row = height[i];
                callback(height, i);
                sum += row[0];
            }
            return sum;
        }
        function main() {
            const first = new Float32Array(1);
            const second = new Float32Array(1);
            first[0] = 3;
            second[0] = 9;
            const owner = {height: [first, second]};
            const runner = {read: selected, mutate: mutate};
            for (let i = 0; i < 40; ++i)
                if (runner.read(owner) !== 384) return false;
            owner.height = [[5], [7]];
            if (runner.read(owner) !== 384) return false;
            owner.height = [{0: 2}, {0: 6}];
            if (runner.read(owner) !== 256) return false;
            owner.height = [new Float32Array(1), new Float32Array(1)];
            owner.height[0][0] = 3;
            owner.height[1][0] = 9;
            return runner.mutate(owner, function(height, i) {
                height[i] = null;
                for (let j = 0; j < 32; ++j) new Float32Array(1);
            }) === 12;
        }
    "#;
    for opt_level in [OptLevel::None, OptLevel::Speed] {
        let mut engine = Engine::with_options(EngineOptions {
            opt_level,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "borrowed-rows.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    }
}

#[test]
fn borrowed_phi_rows_keep_their_owners_alive() {
    let source = r#"
        function selected(choose) {
            const first = [[3]];
            const second = [[9]];
            const row = choose ? first[0] : second[0];
            const scratch = [1, 2, 3];
            return row[0] + scratch[0];
        }
        function main() {
            for (let i = 0; i < 64; ++i) {
                if (selected(true) !== 4 || selected(false) !== 10)
                    return false;
            }
            return true;
        }
    "#;
    for opt_level in [OptLevel::None, OptLevel::Speed] {
        let mut engine = Engine::with_options(EngineOptions {
            opt_level,
            background_compile: false,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "borrowed-phi.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    }
}

#[test]
fn integer_loop_values_widen_and_merge_with_fractions() {
    let source = r#"
        function count(limit) {
            let k = 0;
            while (k < limit) ++k;
            return k;
        }
        function main() {
            let up = 2147483647;
            let down = -2147483648;
            let mixed = 0;
            for (let i = 0; i < 4; ++i) {
                ++up;
                --down;
                mixed += i === 2 ? 0.5 : 1;
            }
            return up === 2147483651 && down === -2147483652 &&
                mixed === 3.5 && (up | 0) === -2147483645 && count(4) === 4;
        }
    "#;
    for mut engine in engines() {
        engine.compile(source, "integer_loops.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    }
}
