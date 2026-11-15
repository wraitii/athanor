use js_engine::{Engine, EngineOptions, Error, Value};

#[test]
fn local_truthiness_and_indexing_preserve_semantics() {
    {
        let mut engine = Engine::with_options(EngineOptions {
            ..Default::default()
        })
        .unwrap();
        engine.compile(
            "function main() { let a = [7]; let z = -0; let n = 0/0; let empty = ''; let text = 'x'; return !z && !n && !empty && !!text && !!a && a[z] === 7 && a[5] === undefined && a['0'] === 7; }",
            "fast_paths.js",
        ).unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        engine
            .compile("function tdz() { return x; let x = 1; }", "tdz.js")
            .unwrap();
        assert!(matches!(engine.call("tdz", &[]), Err(Error::Runtime(_))));
    }
}

#[test]
fn truthiness_preserves_restricted_unions_and_generic_values() {
    for ssa_opt in [false, true] {
        {
            let mut engine = Engine::with_options(EngineOptions {
                ssa_opt,
                ..Default::default()
            })
            .unwrap();
            engine
                .compile(
                    "function branch(value) { if (value) return true; return false; }
                function number_boolean(flag, n, b) {
                    let value = flag > 0 ? +n : b > 0;
                    if (value) return true;
                    return false;
                }
                function string_number(flag, text, n) {
                    let value = flag > 0 ? '' + text : +n;
                    if (value) return true;
                    return false;
                }
                function string_null(flag, text) {
                    let value = flag > 0 ? '' + text : null;
                    if (value) return true;
                    return false;
                }
                function object_null(flag) {
                    let value = flag > 0 ? {} : null;
                    if (value) return true;
                    return false;
                }
                function string_only(text) { if ('' + text) return true; return false; }
                function object_only() { if ({}) return true; return false; }",
                    "truthiness.js",
                )
                .unwrap();
            engine.compile(
                "function main() {
                    const values = [0, -0, 0/0, 1, -1, 1/0, true, false, null, undefined, '', 'x', {}, [], function(){}];
                    const expected = [false, false, false, true, true, true, true, false, false, false, false, true, true, true, true];
                    for (let i = 0; i < values.length; ++i)
                        if (branch(values[i]) !== expected[i]) return false;
                    for (let flag = 0; flag < 2; ++flag) {
                        if (string_number(flag, '', 0) || string_number(flag, '', 0/0)) return false;
                        if (!string_number(flag, 'x', 1)) return false;
                        if (string_null(flag, '')) return false;
                        if (string_null(flag, 'x') !== (flag > 0)) return false;
                    }
                    return !string_only('') && string_only('x');
                }",
                "truthiness_values.js",
            ).unwrap();
            assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
            for flag in [0.0, 1.0] {
                for number in [0.0, -0.0, f64::NAN, 1.0, -1.0, f64::INFINITY] {
                    for boolean in [0.0, 1.0] {
                        let args = [
                            Value::number(flag),
                            Value::number(number),
                            Value::number(boolean),
                        ];
                        let expected = if flag > 0.0 {
                            number != 0.0 && !number.is_nan()
                        } else {
                            boolean > 0.0
                        };
                        assert_eq!(
                            engine.call("number_boolean", &args).unwrap().value(),
                            Value::boolean(expected)
                        );
                    }
                }
                assert_eq!(
                    engine
                        .call("object_null", &[Value::number(flag)])
                        .unwrap()
                        .value(),
                    Value::boolean(flag > 0.0)
                );
            }
            assert_eq!(
                engine.call("object_only", &[]).unwrap().value(),
                Value::TRUE
            );
        }
    }
}
