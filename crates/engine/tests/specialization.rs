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
