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
