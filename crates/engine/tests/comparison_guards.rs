use js_engine::{Engine, EngineOptions, Value};

fn run(source: &str, ssa_opt: bool) -> Result<Value, String> {
    let mut engine = Engine::with_options(EngineOptions {
        ssa_opt,
        ..Default::default()
    })
    .unwrap();
    engine
        .compile(source, "comparison-guards.js")
        .map_err(|error| error.to_string())?;
    engine
        .call("main", &[])
        .map(|value| value.value())
        .map_err(|error| error.to_string())
}

#[test]
fn exposed_number_checks_preserve_comparison_fallbacks() {
    let source = r#"
        function compare(a,b) {
            return (a<b?1:0)+(a<=b?2:0)+(a>b?4:0)+(a>=b?8:0);
        }
        function main() {
            return compare(10,2)===12 && compare("10","2")===3 &&
                compare("2",10)===3 && compare(null,0)===10 &&
                compare(true,1)===10 && compare(undefined,0)===0 &&
                compare(NaN,NaN)===0;
        }
    "#;
    for ssa_opt in [false, true] {
        assert_eq!(run(source, ssa_opt).unwrap(), Value::TRUE);
    }
}

#[test]
fn comparisons_preserve_original_index_keys_and_errors() {
    for key in ["1", "01"] {
        let source = format!(
            r#"
            function read(array,x) {{if(x<0||x>=2)return -1;return array[x];}}
            function main() {{const array=[7,8];read(array,1);return read(array,"{key}");}}
        "#
        );
        let generic = run(&source, false);
        let optimized = run(&source, true);
        assert_eq!(optimized, generic);
        if key == "1" {
            assert_eq!(optimized.unwrap(), Value::number(8.0));
        } else {
            assert!(optimized.is_err());
        }
    }
}
