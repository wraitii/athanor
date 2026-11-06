use js_engine::{Engine, EngineOptions, Value};

fn check(source: &str, expected: f64) {
    for ssa_opt in [true, false] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "inline_references.js").unwrap();
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::number(expected),
            "ssa_opt: {ssa_opt}"
        );
        engine.drain_disposals(usize::MAX);
    }
}

#[test]
fn forwarding_constructor_results_and_mixed_joins_preserve_ownership() {
    check(
        "function Item(value) { this.value = value; return undefined; }
        function forward(value, flag) { return flag ? value : value; }
        function choose(a, b, flag) { return flag ? forward(a, flag) : b; }
        function main() {
            let a = new Item(1);
            let b = new Item(2);
            let sum = 0;
            for (let i = 0; i < 100; ++i) {
                const saved = choose(a, b, i % 2);
                const previous = a;
                a = b;
                b = new Item(previous.value + 1);
                sum += saved.value;
            }
            return sum + a.value + b.value;
        }",
        2753.0,
    );
}

#[test]
fn swapped_loop_values_stay_alive() {
    check(
        "function make(v) { return {v}; }
        function main() {
            let a = make(1);
            let b = make(2);
            let sum = 0;
            for (let i = 0; i < 5; ++i) {
                const t = a;
                a = b;
                b = t;
                sum += a.v * 10 + b.v;
            }
            return sum;
        }",
        87.0,
    );
}

#[test]
fn iterated_elements_outlive_their_source_binding() {
    check(
        "function make() { return [{v: 1}, {v: 5}]; }
        function main() {
            let list = [{v: 1}, {v: 2}, {v: 3}];
            let sum = 0;
            for (const o of list) {
                list = null;
                sum += o.v;
            }
            let last;
            for (const o of make())
                last = o;
            return sum + last.v;
        }",
        11.0,
    );
}

#[test]
fn replaced_loop_values_are_released() {
    check(
        "function main() {
            let current = {v: 0};
            for (let i = 1; i < 500; ++i)
                current = {v: current.v + i, s: 'x' + i};
            return current.v;
        }",
        124750.0,
    );
}

#[test]
fn temporaries_grow_past_their_initial_capacity() {
    let reads = vec!["o.p"; 200].join(", ");
    check(
        &format!(
            "function main() {{
                const o = {{p: {{n: 1}}}};
                let sum = 0;
                for (let i = 0; i < 3; ++i) {{
                    const list = [{reads}];
                    sum += list.length + list[199].n;
                }}
                return sum;
            }}"
        ),
        603.0,
    );
}
