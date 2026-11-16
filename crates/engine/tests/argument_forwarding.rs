use js_engine::{Backend, Engine, EngineOptions, Value};

fn check(source: &str) {
    for backend in [Backend::NativeOnly, Backend::Cranelift] {
        for inlining in [false, true] {
            let mut engine = Engine::with_options(EngineOptions {
                backend,
                inlining,
                background_compile: false,
                ..Default::default()
            })
            .unwrap();
            engine.compile(source, "argument-forwarding.js").unwrap();
            assert_eq!(
                engine.call("main", &[]).unwrap().value(),
                Value::TRUE,
                "{backend:?}, inlining={inlining}"
            );
            engine.drain_disposals(usize::MAX);
        }
    }
}

#[test]
fn inlined_arguments_keep_actual_arity_snapshots_and_owned_values() {
    check(
        "function snapshot(a){a=99;return arguments;}
        function take(){return snapshot(7,{value:11},13);}
        function empty(){return snapshot();}
        function main(){
            let previous=take();
            for(let i=0;i<200;++i){
                let current=take();
                if(current===previous||current.length!==3||current[0]!==7||current[1].value!==11||current[2]!==13)return false;
                previous=current;
            }
            return empty().length===0&&previous[1].value===11;
        }",
    );
}

#[test]
fn forwarding_constructors_keep_receivers_and_overridden_apply() {
    check(
        "function Item(){this.initialize.apply(this,arguments);}
        Item.prototype.initialize=function(a,b){this.value=a.value+b;this.count=arguments.length;};
        function make(){return new Item({value:3},7,11);}
        function main(){
            for(let i=0;i<200;++i){let item=make();if(item.value!==10||item.count!==3)return false;}
            Item.prototype.initialize.apply=function(receiver,args){receiver.value=args[2];receiver.count=args.length;};
            let item=make();
            return item.value===11&&item.count===3;
        }",
    );
}

#[test]
fn forwarded_arguments_preserve_mutations_and_escaping_aliases() {
    check(
        "let saved;
        function target(a,b){return a.value+b;}
        function forward(a){arguments[0]={value:a.value+1};saved=arguments;return target.apply(null,arguments);}
        function main(){
            let total=0;
            for(let i=0;i<200;++i)total+=forward({value:3},7);
            return total===2200&&saved[0].value===4&&saved[1]===7&&saved.length===2;
        }",
    );
}

#[test]
fn fallback_loops_preserve_argument_array_mutations() {
    for backend in [Backend::NativeOnly, Backend::Cranelift] {
        for inlining in [false, true] {
            for allocation in ["[flag,0]", "arguments"] {
                let mut engine = Engine::with_options(EngineOptions {
                    backend,
                    inlining,
                    background_compile: false,
                    ..Default::default()
                })
                .unwrap();
                let source = format!(
                    "let total=0;
                    function mutate(a){{a[1]++;total+=a[1];}}
                    function target(a,b){{return b;}}
                    function main(flag,value){{
                        const args={allocation};
                        if(flag){{
                            for(let i=0;i<3;++i)mutate(args);
                            return total;
                        }}
                        return target(...args);
                    }}"
                );
                engine.compile(&source, "argument-fallback.js").unwrap();
                for (flag, expected) in [(Value::TRUE, 6.0), (Value::FALSE, 0.0)] {
                    assert_eq!(
                        engine
                            .call("main", &[flag, Value::number(0.0)])
                            .unwrap()
                            .value(),
                        Value::number(expected),
                        "{backend:?}, inlining={inlining}, allocation={allocation}"
                    );
                }
                engine.drain_disposals(usize::MAX);
            }
        }
    }
}
