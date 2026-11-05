use js_engine::{Engine, EngineOptions, Value};

fn check(source: &str, expected: f64) {
    for ssa_opt in [true, false] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "property_caches.js").unwrap();
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::number(expected),
            "ssa_opt: {ssa_opt}"
        );
    }
}

#[test]
fn one_site_sees_several_shapes() {
    check(
        "function read(o) { return o.x; }
        function main() {
            const objects = [{x: 1, y: 2}, {y: 20, x: 10}, {x: 100}, {z: 0, y: 0, x: 1000}];
            let sum = 0;
            for (let i = 0; i < 8; ++i)
                sum += read(objects[i % 4]);
            return sum;
        }",
        2222.0,
    );
}

#[test]
fn empty_cache_rejects_dictionary_receivers() {
    check(
        "function read(o) { return o.x; }
        function main() {
            return read({x: 3, 0: 0}) + read({x: 5}) + read({x: 7, 0: 0});
        }",
        15.0,
    );
}

#[test]
fn added_properties_and_dictionaries_miss() {
    check(
        "function read(o) { return o.x; }
        function main() {
            const o = {x: 1};
            let sum = read(o);
            o.y = 2;
            sum += read(o);
            o.x = 4;
            sum += read(o);
            for (let i = 0; i < 70; ++i)
                o['p' + i] = i;
            sum += read(o);
            o.x = 8;
            return sum + read(o) + read({x: 16});
        }",
        34.0,
    );
}

#[test]
fn inherited_properties_follow_their_prototype() {
    check(
        "function C() {}
        C.prototype.x = 5;
        function read(o) { return o.x; }
        function main() {
            const own = new C();
            own.x = 1;
            const inherited = new C();
            let sum = read(own) + read(inherited) + read(own) + read(inherited);
            inherited.x = 100;
            C.prototype.x = 1000;
            return sum + read(inherited) + read(new C());
        }",
        1112.0,
    );
}

#[test]
fn closures_and_primitives_share_sites() {
    check(
        "function f() {}
        f.x = 3;
        function read(o) { return o.x; }
        function method(o) { return typeof o.slice; }
        function main() {
            let sum = read(f) + read({x: 4}) + read(f);
            if (method('text') !== 'function' || method([1]) !== 'function')
                return -1;
            return sum;
        }",
        10.0,
    );
}

#[test]
fn reused_slots_and_retained_values_stay_valid() {
    check(
        "function read(o) { return o.x; }
        function main() {
            let sum = 0;
            for (let i = 0; i < 300; ++i) {
                sum += read({x: i});
                const filler = [i, i];
                sum += filler.length;
            }
            const holder = {x: {v: 7}};
            const kept = [];
            for (let i = 0; i < 3; ++i)
                kept.push(read(holder));
            holder.x = null;
            return sum + kept[0].v + kept[2].v;
        }",
        45464.0,
    );
}

#[test]
fn shadowing_and_prototype_changes_miss() {
    check(
        "function C() {}
        C.prototype.x = 1;
        function read(o) { return o.x; }
        function main() {
            const a = new C();
            const b = new C();
            let sum = read(a) + read(b);
            b.x = 10;
            sum += read(a) + read(b);
            C.prototype.y = 0;
            sum += read(a);
            C.prototype.x = 100;
            sum += read(a);
            for (let i = 0; i < 70; ++i)
                C.prototype['p' + i] = i;
            sum += read(a);
            C.prototype.x = 1000;
            return sum + read(a) + read(b);
        }",
        1224.0,
    );
}

#[test]
fn reused_prototype_slots_stay_valid() {
    check(
        "function read(o) { return o.m; }
        function make(i) {
            function C() {}
            if (i % 2)
                C.prototype.pad = 0;
            C.prototype.m = i;
            return new C();
        }
        function main() {
            let sum = 0;
            for (let i = 0; i < 200; ++i) {
                sum += read(make(i));
                const filler = {m: -1};
                sum += filler.m;
            }
            return sum;
        }",
        19700.0,
    );
}
