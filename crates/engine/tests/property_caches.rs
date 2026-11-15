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
fn partially_filled_overflow_rejects_uncacheable_receivers() {
    check(
        "function read(o) { return o.x; }
        function main() {
            let sum = read({x: 1}) + read({y: 0, x: 2});
            sum += read({x: 4, 0: 0});
            return sum + read({x: 8}) + read({y: 0, x: 16});
        }",
        31.0,
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

#[test]
fn writes_follow_shape_growth_and_preserve_replaced_values() {
    check(
        "function write(o, value) { o.x = value; }
        function main() {
            const o = {};
            write(o, {v: 1});
            const kept = o.x;
            for (let i = 0; i < 80; ++i) {
                o['p' + i] = i;
                write(o, {v: i});
            }
            const other = {y: 0, x: null};
            write(other, kept);
            const fresh = {};
            write(fresh, other.x);
            return o.x.v + other.x.v + fresh.x.v;
        }",
        81.0,
    );
}

#[test]
fn cached_writes_preserve_special_properties() {
    check(
        "function writeLength(o, value) { o.length = value; }
        function writeIndex(o, value) { o.lastIndex = value; }
        function writePrototype(o, value) { o.prototype = value; }
        function C() {}
        function main() {
            const a = [1, 2, 3];
            writeLength(a, 1);
            const r = /a/g;
            writeIndex(r, 2);
            writePrototype(C, {x: 7});
            let keys = 0;
            for (const key in C) ++keys;
            return a.length + r.lastIndex + (new C()).x + keys;
        }",
        10.0,
    );
}

#[test]
fn computed_named_properties_match_named_access() {
    check(
        "function read(f, k) { let sum = 0; for (let i = 0; i < 4; ++i) sum += f[k]; return sum; }
        function main() {
            function f() {};
            f.x = 10;
            let k = 'x';
            let sum = read(f, k);
            f[k] = 42;
            let a = [1, 2, 3];
            let length = 'length';
            a[length] = 1;
            let index = '0';
            a[index] = 7;
            let isArray = 'isArray';
            return sum + f.x + a.length + a[0] + (Array[isArray](a) ? 1 : 0);
        }",
        91.0,
    );
}

#[test]
fn computed_index_probes_follow_index_semantics() {
    check(
        "function read(a, k) { let n = 0; for (let i = 0; i < 4; ++i) n += a[k]; return n; }
        function main() {
            let a = [3, 4], t = new Uint8Array(2);
            t[0] = 5; t[1] = 6;
            let minusZero = '-0';
            return read(a, '0') + read(a, '1') + read(t, '0') + read(t, '1') +
                (t[minusZero] === undefined ? 1 : 0) + (t[-0] === 5 ? 1 : 0);
        }",
        74.0,
    );
    for ssa_opt in [true, false] {
        for expression in ["a['01']", "a['-0']", "t['01']"] {
            let mut engine = Engine::with_options(EngineOptions {
                ssa_opt,
                ..Default::default()
            })
            .unwrap();
            let source = format!(
                "function main() {{ let a = [1], t = new Uint8Array(1); let n = 0; for (let i = 0; i < 4; ++i) n += {expression}; return n; }}"
            );
            engine.compile(&source, "property_caches.js").unwrap();
            assert!(
                engine.call("main", &[]).is_err(),
                "{expression}, ssa_opt: {ssa_opt}"
            );
        }
    }
}

#[test]
fn constructor_additions_cross_capacity_and_keep_children() {
    check(
        "function C(v) {
            this.a = {v: v}; this.b = 'kept' + v;
            this.c = 3; this.d = 4; this.e = 5; this.f = 6;
            this.g = 7; this.h = 8; this.i = 9;
        }
        function main() {
            const kept = [];
            for (let i = 0; i < 100; ++i) kept.push(new C(i));
            let sum = 0;
            for (let i = 0; i < 100; ++i) {
                const o = kept[i];
                sum += o.a.v + o.i;
                if (o.b !== 'kept' + i) return -1;
                let keys = 0;
                for (const key in o) ++keys;
                if (keys !== 9) return -2;
            }
            return sum;
        }",
        5850.0,
    );
}

#[test]
fn addition_site_handles_layouts_prototypes_and_replacements() {
    check(
        "function A() {} function B() {}
        A.prototype.p = 3; B.prototype.p = 5;
        function write(o, v) { o.x = v; }
        function main() {
            let sum = 0;
            for (let i = 0; i < 100; ++i) {
                const o = i % 2 ? new A() : new B();
                if (i % 3) o.pad = 0;
                write(o, {v: i});
                const kept = o.x;
                write(o, kept);
                write(o, null);
                sum += kept.v + o.p;
            }
            return sum;
        }",
        5350.0,
    );
}

fn check_specialized(source: &str, expected: f64) {
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    engine.compile(source, "property_caches.js").unwrap();
    for _ in 0..3 {
        assert_eq!(
            engine.call("main", &[]).unwrap().value(),
            Value::number(expected)
        );
    }
}

#[test]
fn shaped_probes_preserve_nested_reads_and_receiver_fallbacks() {
    check_specialized(
        "function read(o) { return o.x.v + o.y.v; }
        function main() {
            let sum = 0;
            const o = {x: {v: 3}, y: {v: 4}};
            for (let i = 0; i < 20; ++i)
                sum += read(o);
            sum += read({y: {v: 5}, x: {v: 6}});
            o.x = {other: 1, v: 8};
            sum += read(o);
            return sum;
        }",
        163.0,
    );
    let mut engine = Engine::with_options(EngineOptions {
        background_compile: false,
        ..Default::default()
    })
    .unwrap();
    engine
        .compile(
            "function read(o) { return o.x * o.y; }
        function main() {
            for (let i = 0; i < 20; ++i)
                read({x: 3, y: 4});
            return 12;
        }",
            "shaped_probe_error.js",
        )
        .unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::number(12.0)
    );
    let error = engine
        .call("read", &[Value::number(42.0)])
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("cannot read property \"x\" of number"));
}

#[test]
fn shape_guards_fall_back_when_the_layout_changes() {
    check_specialized(
        "function bump(o) { o.n = o.n + 1; return o.n + o.m; }
        function main() {
            let sum = 0;
            for (let i = 0; i < 200; ++i)
                sum += bump({n: i, m: 1});
            sum += bump({m: 5, n: 10});
            sum += bump({n: 1, m: 2, k: 3});
            const base = {m: 100};
            function Child() { this.n = 7; }
            Child.prototype = base;
            sum += bump(new Child());
            return sum;
        }",
        20428.0,
    );
}

#[test]
fn shaped_overwrites_release_replaced_values() {
    check_specialized(
        "function store(o, v) { o.v = v; }
        function main() {
            const o = {v: null};
            let sum = 0;
            for (let i = 0; i < 300; ++i) {
                store(o, {n: i});
                sum += o.v.n;
            }
            store(o, Math.abs);
            store(o, {n: 1000});
            sum += o.v.n;
            store(o, 5);
            return sum + o.v;
        }",
        45855.0,
    );
}

#[test]
fn polymorphic_gets_mix_builtin_methods_and_own_properties() {
    check(
        "function invoke(o){return o.slice(1).length;}
        function main(){
            const a={slice:function(n){return 'abcd';}};
            const b=[0,1,2];
            const c='abcde';
            const d={pad:0,slice:function(n){return 'ab';}};
            const objects=[a,b,c,d];
            let sum=0;
            for(let i=0;i<40;++i)sum+=invoke(objects[i%4]);
            return sum;
        }",
        120.0,
    );
}

#[test]
fn polymorphic_prototype_gets_preserve_retained_values_after_overflow() {
    check(
        "function A(){} function B(){} function C(){} function D(){}
        A.prototype.x={v:1}; B.prototype.x={v:2}; C.prototype.x={v:3}; D.prototype.x={v:4};
        function read(o){return o.x;}
        function main(){
            const objects=[new A(),new B(),new C(),new D()];
            let sum=0;
            for(let i=0;i<40;++i)sum+=read(objects[i%4]).v;
            const kept=[read(objects[0]),read(objects[1]),read(objects[2]),read(objects[3])];
            A.prototype.x={v:100};
            objects[1].x={v:200};
            for(let i=0;i<4;++i)sum+=read(objects[i]).v+kept[i].v;
            return sum;
        }",
        417.0,
    );
}
