use js_engine::{Engine, Value};

fn check(source: &str) {
    let mut engine = Engine::new().unwrap();
    engine.compile(source, "typed.js").unwrap();
    assert_eq!(
        engine.call("main", &[]).unwrap().value(),
        Value::TRUE,
        "{source}"
    );
}

#[test]
fn packed_numeric_conversion_and_bounds() {
    check(
        r#"function main(){
      let u = new Uint16Array(3); u[0] = -1; u[1] = 65537.9; u[2] = NaN;
      let i = new Int8Array(3); i[0] = 255; i[1] = 128; i[2] = -129;
      let u32 = new Uint32Array(2); u32[0] = -1; u32[1] = 4294967297;
      let f = new Float32Array(2); f[0] = 16777217; f[1] = -0;
      let d = new Float64Array(1); d[0] = '2.5';
      return u.length === 3 && u[0] === 65535 && u[1] === 1 && u[2] === 0 &&
        i[0] === -1 && i[1] === -128 && i[2] === 127 && u32[0] === 4294967295 &&
        u32[1] === 1 && f[0] === 16777216 && 1/f[1] === -1/0 && d[0] === 2.5;
    }"#
        .replace("NaN", "(0/0)")
        .as_str(),
    );
    check(
        r#"function main(){
      let a = new Uint8Array(2); a[-0] = 9; a['-0'] = 3; a[-1] = 3; a[1.5] = 3;
      a[9] = 3; a['NaN'] = 3; a['Infinity'] = 3;
      return a[0] === 9 && a['-0'] === undefined && a[-1] === undefined &&
        a[1.5] === undefined && a[9] === undefined && a['NaN'] === undefined &&
        a['Infinity'] === undefined && a.length === 2;
    }"#,
    );
}

#[test]
fn indexed_reads_accept_mixed_element_kinds() {
    check(
        r#"function read(array, index) { return array[index]; }
        function main() {
            const arrays = [
                Uint8Array.from([-1]), Uint16Array.from([-1]), Uint32Array.from([-1]),
                Int8Array.from([255]), Int16Array.from([65535]), Int32Array.from([4294967295]),
                Float32Array.from([1.5]), Float64Array.from([2.5])
            ];
            const expected = [255, 65535, 4294967295, -1, -1, -1, 1.5, 2.5];
            for (let i = 0; i < 80; ++i) {
                const k = i % arrays.length;
                if (read(arrays[k], 0) !== expected[k] ||
                    read(arrays[k], -1) !== undefined ||
                    read(arrays[k], 0.5) !== undefined ||
                    read(arrays[k], 1) !== undefined)
                    return false;
            }
            const floats = Float64Array.from([-0, NaN, Infinity]);
            const nan = read(floats, 1);
            return 1 / read(floats, 0) === -Infinity &&
                nan !== nan && read(floats, 2) === Infinity;
        }"#,
    );
}

#[test]
fn typed_arrays_support_fill_iteration_and_branding() {
    check(
        r#"function main(){
      let A = Uint16Array; let a = new A(4); let alias = a;
      if (a.fill(65537,1,-1) !== a || a[1] !== 1 || alias[2] !== 1) return false;
      let sum=0; for (let n of a) sum += n;
      let f=(...v)=>v.join(',');
      let keys=''; for (let k in a) keys+=k;
      return sum===2 && f(...a)==='0,1,1,0' && a instanceof Uint16Array &&
        !(a instanceof Uint8Array) && !Array.isArray(a) && Array.isArray([]) &&
        typeof a === 'object' && keys==='0123' && Object.keys(a).join(',')==='0,1,2,3' &&
        Object.values(a).join(',')==='0,1,1,0' && 'length' in a && 'fill' in a &&
        !('4' in a) && a.fill===a.fill;
    }"#,
    );
}

#[test]
fn typed_array_from_converts_mapped_iterables_and_array_likes() {
    check(
        r#"function main(){
      let a = Int32Array.from([1.9, -2.9, "3"], (value, index) => value + index);
      let source = Int32Array.from([1, 2]);
      let b = Uint8Array.from(source);
      let c = Int32Array.from({length: 2, 0: 4, 1: 5});
      let d = Float64Array.from([1, 2], value => value / 2);
      return a.length === 3 && a[0] === 1 && a[1] === -1 && a[2] === 32 &&
        b.length === 2 && b[0] === 1 && b[1] === 2 &&
        c.length === 2 && c[0] === 4 && c[1] === 5 &&
        d.length === 2 && d[0] === 0.5 && d[1] === 1 &&
        Int32Array.from === Int32Array.from && typeof Int32Array.from === "function";
    }"#,
    );
    for source in [
        "function main(){ return Int32Array.from(null); }",
        "function main(){ return Int32Array.from([1], 3); }",
        "function main(){ return Int32Array.from([1], () => ({})); }",
    ] {
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "typed.js").unwrap();
        assert!(engine.call("main", &[]).is_err(), "{source}");
    }
}

#[test]
fn typed_arrays_reject_unsupported_construction_and_properties() {
    for source in [
        "function main(){ return new Uint8Array(-1); }",
        "function main(){ return new Uint8Array(1/0); }",
        "function main(){ return new Uint8Array(300000000); }",
        "function main(){ return new Uint8Array([1]); }",
        "function main(){ return Uint8Array(1); }",
        "function main(){ let a=new Uint8Array(1); a.length=3; }",
        "function main(){ return new Uint8Array(1)['01']; }",
        "function main(){ let f=new Uint8Array(1).fill; return f(1); }",
        "function main(){ return new Uint8Array(1).fill({}); }",
        "function main(){ let a=new Uint8Array(1); a[-1]={}; }",
    ] {
        let mut engine = Engine::new().unwrap();
        engine.compile(source, "typed.js").unwrap();
        assert!(engine.call("main", &[]).is_err(), "{source}");
    }
}
