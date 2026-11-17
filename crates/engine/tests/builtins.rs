use js_engine::{Engine, Value};

fn run(source: &str) -> Result<Value, js_engine::Error> {
    let mut engine = Engine::new().unwrap();
    engine.compile(source, "builtins.js").unwrap();
    Ok(engine.call("main", &[])?.value())
}

#[test]
fn array_callbacks_observe_mutations_and_arguments() {
    let source = r#"function main() {
        let a = [1, 2, 3];
        let b = a.map((v, i, array) => {
            if (i === 0) { array[1] = 8; array.length = 2; }
            return v + i + array.length;
        });
        let count = 0;
        let c = b.filter((v, i, array) => { count++; return i === 1 && array === b; });
        return b.length === 3 && b[0] === 3 && b[1] === 11 && b[2] === undefined && count === 2 && c.join() === "11";
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn array_values_survive_callback_replacement_and_appends_are_ignored() {
    let source = r#"function main() {
        let a = [{v: 7}, {v: 9}];
        let b = a.filter((v, i, array) => {
            if (i === 0) { array[0] = null; array[2] = {v: 10}; }
            return true;
        });
        return b.length === 2 && b[0].v === 7 && b[1].v === 9;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn callback_errors_propagate() {
    assert!(run("function main() { return [].map(3); }").is_err());
    assert!(run("function main() { return [1].map(x => { throw 5; }); }").is_err());
    assert_eq!(
        run("function main() { try { [1].filter(x => { throw 5; }); } catch (e) { return e; } }")
            .unwrap(),
        Value::number(5.0)
    );
}

#[test]
fn strings_use_unicode_scalars() {
    let source = r#"function main() {
        let s = "a😀é/β";
        return s.slice(1, -2) === "😀é" && s.startsWith("é/", 2) &&
            s.split("").join("|") === "a|😀|é|/|β" &&
            s.split("/").join(":") === "a😀é:β" &&
            "abcabc".replace("bc", "x") === "axabc";
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn split_preserves_empty_pieces_and_owned_values() {
    let source = r#"function main() {
        const parts = '|texture||template|'.split('|');
        const whole = 'texture'.split();
        const empty = ''.split('');
        for(let i = 0; i < 100; ++i) { let temporary = 'other'.split('t'); }
        return parts.length === 5 && parts[0] === '' && parts[1] === 'texture' &&
            parts[2] === '' && parts[3] === 'template' && parts[4] === '' &&
            whole.length === 1 && whole[0] === 'texture' && empty.length === 0;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn split_retains_results_and_handles_unicode_boundaries() {
    let source = r#"function make() {
        let source = 'terrain' + '/grass';
        return [source.split('|'), source.split(undefined), source.split('/')];
    }
    function main() {
        let results = make();
        for (let i = 0; i < 200; ++i) { let temporary = ('x' + i).split('x'); }
        let a = 'abc'.split('|');
        let b = 'abc'.split('|');
        a[0] = 'changed';
        return results[0][0] === 'terrain/grass' && results[1][0] === 'terrain/grass' &&
            results[2].join('|') === 'terrain|grass' && a !== b && b[0] === 'abc' &&
            ''.split('|').length === 1 && ''.split('|')[0] === '' &&
            ''.split('').length === 0 && '😀é😀'.split('😀').join('|') === '|é|' &&
            '😀é'.split('').join('|') === '😀|é' &&
            'aaaa'.split('aa').join('|') === '||';
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn unsupported_forms_report_errors() {
    for source in [
        "function main() { return 'a'.split(',', 1); }",
        "function main() { return 'a'.split(1); }",
        "function main() { return 'a'.replace('a', '$&'); }",
        "function main() { return [1].join({}); }",
        "function main() { return [1].map(x => x, {}); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn math_functions_and_global_object() {
    let source = r#"function main() {
        let floor = Math.floor;
        let m = Math;
        Math.extra = 7;
        return floor === m.floor && floor(2.9) === 2 && Math.ceil(-1.2) === -1 &&
            Math.sqrt(9) === 3 && Math.abs(-4) === 4 && Math.imul(4294967295, 3) === -3 &&
            Math.sin(0) === 0 && Math.cos(0) === 1 && Math.PI > 3 && Math.PI < 4 && m.extra === 7;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    assert_eq!(
        run("function main(){ let m = Math; Math = 5; return m.floor(2.5) + Math; }").unwrap(),
        Value::number(7.0)
    );
}

#[test]
fn infinity_is_a_global_number() {
    let source = r#"function main() {
        return Infinity === 1 / 0 && -Infinity === -1 / 0 &&
            typeof Infinity === "number" && globalThis.Infinity === Infinity &&
            1 / Infinity === 0 && -1 / Infinity === 0;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn math_edge_cases() {
    let source = r#"function main() {
        let nan = 0/0;
        return 1/Math.round(-0.5) === -1/0 && Math.round(-1.5) === -1 &&
            Math.round(1.5) === 2 && 1/Math.ceil(-0.1) === -1/0 &&
            1/Math.min(0, -0) === -1/0 && 1/Math.max(-0, 0) === 1/0 &&
            Math.min() === 1/0 && Math.max() === -1/0 &&
            Math.min(3, null, -2, true) === -2 && Math.max(-3, null, 2, true) === 2 &&
            Math.sqrt(9, {}) === 3 && Math.pow(2, 3, {}) === 8 &&
            Math.pow(2) !== Math.pow(2) && Math.sqrt() !== Math.sqrt() &&
            Math.min(1, nan) !== Math.min(1, nan) && Math.pow(1, 1/0) !== Math.pow(1, 1/0) &&
            Math.abs() !== Math.abs() && Math.abs(null) === 0 && Math.sign(true) === 1 &&
            Math.abs('-1') === 1 && Math.min(0/0, '1') !== Math.min(0/0, '1') &&
            Math.floor(' 2.5 ') === 2 && Math.ceil('x') !== Math.ceil('x');
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return Math.max(1, {}); }",
        "function main(){ return Math.max(0/0, {}); }",
        "function main(){ return Math.pow(0/0, {}); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn json_parse_and_stringify() {
    let source = r#"function main() {
        let p = JSON.parse;
        let obj = p('{"z":[null,true,false,"é\\n",2],"a":{"x":3}}');
        let text = JSON.stringify(obj);
        let copy = JSON.parse(text);
        return p === JSON.parse && copy.z[0] === null && copy.z[1] === true &&
            copy.z[2] === false && copy.z[3] === 'é\n' && copy.z[4] === 2 &&
            copy.a.x === 3 && text === '{"z":[null,true,false,"é\\n",2],"a":{"x":3}}';
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    assert_eq!(run("function main(){ return JSON.stringify({a:undefined,b:[undefined,0/0,1/0,-0],c:()=>3}) === '{\"b\":[null,null,null,0]}'; }").unwrap(), Value::TRUE);
    assert_eq!(run("function main(){ return JSON.stringify(undefined) === undefined && JSON.stringify(() => 1) === undefined; }").unwrap(), Value::TRUE);
    assert_eq!(run("function main(){ let a = {x:1}; return JSON.stringify([a,a]) === '[{\"x\":1},{\"x\":1}]'; }").unwrap(), Value::TRUE);
}

#[test]
fn json_rejects_unsupported_or_invalid_inputs() {
    for source in [
        "function main(){ return JSON.parse('{'); }",
        "function main(){ return JSON.parse(1); }",
        "function main(){ return JSON.parse('1', () => 1); }",
        "function main(){ return JSON.stringify({}, null); }",
        "function main(){ return JSON.stringify({}, undefined, 2); }",
        "function main(){ let a = {}; a.a = a; return JSON.stringify(a); }",
        "function main(){ return JSON.stringify({toJSON: () => 1}); }",
        "function main(){ return JSON.stringify(new Set()); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn regular_expression_literals_and_matching() {
    let source = r#"function main() {
        let r = /^(a+)(b)?$/i;
        let a = r.exec('AAA');
        let b = 'aab'.match(r);
        return r.test('Aa') && !r.test('no') && a.length === 3 && a[0] === 'AAA' &&
          a[1] === 'AAA' && a[2] === undefined && b[1] === 'aa' && b[2] === 'b' &&
          r.exec('x') === null && 'x'.match(r) === null && typeof r === 'object' &&
          r.test === r.test && /a/ !== /a/;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    assert_eq!(run(r#"function main(){ return /./.exec('😀')[0] === '😀' && /é/u.test('é') && /^b/m.test('a\nb') && /a.b/s.test('a\nb') && /(?<=a)b/.test('ab') && /(.)\1/.test('éé'); }"#).unwrap(), Value::TRUE);
    assert_eq!(run(r#"function main(){ return JSON.stringify(['a b  c'.split(/\s+/), ''.split(/x*/), 'abc'.split(/x*/), 'a1b'.split(/(\d)|(z)/), 'XaX'.split(/X/g), 'aaa'.replace(/a*/g, '-')]) === '[["a","b","c"],[],["a","b","c"],["a","1",null,"b"],["","a",""],"--"]'; }"#).unwrap(), Value::TRUE);
}

#[test]
fn regular_expression_rejects_unsupported_features() {
    for source in [
        "function main(){ return /a/y; }",
        "function main(){ return /a/d; }",
        "function main(){ return /a/v; }",
        "function main(){ return /(?<x>a)/; }",
    ] {
        let mut engine = Engine::new().unwrap();
        assert!(engine.compile(source, "regex.js").is_err(), "{source}");
    }
    for source in [
        "function main(){ return /a/.test(1); }",
        "function main(){ let f=/a/.exec; return f('a'); }",
        "function main(){ return 'a'.match('a'); }",
        "function main(){ return /a/.unknown; }",
        "function main(){ /a/.x = 1; }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn object_enumeration_uses_own_properties_in_order() {
    assert_eq!(
        run(r#"function main(){
        function C(){} C.prototype.inherited=1;
        let o=new C(); o.b={x:7}; o['2']=2; o['1']=1; o.a=3;
        let keys=Object.keys(o); let values=Object.values(o); o=undefined;
        let a=[1,2,3].map((x,i,a)=>{ if(i===0) a.length=1; return x; });
        let names=Object.getOwnPropertyNames(a); let sum=0;
        for (const key of names) sum+=a[key];
        return keys.join() === '1,2,b,a' && values[2].x===7 &&
            sum===4 && 'é😀'[Object.keys('é😀')[1]]==='😀' &&
            Object.keys(a).join()==='0' && Object.getOwnPropertyNames(a).join()==='0,length' &&
            Object.keys('é😀').join()==='0,1' && Object.values('é😀').join()==='é,😀' &&
            Object.getOwnPropertyNames('é😀').join()==='0,1,length' &&
            Object.keys(1).length===0 && Object.values(false).length===0;
    }"#)
        .unwrap(),
        Value::TRUE
    );
    assert_eq!(run(r#"function main(){
        class C { method(){} } C.x=1; let p=C.prototype;
        return Object.keys(p).length===0 && Object.getOwnPropertyNames(p).join()==='constructor,method' &&
            Object.getOwnPropertyNames(C).join()==='prototype,x' &&
            Object.keys(C).join()==='x' && Object.keys(Math).length===0;
    }"#).unwrap(), Value::TRUE);
}

#[test]
fn freezing_is_explicitly_deferred_and_mutation_remains_supported() {
    assert_eq!(
        run(r#"function main(){
        let o={x:1}; let a=[1];
        if(Object.freeze(o)!==o || Object.freeze(a)!==a) return false;
        o.x=2; o.y=3; a[0]=4;
        return !Object.isFrozen(o) && !Object.isFrozen(a) && o.x===2 && o.y===3 && a[0]===4 &&
            Object.freeze(null)===null && Object.isFrozen(null) && Object.isFrozen('a');
    }"#)
        .unwrap(),
        Value::TRUE
    );
    for source in [
        "function main(){ return Object.keys(null); }",
        "function main(){ return Object.values(undefined); }",
        "function main(){ return Object.getOwnPropertyNames(new Set()); }",
    ] {
        assert!(run(source).is_err());
    }
}

#[test]
fn global_regular_expression_last_index_and_unicode() {
    let source = r#"function main() {
      let r = /./g;
      if (!r.test('é😀') || r.lastIndex !== 1) return false;
      if (!r.exec('é😀') || r.lastIndex !== 2) return false;
      if (r.test('é😀') || r.lastIndex !== 0) return false;
      r['lastIndex'] = 1.9;
      if (r.exec('é😀')[0] !== '😀' || r.lastIndex !== 2) return false;
      r.lastIndex = -2;
      if (r.exec('é😀')[0] !== 'é' || r.lastIndex !== 1) return false;
      r.lastIndex = 1/0;
      if (r.test('é😀') || r.lastIndex !== 0) return false;
      let empty = /(?:)/g;
      if (!empty.test('x') || empty.lastIndex !== 0 || !empty.test('x') || empty.lastIndex !== 0) return false;
      let fixed = /x/; fixed.lastIndex = 5;
      return fixed.test('x') && fixed.lastIndex === 5;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    assert!(run("function main(){ let r=/x/g; r.lastIndex='1'; }").is_err());
}

#[test]
fn global_match_and_replace_handle_empty_matches_and_context() {
    let source = r#"function main() {
       let r = /./g; r.lastIndex=1;
       let a='é😀'.match(r);
       if (a.length !== 2 || a[0] !== 'é' || a[1] !== '😀' || r.lastIndex !== 0) return false;
       let empty='é😀'.match(/(?:)/g);
       if (empty.length !== 3 || empty.join('|') !== '||') return false;
       if ('abc'.match(/x/g) !== null) return false;
       r.lastIndex=1;
       if ('é😀'.replace(r, 'x') !== 'xx' || r.lastIndex !== 0) return false;
       if ('é😀'.replace(/(?:)/g, '-') !== '-é-😀-') return false;
       if ('aba'.replace(/^a/g, 'x') !== 'xba') return false;
       if ('xyxy'.replace(/(?<=x)y/g, 'z') !== 'xzxz') return false;
       let one=/a/; one.lastIndex=7;
       return 'aba'.replace(one, 'x') === 'xba' && one.lastIndex === 7;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return 'a'.replace(/a/g, '$&'); }",
        "function main(){ return 'a'.replace(/a/g, ()=>1); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn array_pop_preserves_returned_values_and_holes() {
    let source = r#"function main() {
        let a = [{x: 7}]; let pop = a.pop; let value = a.pop();
        let holes = [1,2].map((v,i,array) => { if (i === 0) array.length=1; return v; });
        let missing = holes.pop();
        let keys=''; for (let k in a) keys+=k;
        return pop === a.pop && value.x === 7 && a.length === 0 && a.pop() === undefined &&
            missing === undefined && holes.length === 1 && keys === '' && 'pop' in a;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    assert!(run("function main(){ let pop = [].pop; return pop(); }").is_err());
}

#[test]
fn array_index_of_uses_strict_equality_and_numeric_start() {
    let source = r#"function main() {
        let o = {}; let a = [3, 'é', o, undefined, 3, -0];
        let holes = [1,2].map((v,i,array) => { if (i === 0) array.length = 1; return v; });
        return a.indexOf === a.indexOf && a.indexOf('é') === 1 &&
          a.indexOf(o) === 2 && a.indexOf({}) === -1 && a.indexOf(undefined) === 3 &&
          holes.indexOf(undefined) === -1 && a.indexOf(0/0) === -1 && a.indexOf(0) === 5 &&
          a.indexOf(3, 1.9) === 4 && a.indexOf(3, -2) === 4 && a.indexOf(3, -1/0) === 0 &&
          a.indexOf(3, 1/0) === -1 && a.indexOf(3, 0/0) === 0 && a.indexOf(3, -100) === 0 &&
          a.indexOf(3, undefined) === 0 && 'indexOf' in a;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return [1].indexOf(1, '0'); }",
        "function main(){ let f=[].indexOf; return f(1); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
    assert_eq!(
        run("function main(){ return [].indexOf(1, {}); }").unwrap(),
        Value::number(-1.0)
    );
}

#[test]
fn number_methods_format_binary_values_and_radices() {
    let source = r#"function main() {
        let fixed = (2.5).toFixed;
        return fixed === (1).toFixed && (2.5).toFixed() === '3' &&
          (-2.5).toFixed() === '-3' && (1.005).toFixed(2) === '1.00' &&
          (2.55).toFixed(1) === '2.5' && (-0).toFixed(2) === '0.00' &&
          (-0.001).toFixed(2) === '-0.00' && (1e21).toFixed(2) === '1e+21' &&
          (1/0).toFixed(2) === 'Infinity' && (-1/0).toFixed() === '-Infinity' &&
          (0/0).toFixed() === 'NaN' && (4294967295).toString(16) === 'ffffffff' &&
          (-255).toString(16) === '-ff' && (-0).toString(16) === '0' &&
          (1e21).toString() === '1e+21' && (1e-7).toString() === '1e-7' &&
          (1e-6).toString() === '0.000001' && (5)['toString'](2) === '101';
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn number_methods_reject_unsupported_arguments_and_receivers() {
    for source in [
        "function main(){ return (1).toFixed(-1); }",
        "function main(){ return (1).toFixed(101); }",
        "function main(){ return (1).toFixed('2'); }",
        "function main(){ return (1).toString(1); }",
        "function main(){ return (1.5).toString(16); }",
        "function main(){ let f=(1).toFixed; return f(2); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn array_concat_copies_values_and_preserves_holes() {
    let source = r#"function main() {
        let object = {x: 4}; let nested = [object];
        let a = [1,2].map((v,i,array) => { if (i === 0) array.length=1; return v; });
        let b = a.concat(3, [4, nested], undefined, []);
        a[0] = 8; b[2] = 9;
        let empty = [].concat();
        return b.length === 6 && b[0] === 1 && !(1 in b) && b[1] === undefined &&
            b[2] === 9 && b[3] === 4 && b[4] === nested && b[4][0] === object &&
            (5 in b) && b[5] === undefined && a[0] === 8 && empty.length === 0 &&
            a.concat === a.concat && 'concat' in a;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    assert!(run("function main(){ let f = [].concat; return f([1]); }").is_err());
}

#[test]
fn array_push_includes_reverse_and_slice() {
    let source = r#"function main() {
        let object = {x:7}; let a = [];
        if (a.push() !== 0 || a.push(object, 0/0, -0) !== 3) return false;
        if (!a.includes(0/0) || !a.includes(0) || a.includes(object, 1) || !a.includes(object, -1/0)) return false;
        let copy = a.slice(0, -1);
        if (copy.length !== 2 || copy[0] !== object || copy === a) return false;
        if (a.reverse() !== a || a[2] !== object || copy[0] !== object) return false;
        let holes = [1,2,3].map((v,i,array) => { if (i===0) array.length=1; return v; });
        if (!holes.includes(undefined) || holes.indexOf(undefined) !== -1) return false;
        let sliced = holes.slice(0, 3);
        if ((1 in sliced) || (2 in sliced) || sliced[0] !== 1) return false;
        holes.reverse();
        if ((0 in holes) || (1 in holes) || !(2 in holes) || holes[2] !== 1) return false;
        let keys = ''; for (let k in []) keys += k;
        return keys === '' && 'push' in a && 'slice' in a && 'includes' in a && 'reverse' in a &&
          a.slice(-1/0,1/0).length === 3 && a.slice(0/0,2.9).length === 2 &&
          a.slice(2,1).length === 0 && a.push === a.push && a.slice === a.slice;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return [1].includes(1, '0'); }",
        "function main(){ return [1].slice(null); }",
        "function main(){ let f=[].reverse; return f(); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn date_now_is_a_stable_receiver_independent_wall_clock() {
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut engine = Engine::new().unwrap();
    engine.compile("function main(){ let now=Date.now; return now('ignored'); } function inspect(){ return Date.now === Date.now && Object.keys(Date).length === 0; }", "date.js").unwrap();
    let result = engine
        .call("main", &[])
        .unwrap()
        .value()
        .as_number()
        .unwrap();
    let reference = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as f64;
    assert!(
        (result - reference).abs() < 10_000.0,
        "{result} vs {reference}"
    );
    assert_eq!(result.fract(), 0.0);
    assert_eq!(engine.call("inspect", &[]).unwrap().value(), Value::TRUE);
}

#[test]
fn array_from_uses_iterables_array_likes_and_mapping() {
    let source = r#"function main() {
        let mapped = Array.from([1, 2, 3], (value, index) => value + index);
        let typed = new Int8Array(2);
        typed[0] = 3; typed[1] = 4;
        let scaled = Array.from(typed, value => value * 2);
        let text = Array.from("a😀é");
        let like = Array.from({ length: 2, 0: "x", 1: "y" });
        let holes = [1, 2].map((value, index, array) => {
            if (index === 0) array.length = 1;
            return value;
        });
        let fromHoles = Array.from(holes);
        let context = { offset: 10 };
        let thisArg = Array.from([1, 2], function(value) { return value + this.offset; }, context);
        let growing = [1];
        let observed = Array.from(growing, (value, index) => {
            if (index === 0) growing.push(2);
            return growing.length;
        });
        return mapped.join() === "1,3,5" && scaled.join() === "6,8" &&
            text.join() === "a,😀,é" && like.join() === "x,y" &&
            fromHoles.length === 2 && fromHoles[0] === 1 && fromHoles[1] === undefined &&
            thisArg.join() === "11,12" && observed.join() === "2,2" &&
            Array.from === Array.from;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn array_constructor_accepts_lengths_and_values() {
    let source = r#"function main() {
        let empty = Array();
        let length = Array(3);
        let constructed = new Array(2);
        let values = Array(1, "2", 3);
        let single = Array("2");
        return empty.length === 0 && length.length === 3 && length[0] === undefined &&
            constructed.length === 2 && Object.keys(length).length === 0 &&
            values.join() === "1,2,3" && single.length === 1 && single[0] === "2";
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
}

#[test]
fn array_from_and_constructor_reject_invalid_inputs() {
    for source in [
        "function main(){ return Array.from(null); }",
        "function main(){ return Array.from([1], 3); }",
        "function main(){ return Array.from(new Set()); }",
        "function main(){ return Array.from([1], () => { throw 5; }); }",
        "function main(){ return Array(2.5); }",
        "function main(){ return Array(-1); }",
        "function main(){ return new Array(1 / 0); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
    assert_eq!(
        run("function main(){ try { Array.from([1], () => { throw 7; }); } catch (e) { return e; } }")
            .unwrap(),
        Value::number(7.0)
    );
}

#[test]
fn array_flat_flattens_to_the_requested_depth_and_skips_holes() {
    let source = r#"function main() {
        let nested = [1, [2, [3, [4]]]];
        let holes = [1, 2].map((value, index, array) => {
            if (index === 0) array.length = 1;
            return value;
        });
        let zero = nested.flat(0);
        let one = nested.flat();
        let two = nested.flat(2);
        let all = nested.flat(1 / 0);
        let empty = holes.flat();
        let original = [1, [2]];
        let copy = original.flat(0);
        return zero.length === 2 && zero[0] === 1 && zero[1] === nested[1] &&
            one.length === 3 && one[1] === 2 && one[2][0] === 3 &&
            two.length === 4 && two[2] === 3 && two[3][0] === 4 &&
            all.join() === "1,2,3,4" && empty.join() === "1" &&
            copy.length === 2 && copy !== original && copy[1] === original[1] &&
            nested.flat === nested.flat && "flat" in nested;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return [1].flat('1'); }",
        "function main(){ let flat = [].flat; return flat(); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn array_every_and_some_short_circuit_and_skip_holes() {
    let source = r#"function main() {
        let a = [1, 2, 3];
        let everyCalls = 0;
        let every = a.every((value, index, array) => {
            everyCalls++;
            return array === a && value === index + 1 && value < 2;
        });
        let someCalls = 0;
        let some = a.some((value, index, array) => {
            someCalls++;
            return array === a && value === index + 1 && value === 2;
        });
        let holes = [1, 2, 3].map((value, index, array) => {
            if (index === 0) array.length = 1;
            return value;
        });
        let holeCalls = 0;
        let everyHoles = holes.every(value => { holeCalls++; return true; });
        let someHoles = holes.some(value => { holeCalls++; return false; });
        let truncated = [1, 2, 3].some((value, index, array) => {
            if (index === 0) array.length = 1;
            return value === 2;
        });
        return !every && everyCalls === 2 && some && someCalls === 2 &&
            everyHoles && !someHoles && holeCalls === 2 && !truncated &&
            [].every(() => false) && ![].some(() => true) &&
            [0, "", null].every(value => !value) &&
            a.every === a.every && a.some === a.some && 'every' in a && 'some' in a;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return [1].every(1); }",
        "function main(){ return [1].some(1); }",
        "function main(){ return [1].every(value => value, {}); }",
        "function main(){ return [1].some(value => value, {}); }",
        "function main(){ let every = [].every; return every(() => true); }",
        "function main(){ let some = [].some; return some(() => true); }",
        "function main(){ return [1].every(() => { throw 5; }); }",
        "function main(){ return [1].some(() => { throw 5; }); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn array_flat_map_flattens_returned_arrays_one_level() {
    let source = r#"function main() {
        let a = [1, 2, 3];
        let nested = a.flatMap((value, index, array) => [value + index, [value * 10]]);
        let scalars = a.flatMap(value => value * 2);
        let holes = [1, 2, 3].map((value, index, array) => {
            if (index === 0) array.length = 1;
            return value;
        });
        let calls = 0;
        let present = holes.flatMap(value => { calls++; return [value]; });
        let empty = [].flatMap(value => [value]);
        return nested.length === 6 && nested[0] === 1 && nested[1][0] === 10 &&
            nested[2] === 3 && nested[3][0] === 20 && nested[4] === 5 &&
            nested[5][0] === 30 && scalars.join() === "2,4,6" && calls === 1 &&
            present.length === 1 && present[0] === 1 && empty.length === 0 &&
            a.flatMap === a.flatMap && 'flatMap' in a;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return [1].flatMap(1); }",
        "function main(){ return [1].flatMap(value => value, {}); }",
        "function main(){ let flatMap = [].flatMap; return flatMap(value => value); }",
        "function main(){ return [1].flatMap(() => { throw 5; }); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn array_for_each_visits_present_values_and_returns_undefined() {
    let source = r#"function main() {
        let a = [1, 2, 3];
        let seen = [];
        let returned = a.forEach((value, index, array) => {
            seen.push(value + index);
            return {ignored: true};
        });
        let holes = [1, 2, 3].map((value, index, array) => {
            if (index === 0) array.length = 1;
            return value;
        });
        let holeCalls = 0;
        let holeResult = holes.forEach(() => { holeCalls++; });
        let truncatedCalls = 0;
        [1, 2, 3].forEach((value, index, array) => {
            truncatedCalls++;
            if (index === 0) array.length = 1;
        });
        return returned === undefined && seen.join() === "1,3,5" &&
            holeCalls === 1 && holeResult === undefined && truncatedCalls === 1 &&
            a.forEach === a.forEach && "forEach" in a;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return [1].forEach(1); }",
        "function main(){ return [1].forEach(value => value, {}); }",
        "function main(){ let forEach = [].forEach; return forEach(() => {}); }",
        "function main(){ return [1].forEach(() => { throw 5; }); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn array_at_reads_relative_indices_and_preserves_holes() {
    let source = r#"function main() {
        let a = [1, 2, 3];
        let holes = [1, 2, 3].map((value, index, array) => {
            if (index === 0) array.length = 1;
            return value;
        });
        return a.at(0) === 1 && a.at(1) === 2 && a.at(-1) === 3 &&
            a.at() === 1 && a.at(9) === undefined && a.at(-9) === undefined &&
            a.at(1 / 0) === undefined && a.at(-1 / 0) === undefined &&
            holes.at(0) === 1 && holes.at(1) === undefined && holes.at(-1) === undefined &&
            a.at === a.at && "at" in a;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return [1].at('0'); }",
        "function main(){ let at = [].at; return at(0); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn array_find_visits_holes_and_short_circuits() {
    let source = r#"function main() {
        let a = [1, 2, 3, 4];
        let calls = 0;
        let found = a.find((value, index, array) => {
            calls++;
            return array === a && value === index + 1 && index === 2;
        });
        let object = {x: 7};
        let objectFound = [object].find(value => value === object);
        let none = a.find(value => value > 9);
        let zero = [0, 1].find(value => value === 0);
        let holes = [1, 2, 3, 4].map((value, index, array) => {
            if (index === 0) array.length = 1;
            return value;
        });
        let holeCalls = 0;
        let holeFound = holes.find(value => { holeCalls++; return value === undefined; });
        let truncatedCalls = 0;
        let truncated = [1, 2, 3].find((value, index, array) => {
            truncatedCalls++;
            if (index === 0) array.length = 1;
            return index === 2;
        });
        return found === 3 && calls === 3 && objectFound === object &&
            none === undefined && zero === 0 && holeCalls === 2 &&
            holeFound === undefined && truncatedCalls === 3 &&
            [].find(() => true) === undefined && a.find === a.find && "find" in a;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return [1].find(1); }",
        "function main(){ return [1].find(value => value, {}); }",
        "function main(){ let find = [].find; return find(() => true); }",
        "function main(){ return [1].find(() => { throw 5; }); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn array_sort_is_stable_and_preserves_undefined_and_holes() {
    let source = r#"function main() {
        let a = [3, 1, 2];
        let same = a.sort((left, right) => left - right);
        let stable = [{k: 2, v: "a"}, {k: 1, v: "b"}, {k: 2, v: "c"}, {k: 1, v: "d"}];
        stable.sort((left, right) => left.k - right.k);
        let defaultOrder = [10, 2, 1].sort();
        let nanStable = [{v: 2}, {v: 1}, {v: 3}];
        let nanCalls = 0;
        nanStable.sort(() => { nanCalls++; return 0 / 0; });
        let holes = [3, 4, 5].map((value, index, array) => {
            if (index === 0) array.length = 1;
            return value;
        });
        holes[1] = undefined;
        let compared = 0;
        holes.sort((left, right) => { compared++; return left - right; });
        let empty = [];
        let one = [1];
        return same === a && a.join() === "1,2,3" &&
            stable[0].v === "b" && stable[1].v === "d" && stable[2].v === "a" &&
            stable[3].v === "c" && defaultOrder.join() === "1,10,2" &&
            nanStable[0].v === 2 && nanStable[1].v === 1 && nanStable[2].v === 3 &&
            nanCalls > 0 && empty.sort() === empty && one.sort() === one && one[0] === 1 &&
            holes[0] === 3 && holes[1] === undefined && (1 in holes) && !(2 in holes) &&
            compared === 0 && a.sort === a.sort && "sort" in a;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return [1].sort(1); }",
        "function main(){ return [1].sort(null); }",
        "function main(){ let sort = [].sort; return sort((a, b) => a - b); }",
        "function main(){ return [1, 2].sort(() => { throw 5; }); }",
        "function main(){ return [{}, 1].sort(); }",
        "function main(){ return [1, 2].sort(() => ({})); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn array_fill_writes_bounds_returns_receiver_and_clears_holes() {
    let source = r#"function main() {
        let object = {x: 7};
        let a = [1, 2, 3, 4];
        let same = a.fill(object, 1, 3);
        let holes = [1, 2, 3].map((value, index, array) => {
            if (index === 0) array.length = 1;
            return value;
        });
        holes.fill(9, 1);
        let empty = [].fill(1);
        let tail = [1, 2, 3].fill(5, -1);
        let none = [1, 2].fill(0, 2, 1);
        let all = [1, 2].fill(8, -1 / 0, 1 / 0);
        return same === a && a[0] === 1 && a[1] === object && a[2] === object &&
            a[3] === 4 && holes.length === 3 && holes[0] === 1 && holes[1] === 9 &&
            holes[2] === 9 && (1 in holes) && empty.length === 0 &&
            tail[0] === 1 && tail[1] === 2 && tail[2] === 5 &&
            none[0] === 1 && none[1] === 2 && all[0] === 8 && all[1] === 8 &&
            a.fill === a.fill && 'fill' in a;
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return [1].fill(1, '0'); }",
        "function main(){ return [1].fill(1, 0, null); }",
        "function main(){ let fill = [].fill; return fill(1); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn string_pad_start_pads_by_unicode_scalars() {
    let source = r#"function main() {
        let padStart = "x".padStart;
        return padStart === "x".padStart && "5".padStart(3, "0") === "005" &&
            "abc".padStart(2, "0") === "abc" && "x".padStart(4) === "   x" &&
            "x".padStart(4, "ab") === "abax" && "é😀".padStart(4, "β") === "ββé😀" &&
            "x".padStart() === "x" && "x".padStart(3, "") === "x";
    }"#;
    assert_eq!(run(source).unwrap(), Value::TRUE);
    for source in [
        "function main(){ return 'x'.padStart('3'); }",
        "function main(){ return 'x'.padStart(3, 1); }",
        "function main(){ let pad = ''.padStart; return pad(2); }",
    ] {
        assert!(run(source).is_err(), "{source}");
    }
}

#[test]
fn exponentiation_special_values() {
    let source = "function main() {
            let nan = 0 / 0, inf = 1 / 0;
            let a = 1 ** inf, b = 1 ** nan, c = Math.pow(1, nan);
            let d = (-1) ** inf, e = Math.pow(-1, -inf);
            return a !== a && b !== b && c !== c && d !== d && e !== e &&
                nan ** 0 === 1 && Math.pow(nan, 0) === 1 &&
                1 / ((-0) ** 3) === -inf && 1 / Math.pow(-0, 3) === -inf &&
                1 / ((-0) ** 2) === inf && 1 / Math.pow(-0, 2) === inf &&
                ((-0) ** -3) === -inf && Math.pow(-0, -3) === -inf &&
                ((-0) ** -2) === inf && Math.pow(-0, -2) === inf;
        }";
    for ssa_opt in [false, true] {
        let mut engine = Engine::with_options(js_engine::EngineOptions {
            ssa_opt,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "builtins.js").unwrap();
        assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
    }
}

#[test]
fn string_conversion_character_codes_and_ranges() {
    assert_eq!(
        run(r#"function main() {
        const from = String.fromCharCode;
        const s = 'a😀bc';
        return String() === '' && String(undefined) === 'undefined' &&
            String(null) === 'null' && String(false) === 'false' &&
            String(-0) === '0' && String(1e21) === '1e+21' && String('é') === 'é' &&
            from(65, 66, 0xd83d, 0xde00) === 'AB😀' && from(65537, -1) === '\u0001\uffff' &&
            from(NaN, Infinity) === '\0\0' && s.charAt('1') === '😀' &&
            s.charAt(-1) === '' && s.charAt(Infinity) === '' && s.charCodeAt(1) === 128512 &&
            s.charCodeAt() === 97 && s.charCodeAt(99) !== s.charCodeAt(99) &&
            s.substring(3, 1) === '😀b' && s.substring(-1, Infinity) === s &&
            s.substr(-3, 2) === '😀b' && s.substr(2, -1) === '' && s.substr(2) === 'bc';
    }"#)
        .unwrap(),
        Value::TRUE
    );
    assert!(run("function main(){ return String({}); }").is_err());
    assert!(run("function main(){ return String.fromCharCode(0xd800); }").is_err());
}

#[test]
fn parse_int_handles_radices_prefixes_and_primitive_inputs() {
    assert_eq!(
        run(r#"function main() {
        return parseInt(' \ufeff-0XfFtail') === -255 && parseInt('1012', 2) === 5 &&
            parseInt('0x20', 10) === 0 && parseInt('010') === 10 &&
            parseInt('z!', 36) === 35 && parseInt(12.9) === 12 &&
            parseInt('11', '2') === 3 && parseInt('0xff', 4294967296) === 255 &&
            1 / parseInt('-0') === -Infinity && parseInt(true, 36) === 1389110 &&
            parseInt('\u00851') !== parseInt('\u00851') && parseInt('1', 1) !== parseInt('1', 1) && parseInt('0x') !== parseInt('0x') &&
            parseInt() !== parseInt() && Math.LN2 > 0.69 && Math.LN2 < 0.70;
    }"#)
        .unwrap(),
        Value::TRUE
    );
}

#[test]
fn javascript_call_forwards_receiver_arguments_and_exceptions() {
    assert_eq!(
        run(r#"function add(a, b) { return this.offset + a + b; }
    function main() {
        function missing(a) { return this === undefined && a === undefined; }
        function throws(value) { throw value; }
        let caught = false;
        try { throws.call(null, 17); } catch (e) { caught = e === 17; }
        let invalid = false;
        try { Function.prototype.call.call(1, null); } catch (e) { invalid = true; }
        return add.call === Function.prototype.call && 'call' in Math.max &&
            Object.keys(Function.prototype).length === 0 &&
            add.call({offset: 4}, 2, 3) === 9 && missing.call() &&
            Math.max.call(null, 1, 9, 3) === 9 && caught && invalid;
    }"#)
        .unwrap(),
        Value::TRUE
    );
}

#[test]
fn javascript_apply_preserves_receivers_and_array_like_arguments() {
    assert_eq!(
        run(r#"function sum(a, b) { return this.offset + a + b; }
    function main() {
        const receiver = {offset: 4};
        function missing(a) { return this === undefined && a === undefined; }
        const methods = {offset: 7, make: function() { return x => this.offset + x; }};
        const arrow = methods.make();
        return sum.apply === Function.prototype.apply && 'apply' in sum && 'apply' in Math.max &&
            Object.keys(Function.prototype).length === 0 &&
            sum.apply(receiver, {0: 2, 1: 3, length: '2.9'}) === 9 &&
            missing.apply(undefined, null) && missing.apply(undefined, {length: -2}) &&
            arrow.apply({offset: 100}, [3]) === 10 && Math.max.apply(null, [1, 9, 3]) === 9 &&
            String.apply(null, [42]) === '42';
    }"#)
        .unwrap(),
        Value::TRUE
    );
}

#[test]
fn javascript_apply_propagates_exceptions_and_keeps_arguments_alive() {
    assert_eq!(
        run(r#"function main() {
        let list = [{value: 8}];
        function f(a, b) { list[0] = null; return a.value + b; }
        const result = f.apply(null, {0: list[0], 1: 2, length: 2});
        function throws(value) { throw value; }
        let caught = false;
        try { throws.apply(null, [17]); } catch (e) { caught = e === 17; }
        let invalid = false;
        try { f.apply(null, 'ab'); } catch (e) { invalid = true; }
        const old = Function.prototype.apply;
        Function.prototype.apply = function() { return 23; };
        const replaced = f.apply(null, []);
        Function.prototype.apply = old;
        function $call(a, b) { return a + b; }
        return result === 10 && caught && invalid && replaced === 23 && $call(3, 4) === 7 &&
            typeof $callSpread === 'undefined';
    }"#)
        .unwrap(),
        Value::TRUE
    );
}

#[test]
fn hot_apply_keeps_target_guards_and_packed_argument_ownership() {
    assert_eq!(
        run(r#"function sum(a, b) { return this.offset + a.value + b; }
    function product(a, b) { return this.offset + a.value * b; }
    function main() {
        let total = 0;
        const args = [{value: 3}, 2];
        const receiver = {offset: 1};
        for (let i = 0; i < 5000; ++i) {
            const target = i < 2500 ? sum : product;
            total += target.apply(receiver, args);
        }
        return total === 32500 && args[0].value === 3 && args.length === 2;
    }"#)
        .unwrap(),
        Value::TRUE
    );
}
