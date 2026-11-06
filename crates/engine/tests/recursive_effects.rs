use js_engine::{Engine, EngineOptions, Value};

fn check(source: &str) {
    for enabled in [false, true] {
        let mut engine = Engine::with_options(EngineOptions {
            ssa_opt: enabled,
            ..Default::default()
        })
        .unwrap();
        engine.compile(source, "recursive_effects.js").unwrap();
        for _ in 0..4 {
            assert_eq!(engine.call("main", &[]).unwrap().value(), Value::TRUE);
        }
    }
}

#[test]
fn descendant_fallback_preserves_owned_values_and_observes_writes() {
    check(
        r#"
        var state;
        var calls;
        function walk(input) {
            const saved = state.value;
            if (typeof input === 'string') return 0;
            const children = input.map(x => walk(x));
            return children[0] + saved.text.length + state.value.text.length;
        }
        function main() {
            state = {value: {text: 'before'}};
            calls = 0;
            walk([['warm']]);
            const leaf = {map: function(callback) {
                calls++;
                state.value = {text: 'after'};
                for (let i = 0; i < 40; ++i) {
                    const garbage = {text: 'temporary' + i};
                }
                return [0];
            }};
            return walk([[leaf]]) === 33 && calls === 1 && state.value.text === 'after';
        }
        "#,
    );
}

#[test]
fn recursive_terrain_fallback_can_replace_constructor_and_target() {
    check(
        r#"
        var calls;
        function SimpleTerrain(text) { this.text = text; }
        function Replacement(text) { this.text = 'new:' + text; }
        function createTerrain(input) {
            return typeof input === 'string' ? new SimpleTerrain(input) : input.map(x => createTerrain(x));
        }
        function replacement(input) { return {text: 'target:' + input}; }
        function main() {
            const originalConstructor = SimpleTerrain;
            const originalTarget = createTerrain;
            calls = 0;
            createTerrain([['warm']]);
            const leaf = {map: function(callback) {
                calls++;
                SimpleTerrain = Replacement;
                return [callback('inside')];
            }};
            const first = createTerrain([[leaf, 'sibling']]);
            const changing = {map: function(callback) {
                calls++;
                createTerrain = replacement;
                return [callback('inside')];
            }};
            const second = createTerrain([[changing, 'sibling']]);
            SimpleTerrain = originalConstructor;
            createTerrain = originalTarget;
            return calls === 2 && first[0][0][0].text === 'new:inside' &&
                first[0][1].text === 'new:sibling' && second[0][0][0].text === 'target:inside' &&
                second[0][1].text === 'target:sibling';
        }
        "#,
    );
}

#[test]
fn recursive_code_guards_recheck_replacements_between_invocations() {
    let padding = "n = n + 0;".repeat(140);
    check(&format!(
        r#"
        var state;
        function left(n) {{
            {padding}
            if (n <= 0) return state.value;
            return right(n - 1) + state.value;
        }}
        function right(n) {{
            if (n <= 0) return state.value;
            return left(n - 1) + state.value;
        }}
        function replacement(n) {{ return 100; }}
        function main() {{
            state = {{value: 2}};
            const before = left(8);
            const saved = right;
            right = replacement;
            const changed = left(8);
            right = saved;
            state.value = 3;
            return before === 18 && changed === 102 && left(8) === 27;
        }}
        "#,
    ));
}

#[test]
fn string_leaf_guards_preserve_constructor_errors_and_single_execution() {
    check(
        r#"
        var separator;
        var constructions;
        var splits;
        function Terrain(texture, template) {
            constructions++;
            if (texture === undefined) throw new Error('missing texture');
            this.texture = texture;
            this.template = template;
        }
        function replacement(texture) {
            constructions++;
            this.texture = 'replacement:' + texture;
        }
        function leaf(input) {
            if (typeof input === 'string') return new Terrain(...input.split(separator));
            return new Terrain(...input.split(separator));
        }
        function main() {
            separator = '|';
            constructions = 0;
            splits = 0;
            const first = leaf('grass|tree');
            const original = Terrain;
            Terrain = replacement;
            const changed = leaf('sand');
            Terrain = original;
            separator = '';
            let caught = false;
            try { leaf(''); } catch (error) { caught = true; }
            const external = leaf({split: function(separator) {
                splits++;
                return ['external', separator];
            }});
            return first.texture === 'grass' && first.template === 'tree' &&
                changed.texture === 'replacement:sand' && caught &&
                external.texture === 'external' && external.template === '' &&
                constructions === 4 && splits === 1;
        }
    "#,
    );
}

#[test]
fn actual_terrain_constructors_preserve_leaf_and_array_validation() {
    let library =
        include_str!("../../../../binaries/data/mods/public/maps/random/rmgen/library.js");
    let start = library.find("function createTerrain(terrain)").unwrap();
    let end = start + library[start..].find("\n}").unwrap() + 2;
    let constructors =
        include_str!("../../../../binaries/data/mods/public/maps/random/rmgen/Terrain.js");
    check(&format!(
        r#"
        var TERRAIN_SEPARATOR = '|';
        {constructors}
        {}
        function main() {{
            TERRAIN_SEPARATOR = '|';
            const result = createTerrain(['grass', ['sand|tree', '']]);
            let emptyArrayError = false;
            try {{ createTerrain([]); }} catch (error) {{ emptyArrayError = true; }}
            TERRAIN_SEPARATOR = '';
            let emptyStringError = false;
            try {{ createTerrain(''); }} catch (error) {{ emptyStringError = true; }}
            return result.terrains[0].texture === 'grass' &&
                result.terrains[1].terrains[0].templateName === 'tree' &&
                result.terrains[1].terrains[1].texture === '' &&
                emptyArrayError && emptyStringError;
        }}
    "#,
        &library[start..end]
    ));
}

#[test]
fn split_element_facts_do_not_survive_mutation_or_escape() {
    check(
        r#"
        function mutate(parts) { parts[0] = 42; }
        function leaf(input, change) {
            if (typeof input !== 'string') return 0;
            const parts = input.split('|');
            const before = parts[0];
            if (change) mutate(parts);
            return typeof before === 'string' && (change ? parts[0] === 42 : parts[0] === input);
        }
        function main() {
            return leaf('grass', false) && leaf('sand', true);
        }
    "#,
    );
}
