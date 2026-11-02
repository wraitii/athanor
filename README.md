# cranelift-exp

Experimental JavaScript engine runners.

## Rust runner

From this directory, run a script's `main` function with
`cargo run -p js-runner -- <script>`, or
use `--eval SOURCE`, `--function NAME`, and `--arg VALUE` to select source,
function, and arguments. Repeat `--load SCRIPT` to compile and initialize files
in order before the main source, sharing one engine and global environment.
Run `cargo run -p js-runner -- --repl` for an
interactive session. Each line is compiled separately, and global declarations
persist between lines. Expressions print their value; enter `exit`, `.exit`, or
EOF to leave. The REPL uses the global name `__repl_result` for expression
results. The engine currently supports only a subset of JavaScript.

The runner installs `hostReadFile`, `hostListFiles`, and `hostEvaluateSource`.
Source evaluation shares the active engine, including nested library loads.
To try the map workload from the repository root:

```sh
cargo run --manifest-path cranelift-exp/Cargo.toml -- \
  --load cranelift-exp/bootstrap.js --eval '
    modRoot = "binaries/data/mods/public";
    hostEvaluateSource(
      hostReadFile(modRoot + "/maps/random/new_rms_test.js")
        .replace("export function* generateMap", "function* generateMap"),
      "new_rms_test.js");
    function main() { return runGenerator(generateMap()); }
  '
```

This exercises file loading and nested source evaluation, but still stops on
unsupported language features in the map libraries; it does not yet generate a
complete map.

## cpp

The C++ runner embeds SpiderMonkey directly and executes 0 A.D. random-map
scripts without linking the game.

```sh
make -C cranelift-exp/cpp
make -C cranelift-exp/cpp run
```

The runner takes an optional script path, data directory, settings JSON, and
bootstrap path:

```sh
cd cranelift-exp/cpp
./build/mapgen-runner \
	--data ../../binaries/data/mods/public \
	../../binaries/data/mods/public/maps/random/new_rms_test.js
```

`make run` executes `new_rms_test.js`. `make run-test` executes
`maps/random/tests/test_Generator.js`. The script path and data directory can be
overridden with `SCRIPT` and `DATA`.

`Engine.GetTemplate` returns a basic synthetic template. This is enough for map
scripts to construct and place entities, but it does not reproduce real game
template values, so the generated map is a workload rather than a faithful map.

Output is a JSON summary with a deterministic hash of the generated heightmap
and tile data, plus generator timing.
