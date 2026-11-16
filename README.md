# Athanor

Athanor is a toy JavaScript engine written in Rust, built around a single graph
representation that is progressively optimized and lowered to native code. It
takes a pure-JIT approach, with no interpreter or bytecode tier, using Oxc for
parsing and Cranelift or an experimental AArch64 backend for code generation.
It supports a growing subset of JavaScript, with 0 A.D. random-map generation
as its main real-world workload.

The experiment is in compiler organization: a single evolving graph, in the
spirit of Maglev, whose operations can live at different abstraction levels.
Skipping optional optimizations should make native compilation cheap enough to
mostly dispense with an interpreter. Longer-term goals include reusable
specialization across languages, JIT and AoT compilation, and extensible passes
with dependencies derived from the components they read and write. This is a
playground for those ideas, with deliberate JS deviations, rather than a bid
for full ECMAScript compatibility. See [the design guide](crates/engine/DESIGN.md).

## Getting started

Clone the repository inside a 0 A.D. checkout as `anathor/`:

```sh
git clone git@github.com:wraitii/anathor.git anathor
cd anathor
```

This is a separate Git checkout: commit and push engine changes from this folder
to Anathor. The current development layout assumes the surrounding 0 A.D. checkout
for map scripts, assets, test fixtures, and the SpiderMonkey reference runner.

Run these commands from `anathor/`:

```sh
cargo run -p js-runner -- --eval 'function main(){return 6*7;}'
cargo run -p js-runner -- script.js
cargo run -p js-runner -- --repl
cargo test --workspace
```

The runner calls `main` by default. Use `--function NAME` and repeat `--arg VALUE`
to choose another entry point. Repeat `--load SCRIPT` to initialize libraries
before the main source in the same engine. The REPL preserves global declarations
between lines; use `exit`, `.exit`, or EOF to leave.

Use `--release` for performance measurements. The dev profile uses `opt-level = 1`;
release builds include line tables for profiling.

## Map generation

For now, keep this folder inside the 0 A.D. checkout to use its map scripts and
assets. From this folder:

```sh
cargo run --release -p js-runner -- --map \
  --data ../binaries/data/mods/public \
  ../binaries/data/mods/public/maps/random/belgian_uplands.js
```

From the 0 A.D. repository root, the equivalent command is:

```sh
cargo run --release --manifest-path anathor/Cargo.toml -- --map \
  binaries/data/mods/public/maps/random/belgian_uplands.js
```

`--settings JSON` and `--bootstrap SCRIPT` override the map settings and shared
bootstrap. The runner prints a JSON summary with a deterministic hash of the
heightmap and tile data, plus generator timing. `Math.random()` uses the game's
rand48 sequence, seeded from `settings.Seed` (zero by default).

`Engine.GetTemplate` supplies synthetic templates, so this is a map-generation
workload rather than a faithful reproduction of game output.

The reference runners embed SpiderMonkey and vendored QuickJS respectively:

```sh
make -C cpp
make -C cpp run
make -C cpp-quickjs
make -C cpp-quickjs run
```

SpiderMonkey requires the 0 A.D. checkout's built SpiderMonkey library.
Both runners accept `--data`, `--settings`, `--bootstrap`, and a script path.
Their Makefiles accept `SCRIPT` and `DATA` overrides; `make run-test` runs
`test_Generator.js`. QuickJS provides an interpreter baseline.

## Inspecting compilation

```sh
cargo run -p js-runner -- --compile --semantic-ir \
  --eval 'function f(a,b){return a*b+1;}'
cargo run -p js-runner -- --compile --filter paint --planner --clif --asm script.js
```

Compile mode does not execute the source unless given `--execute 'JS'`.
`--after PASS` inspects a lowering boundary; `--ast` shows the parsed source.
Use `JS_JIT_OPT=none` to compare against unoptimized semantic graphs.

For a compile-time benchmark without inspection output or execution, use:

```sh
cargo build --release -p js-runner
time target/release/js-runner --compile --no-ir script.js
```

Compile mode eagerly compiles all functions in the file, including uncalled and
nested functions. It does not execute library loads. To benchmark a map script,
remove its `generateMap` export as the map bootstrap does:

```sh
sed 's/export function\* generateMap/function* generateMap/' \
  ../binaries/data/mods/public/maps/random/belgian_uplands.js \
  > /tmp/belgian_uplands.js
time target/release/js-runner --compile --no-ir /tmp/belgian_uplands.js
```

See [DESIGN.md](crates/engine/DESIGN.md) for architecture, supported semantics,
ownership, compiler controls, and detailed inspection and profiling instructions.
See [bench/README.md](bench/README.md) for benchmarks and [AGENTS.md](AGENTS.md)
for contributor orientation.
