# Athanor

Athanor is an experimental pure-JIT JavaScript engine written in Rust, with
0 A.D.'s random-map generation scripts as its main real-world workload. It uses
Oxc for parsing, has no interpreter or bytecode tier, and compiles directly to
native code via a custom AArch64 backend or Cranelift.

It started from the idea that Cranelift makes it easy to write a JIT without an
interpreter. That turned out to be true, but getting decent performance meant
building a dedicated mid-level IR for JS semantics, specialization, and
ownership anyway—at which point lowering into Cranelift meant paying for two
optimizing compilers back-to-back. Athanor evolved around a different model: a
single SSA graph whose operations live at different abstraction levels and are
progressively lowered only as far as the target backend needs.

This is a compiler playground with deliberate JS subset restrictions and
semantic deviations, not a bid for full ECMAScript compatibility. See
[the design guide](crates/engine/DESIGN.md) for details.

## Architecture highlights

```text
JS source ─▶ Oxc ─▶ SSA Graph (JS / Regex / LowLevel / AArch64)
                       ├─▶ direct AArch64 baseline (cold code, with OSR)
                       └─▶ inlining & region versioning ─▶ borrow/RC planning
                           ─▶ progressive lowering ─▶ AArch64 or Cranelift
```

- **No interpreter or bytecode**: cold functions go through a fast direct
  AArch64 baseline (or unoptimized lowering) and tier up via call thresholds or
  loop OSR.
- **One progressively-lowered graph**: high-level language ops (JS and a
  spliced regular-expression dialect), mid-level runtime/memory ops, and
  machine ops coexist in the same CFG. The AArch64 backend can consume
  mid-level ops directly instead of lowering them to scalar primitives only to
  pattern-match them back, while Cranelift runs extra lowering passes.
- **Region versioning instead of interpreter deopt**: speculative optimizations
  version regions on the graph, hoist guards outward, and fall back to generic
  regions or extracted continuations compiled with cheaper baseline lowering.
- **Compiler-planned reference counting**: instead of a tracing GC, static
  borrow analysis and ownership planning run while high-level receiver
  information is still present in the graph, avoiding retain/release traffic
  where borrows can be proved.

## Where it stands

- Runs all eight V8 v7 benchmarks and real 0 A.D. random-map scripts (such as
  `belgian_uplands.js`) with deterministic output.
- Native AArch64 codegen is the default on ARM64; Cranelift is available as a
  backend and fallback on other targets.
- On Apple Silicon (see [bench/README.md](bench/README.md) for full tables):
  - Numeric/array-heavy workloads (`navier-stokes`, `gaussian-blur`) are
    competitive with SpiderMonkey 128 and V8 Maglev.
  - Property- and allocation-heavy workloads (`richards`, `deltablue`,
    `raytrace`) beat V8 Sparkplug and QuickJS, but remain 10–30× behind V8
    TurboFan and JavaScriptCore; `belgian_uplands` runs ~9× faster than QuickJS
    and ~2.5× behind SpiderMonkey 128.
  - Compile times are still a weak point compared to mainstream engines.
- Longer-term directions include more component-based pass/analysis
  invalidation and reusing the core compiler for AoT compilation or other
  frontends.

## Getting started

For standalone scripts, `--eval`, `--repl`, and `--compile`, no surrounding
0 A.D. checkout is required:

```sh
git clone git@github.com:wraitii/athanor.git
cd athanor
cargo run -p js-runner -- --eval 'function main(){return 6*7;}'
cargo run -p js-runner -- script.js
cargo run -p js-runner -- --repl
cargo test --workspace
```

The runner calls `main` by default. Use `--function NAME` and repeat `--arg VALUE`
to choose another entry point. Repeat `--load SCRIPT` to initialize libraries
before the main source in the same engine. The REPL preserves global declarations
between lines; use `exit`, `.exit`, or EOF to leave.

Use `--release` for performance measurements. The dev profile uses `opt-level = 1`.
Both profiles include line tables.

## Map generation

The map-generation workload, map-library tests, and reference runners assume
Athanor is cloned inside a 0 A.D. checkout as `athanor/` (or alongside one as a
sibling `../0ad/` checkout):

```sh
git clone git@github.com:wraitii/athanor.git athanor
cd athanor
cargo run --release -p js-runner -- --map \
  --data ../binaries/data/mods/public \
  ../binaries/data/mods/public/maps/random/belgian_uplands.js
```

From the 0 A.D. repository root, the equivalent command is:

```sh
cargo run --release --manifest-path athanor/Cargo.toml -- --map \
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
cargo run -p js-runner -- --compile \
  --eval 'function f(a,b){return a*b+1;}'
cargo run -p js-runner -- --compile --filter paint --after ownership script.js
cargo run -p js-runner -- --compile --filter paint --after ownership \
  --execute 'main()' script.js
```

Compile mode does not execute the source unless given `--execute 'JS'`, which
runs that code after the script loads and also prints the feedback and OSR
recompilations it triggers.
`--after PASS` inspects a compiler pass, and `--after all` prints every pass.
A stage name such as `inline` or `versioning` also selects its `stage/...`
steps. `--diff` prints the first selected pass in full and then only what each
later pass changed (all passes unless `--after` narrows them), and
`--verbose-ir` adds assumption origins to listings.
Use `--after planner` for optimization decisions.
Use `JS_JIT_OPT=none` to compare against unoptimized semantic graphs.

`web/` is a browser inspector that runs the compiler as WebAssembly and steps
through each pass with diffs and source mapping. It needs `wasm-pack` 0.15 and
`bun`:

```sh
cd web
bun install
bun run wasm
bun run dev
```

Without the `native` feature, the compiler targets AArch64 but never maps or
runs code, and only generic compilations are shown since nothing is observed.

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

Tests using map-library fixtures look first in `../binaries`, then in
`../0ad/binaries` for a sibling 0 A.D. checkout.

See [DESIGN.md](crates/engine/DESIGN.md) for architecture, supported semantics,
ownership, compiler controls, and detailed inspection and profiling instructions.
See [bench/README.md](bench/README.md) for benchmarks and [AGENTS.md](AGENTS.md)
for contributor orientation.
