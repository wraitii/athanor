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

The runner installs `hostReadFile`, `hostListFiles`, and `hostEvaluateSource`,
and provides `print` plus a basic `console.log`. Source evaluation shares the
active engine, including nested library loads.
Set `JS_JIT_JITDUMP=1` to write a `jit-<pid>-<timestamp>-js.dump` file to the
temporary directory. `samply` and `perf` can consume the dump, and debug
information points at the original JavaScript source so they show JavaScript
lines alongside the assembly. Set `JS_JIT_JITDUMP=2` to instead write one `.jsir` listing per compiled
function; debug information then points at these listings, which contain the
complete Cranelift IR in block order with the originating JavaScript source line
interspersed whenever it changes.

Use `--compile --eval SOURCE` or `--compile SCRIPT` to print Cranelift IR
and a per-function helper effect and reentry admission report without executing
the source. Recognized semantic loops are also printed with their fast-path
guards and fallback. The report identifies unresolved calls and current runtime assumptions;
its admission status does not establish purity or analyze raw memory effects.
Add `--ast` for the analyzed Oxc AST and generator resume plans, or `--asm` for
Cranelift machine-instruction output. Use `--no-ir` to omit IR while retaining
the effect report. The compiler emits IR directly from the AST rather than
constructing a separate lowered AST. Eligible functions get a guarded Math fast
path that resolves each distinct native target once per invocation and falls
back to the original body on mismatch. A nonthrowing probe can also cache one
repeated receiver property for the fast body. `JS_JIT_SSA=0` disables these
specializations and numeric SSA simplification.
Both runners provide deterministic `Math.random()` without a Boost dependency.
The shared bootstrap calls `hostSeedRandom(settings.Seed)` before loading scripts;
a missing seed defaults to zero. The generator matches the game's rand48 sequence.
To try the map workload from the repository root:

```sh
cargo run --quiet --manifest-path cranelift-exp/Cargo.toml -- --map
```

`--map` loads the shared bootstrap and calls its `run` entry point with the
same defaults as the C++ runner. Pass a script path positionally, or use
`--data ROOT`, `--settings JSON`, and `--bootstrap SCRIPT` to override the
defaults. This workload now completes and prints the map summary. The dev
profile uses `opt-level = 1`; use `--release` for performance measurements.
Release builds include line tables for profiling.

The JS numeric SSA pass runs after frontend SSA construction and before
Cranelift optimization. It records boxed constants, helper identities, and
numeric guards, propagates facts through branches and block arguments, and
replaces proven guards with constants. Constant branches are rewritten and
unreachable blocks removed before analysis repeats. It also removes frame stores for locals
that only hold non-owning values, excluding generator resume functions. A
separate control-flow analysis removes temporary clears proven empty; unknown
calls and potentially owning reads remain barriers. Error checks remain unchanged. Set `JS_JIT_SSA=0` to disable it, and `JS_JIT_SSA_DUMP=1` to print
the inferred guard and block-parameter kinds to stderr. Combining these settings
runs the analysis without applying its transformations. The corresponding
embedding options are `EngineOptions::ssa_opt` and `EngineOptions::ssa_dump`.
See [Post-SSA numeric analysis](crates/engine/DESIGN.md#post-ssa-numeric-analysis)
for the annotation model, dataflow rules, pass boundary, and future tiering work.


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
