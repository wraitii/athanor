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
semantic IR in block order with the originating JavaScript source line
interspersed whenever it changes. Profile samples are attributed to semantic
instructions and terminators where machine-code source locations are available;
generated code without a semantic location points to the listing header.

Use `--compile --eval SOURCE` or `--compile SCRIPT` to print Cranelift IR
without executing the source. Add `--semantic-ir` to print the semantic graph,
inferred types, and optimization notes, including why a function was not
versioned or optimized. Add `--ast` for the analyzed Oxc AST and generator
resume plans, or `--asm` for disassembled machine code with relocations. Use
`--no-ir` to omit IR. The compiler lowers the AST to a semantic graph before
emitting Cranelift IR. Eligible functions get a guarded Math fast path that
checks native identities once per invocation and falls back on mismatch.
`JS_JIT_SSA=0` disables semantic graph optimization and CLIF cleanup.
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

Numeric facts and representation selection run on the semantic graph before
CLIF emission. Numeric index operands select their specialized fallback helper
directly. A small effect analysis removes guarded temporary clears when every
incoming path is known clean, tracking inline pushes explicitly. Unknown calls
and potentially owning reads remain barriers. `JS_JIT_SSA=0` disables semantic
optimization and CLIF cleanup; `JS_JIT_SSA_DUMP=1` reports removed temporary
clears. The embedding options are `EngineOptions::ssa_opt` and
`EngineOptions::ssa_dump`. See [CLIF cleanup](crates/engine/DESIGN.md#clif-cleanup)
for annotations and temporary-stack effects.


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

Set `JS_JIT_GET_REFINEMENT=0` to disable get feedback and refinement while keeping
other optimizations enabled (`EngineOptions::get_refinement`).

Set `JS_JIT_OSR=1` to enable experimental loop replacement
(`EngineOptions::osr`). Hot eligible loops request a specialized native fragment
and can enter it on a later loop entry during the same function invocation.
The initial implementation supports one loop per emission, numeric outputs,
and transfer at the preheader; it does not transfer an iteration already in
progress. `JS_JIT_RECOMPILE_LOG=1` reports requests, queue priorities, publication,
and the first transfer. `JS_JIT_BACKGROUND=0` compiles fragments synchronously
for deterministic comparisons.

Compilation uses a shared priority queue with two workers by default. Active
loop jobs have highest priority; ordinary function jobs follow; whole-function
recompilation after an OSR request follows those. After synchronous initial
compilation, known callees and calls exposed by inlining are queued at the
lowest priority. Background completion does not recursively prefetch more callees.
Dispatch cancels a queued job when it needs to compile the same function
synchronously; already-running initial results are discarded if synchronous
compilation finishes first. Dependency tickets and inherited priorities support
future callee preparation.

Feedback requests are submitted at function return to the background compiler pool
once the function has completed eight calls. Earlier feedback remains pending until
that threshold is reached; calls before feedback also count. An OSR request makes
the enclosing function eligible immediately and queues it after the loop fragment.
Feedback recompilation and OSR multiply both the shared inlining instruction
budget and the per-callee size limit by two; initial and prefetched compilation
retain the base limits. Set
`JS_JIT_RECOMPILE_INLINE_MULTIPLIER=1` to use the base budget for every compilation
(`EngineOptions::recompile_inline_multiplier`). Zero disables inlining during
recompilation.
Set `JS_JIT_WORKERS=3` to use three compiler threads, or `JS_JIT_WORKERS=1`
to use one (`EngineOptions::compiler_threads`, which must be positive).
Calls continue using the current version until the execution thread installs a
completed replacement during dispatch. When JIT dumping is enabled, each worker
writes a separate dump file. Set `JS_JIT_BACKGROUND=0` to recompile
synchronously at the next dispatch instead (`EngineOptions::background_compile`).

Set `JS_JIT_RECOMPILE_LOG=1` (`EngineOptions::recompile_log`) to trace recompilation
to stderr in either mode. Events include feedback sites and types, requests,
outstanding job counts and queue wait time, plan changes, compilation outcomes and durations,
installation, and worker shutdown. Prefetch events report installation, first
observed use (`sync_compilation=avoided`), synchronous takeover of queued or
running work, and discarded late results. Engine shutdown prints a prefetch
summary; use is detected through completed-call counters, including calls that
bypass runtime dispatch. Each line includes elapsed time since engine
creation; function events include the logical code ID and function name. Requests
made while a job is outstanding are marked `compiling=true`. An `unchanged` plan
keeps the current entry without emitting replacement code. Logging is disabled
by default and can affect timing when enabled.

## cpp-quickjs

The QuickJS runner mirrors the SpiderMonkey C++ runner while embedding QuickJS
instead. QuickJS is vendored under `cpp-quickjs/quickjs` so the folder is
self-contained; see its MIT license there. It implements the same host
functions, CLI arguments, deterministic `Math.random()`, and JSON summary, so
the generated map hashes match the SpiderMonkey runner.

```sh
make -C cranelift-exp/cpp-quickjs
make -C cranelift-exp/cpp-quickjs run
```

`make run-test` executes `maps/random/tests/test_Generator.js`. The script path
and data directory can be overridden with `SCRIPT` and `DATA`.

Because QuickJS has no JIT, the map workload is much slower than under
SpiderMonkey. For reference, `belgian_uplands.js` takes about 7.6 seconds
under QuickJS, versus about 0.34 seconds under SpiderMonkey on the same
machine. It is useful as a baseline for the experimental engine.

### Profiling

QuickJS emits no perf map or jitdump, and its internal bytecode line table is
not exposed to native profilers, so `samply` or `perf` see only the interpreter
and C builtins (`JS_CallInternal`, `JS_GetPropertyInternal`, ...), not JS
function names.

QuickJS has an upstream `OPCODE_ASM_LABEL` debug option that gives each
bytecode dispatch case a global symbol. Build with it to attribute samples to
opcodes:

```sh
cd cranelift-exp/cpp-quickjs
make clean
make QUICKJS_EXTRA_CFLAGS='-DOPCODE_ASM_LABEL -fno-omit-frame-pointer'
samply record ./build/mapgen-runner --data ../../binaries/data/mods/public \
	../../binaries/data/mods/public/maps/random/belgian_uplands.js
```

Samples then show entries such as `label_OP_get_loc_check` and
`label_OP_get_array_el` alongside native helpers. This still does not identify
the JS function or line. For that, sample from `JS_SetInterruptHandler` and use
`JS_GetScriptOrModuleName`, or read an Error's `.stack`; both are in-process
instrumentation rather than profiler output.

Upstream `qjs`'s `-T/--trace` option traces memory allocation, and `-d/--dump`
prints memory usage statistics. Neither is a CPU profiler.
