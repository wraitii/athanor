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

Use `--compile --eval SOURCE` or `--compile SCRIPT` to compile all functions
without executing the source. Output is grouped by function. `--filter NAME`
restricts output to functions whose generated name contains NAME; other functions
remain available for specialization and inlining. A filter with no matches is an
error. Compilation options cannot be combined with `--load`, `--function`,
`--arg`, `--map`, or `--repl`.

- Semantic IR is the default: the actual graph passed to CLIF emission, including
  OSR instrumentation when enabled. `--no-semantic-ir` (also `--no-ir`) suppresses
  it; `--semantic-ir` enables it again. `JS_JIT_SSA=0` selects the unoptimized
  semantic graph and disables engine CLIF cleanup; Cranelift still uses its normal
  optimization level.
- `--clif` adds Cranelift IR before and after Cranelift optimization. The first
  view includes the engine's CLIF cleanup and a helper-call legend; neither view
  includes register allocation.
- `--planner` prints the inline budget, available call/read feedback, selected
  inlines, read refinements, conditional effects/identity guards, OSR plans, and
  optimization notes. It also shows the region tree before versioning and the
  replanned regions immediately before their emission: block membership, native
  requirements, numeric assumptions, read probes, hoisted guards, fallback exits,
  typed receivers/reads, and length checks. Rejected regions remain in the notes.
  After versioning, it lists graph-local region IDs, their block sets and
  generic fallback links. These records describe the versioning result,
  before subsequent simplification; they do not describe every side exit.
  Detailed region reports are collected only with this flag.
  Feedback is empty until code has executed.
- `--ast` prints each function's analyzed Oxc parameters/body and generator resume
  plan, or the module AST for its initializer.
- `--asm` prints native code with function-relative byte offsets, relocations,
  trap information, and semantic-location IDs. Instructions without a semantic
  location are explicitly marked `unattributed`.

Semantic instructions and terminators carry hexadecimal `@NNNN` IDs, shared with
CLIF source locations and assembly annotations within that function. These IDs
are distinct from semantic value/instruction names and are local to each graph.
Planner decisions refer to instruction IDs before their transformations.
Compile-only inspection uses the initial specialization plan and base inline
budget; it does not execute initializers or collect warmup feedback.

```sh
cargo run -p js-runner -- --compile --filter paint --planner --clif --asm SCRIPT
```

Add `--execute 'JS'` to run the source initializer and then a JavaScript setup/call
snippet. The snippet runs inside an immediately invoked arrow function appended
to the source, so it can use loops and local variables. Host functions are available.
This mode uses synchronous compilation and prints initial code plus any variants
compiled from runtime feedback, labeled `.feedback@...`. It does not force a new
variant when the planner has no changes. All ordinary dump flags still apply.

For example, pass a real object and inspect guarded method inlining:

```sh
cargo run -p js-runner -- --compile --filter selected --planner --clif --asm \
  --eval 'function selected(object){return object.method(4);}' \
  --execute 'const object = {method: function(x){return x+1;}}; const runner = {call: selected}; for(let i=0;i<32;++i) runner.call(object);'
```

Call feedback records native identities as well as JavaScript targets. `Array.push`
is lowered to append operations and a resulting length after native feedback,
guarded by the callee identity and array receiver. Directly proven array allocations
use the same lowering at compile time without guards, relying on the engine's
fixed named array methods. Other native identities are recorded for inspection but
do not yet gain this lowering.

The indirect call lets `selected` execute and collect its own feedback; direct
calls may instead be statically inlined into the setup function. With a script
path, the snippet can construct instances of the script's classes in the same way.
Without `--execute`, compilation remains non-executing. Appended snippet locations
refer to their positions after the original source in the same filename.

The embedding API configures these views through
`EngineOptions::compile: Option<Arc<CompileOptions>>`. Options are shared with
compiler workers; `EngineOptions` is cloneable rather than copyable.

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

Loop replacement (`EngineOptions::osr`) is on by default; set `JS_JIT_OSR=0` to
disable it. Every eligible loop polls at its header and, after 1024 iterations
with applicable feedback, requests a specialized native fragment. The running
invocation transfers into it at the next header arrival with its current
loop-carried values, so nested loops move into native fragments inner loop
first. `JS_JIT_RECOMPILE_LOG=1` reports requests, queue priorities, publication,
the first transfer, and why ineligible loops were rejected. `JS_JIT_BACKGROUND=0`
compiles fragments synchronously for deterministic comparisons.

With background compilation, initial bodies are generic by default
(`EngineOptions::generic_first`; set `JS_JIT_GENERIC_FIRST=0` to disable). They
keep semantic optimization and observations but skip speculative versioning.
Specialization comes only from feedback recompilation and OSR fragments.

Compilation uses four regular workers by default plus a separate worker reserved
for demanded lazy bodies. Scheduling priorities are demanded lazy compilation,
OSR/feedback optimization, known lazy callees, and the sweep of remaining lazy
bodies, in that order. Demanded work uses the reserved worker; the other three
priorities use the regular pool.
Dispatch promotes a queued lazy body to the reserved worker's queue and waits
while publishing completed results. If a regular worker has already started
that body, dispatch waits for its result without duplicating compilation.
Independent jobs keep compiling without waiting for publication. Dependency
tickets and inherited priorities support future callee preparation.

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
Set `JS_JIT_WORKERS=3` to use three regular compiler threads, or `JS_JIT_WORKERS=1`
to use one (`EngineOptions::compiler_threads`, which must be positive). The
reserved lazy worker is additional to this count. Parsing, graph construction,
and top-level initializer compilation still run on the execution thread.
Calls continue using the current version until the execution thread installs a
completed replacement during dispatch. When JIT dumping is enabled, each worker
writes a separate dump file. Set `JS_JIT_BACKGROUND=0` to recompile
synchronously at the next dispatch instead (`EngineOptions::background_compile`).

Scheduling priority and compilation tier are separate. Demanded lazy, prefetch,
and unlazify jobs use the baseline tier: Cranelift `opt_level=none`, with semantic
optimization still enabled and speculative versioning only when generic-first is
disabled. OSR and feedback jobs use the
optimized tier and `EngineOptions::opt_level` (default `speed`). Both tiers use
the backtracking register allocator. Initial bodies retain the base inlining
budget; OSR and feedback use the recompilation multiplier described above.
Promoting queued prefetch or unlazify work to a demand retains its baseline
tier. A job already running retains its tier. Optimized recompilation can
replace a baseline body even when its specialization plan is unchanged.
`JS_JIT_SSA=0` still disables semantic optimization globally. Source initializers,
compile-only output, and synchronous compilation with `JS_JIT_BACKGROUND=0` use
the optimized tier. The former `JS_JIT_SYNC_OPT_LEVEL` and
`JS_JIT_SYNC_SINGLE_PASS` settings and their `EngineOptions` fields were removed.

Set `JS_JIT_RECOMPILE_LOG=1` (`EngineOptions::recompile_log`) to trace recompilation
to stderr in either mode. Events include feedback sites and types, requests,
outstanding job counts, priorities, tiers, queue wait time, plan changes,
compilation outcomes and durations,
installation, and worker shutdown. Prefetch events report installation, first
observed use (`sync_compilation=avoided`), and discarded late results. Lazy
misses report wait begin/end events and elapsed wait time, including publication
of other results during the wait. Engine shutdown prints the lazy wait count,
total and maximum wait duration, time spent joining compiler threads, and a
prefetch summary; use is detected through completed-call counters, including calls that
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
