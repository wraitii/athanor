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

- Semantic IR is the default: the graph passed to runtime lowering, including
  OSR instrumentation when enabled. `--no-semantic-ir` (also `--no-ir`) suppresses
  it; `--semantic-ir` enables it again. `JS_JIT_SSA=0` selects the unoptimized
  semantic graph; Cranelift still uses its normal optimization level.
- `--clif` adds a Cranelift IR view before and after backend preparation. For
  native code this view is constructed for inspection; the native backend consumes
  the lowered graph. The first view includes a helper-call legend; neither view
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
- `--ast` prints each function's analyzed Oxc parameters/body, or the module AST for its initializer.
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

Use `--after PASS` to inspect runtime lowering boundaries, including `boxing`,
`finalize-runtime`, `scratch`, `cleanup`, `ownership-lower`, `unbox`,
`scalar-semantics`, `scalar-cleanup`, `legalize-convert`, `bits`, and `cranelift-memory`.
`scalar-cleanup` reuses lowered scalar expressions and loads, and hoists pure
loop invariants. Calls and writes invalidate memory reads.
`scalar-semantics` materializes numeric truthiness, wrapping and saturating
conversions before backend preparation. Ordinary scalar conversions and compound
bit operations remain in the shared graph. `legalize-convert`, `bits`, and
`cranelift-memory` show Cranelift preparation, constructing that view even when
compiling with the native backend. Native preparation resolves scalar conversions
and consumes compound bit operations directly. An unknown pass reports the
available pass names.

With `JS_JIT_BACKEND=native-only`, `--after aarch64-select` shows the graph after
native instruction selection, before allocation. Selected operations expose their
actual register operands, including folded addresses and comparisons. Constants
absorbed into immediate operands and unused intermediate operations are removed.

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
directly. Generated code keeps the references it owns in stack slots and
releases them after their last use, at the end of their semantic block, or on
the way out of a failed one. `JS_JIT_SSA=0` disables semantic optimization; the embedding option is
`EngineOptions::ssa_opt`. See
[CLIF emission and ownership](crates/engine/DESIGN.md#clif-emission-and-ownership)
for details.


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

An integer-speculation failure that requests recompilation puts its owning
function into a persistent number-first policy. Later plans use general-number
guards for speculative numeric reads, including inlined reads, while preserving
proven integer facts such as loop indices. Observed integer feedback is retained
separately from this policy. Repeated non-number failures can still disable
numeric speculation at the affected site.

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
Inlining is enabled by default; set `JS_JIT_INLINE=0` to disable JS function
inlining in every compilation tier (`EngineOptions::inlining`).
Calls left after inlining can dispatch to up to four observed function targets.
Each target uses its ordinary entry or a specialized variant guarded at the caller
on integer/number inputs or observed receiver/input shapes. A variant is requested
only for supported guards emitted at an optimized callee's entry. Caller shape
hints can narrow a callee's polymorphic own-property feedback before this analysis;
only hints that produce entry guards become part of the variant contract. Unknown targets
keep using the generic call. Set `JS_JIT_CALL_VARIANTS=0` to disable this dispatch
and variant selection (`EngineOptions::call_variants`). To keep polymorphic direct
call dispatch while disabling specialized callee variants, set
`JS_JIT_CALL_SPECIALIZATION=0` (`EngineOptions::call_specialization`).
Variants compile on first guarded use, using the ordinary body until background
compilation finishes. `JS_JIT_BACKGROUND=0` compiles them synchronously.
New call targets and numeric guard failures replan both the ordinary body and its
active specialized variants. Feedback received during compilation remains pending.
`--planner` lists call alternatives, and specialized callee IR contains
`entry-assume` facts rather than repeated input guards. Shape assumptions expire
across writes and reentry; internal speculation retains its own guards.
Named get caches keep up to four receiver shapes or builtin kinds, allocating
extra entries only after the first. Overflow and repeated uncacheable reads switch
the site to a persistent megamorphic mode. Feedback removes cache-hit speculation
there and uses regular helper reads. Prototype guards and value retention still
apply to every cached entry.
The specialization planner expands polymorphic own-property feedback into local
shape branches. A later rewrite turns guarded reads into direct slot loads. Inlined accesses use the same
feedback; caller shape facts can eliminate checks and unused alternatives. Unknown
shapes retain the cache-hit/refill fallback. Prototype and builtin reads keep
their runtime cache paths.
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

The native AArch64 path skips Cranelift optimization, including during background
recompilation. It selects and allocates directly from the graph produced by shared
runtime lowering. Cranelift optimization still runs for its own backend, including
native backend fallbacks.

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

Set `JS_JIT_SPECIALIZATION_LOG=lin_solve` to log specialization-plan differences
to stderr for function names containing `lin_solve`; use `1` or `*` for all
functions (`EngineOptions::specialization_log`). This works independently of
`JS_JIT_RECOMPILE_LOG`; enable both to correlate decisions with compilation
times, publication and guard failures.

Each record identifies the logical function and whether compilation is generic,
specialized, an OSR fragment, or skipped. `+` adds a decision, `-` removes one,
and `~` changes its value. Entries include relevant feedback, numeric assumptions,
read refinements, inlines, shape accesses and conditional calls. Locations use
source filenames and byte ranges, including sites in inlined functions.
Function plans are compared with the last installed plan, including across
compiler workers. Initial plans and OSR proposals are reported as additions;
an OSR proposal is not compared with the enclosing function's plan. Snapshots
are retained only for matching functions. These are planning decisions before
region lowering; use `--planner` and semantic IR to inspect guards and fallbacks.

For example, from `cranelift-exp/`:

```sh
JS_JIT_SPECIALIZATION_LOG=lin_solve JS_JIT_RECOMPILE_LOG=1 \
  target/release/js-runner --load bench/v8-v7/base.js \
  --load bench/v8-v7/navier-stokes.js --load bench/v8-v7/driver.js \
  --eval 'function main(){}' 2>specialization.log
```

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

### Heap cycle diagnostics

Set `JS_JIT_HEAP_CYCLES=1` to report a heap snapshot to stderr at engine shutdown,
while globals and other engine roots still exist. The diagnostic fully drains
pending reference-count disposal, subtracts heap-internal owning edges from copied
reference counts, and traces from the remaining external owners. It does not
collect cycles or add work to retain/release paths.

`cyclic_components` counts strongly connected groups with more than one member
or a self-edge, not all possible paths around a cycle. `cyclic_nodes` counts their
members; `unreachable_components` and `unreachable_cyclic_nodes` restrict these
to groups without an external owner. `unreachable` also includes acyclic objects
retained by those groups. Kind counts distinguish cycle members from all retained
garbage. `live` counts allocated positive-reference slots after disposal, not
bytes or arena capacity. `arena_bytes` reports allocated slot storage, excluding
out-of-line payload allocations; `scanned_slots` includes freed slots below the
arena high-water mark. Snapshot time includes draining and graph analysis.

This is a synchronous diagnostic with temporary graph storage, not a production
collector or a bounded-pause operation. It assumes reference counts are correct;
missing releases look like external owners. A single shutdown snapshot measures
retention at that boundary, not peak retention or accumulation across repeated
runs. Host handles and globals intentionally keep objects reachable.

```sh
JS_JIT_HEAP_CYCLES=1 cargo run --release --manifest-path cranelift-exp/Cargo.toml -- \
  --map binaries/data/mods/public/maps/random/belgian_uplands.js
```
