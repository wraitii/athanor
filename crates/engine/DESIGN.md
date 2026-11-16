# Athanor engine design

Athanor is an experimental JIT for a JavaScript subset, driven primarily by
0 A.D.'s random-map scripts. Oxc parses and resolves source; native code is the
only execution tier. There is no interpreter or bytecode representation.
Generic and optimized code both originate in the same semantic graph, although
compilation may also use a direct baseline path. The implementation changes
quickly: implemented mechanisms are described below, with longer-term goals
separated from current behavior. ECMAScript compatibility is not a promise.

The workspace README has build and map commands. `src/engine.rs` is the public
entry point, `src/jit/semantic/` owns the shared graph and transformations, and
`tests/` contains executable examples of supported behavior.

## Design goals

The experiment explores a single evolving graph in the spirit of Maglev,
without requiring one uniform instruction set or abstraction level throughout
compilation. Operations can retain useful semantic structure until a consumer
needs them lowered; the AArch64 backend can consume mid-level instructions.
“Single graph” describes the organization of compilation, not a single fixed IR.

The intention is to mostly skip an interpreter. Varying how much optimization
runs on the graph should offer a range of compilation costs without requiring
another execution tier. Required legalization and correctness work still have
to run when optional optimizations are skipped.

JavaScript is the first frontend and workload, but the design should leave room
for other scripting languages. In particular, specialization machinery should
be as independent of JS semantics as practical, with language-specific facts
and operations supplied by the frontend and runtime. Supporting both JIT and
AoT compilation is a longer-term goal; the current execution model is JIT.

The intended pass architecture is more ECS-like: passes declare the components
they read and write, from which dependencies and invalidation can be derived.
The inspiration is Bevy-style scheduling with some Salsa-like incremental
computation, allowing extensions without a growing manually coordinated pass
sequence. This is design intent; `FunctionState` and the explicit pipeline
currently manage analyses and ordering.

## Embedding and lifetime

`Engine` owns the global environment, compiler/tiering state and function
registry. `compile` prepares source and executes its global initialization;
`compile_only` is for inspection without that execution. `call` invokes a named
function and returns an `OwnedValue`. An optional `Host` supplies file access
and printing; map-specific policy belongs to the runner, not the engine.

A `Value` is a copyable 64-bit token. Numbers are IEEE doubles with NaNs
canonicalized; tagged NaNs encode undefined, null, booleans, strings, objects
and an internal uninitialized value. Object and string tokens borrow their heap
allocation. `OwnedValue` retains the allocation and keeps its heap alive; cloning
retains and dropping releases. Its `value()` method returns a borrowed token,
which must only be used with its originating runtime while an owning handle is
alive. This also applies to arguments passed into the engine and environment.

Public boundaries check heap membership and liveness, but tokens have neither a
generation nor runtime identity. A stale token whose address was reused may be
accepted, including after the original runtime is dropped. Membership checks are
therefore not a substitute for the embedding lifetime contract. See `value.rs`,
`heap/owned.rs`, `environment.rs`, and `engine.rs`.

The heap is reference counted. Heap slots carry metadata used by generated code;
objects may store properties inline, and larger payloads use separate storage.
Releasing the last reference queues destruction. Draining processes a bounded
amount of work and releases outgoing edges incrementally; it is a work budget,
not a real-time latency guarantee. `Engine::drain_disposals` exposes a drain to
embedders. Reference cycles are not collected by this mechanism; the optional
cycle analysis in `heap/cycles.rs` reports them rather than reclaiming them.

## Language boundary

The parser accepts more syntax than the compiler implements. Unsupported forms
are rejected during lowering, often with a source span, rather than interpreted.
`jit/semantic/build.rs` and `jit/syntax.rs` are the authoritative checks; tests
show the currently implemented subset. Examples of explicit rejections include
array literal holes, object spread, optional calls, `finally`, labelled
break/continue, and several destructuring and class forms. These are current
implementation boundaries, not necessarily permanent design decisions.

Supported constructs include closures, classes, generators, exceptions with
`catch`, arrays, typed arrays and a selection of native builtins. Support for a
construct does not imply all standard corner cases or builtins. The globals
installed by `builtins/globals.rs` and the implementations under `builtins/`
define the actual library surface. The runner's seeded random implementation
and synthetic map host are workload-specific. When investigating a discrepancy,
first distinguish rejected syntax, missing library behavior and a bug in a
supported path.

### Builtins

`builtins/mod.rs` declares native builtin identities, their exposure as globals,
namespace members, receiver methods or constructor statics, and their call and
optional construction handlers. The catalog generates static lookup matches and
exhaustive dispatch; it does not introduce a dynamic registry. A normal Rust
builtin needs a catalog entry and its implementation, without changes to the
engine, heap method-name lists, compiler passes or backends. Parameterized Math
and typed-array families share their dispatch routes.

`builtins/globals.rs` installs the catalog into each engine. The heap still owns
value storage and property mechanics, while the library owns method names and
behavior. JIT call helpers forward native calls and construction to the library;
handlers borrow their inputs and return owned results. Implementations still
use `JitCtx` for runtime services, allocation, callbacks and error propagation.

Compiler recognition of selected builtin identities remains an optional fast
path. Every builtin has a runtime implementation that works without recognition.
Adding dedicated IR operations is not a prerequisite for extending the library.
`builtins/setup.js` installs JS library functions using the normal closure and
compilation machinery. Only this trusted setup enables private intrinsics:
`$call` and `$callSpread` lower to ordinary call operations with explicit
receivers, and `$defineHidden` installs a non-enumerable property. Their lowering
is required even without optimizations, and the names are not runtime globals.
Packed calls expose their targets to profiling and guarded inlining. A facility
for registering handwritten IR bodies is not implemented.

`Function.prototype.apply` is a JS wrapper and argument-list helper. Closures and
native callables resolve inherited methods through the shared function prototype;
own closure properties take precedence. `apply` accepts array-like objects and
limits argument lists to 1,048,576 entries. Dynamic `Function` construction and
boxed `String` construction remain unsupported. Array index writes may grow the
backing storage with holes, with a 256 MiB limit; assigning a larger `.length`
remains unsupported.

### Observable differences

Unicode scalar strings are an intentional semantic choice. Other differences
include current implementation gaps; these do not all define a permanent subset:

- Strings are UTF-8 and indexing and length count Unicode scalar values, not
  UTF-16 code units. For example, `'😀'.length` is 1 and `'😀'[1]` is
  `undefined`. Lone surrogate string literals are rejected. String methods and
  iteration also operate on scalars where implemented, including `charAt`,
  `charCodeAt`, `substring` and `substr`. `String.fromCharCode` decodes UTF-16
  arguments into valid Unicode, accepting surrogate pairs and rejecting unpaired
  surrogates.
- General object-to-primitive coercion is currently unsupported. Operations such as
  `[] + 1`, `+[]`, or `[] == true` stop with a runtime error instead of calling
  user conversion methods. Object identity comparisons still work; string
  concatenation has a special case for `Error` values.
- Detached ordinary calls receive `undefined` as `this`, including functions
  written without a strict directive. Member calls receive their evaluated
  base; arrows capture `this` lexically. Global script initialization receives
  `globalThis`.
- `Object.freeze` currently returns its argument without freezing objects or
  arrays. `Object.isFrozen` reports false for those values and true for
  primitives. Mutations after `freeze` therefore still succeed.
- JavaScript `throw` values propagate to `catch`, including through calls and
  callbacks. Engine runtime errors, such as calling `undefined`, bypass JS
  `catch` and return an error to the embedder. Parse and unsupported-syntax
  failures likewise occur outside JS exception handling.

Numbers use binary64, but exact agreement with JS integer arithmetic at precision
limits is not a design requirement. Integer optimizations may depart from JS
where its defined behavior becomes unhelpful, such as a loop whose increment
stops making progress because of floating-point rounding. Such cases are
outside the intended semantic guarantees, even though ECMAScript defines them;
this is not permission to change ordinary numeric behavior or compromise memory
safety. The exact boundary is not yet formalized. See `tests/semantics.rs`,
`tests/receivers.rs`, `tests/builtins.rs`, and `tests/ssa.rs` for current behavior.

Lexical bindings and global object properties have different semantics. Local
bindings can start uninitialized, and captures require shared cells where their
identity or lifetime escapes. Per-iteration scope handling matters for closures.
The global environment maintains bindings and revisions separately from heap
objects; generated code must not treat a cached global as immutable without the
appropriate guards. See `environment.rs`, `semantic/build.rs`, and the closure
and global-cache tests.

## From source to graph

Oxc's AST and semantic information drive graph construction. A graph has basic
blocks, instructions, values, block parameters and explicit terminators. Edges
pass arguments to block parameters, representing SSA merges without embedding
machine-specific phi instructions. Early graph operations still express JS
semantics, scopes, property access and calls; later operations describe runtime
layout and scalar machine operations. This is one progressively lowered graph,
not an independent high-level optimizer followed by a separate backend IR.

`semantic/pipeline.rs` is the ordering authority. Broadly:

1. Build and canonicalize a graph, promoting eligible bindings to SSA and
   simplifying JS expressions.
2. Plan inlining, call variants and specialization using available function
   bodies, effect summaries and observations. Apply guarded versioning and,
   where eligible, instrument loop OSR.
3. Lower JS operations and runtime accesses, with repeated simplification,
   load reuse and facts analysis. Lower cells, properties, typed storage,
   numeric dispatch and helpers.
4. Plan ownership while high-level information needed to prove borrows still
   exists. Lower storage and values, choose representations, insert boxing and
   conversions, then finalize runtime cleanup and ownership operations.
5. Legalize the resulting scalar graph for the selected code generator.

Pass order is significant. Facts, control-flow and effect analyses are cached in
`FunctionState` and must be invalidated when transformations invalidate their
assumptions. Graph verification is performed at selected boundaries; additional
ownership verification is enabled by `JS_JIT_VERIFY=1`. Optimizations are more
restricted for generators and functions with exception handlers. Inspection
with `--semantic-ir --after PASS` is useful when a fact, guard or operation
appears to disappear between stages.

Effects are part of correctness, not just profitability. A reused load or a
hoisted check is valid only while intervening operations cannot mutate what it
depends on. Call summaries and guarded identities permit more precise reasoning;
unknown calls must remain conservative. The graph's facts describe what is
proven on a path, whereas observations describe what has happened at runtime.
An observation alone cannot justify an unchecked operation.

## Speculation, recovery and tiering

Initial code can collect observations at sites such as calls, property accesses
and numeric operations. Compilation uses the observations to choose guarded
specializations, including shape and type checks, direct calls, inlining and
region versions. A guard is a runtime condition; only the success path may use
its associated facts. When it fails, generated code must continue with correct
generic semantics, preserving already performed effects rather than restarting
an effectful computation indiscriminately.

The compiler can create a generic region copy or extract a continuation from a
suitable graph boundary. Continuations receive the live values needed to resume
and may return values for dispatch back to the enclosing graph. Extraction has
explicit eligibility checks: for example, generators, handlers, certain frame
operations and heap views prevent it. Fallbacks and continuations are therefore
choices made under graph constraints, not a universal deoptimization facility.
The relevant implementation is in `semantic/specialization/regions/`,
`semantic/continuation.rs`, `semantic/boundary.rs`, and `semantic/guards.rs`.

Hot eligible loops can also use on-stack replacement. The planner identifies
natural loops with useful observations, computes live inputs, header parameters
and exits, and instruments a poll. A separately compiled loop receives that
state and returns an exit selection and live outputs; if transfer is unavailable
or declined, execution continues in the original loop. This is distinct from a
function-entry replacement and does not apply to every loop. See
`semantic/osr.rs` and `jit/osr.rs`.

Background compilation uses a queue with tickets and dependencies. Workers
produce unpublished code and associated data; the execution side publishes
completed replacements. The engine retains the code and metadata needed by
existing entries while calls or recovery paths can still refer to them. Runtime
state and cache ownership must not be silently shared with worker compilation.
See `jit/recompile.rs`, `jit/compiler.rs`, and the tiering and entry machinery.

## Heap access and ownership in generated code

Shapes describe property layouts and enable cached property access. Fast paths
must check the relevant receiver shape and other conditions before using a cached
slot; writes and layout changes must preserve cache validity. Arrays have
separate indexed storage and distinguish dense from holey arrays. Typed arrays
carry an element kind and require appropriate kind and bounds checks. Heap views
and lowered offsets expose storage to optimized code only while their validity
conditions hold. The slow helpers implement generic behavior when those
conditions are not established. `heap/shape.rs`, `heap/properties.rs`,
`semantic/runtime/` and `jit/helpers/` are the important boundaries.

Generated code cannot rely on Rust's lexical ownership. Operations have
contracts describing borrowed or owned results and operand requirements. The
ownership planner combines those contracts with liveness and proven borrows:
heap values may be rooted by the caller, borrowed from a live receiver, held as
temporaries, or stored in owning frame slots. Slot reuse must release the old
contents, and ownership must transfer correctly across control-flow edges,
returns, helper calls, exceptions and cleanup. A borrowed value must not outlive
the object that keeps it alive or cross an invalidating operation without being
retained. `semantic/borrow.rs`, `semantic/contracts.rs`,
`semantic/ownership.rs`, `semantic/runtime/ownership.rs`, and `jit/session.rs`
show this contract. In particular, helper lowering follows borrow analysis
because it can erase the receiver information that proves a borrow.

Representation selection is separate from semantic type facts and ownership.
Values may be boxed or held in scalar integer, floating-point, boolean or word
representations; operation contracts determine required representations and
boxing/conversion passes bridge boundaries. A representation choice must not
violate the supported coercion and numeric contracts or reference lifetime. See
`semantic/repr.rs`, `semantic/boxing.rs`, and `semantic/contracts.rs`.

## Code generation

The shared graph is finalized with runtime and ownership decisions made before
code generation. Backend consumers need not require every operation to reach
the same abstraction level: AArch64 selection also handles mid-level operations.
Cranelift emission translates the graph to CLIF and uses Cranelift's code generator. The experimental AArch64 backend selects instructions, allocates
registers and emits machine code from the lowered graph; it shares the runtime,
semantic decisions and ownership lowering rather than implementing a second JS
compiler. `jit/finalize/`, `jit/cranelift/`, and `jit/aarch64/` define the
boundary. Backend-specific instruction selection must preserve the shared
operation contracts, ABI and cleanup behavior.

`EngineOptions::backend` selects Cranelift, native, or native-only; the default
is native-only. Native generation is selected on AArch64, while other targets
use Cranelift. Some paths still use Cranelift machinery for signatures, module
management or code without a native graph. Consult `jit/compiler.rs` for the
exact selection and failure behavior rather than assuming all modes provide the
same fallback.

## Inspection and validation

The runner can display AST, graph, planner information, CLIF and assembly. For
example:

```sh
cargo run -p js-runner -- --compile --semantic-ir \
  --eval 'function f(a,b){return a*b+1;}'
cargo run -p js-runner -- --compile --filter f --planner --clif --asm script.js
```

Compile mode does not execute source unless `--execute` is supplied. Use
`--after PASS` to inspect a named pipeline boundary; pass names are listed in
`semantic/pipeline.rs`. `JS_JIT_OPT=none`, `baseline`, or `initial` selects an
optimization level; the default is `full`. Useful switches include
`JS_JIT_BACKEND=cranelift|native|native-only`, `JS_JIT_BACKGROUND=0`,
`JS_JIT_INLINE=0`, `JS_JIT_OSR=0`, `JS_JIT_VERIFY=1`,
`JS_JIT_RECOMPILE_LOG=1`, and `JS_JIT_JITDUMP=1` (or `2` with IR). Defaults
and further controls live in `EngineOptions`.

Run `cargo test --workspace` for compiler or runtime changes. Tests under
`crates/engine/tests/` exercise semantics, ownership, specialization, caches,
continuations and backends. For map-affecting changes, use the release map
command in the workspace README and compare the output hash as well as timing.
The map runner is a useful workload, not a conformance suite.
