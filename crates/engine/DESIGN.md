# JS engine design

## Goals

A small JS engine for 0 A.D., with fast C++ interoperability and controllable
memory reclamation. Map generation is an initial workload; long-lived AI and
simulation state must also work without routine full-heap pauses. Prefer simple
implementation over broad language coverage. Preserve ordinary object identity,
aliasing, and cycles; document deliberate semantic restrictions.

## Current implementation

- `parse.rs` uses Oxc to produce an arena-allocated AST. `jit/compiler.rs` compiles
  it directly to Cranelift native code; there is no interpreter or bytecode tier.
  Parsing storage is temporary, while the compiler retains generated code and
  names referenced by that code. Oxc semantic analysis reports early errors and
  resolves references to binding identities used by the compiler.
- `Engine` owns the compiler, shared global environment, and named function
  registry. Compilation can add functions across source files and execute a
  top-level initializer. Duplicate function names are rejected.
- `Value` is a copyable, NaN-boxed 64-bit word. Heap values contain checked,
  globally unique handles; `OwnedValue` keeps a value and its runtime alive on
  the host. Heap arrays have stable allocations and own their elements. The
  special `globalThis` object remains separate; strings are not implemented.
  Binding cells and closures use the same checked heap handles.
- Generated functions use `(ctx, closure, argv, argc) -> Value`. Functions are
  first-class heap values; calls evaluate the callee before arguments and resolve
  its code ID through the active engine registry. Global function bindings are
  materialized before initializers run, and reassignment affects subsequent
  calls, including `Engine::call`. Missing arguments become undefined.
- Local bindings use Cranelift variables and compiler-managed scopes. Globals
  use runtime helpers and an `Rc<RefCell<...>>` environment containing a string
  map. `globalThis` property access shares those bindings. Each frame retains
  local slots until replacement or return and owns temporary values until the
  end of a full expression or frame; error exits release both.
- The supported subset includes numeric/boolean/null expressions, assignments,
  selected operators, conditionals, basic loops, returns, and dynamic calls.
  Common arithmetic and comparisons have inline numeric paths with helper
  fallbacks. This is not yet a complete implementation of JS binding or coercion
  semantics. Array-to-primitive conversion is not implemented: numeric coercion
  reports a runtime error when object conversion would be required.
- Helpers report runtime errors through `JitCtx`; generated error checks return
  through the function's return block. Parse, compile, unsupported-feature, and
  runtime errors are distinct. Unsupported constructs generally include spans.

Ordinary objects, strings, method calls, and generators are still
missing; arrays and rest alone will not make the bootstrap runnable. The C++
runner independently embeds SpiderMonkey and provides a workload reference,
not Rust-engine bindings.

## Values, objects, and arrays

Heap values use globally unique checked handles in the NaN-box payload. The
runtime maps handles to stable boxed allocations with reference counts. Arrays
contain contiguous `Value` elements; binding cells and closures have separate
heap kinds. Environments and host handles share the
runtime; raw `Value` tokens do not keep it alive. A common object header and
collection metadata are future work for additional heap kinds.

Dense array literals and rest parameters allocate arrays; rest copies trailing
arguments and retains referenced values. Numeric index reads return undefined
past the end. Writes replace an element or append at the current length; sparse
writes are rejected. Length reads and shrinking are supported. Holes, length
growth, string keys, descriptors, and prototype mutation remain unsupported.

For ordinary objects, the intended layout is interned property names, shared
shapes mapping names to slots, and contiguous value slots. Begin with helper
lookup; add guarded shape/slot fast paths when useful. Keep indexed elements
separate from named properties. Typed arrays provide contiguous numeric storage
without per-element reference counting.

Ordinary prototype methods are needed by existing map code.

## Closures and global environments

Arrow functions and anonymous function expressions compile to callable heap
values. Capture analysis follows Oxc binding identities, including transitive
captures used only by a nested closure. Captured bindings use heap cells;
uncaptured locals retain their existing Cranelift variable representation.
Sibling closures share cells, while separate activations allocate separate cells.
Closure allocations own their capture cells, and cells own their current values.
Reads become owned temporaries before side effects can replace their source.

Captured lexical reads and writes check initialization; declarations initialize
cells separately from assignment. Assignment to const bindings is rejected.
Block-local captures get fresh cells each time the block executes. Captured
lexical bindings in a for-loop initializer are explicitly rejected until
per-iteration environments are implemented. Nested named functions/declarations,
`this`, implicit `arguments`, methods, async functions and generators remain
unsupported. Recursion through a captured variable works but can create cycles.

Each engine has one global environment and one code registry. A closure's heap
identity ties it to that engine; foreign heap handles are rejected before code
lookup. Code IDs are engine-local, never executable pointers, and resolve only
against the live engine's registry. Compilations within that engine share global
bindings, so existing closures observe subsequent global writes. A caller's
local bindings do not replace the closure's defining globals.

Execution borrows the engine's code registry and retains its global environment.
Heap closures retain neither the engine nor an Environment, avoiding an external
Rc cycle through the heap. Host handles may outlive the engine and keep captured
values alive, but cannot execute the retired code or transfer it to another
engine. Multiple global environments within one engine are not supported; adding
realms will require explicit originating-environment identity and lifetime rules.
The function metadata queries still describe registered declarations, while
`Engine::call` resolves the current global binding.

Disposal releases capture edges incrementally and releases each cell's value
when that cell is reclaimed. Closure cycles remain allocated until runtime
teardown; cycle collection is still required for persistent workloads.

## Ownership and reclamation

Use non-atomic reference counting on a runtime confined to one thread, with
deferred destruction and separate cycle collection. No moving collector or JIT
stack scanning is required for this ownership model.

- Object fields, array elements, globals, closure captures, and persistent host
  handles own references. Every owning edge must be accounted for.
- `Value` remains a bits-only type: copying its bits does not implicitly retain.
  The compiler and runtime APIs distinguish owned and borrowed values. Initially
  prefer conservative ownership over eliminating retain/release operations.
- Calls borrow arguments while the caller keeps them alive; returned heap
  values transfer an owned reference. Retained arguments must be duplicated.
  Locals release ownership on overwrite, scope exit, or frame exit; temporaries
  release at full-expression boundaries or frame exit, including error paths.
  A borrowed property value cannot survive arbitrary calls unless its lifetime
  is protected.
- Centralize reference stores. Assignment retains the new value before replacing
  and releasing the old one, including self-assignment. Primitive values skip
  reference counting. Native code follows the same ownership rules.
- Environments and host handles retain their runtime. Raw values passed to an
  environment or engine are checked against that runtime; host callers must keep
  an owning handle alive while using a heap value token.

When a count reaches zero, enqueue the object instead of recursively destroying
it. The host drains a bounded amount of disposal work at chosen points. Processing
an object releases its outgoing edges, potentially enqueueing more objects, then
frees its storage. Until those edges are processed, they still retain their
targets. Large arrays need resumable disposal; budget edges/bytes, not just
objects. Deferred destruction trades memory for scheduling control and does not
remove retain/release costs during execution. Allocations and host-handle drops
drain a small slice, and the host can call `Engine::drain_disposals` explicitly.
The budget counts edges and objects, not elapsed time or allocator costs. Cycles
remain allocated until runtime teardown because cycle collection is not yet
implemented.

Do not expose JS finalizers or resurrection initially. Native resource disposal
is explicit. Runtime teardown must also reclaim cycles and pending garbage.

## Cycle collection

Reference counting handles ordinary lifetimes; a separate pass finds groups
whose remaining ownership is entirely internal:

1. Copy candidate objects' reference counts into temporary counters.
2. Subtract each owning edge between candidates from its target's temporary count.
3. Candidates with positive remaining counts have external owners. Preserve
   them and everything reachable from them within the candidate set.
4. The remaining candidates are unreachable and can be reclaimed together.

External ownership includes globals, execution-held values, and C++ handles,
not just globals. Every owning edge within the candidate graph must be enumerable,
including captures and any native payloads participating in that graph. Temporary
counts do not alter real counts. Reclaiming a dead group must handle internal
edges without double releases or accessing already freed members.

Start with a synchronous pass at an outermost host boundary, where no JS frames
are active. This is an initial correctness baseline, not the final latency
solution. A decrement that leaves a positive count can nominate a possible cycle
root; candidate filtering and incremental collection can be refined later.

Incremental analysis cannot simply continue walking a changing graph across
ticks. It needs a consistent accounting protocol for mutations, allocations,
and reclamation between slices. Centralized reference operations leave a place
for that tracking. Scheduling, candidate metadata, and mutation tracking are
deferred decisions; do not claim bounded cycle-collection pauses until implemented
and measured.

## C++ boundary

Expose a C ABI with opaque runtime/handle types and fixed-layout values. Keep
Rust containers and ownership types behind it. C++ RAII handles own references;
borrowed values have explicit lifetimes. Native calls must support reentry and
translate errors without unwinding Rust panics or C++ exceptions through JIT code.

Retain the generic argument-vector convention as a fallback. For registered
native functions, describe signatures once and generate typed adapters so known
calls avoid name lookup and generic dispatch. C++ should likewise resolve and
retain JS callable handles rather than repeatedly looking them up. The call ABI now passes the closure explicitly; methods still need receiver
information.

Pass scalars and IDs directly, copy small value records, and use borrowed numeric
buffers or ownership transfer for bulk data. Borrowing prevents incompatible
resize, destruction, or mutation. Independently owned C++ entities use validated
handles. Native types can have fixed registered fields and methods; arbitrary JS
objects need not share C++ layouts. Batch operations where repeated crossings
and temporary objects dominate.

## Scope and next steps

Defer proxies, weak references, observable GC finalizers, direct eval, and exotic
property behavior. Further restrictions, such as frozen library prototypes,
remain choices to validate against workloads. Unsupported features should fail
clearly rather than accidentally acquiring different semantics.

Implement strings, objects, methods, and native bindings as
workloads require. Add cycle collection before treating the engine
as suitable for persistent arbitrary object graphs; refine pause behavior later.

Use `tests/semantics.rs` for supported behavior and ownership regressions. Exercise
escaping values, self-assignment, error cleanup, native reentry, reachable and
unreachable cycles, and budgeted disposal. Compare supported workloads with the
SpiderMonkey runner. Measure allocation volume, retained/pending bytes,
retain/release traffic, native-call overhead, and worst collection slices on
long-lived workloads; a successful small map does not establish simulation
latency or memory behavior.
