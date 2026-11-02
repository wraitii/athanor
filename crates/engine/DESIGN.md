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
  special `globalThis` object remains separate; immutable UTF-8 strings use
  reference-counted heap handles.
  Binding cells and closures use the same checked heap handles.
- Generated functions use `(ctx, closure, receiver, argv, argc) -> Value`. Functions are
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
- Bitwise and shift operators, including compound assignments, use runtime
  helpers with 32-bit integer conversion and shift counts masked to five bits.
  Unsigned right shift returns a nonnegative number. Compound assignments share
  binary operations and evaluate the target and its old value before the right
  operand. Full string-to-number coercion remains unimplemented.
- `typeof` returns heap strings for supported values, identifying closures and
  built-in Error as functions. Undeclared identifier operands return undefined
  as a type name; uninitialized lexical bindings still report an error.
- Helpers report runtime errors through `JitCtx`; generated error checks return
  through the function's return block. Parse, compile, unsupported-feature, and
  runtime errors are distinct. Unsupported constructs generally include spans.

Most built-in methods and prototypes are still
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
writes are rejected. Length reads and shrinking are supported. `map` can create
internal holes for positions removed before visitation; reads return undefined,
and subsequent `map` and `filter` calls skip them. Array literals with holes,
length growth, string keys, descriptors, and prototype mutation remain unsupported.

Array `filter` and `map` snapshot the initial length and visit present positions
with `(value, index, array)`, observing intervening writes and truncation. Their
optional `thisArg` is unsupported except for undefined. `join` supports primitive
values and separators; object-to-string conversion is unsupported.

Ordinary object literals currently use reference-counted allocations containing
a map of UTF-8 names to owned values, sharing property storage and incremental
disposal with Error instances. Literals evaluate properties in source order;
shorthand, nested values, primitive computed keys, and duplicate-key replacement
are supported. Dot and bracket reads return undefined for absent properties;
stores retain the new value before releasing the old one. Objects preserve
identity and aliasing, and cycles share the existing collection limitation.

Accessors, method shorthand, spread, and prototype-setting literal properties
are explicitly rejected. Computed and shorthand __proto__ keys are ordinary
properties. Object-to-property-key conversion remains unsupported. No prototype
lookup, property descriptors, or enumeration order is implemented.

The intended optimized layout is interned property names, shared
shapes mapping names to slots, and contiguous value slots. Begin with helper
lookup; add guarded shape/slot fast paths when useful. Keep indexed elements
separate from named properties. Typed arrays provide contiguous numeric storage
without per-element reference counting.

Ordinary prototype methods are needed by existing map code.

## Strings

String `slice` and the optional `startsWith` position use scalar indices and
require numeric indices. `split` supports literal string separators or undefined,
without a limit. `replace` supports literal string search and replacement without
`$` substitution patterns. These methods reject unsupported coercions.

Strings will use valid UTF-8 storage rather than JavaScript's UTF-16 code units.
This is a deliberate language restriction, not a transparent representation
change. Length, indexing, slicing, and string iteration will count Unicode scalar
values, not bytes or grapheme clusters. An indexed character is a string; an
out-of-range index returns undefined. Combining marks remain separate scalar
values. Supplementary characters count as one character.

Lone surrogate escapes cannot be represented and must be rejected explicitly
rather than replaced silently. Valid surrogate pairs in source escapes decode
to their corresponding scalar value. Host string inputs must also be valid
UTF-8. No implicit Unicode normalization is performed: equality compares exact
contents, and ordering uses lexicographic scalar-value order.

Use immutable, reference-counted heap strings with checked handles and the same
ownership rules as other heap allocations. String equality compares contents,
not allocation identity. Begin with contiguous storage and helper operations;
scalar indexing may scan the UTF-8 sequence. Add indexing metadata or other
representations only when workloads justify their cost. String literals, concatenation, content equality, scalar length and indexing,
and truthiness are implemented. Slicing and full coercion support
remain future work.

## Loop lowering

`jit/emit/loops.rs` lowers loops directly into Cranelift blocks, preserving Oxc
binding identities and capture analysis. While, do-while, and for loops share a
stack of break and continue targets; unlabeled transfers select the innermost
loop. Continue targets the test for while/do-while and the update for for loops.
Labeled statements and transfers remain unsupported.

For-of currently accepts arrays and strings with a single identifier declared
using var, let, or const. Evaluate the iterable once, then retain an internal
iteration state in a compiler-managed local across temporary cleanup. Arrays
read their current length and elements each step, observing growth, shrinking,
and replacements. Strings advance a UTF-8 byte cursor and yield one scalar as a
string, avoiding repeated scans from the start. Normal completion and break
release the iteration state; return and uncaught exceptions use frame cleanup.
Caught exceptions can retain skipped locals until frame exit, as described below.

Lexical loop bindings are uninitialized while evaluating the right-hand side.
Each iteration creates a fresh lexical binding and, when captured, a fresh cell.
Var bindings remain function-scoped and shared across iterations. For-of continue
advances to the next element. Assignment targets, destructuring, async iteration,
and custom iterables are explicitly unsupported. No iterator protocol or
IteratorClose behavior is implemented; these will be required for generators
and other user-defined iterators.

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
implicit `arguments`, and async functions remain unsupported. Recursion through a
captured variable works but can create cycles.

Each engine has one global environment and one code registry. A closure's heap
identity ties it to that engine; foreign heap handles are rejected before code
lookup. Code IDs are engine-local, never executable pointers, and resolve only
against the live engine's registry. Compilations within that engine share global
bindings, so existing closures observe subsequent global writes. A caller's
local bindings do not replace the closure's defining globals.

Execution retains its global environment and refers to the active engine. Code
lookup copies an entry address under a short registry borrow; no registry or
compiler borrow remains held while generated code executes.
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

## Receivers and lexical this

Generated calls pass an explicit receiver alongside the closure. Dot, bracket,
and parenthesized property calls preserve their evaluated base. Detached calls,
including host Engine::call on ordinary functions, pass undefined. All ordinary
functions use strict receiver semantics: no sloppy-mode global substitution or
primitive boxing is performed, regardless of source directives. Script-level
this is globalThis, including nested hostEvaluateSource initializers. Functions
borrow receivers while their caller keeps them alive.

Arrows ignore the call-site receiver. Capture analysis marks arrows using this
and propagates the requirement through enclosing arrows up to the nearest
ordinary function boundary. Those closures own the lexical receiver; arrows
without such a requirement bind undefined and avoid retaining an unused object.
Receiver ownership is released when the closure is reclaimed, and can form
cycles under the same limitations as captured cells.

Generator factories retain the call receiver in the generator allocation.
Every resume passes that original value, independently of the receiver used to
invoke next. Completion releases it after protecting the return value; abandoned
generators release it during disposal. Arrows created in generator bodies follow
the same lexical capture rules and can outlive completion.

Function.prototype.call, apply, bind, prototype lookup, and super remain
unsupported. Basic construction binds a fresh object but does not yet establish
a prototype chain. Array callbacks still reject a non-undefined thisArg.

## Limited synchronous generators

Generator declarations and anonymous generator expressions lower after Oxc
binding and capture analysis. `jit/generator.rs` constructs a resume plan with
numbered suspension points. Initially only direct yield statements in the
function body are accepted; delegated yields and yields inside expressions,
blocks, loops, or exception handlers are rejected explicitly. Ordinary loops and
try/catch can run between suspension points. Async functions remain unsupported.

Compilation produces a factory and a native resume function. Calling the factory
copies and retains arguments and the original closure into a heap generator;
it does not execute the body. Next enters an execution frame and dispatches to
the saved state. Initialization runs only for state zero. Resume blocks reload
bindings from owned frame slots, including shared cells for captured bindings.
The initial implementation preserves all local slots rather than computing
liveness across suspension points.

Yield retains its result, clears expression temporaries, and transfers the
frame's local ownership to the generator without ordinary return cleanup. Next
returns an object containing value and done. Normal return or an uncaught
exception exits the frame and marks the generator completed, releasing its
arguments and original closure. Subsequent next calls return undefined with
done true. Suspended frame disposal releases locals incrementally if the
generator is abandoned. Generator cycles share the heap's collection limitation.
Resume state belongs to each execution frame, so nested generator calls do not
replace the caller's saved state.

The native next method requires a generator receiver. Non-undefined resume
arguments and reentrant next calls report explicit engine errors. Return and
throw methods are not implemented. Generator objects do not yet participate in
the iterator protocol or for-of. Existing for-of lowering stays in loops.rs;
sharing iteration machinery will require completion and iterator-closing rules,
rather than rewriting indexed array/string loops into generators.

## Exceptions

`throw` and `try/catch` use explicit propagation through `JitCtx`, without native
stack unwinding. The context owns an optional thrown `Value`, separate from its
engine-error state. Even undefined can be thrown. Throwing retains the value
before cleanup; entering catch clears the pending state and transfers ownership
to a temporary, then initializes the optional catch binding. Catch bindings can
be captured by closures.

Generated checks send engine failures directly through frame cleanup; catch
blocks only receive JS exceptions. Calls propagate pending exceptions to the
nearest active handler or exit their frame. The compiler tracks nested handlers;
a catch body executes under its enclosing handler, so rethrows propagate outward.
Temporaries are cleared on entering catch. Locals bypassed by an exceptional
exit currently remain owned until replacement or frame exit. Uncaught exceptions
are released and reported as host runtime errors at the outer call boundary.

Finally and destructuring catch bindings are explicitly unsupported. Finally
needs a representation of pending completion that preserves or replaces return,
throw, break, and continue. Propagation accepts arbitrary values.

## Basic constructors and Error objects

`new` evaluates its callee and arguments in order and dispatches through a
constructor helper. Supported native constructors are the global Error and Set
functions, represented by checked heap handles carrying a NativeFunction enum.
Constructor aliases and reassignment use the same global bindings as other
functions. Calling Error without new also creates an error. Ordinary compiled
functions can also construct a fresh object and receive it as this. An object return replaces that instance; primitive returns keep it.
Arrow and generator functions reject construction. Prototype links remain
unsupported.

Error instances have identity and reference-counted named properties, initially
name and message. Both are writable; absent properties read as undefined.
The default name is Error, and an omitted or undefined message becomes an empty
string. Primitive messages convert to strings; arbitrary object conversion is
still unsupported. Errors support string concatenation and uncaught diagnostics
using the current name and message. Stack traces, Error.prototype, explicit
toString methods, subclassing, and instanceof remain unsupported.

Error property stores retain before releasing overwritten values. Disposal
processes one outgoing property edge at a time, under the existing budget.
Properties can form cycles, which share the existing cycle-collection limitation.
The initial property map is not the planned shape-based ordinary object layout.

## Native Set bindings

Set has a Rust-backed heap kind with hashed membership. Entries own their values.
String keys compare UTF-8 contents, objects compare identity, and numeric keys
use SameValueZero, merging NaNs and signed zeros. New Set accepts no argument,
undefined, or null; iterable construction and Set iteration remain unsupported.
Add returns the receiver, has and delete return booleans, clear returns undefined,
and size is read-only. Arbitrary Set properties are not supported.

Native functions have enum identities separate from compiled code IDs. The heap
interns native functions and roots them for the runtime lifetime, so repeated
method lookup preserves function identity. Error calls and construction share
this dispatch. Property calls evaluate the receiver once and resolve the method
before evaluating arguments. Parenthesized members preserve the receiver;
detached methods receive undefined. Set methods validate their receiver.

Clear detaches the membership map into a pending disposal allocation and empties
the live Set immediately. Disposal releases one entry per step, retaining queued
entries until processed. Deleting an entry releases its value immediately and
queues any resulting destruction. Set cycles share the existing collection
limitation. Insertion-order iteration will need additional storage when added.

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

## Runner host bindings and source evaluation

The engine exposes an opt-in Host trait for file reads and directory listings.
The Rust runner installs filesystem implementations and the global functions
hostReadFile, hostListFiles, and hostEvaluateSource. The engine library performs
no filesystem access by default. File reads require valid UTF-8; listings match
the C++ runner's empty/star, suffix-star, and exact filename filters, optionally
recurse, normalize paths, sort them, and return newline-delimited text. Symlinked
directories are not followed during recursion.

Source evaluation compiles into the active engine and runs its initializer on
the existing JitCtx. Function bindings and code entries are installed before the
initializer executes. Nested evaluation can see existing globals, call back into
running code, and make newly compiled functions immediately callable. Compiler,
code-registry, and function-metadata mutation use short interior borrows that end
before executing any initializer. Function-name enumeration returns an owned
snapshot so callers cannot hold a metadata borrow across evaluation.

Nested JS exceptions propagate through the caller's existing exception state,
preserving the thrown value's identity. Host I/O and missing-argument errors
throw basic Error objects. Parse, unsupported-feature, and compilation failures
remain engine errors and bypass JS catch. Source evaluation currently returns
undefined rather than the last expression's completion value. Host callbacks
use owned Rust strings; arbitrary object-to-string argument conversion remains
unsupported.

## C++ boundary

Expose a C ABI with opaque runtime/handle types and fixed-layout values. Keep
Rust containers and ownership types behind it. C++ RAII handles own references;
borrowed values have explicit lifetimes. Native calls must support reentry and
translate errors without unwinding Rust panics or C++ exceptions through JIT code.

Retain the generic argument-vector convention as a fallback. For registered
native functions, describe signatures once and generate typed adapters so known
calls avoid name lookup and generic dispatch. C++ should likewise resolve and
retain JS callable handles rather than repeatedly looking them up. The generated function ABI passes the closure and receiver explicitly. Native
dispatch and generated functions share the receiver rules described above.

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
