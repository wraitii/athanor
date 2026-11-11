# JS engine design

## Goals

A small JS engine for 0 A.D., with fast C++ interoperability and controllable
memory reclamation. Map generation is an initial workload; long-lived AI and
simulation state must also work without routine full-heap pauses. Prefer simple
implementation over broad language coverage. Preserve ordinary object identity,
aliasing, and cycles; document deliberate semantic restrictions.

## Current implementation

- `parse.rs` uses Oxc to produce an arena-allocated AST. `jit/compiler.rs` compiles
  it through a semantic graph to Cranelift native code; there is no interpreter or bytecode tier.
  Parsing storage is temporary, while the compiler retains generated code and
  names referenced by that code. Oxc semantic analysis reports early errors and
  resolves references to binding identities used by the compiler.
- `Engine` owns the compiler, shared global environment, and named function
  registry. Compilation can add functions across source files and execute a
  top-level initializer. Duplicate function names are rejected.
- `Value` is a copyable, NaN-boxed 64-bit word. Heap values contain the address of a per-runtime arena slot; `OwnedValue` keeps a value and its runtime alive on
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
  end of their block or frame; error exits release both.
- The supported subset includes numeric/boolean/null expressions, assignments,
  selected operators, conditionals, basic loops, returns, and dynamic calls.
  Common arithmetic and comparisons have inline numeric paths with helper
  fallbacks. This is not yet a complete implementation of JS binding or coercion
  semantics. Array-to-primitive conversion is not implemented: numeric coercion
  reports a runtime error when object conversion would be required.
- Numeric facts distinguish Int32, wider integers, and doubles. Integer increments,
  decrements, addition, and subtraction widen to Int64 when Int32 bounds are not
  established; loop merges preserve integer representation without matching a
  specific induction pattern. Indexed lengths and converted array positions are
  also Int64. Inferred integer arithmetic assumes results remain in the exact
  integer range of JavaScript numbers. Stepwise double rounding beyond that range
  and signed 64-bit overflow are unsupported; generated integer operations do not
  guard these limits.
- Bitwise and shift operators, including compound assignments, use guarded
  native operations for exact Int32 operands and runtime helpers for other
  conversions. Shift counts are masked to five bits; unsigned right shift
  returns a nonnegative number. These emission-time fast paths remain enabled
  with `JS_JIT_SSA=0`. Compound assignments share
  binary operations and evaluate the target and its old value before the right
  operand. Numeric operators use heap-aware primitive string conversion.
- `typeof` returns interned heap strings for supported values, identifying closures and
  built-in Error as functions. Undeclared identifier operands return undefined
  as a type name; uninitialized lexical bindings still report an error.
- Helpers report runtime errors through `JitCtx`; generated error checks return
  through the function's return block. Parse, compile, unsupported-feature, and
  runtime errors are distinct. Unsupported constructs generally include spans.

Most built-in methods and prototypes are still
missing; arrays and rest alone will not make the bootstrap runnable. The C++
runner independently embeds SpiderMonkey and provides a workload reference,
not Rust-engine bindings.

## CLIF cleanup

Numeric facts and representation selection belong to the semantic graph and
its emitter. There is no separate numeric dataflow analysis over CLIF. Unknown
boxed operands keep their runtime guards and helper fallbacks; proven operands
use typed emission directly. Mixed relational comparisons still perform their
coercions inside the fallback helper, preserving string comparison when both
operands are strings.

```text
Oxc AST -> semantic optimization -> typed CLIF + annotations
                                -> indexed-read scope and temporary cleanup
                                -> Cranelift optimization and code generation
```

`jit/clif.rs` pairs a CLIF `Function` with emission annotations for helper calls,
indexed reads, invocation roots, inline temporary pushes, and guarded temporary
clears. Annotations refer to that exact body and are consumed before Cranelift
rewrites it. Source locations remain available separately for profiling.

`jit/temporaries.rs` summarizes each block's effect on the temporary stack and
propagates a single may-register bit through the CFG. A guarded clear can be
bypassed only when every incoming path is clean. Both the clear path and its
empty-stack bypass leave the frame clean. Joins and loop backedges preserve any
possible push. Unknown calls, retain helpers and generic addition are
conservative barriers; inline pushes are recorded explicitly, so unrelated
reference-count and frame-slot stores do not dirty the temporary stack. The
cleanup bypasses the whole clear check, leaving unreachable blocks for Cranelift
to remove. It is disabled with `JS_JIT_SSA=0`.

Helpers also have effect summaries distinguishing heap and global reads and
writes, internal bookkeeping and ownership, allocation, errors, and reentry.
Semantic optimization uses the separate operation contracts described under
[Contracts and facts](#contracts-and-facts).

Indexed-read emission records the receiver, key, result, source location,
successful continuation and runtime-read contract as invocation-scoped read
descriptors. The original potentially failing read and its temporary ownership
remain in the CFG. An SSA-enabled scope can reuse a dominating successful read
inside a loop if the whole body has no observable writes, arbitrary reentry,
catch, generator transitions, or unknown calls. A nonthrowing probe establishes
the `DirectNumericIndexStorage` capability for a numeric key on an array or
typed array. It guarantees stable direct reads and that any owning element is
held by the receiver; it does not imply numeric array elements. Other receivers
and keys keep the original read. The later site compares receiver and key bits
in CLIF and falls back to its original helper on mismatch.

Storage and lifetime are separate facts. One bounded frame root holds the receiver, key, and result across
temporary clears, replacing the previous root or releasing it at frame exit.
This extends their lifetimes; the current runtime has no finalizers. The storage
probe must reject accessor or proxy behavior if introduced. Only one pair per
function is currently selected. A future embedder type could provide equivalent
field or method capabilities with explicit effects, result domains, ownership,
and stability contracts; mutations outside the engine would also have to honor
the scope. A typed-array element-domain fact could further specialize numeric
representation. Loop peeling could establish facts on the first iteration for a
later specialized loop, but is not implemented here.

Helper effect contracts let emission omit JS-exception checks for helpers that
cannot throw. General receiver specialization remains unimplemented. Tiered
compilation and feedback specialization are described in the semantic graph section below.

`EngineOptions::ssa_opt` enables rewriting and defaults to true unless
`JS_JIT_SSA=0`. `EngineOptions::ssa_dump` defaults to false unless
`JS_JIT_SSA_DUMP=1`; it prints per-function counts of removed temporary clears
to stderr. The existing IR listings show CLIF after cleanup when rewriting is
enabled.

Compilation tiers build on this boundary. Canonical semantic graphs of ordinary
functions are retained separately from emitted bodies; canonical CLIF bodies are
not retained. Runtime feedback identifies semantic operation sites independently
of CLIF entity IDs, and observed types require runtime guards before becoming
proven facts. Feedback recompilation, entry-point replacement and on-stack
replacement are implemented as described below. General deoptimization remains
future work.

## Values, objects, and arrays

Heap values store the address of a 128-byte arena slot in the NaN-box payload.
Arena chunks never move or shrink while the runtime lives. Each slot starts with
a fixed `repr(C)` header: reference count, kind, shape, and flags, as 32-bit
words. Kinds are bit families (typed array, array, holey, object, closure, ...);
typed arrays encode their element size, float, and signed bits in the low bits.
The header is followed by an element pointer and length at fixed offsets,
mirrored from array and typed-array storage whenever it changes, so generated
code reads elements without a helper. Arrays store holes in-band as the
uninitialized sentinel and count them; the holey kind bit is set exactly while
any hole remains. Objects, arrays, typed-array descriptors, and binding cells
are embedded in the slot; other larger payloads remain boxed. Vacant slots are
reused after incremental destruction completes. Internal lookups trust slot addresses:
a stale value from the same runtime reaches a zero-reference slot, while host
entry points check that an address belongs to the arena. Arrays
contain contiguous `Value` elements; binding cells and closures have separate
heap kinds. Environments and host handles share the runtime; raw `Value` tokens
do not keep it alive; their owner must remain alive while they are used, and
tokens must only be passed to their originating engine.

Public environment writes and engine calls reject heap tokens outside their
runtime's live arena. Live runtimes have disjoint arena allocations, so this
rejects tokens from another live engine. Tokens have no allocation generation
or runtime identity: a stale token can be accepted if its address is reused for
another compatible allocation, including by a different engine after the
original runtime is destroyed. These checks do not replace the requirement to
keep an owning handle alive and use tokens only with their originating runtime.
Raw bit construction is internal to the crate.

Dense array literals and rest parameters allocate arrays; rest copies trailing
arguments and retains referenced values. Numeric index reads return undefined
past the end. Writes replace an element or append at the current length; sparse
writes are rejected. Length reads and shrinking are supported. `map` can create
internal holes for positions removed before visitation; reads return undefined,
and subsequent `map` and `filter` calls skip them. Array literals with holes,
length growth, string keys, descriptors, and prototype mutation remain unsupported.

Array `filter`, `map`, `every`, and `some` snapshot the initial length and visit
present positions with `(value, index, array)`, observing intervening writes and
truncation. `every` and `some` short-circuit and skip holes. `find` instead
visits every index, passing undefined for holes, and returns the first matching
value or undefined. `forEach` visits present positions and returns undefined.
Their optional `thisArg` is unsupported except for undefined.
`sort` is stable and in place;
the default order compares string conversions, undefined values follow sorted
elements, and holes follow undefined. Comparators may return numbers or numeric
strings; object conversions are unsupported. `at` accepts numeric or undefined
relative indices. `join` supports primitive
values and separators; object-to-string conversion is unsupported. `pop` removes
the last element and returns an owned value, or undefined for an empty array or
hole. `indexOf` uses strict equality, skips holes, and accepts numeric or
undefined start positions; other start-position coercions are unsupported.
`concat` shallow-copies arrays and appends scalar arguments, preserving holes and
referenced value identity. Array-like objects, species constructors, and
Symbol.isConcatSpreadable are unsupported. `push` appends values, `reverse`
reorders the receiver in place, and `slice` creates a shallow copy; reverse and
slice preserve holes. `includes` uses SameValueZero and treats holes as
undefined. Slice bounds and the includes start position accept numbers or
undefined; other coercions are unsupported. `flat` supports numeric depths
through infinity and skips source holes. `flatMap` maps present positions and
flattens each returned array one level. `fill` writes a value across numeric
bounds, clears holes, and returns the receiver. `Array.from` consumes the same
iterables as for-of (arrays, typed arrays, and strings), or array-like objects
with a numeric `length`; it supports an optional mapper and receiver. The
`Array` constructor accepts a numeric length or a value list. Set iteration,
sparse array allocation beyond internal holes, and other iterable protocols
remain unsupported.

Ordinary object literals use reference-counted allocations with ordered property
slots and a hash index, sharing property storage and incremental disposal with
Error instances and closures. Literals evaluate properties in source order;
shorthand, nested values, primitive computed keys, and duplicate-key replacement
are supported. Dot and bracket reads return undefined for absent properties;
stores retain the new value before releasing the old one. Objects preserve
identity and aliasing, and cycles share the existing collection limitation.

Accessors, method shorthand, spread, and prototype-setting literal properties
are explicitly rejected. Computed and shorthand __proto__ keys are ordinary
properties. Object-to-property-key conversion remains unsupported. Ordinary objects follow internal prototype links on reads; own properties,
including undefined values, stop lookup. Property enumerability is tracked;
other property descriptor attributes remain unsupported.

Ordinary objects start with eight inline property values. Adding a ninth property
upgrades them to an embedded object with vector storage, preserving identity and
ownership. The `INLINE` kind bit distinguishes small objects while retaining the
`OBJECT` family bit. Object allocation first checks the pending destruction head:
if it is a small object, its remaining values and prototype are released and the
slot is reset in place, leaving unused value bits intact. Other heads use the
normal bounded drain and allocation path. Instances initialize both their prototype
and its root shape directly, including when reusing a pending slot. Closures also use vector storage. Both inline and vector
property values are indexed by their shape. Shapes are shared and immutable:
they record the prototype, the ordered interned property names with enumerability,
and are reached through
transitions keyed by name and enumerability, rooted per prototype. The shape id
lives in the slot header. Adding a property, changing its enumerability, or
setting the prototype changes the shape; replacing a value does not. Objects
with more than 64 properties or an array-index-like name switch to an ordered
dictionary and a reserved shape id that no cache may match. Indexed
elements stay separate from named properties.

The slot header's data and length words take their meaning from the payload:
array elements, typed-array storage, or the shaped property values of objects
and closures (null for dictionaries and other kinds). Only shaped objects and
closures carry a shape id other than `NO_SHAPE` or `DICTIONARY`, so a shape id
alone fixes the value index of each own property. A count reaching zero resets
the shape to `NO_SHAPE`, since disposal pops values without changing it.

Each compiled named read has a bounded polymorphic site cache. One entry is
inline in the cache record; three more are allocated together when a second
receiver shape or builtin kind appears. Each entry holds the receiver's shape,
a holder address (zero for an own property), the holder's shape and a value index,
or a builtin receiver kind and a borrowed native function. Builtin function
values remain owned by the engine's native-function cache.

A shape's root fixes the prototype's address, and receivers keep their prototype
alive, so matching both shapes fixes where the value lives. Generated code checks
the receiver before reading a cached holder and loads the current slot value on a
match. Heap results are retained unless borrowed or stored directly in a slot.
The separate semantic cache guard and load reselect the same stable entry by
receiver; they do not communicate through mutable last-hit state. Entries are
refreshed for prototype shape changes without consuming another receiver slot.
Arena chunks remain mapped for the heap's lifetime.

A refill caches shaped own or direct-prototype properties and returns the value
from that lookup, avoiding a second generic read. Deeper prototype chains, absent
properties and dictionary receivers use the generic helper. A fifth distinct
receiver or eight uncacheable misses permanently marks the site megamorphic.
Such a site skips probing and refilling the small cache. Existing entries remain
allocated so an earlier dominating cache guard still supports its load.

Megamorphic shape feedback is terminal and tracked per observing function, including
inlined copies. Recompilation removes cache-hit speculation at the affected site
and emits ordinary helper reads. Cache-miss paths have normal rather than cold
block placement: they can be routine at a polymorphic or megamorphic site.
Typed arrays provide contiguous numeric storage without per-element reference
counting.

Named writes pass a site cache to the heap helper. It caches the property's atom
and the last receiver shape's own-slot lookup, including an absent slot. A weak
heap reference keeps these IDs scoped to their heap without retaining its values.
The cache also remembers the shape transition for adding the property, keyed by
the source shape and enumerability, for both inline and vector storage.
Stores still use the normal retain/release, shape transition and invalidation
paths; dictionary writes and special receiver properties retain their usual
behavior.

Ordinary prototype methods are needed by existing map code.

## Strings

String `slice` and the optional `startsWith` position use scalar indices and
require numeric indices. `split` supports literal string separators or undefined,
without a limit. `replace` supports literal string search and replacement without
`$` substitution patterns. `padStart` pads by Unicode scalars with a literal
string. These methods reject unsupported coercions.

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
not allocation identity, borrowing both strings without copying their contents.
The semantic optimizer replaces strict comparisons of a single-use `typeof`
result with the literal `"string"` by a string tag test, in either operand order.
It preserves initialization checks at the original `typeof` evaluation point.
Begin with contiguous storage and helper operations;
scalar indexing may scan the UTF-8 sequence. Add indexing metadata or other
representations only when workloads justify their cost. String literals, concatenation, content equality, scalar length and indexing,
and truthiness are implemented. Slicing and full coercion support
remain future work.

## Loop lowering

`jit/emit/loops.rs` lowers loops directly into Cranelift blocks, preserving Oxc
binding identities and capture analysis. While, do-while, and for loops share a
stack of break and optional continue targets. Break selects the innermost
loop or switch; continue selects the innermost loop. Continue targets the test for while/do-while and the update for for loops.
Labeled statements and transfers remain unsupported.

For-of currently accepts arrays and strings with an identifier or flat array
binding pattern declared using var, let, or const. Array patterns reuse declaration
destructuring, including elisions and undefined for exhausted elements. Each
iteration has fresh lexical bindings. Defaults, rest, and nested patterns remain
unsupported. Evaluate the iterable once, then retain an internal
iteration state in a compiler-managed local across temporary cleanup. Arrays
read their current length and elements each step, observing growth, shrinking,
and replacements. The semantic builder checks `is_indexed` once: arrays and
typed arrays then walk a hidden index with `indexed_length` and `get_indexed`,
without allocating iteration state, while other iterables take the iterator
path. Both paths share one copy of the body, so the loop stays closed for
region optimization. Strings advance a UTF-8 byte cursor and yield one scalar as a
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
lexical loop bindings use per-iteration environments. Named function expressions
and async functions remain unsupported. Recursion through a
captured variable works but can create cycles.

Each engine has one global environment and one code registry. A closure belongs
to that engine; callers must not pass its value token to another engine. Code IDs
are engine-local, never executable pointers, and resolve only against the live
engine's registry. Compilations within that engine share global
bindings, so existing closures observe subsequent global writes. A caller's
local bindings do not replace the closure's defining globals.

Global reads cache binding locations in separately allocated cells, which remain
stable when the environment's name maps grow. Generated code checks the cached
environment identity and binding-resolution revision, then loads the current
cell value directly. Assignments update that cell without invalidating the
location. New ordinary or lexical declarations advance a revision shared by
the environment and its descendants, so inherited and shadowed bindings are
resolved again. This revision is separate from the heap epoch used by native
identity guards.

Cache misses resolve lexical bindings before ordinary bindings and refresh the
location. Missing bindings and uninitialized lexical cells use the existing
error path; optional reads suppress only missing-name errors. Heap values are
still retained as temporaries on cache hits. Each cache retains its environment
to keep its identity and parent chain alive; cached lexical locations are used
only after revision validation. The extra storage is one boxed cell per binding
and one cache per emitted read site, including retained compiled versions.

Execution retains its global environment and refers to the active engine. Code
lookup copies an entry address under a short registry borrow; no registry or
compiler borrow remains held while generated code executes.
Fixed-arity calls guard a predicted code ID and load its current published entry.
For calls with matching argument counts up to eight, a shared miss stub resolves
the actual callee's code ID through the same table. Its published argument adapter
pads missing arguments and discards extras before entering fixed-arity code,
without Rust dispatch. The adapter returns an owned result; only runtime results
need an additional retain in the miss stub. Native functions, closures requiring
lexical receiver resolution, and targets without a published adapter use the
runtime. Feedback withdrawal clears both published pointers so the next call
still triggers pending recompilation. The table uses two pointer words per code ID.
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

Function.prototype.call, apply, bind, and super remain unsupported. Ordinary
closures have writable named properties and lazily materialize a stable prototype
object. Construction links the instance to the current constructor prototype;
primitive prototype values fall back to a null internal prototype. Object.prototype
and the default prototype.constructor back-reference remain unimplemented.
Exotic constructor prototype values (arrays, functions, and native heap kinds)
remain unsupported. Prototype links and function properties own their values
and participate in incremental disposal; cycles retain the existing limitation. Array callbacks still reject a non-undefined thisArg.

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
Arrow and generator functions reject construction.

Error instances have identity and reference-counted named properties, initially
name and message. Both are writable; absent properties read as undefined.
The default name is Error, and an omitted or undefined message becomes an empty
string. Primitive messages convert to strings; arbitrary object conversion is
still unsupported. Errors support string concatenation and uncaught diagnostics
using the current name and message. Stack traces, Error.prototype, explicit
toString methods and subclassing remain unsupported.

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
  engine are checked against that runtime's arena; host callers must keep
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

## Instance checks and diagnostics

Ordinary instanceof checks walk internal object prototype links against the
right operand's current prototype property. Primitive left operands return false
for compiled functions. Invalid right operands and non-object prototypes report
engine runtime errors rather than catchable TypeErrors. Bound functions, proxies,
and Symbol.hasInstance are unsupported. Native Array has a global callable
identity for instanceof checks, including aliases; these checks use the array
heap kind. Array construction, mutable Array.prototype, and builtin inheritance
remain unsupported, as do instanceof checks against other native constructors.

Compilation errors and unsupported-feature errors include the source filename
and byte span. All parsed binary operator kinds now have lowering paths.
Generated error checks attach the innermost active filename and byte span to
runtime errors, preserving an origin already reported by nested functions or
source evaluation. This runs only on error paths. Errors raised outside checked
operations, including uncaught JS exceptions, may lack a location; stack traces
are not implemented. Unsupported property diagnostics identify the receiver kind.

## Default parameters

Identifier and flat array-pattern parameters support lazy whole-parameter defaults
for missing or undefined arguments. Array patterns accept identifiers and
elisions, and iterate arrays or strings using the same binding logic as array
declarations. Parameter bindings in non-simple lists begin uninitialized and are
initialized left to right; defaults can use earlier parameters, globals, and
captured bindings. Captures are installed before parameter initialization.

Nested patterns, element defaults, array rest patterns, and object patterns are
unsupported. Generator default and destructured parameters are rejected until
initialization can run at factory call time rather than first next. Parameter
and body var environments do not fully model JavaScript's separate environments
for non-simple parameter lists.

## Property presence

The in operator uses a runtime presence lookup, preserving properties whose
values are undefined and following ordinary object prototype links. Closures
expose named properties and their lazy constructor prototype. Arrays recognize
present canonical indices, excluding holes, and the supported length, filter,
map, every, some, find, forEach, at, sort, join, pop, indexOf, concat, push,
includes, reverse, slice, flat, flatMap, and fill properties.
globalThis checks global binding presence. Primitive
right operands, unsupported native receivers, and object-to-key conversion
report explicit engine errors. Proxies and additional builtin prototypes remain
unsupported.

## Basic classes

Named class declarations at script and block scope lower to ordinary closures
and prototype objects. Explicit constructors and synthesized empty constructors
require new; instance methods are shared non-constructable closures. Class-name
references capture a separate inner binding, retaining the original constructor
when the outer binding is reassigned. Assignments to the inner name are rejected
at compilation rather than producing a catchable TypeError. Outer bindings remain
uninitialized until declaration evaluation; local reads check initialization.

Inheritance, super, class expressions, static methods and blocks, fields,
accessors, private names, computed or literal method names, decorators, async
methods, and generator methods remain unsupported. Methods and the prototype constructor back-reference are nonenumerable and use
ordinary writable property storage; the class prototype property remains writable.
Other property descriptor attributes are not implemented.
Top-level class bindings currently share the global object binding storage.

The prototype constructor back-reference forms a strong cycle. Class graphs,
including their captures, therefore share the existing limitation of remaining
allocated until runtime teardown unless their cycles are explicitly broken.
They do not retain the engine or runtime itself. Cycle collection remains
required before persistent workloads can reclaim these graphs automatically.

## Destructuring declarations

Function-local var, let, and const declarations accept flat array patterns with
identifier bindings and elisions, and object patterns with static keys and
identifier bindings. The initializer is evaluated once. Array patterns use the
same array and scalar-string iteration as for-of; exhausted elements become
undefined. Object patterns read properties in source order with the ordinary
property lookup path. Captured bindings use the existing cells. Defaults, rest,
nested patterns, assignment patterns, computed object keys, custom iterables,
and top-level destructuring remain unsupported. Iterator closing is not
implemented.

## Top-level lexical declarations

Top-level identifier let and const declarations have a separate global lexical
store shared by subsequent source evaluations. Bindings are installed before
initializers execute, remain uninitialized until their declaration executes,
and reject reads or writes in the temporal dead zone. Let bindings are mutable;
const assignments fail. Same-source const writes may be rejected at compilation,
while cross-source writes fail at runtime. Neither binding appears as a
property of globalThis. Host Environment::get resolves lexical bindings, while
Environment::set and define operate on object bindings.

Cross-source redeclarations conflicting with lexical bindings are rejected.
Lexical declarations also reject existing object bindings, including builtins;
this is stricter than JavaScript's property-descriptor-dependent rules. Failed
initializers leave their bindings installed and possibly uninitialized. Global
class declarations still use the older global object storage. Top-level
destructuring and catchable TypeErrors for const writes remain unsupported.

## Spread arguments

Calls and construction accept spread arguments mixed with ordinary arguments.
The callee and receiver are evaluated once. Each spread lowers to an explicit
`Spread` operation that snapshots its iterable into an internal pack. Ordinary
functions keep the vector descriptor on the native stack, without a heap slot
or a JS value root. Each producer has a descriptor initialized at function entry.
Mixed arguments append ordinary values and expanded packs in source order into
an argument pack, without constructing a temporary JavaScript array. Arrays,
typed arrays and strings use the existing iteration behavior; other iterables
fail explicitly. Custom iterators and iterator closing remain unsupported.
Calls without spread retain the fixed-argument path. Pack calls pass the owned
vector's pointer and length directly to the invocation. Inlined constructors
extract referenced arguments with `PackTake` before initializers or the body;
missing arguments read as undefined. Extraction replaces the pack entry with
undefined and transfers its owned reference into the argument root, without
incrementing the refcount. Repeated parameter reads reuse the extracted SSA value.
`PackRelease` releases remaining references immediately after the inlined argument
reads and returns the cleared buffer to a context-local `Vec<Vec<Value>>` pool.
Split and spread producers fill checked-out buffers directly. Nested calls keep
their active buffers out of the pool; an empty pool falls back to `Vec::new()`.
Zero-capacity buffers are not pooled.
Generic invocations take the vector into a Rust owner that releases its values
and returns the cleared buffer to the pool on return, including error returns. Array and pack appends consume
the source pack. Catch entry and function exit release any unfinished stack
packs; cleanup is idempotent. Generators materialize packs as internal arrays
so they can survive suspension in the existing frame slots.
A native split whose result reaches only a spread through single-input joins
can produce `StringSplitParts` directly, eliminating the split array. Other
split results retain ordinary materialization. This optimization does not depend
on the global Array binding; explicit calls to Array still use the captured
callee and ordinary native-constructor dispatch.

Array literals also accept spread mixed with ordinary elements, using explicit
spread packs and appending their values in source order. Array holes read during spread
become present undefined elements. Literal elisions remain unsupported. Arrays
without spread retain their fixed-size construction path.


## Implicit arguments

Ordinary functions that reference implicit arguments create an owned snapshot of
the supplied arguments, independently of parameter bindings. Nested arrows
share that binding through capture cells, including after the outer call exits.
Constructors and generator resumes use the same mechanism. Explicit bindings
named arguments follow ordinary name resolution.

The snapshot uses the supported array representation, so it exposes array
identity and methods and inherits its dense index and length restrictions; it is
not a JavaScript arguments exotic object. Parameter aliasing, descriptors, callee,
and the complete arguments prototype and iterator semantics are unsupported.

## Math

The global Math object has stable native methods for abs, ceil, floor, round,
trunc, sqrt, sin, cos, tan, asin, acos, atan, atan2, exp, log, pow, min, max,
imul, sign, and random, plus PI and E. Methods ignore their receiver. Primitive
arguments, strings included, use the engine's primitive number conversion;
object coercion is explicitly unsupported. Min and max preserve
signed zero and propagate NaN; round uses ties toward positive infinity.

Math and its properties use ordinary mutable global and object storage; the
standard property descriptors are not implemented. Transcendental results use
Rust's platform math implementation. Other Math operations are unsupported.
The game's square and clamp helpers are defined by its globalscript rather than
being native Math methods.

Math.random uses per-engine 48-bit linear congruential state, initialized with
seed zero and preserved across calls and source evaluation. Seeding sets the
state to `(uint32(seed) << 16) | 0x330e`. Each draw multiplies by 0x5deece66d,
adds 11, and keeps the low 48 bits; the result is the upper 31 bits divided by
2147483648. This matches the game's rand48 sequence without depending on Boost.
The method ignores its arguments and receiver.

The hostSeedRandom binding resets the state using primitive ToUint32 conversion;
object conversion remains unsupported in the Rust engine. The shared map
bootstrap seeds from settings.Seed before loading global or map scripts, with
undefined selecting zero. The C++ runner implements the same generator and
seeding binding. Identical random sequences do not guarantee identical complete
maps when other engine behavior or transcendental results differ.

## JSON

JSON.parse accepts string input and constructs owned primitive values, arrays,
and ordinary objects. JSON.stringify supports these values, omits undefined and
callable object properties, substitutes null for undefined or callable array
entries and non-finite numbers, and rejects cycles. Repeated references that do
not form cycles are supported. JSON and its two native methods have stable
identity and use ordinary mutable object properties.

Object properties are emitted in the engine's ordered enumerable own-property
order. Number formatting follows serde_json and can differ
from ECMAScript exponent formatting. Parsing uses serde_json's nesting and
numeric-range limits; lone surrogate escapes cannot be represented. Stringify
limits nesting to 128. Revivers, replacers, spacing, callable toJSON methods,
non-string parse input, and exotic heap objects are explicitly unsupported.
Errors are engine errors rather than catchable JavaScript exceptions.


## Nested function declarations

Function-body declarations are initialized after parameters and before body
statements, sharing function bindings with var declarations. Declarations in
nested blocks are initialized on block entry and have lexical block scope.
Closures share captured cells, supporting self recursion, mutual recursion, and
escaping declarations; recursive closure cycles retain the existing collection
limitation. Nested generators use the existing generator machinery. Named
function expressions and declarations used directly as non-block statements
remain unsupported, and sloppy Annex B block-function semantics are not modeled.


## Ordered properties and for-in

Objects and closures use insertion-ordered property slots indexed by name.
Canonical array indices through 4294967294 enumerate first in numeric order;
other names follow insertion order. Replacing a value preserves order and
enumerability. Layout revision changes when a property is added or its
enumerability changes. Function prototypes, class methods and constructor,
Math and JSON builtins, and Error name and message are nonenumerable. Other
descriptor attributes and property deletion are unsupported.

For-in snapshots each receiver's own keys as it is reached, checks presence and
enumerability when visiting each key, and suppresses previously seen names
across the prototype chain. Nonenumerable own names suppress inherited names.
The owned enumeration state retains its current receiver and is released with
the frame or loop slot. Added keys need not appear after their receiver has
been snapshotted. Arrays expose present indices; truncation and holes are
observed. Null, undefined, numbers, and booleans have no keys. Strings and
exotic receivers are rejected. Identifier var, let, and const loop bindings
are supported, including fresh captured lexical bindings. Assignment targets,
destructuring, proxies, symbols, and prototype mutation are unsupported.
Generator loops without suspension are supported; nested yields are rejected by
the generator planner.


## Regular expression literals

Regular expression literals use regress 0.12.0. Each evaluation produces a fresh
heap object, while its compiled pattern is shared through a per-heap cache keyed
by source and flags. The cache owns compiled patterns, not JS heap values, and
persists for the heap lifetime. The supported flags are g, i, m, s, and u;
y, d, and v are rejected at compilation. Named captures are also rejected.

RegExp test and exec accept string input. Global regexes use a numeric lastIndex
measured in Unicode scalar values, update it on success, and reset it on failure;
zero-width success does not advance it. Non-global matching leaves it unchanged.
Exec and non-global String match return null or an array containing the whole
match and positional captures, using undefined for unmatched groups. Global
String match returns all full matches or null, resets lastIndex, and advances one
scalar after an empty match. Result metadata including index, input, and groups
is not implemented.

String replace supports regex search and a literal string replacement; global
replacement resets lastIndex and advances after empty matches. Dollar
substitutions and replacement callbacks are rejected. Matching and capture
ranges operate on UTF-8 Unicode scalar values even without u, rather than
JavaScript's UTF-16 code units; u still affects regex syntax and matching rules.
The RegExp constructor, arbitrary property mutation, and regex integration with
split remain unsupported. Regress is a backtracking engine and its execution is
not bounded by the runtime.


## Switch statements

Switch evaluates its discriminant once before entering the shared case scope.
Case expressions are evaluated in source order using strict equality until a
match is found; default is selected only after all tests fail. Case bodies use
ordinary Cranelift blocks with fallthrough. There is no new runtime operation
or jump-table specialization.

Let, const, class, and function declarations share one lexical scope across all
cases. Lexical bindings start uninitialized; functions are initialized before
case tests. Var declarations are hoisted to the enclosing function. Break exits
the innermost switch or loop, while continue targets the enclosing loop.
Labeled control transfers and suspension inside a switch remain unsupported.


## Object helpers and deferred freezing

Object is currently a mutable namespace with stable native keys, values,
getOwnPropertyNames, freeze, and isFrozen methods. It is not yet a callable
constructor, and Object.prototype is not implemented.

Keys and values enumerate own enumerable properties in numeric-index and then
insertion order. GetOwnPropertyNames includes nonenumerable own properties.
Ordinary objects, closures, arrays, and primitive strings are supported. Arrays
include present indices and a nonenumerable own length, excluding inherited
methods. String indices use Unicode scalar values. Numbers and booleans have
no own keys; null and undefined are rejected. Other exotic receivers and the
global object are unsupported. Closure name and length properties are not yet
modeled; the existing lazy prototype property is included. Result arrays own
their elements. Computed array and string reads accept canonical numeric string
keys; named reads such as array["length"] use ordinary property lookup.

Freezing enforcement is intentionally deferred for this experiment: freeze
returns its argument unchanged and objects remain mutable. IsFrozen returns
false for objects and true for primitives. The bootstrap deepfreeze traversal
can run, but does not provide immutability. This is an explicit deviation from
JavaScript and must be revisited before relying on frozen-object guarantees.


## Optional chaining

Optional named members, computed members, and calls lower to nullish checks
and branches. A null or undefined receiver skips the rest of its chain and
returns undefined. Computed keys and call arguments, including spreads, are
only evaluated on the live path. Bases and callees are evaluated once.
Calls preserve the receiver for both ordinary and optional method references,
including parenthesized chain references. Parentheses otherwise delimit the
short-circuit region: (a?.b).c still reads c after the inner chain finishes.
Unresolved variables, uninitialized bindings, noncallable values, and errors
on the live path retain their normal diagnostics. Existing primitive-property
boxing limitations still apply. Private fields, optional delete, and tagged
templates remain unsupported. No new runtime operation is required.

## Number formatting

Numeric primitives expose stable native `toFixed` and `toString` methods.
`toFixed` accepts numeric or undefined precision, truncates it to an integer
from 0 through 100, and rounds the exact binary64 value with ties toward the
larger magnitude. Values of magnitude at least 1e21 use shortest decimal output.
Decimal `toString` uses ECMAScript shortest formatting; nondecimal radices from
2 through 36 support integral finite values, zero, NaN, and infinities. Fractional
nondecimal values and non-numeric formatting arguments are rejected. Errors are
engine errors rather than catchable RangeErrors.

Unary plus now converts formatted strings through the primitive numeric
conversion path, including `toFixed` output. Implicit string conversion of
numbers (concatenation, `join`, property keys) follows the same ECMAScript
Number::toString rules, including `Infinity`, `-0` as `0` and exponent forms.


## Primitive numeric conversion

Numeric operators read heap strings before conversion instead of using the
heap-independent Value conversion. Complete decimal strings, exponents, signed
Infinity, unsigned hexadecimal/binary/octal prefixes, empty strings, and
ECMAScript whitespace are supported. Invalid strings return NaN and negative
zero is preserved. Arithmetic, unary numeric operators, bitwise operators, mixed
primitive comparisons and loose equality share this conversion. Two-string
relational comparisons use the runtime's Unicode scalar string order. Addition
with a string remains concatenation. Object-to-primitive conversion remains
unsupported, and builtin-specific argument coercion restrictions still apply.

## Date.now

The global Date namespace exposes a stable, receiver-independent `now` native
returning the current wall-clock Unix time in integer milliseconds. Times before
the epoch are floored to the preceding millisecond. The wall clock is not
monotonic. Date construction, instances, parsing, and other Date methods remain
unsupported; Date is currently an ordinary namespace object.

## Typed arrays

Uint8Array, Uint16Array, Uint32Array, Int8Array, Int16Array, Int32Array,
Float32Array, and Float64Array construct fixed-length, zero-initialized packed
storage from a numeric length or undefined. Lengths use truncation, with NaN
as zero; negative integral lengths and infinities fail. Allocations use fallible
reservation and are limited to 256 MiB per array. Static `from` accepts the same
iterable, array-like, mapper, and receiver inputs as `Array.from`, then converts
elements numerically. Other constructor forms and Uint8ClampedArray are
unsupported.

Indexed reads and writes use canonical numeric indices. Out-of-range and
invalid canonical reads return undefined and writes are ignored after value
conversion. Noncanonical named properties are unsupported, and length is
read-only. Integer stores truncate and wrap to their width; float stores use
binary32 or binary64. Primitive values use the engine's numeric conversion;
object-to-primitive conversion remains unsupported. `fill` mutates in place,
and typed arrays support indexed own keys, for-in, for-of, and spread. Their
constructor identities support the existing brand-based `instanceof` path;
`Array.isArray` returns false for them.

Shared ArrayBuffer storage, views, offsets, detachment, typed-array prototypes,
custom properties, and other constructor or method forms are unsupported.
This initial implementation stores each typed array independently.

Bitwise and shift operators have guarded int32 IR paths emitted through shared boxing and
unboxing routines in `jit/emit/ir_helpers.rs`. Unboxing uses saturating conversion
and compares the boxed round trip, rejecting nonnumeric values, fractions,
out-of-range numbers, and negative zero. Semantic Int32 operands use typed
emission without a boxed guard. Unknown operands keep the guard, fallback
helper and error check.

During emission, numeric index operands select `IndexGetNumeric` for the
fallback helper, passing the index as f64. The generic and numeric entrypoints share the numeric read implementation and the slow
receiver/property fallback. Specialization removes the boxed-index type test
without assuming an array receiver, integer index, or valid bounds. Ownership
and engine-error checks remain in place. Unknown index kinds keep IndexGet.
Numeric element indices are checked against the receiver's limit and round-trip
through `usize` to reject fractions; negative zero is accepted as index zero.

After semantic hoisting, known-kind typed reads with numeric keys expose their
checks in the graph: `ArrayPosition` computes a saturating integer position,
`IsExactPosition` checks its numeric round trip when needed, and `IsInBounds`
compares position and length as unsigned integers. Failure produces undefined;
success executes a bounded `TypedIndex` using the position. Region block sets
include the generated branches. Load reuse shares position and exactness work
for identical SSA keys within optimized regions, while bounds remain specific
to each receiver. View placement and simplification run afterward. The bounded
load's storage access, element conversion and result boxing still use the
existing emitter; other typed reads retain their emission-time checks.

The numeric getter keeps primitive and borrowed reads in its main body. Character
allocation and the general property fallback are cold, non-inlined functions so
those paths do not inflate the common getter's register saves and stack frame.

## Semantic graph

The compiler lowers supported functions, nested callables, generator resumes,
and script initializers through `jit/semantic/` before emitting Cranelift IR.
`--compile` prints the graph actually passed to emission by default, with
inferred types and semantic-location IDs. `--planner` prints specialization
plans, available feedback, the region tree and per-region assumptions, and pass
notes separately. Detailed region snapshots are captured before versioning and
after replanning each emitted region, only when requested. `jit/inspect.rs` formats
the function views; `--filter NAME` restricts their output by name substring.
`--clif` adds dumps from both sides of Cranelift optimization, and assembly includes
byte offsets and semantic-location IDs. The graph verifier checks scope and
control-flow invariants. Semantic
optimization is disabled by `JS_JIT_SSA=0`, for graphs with exception handlers and for
generator graphs; the shared CLIF optimization passes still apply.

```text
Oxc AST -> build -> promote -> simplify -> canonical graph
        -> plan specialization -> inline -> simplify -> plan regions -> version
        -> reuse loads -> hoist -> coercions and bit tests -> reuse loads -> hoist
        -> views -> simplify -> borrow
```

Canonicalization, planning and optimization operate on the same `FunctionState`,
containing the working graph, its facts, specialization plan, OSR plan and
diagnostic notes. The repository retains graph and notes at the chosen reusable
snapshot point; fetching a body creates a working state with empty analysis
data. This snapshot boundary does not introduce a separate graph or state type.
Passes explicitly refresh facts or control-flow analysis before using them. There
is no automatic invalidation or cross-pass reuse policy. A control-flow refresh
does not refresh other facts; each pass remains responsible for the
validity of the analysis snapshot it uses while transforming the graph.
`Facts` currently owns control-flow analysis as well as types, identities,
effects and lazy freshness data. Consumers share its control-flow analysis
within that snapshot, including final CLIF emission. Speculative analyses remain
local to their graph and assumptions.

### Representation

Values are SSA: each is defined once, by an instruction result or a block
parameter, and every use is dominated by its definition. Control-flow merges
(default parameters, `||`, `&&`, `??`, `?:`) pass values as branch arguments.
Blocks, instructions, values, bindings and scopes are Cranelift entity IDs, so
instruction IDs are stable within a graph and could later name operation sites.
Instructions and terminators carry source byte spans and preserve source
evaluation order. Calls record the receiver separately and resolve member
callees before evaluating arguments.

Bindings are interned per graph with their kind, owning scope, whether a closure
captures them (from `CaptureAnalysis`).
`Read`, `Write` and `Initialize` act on binding storage and check
initialization. Unresolved references and references to script-level
declarations use `GlobalRead`. Unresolved `undefined`, `Infinity` and `NaN` fold
to constants; the runtime defines `Infinity` and `NaN` as read-only globals, and
top-level lexical declarations cannot shadow existing global bindings.

A scope is created only when it declares bindings. `EnterScope(s)` creates its
bindings uninitialized; parameters, the function body, blocks and lexical `for`
heads each have their own scope. `NextIteration(s)` creates the next loop
environment, copying its bindings, before the first test and before each update.
Every normal or abrupt exit (`return`, `break`, `continue`) emits `LeaveScope`
for each open scope it leaves, innermost first; a return evaluates its value
first and leaves all scopes. The verifier checks that each block is entered with
a single scope stack, that scope operations nest, and that returns close every
scope.

Exception handlers have explicit exceptional successors that unwind lexical
scopes before entering a catch block; runtime errors bypass JavaScript catches.
Generator yields have resume successors and restore graph values and bindings
from the generator frame. The supported subset includes closures, `var`, `this`,
construction, classes, switch, do-while, and selected destructuring forms.
Labels, templates, `finally`, and more general destructuring remain unsupported.

### Contracts and facts

`Graph::contract(op, facts)` describes an operation: reentry, heap and global
effects, allocation, internal bookkeeping, whether it may throw, its runtime
assumptions and the ownership of its result (unowned, borrowed, owned, or maybe
owned). Runtime assumptions use the `Assumption` enum shared with the helper
effect report. Property writes are classified as closed under the same accessor
and coercion assumptions as property reads. `Define` only targets the object
literal being built, so it does not count as a write to existing heap state. It retains the property
read and callback assumptions of the shared property setter.

Facts are sets of possible value types: int32, other doubles, boolean,
undefined, null, string, object, other values (symbols and BigInts), the
uninitialized-binding sentinel and internal typed-array views. They are computed flow-insensitively over SSA
values to a fixed point; block parameters join their incoming arguments. Arguments
forwarded by a branch on their own type test retain the corresponding type
restriction on each edge, including undefined, nullish, string, number and
uninitialized tests. Strict equality and inequality against a value proven to
be undefined lower to `IsUndefined` and, for inequality, boolean negation.
This uses value facts rather than the spelling of an identifier; loose equality
retains its coercion semantics.

Type-test restrictions also follow the tested SSA value through intervening
blocks when the corresponding branch successor has a single predecessor and
dominates the forwarding edge. These restrictions are derived from the current
CFG during fact computation, not stored as persistent region guarantees. They
narrow incoming block arguments without changing the original value's facts;
joins still combine all incoming possibilities. Thus the result of
`x === undefined ? 0 : x` is numeric when `x` is number-or-undefined, even when
each arm has its own block. For-of elements reached through the initialized iterator
path therefore need no initialization checks. The
engine has no BigInt, so numeric operators always produce numbers. Facts
refine contracts: a coercing operation on numbers is pure, one on other
primitives needs no callback assumption, and an initialization check of a value
that cannot be uninitialized is pure.

### Initial call specialization

The compiler retains canonical graphs of ordinary top-level and nested functions, keyed by
their code IDs, before specialization or region copying. Emission optimizes a
clone. The repository also records the originating compilation. Cross-source
inlining preserves the callee's error locations and resolved closure code IDs.

`semantic/specialize.rs` separates specialization requests, guard planning, and
application. The static producer recognizes global bindings and known local
closures through their SSA definitions. Inlining substitutes ordinary argument
values without numeric guards or type-based eligibility restrictions. The
planner bounds initial expansion to 128 callee instructions per caller, with at
most 32 instructions per eligible callee. Recompilation multiplies both limits as described below. These counts use canonical semantic operations, including
arguments, constants and scopes, but not block parameters or terminators.

`semantic/inlining.rs` copies callee blocks, values, bindings and scopes, and
joins returns through a continuation parameter. Mutable global targets keep a
code-identity guard and the original call as fallback. Direct local closure
creation proves the code identity without a repeated guard, except when class
marking requires the ordinary call checks. Arguments and ordinary receivers are
substituted at the call site; an arrow uses its closure's lexical receiver.
Known local closures access the original caller bindings directly when their
captured environments have not changed since closure creation. Other captured
scopes use an explicit closure operand to load the original cells, preserving
shared writes, initialization checks and per-iteration environments.
Nested calls, heap operations, closure creation and ordinary control flow can be
copied. Closure operations carry their resolved code ID, capture bindings and
lexical receiver; copied closures remap the captures and take the inlined
activation's receiver. Rest and arguments snapshots, generators, and exception
handlers remain unsupported by the inliner.

Calls to a property named `split` get a guard on the evaluated native identity.
The guarded path calls the existing string split implementation with the original
receiver and arguments; replacements keep the generic call. The operation reads
the heap, allocates its owned array result, and can fail. Only that fast result
is known to be an object; the merge with an arbitrary fallback remains unknown.

`semantic/arrays.rs` exposes `map`, `some`, `every`, and `filter` calls with known
closures as semantic loops behind a guard on the evaluated native identity and
array receiver. Unsupported
receivers, replacement methods, and unsupported thisArg values retain the generic
call. The loop snapshots the initial length, skips absent indices at visitation,
and passes `(value, index, array)`. `map` writes into a result initially
containing holes; `filter` appends matching values to a dense result. `some` and
`every` short-circuit on callback truthiness. Reads observe callback mutations.
Exposing the callback lets ordinary inlining remove its dispatch and frame. Later rounds can inline calls made by its body, including bounded
recursive calls such as `createTerrain(t)` in the rmgen library. Exception and
engine-error exits retain the caller's normal frame cleanup.

After specialization, closure allocations with no reachable uses are removed. An allocation used only in one cold fallback block can move to
that block when its loop nesting is unchanged; this avoids duplicating observable
closure identities across iterations. Inlined paths then avoid materializing an
otherwise unused callback. Captured closures can also sink when their original
binding environments remain available. Local bindings needed only by fallback
closures start as ordinary values and acquire a shared cell on first closure
creation. Subsequent accesses use that cell, and iteration changes reset the
current cell while escaped closures retain the previous environment. Bindings
with no remaining closure uses need no cell. Ownership analysis runs after these
transformations.

For example, `square(x)` with `square(x) { return x * x; }` gets a code check,
an inline multiply with generic semantics, and the original call as fallback.
Subsequent region planning discovers numeric assumptions from the inlined
operations and combines guards whose dependencies are available at a shared
entry. `square(x * 2)` needs no numeric guard because its argument is already
known numeric. Inlining decisions retain target identity and expansion costs;
numeric specialization belongs to region planning.

`planner.rs` collects feedback independently of code emission. Fixed argument
calls and named or indexed reads in ordinary functions have a stable site
identified by source and byte span, including calls copied from an inlinee.
A boxed cache records its owning code ID. For calls, its first execution records
a JS code ID and queues the owner;
native and invalid callees close the cache without requesting inlining. This
initial policy keeps one candidate per source site and does not adapt to later
target changes. Code IDs are metadata and do not retain closure objects.

Semantic planning inserts explicit `Observe` operations before region versioning.
Each operation carries its value, observation kind, and original source site;
versioning and cleanup preserve that intent when copying or replacing reads.
The emitter only lowers these operations to one-shot latches. OSR consumes the
same operations, including dominating observations of loop inputs, rather than
rediscovering observation opportunities from reads and calls. Replanning removes
old observers and emits only those still missing feedback. Observers update
compiler metadata without retaining values or changing JavaScript heap state.

Numeric entry guards also emit one-shot number observations after success.
They distinguish safe integral numbers from other numbers, excluding negative
zero. Replanning represents observed types with explicit `ReadHint` metadata;
hints choose speculation but do not narrow facts. Region checks carry their
required types into candidate analysis, guard generation, and refinements.
An integer choice emits `IsInteger` and `AsInteger`, preserving native integer
arithmetic while retaining the generic continuation when the guard fails.

Inlining, read refinements, numeric observations, and conditional calls form one
comparable `SpecializationPlan`. Recompilation compares that complete plan, so a
numeric-only change can replace a body or produce an OSR fragment. Extracted
loops carry both pending observations and known read hints for transferred inputs
at their original sites. Array and numeric feedback therefore remain available
when the producing read or guard is outside the extracted loop.

Read observations run after successful lookup, including named-property IC hits.
The first result closes the observation cache; array results record their storage
category (ordinary array or typed array with element kind) and queue the owner.
Indexed reads also observe receiver categories, including generic typed reads
whose element kind is still unknown. This retains feedback when load reuse
eliminates the producing read or an existing loop version already establishes
that the receiver is typed. Planning prefers a read's own receiver observation,
then uses its producer's result observation, then the element kind statically
known from the receiver's construction (`new Uint8Array(...)`,
`Float32Array.from(...)`). Receivers with such a static kind emit no receiver
observation, so read planning must consume the static kind itself; otherwise
the read stays generic until other code observes it. In `height[x][y]`, the first read
can select guarded `ArrayLoad` operations and the second can select `TypedIndex` independently.

Where needed, a number guard selects numeric keys before read specialization.
For ordinary arrays, `ArrayPosition` converts the numeric key to an integer.
`IsArrayIndex` checks that this position exactly represents the original key,
is in bounds, and matches the packed or holey layout. After versioning, lowering
exposes `IsExactPosition`, `IndexedLength`, `IsInBounds` and `IsArray`, joined by
`BooleanAnd` predicates. Load reuse shares exactness and bounds between the packed
and holey alternatives; length and layout reuse are invalidated by conflicting
mutations. The resulting predicate selects a matching `ArrayLoad`, which uses
the shared integer position directly without repeating layout or bounds checks. Packed loads read the element directly; holey loads also map
holes to undefined. These loads require the matching successful check with no
intervening array mutation, and retain the existing heap-result ownership rules.
Other receivers and keys use the original read. Numeric speculation combines
IR facts with guarded integer observations. Typed element kinds
select a matching kind guard and a direct element load, retaining index validity
and bounds checks unless the region proves them redundant. Feedback does not prove a property
immutable or retain the observed array. Read decisions join the shared specialization plan. This initial policy does not propagate producer
observations through SSA merges or specialize writes.

Observations request compilation independently of scheduling. At function return,
an eligible request submits an immutable snapshot of the canonical repository
and accumulated feedback to a background worker pool. At most one job per logical
code ID is outstanding; further requests remain pending and are coalesced into
a follow-up job after completion. Canonical bodies and source text are shared
between snapshots, while metadata uses copy-on-write updates. Each worker owns
its Cranelift module and never accesses the runtime heap or published caches.
Observations supply speculative candidates,
with the evaluated callee checked by the existing code-identity guard. The planner
runs an initial inlining pass and up to two further rounds with one shared
128-instruction base budget, allowing calls exposed by earlier rounds to inline
without expanding fallback calls again.
Both the shared instruction budget and per-callee size limit are multiplied by
`EngineOptions::recompile_inline_multiplier` (default two) for feedback
recompilation and OSR. Initial and prefetched compilation use the base budget.
The multiplier leaves the number of rounds unchanged.
The ordered inlining and read decisions form the decision record for the emitted
version.
Planning includes array, string split and construction expansion and simplification between
inlining rounds, so later decisions can use sites exposed by earlier inlines.
Expansion tracks processed instructions within the planning run; preserved
generic fallback sites are not expanded again. If the decision record is
unchanged, recompilation stops before region versioning, ownership analysis and
machine-code emission. Otherwise those passes continue on the planned graph.

Call-effect analysis also follows canonical bodies independently of inlining.
It collects reads, writes, allocation, errors and engine effect assumptions, then
propagates effects and identity obligations through reachable calls to a fixed
point. Recursive edges do not themselves introduce unknown effects. Unresolved
calls retain arbitrary effects; global code and recognized native identities
remain explicit conditions rather than facts inferred from observations.

Exact local closure calls can use unconditional closed, read-only summaries.
Conditional call proofs participate in the same bottom-up region planning as
numeric, native and typed assumptions. Each call retains its transitive global
code requirements. A candidate loop or function checks only its own requirements,
merges covered child proofs and emits one combined guarded copy. The region must
remain closed and preserve the guarded identities; unrelated writes are allowed
when the ordinary region stability checks permit them. Callee summaries still
require read-only descendants and reject unresolved calls, native identity
obligations, generators and handlers. Only calls in the validated copy receive
the conditional proof; guard failure enters the original region before its
effects execute. Value substitution and nested copying preserve those proofs.
Feedback recompilation retains the native entry of the previously emitted
version. A function specialization sends entry guard failure directly to that
previous version, passing the original closure, receiver and arguments. When
all fallback guards move to entry, it reuses the planned fast body in place.
Generic code that a specialized region bails out to is split into native
continuation pieces built lazily from the canonical graph, at most once per
compilation:

- `loop.H` runs loop `H` from its header with the loop-carried values. Nested
  loops are called as their own `loop` pieces rather than copied. Every exit
  stores its live outputs into the caller's argument buffer and returns an exit
  index. Exits are keyed by target block, plus one shared exit for `return`, so
  a `break` to an outer label simply becomes an exit of each enclosing loop.
- `next.H` calls `loop.H`, dispatches on the exit index and runs the generic
  code after the loop up to the next loop header, which hands over to that
  header's `next` piece.
- `resume.T` starts at a failed guard's target with the live SSA values,
  finishes the interrupted iteration and hands over to `next` at the first loop
  header it reaches.

A guard stub stores its live values and makes one call. Loop headers of the
in-function generic copy are replaced by a call to their `next` piece. Pieces
only call outward along loop exits, so the native stack depth of a bailout is
bounded by the number of loops on its path, independent of iteration counts.
Stubs share one cold block when they resume at the same target with the same
values.

Continuations use the canonical graph's facts to type their inputs and outputs:
a piece's own control flow omits the paths around it and would infer unsound
loop-carried types. Nested versions compose block and value mappings back to
that graph, so a bailout does not inherit an enclosing specialization's
assumptions. Each continuation uses the existing vector calling convention, with
boxed live values in a caller-owned native stack buffer; a loop call that
returns outputs gets its own buffer. The original receiver is forwarded through
each continuation, so ordinary `this` reads need no extra materialization.
Borrowed inputs acquire temporary ownership at the transfer boundary. Exit
outputs are retained by the callee and owned by the caller's temporaries.
Continuations own their results through the ordinary return path; only the
original invocation reports function completion.

Extraction currently requires promoted SSA state. Materialized scopes, other
implicit frame reads, generators, handlers, heap views, and OSR polls keep their
existing in-function fallback paths, and so does any
guard whose path reaches a loop that cannot be split; planner notes explain
rejected extractions. Graphs that already store exit outputs, such as OSR
fragments, do not extract continuations: a piece would store into its own buffer
rather than the fragment's caller buffer. Continuations compile eagerly and have separate semantic
IR and assembly listings under the enclosing function's name.
`JS_JIT_NO_CONTINUATIONS=1` disables extraction and keeps every bailout in the
function's own generic copy.

Only function-entry failures restart the previous native version. Fixed entries
use a native tail call; generic entries call and return after releasing the
speculative frame. That fallback bypasses mutable dispatch tables.

Load reuse, hoisting and borrowing consume instruction-specific proofs, while
generic calls retain their reentry and ownership barriers. Conditional decisions
participate in the tiering decision record; inline budgets are unchanged.

String split calls have a conditional native effect summary with an explicit
callee identity obligation. Within a proven string branch, the successful native
split can enter a specialized continuation. Its result is then separate from
the arbitrary replacement method's result. Constructor identity, prototype,
storage and undefined-texture failures leave this copy for the original
continuation at the corresponding point; the split is not replayed. This pass
selects at most one continuation of at most 512 instructions, and only when no
other optimization region was selected. Default argument initialization stays in
the specialized copy so its joins retain the split element types.

Successful split results have indexed storage. Spread reads yield strings or
undefined; the first element is a string when the separator is statically absent,
undefined or a nonempty string. Element facts survive only when no mutation or
escape through an SSA alias can reach the read, including through loop backedges.
Inlined constructor stores can remain private when their fresh instance reaches
only a terminal constructor result and return. Freshness follows block parameters,
including loop backedges, only when every incoming value is private. Escaping any
alias invalidates the whole group; writes through aliases still clobber reads
through the original value. Dynamic separators retain the
undefined-argument error exit.

This does not yet prove the full `createTerrain` workload closed: array method
replacements and recursive array inputs remain generic barriers. Diagnostics
trace rejected calls into their canonical dependencies.
Supporting writes that preserve particular dependencies, recursive native guards,
and descendant escapes with continuation recovery remains future work.

Re-emission declares a new machine function. Generated call dispatch checks a
shared atomic completion flag before loading its target and enters the runtime
only when results are ready. The execution thread takes ownership of the new
version's previously unused caches before replacing the entry for the original
logical code ID. Return-time scheduling also polls completions. Closures retain
their identity, captures, and prototype. Active frames finish in the old code,
which the JIT modules retain. Engine destruction joins the worker before
releasing runtime resources. Failed recompilation reports a diagnostic and keeps
the previous entry. `EngineOptions::background_compile = false` or
`JS_JIT_BACKGROUND=0` retains synchronous recompilation at the next dispatch for
deterministic tests and comparisons. Property writes are not profiled yet.

The compilation service separates target identity from attempt identity. A
function code ID or an emission's loop site identifies the target; a unique
ticket identifies each queued attempt. The shared queue orders ready jobs by
priority and submission order, coalesces duplicate targets, and supports
promotion without changing an attempt's snapshot. Dependencies refer to attempt
tickets and inherit the requesting job's priority transitively. Taking a job
does not release dependents: the execution thread completes its ticket after
handling publication or failure. Further feedback for a running target remains
pending for a later attempt. The service starts four regular workers by default,
configurable through `EngineOptions::compiler_threads` or `JS_JIT_WORKERS`, plus
one additional worker reserved for demanded lazy compilation. Each owns its
compiler and code module. Regular and demanded jobs have separate ready heaps;
promoting a queued lazy job preserves its ticket and snapshot while moving it
to the reserved lane. Only the reserved worker takes demanded jobs, and it
sleeps when no such job is ready. Publication remains confined to the execution
thread. A worker exit stops the pool, joins the remaining workers, and cancels
outstanding work
after discarding remaining unpublished results; previously installed code remains
usable.

Scheduling has four priorities: demanded lazy compilation, OSR/feedback,
prefetch, and unlazify. Compilation tier is carried separately on each job.
Demanded lazy, prefetch, and unlazify jobs use the baseline tier with Cranelift
`opt_level=none`; OSR and feedback jobs use the optimized tier with the configured
`opt_level`. Both use the backtracking allocator; the single-pass allocator
measured slower overall at the baseline tier.

With background compilation, initial bodies are generic by default: they keep
semantic optimization and observations but skip region versioning, so they carry
no speculative guards or continuations. Specialization happens only through
feedback recompilation (the first observation requests it, and it is submitted
once the function has returned eight times) or through OSR fragments for loops
still running. Nothing queues specialization merely because an initial body was
installed. `EngineOptions::generic_first` (`JS_JIT_GENERIC_FIRST=0` disables it)
restores speculative initial bodies. Recompilation of a generic body always
proceeds, even when its specialization plan is unchanged. Workers retain a compiler/module per tier so emitted
code outlives subsequent jobs. Promoting a queued lazy job to demanded priority
retains its baseline tier and snapshot. Running jobs
retain their original tier. Installed emissions record their tier, allowing
optimized recompilation to replace baseline code even if the specialization
plan has not changed. Main-thread compilation uses the optimized tier.

Dispatch of a lazy function queues known callees that still have lazy bodies
at prefetch priority, below OSR/feedback optimization. Loading source
also queues all remaining unattempted lazy bodies below prefetch priority. Callee discovery
combines canonical call sites with calls remaining in the planned graph after
inlining, including calls exposed inside inlined bodies. Original targets remain
useful for later calls from other sites even when this caller inlines them.
Discovery resolves global functions and local closures, including constructor
and spread calls. Prefetch uses each callee's
source-load repository snapshot and emits an initial body with its generic
continuations, rather than a feedback version requiring an existing entry.
Workers declare their own physical code IDs while preserving logical function
identity and lexical receiver behavior. Background installation removes the lazy
body but does not recursively discover more callees.

Dispatch promotes a queued lazy attempt to the reserved lane and waits for its
entry. If a regular worker has already taken the attempt, dispatch waits for
that result instead of compiling it again. While waiting, the execution thread
publishes completed results and releases their dependency tickets. Independent
jobs continue compiling without publication. Initial compilation failures are
retained and returned when the function is called. With background compilation
disabled, initial bodies still compile synchronously. Lazy wait logging records
per-call durations and aggregate count, total, and maximum duration. Shutdown
stops the pool without joining: running compilations finish detached on their
owned snapshots and their results are discarded. Workers embed engine addresses
in generated code but never dereference them.

`EngineOptions::osr` enables loop replacement; it is on by default and
`JS_JIT_OSR=0` disables it.
Natural-loop discovery runs independently of the region-plan tree after guard
placement and optimization. Merging region plans therefore does not suppress an
inner loop that still has explicit call or indexed-read observations.
The descriptor retains that emission's graph and records each loop's external
SSA inputs, header parameters, and exit shape. An observation belongs to a loop
when it executes inside the body, or dominates its header and observes one of
its external inputs. Read-result opportunities are restricted to values feeding
an unresolved indexed receiver. Loops without remaining observations do not
receive counters.

Every eligible loop is instrumented. Eligibility requires promoted state, inputs
dominating the header, outputs dominating their exits, and at least one exit.
Generators, handlers, materialized scopes, closure creation, and implicit
function-frame reads other than `this` are excluded; the planner report names
the rejecting operation. Exits follow the loop-continuation convention: each
edge leaving the body and each return gets an exit index, and the fragment
stores that exit's live outputs and target arguments with `ContinuationStore`.

All edges into the header, from the preheader and every backedge, pass through
a poll block whose parameters replace the header's. Its `OsrStart` operation
exposes external inputs and the current header parameters to ownership
analysis, and returns -1 when no transfer happened or the fragment's exit index.
Dispatch then reads outputs with `ContinuationOutput`, unboxes them by their
facts, and jumps to the original exit or returns; SSA repair merges these
definitions with the loop's own. A native stack buffer passes boxed words in
both directions, borrowed inputs acquire temporary ownership, and no JS
container is allocated. The caller's receiver is forwarded, so `this` reads in
the loop need no materialization. The parent frame remains suspended and roots
its values while a separately framed fragment executes. Errors unwind the
fragment frame before propagating through the parent's existing failure path.

The poll counts header arrivals, so it measures iterations rather than loop
entries. Every 1024 arrivals a site with applicable feedback can request one
compilation attempt; installed sites transfer on the next arrival. Because
transfer happens at the header with the current loop-carried values, a running
invocation can leave generic code mid-loop. Nested loops cascade: the inner
loop reaches the threshold first, and by the time the outer loop transfers, its
fragment is compiled with the inner loop already observed. Fragments themselves
are not instrumented. The extracted graph inherits facts established at the
header. Constants are rematerialized, established numeric inputs become numeric
SSA values, and typed operations retain the enclosing guards' guarantees.
Address views are discarded and rebuilt during fragment optimization, so
re-versioning cannot retain stale view definitions. New feedback introduces
guards and local generic paths inside the fragment. A failed guard never
restarts the parent function or replays completed iterations.

Loop jobs use active priority. Ordinary function recompilation uses normal
priority; after an OSR request, encompassing-function jobs use next-call priority
for a future invocation, above speculative callee prefetch. Fragments record observations against their logical
owner but do not report a parent function return. Publishing a fragment changes
only its emission's loop entry cell, leaving the active function entry intact.

### Passes

Promotion turns every non-captured binding into SSA values, adding block
parameters at merges and removing trivial ones. Each promoted read or write
becomes a `CheckInitialized` of the reaching value; the entry value is an
explicit uninitialized sentinel. Simplification removes pure operations whose
results are unused, including checks of initialized values and the scope
operations of scopes without captured bindings, and replaces numeric coercions
of numbers by their operand.

Relational comparisons with unknown numeric operands expose their `is_number`
checks in the semantic graph. `GuardedCompare` consumes those checks and keeps
the original operands for generic comparison fallback. Dominating checks are
shared with later comparisons and refined indexed reads without adding semantic
blocks. Known numeric comparisons still lower directly. Simplification folds
number checks once their operands' types are proven, removes redundant
`as_number` assertions, and prunes constant branches. It recomputes facts after
pruning so newly narrowed joins can simplify further. Choosing a shared fast
region around several guarded operations remains a planning decision; a
successful coercing comparison does not change the original index key's type.

Optimization works on regions: the whole function, or a loop whose body is
closed. A region is closed when nothing in it reenters JavaScript and nothing
writes state the region is guarded on. Typed versions of a region are selected
by checks that run before it is entered. Varying inline guards can transfer to
the original continuation before reentry. Each versioned region keeps exactly
one generic sibling, which is not versioned
further, so the result is a tree of loop versions that join again at loop exits.
Values defined inside the version and used after it reach those exits through
block parameters, placed by dominance-based SSA reconstruction.

Versioning stores these block sets in graph-owned `Region` records, indexed by
`RegionId`. Each record can identify its generic fallback as another region or an
existing compiled entry. The link does not enumerate individual side exits.
Load reuse and hoisting receive region IDs and use the same block sets as before;
their existing analyses establish safety.
These records describe versioning results, not maintained structural ownership
or proofs. Later CFG transformations do not automatically extend membership or
repair fallback relationships. Planner output reports the records at versioning
time.

A call whose callee is a property read of a global, such as `Math.min`, reenters
unless the callee is the expected native function. Versioning collects these
requirements for the recognized `Math` natives while planning each region. The copy calls `CallNative` and drops
its now unused callee reads. Its entry compares a per-guard cell against the
realm epoch. On a mismatch the global and method probes run without invoking
JavaScript, and on success the cell records the epoch; any identity failure
selects the original. The heap bumps the epoch when a property holding a native
function is replaced, when a global binding holding an object is replaced, and
on every global definition, so a script that redefines builtins once at startup
revalidates each guard once. Callees captured before the region are checked
by value on each entry. When only the receiver was captured, its property is
probed on each entry instead of looking up the current global. A guarded copy
must not write a guarded global or a property with a guarded name. Property
writes through receivers that may be `globalThis` also affect globals with the
same name; unknown keys affect all globals. Numeric keys cannot overwrite
recognized Math names. Unescaped local allocations do not alias globals.
`Math.random` is also a recognized guarded native. Its RNG mutation has the
internal-state effect, so it stays in execution order and cannot be removed or
hoisted. It does not clobber globals or heap properties, allowing their reads
to move outside closed loops.

The same entry-check planner handles loops and the function body. Arguments and `this` seed its
probe dependencies; preserved global and property reads can then feed code and
number guards at entry. Repeated probes and identical code or number checks are
shared. The copy takes hoisted guard branches unconditionally. Guards that
depend on values produced inside the region remain in place; failure transfers
to the original slow continuation before reentry, with SSA repair carrying the
current values. The closed copy retains no reachable reentrant fallback. If the closure proof fails, the candidate is
discarded. Failed entry checks select the original function before any of its
effects run. Only reachable copied blocks participate in subsequent write
summaries and load optimization.
Property probes use the same shape and holder cache as ordinary named reads,
with a non-throwing, non-retaining miss path. Borrow analysis includes property
and global probes; reentry and heap or global writes end those borrows. A path
that never writes a frame slot skips the local-slot release scan on return,
even when its generic sibling reserves slots. Temporary and indexed roots still
receive their normal cleanup.

Property caches also retain a builtin-method entry guarded by the receiver's
exact heap kind. Arrays, typed arrays, strings, sets, regexps and generators
resolve these methods by kind; native constructors remain on the ordinary miss
path because their methods also depend on constructor identity. The native
function cache keeps cached method values alive. Hits load the function value
directly and use the existing ownership rules for the result.
Lazy binding reads check initialization inline and retain heap temporaries with
the inline retain-and-push path; immediate values need no runtime helper.

Region planning visits natural loops from innermost to outermost, then the
function body. Each plan records native identities, numeric assumptions,
preserved read probes, existing inline guards, typed receivers and bounds checks
without copying control flow. Arguments and `this` can seed function-entry
probes. Numeric assumptions feed a separate fact analysis for the candidate;
they do not narrow the generic graph. Typed-read selection uses these facts,
so numeric and typed conditions share a single entry decision.

A parent absorbs a child whenever its entry assumptions establish the child's
number facts, native identities, inline guards and typed-read guarantees. There
is no profitability threshold yet. A receiver defined by an outer iteration, a
read clobbered by parent writes, or a bound unavailable at parent entry can keep
a child region separate. Preserved reads can be probed speculatively even when
their original path is not taken: probes do not throw or invoke JavaScript, and
failure selects the original region before its effects run.

Only the resulting region tree is emitted. Each speculative node gets one copy
behind all its guards; children already covered by an ancestor emit no copy.
Separate child plans are remapped into the parent's fast body. Their entry
assumptions are refreshed against the repaired SSA graph before emission, so
exit phis and probes or guards already handled by an ancestor are respected. Native callee reads made unused by the
planned native calls are excluded from probe discovery. Load reuse and hoisting
run after emission. Capture-cell accesses have their own alias category, so
writing a binding does not invalidate global or property guards. Unused
closures and unnecessary capture cells are removed before region planning, and
bindings that no longer need cells are promoted to SSA again. Unknown reentrant
calls still prevent closed-region optimization.

A typed-array receiver created by a typed-array constructor or its `from` or
`of`, looked up by global name, is guessed to have that element kind; the entry
check then compares the exact kind, so a wrong guess or a replaced constructor
only selects the generic sibling. A read whose key is a loop counter starting at a
non-negative integer constant, stepping by one, and tested with `<` against a
number defined before the copy, on a header whose false edge leaves the loop, is
also checked once on entry: the bound must not exceed the receiver's length.
Typed arrays never change length, so such a read is an in-bounds integral index.
With a known kind it compiles to a single load with an unboxed result, an Int32
for element kinds that fit.

Write effects are classified by access: a named property, `length`, elements
under a numeric key, iteration state, or unknown. Element writes conflict with
element and `length` reads. Numeric-looking names, including `NaN` and
`Infinity`, conservatively overlap numeric accesses. Other named writes
conflict with reads of the same name, and iterator and enumerator steps with no read, since their state is not visible to
JavaScript. Escape is judged within the region: an allocation escapes when it
is used other than as the receiver of a property access, in an identity test or
as an iteration source, or when it is passed to a block inside the region.
Iterators and enumerators created in the region are allocations too, and a
source escapes if its iterator does; this relies on iteration using direct
engine state rather than JS callbacks. Array-pattern declarations initialized
by an array literal without spreads or holes bind the evaluated elements
directly.

Within a region, a property, index or global read reuses an equal dominating
read if no write in the region may alias it; `this`, argument reads and pure
binary operations are reused too. Loop hoisting moves reads and pure operations
with loop-invariant operands from a loop header to its single preheader,
stopping at the first operation it cannot move. Other invariant reads use
non-erroring probes before a versioned loop. Every probe must succeed to enter
the specialized copy, which uses the probed values without per-read checks;
failure selects the original loop and preserves when errors occur. The internal
uninitialized sentinel distinguishes failure from a successful `undefined`.
Probes support directly stored values, numeric indexed values and simple builtin
properties such as `length`; accesses requiring materialization fall back.

A view is the receiver's slot address, or a static all-zero slot for
non-objects and the global object, so kind checks fail and the length is zero.
A loop indexing an object defined before it computes the view in its preheader;
other element reads compute it in place. Each access checks for an integral key
within the slot length and switches on the kind: typed arrays load or store
inline, packed and holey arrays load inline (holes read as undefined, heap
elements are retained as temporaries like the helper result), and everything
else falls back to the generic helper at the same point. Array `length`
reads, `is_indexed` and `indexed_length` load the header directly. Integer stores check integrality and the signed
64-bit conversion range before truncating to the element width; other values
use the generic modulo conversion. Typed-array storage never moves or resizes
(`FixedTypedStorage`), and the frame keeps the array alive while the loop runs.

The emitter lowers recognized Math natives (floor, ceil, sqrt, min, max) to
inline f64 instructions behind tag checks, with the native helper as fallback;
numeric equality takes the same inline path as relational comparison. Branches
on known booleans compare directly. Fallback versions and the probe path are
cold blocks, laid out after the hot code. Graph values and block parameters whose facts admit strings, objects
or other heap values are rooted in frame slots; `this` and arguments are
borrowed from the caller and are not rooted. Generator graphs root every value
because resumption restores them from the frame. Incoming block parameters are
retained before replacing their slots when they hold heap values. After slot
assignment, a parameter whose arguments already own its destination slot on every
incoming edge forwards that ownership without a retain, store or release. Other
parameters still retain in parallel before any slot is replaced, preserving swaps.
Temporary
values are cleared once per block, before its terminator, skipping the call when
no temporary is pending, so they live at most until the end of their block;
every heap result they could release early is also rooted. Frame exit releases
the slots. This conservative scheme keeps each value rooted until it is
overwritten or the frame exits.

Property and element reads can instead be borrowed: they are neither rooted nor
retained by the inline element path. The last semantic pass accepts a read when,
throughout its live range, nothing writes the heap, reenters, or redefines a
receiver along its chain of reads (`height` for `height[x][y]`). A store may be
the last use, since stores retain before releasing. Either every receiver in the
chain up to an owning receiver is known to be an object or reads it as an indexed
iteration element (only arrays and typed arrays reach those reads), so it owns
the result, or the read dies within its block before the temporary clear, which
covers fresh slow-path results such as string characters. Other block parameters and returns still retain what they
receive. Trivial block parameters introduced by specialization are folded before
borrow analysis, preserving borrows through guard blocks. `--semantic-ir` lists
the borrowed reads.

Frames share one contiguous slot stack and one temporary stack in `JitCtx`;
ordinary functions number their rooted values densely and reserve that many
slots on entry, and `JitCtx` exposes the current frame's slot base and the
temporary stack's storage, length, capacity and frame floor. Frame exit releases
slots and temporaries under one heap borrow. A rooted graph value's slot takes
a reference of its own: ordinary stores increment inline, and a temporary holding
the same value keeps its reference until the block's clear. Object, array, map
result, instance and string-split allocations instead use owned-result helpers
when slotted. The frame adopts the allocation's reference directly; these helpers
never register temporaries. Property, element and global reads stored this way
skip temporary registration and acquire the slot's reference before replacing its
old value. Global cache misses have a matching borrowed-result helper. Constructor
results always have object facts; a proven primitive return selects the instance,
and a proven object return selects that object before slot assignment. A slot only
holds an old value to release when its defining block lies on a control-flow
cycle (a strongly connected component, so hoisting out of a loop removes the
release); functions with handlers or generators treat every block as cyclic.
Other block parameters follow the same rule, with every incoming heap value
incremented first so swapped parameters stay alive. Releases decrement inline
while other references remain, and otherwise call the runtime, which queues the
slot for disposal. Inline temporary pushes call the runtime only to grow the
stack. Bindings without SSA values (written by assignment or scope entry) keep
the runtime store, which takes over the most recent temporary. Generator frames
keep the helper path. Error and
exception checks load fixed `JitCtx` fields inline. Scalar replacement of
unescaped objects, general branch-sensitive facts, structural facts about loaded
values (such as an array of `{x: number}`) and object kinds are not
implemented.

## Compiled variant identity

Closures identify logical functions. `VariantId` identifies a stable entry slot,
while `Entry` describes the native body currently implementing that slot.
Compilation jobs, publication, direct-call targets, and current emission plans
carry the variant separately from the logical function. Recompilation replaces
the body without changing either identity; fallback addresses continue to name
specific older bodies. OSR fragments retain their separate site-based entries.

Ordinary variant slots equal logical function IDs, preserving generic dispatch
stubs and the direct-entry layout. Specialized variants use stable atomic entry
cells owned by a shared registry; their IDs identify those cells. The registry
is shared across compiler snapshots and retains cells until engine shutdown.
It interns contracts by logical function and ordered input requirements, bounded
to four variants per function and 128 per engine. Ordinary and specialized native
bodies are published independently. Warmup, pending feedback, and observations
still belong to logical functions.

After inlining, fixed-arity calls can select variants for integer/number inputs
or observed shapes of the receiver and arguments. Requirements come from actual
guards emitted for an optimized callee's entry, after versioning has applied its
profitability checks. The shared assumption/versioning guard emitter reports its
supported input facts; shape guard emission uses its existing preservation analysis
to report input shape checks eligible at entry. Call specialization consumes this
report rather than recognizing a second set of prologue instructions. Analysis
disables inlining and nested call-variant selection, and is reused per target and
offered input shapes within each caller plan. Caller shape hints come from explicit
shape checks, entry assumptions, and monomorphic property feedback. They are planning
hints, so callers still emit guards even when an intervening effect could have
invalidated the original fact. The ordinary callee need not already have a matching
entry guard: own-property feedback retains up to four shape/slot alternatives,
and an offered shape can select one during contextual planning. Only supported
guards emitted at entry enter the resulting contract; unused hints and guards
after reentry are excluded. Prototype reads do not supply own-slot alternatives.
Arithmetic uses or property feedback alone do not create requirements. Caller guards are explicit
semantic branches, with the ordinary call as fallback. The callee retains the
original closure and boxed ABI. Calls with rest/arguments, generators, exception
handlers, lexical receiver resolution, or mismatched arity are excluded for now.
Shape contracts use feedback available when the caller is compiled; callee
feedback alone does not currently trigger recompilation of its callers.

A specialized body starts with `EntryAssume` facts, distinct from the speculative
`Assume` proposals. Numeric input facts persist through optimization. Shape facts
participate in available-check analysis and expire across invalidating effects.
The assumption's boolean result is always true, allowing redundant shape checks
and branches to simplify without discarding the entry contract.

First guarded use installs the ordinary body into the variant cell and requests
optimized compilation. Execution continues through that body until the execution
thread publishes the specialized entry. Synchronous compilation uses the same
contract and publication path. Recursive calls reuse interned variants without
waiting for their specialized bodies. Compilation failure retains the ordinary
body. Variant choices are part of specialization-plan equality; fallback addresses
still reference immutable body generations rather than mutable variant cells.

`JS_JIT_CALL_VARIANTS=0` disables selection. Caller-side dispatch does not move
unrelated internal guards or introduce context-specific feedback.

### Guarded alternatives and polymorphic calls

Semantic dispatch lowers a bounded list of guarded alternatives around an operation.
Each alternative carries facts about evaluated SSA values and an implementation.
The shared guard vocabulary produces tests, each miss advances to the next
alternative, and all successful results join the original operation's continuation.
The original operation remains the final fallback. Inlining, property and array
transforms share the same operation-splitting helper. Unsupported fact tests reject
an alternative before rewriting the graph.

Calls currently use up to four observed code targets, in discovery order. Each
contributes a code check, an optional variant with useful callee entry requirements,
and an ordinary direct-call alternative. A contract miss therefore retains direct
dispatch for the known function. Code checks are visible to semantic analysis;
emission omits its defensive target check when dominating facts already establish
that code. The actual closure and receiver are preserved on every path.

Generic fallback calls retain observation until the target set reaches its bound.
Observation caches suppress repeated identical callees; per-owner target-set
versions prevent repeated recompilation without losing changes seen by another
owner. Saturation freezes the selected set, leaving further targets generic.

Call feedback growth and numeric despecialization share an update result indicating
whether the policy changed and whether observation is complete. Caches separately
count failures before disabling numeric speculation. Feedback changes replan the
logical function's ordinary body and all active specialized variants, preserving
immutable entry contracts. Background requests remain pending while any variant
of that function is compiling and are reconsidered after publication. Synchronous
invalidation clears variant cells so the next guarded call services pending work.

This first use of guarded alternatives does not add polymorphic inlining or
context-specific shape observations. Callee feedback remains shared; several
receiver contracts for the same function need further feedback support.

`JS_JIT_CALL_SPECIALIZATION=0` suppresses callee contract analysis and specialized
variant requests while retaining bounded polymorphic direct-call dispatch and
get PICs. It is independent of `JS_JIT_INLINE`; together these switches separate
callee specialization from the other effects of inlining. Ordinary bodies retain
their own speculative optimization and feedback recompilation.
