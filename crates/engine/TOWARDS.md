**The intended end-state**

This document proposes an architectural direction and migration, rather than describing features already available. DESIGN.md describes the current implementation. Details explicitly left open here should be resolved as the corresponding stage is implemented.

The compiler should progressively make a program concrete within one shared graph infrastructure, from language operations down toward machine instructions. Lowering replaces operations locally and introduces implementation constraints as they become relevant. Different abstraction levels can coexist, so target information can inform optimization before decisions become expensive to undo.

The structural backbone remains an ordered CFG with SSA values:

```text
Module → Function → Region → Block → Operation
```

Regions can nest. Values connect operation results, operands and block parameters across this structure. Machine instructions are another operation family, rather than a separate child hierarchy beneath semantic instructions. A lowering can introduce several operations or blocks; provenance records where they came from without restricting where they may subsequently move. A module groups compilation data without necessarily defining an isolated JS environment.

The hierarchy describes organization, not nested storage ownership. Functions own their blocks and region records. Each block belongs directly to one region; a parent region includes its descendants' blocks transitively. Initially, regions form a tree rooted at the function region, using the existing loop/function candidates. Supporting arbitrary overlapping or multi-entry regions is outside the initial migration.

Regions make the scope of reasoning explicit. They record entry guarantees, effects and the guarantees preserved through their blocks, together with boundaries and fallback relationships. Blocks retain explicit execution order; ordinary value dependencies and effect contracts determine which transformations are legal. This gives us a place to preserve knowledge through lowering without requiring a Sea of Nodes or a global effect graph.

A guarantee applies only where every incoming execution path establishes it and execution has not invalidated it. Entry guards establish facts on their successful edges; guard failures and fallback paths do not inherit those facts. Exceptional exits and joins need their own valid incoming facts. Facts about immutable SSA values can survive effects that invalidate facts about mutable storage. Region membership alone is never sufficient evidence for a guarantee.

The design is data-oriented: indexed entities, associated data and systems that analyze or transform it. The ECS analogy is a mental model, not a requirement for an ECS implementation or a new framework for declaring pass contracts. An enum of operation families, existing Facts and ordinary pass functions are enough to start. A pass acts on the operations and information it understands; lowering replaces some operations while keeping their replacements in the same graph. Simplification, parameter cleanup and applicable load/motion analyses can then operate on the exposed work. Their applicability still depends on the operations' semantics and on valid analysis data.

There is no universal sequence of operation families that every frontend must traverse. A frontend using our runtime can produce runtime operations; another can produce memory or arithmetic operations directly, without sharing any JS operations. Likewise, a backend can consume a mixture of compound and primitive operations. Further lowering is necessary only for operations that its chosen implementation cannot consume. Enum boundaries organize the data; pass boundaries express concrete transformations.

The common infrastructure uses indexed entities and associated analyses. Typical operations and generic graph algorithms form a common core. Frontend, runtime and target extensions provide additional operation families, contracts and rules. JS coercion semantics, this engine's NaN boxing and a target's register constraints belong to different layers, but can participate in the same infrastructure. Extensibility can begin with statically compiled Rust code; it does not require a dynamic plugin system.

There are two useful kinds of explicit effects. Program operations describe what execution reads, writes or invalidates. Compiler passes describe which graph data they read and change, allowing affected analyses to be invalidated and recomputed. The ECS-like aspect is composing behavior around explicit data access; the Salsa-like aspect is tracking dependencies of derived information. Neither requires making every algorithm a rewrite rule or every mutation a fine-grained incremental query.

Semantic rules establish legal alternatives; target constraints and costs help choose between them. Sharing a computation, hoisting it or choosing a representation should remain revisable when its register pressure becomes apparent. The longer-term ambition is to carry this continuity through scheduling and allocation, progressively adding placement constraints and committing decisions where enough is known. E-graph alternatives and compiled rule matching could support this, but are optional mechanisms rather than the foundation.

Cranelift and our AArch64 backend currently share CLIF storage for the optimized path. The experimental AArch64 baseline instead consumes the semantic graph directly. The immediate goal is to make more emitted work visible in the graph and let each backend consume the operations it supports, without introducing a parallel semantic operation model at the CLIF boundary.

Shared runtime lowering now produces a distinct low-level operation family within `Op`, using the graph's ordinary values and block parameters. It preserves compound operations that a backend can consume directly. Cranelift-specific expansion of `IndexedLoad` follows that shared lowering. At this intermediate migration stage both optimized backends still consume the resulting CLIF; the next step is for AArch64 to consume the lowered graph, including retained indexed loads, directly. Both routes can continue using `JITModule` for symbols, relocation and executable memory. Owning the graph and native allocation does not require replacing that loader.

The semantic graph already has SSA construction and binding promotion. Runtime lowering now builds SSA for its additional control flow and mutable temporaries in the same graph infrastructure, including multiple call results and scalar representations. It builds a replacement graph and preserves source provenance. This rewrite boundary is an implementation choice: earlier lowering passes can progressively shrink it without changing the graph's semantics or introducing a mandatory sequence of operation families.

On the Cranelift path, our graph can use target constraints and pressure estimates before emission, but cannot assume feedback from its allocator or revise choices during allocation. The native path owns allocation, although its current CLIF input still separates those decisions from the semantic graph.

**Examples of the intended benefits**

These are benefits to work toward. Some become available by exposing lowering in the graph; allocation-driven decisions require deeper backend integration.

**Preserve a stability proof while revising how a value is computed and stored.** Consider a loop repeatedly accessing `object.buffer[i]`. Entry guards and the region's effect summary establish that resolving `object.buffer` is safe and its value remains unchanged. We can resolve that property once before the loop and carry the preservation proof through lowering into cache checks and memory accesses.

That does not require hoisting every derived address or keeping every shared calculation live. Depending on register pressure, we could keep the buffer reference in a register, recompute cheap element addresses near their uses, or fold offsets into machine addressing modes. If keeping the buffer reference live is expensive, the same proof could permit reloading it near a use, provided the lowered read is safe and the receiver stays alive. This offers an alternative to spilling the previously loaded value to the stack.

Such a reload must be equivalent to the established direct access: it cannot repeat arbitrary JS property lookup with new accessor calls, exceptions or reentry. Its storage and lifetime dependencies must still hold at the new location.

The region establishes which implementations are legal; target costs help choose between them. Hoisting and common-subexpression elimination expose opportunities that remain revisable as machine constraints become concrete. The hoped-for improvement is fewer checks and loads where they matter, without paying for unnecessarily long live ranges in hot loops.

**Let shape specialization and hoisting cooperate through assumptions.** Several property reads on the same object can share a shape guard and become direct slot reads. But lowering the guard and its fallback first can make the region appear effectful and block hoisting that was previously possible. Hoisting preserved reads first reduces the number of accesses, which can make shape specialization less worthwhile. The transformations change each other's profitability, and their ordering can obscure the proofs needed to combine them.

Represent specialization as a conditional region hypothesis: if the receiver has shape S and the relevant layout remains valid, these property reads can use direct slots. Keep separate the cost and failure edge of establishing that assumption, the effects that could invalidate it, and the operations that exploit it. Stable shape does not imply stable property values: hoisting a slot load still requires a no-clobber proof for that property, alongside the usual execution and lifetime constraints.

The successful region should be analyzed under the established assumption without automatically inheriting effects from a fallback outside it. Moving a shape check still requires proving its own dependencies and failure behavior safe at the new location. Introducing guards should preserve the reasoning needed to move accesses; moving accesses should trigger reconsideration of guard profitability.

This is an opportunity for e-graph-like alternatives under explicit assumptions: compare guarded direct reads, hoisted reads, and combinations before committing. A generic property read and a slot load are equivalent only under the required guards and preservation proofs. The goal is to reduce dependence on pass order without requiring an e-graph implementation as the first step.

**Optimize runtime bookkeeping alongside the operations that need it.** A specialized numeric path may still acquire boxing, temporary roots or cleanup during emission. Making those explicit graph operations lets facts and simplification act on them. A proven immediate value needs no heap root; a value used only by numeric operations may remain unboxed until a fallback boundary. The benefit is eliminating overhead introduced after much of today's semantic optimization has finished.

**Extend the operation vocabulary through contracts.** A hypothetical statically compiled extension for a host-provided numeric buffer could supply operations with explicit result types, read/write effects, lifetime requirements and lowering rules. Region analysis, load reuse and motion could then use those contracts, with buffer stability guaranteed by the host integration where required. The extension supplies its semantics while existing algorithms provide optimization, reducing the need to add operation-specific cases throughout the compiler. This example does not imply an existing runtime plugin API.

**Recompute analyses according to what changed.** Replacing arithmetic inside a loop may invalidate value facts while leaving its CFG and dominance unchanged. Retargeting a branch can invalidate dominance and region boundaries without affecting unrelated functions. Explicit mutation contracts make those dependencies enforceable. Coarse invalidation can first prevent stale-analysis mistakes; finer dependency tracking can later avoid redundant work where measurements justify it.

**Progressive migration**

The broad structural implementation of steps 1–3 was explored and set aside:
it introduced ownership and analysis machinery before a concrete optimization
needed it. The retained refactor gives existing regions identities, block sets
and fallback links. It does not establish a region hierarchy or persistent
proofs, and does not complete the stages below.

Treat these stages as architectural options, not a prerequisite checklist. In particular, the explicit pass contracts and dependency framework outlined below are optional future mechanisms, not requirements for the current operation-family migration. Ordinary pass ordering and existing analysis invalidation are sufficient for its first slices.
Experiments should start with a concrete optimization interaction, identify
which proof or effect information is lost, and introduce only the representation
needed to improve that example. Shape specialization and hoisting remain one
such target; checked typed-array reads provide a smaller first lowering slice.

I’d migrate by making the existing architecture explicit, then extending it downward. Each stage should leave a working compiler and provide an immediate benefit.

**1. Establish the structural model**

Keep the current indexed CFG/SSA storage. Introduce explicit function and region identities, and separate structural data from JS/runtime metadata.

- Modules own or reference functions.
- Functions own blocks and a root region.
- Nested regions group blocks; a block has one direct enclosing region.
- Operations remain ordered within blocks.
- Values remain operation results or block parameters.
- Source provenance stays independent of containment.

Initially, modules can be thin containers over existing compilation units. Avoid changing global-environment behavior or code publication as part of this work.

**Result:** a clearer ownership model without changing compilation behavior.

**2. Materialize existing regions**

Turn the transient region plans and returned block sets into persistent region records.

Separate:

- Structural boundaries: membership, entry, exits, nesting.
- Entry guarantees: facts established before execution.
- Effect summaries: what execution may change.
- Preserved guarantees: which facts remain valid throughout.
- Version relationships: specialized region and fallback.

Keep the existing loop/function region discovery initially. Don’t attempt arbitrary overlapping regions yet.

Have versioning produce region identities, then make load reuse and hoisting consume them. Preserve explicit guards in the CFG: region metadata describes their consequences; it does not replace their execution.

A region identity names a record, not a permanent proof. Transformations must assign membership to new or split blocks and invalidate affected boundaries and guarantees before they are consumed again. Preserve identity where the region still has a coherent meaning; allow rebuilding or retiring records when structure changes. The exact repair mechanism remains an implementation choice.

**Result:** later passes can inspect why a region is optimizable instead of reconstructing that reasoning from block sets.

**3. Introduce mutation and analysis contracts**

Give graph mutations a small common interface that records what changed: operations, operands, CFG edges, membership or contracts.

Use that to invalidate derived data:

```text
CFG changes       → dominance, loops, region boundaries
Operation changes → facts, effects, cost estimates
Guard changes     → dependent region guarantees
```

These are examples, not an exhaustive dependency table: operand changes can affect effects, CFG changes can affect facts, and membership or contract changes can affect region summaries. Mutating passes must report changes through the common interface; analyses must declare their dependencies or conservatively depend on the enclosing function revision. Direct edits that bypass invalidation cannot be part of the supported pass interface.

Start conservatively, with function-level or region-level revisions and recomputation. Add finer dependency tracking only where compilation profiles justify it (probably nowhere, for now).

Distinguish program effects from compiler-pass accesses. A pass can declare that it reads CFG/facts and writes operations, but prerequisites and ordering remain explicit.

**Result:** analysis validity becomes an enforced convention rather than something every pass author must remember independently.

**4. Separate mechanisms from operation rules**

Progressive lowering already happens within the semantic graph. In [arrays.rs](src/jit/semantic/arrays.rs), eligible `Array.map` calls expand behind native/receiver checks into an explicit loop, result allocation, element reads, callback calls and result writes, retaining the original call as fallback. Related array methods use the same machinery. Recognized `Array.push` calls become `ArrayAppend` operations and a resulting length, using guards for feedback-based recognition or the engine's fixed-method guarantee for directly proven array allocations. This exposes builtin work to ordinary graph optimization instead of leaving it hidden inside a call.

These expansions are compiler-intrinsic behavior expressed as graph transformations. A builtin's general execution path and its specialized expansion implement the same operation under different established conditions. Recognition, guards and semantic rules justify the expansion; inlining, simplification and region analysis can then optimize its pieces. Generalizing this existing pattern is as important as moving later CLIF-emission decisions into the graph.

The JIT already has much of the machinery behind treating runtime helpers and compiler intrinsics as implementations of known operations. [helpers.rs](src/jit/helpers.rs) defines helper identities, implementation pointers and ABI signature shapes. [effects.rs](src/jit/effects.rs) describes their reentry, reads/writes, allocation, errors, assumptions and temporary-stack effects. Separately, semantic operation contracts describe effects and result ownership using operand facts, while emission selects typed implementations or helper fallbacks. These are existing foundations to connect and generalize, rather than a new contract system to build from scratch.

The intended distinction is between an operation's semantics and its implementation: a helper call can provide a general implementation, while specialization lowers proven cases into explicit graph operations. Helper bookkeeping such as retaining results, registering roots or updating error state must remain accounted for when replacing that call. ABI and temporary-stack details belong to the implementation contract; they should not become universal properties of every semantic operation.

This also makes baseline and optimized compilation policies of the same compiler. A baseline policy can select general call implementations and perform only the lowering needed to emit executable code. An optimized policy can spend more time establishing assumptions, expanding operations, inlining callbacks and choosing specialized implementations. Both share operation semantics, correctness contracts and lowering machinery; feedback changes the available facts and the budget for exploiting them. Every policy must still discharge the guards, ownership and error-handling obligations of the implementations it selects.

The experimental [AArch64 baseline](src/jit/aarch64/baseline.rs) already implements a cheaper terminal path: it consumes Graph directly, uses boxed stack slots and runtime helpers, and emits machine code without CLIF construction or general register allocation. It demonstrates that higher-level operations can be valid backend input. Its implementation selection, ownership bookkeeping and target emission remain fused; extracting shared lowering must not force baseline through expensive expansion or analysis that it does not need.

[Generator suspension lowering](src/jit/semantic/suspension.rs) provides an existing graph-to-graph example. It turns yields into resume-region graphs with explicit saved values and binding restoration. Those ordinary graphs can use either baseline or CLIF emission. The transformation changes what the backend must implement while retaining the graph vocabulary and infrastructure.

Retain a statically defined operation vocabulary initially. No plugin loader or dynamic ECS is necessary.

Extract common interfaces for:

- Operands, results and successors.
- Effects and access resources.
- Fact propagation.
- Verification.
- Rewrite and lowering rules.

Move JS knowledge behind these interfaces progressively. Keep specialized planners where useful; generic graph algorithms need not become rewrite rules.

The main operation families would be:

| Family | Responsibilities |
|---|---|
| Core | CFG, SSA, basic typed operations |
| JS | Coercions, bindings, properties, calls, constructors |
| Runtime | Boxing, shapes, ownership, roots, helper conventions |
| Target | Machine operations and physical constraints |

Add multiple-result support when introducing operations that benefit from it. Separate semantic value facts from physical representations.

**Result:** reusable algorithms stop depending directly on the full JS operation enum.

**5. Move one lowering path into the graph**

The first slice now expands known-kind, unbounded `TypedIndex` operations with
proven numeric keys into position conversion, exactness and bounds checks,
a bounded read, and an undefined result on failure. Existing region membership
is extended to the generated blocks. Load reuse can share position conversion
and exactness checks for the same SSA key; each receiver keeps its own bounds
check. This runs after hoisting and before view placement. A subsequent late
lowering now exposes storage extraction, runtime-agnostic indexed loads and
numeric result conversions in the graph. The native backend retains indexed
loads; a separate graph expansion supplies addresses and scalar loads for
Cranelift. Raw values currently stay within their block, and a keepalive protects
the runtime receiver. This does not establish persistent guard proofs or complete
this stage.

The next small interaction exposes strict undefined comparisons as the existing
`IsUndefined` predicate. Fact analysis carries its edge restrictions through
dominated forwarding blocks, allowing a typed read's number-or-undefined result
to become numeric after an undefined default. This keeps the semantic rule in
comparison lowering and the propagation mechanism in fact analysis. It adds no
operation family or persistent proof storage; the restrictions are recomputed
from the CFG.

Choose a narrow hot path—typed-array numeric loops are a reasonable candidate.

Replace emission-time decisions with graph transformations:

```text
JS operation
  → specialized operation
  → explicit representation conversions and runtime accesses
  → typed arithmetic, addresses and loads
  → CLIF
```

Allow these operation families to coexist. The indexed-load slice lowers the runtime-specific `TypedIndex` into explicit storage extraction and a runtime-agnostic indexed load, with numeric conversions separate. The load takes a base pointer, integer element index and scalar format; the producing lowering establishes access validity and storage lifetime. It does not know typed-array headers or JS indexing semantics.

The AArch64 backend consumes this indexed load directly. The Cranelift path expands it into address arithmetic and a scalar load. Both forms belong to the same operation vocabulary, and another frontend can produce the memory operation without using our runtime. Baseline can retain its direct higher-level implementation. This is a bounded migration of emission decisions into graph passes, not a requirement to rewrite every emitter at once.

Run existing simplification and applicable load/motion passes after lowering. Preserve region identities where they remain meaningful, and update or invalidate their summaries.

**Result:** a concrete test of shared graph lowering with different backend operation sets.

**6. Expand progressive lowering**

Move additional emitter responsibilities into explicit operations as needed:

- Property-cache fast and slow paths.
- Direct and helper calls.
- Boxing and unboxing.
- Retains, releases and rooting.
- Exception checks and fallback control flow.
- Temporary storage and lifetime boundaries.

Use the same operation storage rather than nesting generated instructions under their source operation. Track provenance separately.

**Result:** the graph increasingly represents the actual program delivered to Cranelift, including its costs and lifetime pressure.

**7. Extend target feedback**

Introduce register classes, fixed-register constraints, rematerialization candidates and pressure estimates where useful. Use those to inform hoisting, sharing and representation choices.

The AArch64 backend already owns machine lowering and allocation. Extend its connection to graph decisions only where measurements justify it, while preserving the Cranelift path. Shared graph lowering should not depend on completing that deeper integration.

Likewise, compiled rule dispatch, e-graph alternatives and fine-grained incremental queries can come later. None is required to establish the model.

**Validation throughout**

Preserve behavioral tests and map-output checks. Add structural verification for region boundaries, guarantee validity and SSA after transformations. For migrated hot paths, compare compilation time, lowered graphs, assembly, spills and execution time against the existing path.

**The next milestone is a concrete optimization that benefits from preserving reasoning across transformations.** Extend the checked typed-read slice only where an example benefits; use the shape-specialization and hoisting example to expose its actual obstruction before introducing more structural or analysis infrastructure.
