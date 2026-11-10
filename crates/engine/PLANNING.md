# Planning, guards and versioning

Sketch of where the speculation pipeline is going. It describes the target shape, what exists today,
and an ordered migration. Nothing here is implemented unless it says so.

## Principles

- **Effects are what an op does.** `Effects { memory, control }` comes from `contract(op, facts)`.
  Every pass reads effects; nothing but the planner reads requirements.
- **Facts are what we know.** `Facts` holds value facts (types, identities, aliasing). Contracts read
  facts, so knowing more makes ops cheaper without any pass having to be told.
- **A guard is a check whose success edge adds facts.** Type refinements, effect removal (via
  identities feeding contracts) and aliasing facts are all "facts on an edge". A null check and a
  speculative shape check are the same thing; what differs is whether the failure edge is cold and
  speculated on, which is a cost-model property.
- **"Guard" is a role, not a kind of op.** Any check op (`IsNumber`, `IsShape`, `IsCode`, ...) feeding
  a branch is a guard when its success edge implies facts. There is no separate guard op or guard
  structure: generic passes (LICM, GVN, PRE, DCE) move, merge and remove these checks like any other
  op, using their effects and `Access` conflicts, without knowing about JS.
- **Value facts only.** Facts about locations ("`Math.sqrt` is still the builtin") become value
  facts by loading once (a probe) and guarding the loaded value. That a later read of the location
  is still that value is load reuse: GVN plus `conflicts()`. `Facts` stays a dominance-based
  dataflow without kill sets.
- **Regions are an output.** A region is code dominated by a set of entry guards. It is produced by
  guard placement and specialisation, not planned up front.
- **The planner is smarter, not lower-level.** It decides which guards are worth creating, what to
  observe when it lacks information, and where duplicating code pays off. It does not move code.
- **Guard failure lowering is late and orthogonal.** Whether a failing guard runs an inline slow path
  or leaves for a continuation (continuations and sibling copies are the same mechanism) is a
  compile-time cost decision made after optimisation. ICs are a localised form of the same logic;
  whether a site is an IC or a guard should fall out of the cost model.

## Layers

```
runtime            jit/profile.rs (today jit/planner.rs)
                   feedback: call targets, natives, read types, shapes, guard failures
                   recompile queue
                      |
                      v  snapshot (Compiler::update_feedback -> specialize::Repository)
1. inline          specialise/inline rounds from call-target feedback (budget)
                   the call target's code identity is checked at the call site
                      |
2. normalise       generic transforms + JS lowering to a fixpoint, no new speculation
                   GVN / load reuse, LICM, DCE, typed_lowering, comparisons, coercions, folding
                      |
3. plan checks     planner (semantic/planner/)
                   - candidates from the guard table + profile
                   - evaluation on the normalised graph: Facts::assuming(candidate) vs Facts,
                     loop-weighted, minus check cost and failure rate
                   - insert chosen checks at their use sites (cold failure edge)
                   - observation injection where feedback is missing
                      |
4. optimise        generic transforms + JS lowering again
                   check PRE / LICM: place each check as high as it stays valid
                      |
5. specialise      planner: duplicate code where a check was blocked by a conflicting write
                   and the cost model says the copy pays off; rerun 4 on the copies
                      |                         (3-5 may iterate under a budget)
6. late lowering   guard failure edges: inline slow path vs continuation, from the cost model
                      |
emit               emit/
```

Ownership: steps 1, 3 and 5 are the planner (runtime-informed decisions); steps 2 and 4 are
`semantic/transforms/` and `semantic/js/`; step 6 is runtime lowering. The planner never moves code,
and the generic passes never decide what to speculate on.

## Components

### Facts (semantic/facts.rs)

| Component | Today | Target |
|---|---|---|
| `types` | `Types` per value, edge refinement for some checks | unchanged |
| `memory`, `control` | per-inst columns from `inst_contract` | unchanged, derived from contracts that read the components below |
| identities | none; `closed_calls` side table, `plan_region`'s pretend-closed clone, `EntryGuard` | per value: known code (`FuncId`), native, or shape |
| aliasing | `unescaped()` in `optimize/analysis.rs`, computed on demand per block scope | per value and point: fresh / distinct allocation root, not escaped before this point |
| `indexed`, bounds | `string_split` set, induction bounds | unchanged |

Edge refinement becomes table-driven: `implies(op) -> (facts on true edge, facts on false edge)`.
`IsNumber`, `IsCode`, `IsNative`, `IsShape`, `IsTyped`, `IsUndefined`, ... all go through it.
`guard_fallback` and `EntryGuard::of` disappear.

`contract(op, facts)` consults identities: a `Call` whose callee is a known `FuncId` takes that
function's summary effects; a native takes the native's effects; a `Get` on a known shape takes
`ShapedGet`'s effects. Summaries move onto the graph keyed by `FuncId` (instead of `closed_calls`
keyed by `Inst`).

### Effects and accesses (jit/effects.rs, semantic/access.rs)

Done:
- `Effects { memory: Memory, control: Control }`; `Memory` reads/writes `Locations` (heap, global,
  code); reentry is "writes code".
- `Access { location, base, field }` refines a `Locations` class for one object and one field;
  `Writes::clobbers` and `Guarded::breaks` use it.

Next:
- A `conflicts(write, read)` usable by generic passes directly, not only through `Writes`.
- `removable(inst)` for DCE, once side-table uses (`graph.views`) are real uses.

### Guard table (planner-owned)

```
op kind -> [ { observe: feedback needed,
               guard:   check op to insert,
               implies: value facts on success,
               becomes: cheaper op (optional, else JS lowering picks it up from facts) } ]
```

Examples:
- `Call` to a global → observe call target → `IsCode(callee, f)` → identity → summary effects.
- `Math.x(...)` → observe native → `IsNative` → identity → native call, coercion-free if typed.
- `Get` → observe shape → `IsShape` → shape identity → `ShapedGet`.
- `GetIndex` → observe read type / storage → `IsTyped` + bounds → `TypedIndex`.
- numeric use of an unknown value → observe number → `IsNumber`/`IsInteger` → `Types` refinement.

Runtime invariants (`StableIndexStorage`, `FixedTypedStorage`, `DirectIteration`) stay listed in
`effects::assumptions(helper)`; they are not guards.

### Profile (jit/planner.rs → jit/profile.rs)

`jit/planner.rs` is runtime feedback plus a recompile queue; it plans nothing. Rename to
`jit/profile.rs` with:
- `Profile`: call targets, natives, read types, shapes, call counts (the current `Planner` maps).
- `Recompiles`: the pending queue (`request` / `take` / `has_pending`).
- New: guard failure counts. `bailouts.rs` already counts continuation transfers behind
  `JS_JIT_BAILOUT_LOG`; the counts should feed the profile so a guard that keeps failing stops being
  speculated on.

`Compiler::update_feedback` keeps copying a snapshot into `specialize::Repository`; the four
`*_feedback` maps there become one `Profile` snapshot. That needs `Profile` split into a cloneable
feedback part (targets keyed by `FuncId`, natives, reads, shapes) and runtime-only bookkeeping
(`calls` keyed by pointers, the `observed*` dedup sets), and the `get_refinement` gating moved to the
readers. Deferred to step 7, where the readers change anyway.

## Pipeline

Today (`semantic/mod.rs`):

```
plan_canonical_with_budget
  expansions, specialise/inline rounds, js::simplify
  prove_closed_calls -> graph.closed_calls
  direct_calls
  observations::plan  -> numbers
  reads::plan         -> typed reads
  shapes::plan        -> shape accesses
  conditional_plan    -> conditional calls (+ identities)
  arrays::push_calls
  => SpecializationPlan { conditional, inlines, reads, shapes, numbers }

optimize_planned
  caches::apply, reads::apply, comparisons::expose, sink/lazy captures, ...
  version_with_report  (optimize/planning.rs + optimize/versioning.rs)
  js::simplify, reuse_loads, hoist per region, ...
```

Target:

```
inline       expansions, specialise/inline rounds (unchanged)
normalise    JS lowering, GVN / load reuse, LICM, DCE to a fixpoint
plan checks  candidates = guard table x profile   (replaces observations/reads/shapes/conditional plans)
             evaluate   = Facts::assuming(candidate) diff on the normalised graph
             insert     chosen checks at their use sites (cold failure edge)
             observe    ops with table entries but no feedback get Observe
optimise     JS lowering under the new facts; GVN; check PRE / LICM
specialise   versioning where a check is blocked and the copy pays off; optimise the copies
             (plan checks .. specialise may iterate under a budget)
lower        per guard failure edge: inline / continuation / sibling copy
```

The current `plan_canonical_with_budget` / `optimize_planned` split no longer matches: only inlining
stays before optimisation. `SpecializationPlan` stops being a precomputed bundle handed to
`optimize_planned`; the planner runs between optimisation rounds instead.

## Where today's code goes

| Today | Concern | Target |
|---|---|---|
| `planning.rs` `invariant_guards` | hoisting existing guards | guard LICM / PRE |
| `analysis.rs` `probe_reads` | hoisting loads | LICM over `Access` |
| `analysis.rs` `closure_failure` | legality | generic conflicts check |
| `planning.rs` `entry_checks`, typed receivers, bounds | guards from feedback | planner candidates |
| `analysis.rs` `requirements`, conditional identities | guards from summaries | planner candidates |
| `planning.rs` `covers` / `merge` | guard subsumption | GVN / PRE on guards |
| `versioning.rs` block copy, SSA repair | duplication | generic versioning transform |
| `versioning.rs` entry chain | emitting guards at entry | planner insert + placement |
| `versioning.rs` `TypedIndex` / `CallNative` / `AsNumber` rewrites | ops under guards | JS lowering from facts |
| `versioning.rs` continuations, cache priming, previous-entry fallback | failure handling | late failure lowering |
| `GuardValid` / `GuardCommit` cells | cached guard validation | kept; a check op with a cheap fast path |
| `specialize.rs` summaries | callee effects | facts identities + summaries on the graph |
| `graph.closed_calls` | per-call effect override | removed (identity facts) |
| `jit/planner.rs` | runtime feedback | `jit/profile.rs` |

## Migration

Each step should keep tests green and the belgian_uplands hashes unchanged, and is checked with
`--compile --planner` notes on small loops.

1. **Rename `jit/planner.rs` to `jit/profile.rs`.** Split `Profile` from the recompile queue.
   Mechanical.
2. **Identity facts.** Add the identity component, refine it on `IsCode` / `IsNative` / `IsShape`
   edges, move summaries onto the graph by `FuncId`, have `contract(Call)` read them. Remove
   `closed_calls` and the pretend-closed clone in `plan_region`.
3. **Guard table for edge refinement.** `implies(op)` drives `Facts` refinement; replace
   `guard_fallback` and `EntryGuard::of`.
4. **Aliasing facts.** Folded into step 5. `unescaped()` is an optimistic fixpoint over a block
   scope (region, body, or body minus fallbacks): uses and alias edges outside the scope are not
   seen, so the same value can be fresh in one scope and escaped in another. A scope-free value fact
   needs a flow-sensitive definition ("not escaped before this point", from escape points and
   dominance) and only makes sense once `hoist`/`reuse_loads` stop being region-scoped.
5. **Generic LICM / GVN over checks.** Checks hoist like any read, carrying their branch; `hoist`/`reuse_loads`
   stop being region-scoped, and `unescaped()` becomes the flow-sensitive aliasing component in
   `Facts`. Fix side-table uses and switch DCE to `removable`.
6. **Guard placement.** PRE-style placement replaces `invariant_guards`, `probe_reads` and
   `covers`/`merge`. Conditional summaries (callee bodies that read globals) apply when a dominating
   `IsCode` guards a `GlobalRead` of each required name; after step 2 this trusts region legality
   (`closure_failure`) for "no write in between". Placement should make that a `conflicts` check.
   The same shortcut gives any `GlobalRead`/`ProbeGlobal(name)` the identity of a dominating guarded
   probe of that name, because `apply_version` clones callees as fresh reads instead of using the
   guarded value; rewriting in-region callees to the guarded value (or a `conflicts` check) removes it.
7. **Planner candidates.** Guard table x profile, evaluated with `Facts::assuming` on the
   normalised graph, so the planner moves from `plan_canonical_with_budget` to between optimisation
   rounds. Replaces `entry_checks`, typed receivers, bounds, requirements and identities;
   `reads::plan`, `shapes::plan`, `observations::plan` and `conditional_plan` fold in.
8. **Versioning as a transform.** `apply_version` reduces to duplication plus SSA repair, driven by
   planner specialisation requests; regions are recorded from placement.
9. **Late failure lowering.** Continuation wiring, cache priming and previous-entry fallback move out
   of `apply_version` into a pass after optimisation.
10. **Cost model and failure feedback.** Guard failure counts in the profile; loop-weighted
    evaluation; IC versus guard decided by cost.

Steps 1–3 are refactors with no intended behaviour change. Steps 5–9 change what gets optimised and
need before/after comparisons of notes and timings, not just hashes.

## Open questions

- **Resume state through transforms.** Late failure lowering needs the canonical mapping
  (`Canonical` blocks/values, `continuation::Pieces`) to stay valid across the passes between
  guard insertion and lowering. Expected to be mostly available already; needs checking.
- **No continuation available.** Handlers, generators and some frame states can't resume. Guards
  there either can't be speculative or need an inline fallback.
- **Compile time.** Handled by a budget: rounds and passes can be skipped when it runs out.
  `Facts::compute` (about 45 calls per compile today) can be cached later if it shows up.
- **Recursive inlining.** Inlining should eventually take part in the plan/optimise iteration too
  (inline, optimise, discover new call targets, inline again), which needs per-callee caching of
  optimised bodies.
- **Cost model units.** Loop weight, guard cost, failure rate and the size of what a guard unlocks
  need a common scale. Start crude (loop depth, counts) and make it visible in `--planner` notes.
- **Shape identities.** Shapes are runtime objects; whether a shape fact is a value fact on the
  object (permanent until a write changes the shape) or needs location treatment.
- **Observation placement.** Today `Observe`/`ReadHint` are emitted at build time. Moving
  observation to the planner means build stops guessing, but the first (unoptimised) compile still
  needs observation points.
