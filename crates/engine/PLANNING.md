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
  assumption placement and specialisation, not planned up front.
- **The planner proposes, specialisation commits.** The planner records "we would win X if Y held
  here" as inert `Assume` ops. Nothing reads them except placement and specialisation, so they can
  be moved, merged or dropped freely. Only specialisation turns them into real checks, in a copy,
  and only real checks add facts. The planner never changes what code does.
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
3. plan             planner (semantic/planner/)
                   - candidates from the guard table + profile
                   - evaluation on the normalised graph: Facts::assuming(candidate) vs Facts,
                     loop-weighted, minus check cost and failure rate
                   - record chosen candidates as inert Assume(v, fact) ops at their use sites,
                     with estimated gain and provenance (profile site, table row)
                   - observation injection where feedback is missing
                      |
4. place           hoist each Assume as high as it stays valid (operand available; for
                   probe-based facts, no conflicting write in between); merge duplicates and
                   subsumed ones (Fact::implies), summing gains where they collect
                      |
5. specialise      where collected gain beats the cost (copy size, check cost, failure rate):
                   duplicate from that point, turn its Assumes into real entry checks in the
                   copy, rerun 2 on the copy. Leftover Assumes are either specialised in place
                   with inline recovery (an IC: today's reads::apply, shapes::apply, push calls)
                   or dropped
                      |                         (3-5 may iterate under a budget)
6. failure mode    per check in a copy: continuation to the generic copy, or inline recovery
                   (slow path inline, merge back; the region ends at the merge, which is an IC)
                      |
emit               emit/
```

The line between lowering and planning is what justifies a rewrite: facts already proven (no
fallback) is lowering; a check with a fallback is speculation. Some JS lowering passes (`caches::apply`,
`arrays::push_calls`, `reads::apply`) are a local planner decision plus IC specialisation in one step;
that is fine while the choice is obvious, and they move onto `Assume` only when a cost model should be
able to refuse them or their checks should hoist and merge. `observations::plan` is likewise a planner
phase ("find out"); others of that shape fit the same slot.

Ownership: steps 1, 3 and 5 are the planner (runtime-informed decisions); step 2 is
`semantic/transforms/` and `semantic/js/`; step 4 is generic placement over `Assume` ops; step 6 is
runtime lowering. Nothing depends on an `Assume`, so dropping one is always sound; code only changes
behind a real check. For now specialisation always uses continuations, as versioning does today;
inline recovery comes with the cost model.

`Assume` ops have no effects and their operand is not a counted use (DCE, `trim`), but uses are
rewritten by `replace_uses` and SSA repair, and removing the defining inst removes its `Assume`s.
Location facts sit on a probe value, so they stay value facts and placement keeps them valid with
`conflicts`. OSR copies them. `ReadHint` stays separate: it is def-site feedback that must survive
optimisation and can say "unknown, don't speculate", which an `Assume` (a use-site proposal) cannot.

The recompile gate in `compiler.rs` compares plans to decide whether feedback changed, so planning
has to run before it and produce a stable signature (chosen `Assume`s keyed by site, not by
`Inst`/`Value`). The plan step therefore stays cheap and early in practice: inline, normalise, plan,
then the gate, then place onward.

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
  `Writes::clobbers`, `closure_failure` and the available-loads dataflow use it.
- `Facts::removable(inst)` drives DCE; `graph.views` entries are real uses.

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
plan         candidates = guard table x profile   (replaces observations/reads/shapes/conditional plans)
             evaluate   = Facts::assuming(candidate) diff on the normalised graph
             record     chosen candidates as inert Assume ops at their use sites
             observe    ops with table entries but no feedback get Observe
place        hoist / merge Assumes (replaces entry_checks, invariant_guards, probe_reads, covers/merge)
specialise   duplicate where collected gain pays off; Assumes become entry checks in the copy;
             rerun normalise on the copy; drop leftover Assumes
             (plan .. specialise may iterate under a budget)
lower        per check failure: continuation (today) / inline recovery (later)
```

The current `plan_canonical_with_budget` / `optimize_planned` split no longer matches: only inlining
stays before optimisation. `SpecializationPlan` stops being a precomputed bundle handed to
`optimize_planned`; the planner runs between optimisation rounds instead.

## Where today's code goes

| Today | Concern | Target |
|---|---|---|
| `planning.rs` `invariant_guards` | hoisting existing guards | `Assume` placement |
| `analysis.rs` `probe_reads` | hoisting loads | `Assume` placement on probes |
| `analysis.rs` `closure_failure` | legality | generic conflicts check |
| `planning.rs` `entry_checks`, typed receivers, bounds | guards from feedback | planner candidates → `Assume` |
| `analysis.rs` `requirements`, conditional identities | guards from summaries | planner candidates → `Assume` |
| `planning.rs` `covers` / `merge` | guard subsumption | `Assume` merging (`Fact::implies`) |
| `versioning.rs` block copy, SSA repair | duplication | generic versioning transform |
| `versioning.rs` entry chain | emitting guards at entry | `Assume`s at the copy point become checks |
| `versioning.rs` `TypedIndex` / `CallNative` / `AsNumber` rewrites | ops under guards | JS lowering from facts |
| `versioning.rs` continuations, cache priming, previous-entry fallback | failure handling | late failure lowering |
| `GuardValid` / `GuardCommit` cells | cached guard validation | kept; a check op with a cheap fast path |
| `specialize.rs` summaries | callee effects | facts identities + summaries on the graph |
| `graph.closed_calls` | per-call effect override | removed (identity facts) |
| `jit/planner.rs` | runtime feedback | `jit/profile.rs` |

## Migration

Each step should keep tests green and the belgian_uplands hashes unchanged, and is checked with
`--compile --planner` notes on small loops.

Status: steps 1-3 are done. Step 5 is partly done: side-table uses and `removable` DCE, flow-sensitive
aliasing (`Facts::fresh`, replacing `unescaped`; step 4 is folded in here), whole-function
`reuse_loads` as an available-loads dataflow, GVN of value checks, and folding of branches decided
by a dominating edge. Whole-function `hoist` is deferred to step 8: it moves code across blocks,
which needs resume state to survive the transforms. Check hoisting with its branch is no longer
needed (placement moves inert `Assume`s instead, step 7).
Step 6a is done: subsumption is `Fact::implies` (used by `covers` and the entry-guard dedupe),
`closure_failure` reads `Access` conflicts and `fresh.at` over a guard list, and the name-keyed global
identity rule is gone because entry probes reuse the `IsCode`-checked probe. `guarded_global` no
longer trusts region legality: the available-loads dataflow (`semantic/available.rs`, shared with
`reuse_loads`) must still hold the guarded probe at the call, with summarised calls decided
optimistically and re-checked until stable.

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
6. **Guard placement.** Split in two, because a guard can only be placed where a failure target
   exists, and today only region entries have one.
   - **6a (refactor).** Placement points stay region entries. `invariant_guards`, `probe_reads` and
     `covers`/`merge` are rebuilt on the generic pieces: invariance and legality from `Access`
     conflicts and `fresh.at` instead of `closure_failure`'s scoped walk, implied facts from
     `guards::implies`, duplicates from check GVN, subsumption (`IsInteger` covers `IsNumber`) as a
     table rule. Conditional summaries (callee bodies that read globals) apply when a dominating
     `IsCode` guards a `GlobalRead` of each required name; after step 2 this trusts region legality
     for "no write in between", and gives any `GlobalRead`/`ProbeGlobal(name)` the identity of a
     dominating guarded probe of that name, because `apply_version` clones callees as fresh reads.
     Rewriting in-region callees to the guarded value and checking `conflicts` removes both.
   - **6b.** Absorbed into step 7: placement works on inert `Assume` ops, so real checks never need
     to move with their branch.
7. **Planner candidates and `Assume`.** Guard table x profile, evaluated with `Facts::assuming` on
   the normalised graph, recorded as inert `Assume` ops; placement hoists and merges them with the
   6a machinery. The planner moves from `plan_canonical_with_budget` to after normalisation.
   Today's versioning consumes the `Assume`s that reach a region entry instead of `entry_checks`,
   `invariant_guards`, typed receivers, bounds, requirements and identities; `reads::plan`,
   `shapes::plan`, `observations::plan` and `conditional_plan` fold in. Output should stay close
   to today's.
8. **Versioning as a transform.** `apply_version` reduces to "duplicate from an `Assume` collection
   point, turn those `Assume`s into entry checks, repair SSA", then reruns normalisation on the
   copy. The gain-versus-cost decision is a hook that is ready for a cost model but not
   implemented: it keeps today's choices.
9. **Failure mode per check.** Continuation wiring, cache priming and previous-entry fallback move
   out of `apply_version` into a pass after optimisation. Continuations stay the only mode for
   now; inline recovery (an IC-sized region) is added with the cost model.
10. **Cost model and failure feedback.** Guard failure counts in the profile; loop-weighted
    evaluation; specialise-or-drop and continuation-or-inline decided by cost. This doubles as
    tiering: with fresh feedback every `Assume` is an in-place IC (cheap, cannot deopt); once its
    IC has been stable it may hoist and specialise a region with continuations; a region guard that
    keeps failing goes back to an IC or is dropped. Tier changes are plan-signature changes, so the
    existing recompile gate drives them.

Steps 1–3 and 6a are refactors with no intended behaviour change. Steps 5–9 change what gets optimised and
need before/after comparisons of notes and timings, not just hashes.

## Open questions

- **Resume state through transforms.** Continuations need the canonical mapping (`Canonical`
  blocks/values, `continuation::Pieces`) to stay valid across the passes that run on a copy after
  specialisation. Expected to be mostly available already, since today's versioning optimises
  copies; needs checking once normalisation reruns on them.
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
- **Observation placement.** `Observe`/`ReadHint` are injected by `observations::plan` after
  inlining; the first (unoptimised) compile still needs observation points.
- **Planning under assumptions.** Evaluating candidates with `Facts::assuming(already chosen
  Assumes)` would find chains that only pay off together (a number check that enables a typed
  read that enables a bound). Theoretical for now; the evaluation API already allows it.
- **Contradictory assumptions.** Because `Assume`s are inert, two that cannot both hold (`v` is an
  integer here, a string there) can coexist. Where they collect at one point, specialisation should
  pick a consistent subset or split into one copy per subset (polymorphic dispatch), never emit both
  as entry checks. Not reachable today (one hint per value), but materialisation should detect it.
- **Speculative CFG transforms as assumptions.** Some transforms could be proposals too: LICM of a
  load past a write that probably does not alias, "this loop runs at least once", a branch that is
  never taken in the profile. Each is an `Assume` whose materialisation adds a check.
- **Identities beyond code.** Only `Identity::Code` exists; native and shape identities are added
  when a contract reads them. `plan_region` still clones the graph to rewrite `Call` to
  `CallNative` for the requirements analysis; that goes with native `Assume`s (step 7e).
