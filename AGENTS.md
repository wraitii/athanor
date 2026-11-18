# Athanor contributor guide

This folder is the Athanor Cargo workspace, an experimental JS JIT engine for
fun and a possible “Beating* V8 in two weeks” blog post, with a substantial asterisk.
The user is wraitii, a long-term 0 A.D. maintainer. Work from this folder by default.
It lives at `athanor/` inside a 0 A.D. checkout, as a separate Git checkout
of `https://github.com/wraitii/athanor`. Commit and push engine changes from this
folder to Athanor, rather than to the surrounding 0 A.D. repository. This layout
is required for map scripts, assets, and test fixtures; building the game is not
required for Rust engine changes.

## Coding practices

- No comments - the code should be simple and readable on its own. Write all code without comments in the first pass. Only add comments as a second pass for clarification where necessary, including in particular unexpected or tricky behaviour.
- I repeat the above - no comments on the initial code writing pass, only add spare comments in a second pass for clarification where necessary.
- Maintain house-style but keep things clean.
- On the C++ side, be especially careful about memory.
- On the JS side, be especially careful about typing - this is a raw JS codebase using JSDOC style annotations.
- Performance matters heavily, but remains generally second to simplicity.
- Prefer early returns. Prefer clean flow.
- Avoid unrelated cleanups.
- The codebase is relatively untested and proper testing is difficult. Avoid overly-specific unit tests. Tests can be written to validate implementation/fixes and then not staged when committing if keeping them around does not seem useful.

## Committing guidelines

- Separate subject from body with a blank line
- Limit the subject line to 50 characters where possible
- Capitalize the subject line
- Do not end the subject line with a period
- Use the imperative mood in the subject line
- Wrap the body at 72 characters
- Use the body to explain what and why vs. how
- Write atomic, logically separated commits
- Use imperative, present tense in commit summaries
- Link relevant issues with Refs: #1234 or Fixes: #1234
- (personal) I like to mention the component/code part in the commit message sometimes, such as 'UnitAI: ...'

**Example**

Add workaround to turn off nursery size heuristic

SpiderMonkey 98 introduced a size heuristic for the nursery GC region
(https://phabricator.services.mozilla.com/D136637). As this heuristic
uses a wall-clock time duration, it results in a severe performance
regression on slower systems for our use case.

This commit adds a workaround to turn off that heuristic, by telling
SpiderMonkey that a "page load" (something which doesn't have a meaning
in the context of pyrogenesis) is in progress, as that heuristic is
disabled for page loads.

Fixes: #7714

---

## Orientation

Read `README.md` for commands, then `crates/engine/DESIGN.md` for architecture,
supported semantics, compiler controls and remaining design directions.

The workspace contains `engine` and `runner` (`js-runner`). Source paths below
are relative to `crates/engine/src/` unless stated otherwise.

1. Public API: `lib.rs`, `engine.rs`, and `parse.rs`.
2. Values and heap: `value.rs`, `heap/mod.rs`, `heap/object.rs`,
   `heap/shape.rs`, then the remaining heap types and property implementations.
3. Compilation: `jit/compiler.rs`, `jit/semantic/build.rs`,
   `jit/semantic/mod.rs`, `jit/semantic/pipeline.rs`, and the semantic
   optimization, specialization and transform modules.
4. Lowering: `jit/finalize/`, `jit/semantic/runtime/`, and
   `jit/semantic/ownership_lowering.rs`. The shared graph feeds
   `jit/cranelift/` or `jit/aarch64/`; the latter owns selection and allocation.
5. Runtime: `jit/session.rs`, `jit/helpers/`, and `builtins/`.
6. Inspection: `jit/jitdump.rs`, `jit/semantic/listing.rs`, and
   `crates/runner/src/main.rs`.
7. Behavioral specs: `crates/engine/tests/`, `heap/tests.rs`, and
   `crates/runner/tests/`.

## Validation

From this folder:

```sh
cargo test --workspace
cargo run -p js-runner -- --compile --eval 'function f(a,b){return a*b+1;}'
cargo run --release -p js-runner -- --map \
  --data ../binaries/data/mods/public \
  ../binaries/data/mods/public/maps/random/belgian_uplands.js
```

Use `JS_JIT_OPT=none` to compare semantic optimization and `--after PASS`
to inspect changes. Use release builds for
performance comparisons. For compiler changes, run the Rust tests; for changes
that affect the map workload, compare its output hash as well as timing.

Check documented subset restrictions before treating unsupported JS as a bug.
Keep current behavior distinct from design intent. Heap handles have no generation
or runtime identity; stale handles remain an embedding safety caveat.
