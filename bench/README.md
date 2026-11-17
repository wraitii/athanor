# Benchmarks

Run commands from `athanor/` unless noted otherwise.

## V8 benchmark suite v7

`v8-v7/` holds the V8 Benchmark Suite version 7 (the predecessor of Octane):
`base.js` (harness), one file per benchmark, `driver.js` (prints `Score:` via
`print`) and `shell.cjs` (loads the same files in node/bun through `vm`).
Scores are relative to the reference machine; higher is better.

```sh
bench/run.sh                       # navier-stokes richards
bench/run.sh richards deltablue    # any subset
```

`run.sh` builds `js-runner` in release mode, runs each benchmark with a 180 s
timeout and prints the JIT score next to `node` and `bun` when they are on
`PATH`. The JIT invocation is:

```sh
target/release/js-runner --load bench/v8-v7/base.js --load bench/v8-v7/NAME.js \
  --load bench/v8-v7/driver.js --eval 'function main(){}'
```

To isolate V8 tiers, run node directly:

```sh
node --jitless   bench/v8-v7/shell.cjs richards   # Ignition, no JIT at all
node --max-opt=1 bench/v8-v7/shell.cjs richards   # + Sparkplug (baseline)
node --max-opt=2 bench/v8-v7/shell.cjs richards   # + Maglev
node --max-opt=3 bench/v8-v7/shell.cjs richards   # + TurboFan (default)
```

To run QuickJS through the C++ runner:

```sh
make -C cpp-quickjs
cpp-quickjs/build/mapgen-runner --bootstrap bench/v8-v7/quickjs.js bench/v8-v7/richards.js
cpp-quickjs/build/mapgen-runner --bootstrap bench/v8-v7/quickjs.js bench/v8-v7/navier-stokes.js
```

### Ant

Use the official macOS ARM64 release binary; its JIT is enabled by default.
The results below use [Ant 16.5.c41757b4.0](https://github.com/theMackabu/ant/releases/tag/v16.5.c41757b4.0).
Combine the existing harness, benchmark, and driver into one entry script:

```sh
{ printf 'function print(text) { console.log(text); }\n';
  cat bench/v8-v7/base.js bench/v8-v7/navier-stokes.js bench/v8-v7/driver.js;
} > /tmp/ant-navier-stokes.js
/path/to/ant /tmp/ant-navier-stokes.js
```

Substitute the benchmark path for Richards or Gaussian blur.

### Gaussian blur

`gaussian-blur.js` benchmarks a 3×3 Gaussian convolution over the interior of
an ordinary 128×128 grayscale array. The image, kernel, and output are passed
as arguments; the output buffer is reused and border pixels remain unchanged.
The reference value is arbitrary, so compare scores only across engines running
this benchmark.

```sh
target/release/js-runner --load bench/v8-v7/base.js --load bench/gaussian-blur.js \
  --load bench/v8-v7/driver.js --eval 'function main(){}'
node bench/v8-v7/shell.cjs ../gaussian-blur
cpp-quickjs/build/mapgen-runner --bootstrap bench/v8-v7/quickjs.js bench/gaussian-blur.js
```

### Results

Apple Silicon; node 23.9.0, bun 1.3.1, SpiderMonkey 128,
QuickJS 2026-06-04, Ant 16.5.c41757b4.0. Higher is better.
JIT, default Node and Bun results were measured on 2026-11-17, with Athanor
at `56d6718`. Other results are retained from 2026-11-16.

| benchmark | jit | node jitless | node ≤Sparkplug | node ≤Maglev | node | bun | Ant | SpiderMonkey 128 | QuickJS |
|-----------|----:|-------------:|----------------:|-------------:|-----:|----:|----:|----------------:|--------:|
| richards | 4204 | 1753 | 2835 | 33057 | 44947 | 51484 | 5454 | 15621 | 1139 |
| navier-stokes | 23869 | 2370 | 3379 | 27788 | 40096 | 33700 | 20755 | 21390 | 1783 |
| gaussian-blur | 289 | 12.9 | 19.9 | 229 | 645 | 1282 | 153 | 260 | 7.93 |
| deltablue | 3253 | 1849 | 3124 | 47307 | 118384 | 77133 | — | 17515 | 1291 |

All scores are medians of three runs, with benchmark processes run sequentially.
The standard harness warms up each benchmark for one second before measurement.
All engines produced a Gaussian blur output buffer sum of `2019780`.
DeltaBlue was measured separately on 2026-11-17 with the `Function.prototype.call`
change; Ant was unavailable for that run.

Unsupported benchmarks:

| benchmark | first error |
|-----------|-------------|
| earley-boyer | `unsupported unary operator in unary expression` |

Additional scores on the same machine, medians of three sequential runs per
engine and benchmark, using the same dates and release JIT build as above.
Node and Bun use their default configurations; QuickJS uses the C++ runner and
the same benchmark harness:

| benchmark | jit | node | bun | QuickJS |
|-----------|----:|-----:|----:|--------:|
| splay | 7774 | 20715 | 72421 | 5827 |
| raytrace | 5119 | 105226 | 201055 | 2905 |
| crypto | 8679 | 63546 | 76789 | 979 |

All three benchmarks completed successfully in every engine.

`Function.prototype.apply` is implemented in JS with a compiler-visible target
call; string conversion, character/range methods and `parseInt` use Rust
implementations.

### Compatibility audit, 2026-11-17

Release build `c1a59fa`, default compiler settings, one sequential run per
benchmark. These scores are smoke checks, not new performance medians:

| benchmark | result |
|-----------|-------:|
| richards | 4413 |
| navier-stokes | 24758 |
| splay | 4079 |
| raytrace | 5624 |
| crypto | 7249 |
| regexp | 5146 |
| gaussian-blur | 293 |

DeltaBlue still fails because `Object` is a namespace without a `prototype`.
Temporary copies that redirect `Object.prototype.inheritsFrom` to
`Function.prototype.inheritsFrom` and replace `new Object()` with `{}` expose
another missing builtin: `Function.prototype.call`. Adding a temporary JS
`call` wrapper around `apply` lets the modified benchmark complete its
correctness checks and score 2900. This is a diagnostic result, not a passing
run of the original benchmark or an implementation of Object semantics.

Earley–Boyer first rejects `delete ht[key]` in `sc_rempropBang` at line 131.
Replacing `delete` with `void` in a temporary copy, solely to probe compilation,
exposes unsupported `finally` at line 1719. Both language features need work;
runtime compatibility beyond these compile failures remains unchecked.

Map probes use size 128, seed 0 and two players (`athen`, `brit`), with the same
shared bootstrap in Athanor and SpiderMonkey. Belgian Uplands completes with
identical map summaries, including all three hashes and 1137 entities:
`heightHash: dd1de906`, `tileIndexHash: 3688e6a0`,
`tilePriorityHash: 3688e6a0`. Generation takes 773 ms in Athanor versus 336 ms
in SpiderMonkey in this single run.

Several other maps first hit harness or settings gaps shared by both runners:
`Engine.FileExists` is missing, the default settings omit `Biome` and
`PlayerPlacement`, and the global `clone` helper is missing for Schwarzwald
and Caledonian Meadows. Ambush and Mediterranean use imports that the
bootstrap's script loader does not handle.

With a temporary bootstrap implementing `Engine.FileExists` through directory
listing, and settings `Biome: "generic/temperate"` and
`PlayerPlacement: "circle"`, the following engine boundaries become visible:

| maps | Athanor blocker | reference outcome |
|------|------------------|-------------------|
| mainland, islands, rivers, ardennes_forest, lake, cantabrian_highlands | Destructuring assignment in `rmbiome/generic/aegean.js` | SpiderMonkey completes |
| alpine_lakes | Same biome-library destructuring | SpiderMonkey also fails later with undefined terrain under these settings |
| oasis, northern_lights | Missing `Array.prototype.shift` in `SmoothElevationPainter` | SpiderMonkey completes |
| latium | Object-to-primitive coercion at `t == tGrass` (line 314) | SpiderMonkey completes |
| unknown | Object destructuring assignment (line 137) | Default reference run is blocked by missing `Engine.FileExists` |
| caledonian_meadows | Destructured callback parameter (line 599) | Reference run is blocked by missing `clone` |
| alpine_valley | Nested yield rejected | Reference probe also fails with undefined terrain |

The biome-library error occurs while loading the generic library, including
the uncalled Aegean setup function, even when selecting the temperate biome.
Oasis, Northern Lights and Alpine Valley were probed with the temporary
bootstrap and explicit biome, before adding the circle placement setting;
Latium, Unknown and Caledonian Meadows used the original bootstrap and defaults.
The diagnostic bootstrap and modified benchmark copies are not checked in.

### Object root and constructor check, 2026-11-17

The object root and native Object constructor now let unmodified DeltaBlue
initialize its object model. Its remaining failure is
`EqualityConstraint.superConstructor.call(...)`: `Function.prototype.call`
is still missing. Primitive boxing and specialized builtin prototype chains
remain outside this change.

All seven previously supported benchmarks complete on the release build.
A final paired run against `c1a59fa` gives object-churn scores of 3019 → 3121
and missing-property-read scores of 2491 → 2617. These are single paired
samples with the standard harness, using synthetic loops that publish each
allocated object to a global. They are not performance medians.

Allocation caches the default root shape, reuse preserves an unchanged prototype
reference, and generic misses skip the shared root while its live shape is
empty. Unknown property atoms skip shaped lookup while retaining dictionary
lookup. Earlier focused comparisons exposed the extra root traversal cost and
motivated these paths. Belgian Uplands still produces all three expected hashes
and 1137 entities. The workspace's 787 Rust tests pass.

### Allocator loop liveness comparison

Apple Silicon ARM64, 2026-11-17; release builds, baseline `5d919de` versus
allocator loop summaries. Benchmark processes run sequentially, alternating
before/after order. Compile times are medians of five process wall times using
`js-runner --compile --no-ir PATH`, including startup and eager compilation of
all functions. Scores use the standard harness and are medians of three runs,
except Splay, which uses nine because the first three were noisy.

| benchmark | compile before (ms) | compile after (ms) | time change | score before | score after | score change |
|-----------|--------------------:|-------------------:|------------:|-------------:|------------:|-------------:|
| richards | 81.4 | 81.2 | -0.3% | 4385 | 4367 | -0.4% |
| navier-stokes | 89.7 | 91.7 | +2.2% | 24189 | 24338 | +0.6% |
| splay | 76.0 | 75.9 | -0.1% | 3993 | 3903 | -2.3% |
| raytrace | 224.7 | 223.3 | -0.6% | 5675 | 5698 | +0.4% |
| crypto | 575.7 | 581.3 | +1.0% | 7233 | 7222 | -0.2% |
| regexp | 1400.2 | 1036.0 | -26.0% | 5237 | 5237 | +0.0% |
| gaussian-blur | 23.9 | 23.6 | -1.5% | 290 | 290 | +0.0% |

RegExp compilation improves by 26.0%; a second five-run comparison on the
final build gives 1418.2 → 1056.9 ms (25.5% faster). Other compile times and
runtime scores show no clear general improvement. Splay's initial three-run
score fell 9.1%, but the difference shrinks with six additional pairs of runs.

A synthetic loop with 96 values loaded before the loop and 96 conditional uses
inside it compiles in 101.9 → 97.3 ms (medians of five runs).

Belgian Uplands generation remains 758 → 758 ms (medians of three runs).
All six map runs produce `heightHash: dd1de906`,
`tileIndexHash: 3688e6a0`, and `tilePriorityHash: 3688e6a0`.

### DeltaBlue call support, 2026-11-17

`Function.prototype.call` lets the unmodified DeltaBlue benchmark complete its
correctness checks. Median scores from three sequential runs per engine are in
the results table above.

### Published results

Selected macos-arm64 results from
[ahaoboy/js-engine-benchmark](https://github.com/ahaoboy/js-engine-benchmark).
Hardware is unspecified; compare ratios to node rather than raw
scores across machines. Its node version is 27 alpha versus 23.9 locally.

| engine         | Richards | NavierStokes | Score |
|----------------|----------|--------------|-------|
| v8 15.9.118    | 38603    | 35913        | 50674 |
| JavaScriptCore | 49610    | 28909        | 54154 |
| node 27.0.0-alpha | 30543 | 24413        | 33703 |
| spidermonkey 147 | 9743   | 14589        | 14644 |
| ant            | 4978     | 15932        | 7132  |
| quanta 2.0.0-alpha | 3195 | 11097        | 3195  |
| quickjs 2026.06.04 | 1167 | 2180         | 1588  |
| quickjs-ng 0.18.0 | 1171  | 2637         | 1556  |
| hermes 0.12.0  | 759      | 1121         | 1322  |

## Map generation

Run random map generation from the repository root:

```sh
cargo run --release --manifest-path athanor/Cargo.toml -- --map \
  binaries/data/mods/public/maps/random/belgian_uplands.js
```

The C++ runners in `cpp/` (SpiderMonkey 128, as embedded by the game) and
`cpp-quickjs/` (QuickJS) run the same script. Build them with `make`, then from
`athanor/`:

```sh
cpp/build/mapgen-runner --bootstrap bootstrap.js --data ../binaries/data/mods/public \
  ../binaries/data/mods/public/maps/random/belgian_uplands.js
```

`--bootstrap` is required when not running from the repository root.
All engines must report the same `heightHash` (`dd1de906` for belgian_uplands).

| engine          | belgian_uplands |
|-----------------|-----------------|
| SpiderMonkey 128 | 330 ms         |
| jit             | 824 ms         |
| Ant             | 2195 ms        |
| QuickJS         | 7537 ms         |

JIT and Ant times are medians of three sequential runs; C++ runners were run once each.
The JIT result was measured on 2026-11-17 at `56d6718` (823, 824 and 828 ms);
other engine results are retained from 2026-11-16.
Ant uses the same bootstrap and settings through JavaScript file-loading and
seeded-random host adapters. All runs produced the expected `heightHash`.
