# Benchmarks

Run commands from `cranelift-exp/` unless noted otherwise.

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

Apple Silicon, 2026-11-16; node 23.9.0, bun 1.3.1, SpiderMonkey 128,
QuickJS 2026-06-04, Ant 16.5.c41757b4.0. Higher is better.

| benchmark | jit | node jitless | node ≤Sparkplug | node ≤Maglev | node | bun | Ant | SpiderMonkey 128 | QuickJS |
|-----------|----:|-------------:|----------------:|-------------:|-----:|----:|----:|----------------:|--------:|
| richards | 4162 | 1753 | 2835 | 33057 | 44911 | 51315 | 5454 | 15621 | 1139 |
| navier-stokes | 24462 | 2370 | 3379 | 27788 | 40392 | 34280 | 20755 | 21390 | 1783 |
| gaussian-blur | 289 | 12.9 | 19.9 | 229 | 648 | 1286 | 153 | 260 | 7.93 |

All scores are medians of three runs, with benchmark processes run sequentially.
The standard harness warms up each benchmark for one second before measurement.
All engines produced a Gaussian blur output buffer sum of `2019780`.

Unsupported benchmarks:

| benchmark | first error |
|-----------|-------------|
| deltablue | `cannot assign property "inheritsFrom" of undefined` |
| crypto | `cannot read property "charCodeAt" of string` |
| raytrace | `value is not callable in this engine` |
| earley-boyer | `unsupported unary operator in unary expression` |
| regexp, splay | ``unknown variable `String` `` |

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
cargo run --release --manifest-path cranelift-exp/Cargo.toml -- --map \
  binaries/data/mods/public/maps/random/belgian_uplands.js
```

The C++ runners in `cpp/` (SpiderMonkey 128, as embedded by the game) and
`cpp-quickjs/` (QuickJS) run the same script. Build them with `make`, then from
`cranelift-exp/`:

```sh
cpp/build/mapgen-runner --bootstrap bootstrap.js --data ../binaries/data/mods/public \
  ../binaries/data/mods/public/maps/random/belgian_uplands.js
```

`--bootstrap` is required when not running from the repository root.
All engines must report the same `heightHash` (`dd1de906` for belgian_uplands).

| engine          | belgian_uplands |
|-----------------|-----------------|
| SpiderMonkey 128 | 330 ms         |
| jit             | 829 ms         |
| Ant             | 2195 ms        |
| QuickJS         | 7537 ms         |

JIT and Ant times are medians of three sequential runs; C++ runners were run once each.
Ant uses the same bootstrap and settings through JavaScript file-loading and
seeded-random host adapters. All runs produced the expected `heightHash`.
