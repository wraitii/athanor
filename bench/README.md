# Benchmarks

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

### Results (2026-11-10, Apple Silicon, node 23.9.0, bun 1.3.1)

| benchmark     | jit  | node jitless | node ≤Sparkplug | node ≤Maglev | node  | bun    |
|---------------|------|--------------|-----------------|--------------|-------|--------|
| richards      | 3329 | 1753         | 1952            | 33717        | 46221 | 53147  |
| navier-stokes | 4773 | 2503         | 3620            | 28048        | 40662 | 34429  |

JIT scores are medians of three runs after adding inline packed-array stores:
richards 3340/3149/3329, navier-stokes 4706/4773/4801. Other engines and tiers
were run once. These are about 20% and 16% above the previously recorded JIT
scores (2780 and 4100), respectively; this is not an isolated before/after build
comparison.

- The JIT beats V8's interpreter and baseline tiers on both, and is roughly
  6–14x behind Maglev/TurboFan.
- The other benchmarks do not run yet (first error reported):
  - deltablue: `cannot assign property "inheritsFrom" of undefined`
  - crypto: `cannot read property "charCodeAt" of string`
  - raytrace: `value is not callable in this engine`
  - earley-boyer: `unsupported unary operator in unary expression`
  - regexp, splay: ``unknown variable `String` ``

### Published results

[ahaoboy/js-engine-benchmark](https://github.com/ahaoboy/js-engine-benchmark)
runs the same suite (forked from mozilla/arewefastyet `benchmarks/v8-v7`) on
~40 engines for ubuntu, macos-arm64 and windows, with results in
`macos-arm64.json`, `ubuntu.json` and `windows.json`. Hardware is not
published (presumably CI runners), so compare ratios to node, not raw scores.
Selected macos-arm64 rows as published (dated 11/10/2026):

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

Relative to node on the same machine, the JIT is at ~7% on richards and ~12%
on navier-stokes. On richards that places it above QuickJS (~4% of node) and
Hermes (~2.5%) and below quanta (~10%); on navier-stokes it is slightly above
quickjs-ng (~11%) and well below SpiderMonkey (~60%). Node versions differ (23.9 locally, 27 alpha
there), so this placement is approximate.

## Map generation

The real workload is random map generation. From the repository root:

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
| jit             | 979 ms         |
| QuickJS         | 7537 ms         |

Refreshed 2026-11-10. JIT time is the median of three runs (975/979/981 ms);
the C++ runners were run once each. All five runs produced `heightHash`
`dd1de906`.
