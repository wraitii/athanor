Vendored QuickJS 2026-06-04.

Source: https://bellard.org/quickjs/quickjs-2026-06-04.tar.xz

Only the core library sources needed by the embedding runner are included.
QuickJS is distributed under the MIT license; see LICENSE.

Local patch: the upstream `OPCODE_ASM_LABEL` debug option now prefixes its
inline-assembly labels with an underscore on macOS so they survive linking and
can be used for perf/samply attribution.
