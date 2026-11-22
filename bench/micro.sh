#!/bin/sh
set -e
cd "$(dirname "$0")/.."
runner=target/release/js-runner
cargo build --release -p js-runner --quiet

jsc=""
if command -v jsc >/dev/null 2>&1; then
  jsc=$(command -v jsc)
elif [ -x /System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc ]; then
  jsc=/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc
fi

suites=${*:-"properties globals"}

for suite in $suites; do
  file="bench/$suite.js"
  echo "=== $suite (ns/op, lower is better) ==="
  jit_out=$("$runner" "$file" 2>&1 | grep ': .* ns/op$')
  node_out=""
  if command -v node >/dev/null 2>&1; then
    node_out=$(node "$file" 2>&1 | grep ': .* ns/op$')
  fi
  jsc_out=""
  if [ -n "$jsc" ]; then
    jsc_out=$("$jsc" "$file" 2>&1 | grep ': .* ns/op$')
  fi

  printf '%-20s %10s %10s %10s %9s %9s\n' "benchmark" "jit" "node" "jsc" "jit/node" "jit/jsc"
  echo "$jit_out" | while IFS= read -r line; do
    name=${line%%:*}
    j_val=$(printf '%s\n' "$line" | awk '{print $2}')
    n_val=$(printf '%s\n' "$node_out" | awk -v k="$name:" '$1 == k {print $2}')
    c_val=$(printf '%s\n' "$jsc_out" | awk -v k="$name:" '$1 == k {print $2}')
    r_node=$(LC_ALL=C awk -v a="$j_val" -v b="$n_val" 'BEGIN { if (b+0 > 0) printf "%.2fx", a/b; else print "-" }')
    r_jsc=$(LC_ALL=C awk -v a="$j_val" -v b="$c_val" 'BEGIN { if (b+0 > 0) printf "%.2fx", a/b; else print "-" }')
    printf '%-20s %10s %10s %10s %9s %9s\n' "$name" "$j_val" "${n_val:--}" "${c_val:--}" "$r_node" "$r_jsc"
  done
  echo
done
