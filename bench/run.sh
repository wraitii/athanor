#!/bin/sh
cd "$(dirname "$0")/.."
runner=target/release/js-runner
cargo build --release -p js-runner --quiet || exit 1
benchmarks=${*:-"navier-stokes richards"}
for name in $benchmarks; do
	printf '%-14s' "$name"
	printf 'jit %-8s' "$(perl -e 'alarm 180; exec @ARGV' $runner --load bench/v8-v7/base.js --load bench/v8-v7/$name.js --load bench/v8-v7/driver.js --eval 'function main(){}' 2>&1 | sed -n 's/^Score: //p;s/.*rror.*/ERR/p' | head -1)"
	for engine in node bun; do
		command -v $engine >/dev/null || continue
		printf '%s %-8s' $engine "$($engine bench/v8-v7/shell.cjs $name 2>&1 | sed -n 's/^Score: //p' | head -1)"
	done
	echo
done
