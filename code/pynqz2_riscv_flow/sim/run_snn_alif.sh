#!/usr/bin/env bash
# run_snn_alif.sh -- generate the golden_alif.py vector sets and run tb_snn_alif.v
# on each (iverilog + vvp).   ./run_snn_alif.sh [set ...]   (default: all sets)
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
OUT="${TMPDIR:-/tmp}/snn_alif.$$"
mkdir -p "$OUT"
trap 'rm -rf "$OUT"' EXIT
PY="$(command -v python3 || command -v python)"

"$PY" golden_alif.py regress --base "$OUT"
SETS="${*:-small fast edges spikes full keyword tie}"
fail=0
for s in $SETS; do
    d="$OUT/vec_$s"
    iverilog -g2012 -I "$d" -o "$d/tb.vvp" tb_snn_alif.v ../rtl/snn_layer.v
    r=$(cd "$d" && vvp -n tb.vvp | grep -E "^(TB:|MISMATCH|RESULT)")
    printf '=== %s\n%s\n' "$s" "$r"
    echo "$r" | grep -q "^RESULT: PASS" || fail=1
done
[ $fail = 0 ] && echo "ALL SETS PASS" || { echo "SOME SETS FAILED"; exit 1; }
