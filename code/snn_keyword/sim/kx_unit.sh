#!/usr/bin/env bash
# kx_unit.sh -- instruction-level test of rtl/kdot_pcpi.v (sim/tb_kx.sv, sim/kx_vectors.py).
#   sim/kx_unit.sh [seed ...]          KX=1 and KX=0 builds of this tree's kdot_pcpi.v
#   RTL=<dir> NO_KX_PARAM=1 KX=0 sim/kx_unit.sh 1   legacy-only stream against another RTL
set -euo pipefail
cd "$(dirname "$0")/.."
RTL="${RTL:-../pynqz2_riscv_flow/rtl}"
PY="${PYTHON:-python}"
SEEDS="${*:-1 2 3}"
KXS="${KX:-1 0}"
fail=0
for kx in $KXS; do
  for seed in $SEEDS; do
    d="build/kx_unit/kx${kx}_s${seed}"
    mkdir -p "$d"
    "$PY" sim/kx_vectors.py "$d" --seed "$seed" --kx "$kx" > "$d/gen.log"
    defs=""; [ -n "${NO_KX_PARAM:-}" ] && defs="-DNO_KX_PARAM"
    iverilog -g2012 $defs -I "$d" -s tb_kx -o "$d/tb" sim/tb_kx.sv "$RTL/kdot_pcpi.v"
    out=$(cd "$d" && vvp -n tb)
    echo "KX=$kx seed=$seed: $(cat "$d/gen.log" | sed 's/ -> .*//')"
    echo "$out" | grep -E "^(BAD|instructions|RESULT)" | sed 's/^/    /'
    echo "$out" | grep -q "^RESULT: PASS" || fail=1
  done
done
[ $fail = 0 ] && echo "KX UNIT: ALL PASS" || { echo "KX UNIT: FAIL"; exit 1; }
