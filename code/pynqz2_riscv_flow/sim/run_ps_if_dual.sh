#!/usr/bin/env bash
# run_ps_if_dual.sh -- run sim/tb_ps_if_dual.v (iverilog + vvp).
# The lockstep regression needs the original single-core ps_if.v; it is taken
# from git (REF_REV, default ed0d724 = the last single-core version) and
# renamed to ps_if_ref. Its 'rstate' declaration is moved above its first use
# (Vivado accepts use-before-declaration, iverilog does not), and its syscon
# VERSION line (reg 7) is replaced by the current one, since the ABI bits grew
# after REF_REV (neuron engine, KX); nothing else changes.
#   ./run_ps_if_dual.sh            (from anywhere)
set -euo pipefail
cd "$(dirname "$0")"
REF_REV="${REF_REV:-ed0d724}"
OUT="${TMPDIR:-/tmp}/ps_if_dual.$$"
mkdir -p "$OUT"
trap 'rm -rf "$OUT"' EXIT

VERSION_LINE="$(tr -d '\r' < ../rtl/ps_if.v | grep "^ *5'd7: ")"
git show "$REF_REV:code/pynqz2_riscv_flow/rtl/ps_if.v" | tr -d '\r' | awk -v ver="$VERSION_LINE" '
    /^module ps_if / { sub(/^module ps_if /, "module ps_if_ref ") }
    /^    reg \[1:0\] rstate;/ { next }
    /^ *5.d7: / { print ver; next }
    { print }
    /^    reg        aw_got, w_got, b_wait;$/ { print "    reg [1:0] rstate;"; print "    localparam KX = 1;" }
' > "$OUT/ps_if_ref.v"
grep -q "^module ps_if_ref " "$OUT/ps_if_ref.v" || { echo "could not build ps_if_ref from $REF_REV"; exit 1; }
[ "$(grep -c 'reg \[1:0\] rstate;' "$OUT/ps_if_ref.v")" = 1 ] || { echo "rstate move failed"; exit 1; }
[ "$(grep -c "5'd7: " "$OUT/ps_if_ref.v")" = 1 ] || { echo "VERSION line replace failed"; exit 1; }

iverilog -g2012 -Wall -Wno-timescale -o "$OUT/tb.vvp" \
    tb_ps_if_dual.v ../rtl/ps_if.v "$OUT/ps_if_ref.v"
vvp -n "$OUT/tb.vvp" | tee "$OUT/log.txt"
grep -q "^RESULT: PASS" "$OUT/log.txt"
