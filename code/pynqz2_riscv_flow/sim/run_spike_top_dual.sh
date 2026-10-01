#!/usr/bin/env bash
# run_spike_top_dual.sh -- system simulation of the two-core spike_top running
# firmware/spike.bin on both cores (sim/tb_spike_top_dual.v, iverilog + vvp).
# Also the elaboration check of the whole rtl/ hierarchy with two spike_soc.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
OUT="${TMPDIR:-/tmp}/spike_top_dual.$$"
mkdir -p "$OUT"
trap 'rm -rf "$OUT"' EXIT

python3 - "$HERE/../firmware/spike.bin" "$OUT/spike_bin.hex" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
data += b"\0" * (-len(data) % 4)
with open(sys.argv[2], "w") as f:
    for (w,) in struct.iter_unpack("<I", data):
        f.write("%08x\n" % w)
    for _ in range(len(data) // 4, 16384):      # pad to the 64 KB image area
        f.write("00000000\n")
PY

iverilog -g2012 -Wno-timescale -s tb_spike_top_dual -o "$OUT/tb.vvp" \
    tb_spike_top_dual.v ../rtl/spike_top.v ../rtl/ps_if.v ../rtl/spike_soc.v \
    ../rtl/picorv32.v ../rtl/poisson.v ../rtl/snn_layer.v
(cd "$OUT" && vvp -n tb.vvp) | tee "$OUT/log.txt"
grep -q "^RESULT: PASS" "$OUT/log.txt"
