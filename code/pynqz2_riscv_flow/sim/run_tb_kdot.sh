#!/bin/bash
# Runs the kdot_pcpi unit test (tb_kdot.sv) in Vivado's simulator (xvlog/xelab/xsim).
# Usage: bash run_tb_kdot.sh [vivado-bin-folder]   (default: Vivado 2026.1 on this PC)
# Simulator output goes to a temporary work folder; the last line is PASS or FAIL.
set -euo pipefail
BIN="${1:-/c/AMDDesignTools/2026.1/Vivado/bin}"
SIM="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
cp "$SIM/tb_kdot.sv" "$SIM/../rtl/kdot_pcpi.v" "$WORK/"
cd "$WORK"
"$BIN/xvlog.bat" --sv tb_kdot.sv kdot_pcpi.v > xvlog.log || { cat xvlog.log; exit 1; }
"$BIN/xelab.bat" tb_kdot -s tb_kdot > xelab.log || { cat xelab.log; exit 1; }
"$BIN/xsim.bat" tb_kdot -runall > xsim.log || true
grep -E "ERROR|ksdot with|PASS|FAIL" xsim.log
grep -q "^PASS" xsim.log
