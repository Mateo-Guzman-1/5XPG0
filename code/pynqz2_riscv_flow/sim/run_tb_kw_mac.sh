#!/bin/bash
# run_tb_kw_mac.sh — compile and run the MAC unit test (tb_kw_mac.v) in xsim.
# Prints the per-neuron sums and PASS / FAIL. Work files go to sim/work/.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VIV="$HERE/../vivado/vivado2024.sh"
mkdir -p "$HERE/work" && cd "$HERE/work"

VIVADO_TOOL=xvlog "$VIV" --nolog "$HERE/../rtl/kw_mac_pcpi.v" "$HERE/tb_kw_mac.v" > /dev/null
VIVADO_TOOL=xelab "$VIV" --nolog --timescale 1ns/1ps -debug typical tb_kw_mac -s tb_kw_mac > /dev/null
VIVADO_TOOL=xsim  "$VIV" tb_kw_mac --nolog -R | grep -E "ERROR|neuron|PASS|FAIL"