#!/bin/bash
# snn_smoke_test.sh -- on-board check of the ALIF snn_layer (16 neurons per core).
# Runs ON THE PYNQ-Z2, in the directory holding spike_top.bit, snn_smoke.bin, spike_pynq.py.
#
# Usage: [CORE=1] ./snn_smoke_test.sh [bitstream] [test-firmware]
#        (defaults: core 0, spike_top.bit, snn_smoke.bin)
#
# PASS means: the fabric loads, the CPU of that core runs the test, and every neuron produced
# exactly the golden-model results (golden_alif.py: fire vector, membrane u and adaptation
# trace a of every neuron at every tick, final counters) for three sets -- the trained
# sheila layer 1, polled, and two random sets pushed without polling.

set -u
cd "$(dirname "$0")"

PY=/usr/local/share/pynq-venv/bin/python3
BIT="${1:-spike_top.bit}"
BIN="${2:-snn_smoke.bin}"

CORE="${CORE:-0}"

run() { sudo bash -c "source /etc/profile.d/xrt_setup.sh && $PY spike_pynq.py --core $CORE $*"; }
fail() { echo; echo "SNN SMOKE TEST FAILED: $*"; exit 1; }
pass() { echo "ok  - $*"; }

[ -f "$BIT" ] || fail "no bitstream $BIT"
[ -f "$BIN" ] || fail "no test firmware $BIN (build it with firmware/build_smoke.sh)"

echo "== 1. load bitstream ($BIT)"
run load-bit "$BIT" || fail "bitstream download"

echo "== 2. syscon sanity"
S=$(run status) || fail "syscon read"
echo "$S"
echo "$S" | grep -q "0x534b454c" || fail "MAGIC is not 0x534b454c (SKEL) - wrong bitstream"
pass "syscon MAGIC ok"

echo "== 3. load test firmware ($BIN)"
run load-elf "$BIN" || fail "program load"

echo "== 4. start CPU (the test takes a few tens of ms)"
run start || fail "start"
sleep 2

echo "== 5. console report"
C=$(run console)
echo "$C"
[ -n "$C" ] || fail "empty console - the CPU is not running the test (bus/decode/timing problem?)"
echo "$C" | grep -q "RESULT: PASS" || fail "console does not report RESULT: PASS (see the BAD lines above)"
pass "console says RESULT: PASS"

echo "== 6. verdict over the mailbox (time, checks, verdict, mismatches)"
R=$(run cmd status) || fail "mailbox status"
echo "$R"
read -r w0 w1 w2 w3 <<< "$R"
[ "${w2:-}" = "0x50415353" ] || fail "mailbox verdict is not 0x50415353 (PASS)"
[ "${w3:-}" = "0x0" ]        || fail "mailbox reports mismatches: $w3"
pass "verdict PASS, ${w1} checks, 0 mismatches"

echo
echo "SNN SMOKE TEST PASSED (core $CORE) - weights, ALIF parameters, events, ticks and readback of every neuron match the golden model on the FPGA."
