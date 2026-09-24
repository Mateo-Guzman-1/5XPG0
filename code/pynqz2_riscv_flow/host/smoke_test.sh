#!/bin/bash
# smoke_test.sh — bring-up test for the minimal RISC-V / SNN design.
# Runs ON THE PYNQ-Z2 (copied there by install.sh). Needs sudo + the XRT
# environment, both handled below. No external hardware required.
#
# Usage: ./smoke_test.sh   (from the directory containing spike_top.bit,
#                           spike.bin and spike_pynq.py)

set -u
cd "$(dirname "$0")"

PY=/usr/local/share/pynq-venv/bin/python3

run() {
    sudo bash -c "source /etc/profile.d/xrt_setup.sh && $PY spike_pynq.py $*"
}

fail() { echo "SMOKE TEST FAILED: $*"; exit 1; }
pass() { echo "ok  - $*"; }

echo "== 1. load bitstream"
run load-bit spike_top.bit || fail "bitstream download"

echo "== 2. syscon sanity (CPU should be held in reset)"
S=$(run status) || fail "syscon read"
echo "$S"
echo "$S" | grep -q "0x534b454c" || fail "MAGIC is not 0x534b454c (SKEL) - wrong bitstream"
pass "syscon MAGIC ok"

echo "== 3. load firmware image"
run load-elf spike.bin || fail "program load"

echo "== 4. start CPU"
run start || fail "start"
sleep 1

echo "== 5. status (CPU must be running)"
S=$(run status)
echo "$S"
echo "$S" | grep -E "^STATUS" | grep -q "0x1$" || fail "core not running (STATUS)"

echo "== 6. console banner"
C=$(run console)
echo "$C"
echo "$C" | grep -q "single LIF neuron" || fail "no console output - CPU not executing firmware"

echo "== 7. mailbox echo"
E=$(run cmd echo 3735928559 7 8)
echo "$E"
echo "$E" | grep -qi "deadbeef" || fail "mailbox echo did not return 0xdeadbeef"

sleep 1
echo "== 8. output spikes from the Poisson-stimulated neuron"
OUT=$(run spikes) || fail "spike log read"
printf '%s\n' "$OUT" | head -20
N=$(printf '%s\n' "$OUT" | grep -c "^out=" || true)
echo "collected $N new spikes (expected > 0)"
[ "$N" -gt 0 ] || fail "no output spikes - Poisson generator or neuron not running"

echo
echo "SMOKE TEST PASSED - fabric, CPU, XIP, console, mailbox and spiking all work."
