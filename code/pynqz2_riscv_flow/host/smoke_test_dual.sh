#!/bin/bash
# smoke_test_dual.sh — bring-up test for the TWO-core design (rtl/spike_top.v
# with two independent spike_soc instances). Runs ON THE PYNQ-Z2. Needs sudo +
# the XRT environment, both handled below. No external hardware required.
#
# Loads the same, unmodified firmware (spike.bin, the single-LIF-neuron demo)
# onto both cores and checks
#   A. independence: with only core 0 started, core 1 stays in reset the whole
#      time (catches the two core resets being tied together), and core 0
#      passes the normal smoke-test checks on its own;
#   B. concurrency: both cores started back to back run at the same time, each
#      passes the normal checks on its own BRAM / syscon / mailbox / spike log,
#      and the two cores' results are not the same data read twice.
#
# Usage: ./smoke_test_dual.sh [bitstream]   (default spike_top.bit; run from the
#        directory containing the bitstream, spike.bin and spike_pynq.py)

set -u
cd "$(dirname "$0")"

PY="${PY:-/usr/local/share/pynq-venv/bin/python3}"
XRT_SETUP="${XRT_SETUP:-/etc/profile.d/xrt_setup.sh}"
BIT="${1:-spike_top.bit}"

run() {
    sudo bash -c "source $XRT_SETUP && $PY spike_pynq.py $*"
}
# python on stdin, with spike_pynq importable (same environment as run)
run_py() {
    sudo bash -c "source $XRT_SETUP && $PY -"
}

fail() { echo; echo "DUAL-CORE SMOKE TEST FAILED: $*"; exit 1; }

# Separate-BRAM check: raw console head of each core (not the consuming console
# read), then a different marker into one unused word of each BRAM (0x3C000:
# above the spike log, below the stack) and both read back.
bram_probe() {    # prints: HEAD0=.. HEAD1=.. MARK0=0x.. MARK1=0x..
    run_py <<'PYEOF'
import spike_pynq as sp
b0, b1 = sp.Bram(0), sp.Bram(1)
h0, h1 = b0.rd32(sp.CONSOLE_BASE), b1.rd32(sp.CONSOLE_BASE)
b0.wr32(0x3C000, 0xB0B0B0B0)
b1.wr32(0x3C000, 0xB1B1B1B1)
print(f"HEAD0={h0} HEAD1={h1} MARK0={b0.rd32(0x3C000):#x} MARK1={b1.rd32(0x3C000):#x}")
PYEOF
}
check_marks() {   # check_marks <step>
    [ "$MARK0" = "0xb0b0b0b0" ] && [ "$MARK1" = "0xb1b1b1b1" ] \
        || fail "$1 BRAM crosstalk: wrote 0xb0b0b0b0 to core 0 and 0xb1b1b1b1 to core 1 at 0x3C000, read back $MARK0 / $MARK1 (both cores decode to the same BRAM?)"
}
pass() { echo "ok  - $*"; }

status_word() {   # status_word <core> -> the STATUS value, e.g. 0x1
    run --core "$1" status | awk '$1 == "STATUS" { print $2 }'
}

# the normal smoke_test.sh checks (steps 5-8) for one core; the echo arguments
# are per core, so each core must answer with its own values
core_checks() {   # core_checks <core> <echo a1> <echo a2> <step label>
    local c="$1" a1="$2" a2="$3" L="$4" S C E OUT N

    echo "== $L.a core $c: status (must be running)"
    S=$(run --core "$c" status) || fail "$L core $c: syscon read"
    echo "$S"
    echo "$S" | grep -q "0x534b454c" || fail "$L core $c: MAGIC is not 0x534b454c (SKEL)"
    echo "$S" | grep -E "^STATUS" | grep -q "0x1$" || fail "$L core $c: not running (STATUS)"
    pass "core $c running, MAGIC ok"

    echo "== $L.b core $c: console banner"
    C=$(run --core "$c" console)
    echo "$C"
    echo "$C" | grep -q "single LIF neuron" \
        || fail "$L core $c: no 'single LIF neuron' banner - CPU $c not executing firmware"
    pass "core $c console banner"

    echo "== $L.c core $c: mailbox echo 0xdeadbeef $a1 $a2"
    E=$(run --core "$c" cmd echo 3735928559 "$a1" "$a2")
    echo "$E"
    echo "$E" | grep -qi "^0xdeadbeef $(printf '%#x %#x' "$a1" "$a2") " \
        || fail "$L core $c: mailbox echo did not return 0xdeadbeef $(printf '%#x %#x' "$a1" "$a2")"
    pass "core $c mailbox echo returns its own arguments"

    sleep 1
    echo "== $L.d core $c: output spikes from its Poisson-stimulated neuron"
    OUT=$(run --core "$c" spikes) || fail "$L core $c: spike log read"
    printf '%s\n' "$OUT" | head -5
    N=$(printf '%s\n' "$OUT" | grep -c "^out=" || true)
    echo "core $c: $N new spikes (expected > 0)"
    [ "$N" -gt 0 ] || fail "$L core $c: no output spikes - Poisson generator or neuron not running"
    pass "core $c spiking"
    SPIKES[$c]="$OUT"
}
declare -a SPIKES

[ -f "$BIT" ]       || fail "no bitstream $BIT"
[ -f spike.bin ]    || fail "no spike.bin"

echo "== 0. load bitstream ($BIT; one download configures both cores)"
run load-bit "$BIT" || fail "bitstream download"

echo "== 1. syscon sanity on both cores (both CPUs held in reset)"
for c in 0 1; do
    S=$(run --core "$c" status) \
        || fail "core $c syscon read failed (a single-core bitstream answers core 1 with a bus error)"
    echo "$S"
    echo "$S" | grep -q "0x534b454c" || fail "core $c MAGIC is not 0x534b454c (SKEL) - wrong bitstream"
    echo "$S" | grep -E "^CTRL"   | grep -q "0x1$" || fail "core $c not held in reset after configuration (CTRL)"
    echo "$S" | grep -E "^STATUS" | grep -q "0x0$" || fail "core $c running before it was started (STATUS)"
done
pass "both cores: MAGIC ok, held in reset"

# =====================================================================
echo
echo "########## A. independence: only core 0 is started ##########"
echo "== A.1 load spike.bin onto both cores (both stay in reset)"
run --core 0 load-elf spike.bin || fail "A.1 program load, core 0"
run --core 1 load-elf spike.bin || fail "A.1 program load, core 1"

echo "== A.2 start core 0 only"
run --core 0 start || fail "A.2 start core 0"

echo "== A.3 core 1 must stay in reset while core 0 runs"
for i in 1 2 3 4 5; do
    W=$(status_word 1) || fail "A.3 core 1 status read"
    echo "   poll $i: core 1 STATUS $W"
    [ "$W" = "0x0" ] || fail "A.3 RESET-COUPLING BUG: core 1 STATUS is $W (running) although only core 0 was started - the two core_rst_n in spike_top.v are not independent; the rest of this test is meaningless until that is fixed"
    sleep 0.4
done
pass "core 1 stayed in reset over 5 polls"

core_checks 0 7 8 "A.4"

echo "== A.4.e core 1 untouched while core 0 ran; the two BRAMs are separate"
W=$(status_word 1)
[ "$W" = "0x0" ] || fail "A.4 RESET-COUPLING BUG: core 1 STATUS is $W after core 0's checks"
P=$(bram_probe) || fail "A.4 BRAM probe"
echo "   $P"
eval "$P"
[ "$HEAD0" -gt 0 ] || fail "A.4 core 0 console head is 0 although core 0 printed its banner"
[ "$HEAD1" -eq 0 ] || fail "A.4 core 1 console head is $HEAD1 although core 1 never ran (core 0's output reached core 1's BRAM?)"
check_marks "A.4"
pass "core 1 still in reset, its console never written, each core has its own BRAM"

echo "== A.5 stop core 0, reset both"
run --core 0 stop || fail "A.5 stop core 0"
run --core 1 stop || fail "A.5 stop core 1"

# =====================================================================
echo
echo "########## B. concurrency: both cores run at the same time ##########"
echo "== B.0 reload spike.bin onto both cores (fresh console, mailbox, spike log)"
run --core 0 load-elf spike.bin || fail "B.0 program load, core 0"
run --core 1 load-elf spike.bin || fail "B.0 program load, core 1"

echo "== B.1 + B.2 start core 0 then core 1 back to back, read both TIMEs twice"
# One process for both CTRL writes, so they are microseconds apart (a separate
# spike_pynq.py call per core costs about a second of sudo + python start-up).
T=$(run_py <<'EOF'
import time
import spike_pynq as sp
r0, r1 = sp.Regs(0), sp.Regs(1)
r0.ctrl_reset(False)
r1.ctrl_reset(False)
a0, a1 = r0._rd(sp.TIME), r1._rd(sp.TIME)
time.sleep(0.2)
b0, b1 = r0._rd(sp.TIME), r1._rd(sp.TIME)
print(f"T0A={a0} T1A={a1} T0B={b0} T1B={b1}")
EOF
) || fail "B.1 start / TIME read"
echo "   $T"
eval "$T"
[ "${T0A:-0}" -gt 0 ] && [ "${T1A:-0}" -gt 0 ] || fail "B.2 a TIME register reads 0 (core 0: ${T0A:-?}, core 1: ${T1A:-?})"
# TIME is a 32-bit counter at 100 MHz (wraps every ~43 s): differences are taken mod 2^32
d0=$(( (T0B - T0A) & 0xFFFFFFFF )); d1=$(( (T1B - T1A) & 0xFFFFFFFF ))
echo "   core 0 TIME advanced $d0 ticks, core 1 TIME advanced $d1 ticks in ~0.2 s"
[ "$d0" -gt 1000000 ] || fail "B.2 core 0 TIME not advancing ($T0A -> $T0B)"
[ "$d1" -gt 1000000 ] || fail "B.2 core 1 TIME not advancing ($T1A -> $T1B)"
pass "both TIME registers non-zero and advancing"

for c in 0 1; do
    W=$(status_word "$c")
    [ "$W" = "0x1" ] || fail "B.1 core $c not running right after the back-to-back start (STATUS $W)"
done
pass "both cores running"

core_checks 0 16 17 "B.3"     # 0x10 0x11
core_checks 1 32 33 "B.3"     # 0x20 0x21

echo "== B.4 the two cores' results are separate data"
[ "${SPIKES[0]}" != "${SPIKES[1]}" ] \
    || fail "B.4 core 0 and core 1 spike logs are identical - the same BRAM read twice (address decode bug)?"
pass "spike logs differ (first spike: core 0 '$(printf '%s\n' "${SPIKES[0]}" | head -1)', core 1 '$(printf '%s\n' "${SPIKES[1]}" | head -1)')"
X=$(run_py <<'EOF'
import spike_pynq as sp
r0, r1 = sp.Regs(0), sp.Regs(1)
r0._wr(sp.SCRATCH, 0xC0C0C0C0)
r1._wr(sp.SCRATCH, 0xC1C1C1C1)
print(f"{r0._rd(sp.SCRATCH):#x} {r1._rd(sp.SCRATCH):#x}")
EOF
) || fail "B.4 SCRATCH access"
echo "   SCRATCH after writing 0xc0c0c0c0 to core 0 and 0xc1c1c1c1 to core 1: $X"
[ "$X" = "0xc0c0c0c0 0xc1c1c1c1" ] || fail "B.4 SCRATCH crosstalk between the cores' syscon: read back $X"
pass "each core keeps its own SCRATCH"
P=$(bram_probe) || fail "B.4 BRAM probe"
echo "   $P"
eval "$P"
[ "$HEAD0" -gt 0 ] && [ "$HEAD1" -gt 0 ] || fail "B.4 a console head is 0 although both cores printed (core 0: $HEAD0, core 1: $HEAD1)"
check_marks "B.4"
pass "each core keeps its own BRAM word"

echo
echo "DUAL-CORE SMOKE TEST PASSED - two independent cores: separate resets, BRAMs, consoles, mailboxes, spike logs and syscon, running concurrently."
