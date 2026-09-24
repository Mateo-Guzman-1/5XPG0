#!/bin/bash
# install.sh — one-shot installer for the minimal RISC-V / SNN bring-up
# design on a PYNQ-Z2.
#
# Given the board IP, this copies everything to the board and leaves it ready:
#   - the prebuilt bitstream           (vivado/spike_top.bit)
#   - the RISC-V firmware              (firmware/spike.bin)
#   - the host tool + smoke test       (host/spike_pynq.py, host/smoke_test.sh)
#   - Python deps in the PYNQ venv     (requirements.txt: pyzmq)
#
# Usage:
#   ./install.sh <board-ip> [--smoke-test]
#
# PREREQUISITES (your responsibility, done once per board):
#   * a PYNQ-Z2 running the stock PYNQ 3.1 image with an Ethernet cable,
#   * SSH key auth for user `xilinx` at <board-ip> (so `ssh xilinx@<ip>` needs
#     no password),
#   * passwordless sudo on the board (the stock image ships the credentials
#     xilinx / xilinx; fingerprint it below with `sudo -n true`),
#   * the PC and the board on the same network, internet reachable from the
#     board (pip install).
#
# <board-ip> is a plain IP such as 192.168.2.99. It is remembered in
# .board-ip for later runs (just run ./install.sh --smoke-test to reuse it).
#
# If the prebuilt bitstream is missing, build it first with:
#   make -C .. bitstream      (Vivado 2024.1, ~15 min)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

REMOTE_USER=xilinx
REMOTE_DIR=/home/xilinx/snn
VENV_PY=/usr/local/share/pynq-venv/bin/python3
PREBUILT_BIT="$HERE/vivado/spike_top.bit"
FIRMWARE_BIN="$HERE/firmware/spike.bin"

usage() { sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

BOARD_IP=""
SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --smoke-test) SMOKE=1 ;;
        -h|--help) usage 0 ;;
        -*) echo "unknown option: $arg" >&2; usage 2 ;;
        *)  if [ -n "$BOARD_IP" ]; then
                echo "unexpected argument: $arg" >&2; usage 2
            fi
            BOARD_IP="$arg" ;;
    esac
done
if [ -z "$BOARD_IP" ]; then
    BOARD_IP="$(cat "$HERE/.board-ip" 2>/dev/null || true)"
    [ -n "$BOARD_IP" ] || usage 2
fi
BOARD_HOST="$REMOTE_USER@$BOARD_IP"

step() { printf '\n== %s\n' "$*"; }
die()  { echo "INSTALL FAILED: $*" >&2; exit 1; }

if [ ! -f "$PREBUILT_BIT" ]; then
    die "no prebuilt bitstream at $PREBUILT_BIT — build it with: make -C $HERE bitstream"
fi
if [ ! -f "$FIRMWARE_BIN" ]; then
    die "no firmware at $FIRMWARE_BIN — build it with: make -C $HERE/firmware"
fi

step "0. check access to $BOARD_HOST"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$BOARD_HOST" true \
    || die "cannot ssh to $BOARD_HOST (needs key auth, user $REMOTE_USER)"
ssh "$BOARD_HOST" '[ -x '"$VENV_PY"' ]' \
    || die "$VENV_PY not found — not a PYNQ 3.1 board?"
ssh "$BOARD_HOST" 'sudo -n true' \
    || die "passwordless sudo missing (stock PYNQ: user xilinx, password xilinx)"
echo "$BOARD_IP" > "$HERE/.board-ip"
echo "ok  - ssh + passwordless sudo"

step "1. deploy bitstream, firmware, host tool"
ssh "$BOARD_HOST" "mkdir -p $REMOTE_DIR"
scp -q "$PREBUILT_BIT" "$FIRMWARE_BIN" \
    "$HERE/host/spike_pynq.py" "$HERE/host/smoke_test.sh" "$HERE/host/demo.sh" \
    "$BOARD_HOST:$REMOTE_DIR/"
ssh "$BOARD_HOST" "chmod +x $REMOTE_DIR/smoke_test.sh $REMOTE_DIR/demo.sh"
echo "deployed to $BOARD_HOST:$REMOTE_DIR"

step "2. install Python deps in the PYNQ venv"
scp -q "$HERE/requirements.txt" "$BOARD_HOST:$REMOTE_DIR/"
ssh "$BOARD_HOST" "sudo $VENV_PY -m pip install -q -r $REMOTE_DIR/requirements.txt" \
    || die "pip install failed (board needs internet)"

if [ "$SMOKE" = 1 ]; then
    step "3. smoke test on the board"
    ssh "$BOARD_HOST" "cd $REMOTE_DIR && ./smoke_test.sh" \
        || die "smoke test failed"
fi

cat <<EOF

Board $BOARD_IP is ready. On the board ($REMOTE_DIR):
  sudo bash -c "source /etc/profile.d/xrt_setup.sh && \\
      /usr/local/share/pynq-venv/bin/python3 spike_pynq.py load-bit spike_top.bit"
  ... spike_pynq.py load-elf spike.bin
  ... spike_pynq.py start
  ... spike_pynq.py console      # firmware banner
  ... spike_pynq.py spikes       # output spikes from the demo neuron

Easiest: from this PC run  ./run_demo.sh $BOARD_IP
EOF
