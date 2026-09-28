#!/bin/bash
# demo.sh — run the "sheila" keyword demo on the board (Runs ON THE PYNQ-Z2).
# Loads the bitstream + firmware, starts the CPU, then bridges spectrogram
# frames from the PC (pc_keyword_demo.py <board-ip>) into the RISC-V core.
#
# Usage: ./demo.sh [-v]     (-v prints the result of every frame)

set -u
cd "$(dirname "$0")"

PY=/usr/local/share/pynq-venv/bin/python3

run() { sudo bash -c "source /etc/profile.d/xrt_setup.sh && $PY $*"; }

run spike_pynq.py load-bit spike_top.bit || exit 1
run spike_pynq.py load-elf spike.bin     || exit 1
run spike_pynq.py start                  || exit 1
sleep 1

echo "--- firmware console ---"
run spike_pynq.py console

echo
run keyword_bridge.py "$@"
