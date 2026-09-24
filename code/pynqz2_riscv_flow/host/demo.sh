#!/bin/bash
# demo.sh — run the full bring-up demo on the board (Runs ON THE PYNQ-Z2).
# Loads the bitstream + firmware, starts the CPU, prints the banner and then
# dumps output spikes collected over a few seconds.

set -u
cd "$(dirname "$0")"

PY=/usr/local/share/pynq-venv/bin/python3

run() { sudo bash -c "source /etc/profile.d/xrt_setup.sh && $PY spike_pynq.py $*"; }

run load-bit spike_top.bit || exit 1
run load-elf spike.bin     || exit 1
run start                  || exit 1
sleep 1

echo "--- firmware console ---"
run console

echo
echo "--- collecting output spikes for 3 s ---"
sleep 3
run spikes | head -40

echo
echo "--- run 'spike_pynq.py spikes' again to see more (the ring keeps filling) ---"
