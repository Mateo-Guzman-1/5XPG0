#!/bin/bash
# Load the Group-2 Sheila detector and run its Ethernet bridge at board boot.

set -euo pipefail

REMOTE=/home/xilinx/snn_keyword
PY=/usr/local/share/pynq-venv/bin/python3

cd "$REMOTE"
source /etc/profile.d/xrt_setup.sh

"$PY" spike_pynq.py load-bit spike_top.bit
"$PY" spike_pynq.py load-elf keyword.bin
"$PY" spike_pynq.py start
sleep 1
"$PY" spike_pynq.py console

exec "$PY" keyword_bridge.py --bind 'tcp://*:5556'
