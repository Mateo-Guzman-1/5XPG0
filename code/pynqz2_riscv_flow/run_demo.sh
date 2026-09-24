#!/bin/bash
# run_demo.sh — from the PC: load and run the demo on the board over SSH.
#
# Usage: ./run_demo.sh [board-ip]      (uses .board-ip if omitted)

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IP="${1:-$(cat "$HERE/.board-ip" 2>/dev/null || true)}"
if [ -z "$IP" ]; then
    echo "usage: $0 <board-ip>" >&2
    exit 2
fi
exec ssh -t "xilinx@$IP" "cd /home/xilinx/snn && ./demo.sh"
