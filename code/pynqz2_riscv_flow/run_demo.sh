#!/bin/bash
# run_demo.sh — from the PC: load and run the keyword demo on the board over SSH.
#
# Usage: ./run_demo.sh [board-ip] [-v]     (uses .board-ip if omitted)
# Then start code/snn_keyword/pc_keyword_demo.py <board-ip> on the PC.
# The board login defaults to xilinx; override with PYNQ_USER=student.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE_USER="${PYNQ_USER:-xilinx}"
IP=""
if [ $# -ge 1 ] && [ "${1#-}" = "$1" ]; then
    IP="$1"; shift
fi
[ -n "$IP" ] || IP="$(cat "$HERE/.board-ip" 2>/dev/null || true)"
if [ -z "$IP" ]; then
    echo "usage: $0 <board-ip> [-v]" >&2
    exit 2
fi
exec ssh -t "$REMOTE_USER@$IP" "cd /home/$REMOTE_USER/snn && ./demo.sh $*"
