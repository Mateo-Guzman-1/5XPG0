#!/bin/bash
# Board side of run_board_demo.ps1. Run as root from a login shell (sudo -i)
# so the PYNQ environment is loaded. Usage: board_run.sh keyword|keyword_kdot
set -u
cd "$(dirname "$0")"
IMAGE=${1:-keyword}
LOG=/tmp/board_server.log

pkill -f board_server.py && sleep 1
python3 -c "from pynq import Bitstream; Bitstream('$PWD/deploy/$IMAGE.bit').download()" || exit 1
echo "Loaded deploy/$IMAGE.bit"

# Detach fully (setsid, no inherited stdio) so the SSH session can close.
setsid nohup python3 -u board_server.py --bind 0.0.0.0 --firmware "deploy/$IMAGE.bin" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
for _ in $(seq 60); do
    if grep -q Listening "$LOG"; then cat "$LOG"; exit 0; fi
    if ! kill -0 "$PID" 2>/dev/null; then cat "$LOG"; exit 1; fi
    sleep 0.5
done
cat "$LOG"
echo "board_server.py did not start listening within 30 s"
exit 1
