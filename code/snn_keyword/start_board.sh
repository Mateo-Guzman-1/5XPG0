#!/bin/bash
# Runs ON the PYNQ-Z2 (copied there by run_board.ps1, or by hand with the files
# listed in README "Run on the PYNQ board"). Loads the keyword bitstream, then
# (re)starts board_server.py in the background and waits until it listens.
# Usage: bash start_board.sh [kdot|base] [firmware.bin] [bitstream.bit]
#        (default kdot, deploy/<image>.bin, deploy/<image>.bit)
#        bash start_board.sh stop
# The optional firmware path (relative to this folder) runs another image on the
# variant's bitstream; research/tools/bench_board.py uses it for experiment models.
# The optional bitstream path loads an experimental fabric instead (research S6).
set -euo pipefail
cd "$(dirname "$0")"
# sudo drops the variables /etc/profile.d normally sets; PYNQ finds no device without them.
PY="env XILINX_XRT=/usr BOARD=Pynq-Z2 /usr/local/share/pynq-venv/bin/python3"

stop_server() {
    # [b] keeps the pattern from matching this sudo/pkill command line itself.
    sudo pkill -f '[b]oard_server.py' || true
    for _ in $(seq 50); do pgrep -f '[b]oard_server.py' >/dev/null || return 0; sleep 0.1; done
    echo 'board_server.py did not stop' >&2; exit 1
}

case "${1:-kdot}" in
    kdot) name=keyword_kdot ;;
    base) name=keyword ;;
    stop) stop_server; echo 'Stopped'; exit 0 ;;
    *) echo "usage: $0 [kdot|base|stop] [firmware.bin]" >&2; exit 2 ;;
esac
firmware="${2:-deploy/$name.bin}"
[ -f "$firmware" ] || { echo "No firmware $firmware" >&2; exit 2; }
bitstream="${3:-deploy/$name.bit}"
[ -f "$bitstream" ] || { echo "No bitstream $bitstream" >&2; exit 2; }

stop_server
sudo $PY -c "from pynq import Bitstream; Bitstream('$bitstream').download()"
echo "Loaded $bitstream"
echo "Firmware $firmware"
sudo nohup $PY board_server.py --bind 0.0.0.0 --firmware "$firmware" > server.log 2>&1 &
for _ in $(seq 100); do
    if grep -q Listening server.log; then cat server.log; exit 0; fi
    pgrep -f '[b]oard_server.py' >/dev/null || break
    sleep 0.1
done
echo 'board_server.py failed to start:' >&2
cat server.log >&2
exit 1
