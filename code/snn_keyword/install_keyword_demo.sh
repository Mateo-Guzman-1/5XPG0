#!/bin/bash
# Build, deploy, and start the complete Group-2 keyword path.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FLOW="$HERE/../pynqz2_riscv_flow"
BOARD_IP="${1:-192.168.2.99}"
BOARD="xilinx@$BOARD_IP"
REMOTE=/home/xilinx/snn_keyword
PY=/usr/local/share/pynq-venv/bin/python3

step() { printf '\n== %s\n' "$*"; }
die() { echo "KEYWORD INSTALL FAILED: $*" >&2; exit 1; }

step "1. locate quantized keyword firmware"
[ -f "$FLOW/firmware/keyword_model.h" ] \
    || die "keyword_model.h is missing; run train_keyword_snn.py first"
if [ ! -f "$FLOW/firmware/keyword.bin" ]; then
    echo "keyword.bin is missing; building it with the RISC-V toolchain"
    make -C "$FLOW/firmware" keyword.bin
else
    echo "using prebuilt $FLOW/firmware/keyword.bin"
fi

step "2. verify board access"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$BOARD" true \
    || die "cannot reach $BOARD with SSH key authentication"
ssh "$BOARD" 'sudo -n true' \
    || die "passwordless sudo is required"

step "3. deploy bitstream, firmware, and bridge"
ssh "$BOARD" "mkdir -p '$REMOTE'"
scp -q \
    "$FLOW/vivado/spike_top.bit" \
    "$FLOW/firmware/keyword.bin" \
    "$FLOW/host/spike_pynq.py" \
    "$FLOW/host/keyword_bridge.py" \
    "$FLOW/host/keyword_service.sh" \
    "$HERE/audio_features.py" \
    "$BOARD:$REMOTE/"
ssh "$BOARD" "chmod +x '$REMOTE/keyword_service.sh' '$REMOTE/keyword_bridge.py'"

step "4. load fabric and keyword firmware"
ssh "$BOARD" "cd '$REMOTE' && ./keyword_service.sh stop >/dev/null 2>&1 || true"
ssh "$BOARD" "cd '$REMOTE' && sudo -n bash -c 'source /etc/profile.d/xrt_setup.sh && \
    $PY spike_pynq.py load-bit spike_top.bit && \
    $PY spike_pynq.py load-elf keyword.bin && \
    $PY spike_pynq.py start'"
sleep 1
ssh "$BOARD" "cd '$REMOTE' && sudo -n bash -c 'source /etc/profile.d/xrt_setup.sh && \
    $PY spike_pynq.py console'"

step "5. start Ethernet-to-BRAM bridge"
ssh "$BOARD" "cd '$REMOTE' && ./keyword_service.sh start"

cat <<EOF

Keyword detector is ready at tcp://$BOARD_IP:5556.
Microphone demo:
  python pc_keyword_demo.py $BOARD_IP
WAV test:
  python pc_keyword_demo.py $BOARD_IP --wav path/to/sample.wav
EOF
