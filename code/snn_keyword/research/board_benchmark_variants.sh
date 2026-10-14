#!/usr/bin/env bash
# Benchmark firmware variants on the board (same steps as install_keyword_demo.sh).
#
#   research/board_benchmark_variants.sh <BOARD_IP> [variant ...]
#
# Loads vivado/spike_top.bit once, then for every variant (default: the four
# compared with the MAC unit) loads research/results/quant/build/<variant>.bin,
# restarts the bridge and replays the 40 board vectors of the matching model
# with benchmark_candidate_board.py. Results: research/results/quant/board/
# <variant>.json (cycles + layer1_cycles, 40/40 must match). Afterwards the
# deployed keyword.bin is loaded again, so the live demo keeps working.
# Build the variants first: research/build_quant_firmware.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
KW="$HERE/.."
FLOW="$KW/../pynqz2_riscv_flow"
QUANT="$HERE/results/quant"
BOARD_IP="${1:?usage: $0 <BOARD_IP> [variant ...]}"
shift
[ $# -gt 0 ] && VARIANTS=("$@") || VARIANTS=(v0_int8 v3_int4 v4_int8 v5_int4)
BOARD="xilinx@$BOARD_IP"
REMOTE=/home/xilinx/snn_keyword
PY=/usr/local/share/pynq-venv/bin/python3
LOCAL_PY="$KW/.venv/bin/python"

step() { printf '\n== %s\n' "$*"; }
on_board() { ssh "$BOARD" "cd '$REMOTE' && sudo -n bash -c 'source /etc/profile.d/xrt_setup.sh && $1'"; }

load_firmware() {   # $1 = local .bin
    scp -q "$1" "$BOARD:$REMOTE/keyword.bin"
    ssh "$BOARD" "cd '$REMOTE' && ./keyword_service.sh stop >/dev/null 2>&1 || true"
    on_board "$PY spike_pynq.py load-elf keyword.bin && $PY spike_pynq.py start"
    sleep 1
    on_board "$PY spike_pynq.py console" | tail -2
    ssh "$BOARD" "cd '$REMOTE' && ./keyword_service.sh start" >/dev/null
}

step "deploy bitstream and bridge"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$BOARD" true
ssh "$BOARD" "mkdir -p '$REMOTE'"
scp -q "$FLOW/vivado/spike_top.bit" "$FLOW/host/spike_pynq.py" \
    "$FLOW/host/keyword_bridge.py" "$FLOW/host/keyword_service.sh" \
    "$KW/audio_features.py" "$BOARD:$REMOTE/"
ssh "$BOARD" "chmod +x '$REMOTE/keyword_service.sh' '$REMOTE/keyword_bridge.py'"
ssh "$BOARD" "cd '$REMOTE' && ./keyword_service.sh stop >/dev/null 2>&1 || true"
on_board "$PY spike_pynq.py load-bit spike_top.bit"

for variant in "${VARIANTS[@]}"; do
    case "$variant" in
        *int8) vectors="$QUANT/export_int8/board_vectors.npz" ;;
        *int4) vectors="$QUANT/export_int4/board_vectors.npz" ;;
        *) echo "unknown model for $variant" >&2; exit 1 ;;
    esac
    step "$variant"
    load_firmware "$QUANT/build/$variant.bin"
    "$LOCAL_PY" "$HERE/benchmark_candidate_board.py" "$BOARD_IP" "$vectors" \
        --bands 8 --label "$variant" --output "$QUANT/board/$variant.json"
done

step "restore the deployed keyword firmware"
load_firmware "$FLOW/firmware/keyword.bin"
echo "results in $QUANT/board/"
