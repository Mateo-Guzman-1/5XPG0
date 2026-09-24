#!/bin/bash
# vivado2024.sh — run Vivado 2024.1 (the only install on this PC that has
# Zynq-7000 parts; the 2026.1 install in ~/tue/Xilinx dropped the family).
#
# Two gotchas handled here:
#  * the /storage install's settings64.sh points at an unmounted external
#    drive, so we set the environment manually;
#  * Vivado 2024.1 needs libtinfo.so.5, which Ubuntu 26.04 does not ship.
#    The library is extracted once into /home/federico/tue/vivado2024-libs.

export XILINX_VIVADO=/storage/Xilinx/Vivado/2024.1
export PATH="$XILINX_VIVADO/bin:$PATH"
export LD_LIBRARY_PATH="/home/federico/tue/vivado2024-libs:$XILINX_VIVADO/lib/lnx64.o${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$XILINX_VIVADO/bin/vivado" "$@"
