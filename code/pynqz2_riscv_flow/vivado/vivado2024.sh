#!/bin/bash
# vivado2024.sh — run Vivado 2024.1 (needed for the Zynq-7000 parts; newer
# installs such as 2026.1 dropped the family).
#
# The install is found in this order:
#   1. $XILINX_VIVADO, if set (e.g. XILINX_VIVADO=/opt/Xilinx/Vivado/2024.1)
#   2. the known install locations below (first one that exists wins)
#
# VIVADO_TOOL runs another tool from the same install with the same
# environment, e.g. VIVADO_TOOL=xvlog ./vivado2024.sh ... (used by ../sim).
#
# Two gotchas handled here:
#  * some installs' settings64.sh point at an unmounted external drive, so we
#    set the environment manually instead of sourcing it;
#  * Vivado 2024.1 needs libtinfo.so.5. Ubuntu 24.04 ships it in the system
#    library path; on Ubuntu 26.04 it is extracted once into a private
#    directory (VIVADO_EXTRA_LIBS, default /home/federico/tue/vivado2024-libs).

if [ -z "$XILINX_VIVADO" ]; then
    for candidate in \
        /storage/Xilinx/Vivado/2024.1 \
        "$HOME/Vivado/Vivado/2024.1" \
        /tools/Xilinx/Vivado/2024.1 \
        /opt/Xilinx/Vivado/2024.1; do
        if [ -x "$candidate/bin/vivado" ]; then
            XILINX_VIVADO=$candidate
            break
        fi
    done
fi
if [ -z "$XILINX_VIVADO" ] || [ ! -x "$XILINX_VIVADO/bin/vivado" ]; then
    echo "vivado2024.sh: Vivado 2024.1 not found; set XILINX_VIVADO" >&2
    exit 1
fi
export XILINX_VIVADO
export PATH="$XILINX_VIVADO/bin:$PATH"

extra_libs=${VIVADO_EXTRA_LIBS:-/home/federico/tue/vivado2024-libs}
libs="$XILINX_VIVADO/lib/lnx64.o"
[ -d "$extra_libs" ] && libs="$extra_libs:$libs"
export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$XILINX_VIVADO/bin/${VIVADO_TOOL:-vivado}" "$@"
