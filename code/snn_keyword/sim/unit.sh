#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
iverilog -g2012 -s tb_axi -o build/tb_axi sim/tb_axi.sv ../pynqz2_riscv_flow/rtl/ps_if.v
vvp build/tb_axi
iverilog -g2012 -s tb_led -o build/tb_led sim/tb_led.sv ../pynqz2_riscv_flow/rtl/spike_soc.v ../pynqz2_riscv_flow/rtl/picorv32.v ../pynqz2_riscv_flow/rtl/poisson.v ../pynqz2_riscv_flow/rtl/kdot_pcpi.v
vvp build/tb_led
