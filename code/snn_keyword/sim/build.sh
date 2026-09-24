#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
make -C firmware
gcc -O3 -Wall -Wextra -Werror -Ibuild -Ifirmware sim/native.c firmware/inference.c -o build/native
verilator --cc --exe --build -j 8 -Wno-fatal --top-module spike_soc \
  --Mdir "$PWD/build/obj_dir" -CFLAGS '-O3' \
  ../pynqz2_riscv_flow/rtl/spike_soc.v ../pynqz2_riscv_flow/rtl/picorv32.v \
  ../pynqz2_riscv_flow/rtl/poisson.v ../pynqz2_riscv_flow/rtl/kdot_pcpi.v "$PWD/sim/soc_main.cpp"
