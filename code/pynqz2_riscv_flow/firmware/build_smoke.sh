#!/usr/bin/env bash
# build_smoke.sh -- build the on-board neuron test as firmware/snn_smoke.bin.
# Same flags as the firmware Makefile; does NOT touch spike.bin, main.o or the Makefile.
# Needs the same RISC-V toolchain that `make -C firmware` uses (Linux or WSL).
#   CROSS=riscv-none-elf- ./build_smoke.sh      # if your toolchain has another prefix
set -euo pipefail
cd "$(dirname "$0")"

CROSS="${CROSS:-riscv64-unknown-elf-}"
ARCH="-march=rv32im -mabi=ilp32"
CFLAGS="$ARCH -O2 -Wall -Wextra -Werror -ffreestanding -fno-builtin -nostdlib -nostartfiles -g -I."
LDFLAGS="$ARCH -T linker.ld -static -nostdlib -Wl,--build-id=none -Wl,--gc-sections -Wl,-Map=snn_smoke.map"

# snn_vectors.h ships pre-generated; only regenerate if it is missing
if [ ! -f snn_vectors.h ]; then
    python3 ../sim/golden_alif.py hwvec --out snn_vectors.h
fi

${CROSS}gcc $CFLAGS -c -o start_smoke.o start.S
${CROSS}gcc $CFLAGS -c -o snn_smoke.o snn_smoke.c
${CROSS}gcc $LDFLAGS start_smoke.o snn_smoke.o -o snn_smoke.elf -lgcc
${CROSS}size snn_smoke.elf
${CROSS}objcopy -O binary snn_smoke.elf snn_smoke.bin
echo "built: $(pwd)/snn_smoke.bin  ($(stat -c %s snn_smoke.bin) bytes; limit 65280)"
