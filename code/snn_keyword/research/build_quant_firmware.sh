#!/usr/bin/env bash
# Build the stage-1 firmware variants (research/export_quant.py models) into
# research/results/quant/build/. Uses the flags of firmware/Makefile and never
# touches firmware/keyword.bin or firmware/keyword_model.h.
#
#   name          loop (KW_VARIANT)            model
#   default       auto (no defines at all)     firmware/keyword_model.h (QAT int4 -> loop 3)
#   v0_damien     0  one lb per weight         Damien's original model (sheila_v2_deployment)
#   v0_int8       0  one lb per weight         export_int8
#   v1_int8       1  unrolled 4x, lb           export_int8
#   v2_int8       2  packed, lw per 4 weights  export_int8
#   v0_int4       0  one lb per weight         export_int4
#   v3_int4       3  packed, lw per 8 weights  export_int4
#
# Per build: <name>.bin (load with spike_pynq.py load-elf), .elf, .dis (objdump),
# and one line in sizes.txt.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FW="$HERE/../../pynqz2_riscv_flow/firmware"
QUANT="$HERE/results/quant"
OUT="$QUANT/build"
CROSS="${CROSS:-riscv64-unknown-elf-}"

ARCH="-march=rv32im -mabi=ilp32"
CFLAGS="$ARCH -O2 -Wall -Wextra -Werror -ffreestanding -fno-builtin -nostdlib -nostartfiles -g -I$FW"
LDFLAGS="$ARCH -T $FW/linker.ld -static -nostdlib -Wl,--build-id=none -Wl,--gc-sections"

mkdir -p "$OUT"
"${CROSS}gcc" $CFLAGS -c -o "$OUT/start.o" "$FW/start.S"
: > "$OUT/sizes.txt"

build() {
    local name=$1; shift
    "${CROSS}gcc" $CFLAGS "$@" -c -o "$OUT/$name.o" "$FW/keyword_main.c"
    "${CROSS}gcc" $LDFLAGS -Wl,-Map="$OUT/$name.map" "$OUT/start.o" "$OUT/$name.o" -o "$OUT/$name.elf" -lgcc
    "${CROSS}objcopy" -O binary "$OUT/$name.elf" "$OUT/$name.bin"
    "${CROSS}objdump" -d -S "$OUT/$name.elf" > "$OUT/$name.dis"
    printf '%-12s %s\n' "$name" "$("${CROSS}size" "$OUT/$name.elf" | tail -1)" >> "$OUT/sizes.txt"
}

model() { echo "-DKW_MODEL_HEADER=\"$QUANT/export_$1/keyword_model.h\""; }

build default
build v0_damien "-DKW_MODEL_HEADER=\"$HERE/results/sheila_v2_deployment/keyword_model.h\""
build v0_int8 -DKW_VARIANT=0 "$(model int8)"
build v1_int8 -DKW_VARIANT=1 "$(model int8)"
build v2_int8 -DKW_VARIANT=2 "$(model int8)"
build v0_int4 -DKW_VARIANT=0 "$(model int4)"
build v3_int4 -DKW_VARIANT=3 "$(model int4)"

echo "        text    data     bss     dec     hex filename"
cat "$OUT/sizes.txt"
echo "built into $OUT"
