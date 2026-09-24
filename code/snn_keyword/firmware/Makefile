CROSS ?= riscv64-unknown-elf-
BUILD ?= ../build
CC := $(CROSS)gcc
FLAGS := -march=rv32im -mabi=ilp32 -O3 -Wall -Wextra -Werror -ffreestanding -fno-builtin -fno-tree-loop-distribute-patterns -nostdlib -msmall-data-limit=0 -I$(BUILD) -I.
SRC := main.c inference.c runtime.c inference.h linker.ld $(BUILD)/model_data.h $(BUILD)/model_config.h ../../pynqz2_riscv_flow/firmware/start.S
# keyword.bin: pure RV32IM. keyword_kdot.bin: uses the kdot PCPI coprocessor (needs ABI bit0).
all: $(BUILD)/keyword.bin $(BUILD)/keyword_kdot.bin
$(BUILD)/keyword.elf: $(SRC)
	$(CC) $(FLAGS) -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword.map ../../pynqz2_riscv_flow/firmware/start.S main.c inference.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
$(BUILD)/keyword_kdot.elf: $(SRC)
	$(CC) $(FLAGS) -DUSE_KDOT -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_kdot.map ../../pynqz2_riscv_flow/firmware/start.S main.c inference.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
$(BUILD)/%.bin: $(BUILD)/%.elf
	$(CROSS)objcopy -O binary $< $@
.PHONY: all
