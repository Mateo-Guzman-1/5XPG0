CROSS ?= riscv64-unknown-elf-
BUILD ?= ../build
CC := $(CROSS)gcc
FLAGS := -march=rv32im -mabi=ilp32 -O3 -Wall -Wextra -Werror -ffreestanding -fno-builtin -fno-tree-loop-distribute-patterns -nostdlib -msmall-data-limit=0 -I$(BUILD) -I.
all: $(BUILD)/keyword.bin
$(BUILD)/keyword.elf: main.c inference.c runtime.c inference.h linker.ld $(BUILD)/model_data.h $(BUILD)/model_config.h ../../pynqz2_riscv_flow/firmware/start.S
	$(CC) $(FLAGS) -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword.map ../../pynqz2_riscv_flow/firmware/start.S main.c inference.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
$(BUILD)/keyword.bin: $(BUILD)/keyword.elf
	$(CROSS)objcopy -O binary $< $@
.PHONY: all
