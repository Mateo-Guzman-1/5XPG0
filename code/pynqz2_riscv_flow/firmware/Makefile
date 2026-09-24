# firmware Makefile — minimal RISC-V / SNN firmware (PicoRV32, RV32IM)

CROSS   ?= riscv64-unknown-elf-
CC      := $(CROSS)gcc
OBJCOPY := $(CROSS)objcopy
SIZE    := $(CROSS)size

ARCHFLAGS := -march=rv32im -mabi=ilp32
CFLAGS    := $(ARCHFLAGS) -O2 -Wall -Wextra -Werror -ffreestanding -fno-builtin \
             -nostdlib -nostartfiles -g -I.
LDFLAGS   := $(ARCHFLAGS) -T linker.ld -static -nostdlib -Wl,--build-id=none \
             -Wl,--gc-sections -Wl,-Map=spike.map

OBJS := start.o main.o

all: spike.bin

%.o: %.S
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c board.h hal.h
	$(CC) $(CFLAGS) -c -o $@ $<

spike.elf: $(OBJS) linker.ld
	$(CC) $(LDFLAGS) $(OBJS) -o $@ -lgcc
	$(SIZE) $@

spike.bin: spike.elf
	$(OBJCOPY) -O binary $< $@

clean:
	rm -f $(OBJS) spike.elf spike.bin spike.map

.PHONY: all clean
