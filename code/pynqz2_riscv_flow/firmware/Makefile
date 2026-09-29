# firmware Makefile — minimal RISC-V / SNN firmware (PicoRV32, RV32IM)

CROSS   ?= riscv64-unknown-elf-
CC      := $(CROSS)gcc
OBJCOPY := $(CROSS)objcopy
SIZE    := $(CROSS)size

ARCHFLAGS := -march=rv32im -mabi=ilp32
CFLAGS    := $(ARCHFLAGS) -O2 -Wall -Wextra -Werror -ffreestanding -fno-builtin \
             -nostdlib -nostartfiles -g -I.
LDFLAGS   := $(ARCHFLAGS) -T linker.ld -static -nostdlib -Wl,--build-id=none \
             -Wl,--gc-sections

DEMO_OBJS    := start.o main.o
KEYWORD_OBJS := start.o keyword_main.o

all: spike.bin keyword.bin

%.o: %.S
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c board.h hal.h
	$(CC) $(CFLAGS) -c -o $@ $<

keyword_main.o: keyword_main.c keyword_model.h board.h hal.h
	$(CC) $(CFLAGS) -c -o $@ $<

spike.elf: $(DEMO_OBJS) linker.ld
	$(CC) $(LDFLAGS) -Wl,-Map=$(@:.elf=.map) $(DEMO_OBJS) -o $@ -lgcc
	$(SIZE) $@

spike.bin: spike.elf
	$(OBJCOPY) -O binary $< $@

keyword.elf: $(KEYWORD_OBJS) linker.ld
	$(CC) $(LDFLAGS) -Wl,-Map=$(@:.elf=.map) $(KEYWORD_OBJS) -o $@ -lgcc
	$(SIZE) $@

keyword.bin: keyword.elf
	$(OBJCOPY) -O binary $< $@

clean:
	rm -f $(DEMO_OBJS) $(KEYWORD_OBJS) spike.elf spike.bin spike.map \
		keyword.elf keyword.bin keyword.map

.PHONY: all clean
