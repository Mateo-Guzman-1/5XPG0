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

OBJS := start.o main.o snn.o

HOSTCC ?= cc

all: spike.bin

%.o: %.S
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c board.h hal.h snn.h weights.h
	$(CC) $(CFLAGS) -c -o $@ $<

# host-side check that snn.c matches the Python integer model
test: test_snn.c snn.c snn.h weights.h test_vectors.h
	$(HOSTCC) -O2 -Wall -Wextra -I. test_snn.c snn.c -o test_snn
	./test_snn

spike.elf: $(OBJS) linker.ld
	$(CC) $(LDFLAGS) $(OBJS) -o $@ -lgcc
	$(SIZE) $@

spike.bin: spike.elf
	$(OBJCOPY) -O binary $< $@

clean:
	rm -f $(OBJS) spike.elf spike.bin spike.map test_snn test_snn.exe

.PHONY: all clean test
