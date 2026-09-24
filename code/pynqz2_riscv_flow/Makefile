# Makefile — build + deploy + test the minimal RISC-V / SNN bring-up design.
#
#   make bitstream   build vivado/spike_top.bit        (Vivado 2024.1, ~15 min)
#   make firmware    build firmware/spike.bin           (riscv gcc, seconds)
#   make install     install.sh <BOARD_IP>
#   make smoke       install + run the smoke test on the board
#   make demo        run the interactive demo on the board
#   make clean       remove build artifacts
#
# BOARD_IP defaults to the value remembered in .board-ip.

BOARD_IP  ?= $(shell cat .board-ip 2>/dev/null || echo 192.168.2.99)

.PHONY: all bitstream firmware install smoke demo clean help

all: bitstream firmware

bitstream: vivado/spike_top.bit

vivado/spike_top.bit: rtl/*.v vivado/spike_top.xdc vivado/build.tcl vivado/vivado2024.sh
	mkdir -p vivado/build-logs
	cd vivado && ./vivado2024.sh -mode batch -source build.tcl \
	    -log build-logs/vivado.log -journal build-logs/vivado.jou
	@test -f vivado/spike_top.bit

firmware: firmware/spike.bin

firmware/spike.bin: firmware/*.c firmware/*.h firmware/start.S firmware/linker.ld firmware/Makefile
	$(MAKE) -C firmware

install:
	./install.sh $(BOARD_IP)

smoke:
	./install.sh $(BOARD_IP) --smoke-test

demo: install
	./run_demo.sh $(BOARD_IP)

clean:
	rm -rf vivado/rvproj vivado/spike_top.bit vivado/build-logs
	$(MAKE) -C firmware clean

help:
	@sed -n '2,13p' Makefile | sed 's/^# \{0,1\}//'
