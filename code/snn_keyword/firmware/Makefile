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
# Streaming model (ABI v3): headers from export_model.py into $(BUILD)/stream.
STREAM_SRC := stream_main.c stream_infer.c stream_infer.h runtime.c linker.ld $(BUILD)/stream/model_stream_data.h $(BUILD)/stream/model_stream_config.h ../../pynqz2_riscv_flow/firmware/start.S
stream: $(BUILD)/keyword_stream.bin $(BUILD)/keyword_stream_kdot.bin $(BUILD)/keyword_stream_engine.bin $(BUILD)/keyword_stream_engine_l1.bin
$(BUILD)/keyword_stream.elf: $(STREAM_SRC)
	$(CC) $(FLAGS) -I$(BUILD)/stream -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_stream.map ../../pynqz2_riscv_flow/firmware/start.S stream_main.c stream_infer.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
$(BUILD)/keyword_stream_kdot.elf: $(STREAM_SRC)
	$(CC) $(FLAGS) -DUSE_KDOT -I$(BUILD)/stream -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_stream_kdot.map ../../pynqz2_riscv_flow/firmware/start.S stream_main.c stream_infer.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
# Layer 1 with kdot, layer 2 and readout in rtl/neuron_engine.v (ABI bit1).
$(BUILD)/keyword_stream_engine.elf: $(STREAM_SRC)
	$(CC) $(FLAGS) -DUSE_KDOT -DUSE_ENGINE -I$(BUILD)/stream -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_stream_engine.map ../../pynqz2_riscv_flow/firmware/start.S stream_main.c stream_infer.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
# Layers 1 and 2 and the readout in rtl/neuron_engine.v (CTRL bit2, ABI bit2; branch ALIF-Layer1).
$(BUILD)/keyword_stream_engine_l1.elf: $(STREAM_SRC)
	$(CC) $(FLAGS) -DUSE_KDOT -DUSE_ENGINE -DUSE_ENGINE_L1 -I$(BUILD)/stream -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_stream_engine_l1.map ../../pynqz2_riscv_flow/firmware/start.S stream_main.c stream_infer.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
.PHONY: all stream
# Verifier track: stage 1 (kdot + engine, or pure RV32IM) with the phoneme verifier, command 6.
# Headers: build/stream (export_model.py) and build/verifier (verifier_export.py).
VER_SRC := $(STREAM_SRC) verifier.c verifier.h $(BUILD)/verifier/verifier_data.h $(BUILD)/verifier/verifier_config.h
verifier: $(BUILD)/keyword_stream_verifier_engine.bin $(BUILD)/keyword_stream_verifier.bin
$(BUILD)/keyword_stream_verifier_engine.elf: $(VER_SRC)
	$(CC) $(FLAGS) -DUSE_KDOT -DUSE_ENGINE -DUSE_VERIFIER -I$(BUILD)/stream -I$(BUILD)/verifier -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_stream_verifier_engine.map ../../pynqz2_riscv_flow/firmware/start.S stream_main.c stream_infer.c verifier.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
$(BUILD)/keyword_stream_verifier.elf: $(VER_SRC)
	$(CC) $(FLAGS) -DUSE_VERIFIER -I$(BUILD)/stream -I$(BUILD)/verifier -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_stream_verifier.map ../../pynqz2_riscv_flow/firmware/start.S stream_main.c stream_infer.c verifier.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
# Cascade: stage 1 proposes at CASCADE_T1, the verifier confirms (kdot + engine); LED on confirmed detections.
cascade: $(BUILD)/keyword_stream_cascade_engine.bin $(BUILD)/keyword_stream_cascade_engine_l1.bin
$(BUILD)/keyword_stream_cascade_engine.elf: $(VER_SRC)
	$(CC) $(FLAGS) -DUSE_KDOT -DUSE_ENGINE -DUSE_VERIFIER -DCASCADE -I$(BUILD)/stream -I$(BUILD)/verifier -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_stream_cascade_engine.map ../../pynqz2_riscv_flow/firmware/start.S stream_main.c stream_infer.c verifier.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
$(BUILD)/keyword_stream_cascade_engine_l1.elf: $(VER_SRC)
	$(CC) $(FLAGS) -DUSE_KDOT -DUSE_ENGINE -DUSE_ENGINE_L1 -DUSE_VERIFIER -DCASCADE -I$(BUILD)/stream -I$(BUILD)/verifier -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_stream_cascade_engine_l1.map ../../pynqz2_riscv_flow/firmware/start.S stream_main.c stream_infer.c verifier.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
.PHONY: verifier cascade
# KX extension of kdot_pcpi (rtl/kdot_pcpi.v KX = 1, ABI bit3): the verifier and cascade with
# -DUSE_KX (firmware/verifier.c). KXFLAGS passes e.g. -DKX_LEVEL=1 for stage-by-stage measurements.
KXFLAGS ?=
verifier_kx: $(BUILD)/keyword_stream_verifier_engine_kx.bin
$(BUILD)/keyword_stream_verifier_engine_kx.elf: $(VER_SRC)
	$(CC) $(FLAGS) -DUSE_KDOT -DUSE_ENGINE -DUSE_VERIFIER -DUSE_KX $(KXFLAGS) -I$(BUILD)/stream -I$(BUILD)/verifier -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_stream_verifier_engine_kx.map ../../pynqz2_riscv_flow/firmware/start.S stream_main.c stream_infer.c verifier.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
cascade_kx: $(BUILD)/keyword_stream_cascade_engine_kx.bin $(BUILD)/keyword_stream_cascade_engine_l1_kx.bin
$(BUILD)/keyword_stream_cascade_engine_kx.elf: $(VER_SRC)
	$(CC) $(FLAGS) -DUSE_KDOT -DUSE_ENGINE -DUSE_VERIFIER -DCASCADE -DUSE_KX $(KXFLAGS) -I$(BUILD)/stream -I$(BUILD)/verifier -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_stream_cascade_engine_kx.map ../../pynqz2_riscv_flow/firmware/start.S stream_main.c stream_infer.c verifier.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
$(BUILD)/keyword_stream_cascade_engine_l1_kx.elf: $(VER_SRC)
	$(CC) $(FLAGS) -DUSE_KDOT -DUSE_ENGINE -DUSE_ENGINE_L1 -DUSE_VERIFIER -DCASCADE -DUSE_KX $(KXFLAGS) -I$(BUILD)/stream -I$(BUILD)/verifier -T linker.ld -Wl,--build-id=none,-Map=$(BUILD)/keyword_stream_cascade_engine_l1_kx.map ../../pynqz2_riscv_flow/firmware/start.S stream_main.c stream_infer.c verifier.c runtime.c -o $@ -lgcc
	$(CROSS)size $@
.PHONY: verifier_kx cascade_kx
