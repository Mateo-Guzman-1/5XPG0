# 5XPG0 — Group 2 keyword spotter (brief for Claude Code sessions)

The user (Pedro) is a beginner with git, VS Code and hardware tools: explain steps in
plain language, say what a long command will do before running it, and report real
numbers only.

## What this project is
A spiking neural network (SNN) detects the word **"yes"**. The PC records the microphone,
turns each 1 s window (every 250 ms) into a 24×32 mel spectrogram (768 bytes,
`code/snn_keyword/features.py`) and sends it over TCP to a **PYNQ-Z2**. There, a
**PicoRV32 RISC-V soft core** in the FPGA runs the integer SNN from BRAM and lights LED0 for
1 s on a detection. Details: `code/snn_keyword/README.md`, `REPORT.md`, `JOURNAL.md`.

## Research programme (the current work)
Goal: how input encoding and network size trade off against detection quality and
latency on the RISC-V core. The plan is split into experiments E0–E8, one per session:

1. Read `code/snn_keyword/research/PLAN.md` — Parts A–B (primer, method rules) and the
   experiment the user asked for.
2. Read `code/snn_keyword/research/STATUS.md` — what is done, key numbers, open issues.
3. Do that experiment only. Finish with its report in `research/reports/`, raw data in
   `research/results/E<n>/`, an updated `STATUS.md`, and a local commit.

## Git rules
- Work only on branch **`Pedro`**. Never check out, commit to or push **`Mateo_AI`**
  (or any other branch).
- Commit locally when an experiment is finished. **Never push without asking the user.**
- Do not modify `code/snn_keyword/deploy/` (the release firmware/bitstreams) unless an
  experiment explicitly promotes a new model (E8), and then only after asking.

## Machine (Windows 11, this PC)
- Repo: `C:\Users\pedro\Desktop\Projects\Semiconductor Technology\5XPG0`
- Python venv: `.venv` at the repo root (`.venv/Scripts/python.exe`). The PC side needs
  `numpy` and `sounddevice`; training needs PyTorch (CPU build; the GPU is AMD, so no CUDA).
- RISC-V compiler (from Vitis 2026.1, reproduces the release firmware byte for byte):
  `C:/AMDDesignTools/2026.1/gnu/riscv/nt/bin/riscv64-unknown-elf-gcc.exe`
  (+ `-objcopy`, `-size`, `-objdump`). There is **no `make`, no WSL, no Verilator**; the
  flags are in `code/snn_keyword/firmware/Makefile` (use `research/tools/build_firmware.py`
  once E0 has created it).
- Vivado/xsdb 2026.1: `C:/AMDDesignTools/2026.1/Vivado/bin/` (only needed for RTL changes
  or the JTAG fallback).
- Shell tips: run `ssh`/`scp` commands through the **Bash tool** (PowerShell mangles nested
  quotes). The PowerShell tool may start in a different folder: use absolute paths.

## Board (PYNQ-Z2, PYNQ Linux 3.0.1)
- Direct Ethernet cable: board **10.43.0.1**, PC 10.43.0.2. `ssh pynq` works without a
  password (user `student`, passwordless sudo). Serial console: COM4, 115200 baud.
- Start/restart the demo server (also after every board power cycle):
  `code/snn_keyword/run_board.ps1` (`-Variant kdot|base|stop`). On the board this runs
  `~/snn_keyword/start_board.sh`, which loads the bitstream and starts `board_server.py`
  on port 5556 (log: `~/snn_keyword/server.log`).
- Under `sudo`, PYNQ needs `env XILINX_XRT=/usr BOARD=Pynq-Z2` (see `start_board.sh`).
- Live demo on the PC:
  `.venv/Scripts/python.exe code/snn_keyword/pc_keyword_demo.py 10.43.0.1`.
- Board memory map, mailbox and arithmetic: README "Interface and arithmetic". The
  model section is 0x18000–0x3bfff (144 KiB); the budget per 250 ms hop is 25 M cycles at 100 MHz.
- At the end of a session, restore the release demo on the board: `run_board.ps1`.
