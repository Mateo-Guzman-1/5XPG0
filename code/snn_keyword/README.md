# Group 2: "yes" keyword detection on PicoRV32

The completed local pipeline trains a spiking keyword detector on the full
Speech Commands v0.02 dataset, exports an integer model, executes its C
inference kernel on the PicoRV32 RTL, and builds the PYNQ-Z2 bitstream.
The PC computes one-second mel spectrograms every 250 ms; the RISC-V core
classifies them and a hardware countdown lights LED0 for exactly one second.

See [REPORT.md](REPORT.md), [presentation.pdf](presentation.pdf), and the
machine-readable [results](results/). These are simulation and implementation
results. The board and microphone have not been tested physically.

## What was measured

- Six GPU training runs: current/rate encoding, seeds 0/1/2, 35 epochs each,
  including ten epochs of quantization-aware training.
- Official speaker-disjoint split: 84,843 training, 9,981 validation,
  11,005 test clips; "yes" versus all other 34 words.
- Deployed current model: 90.15% precision, 85.20% recall, 87.61% F1.
- Worst of 40 RTL vectors: 5,981,498 cycles = 59.82 ms at 100 MHz.
- The best rate model took up to 715.52 ms and misses the 250 ms hop budget.
- Full-test native C and RTL stress vectors match the integer oracle exactly.
- Zero detections on 398 separate background-noise windows; this is not a
  continuous-speech false-accepts-per-hour measurement.

## Local environment

From the repository root on Windows (the environment used for this release):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r code/snn_keyword/requirements.txt
wsl -d Ubuntu -u root -- apt-get install -y gcc g++ make gcc-riscv64-unknown-elf verilator iverilog
```

The cross-compiler executable is `riscv64-unknown-elf-gcc`.
CUDA training runs natively on Windows; compilation
and Verilog simulation run in WSL. Native Linux works with the same Python
scripts and the Bash scripts directly. `setup_venv.ps1`/`.sh` remain available
for a separate environment inside this folder; install CUDA PyTorch first if
using that environment for GPU training.

## Verify the bundled release without training data

```powershell
.venv/Scripts/python.exe code/snn_keyword/export_model.py code/snn_keyword/deploy/model.npz
wsl -d Ubuntu -- bash code/snn_keyword/sim/build.sh
wsl -d Ubuntu -- bash code/snn_keyword/sim/unit.sh
.venv/Scripts/python.exe -m pytest code/snn_keyword/tests -q
.venv/Scripts/python.exe code/snn_keyword/verify.py --release-only
```

The release-only check uses the 40 bundled feature vectors; the original
full-test verification evidence remains in `results/verification.json`.
The RTL run executes a complete 100,000,000-cycle LED pulse; allow a few
minutes. It does not contact a PYNQ board.

## Reproduce all training and experiments

```powershell
.venv/Scripts/python.exe code/snn_keyword/prepare_data.py
.venv/Scripts/python.exe code/snn_keyword/train_keyword_snn.py
.venv/Scripts/python.exe code/snn_keyword/evaluate.py
wsl -d Ubuntu -- bash code/snn_keyword/sim/build.sh
.venv/Scripts/python.exe code/snn_keyword/verify.py
.venv/Scripts/python.exe code/snn_keyword/export_model.py code/snn_keyword/runs/current_seed2/model.npz --out code/snn_keyword/build/current
wsl -d Ubuntu -- make -C code/snn_keyword/firmware BUILD=../build/current
wsl -d Ubuntu -- code/snn_keyword/build/obj_dir/Vspike_soc code/snn_keyword/build/current/keyword.bin code/snn_keyword/build/vectors.bin code/snn_keyword/results/rtl_current.csv 2674
.venv/Scripts/python.exe code/snn_keyword/select_deployment.py
wsl -d Ubuntu -- bash code/snn_keyword/sim/build.sh
.venv/Scripts/python.exe code/snn_keyword/verify.py
.venv/Scripts/python.exe code/snn_keyword/robustness.py
```

The explicit current model and threshold above reproduce this release's
fixed seeds. If changing training settings, choose the current run with the
best **validation** F1 and use its `decision_threshold` from `training.json`.
Selection must not use test accuracy. Download is about 2.43 GB; extracted
audio, caches, CUDA packages, and Vivado output need substantially more space.
Dataset SHA256 is checked. Raw audio and training checkpoints stay ignored;
six exported integer models and training histories are retained in `results/`.

## Build the FPGA image

Vivado **2025.2** with the XC7Z020 device was used successfully locally.
The earlier claim that this design requires only 2024.1 is not applicable
to this installed toolchain. Run from `code/pynqz2_riscv_flow/vivado`:

```powershell
& C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat -mode batch -source build.tcl -tclargs ../../snn_keyword/build/vivado_rebuild ../../snn_keyword/deploy/keyword.bit
```

Use a fresh project directory on each build. The Tcl script emits routed
timing, utilization, and DRC reports. This bitstream adds the LED duration
register, fixes AXI read/write arbitration and byte enables, and exposes
the Zynq DDR/fixed I/O ports. Do not use the original `spike_top.bit` for
this demo: the server requires hardware ABI version 2.

## Custom instruction: `kdot` dot-product coprocessor

The input layer (64 × 768 int16 × uint8 multiply-accumulates) was about
99% of inference time. Every CPU read waits on the BRAM bus, so the
pure-RV32IM loop costs about 120 cycles per MAC. `rtl/kdot_pcpi.v`
(in `pynqz2_riscv_flow`) attaches to PicoRV32's PCPI port and adds two
custom-0 (opcode `0x0B`) R-type instructions:

| Instruction | Encoding | Effect |
|---|---|---|
| `klen x0, rs1, x0` | funct3=1 | length register ← rs1 (elements, multiple of 4) |
| `kdot rd, rs1, rs2` | funct3=0 | rd ← Σ int16 w[rs1+2i] · uint8 x[rs2+i], 32-bit wrap |

The core stalls on `kdot`. Meanwhile the unit borrows BRAM port B, streams
one input word and two weight words per four elements, and keeps a
registered DSP multiply stage and an accumulate stage. The result is
bit-exact with the C loop. `firmware/Makefile` builds both
`keyword.bin` (pure RV32IM) and `keyword_kdot.bin` (`-DUSE_KDOT`).
The fabric advertises the unit through ABI bit 0 (`0x00020001`).
The baseline image still runs unchanged on the new fabric.

RTL simulation, 40 verification vectors, same SoC:

| Firmware | Cycles (max) | Latency at 100 MHz | Scores vs oracle |
|---|---|---|---|
| `keyword.bin` (RV32IM) | 5,981,498 | 59.81 ms | 40/40 exact |
| `keyword_kdot.bin` | 267,370 | 2.67 ms | 40/40 exact |

That is a 22.4× worst-case speedup. What remains is mostly the LIF time
loop and the readout. On the physical PYNQ-Z2 (loaded over JTAG, see
`results/board_jtag_kdot.csv`), all 40 vectors match the oracle, and
scores, spikes and cycle counts are identical to RTL simulation.
Implementation meets timing (WNS +0.745 ns at 100 MHz). The unit costs
about 180 LUTs and 150 flip-flops over the baseline. `deploy/keyword_kdot.bit` is the matching bitstream;
the original `keyword.bit`/`keyword.bin` release is unchanged.

## PC simulation demo

In one terminal:

```powershell
.venv/Scripts/python.exe code/snn_keyword/board_server.py --simulate
```

In another terminal:

```powershell
.venv/Scripts/python.exe code/snn_keyword/pc_keyword_demo.py 127.0.0.1 --wav path/to/yes.wav
# Or use the actual microphone:
.venv/Scripts/python.exe code/snn_keyword/pc_keyword_demo.py 127.0.0.1
```

WAV input must be mono PCM16 at 16 kHz. The WAV command classifies one
one-second window, padding/truncating as needed. The microphone uses a
bounded rolling buffer and sends one-second windows with a 250 ms hop.
The callback only queues audio; feature extraction and networking happen
on the main thread. Responses contain scores, detection, and hidden spikes.
Simulation reports zero cycles; only RTL/board execution measures cycles.
Live microphone mode uses "2 of 3" confirmation (below); `--single`
restores per-window decisions. WAV mode always classifies a single window.

## Live confirmation and confusable words

**"2 of 3" confirmation.** A real "yes" spans several overlapping 250 ms-hop
windows, while most false accepts fire in only one. Protocol mode 1 maps to
firmware command 3. A window then counts as a detection only if it and one
of the two previous stream windows (within 750 ms, by the fabric timer)
reach the model's `stream_threshold`. Only confirmed detections light LED0.
`detected` bit 0 is the confirmed decision and bit 1 is the window alone.
Command 1 (single window: WAV mode and all verification) is unchanged.
`tune_stream.py <model.npz>` picks the stream threshold on validation
clips only. It places each clip in 2.25 s of background noise, cuts
consecutive 250 ms-hop windows, and chooses the lowest threshold with at
most 0.3% confirmed false accepts. `export_model.py` compiles it as
`MODEL_STREAM_THRESHOLD`.

Held-out test clips, placed the same way, over consecutive 250 ms windows:

| Release model (`deploy/model.npz`) | "yes" detected | Other words accepted |
|---|---|---|
| Single window, threshold 2674 | 74.5% | 1.07% |
| 2 of 3, stream threshold 1553 | 67.3% | 0.13% |

**Confusable endings.** The release model also fires on "yeets", "yets",
"pizza", "eats", "yetch" and "yeah". Speech Commands has no such words,
and its clips are centred while the demo slides its window.
`augment.py` edits real "yes" recordings, keeping their official splits,
into negatives: /t/ closure plus burst before the /s/ ("yets"), /ʃ/-shifted
endings ("yetch", "yesh"), a removed /s/ ("yeh"), and window-edge cuts.
It also makes positives with the full word at random positions.
`train_keyword_snn.py --aug-fraction 0.3` (default) mixes them into training.

The trial model (current encoding, seed 2) was checked against synthesized
Windows voices that were never used in training, at the real 250 ms hop:

| Share of utterances detected | "yes" | yeets/yets/yetz | pizza(s) | eats/its | ch words | yeah |
|---|---|---|---|---|---|---|
| Release model, single window | 91% | 75% | 31% | 66% | 26% | 33% |
| Release model, 2 of 3 | 77% | 53% | 11% | 44% | 5% | 15% |
| Augmented model, 2 of 3 | 56% | 12% | 3% | 7% | 1% | 3% |

On real held-out speakers the augmented model with 2 of 3 detects 60.1% of
"yes" and accepts 0.50% of other words. It trades recall for rejecting
these confusions and is not yet the release model.

**Finer time bins do not help this network.** 64 time bins (about 16 ms,
1536-byte input, 48 or 56 hidden neurons to fit BRAM) scored below 32 bins
on validation F1 (at most 0.819 against 0.837) and on live recall,
including with 60 epochs. The dense layer has no time-shift invariance, so
twice the weights per neuron add variance without new usable structure. A
time-convolutional front layer is the next step. The firmware already
publishes its input size (mailbox word 14), and the RTL harness and JTAG
relay adapt to it.

## When the PYNQ becomes available

No board upload has been attempted. Copy this folder's `deploy/`,
`board_server.py`, `protocol.py`, and `features.py` to the board. Use the
PYNQ environment (which already provides NumPy) to load the released image:

```bash
sudo /usr/local/share/pynq-venv/bin/python3 -c "from pynq import Bitstream; Bitstream('deploy/keyword.bit').download()"
sudo /usr/local/share/pynq-venv/bin/python3 board_server.py --bind 0.0.0.0 --firmware deploy/keyword.bin
```

Then run `pc_keyword_demo.py <board-ip>` on the PC. The server validates
the fabric magic, ABI version, and advertised clock, holds the CPU in reset,
clears BRAM, loads the complete image (including weights), and releases reset.
It also sets and reads back the physical FCLK0 at 100 MHz while the core is
held in reset, using [PYNQ Clocks](https://pynq.readthedocs.io/en/latest/pynq_package/pynq.ps.html).
An inference timeout or trap stops the core and terminates the server,
preventing a new frame from overwriting input still in use.
Each positive single-window request, or each confirmed stream window,
retriggers LED0 for one second from that detection.
Use the lab network; the small demo protocol has no authentication.

The remaining physical checks are Ethernet connectivity, actual clock setup,
microphone behavior, LED observation, and board timing. Do not replace
simulation results with claims of board measurement.

## Interface and arithmetic

Frontend: 16 kHz, periodic Hann 400 samples, hop 160, FFT 512; 24 normalized
HTK mel triangles from 80 to 7600 Hz; log power clipped to [-80,0] dB;
32 mean-pooled time bins; mel-major 768-byte unsigned input. Exactly the
same function is used for training data, WAV input, and live audio.

Weights are signed Q10 int16, biases and accumulators int32. For constant
current, `drive = trunc(sum(weight * input)/255) + bias` is computed once.
Each of 12 steps performs `v = trunc(7*v/8) + drive`; `v >= 1024` emits
a spike and immediately subtracts 1024. A linear accumulated readout sums
hidden spike counts. The score margin is compared with threshold 2674
(stream windows: 1553, see above).
Rate encoding uses a deterministic integer phase accumulator per input.

| RISC-V address | Purpose |
|---|---|
| 0x00000–0x0ffff | Code and working globals |
| 0x10400 | Mailbox: request/ack sequence, command, length, result; words 12–15 = magic, threshold, input bytes, stream threshold |
| 0x10800–0x10aff | 768 input bytes |
| 0x18000–0x3bfff | Read-only model section (98,824 bytes used) |
| 0x3c000–0x3ffff | Reserved 16 KiB stack |
| 0x10002004 | LED0 countdown duration in fabric clock cycles |

The PS maps BRAM at 0x40000000 and syscon at 0x40040000. Syscon offset
0x1c returns 0x00020000, or 0x00020001 when the fabric includes `kdot`.
TCP framing is versioned and length-bounded; the header's 16-bit mode field
(0 single, 1 stream) reads as 0 for clients using the original 32-bit
length. See `protocol.py`. Firmware publishes scores before acknowledging the sequence.
Only one request may be outstanding; frames stay owned by the host until ack.
