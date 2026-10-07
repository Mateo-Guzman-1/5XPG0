# Group 2 - single-keyword detection SNN

This directory contains the deployed reference path for detecting **"Sheila"**
from Google Speech Commands v2:

```text
PC microphone/WAV
  -> 8x16 log-frequency feature map (40-5120 Hz, 1.0 s, 8 bit)
  -> uint8 request over ZeroMQ/TCP
  -> PYNQ Linux bridge
  -> dual-port BRAM + mailbox
  -> quantized two-layer LIF SNN on PicoRV32
  -> decision reply + board LEDs on for one second
```

The network really executes on the RISC-V soft core. The ARM processing system
only terminates Ethernet, copies 128 input bytes into BRAM, and relays the
mailbox result.

## Current reference result

The data preparation keeps every v2 `sheila` clip and a deterministic,
label-balanced sample from all other command directories. The official
speaker-disjoint split contains 9,606 training, 2,704 validation, and 3,212
test clips. Across three floating-model seeds, the selected frontend reached
**98.27% mean test accuracy**. The deployed seed-0 integer model reached
**98.07% test accuracy**, **85.71% F1**, and **93.27% balanced accuracy**.

On synthetic additive noise, the deployed integer recurrence reached **97.38%**
accuracy at 10 dB SNR and **96.73%** at 5 dB SNR. It produced **0/400** false
accepts on generated background and silence, including **0/100** exact-silence
frames.

The PYNQ-Z2 reproduced the Python integer spike counts for **40/40** exported
golden vectors. Mean PicoRV32 inference latency is **10.39 ms** at 100 MHz;
mean Ethernet request/reply latency was **13.72 ms** on the test connection.
The quantized parameters occupy **6,440 bytes**, the input frame 128 bytes,
and the complete firmware binary 8,192 bytes.

The current research plan and full results are in
[`research/SHEILA_V2_PLAN.md`](research/SHEILA_V2_PLAN.md) and
[`research/SHEILA_V2_REPORT.md`](research/SHEILA_V2_REPORT.md).

## Files

| File | Purpose |
|---|---|
| `audio_features.py` | Shared WAV/microphone preprocessing and packet format |
| `research/prepare_speech_commands_v2.py` | Reproducible v2 Sheila subset preparation |
| `research/frontend_sweep.py` | Encoding sweep and three-seed confirmation |
| `research/export_frontend_candidate.py` | Integer validation and C-header export |
| `train_keyword_snn_clip.py` | Legacy Mini Speech Commands training baseline |
| `pc_keyword_demo_zmq.py` | Live microphone or WAV client |
| `install_keyword_demo.sh` | Builds if needed, deploys, loads, and starts the board chain |
| `../pynqz2_riscv_flow/firmware/keyword_main.c` | Integer LIF inference and one-second LED control |
| `../pynqz2_riscv_flow/firmware/keyword_model.h` | Generated quantized weights |
| `../pynqz2_riscv_flow/host/keyword_bridge.py` | Board-side TCP/BRAM/mailbox bridge |

## Run the checked-in demo

Create the PC environment once (PowerShell):

```powershell
Set-Location code\snn_keyword
.\setup_venv.ps1
```

Deploy from Git Bash or MSYS2. The prebuilt `keyword.bin` means deployment
does not require a RISC-V compiler:

```bash
cd code/snn_keyword
./install_keyword_demo.sh 192.168.2.99
```

Then run the live PC microphone client in PowerShell and say **"Sheila"**:

```powershell
Set-Location code\snn_keyword
.\.venv\Scripts\python.exe .\pc_keyword_demo_zmq.py 192.168.2.99
```

For a deterministic test, send one WAV file:

```powershell
.\.venv\Scripts\python.exe .\pc_keyword_demo_zmq.py 192.168.2.99 --wav .\sample.wav
```

The reply prints both output-neuron spike counts, the PicoRV32 cycle count,
inference milliseconds, and the LED register. A positive result should report
`led=0x3ff`; after one second the firmware returns to the idle `led=0x1`.

Board service checks:

```bash
ssh xilinx@192.168.2.99 'cd /home/xilinx/snn_keyword && ./keyword_service.sh status'
ssh xilinx@192.168.2.99 'cd /home/xilinx/snn_keyword && ./keyword_service.sh log'
ssh xilinx@192.168.2.99 'cd /home/xilinx/snn_keyword && ./keyword_service.sh stop'
```

## Reproduce, retrain, and rebuild

Prepare the v2 subset, run the controlled encoding sweep, and export the
selected seed-0 8-band model. Data, caches, and checkpoints are git-ignored.

```powershell
Set-Location code\snn_keyword
.\.venv\Scripts\python.exe .\research\prepare_speech_commands_v2.py `
    --archive .\data\speech_commands_v0.02.tar.gz `
    --output .\data\speech_commands_v2_sheila_subset
.\.venv\Scripts\python.exe .\research\frontend_sweep.py `
    --dataset .\data\speech_commands_v2_sheila_subset `
    --keyword sheila `
    --cache-dir .\data\frontend_sweep_cache_sheila_v2 `
    --output .\research\results\frontend_sweep_sheila_v2.json
.\.venv\Scripts\python.exe .\research\export_frontend_candidate.py `
    .\research\results\frontend_sweep_sheila_v2_models\b8_f8_hz40-5120_t1000_seed0.pt `
    --dataset .\data\speech_commands_v2_sheila_subset `
    --keyword sheila `
    --feature-cache .\data\frontend_sweep_cache_sheila_v2\b8_f8_hz40-5120_t1000.npy `
    --output .\research\results\sheila_v2_deployment
```

Copy the exported `keyword_model.h` into the firmware directory, then rebuild
with an RV32-capable GNU toolchain:

```bash
cd code/pynqz2_riscv_flow/firmware
make keyword.bin
```

The export uses signed int8 weights, int32 membrane/current state, Q8 leak
(`230/256`), 48 hidden LIF neurons, two output LIF neurons, and 16 simulation
steps. The first-layer current is constant over those steps and is computed
once per inference, which preserves the recurrence while avoiding repeated
dot products. The audio frontend rejects RMS levels at or below `0.003` and
uses a logarithmic activity factor up to RMS `0.04`, so near-silence is not
peak-normalised into a full-strength spectrogram.

## Interface contract

- Feature shape: 8 frequency bands by 16 time bins, row-major, uint8.
- Network transport: request/reply ZeroMQ on board TCP port 5556.
- BRAM input window: RISC-V/BRAM offset `0x22000`, 128 bytes used.
- Mailbox command: `MB_CMD_CLASSIFY` (`8`), returning decision, two spike
  counts, and inference cycles.
- Positive class: output neuron 1 must emit at least two spikes and lead the
  non-keyword neuron by at least two spikes. Passing that confidence rule
  lights all ten LEDs for one second.

Keep `audio_features.py`, the trainer, and the live client together. Training
and inference with different feature extraction is a silent but serious model
mismatch.

## Recommended experiment plan

1. Measure precision, recall, false accepts per hour, and missed detections on
   continuous audio; add temporal debounce on top of the existing two-spike
   confidence margin.
2. Compare direct current input against rate, latency, and delta encoding at a
   fixed memory/latency budget. Report several seeds and confidence intervals.
3. Implement a packed 4-bit input ABI, then remeasure memory and Ethernet
   latency; the current ABI cannot realize the theoretical 64-byte frame.
4. Sweep hidden width, time steps, and int8/int16 weight quantization; measure board
   cycles and memory alongside accuracy.
5. Replace or supplement generated noise with recorded room, fan, keyboard,
   music, and overlapping-speech backgrounds plus locally recorded speakers.
