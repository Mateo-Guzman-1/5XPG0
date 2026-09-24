# Group 2 - single-keyword detection SNN

This directory now contains a complete reference path for detecting **"yes"**:

```text
PC microphone/WAV
  -> 16x16 log-frequency feature map
  -> uint8 request over ZeroMQ/TCP
  -> PYNQ Linux bridge
  -> dual-port BRAM + mailbox
  -> quantized two-layer LIF SNN on PicoRV32
  -> decision reply + board LEDs on for one second
```

The network really executes on the RISC-V soft core. The ARM processing system
only terminates Ethernet, copies 256 input bytes into BRAM, and relays the
mailbox result.

## Current reference result

The checked-in model was trained with seed 0 on a balanced subset of
TensorFlow Mini Speech Commands: 800 `yes` clips and 800 clips sampled from
the other seven commands. On its fixed 320-clip validation split it reached:

- floating SNN accuracy: **92.50%**;
- exported integer SNN accuracy: **91.56%**;
- floating/integer prediction agreement: **97.19%**.

A full replay of those 320 held-out feature maps through the physical PYNQ-Z2
matched the Python integer model's output spike counts for **320/320** frames.
Mean PicoRV32 inference time was **17.56 ms** at 100 MHz.

These figures establish a reproducible baseline, not a final scientific
claim. The split is clip-random rather than speaker-independent, only the
eight-command mini dataset is used, and continuous false accepts have not yet
been measured.

## Files

| File | Purpose |
|---|---|
| `audio_features.py` | Shared WAV/microphone preprocessing and packet format |
| `train_keyword_snn.py` | Real-data training, integer validation, C-header export |
| `pc_keyword_demo.py` | Live microphone or WAV client |
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

Then run the live PC microphone client in PowerShell and say **"yes"**:

```powershell
Set-Location code\snn_keyword
.\.venv\Scripts\python.exe .\pc_keyword_demo.py 192.168.2.99
```

For a deterministic test, send one WAV file:

```powershell
.\.venv\Scripts\python.exe .\pc_keyword_demo.py 192.168.2.99 --wav .\sample.wav
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

## Retrain and rebuild

Training automatically downloads the approximately 182 MB Mini Speech
Commands archive on first use. Data and run artifacts are git-ignored.

```powershell
Set-Location code\snn_keyword
.\.venv\Scripts\python.exe .\train_keyword_snn.py `
    --keyword yes --epochs 12 --max-per-class 800
```

This writes `runs/keyword_snn.pt`, validation examples, and overwrites the
firmware's generated `keyword_model.h`. Rebuild with an RV32-capable GNU
toolchain:

```bash
cd code/pynqz2_riscv_flow/firmware
make keyword.bin
```

The export uses signed int8 weights, int32 membrane/current state, Q8 leak
(`230/256`), 48 hidden LIF neurons, two output LIF neurons, and 16 simulation
steps. The first-layer current is constant over those steps and is computed
once per inference, which preserves the recurrence while avoiding repeated
dot products.

## Interface contract

- Feature shape: 16 frequency bands by 16 time bins, row-major, uint8.
- Network transport: request/reply ZeroMQ on board TCP port 5556.
- BRAM input window: RISC-V/BRAM offset `0x22000`, 256 bytes.
- Mailbox command: `MB_CMD_CLASSIFY` (`8`), returning decision, two spike
  counts, and inference cycles.
- Positive class: output neuron 1; a strict `keyword > not-keyword` spike-count
  comparison lights all ten LEDs for one second.

Keep `audio_features.py`, the trainer, and the live client together. Training
and inference with different feature extraction is a silent but serious model
mismatch.

## Recommended experiment plan

1. Replace the clip-random split with a speaker-disjoint train/validation/test
   manifest and reserve the test set before tuning.
2. Measure precision, recall, false accepts per hour, and missed detections on
   continuous audio; tune a confidence margin and temporal debounce rather
   than relying only on `argmax`.
3. Compare direct current input against rate, latency, and delta encoding at a
   fixed memory/latency budget. Report several seeds and confidence intervals.
4. Sweep hidden width, time steps, and int8/int16 quantization; measure board
   cycles and memory alongside accuracy.
5. Add background noise, silence, unknown-word, and locally recorded samples
   before treating the demo as robust.
