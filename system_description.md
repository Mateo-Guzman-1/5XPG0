# System description: "sheila" keyword detection with a spiking neural network on the PYNQ-Z2

This document describes the Group 2 keyword detector block by block. For each
block it gives:
- what the block does;
- its components;
- its inputs and outputs;
- the alternatives for it, with their pros and cons.

It describes the system on branch `merge-all` as of 2026-10-01 (commit `9da91a4`).
Paths are relative to the repository root on that branch. On `main` the code is not
present.

Numbers are measured in this repository unless they are marked *est.* (an estimate).
Their sources:
- `code/snn_keyword/JOURNAL.md`, cited as JOURNAL n;
- `code/snn_keyword/results/`;
- `MERGE_SUMMARY.md`.

How the repository is organised around these blocks is described in
`Repo-Architecture.md` (branch `merge-all`).

## 1. Requirements

| Requirement | Target | Source |
|---|---|---|
| Live "sheila" detected (test clips in noise, 250 ms hop) | ≥ 85% | `code/snn_keyword/IMPLEMENTATION_PLAN.md` |
| Other words accepted | ≤ 0.2% | same |
| False alarms on running speech (18-19 h of negatives) | ≤ 2 per hour | same |
| Compute per 250 ms of audio (stage 1) | ≤ 25 ms | same |
| Speakers, microphones, rooms | unseen ones; no training on the user's voice | `code/snn_keyword/OPTIMIZATION_REPORT.md` |
| Output | LED0 on for 1 s per detection | project task |
| Platform | an SNN on a RISC-V soft core on the PYNQ-Z2 (XC7Z020: 53,200 LUTs, 140 BRAM36, 220 DSP slices) | `ProjectPitch/5XPG0_Projects_ES_Group.pdf` |
| Correctness | Python integer model, native C, RTL and board agree bit for bit | project rule |

## 2. Top-level block diagram

```
OFFLINE (PC + GPU)
  B0 Training & export ──► int8 weights, thresholds, C headers, engine tables, test vectors
                               │ compiled into the firmware image, loaded once at start
RUNTIME                        ▼
PC         mic ─► B1 Audio capture ─PCM16 16 kHz─► B2 Front end ─24 B per 10 ms─► B3 Client
                                                       TCP: 25 frames (600 B) per 250 ms ↓ ↑ 40 B reply
PS (ARM)   B4 board_server.py: loads bitstream, clock, firmware; relays the mailbox
                               │ AXI4-Lite, 0x4000_0000
PL         B5 ps_if + syscon ──port A──► 256 KB BRAM: code · mailbox · frames · model
(100 MHz)                                    │ port B
           B6 PicoRV32 RV32IM + firmware ─┬─ PCPI ─► B7 kdot: layer 1 (24 → 128 ALIF)
                                          ├─ MMIO ─► B8 neuron engine: delays 0-31 frames,
                                          │          layer 2 (128 ALIF, recurrent), readout → score
                                          ├─ B9 decision (firmware): score ≥ t1 → proposal
                                          ├─ B10 verifier (firmware + kdot): GRU on last 1.5 s ≥ t2
                                          └─ MMIO ─► B11 LED countdown ─► LED0 (1 s)
Across all blocks: B12 verification (integer oracle = native C = Verilator RTL = board)
```

Stage 1 is B7, B8 and B9: a streaming SNN that proposes detections. Stage 2 is B10:
a verifier that confirms them. B9 and B10 are firmware on the B6 core. B7 and B8
are hardware attached to it.

## 3. The blocks

In every table, the option marked **(current)** is the one built on `merge-all`.

### B0 · Offline training and export (PC, GPU)

Produces every number the board uses.

- **Components:**
  - **Data:**
    - Speech Commands v2: 2,022 "sheila" clips, 1,606 of them for training;
    - MSWC, for near-miss words such as "she", "sheep" and "shield";
    - LibriSpeech, for negatives and an 18-19 h false-alarm stream;
    - 55,000 Piper TTS utterances;
    - MUSAN noise and room impulse responses.
  - **Augmentation**, applied on the fly: microphone EQ and band limits, rooms,
    noise, speed changes, SpecAugment.
  - **Training:** surrogate-gradient training through time. Stage 1 learns 35
    words; stage 2 learns 3 classes ("sheila", `_silence_`, `_unknown_`). The
    learnable delays are rounded to single taps.
  - **Quantization:** quantization-aware training to int8.
  - **Threshold choice:** on validation data only (`stream_select.py`,
    `verifier_cascade.py`).
  - **Exporters:** `export_model.py` and `verifier_export.py`.
- **In:** labelled audio.
- **Out:**
  - the int8 model (`.npz`);
  - thresholds: t1 = 3785 and t2 = 1524 for the cascade, 18,462 for stage 1 alone;
  - C headers;
  - the engine's sparse synapse tables;
  - test vectors.

| Alternative | Pros | Cons |
|---|---|---|
| **Surrogate-gradient training through time (current)** | Best accuracy for ALIF neurons with learnable delays | GPU hours; long sequences need a lot of memory |
| ANN-to-SNN conversion | Reuses ANN tooling | Rate coding needs many time steps, which does not suit streaming audio |
| Local learning rules (STDP, e-prop) | Learning on the chip becomes possible | Far lower accuracy for keyword spotting; on-chip learning is not needed for a detector that must not adapt to one user |
| **Isolated-word positives (current)** | Real recordings from many speakers | Few positives; the model knows only the word said on its own, so it detects 6% of "sheila" in the middle of a sentence (JOURNAL 34) |
| The keyword inside synthetic sentences | Attacks the main open problem at its source | Synthetic voices differ from real ones; needs a new data path |
| **Quantization-aware int8 (current)** | No accuracy loss (JOURNAL 15) | — |
| int4 weights | Half the weight memory, and BRAM is the scarce resource (§5) | Some accuracy risk; the engine must unpack 4-bit weights |
| Distillation from a BC-ResNet-8 teacher | — | Tried: it cost 10-15 points (JOURNAL 10, 14) |

### B1 · Audio capture

- **Components:** the PC microphone; a `sounddevice` callback into a bounded ring
  buffer; WAV replay for tests (`pc_keyword_demo.py`).
- **In:** sound.
- **Out:** PCM16 mono at 16 kHz (32 KB/s).

| Alternative | Pros | Cons |
|---|---|---|
| **PC microphone (current)** | No hardware work; any microphone; reproducible WAV tests | The board cannot work alone. The microphone changes per PC; augmentation covers this (86.3% detected on held-out microphones). |
| PYNQ-Z2 on-board ADAU1761 codec (I²S data, I²C set-up) | A standalone device with one known microphone chain; no network in the loop | Needs an I²S receiver and the codec set-up, and B2 must move to the board. The plan defers it. |
| PDM MEMS microphone on a Pmod | Digital and cheap; close to a real edge product | Needs a decimation filter (CIC and FIR) in the fabric, and new hardware |

### B2 · Front end: features and spike encoding

- **Components** (`features.py`, one function for training and live use):
  - a 400-sample Hann window with a 160-sample hop (10 ms);
  - a 512-point FFT and its power spectrum;
  - 24 HTK mel bands from 80 to 7600 Hz;
  - a log, clipped to [−80, 0] dB and scaled to uint8.
- **In:** PCM16 at 16 kHz.
- **Out:** one 24-byte frame every 10 ms (2.4 KB/s).

*Where it runs:*

| Alternative | Pros | Cons |
|---|---|---|
| **PC, Python (current)** | Easy to change; training and live features come from one function and cannot drift apart | Needs a PC |
| ARM processor on the board (PS) | Floating point and NEON; removes the PC if B1 moves to the board | Still depends on Linux |
| PicoRV32 firmware (integer FFT) | Everything runs on the soft core | *est.* 50-100k cycles per frame, 5-10% of the core, competing with the verifier. Fixed-point features need retraining. |
| Fabric (FFT IP, mel accumulation, log table) | Deterministic, and frees the CPU; pairs with the codec | The most RTL effort; the fixed-point features must be matched bit for bit, and the model retrained |
| IIR filter bank ("cochlea") emitting spikes | Spike-native; no FFT; cheap in the fabric | Different features, so a full retrain |

*What it computes, and how it becomes spikes:*

| Alternative | Pros | Cons |
|---|---|---|
| **Log-mel, 24 bands, absolute dB (current)** | Measured best for this network | Depends on input level |
| PCEN or level normalisation (AGC) | Level and microphone invariance in theory | Tried: each fixed one level range and lost at normal level (JOURNAL 22, 25) |
| 8 bands | 3× smaller input; "sheila" carries no information above about 5 kHz | Tested only on the clip model (`code/snn_keyword/research/FRONTEND_SWEEP_REPORT.md`) |
| **uint8 values into a dense layer 1 that spikes (current)** | No encoder; layer 1 learns the encoding | Layer 1 needs multiply-accumulates, and it is now the main cost (B7) |
| Poisson rate coding (`poisson.v` is in the SoC) | Real spikes at the input | Needs many time steps (the rate window model took 715 ms per window); random, so bit-exact checks are hard |
| Delta / threshold-crossing events | Sparse; layer 1 becomes event-driven, with additions instead of multiplications | Retraining; loses the absolute level |

### B3 · Host link and protocol

- **Components:** the client `pc_keyword_demo.py` and the protocol `protocol.py`
  (ABI v3).
  - The 12-byte header holds the magic `KWS3`, a sequence number, the payload
    length and the mode (INFO, FRAMES or RESET).
  - The 40-byte reply holds the best and last score, cycles, spikes, detection
    bits, synaptic events and the frame of the detection.
  - The protocol also serves ABI v2, the "yes" window release.
- **In:** frames.
- **Out:** requests; replies come back.

| Alternative | Pros | Cons |
|---|---|---|
| **TCP over Ethernet (current)** | 7 ms round trip; the standard PYNQ path; bounded, versioned framing | PYNQ Linux must boot; no authentication, so lab networks only |
| ZeroMQ (the clip pipeline) | A simpler messaging library | 13.7 ms average round trip; a second ABI to maintain |
| JTAG relay through xsdb (fallback) | Works with no OS on the board; it was used when the SD card would not boot | 30-45 ms per hop; the cable is shared with Vivado |
| UART | No network or Linux needed | 115,200 baud ≈ 11.5 KB/s: enough for features, not for audio |
| **25 frames per 250 ms request (current)** | Few requests | The PC waits for 25 frames before it sends, so LED0 can light up to about 240 ms after the frame that detected |
| 5 frames per 50 ms request | Cuts that wait to about 40 ms | 5× more requests (still well within a 7 ms round trip) |

### B4 · Board software on the ARM (PS)

- **Components** (`board_server.py`, started by `start_board.sh`):
  - the PYNQ bitstream loader;
  - FCLK0 set and read back at 100 MHz;
  - the firmware loader: hold the core in reset, clear BRAM, load the image,
    release;
  - the mailbox handshake: sequence word MB[0], acknowledge MB[1];
  - timeout and trap handling;
  - the TCP server.

  All accesses are 32-bit. Byte-wise writes ran every request twice
  (JOURNAL 18b).
- **In:** the bitstream, the firmware image, TCP requests.
- **Out:** AXI reads and writes; replies.

| Alternative | Pros | Cons |
|---|---|---|
| **Python server on PYNQ Linux (current)** | Quick to write; PYNQ handles clocks and bitstreams | Linux must boot; Python timing jitter |
| C server on Linux (UIO, with a PL interrupt) | Lower jitter; no polling | More code |
| Bare-metal ARM program (Vitis standalone) | Boots in seconds; deterministic | Needs lwIP for networking; no PYNQ |
| PS outside the data path (PL only, with the codec) | Truly standalone | The PS is still needed to boot the PL, and B1 and B2 must move to the board |

### B5 · PS–PL bridge

- **Components** (`code/pynqz2_riscv_flow/rtl/ps_if.v`, `spike_top.v`):
  - PS7 `M_AXI_GP0`, an AXI interconnect, then `ps_if`, an AXI4-Lite slave;
  - a window onto the BRAM at 0x4000_0000;
  - syscon at 0x4004_0000: core reset, status, timer, clock rate, the magic `SKEL`
    and the LED state.

  The CPU reads the ABI version `0x00020003` (bit 0: kdot, bit 1: neuron engine).
- **In:** AXI transactions.
- **Out:** BRAM port A accesses; the core reset.

| Alternative | Pros | Cons |
|---|---|---|
| **Custom AXI4-Lite slave into the shared BRAM (current)** | Tiny; the host sees the core's memory directly | One word per transaction, and polling. Enough for 600 B per hop. |
| Xilinx AXI BRAM Controller IP | Standard; supports bursts | More block-design work for no gain at this data rate |
| AXI-Stream and DMA on an HP port into a FIFO | High bandwidth; interrupts | Far beyond the 2.4 KB/s needed |

### B6 · Controller SoC: PicoRV32, memory, firmware

- **Components** (`code/pynqz2_riscv_flow/rtl/spike_soc.v`,
  `code/snn_keyword/firmware/stream_main.c`):
  - **Core:** PicoRV32 RV32IM, with fast multiply, divide, the PCPI port and
    counters, but no barrel shifter.
  - **Memory:** 256 KB of true dual-port BRAM (64 BRAM36). Port A belongs to the
    PS; port B to the CPU, which `kdot` borrows. One wait state (`READ_WAIT = 1`).
  - **Peripherals:** SYSCTRL, TIMER, LED, the neuron engine at 0x1000_4000, and the
    Poisson generator (unused by this system).
  - **Firmware:** the mailbox loop; the network state stays on the core between
    requests.
  - **Memory map:**
    - code: 0x00000-0x0FFFF;
    - mailbox: 0x10400;
    - frames: 0x10800;
    - model: 0x18000-0x3BFFF (121 of 144 KB used by both stages);
    - stack: 16 KB.
- **In:** frames and commands in the mailbox.
- **Out:** results in the mailbox; engine and LED register writes.

| Alternative | Pros | Cons |
|---|---|---|
| **PicoRV32 (current)** | Small and verified; the PCPI port takes custom instructions | Several cycles per instruction; slow shifts, which the ALIF update uses heavily |
| PicoRV32 with `BARREL_SHIFTER = 1` | One parameter; speeds up the software ALIF update | *est.* a few hundred LUTs; timing must be re-checked |
| VexRiscv (5-stage pipeline) | Close to one instruction per cycle; custom-instruction plugins | SpinalHDL generator; new integration and verification |
| CV32E40P (CORE-V) with the Xpulp SIMD extension | Four int8 multiply-accumulates per instruction | Larger; SystemVerilog; more integration |
| Two cores: one for stage 1, one for the verifier | The verifier no longer stalls stage 1; matches the brief's "multi-core scaling" | The BRAM has two ports and the PS uses one, so a second memory or an arbiter is needed, plus synchronisation |
| Separate instruction and data BRAM | No contention between fetches and data accesses | More BRAM; the firmware layout changes |

### B7 · Layer 1: dense 24 → 128 ALIF neurons

- **Components:**
  - The `kdot` coprocessor on the PCPI port
    (`code/pynqz2_riscv_flow/rtl/kdot_pcpi.v`). It uses the custom-0 opcode:
    `klen` sets the length, and `kdot` returns Σ int16 w × uint8 x at 3 cycles per
    4 multiply-accumulates. Cost: +180 LUTs, +147 flip-flops, +2 DSPs.
  - The ALIF update of each neuron, in software.
  - Writes of the spiking neurons' indices to the engine.
- **In:** a 24-byte frame, 24 × 128 int16 weights, the neuron state.
- **Out:** the indices of the layer-1 neurons that spiked.
- **Cost:**
  - With the engine in place, stage 1 takes about 40k cycles per frame (988,705
    per 25-frame hop), and about 37k of them are layer 1 (profile, JOURNAL 13).
  - The 3,072 `kdot` multiply-accumulates take only about 2.3k cycles.
  - **The rest is the software ALIF update and loop overhead**, about 270 cycles
    per neuron.

| Alternative | Pros | Cons |
|---|---|---|
| Plain RV32IM | No hardware | Many times slower on the dot products |
| **`kdot` custom instruction (current)** | 22× on the dense window model; the verifier reuses it | The CPU stalls during the instruction; the ALIF update stays in software |
| An ALIF custom instruction | Small RTL; targets the actual cost | Specific to this neuron model |
| Layer 1 inside the engine (16-lane multiply-accumulate, about 320 cycles per frame) | Stage 1 goes from 10 ms to *est.* about 1 ms per hop; the CPU is free | DSPs, RTL and a new bit-exact check |
| Spiking input (delta coding), so layer 1 becomes event-driven | One event-driven design throughout | Retraining; a front-end change |

### B8 · Neuron engine: delays, layer 2, readout

- **Components** (`code/pynqz2_riscv_flow/rtl/neuron_engine.v`):
  - a delay ring: 32 slots, each holding the list of layer-1 neurons that spiked
    in that frame;
  - a sparse synapse table (CSR) per (neuron, delay): offsets plus {int8 weight,
    target}, up to 16,384 synapses;
  - 128 accumulators, with a pipelined read-add-write and forwarding;
  - recurrent weights, 128 × 128 int8, read 16 lanes per clock;
  - a 5-stage ALIF pipeline, one neuron per clock: adaptation, threshold
    θ + (bq·a)>>8, leak by shift, int16 saturation, subtract-reset;
  - a leaky readout, with score = o[sheila] − max o[other];
  - registers: CTRL, STATUS, SPIKE, SCORE, SPIKES2, EVENTS, CYCLES, ID, table
    loading, CONFIG.
- **In:** the layer-1 spike indices of each frame and a start command; the tables
  at start-up.
- **Out:** an int32 score per frame, plus spike, event and cycle counts.
- **Cost:**
  - About 1.6k synaptic events per frame. That is 20× fewer operations than a
    dense equivalent: 40.6k against 829k per hop (JOURNAL 16).
  - The engine uses about 10-12k LUTs, 24 BRAM36 and 4 DSPs.
  - The whole design meets timing at 100 MHz with the default Vivado strategy
    (WNS +0.406 ns, JOURNAL 31).

*Architecture:*

| Alternative | Pros | Cons |
|---|---|---|
| Software on the PicoRV32 | No RTL | 77 ms per hop even with `kdot`, which leaves no time for a verifier |
| **Event-driven, one synaptic event per clock (current)** | The cost follows the spikes; small; bit-exact | The latency depends on activity, but stays far below the budget |
| Parallel neuron array (`code/pynqz2_riscv_flow/rtl_sketches/snn_layer.v`: 64 LIF neurons, row-wide weights) | 64× throughput per input event | No delays, adaptation or recurrence; not verified; uses the engine's address; that throughput is not needed |
| Dense, time-stepped multiply-accumulate array | Fixed latency; simple control | 20× more operations; ignores sparsity, which is the point of an SNN |
| High-level synthesis (HLS) from the existing C kernel | Faster to write; the C is already bit-exact | Less control over BRAM use and timing |

*Delays and neuron model:*

| Alternative | Pros | Cons |
|---|---|---|
| **Spike history plus a CSR table per delay (current)** | An exact delay per synapse | Two table reads per event |
| Dendritic ring: at spike time, acc[t + d][target] += w | One write per event; no history scan | 32 × 128 accumulator RAM; read-modify-write per event |
| One delay per neuron (axonal delay) | Much smaller tables | Less expressive; retraining |
| No delays, or LIF instead of ALIF | Simpler | The ablations put delays, adaptation and recurrence at 10-36 points each (JOURNAL 14) |

### B9 · Stage-1 decision (firmware)

- **Components:**
  - a threshold on the frame score, summed over the last W frames (W = 1 for
    "sheila");
  - a 1 s hold-off;
  - in the cascade, a low proposal threshold, t1 = 3785. Stage 1 alone needs
    18,462.
- **In:** the score of each frame.
- **Out:** a proposal (cascade) or a detection (stage 1 alone), and its frame.

| Alternative | Pros | Cons |
|---|---|---|
| **Threshold and hold-off (current)** | Simple; the threshold is chosen on validation data under the false-alarm rule | — |
| Sum of the last W scores | +5 points for "yes" with W = 20 (JOURNAL 19) | No gain for "sheila" |
| "2 of 3" windows | 8× fewer other words accepted for the window model (JOURNAL 5) | Adds latency; the verifier replaced it |
| A comparator in the engine | No CPU involvement | The firmware cost is already negligible |

### B10 · Stage-2 verifier (firmware + kdot)

- **Components** (`code/snn_keyword/firmware/verifier.c`):
  - a 1.5 s history of frames: 150 frames in a 7.2 KB double ring;
  - two GRU layers of 64, at 20 ms steps over 2 stacked frames;
  - int8 weights with a power-of-two scale per row; Q10 int16 activations;
  - a phoneme readout (SH IY L AH) and a keyword head, taking the maximum over
    time;
  - the confirmation threshold t2 = 1524.
- **In:** the last 1.5 s of frames when stage 1 proposes, about 2,200 times per
  hour of speech.
- **Out:** confirm or reject, which drives the LED.
- **Effect** (test split, JOURNAL 27 and 32):
  - live recall rises from 63.7% to 86.8%;
  - false alarms fall from 1.58 to 1.42 per hour;
  - near-miss "she-" words accepted fall from 23 to 6 of 183.
- **Cost:** 107 ms per check; the worst request is 117 ms per 250 ms.

| Alternative | Pros | Cons |
|---|---|---|
| No verifier | A pure SNN; 10 ms per hop | 63.7% recall, below the 85% target |
| **GRU verifier on the PicoRV32 (current)** | The only set-up that has met both false-alarm targets. It runs only on proposals (*est.* about 7% average load). | **Not spiking**, which weakens the claim that an SNN detects "sheila"; 117 ms bursts; it knows only the isolated word |
| Spiking verifier: a second ALIF network on the engine | The whole system stays an SNN | Untested; needs BRAM, of which 52 BRAM36 are left |
| Verifier on the ARM | *est.* about 10× faster | The detector is no longer on the RISC-V core |
| One larger stage-1 SNN trained on sentences | One model. It attacks the real limit: stage 1 proposes for only 39-65% of mid-sentence keywords (JOURNAL 34). | Data work; more engine memory |

### B11 · Output

- **Components:**
  - a hardware LED countdown at 0x1000_2004: 100,000,000 cycles is 1 s; writing it
    again restarts the countdown; CPU load does not affect it;
  - the detection bits in MB[11]: bit 0 confirmed, bit 1 stage 1 proposed, bit 2
    the verifier ran;
  - the reply to the PC.
- **In:** a confirmed detection.
- **Out:** LED0 on for 1 s; the reply fields.

| Alternative | Pros | Cons |
|---|---|---|
| **Hardware countdown (current)** | Exactly 1 s, even while the verifier runs | — |
| Software timer in the firmware | No RTL | Jitter while the core is busy in the verifier |
| LED driven from the ARM | No PL change | Linux latency |

### B12 · Verification chain (across all blocks)

- **Components:** the Python integer oracle (`model.py`, `verifier_model.py`),
  then native C (`verify_stream.py`), then the full SoC in Verilator
  (`verify_stream_rtl.py --engine`, `verify_verifier_rtl.py --cascade`), then the
  board over JTAG or Ethernet (`jtag/stream_board_test.tcl`).
- **Status:**
  - The oracle equals the RTL: 355 hops for stage 1 and 934 requests for the
    cascade.
  - Native C on the full test set and the board run have not yet been done for
    "sheila".

| Alternative | Pros | Cons |
|---|---|---|
| **Bit-exact integer oracle (current)** | Any mismatch is a bug; all four levels compare against one model | Every arithmetic change touches four implementations |
| Comparison with a float model within a tolerance | Easier | Hides off-by-one errors in shifts and saturation |
| cocotb test benches | More structured RTL tests | Replaces a harness that works |

## 4. Whole-system alternatives

| Architecture | Pros | Cons |
|---|---|---|
| **S1 (current):** front end on the PC, Ethernet, PicoRV32 with `kdot`, the neuron engine and a GRU verifier | Meets both false-alarm targets with 86.8% recall; bit-exact up to the RTL | Not yet run on a board; part spiking, part GRU; isolated words only |
| S2 Standalone: codec and front end on the board, with the S1 back end | A real edge device with no PC | The front end must be rebuilt and matched bit for bit; deferred |
| S3 Fabric dataflow: layer 1, engine and decision in the PL; the CPU only configures them | Fastest (*est.* about 1 ms per hop); lowest energy | A weaker RISC-V co-design story; less flexible |
| S4 RISC-V software only | Simplest | About 113 ms per hop for the SNN alone; no room for a verifier |
| S5 Clip classifier: 1 s windows of 8 × 16 features (`code/snn_keyword/README_clip_pipeline.md`) | Tiny (6.4 KB of parameters); 98% on isolated clips | 52-366 false alarms per hour on ordinary speech; 7% recall at ≤ 2 per hour (JOURNAL 27) |
| S6 ARM only (NEON) | Quick to develop | No SNN hardware; outside the course's goal |

## 5. Budgets

| Resource | Used | Available | Note |
|---|---|---|---|
| LUTs | 14,627 (27%) | 53,200 | The engine is about 10-12k of them |
| BRAM36 | 88 (63%) | 140 | SoC memory 64, engine 24. **The scarce resource**: 52 left. |
| DSP slices | 12 (5%) | 220 | |
| Timing | WNS +0.406 ns at 100 MHz | | Default Vivado strategy |
| Model memory | 121 KB | 144 KB model region | Both stages |
| Compute per 250 ms hop | 10.3 ms (stage 1); 117 ms worst case with the verifier | 250 ms | Target ≤ 25 ms for stage 1 |
| Network round trip | 7 ms | | Ethernet; JTAG relay 30-45 ms |
| Median detection after the word | 0.04 s (stage 1), 0.18 s (cascade) | | Test split |

## 6. Recommended next steps

1. **Run the board test of S1 first** (`MERGE_SUMMARY.md`, "Still to do", item 1).
   Every other choice depends on the system working on real hardware.
2. **The accuracy limit is in B0, not in the hardware.** The change that matters is
   training stage 1 with the keyword inside sentences: in the middle of a
   sentence, 6% of "sheila" is detected today.
3. **The compute limit is the software ALIF update (B7) and the verifier's bursts
   (B10).** A barrel shifter or an ALIF instruction is the cheap fix. Moving layer
   1 into the engine is the larger one. A second core, or a spiking verifier,
   removes the 117 ms bursts.
4. **BRAM, not logic, limits the design** (63% against 27% of LUTs). Any
   alternative that adds a memory or a model needs a BRAM budget first; int4
   weights would make room.
5. **Say plainly that stage 2 is a GRU**, or test a spiking verifier, so that the
   claim "an SNN detects sheila" holds.
