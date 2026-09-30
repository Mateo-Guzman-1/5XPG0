# Development journal: board bring-up, `kdot`, confusable words

A record of each change after the initial release (commit `714db5c`): why
it was made, what changed, and the result, including attempts that did not
work. The numbers come from the files named in each entry. Two probe metrics
appear below; they are not interchangeable:

- **any-window probe** (entries 4a–4d): a clip counts as detected if any 1 s
  window at a 62.5 ms hop fires. Early diagnosis only.
- **250 ms-hop probe** (`confusables.py`): the demo's real hop, averaged over
  four hop phases, single window or "2 of 3". All tables in the README use it.

The "synthesized words" are offline Windows voices (David, Zira; 24 words × 4
rates × 3 pitches; `make_tts_probe.ps1`). They were never used for training.
With only two voices, they indicate trends, not accuracy.

---

## 2026-09-24

### 1. Running the implementation on the PYNQ-Z2: SD boot failure, JTAG workaround

- **Motivation.** Run the release on the physical board. The README path
  needs PYNQ Linux (`board_server.py` over Ethernet).
- **Finding.** The UART on COM8 was silent and 192.168.2.99 did not answer.
  Read over JTAG: boot mode register `0xF800025C` = 5 (SD), but BootROM
  status `0xF8000258` = `0x0040200A`, error `0x200A`, meaning boot from SD
  failed. Both Cortex-A9 cores sat in the BootROM, so Linux never ran. This is
  a problem with this particular board or SD card. **Reflashing the SD card
  with the PYNQ-Z2 v3.1 image will most likely fix it**, after which the
  standard Ethernet path applies.
- **Change.** A JTAG workaround replaces the boot chain (`jtag/bringup.tcl`).
  It runs the Vivado-generated `ps7_init` (FCLK0 = 100 MHz), programs the
  bitstream, runs `ps7_post_config`, and writes the firmware into BRAM through
  the PS AXI port with the core held in reset. `jtag/board_test.tcl` runs the
  40 verification vectors.
- **Outcome.** 40/40 vectors bit-exact on the physical board, cycle counts
  identical to RTL (worst 5,981,498), and LED0 off 1002–1005 ms after
  detection (JTAG polling). `results/board_jtag.csv`.

### 2. Live microphone demo over a JTAG relay

- **Motivation.** Connecting the Ethernet cable could not help: the board has
  no running OS. `jtag_server.py` stands in for `board_server.py` on the PC
  and moves each demo frame through xsdb into the mailbox.
- **Bugs found and fixed** (`jtag_server.tcl`, `jtag_server.py`):
  1. The frame check `regexp {^[0-9a-f]{1536}$}` does not compile in Tcl,
     which caps repetition counts at 255. The handler died silently and every
     request timed out. It now checks the length separately.
  2. `Popen.terminate()` killed only `xsdb.bat`. The orphaned `xsdb.exe` kept
     port 5557, and every restarted server connected to the stale process. It
     now kills the whole process tree (`taskkill /T`).
  3. (2026-09-25) "Invalid context" and "no targets found" errors appeared
     when Vivado's Hardware Manager refreshed the shared cable. The relay now
     re-selects the CPU target and retries for up to 5 s.
- **Outcome.** The demo works over JTAG. The round trip was 80–105 ms per
  window with plain RV32IM, and 30–45 ms after `kdot`, dominated by JTAG.
  5/5 "yes" and 5/5 other clips were classified correctly.

### 3. `kdot` custom instruction (hardware acceleration)

- **Motivation.** The core was confirmed to run no custom instruction. The
  input layer (64 × 768 multiply-accumulates) took about 99% of 5.98 M cycles.
  The SoC is memory-bound: every CPU read waits `READ_WAIT = 8` cycles, about
  120 cycles per MAC.
- **Change.** `rtl/kdot_pcpi.v` is a PCPI coprocessor with custom-0
  instructions `klen` and `kdot`. It borrows BRAM port B while the CPU
  stalls, reads 3 words per 4 MACs, and has registered DSP multiply and
  accumulate stages, so results are bit-exact with the C loop. The fabric
  reports ABI `0x00020001` (bit 0 = `kdot`). The firmware builds
  `keyword.bin` and `keyword_kdot.bin`. The original `keyword.bit` is kept.
- **Outcome.**
  - Worst case 267,370 cycles, 2.67 ms: **22.4× faster**. Bit-exact in RTL
    and on the board, with cycles identical to RTL
    (`results/board_jtag_kdot.csv`).
  - The original firmware runs on the new fabric with unchanged cycles.
  - Timing met, WNS +0.745 ns (baseline +0.517). Cost: +180 LUT, +147 FF,
    +2 DSP (`results/hardware_kdot.json`).
- **Profile after `kdot`**, measured by building with 6 and 12 LIF steps:
  12 LIF steps ≈ 196k cycles (72%), 64 `kdot` calls ≈ 38k (14%), division,
  bias, readout and mailbox ≈ 39k (14%).
- **Not pursued.** Lowering `READ_WAIT` (about 3× on all remaining code) and a
  LIF update unit (about 3.5× more). Inference already uses 2.7 ms of a
  250 ms budget and the live demo is limited by JTAG. The spare time is
  better spent on a model with time structure.

### 4. False "yes" on "yeets", "pizza": diagnosis and augmentation

- **Motivation.** User report: "yeets" and the "eetz" part of "pizza" were
  detected as "yes". Requirement: accept the plain /s/ ending, reject /ts/ and
  /tʃ/ ("ch").
- **Diagnosis** (any-window probe, release model):
  - Words fired far too often: "yetz" 92%, "yets"/"eats" 88%, "yeets" 75%,
    "yeah" 67% (no final consonant at all), "yetch" 62%.
  - Speech Commands has no negatives that differ from "yes" only in the
    ending. Its clips are also centred, while the demo slides its window.
  - Windows that cut "yes" off before its /s/ still fired. The model had
    learned "e/ee vowel near the right edge plus friction", not "yes".
- **Attempts** (configurations and validation F1: `results/experiment_sweep.json`;
  held-out scores: `results/confusables.json`):
  - **a. Ending-only negatives, 64 hidden.** Edits made from real "yes"
    recordings in the train split: /t/ closure 30–80 ms plus burst ("yets"),
    /ʃ/-shifted frication ("yetch", "yesh"), and a removed /s/ ("yeh").
    Synthetic "yets" false accepts dropped from 85% to 5–10%, but synthesized
    /ts/ words barely moved (any-window 49–75%) and clip recall fell from
    85% to 67–71%. Rejected; the runs were superseded and deleted.
  - **b. ts/ch only, fraction 0.15, 88 hidden** (`tsch*`, `all_h88`).
    Similar result: recall about 70–77%, /ts/ words still high.
  - **c. Window-position analysis.** Many remaining fires came from
    edge-truncated windows. `augment.py` adds `tail` (window ends before or at
    the /s/), `head` (window starts after the /j/ onset) and `shift` (positive:
    complete word at a random position) (`aug_h*`).
  - **d. The closures were too short.** In the features, the synthesized
    /t/ closure lasts 60–120 ms (2–4 time bins), while the variants used
    30–80 ms. Widened to 30–130 ms (`aug2_*`).
  - **e. Capacity test** with 88, 176 and 256 hidden neurons: no systematic
    gain. The limit is not size. One dense layer has no time-shift
    invariance and cannot detect "vowel, gap, hiss" at any position.
  - **f. Without `head`** (`nohead*`): /ts/ words and live false accepts got
    worse. Rejected.
- **Outcome: trial model** `aug2_h64` seed 2, selected by validation F1
  (`results/models/augmented_current_seed2.npz`, threshold 2067). 250 ms-hop
  probe, single window, compared with the release model:

  | | Release | Trial |
  |---|---|---|
  | Synthesized "yes" | 91% | 72% |
  | yeets/yets/yetz | 75% | 22% |
  | pizza(s) | 31% | 7% |
  | eats/its | 66% | 22% |
  | ch words | 26% | 2% |
  | yeah | 33% | 1% |
  | Clip F1 | 0.876 | 0.820 |
  | Live "yes" / other words (single window) | 74.5% / 1.07% | 66.1% / 0.83% |

  It is loaded on the board (bit-exact, 40/40) but **not promoted to
  release** because of the recall cost.

## 2026-09-25

### 5. "2 of 3" window confirmation

- **Motivation.** A real "yes" spans several overlapping 250 ms-hop windows,
  and most false accepts fire in only one. Confirmation cuts false alarms
  without retraining.
- **Change.**
  - Firmware command 3 confirms a window only if it and one of the two
    previous stream windows (within 750 ms, by the fabric timer) reach the
    stream threshold. Only confirmed windows light LED0. `detected` bit 0 =
    confirmed, bit 1 = this window alone. Command 1 is unchanged.
  - The protocol gained a 16-bit mode field, read as 0 by the original client
    header.
  - `tune_stream.py` sets the stream threshold on validation clips only (at
    most 0.3% confirmed false accepts): 1553 for release, 652 for trial.
  - The firmware publishes the stream threshold (mailbox word 15). The RTL
    harness checks the +,+,−,+ sequence and the LED.
- **Outcome** (held-out test clips in noise, 250 ms hop; `results/confusables.json`):

  | | "yes" detected | Other words accepted |
  |---|---|---|
  | Release, single window | 74.5% | 1.07% |
  | Release, 2 of 3 | 67.3% | 0.13% |
  | Trial, 2 of 3 | 60.1% | 0.50% |

  Trial with 2 of 3 on synthesized words: "yes" 56%, yeets/yets/yetz 12%,
  pizza 3%, eats/its 7%, ch 1%, yeah 3%. On the board, +,+,−,+ gave
  unconfirmed, confirmed, none, confirmed, and history expired after 1 s.
  Cost: 23 cycles per inference.
- **Manual microphone test (user, trial model on the board over JTAG).**
  Made-up words ending in /s/-like sounds, such as "mes", "ras", "tes", "tos"
  and "ex", were often detected as "yes" when each window decided alone. With
  "2 of 3", most of these false detections stopped, but detection is still
  far from perfect. This was an informal test with one speaker and one
  microphone, neither of which is in the training data.

### 6. 64 time bins (finer time resolution): not adopted

- **Motivation.** A 60–120 ms /t/ closure covers only 2–4 of the 31 ms time
  bins.
- **Change.** The experiment-only `KWS_TIME_BINS=64` gives about 16 ms bins
  and a 1536-byte input. To fit BRAM with int16 weights, the model has 48 or
  56 hidden neurons. It was also trained for 60 epochs, with the aug2 data.
- **Outcome.** Worse throughout (`results/confusables_64bins.json`):
  - Best validation F1 0.819, against 0.837 at 32 bins.
  - Live single-window "yes" 61.6–70.9%, against 66–72%.
  - No gain on /ts/ words.

  Twice the weights per neuron add variance without new structure the dense
  layer can use. The input-size plumbing stays: the firmware publishes its
  input size (mailbox word 14), and the RTL harness and relay adapt. Finer
  bins should come back together with a time-convolutional model.

### 7. Record-keeping

- Full verification rerun (`verify.py`): native C 11,005/11,005, RTL 40/40,
  new hashes in `results/verification.json`.
- `deploy/model.npz` gained `stream_threshold` (weights unchanged). Hashes
  refreshed in `results/evaluation.json` and `deploy/manifest.json`, which now
  also covers `keyword_kdot.*`, `ps7_init.tcl` and the JTAG board status.
- `train_keyword_snn.py --aug-fraction` defaults to 0 (release training); the
  trial model used 0.3.
- The JTAG scripts moved from ignored build files into `jtag/`, and were
  retested on the board (40/40).

### 8. Phase 0: robustness harness and baseline (IMPLEMENTATION_PLAN.md)

- **Motivation.** Measure speaker, microphone, corpus and continuous-stream
  behaviour before changing the model, so that every later phase is judged on
  the same numbers.
- **Change.**
  - `robust_eval.py` scores any detector through one interface (waveform in,
    per-step decision score out), so window models and the Phase 3 streaming
    models share every test. Tests: live clips (the `confusables.py` set,
    same seeds), the same clips through 16 held-out microphones, 218 real
    RIRs, and both (`channels.py`); MSWC English test clips, never trained
    on; a one-hour stream of Speech Commands words, LibriSpeech test-clean
    speech (utterances containing "yes" removed) and 138 inserted test "yes"
    over varying noise; the TTS probe.
  - `channels.py`: held-out microphones use odd third-octave bands for every
    corner and centre frequency, training microphones even bands; real RIRs
    are held out, simulated RIRs are for training.
  - `fetch_corpora.py` downloads MSWC (selected clips only), LibriSpeech,
    OpenSLR 28 and MUSAN. The hub drops multi-GB transfers without an
    error, so downloads resume by HTTP range until the file has the
    server's size.
- **Checks.** The clean live and TTS numbers equal `confusables.json` to the
  last digit for both models, so the new harness measures the same thing.
- **Finding: MSWC's official splits share speakers** (about 10k speakers in
  both train and test). `prepare_multicorpus.py` removes test speakers from
  MSWC dev and train, and dev speakers from train.
- **Outcome** (`results/robust_baseline.json`, test split, "2 of 3"):

  | | Release | Trial |
  |---|---|---|
  | Live "yes" / other words, clean | 67.3% / 0.13% | 60.1% / 0.50% |
  | Held-out microphones: recall drop | 17.7 points | 16.7 points |
  | Held-out real rooms: recall drop | 9.1 points | 7.6 points |
  | Microphone + room: recall drop | 29.4 points | 25.1 points |
  | MSWC "yes" / random words accepted | 58.7% / 1.0% | 50.0% / 0.9% |
  | MSWC recall drop at equal false-accept rate | 26.5 points | 13.5 points |
  | MSWC near-miss accepted, /s/-final, /ts/-final | 14.5%, 9.3% | 12.2%, 6.2% |
  | Stream: false accepts per hour | 52 | 105 |
  | Stream: recall, median latency after word end | 76.1%, 0.19 s | 68.1%, 0.19 s |
  | Stream: recall at ≤ 1 false accept per hour | 43.5% | 27.5% |

  - Both models are far from the device (≤ 3 points) and corpus
    (≤ 5 points) targets. The microphone shift alone costs as much as the
    trial's whole confusable gain.
  - 45 of the release's 52 stream false accepts come from LibriSpeech read
    speech: neither model has seen continuous speech. The trial's
    lower stream threshold doubles them.
  - **Proposed stream target** (the plan set it after Phase 0): at most
    2 false accepts per hour on this stream, at the recall targets.
- The harness took 47 min for two models with the CPU shared with TTS
  synthesis; about 10 min alone.

### 9. Phase 1: multi-corpus data, online augmentation, and the data-only gate

- **Motivation.** Give training real audio from several collections,
  microphones, rooms and near-miss words, and measure what data alone does
  before the architecture changes (plan, Phase 1 gate).
- **Change.**
  - Corpora (`fetch_corpora.py`, `prepare_multicorpus.py`): Speech
    Commands; MSWC English train/dev (only the 37 of 88 train archives
    holding "yes" and the near-miss words, with random negatives from the
    same archives); LibriSpeech train-clean-100 (20 h) and dev-clean; MUSAN
    noise and music; OpenSLR 28 point-source noises. MSWC speakers made
    disjoint across splits: this removed 23,253 of 60,260 train clips
    (37k clips from 14.5k speakers remain) and 2,584 of 5,887 dev clips.
    MSWC "yes" whose content ends within 15 dB of its peak (cut /s/, about
    5%) are dropped; 28 in train.
  - `make_tts_negatives.py`: 55,000 Piper utterances. 234 negative words
    (every single-letter edit of "yes", minus edits espeak pronounces as
    "yes" or with only a voiced final /z/, plus real /s/, /ts/, /tʃ/ and
    "ye-" words) and "yes" itself, so a synthetic voice is no cue. 904
    LibriTTS-R speakers split 80/10/10; VCTK, L2-ARCTIC and ARCTIC voices
    are held out whole.
  - `augment_online.py` (GPU): speed, simulated rooms, noise at 0-30 dB SNR,
    level, training microphones (even bands), VTLP, SpecAugment; its front
    end equals `features()` exactly for window features.
  - `features.py`: frame-level log-mel and PCEN front ends (`features()`
    unchanged).
- **Gate** (`train_dense_online.py`): the release's dense 64-neuron model,
  same export, only the data changed; stream threshold by `tune_stream.py`.
  Test split (`results/phase1_gate_interim.json`, `results/phase1_gate_all.json`):

  | | Release | Control: release recipe | SC + TTS + LibriSpeech, augmented | + MSWC (all) |
  |---|---|---|---|---|
  | Live "yes" / other words, clean | 67.3% / 0.13% | 68.3% / 0.33% | 56.6% / 0.50% | 58.7% / 0.60% |
  | Recall drop: mic / room / both | 17.7 / 9.1 / 29.4 | 14.1 / 5.7 / 22.9 | 5.0 / 4.5 / 15.8 | 7.9 / 6.7 / 17.9 |
  | MSWC drop at equal false-accept rate | 26.5 | 22.2 | 2.8 (held-out corpus) | -1.4 (in-corpus) |
  | MSWC near-miss /s/, /ts/ accepted | 14.5%, 9.3% | 12.5%, 9.0% | 5.7%, 1.9% | 5.5%, 1.2% |
  | Stream false accepts per hour | 52 | 88 | 10 | 17 |
  | TTS yeets/yets/yetz, eats/its | 53%, 44% | 53%, 36% | 11%, 2% | 16%, 3% |
  | TTS "yes" | 77% | 82% | 46% | 56% |

- **Outcome.** Data alone delivers most of the robustness: device drops
  fall to a third, the held-out corpus gap closes (22 to 3 points), stream
  false alarms fall 5-9×, and /ts/ confusions nearly vanish. But clean
  recall drops by about 10 points: one dense layer over a 1 s snapshot
  cannot hold the wider data. This is the separation the gate asks for
  (data gains against architecture gains), and it supports Phase 3.
  None of these models is promoted.

### 10. Phase 2: BC-ResNet-8 teacher

- `teacher.py`: BC-ResNet-8 [8] as in the reference implementation (327k
  parameters), 40-band log-mel of the same augmented audio as the student,
  37 classes, SGD with warm-up and cosine decay, 30 × 400 steps.
- **Outcome.** 96.5% on the 35-word Speech Commands validation split
  (target ≥ 95%). `runs_teacher/bcresnet8.pt`. Training only, never deployed.

### 11. Phase 3: streaming SNN, first model and gate

- **Change.** `snn_stream.py`: 24-band log-mel frames every 10 ms → 128 ALIF
  (dense input, the `kdot` layer) → learnable delays 0-31 frames on the
  layer 1 → 2 synapses (DCLS [4], rounded to one tap for the last 20% of
  stage 1) → 128 recurrent ALIF → leaky readout; 58k parameters.
  Leak and adaptation are 1 − 2^−k per neuron so Phase 4 can round them to
  shifts. The delays sit on the spiking synapses, not on the analog input,
  to match the spike-history ring buffer of the planned engine.
  `train_stream.py`: stage 1 on 37 classes (30 × 300 steps, with or without
  distillation); stage 2 on 3 s streams (20 × 200 steps), "yes" rewarded at
  its best frame within 0.35 s after the word, other frames labelled
  speech or silence, 25% of the clips swapped for "yes". `stream_select.py`
  sets the threshold on validation data only: best live recall with live
  false accepts ≤ 0.2% and ≤ 2 stream false accepts per hour.
- **Stage 1**, 35-word validation accuracy: 77.6% without distillation,
  74.6% with it.
- **Gate** (validation proxies, same rule for every model,
  `results/stream_selection_*.json`):

  | | Live recall | Stream recall | MSWC dev recall | Median latency |
  |---|---|---|---|---|
  | Release | 38.3% | 46.4% | 29.8% | 0.19 s |
  | Dense, Phase 1 data | 51.9% | 60.9% | 37.3% | 0.30 s |
  | Stream SNN (no distillation) | 54.7% | 58.7% | 42.5% | 0.13 s |

  The SNN clearly beats the release on the validation proxies, so Phase 3
  continues. Most of the gain is from data; the architecture adds 3-5
  points over the data-only dense model and faster detection.
- **Test split** (`results/phase3_snn_s2_nokd.json`): live 55.1% / 0.27%;
  near-miss words 0.6-2.8% accepted (/ts/ 0.6%, release 9.3%); MSWC
  equal-FA drop 4.7 points; room drop 4.1; stream 5 false accepts per hour
  at 56.5% recall, latency 0.11 s. Weak: **microphone drop 14.6 points**,
  TTS "yes" 33%, and at ≤ 2 false accepts per hour on the test stream
  recall is 20%, against 42% for the release (the release is better at
  the strictest budgets; the SNN from 5 per hour up).
- Fixed on the way: the stream recall/false-accept curve used quantiles of
  per-step scores, too coarse at 10 ms steps; it now uses per-second maxima.
- **Next.** The plan's ablations (PCEN, no delays, LIF, no recurrence, no
  TTS, 64 neurons, distillation), each through both stages, on validation.

### 12. Phase 4 groundwork: integer oracle and C kernel, bit-exact

- **Change.**
  - `model.quantize_stream()` / `integer_forward_stream()`: int8 weights,
    one scale per neuron folded into an integer threshold in [1024, 2048)
    (layer 1: current = (x · w1) >> r; layer 2: sum of arriving int8
    weights << p); leak and adaptation as exact arithmetic shifts; Q8
    adaptation traces; int16 saturating membranes; int32 readout with one
    scale for all classes, so the decision score and threshold are integers.
  - `snn_stream.py`: QAT switches (int8 fake quantization per row, joint
    scale for each layer-2 neuron's delayed and recurrent inputs, rounded
    shifts); `train_stream.py --qat`; `quantize_stream_model.py`;
    `IntegerStreamDetector` so `robust_eval.py` and `stream_select.py`
    score the integer model itself.
  - `firmware/stream_infer.c`: layer 1 dense (int16 weights, the `kdot`
    format); layer 2 event-driven through a CSR table of synapses grouped
    by (source neuron, delay); recurrent and readout weights added per
    spiking neuron. `export_model.export_stream()` writes the headers and
    rejects any model whose int32 intermediates could overflow.
    `sim/native_stream.c` + `verify_stream.py` compare C with the oracle.
- **Outcome** (baseline SNN exported without QAT, as a pipeline test):
  native C equals the oracle on all 732,332 frames (3,419 live test clips and
  a 2-minute stream). 16,132 non-zero delayed synapses; per 10 ms frame 4.8
  layer-1 and 6.5 layer-2 spikes, 1,419 synaptic events on average (4,724
  maximum), about 35k per 250 ms hop (the report's sizing: 41k).
- **Ablations so far** (validation, `results/stream_ablation.json`):
  distillation −10 / −11 / −15 points (live / stream / MSWC recall);
  PCEN +2 / −6 / −5 (log-mel kept, as the plan's validation rule decides);
  no recurrence −10 / −25 / −11.
- A QAT run beside the two ablation runs ran out of the 8 GB of GPU memory;
  QAT waits for the ablations and runs on the configuration they favour.

## 2026-09-26

### 13. Phase 5: streaming firmware, ABI v3, bus fix (RTL; board: entry 18)

- **Firmware** (`firmware/stream_main.c`, `make -C firmware stream`):
  mailbox magic "KWS3" (written last, after the words it announces; a
  first version wrote it first and the RTL harness read the frame size too
  early). Command 4 streams 1-100 frames of 24 bytes with the network state
  kept on the core, 5 resets it, 2 is status. Detection with a hold-off of
  100 frames (1 s) counted in frames, so the board decision equals the
  harness's. `USE_KDOT` computes layer 1 with the existing `kdot`
  instruction (24 int16 weights × 24 input bytes).
- **Protocol** (`protocol.py`, `board_server.py`, `jtag_server.py/.tcl`,
  `pc_keyword_demo.py`): ABI v3 requests INFO, FRAMES, RESET beside the
  unchanged v2 window protocol. The server detects the firmware from its
  magic; a request for the other ABI gets status 2. The client asks INFO
  and falls back to v2 for older servers. Tested with the simulated
  backend: v3 detects a test "yes", v2 still serves the release, a v2
  request to a v3 server is refused; live chunked framing equals the
  offline front end frame for frame.
- **RTL** (`sim/soc_stream.cpp`, `verify_stream_rtl.py`): the firmware on the
  PicoRV32 SoC in Verilator, every 25-frame hop checked against the oracle
  (best and last score, spikes, detection), plus malformed requests, reset
  and duplicate sequence numbers. Bit-exact on all hops tested. Cycles per
  250 ms hop, 4 test streams:

  | | Worst | Mean |
  |---|---|---|
  | RV32IM, `READ_WAIT` 8 | 327 ms | 293 ms |
  | `kdot`, `READ_WAIT` 8 | 236 ms | 202 ms |
  | RV32IM, `READ_WAIT` 1 | 135 ms | 121 ms |
  | `kdot`, `READ_WAIT` 1 | 98 ms | 84 ms |
  | + per-frame spike lists (below) | **69 ms** | **54 ms** |

  A profiling build (`-DPROFILE`, section cycles in mailbox words 20-24)
  showed 62% of the time in the delayed synapses, mostly scanning 32 × 128
  history bytes. Keeping per frame the list of layer-1 neurons that spiked
  cut that stage from 209k to 88k cycles per frame. Remaining per frame:
  delayed 88k, recurrence 49k, layer 1 37k, layer 2 neurons 34k, readout 6k
  (about 56 cycles per synaptic event; the core has no barrel shifter).
- **Bus fix** (report E2): `READ_WAIT` 8 → 1 in `rtl/spike_soc.v` (the CPU
  address is stable from the accept and the BRAM output is registered, so
  data is valid one cycle later). `tb_axi` and `tb_led` pass; the release
  firmware passes its full RTL harness (40 vectors bit-exact, LED pulse,
  "2 of 3") and gets faster: RV32IM 59.8 → 24.7 ms, `kdot` 2.67 → 1.45 ms.
  Vivado 2025.2: timing met, WNS +0.365 ns (was +0.745), 2,881 LUTs, 64
  BRAM36 (`build/keyword_rw1.bit`; `deploy/` unchanged).
- **Outcome.** Real time on the RV32 software path (worst 69 ms of the
  250 ms hop) but not the 25 ms target: per the plan, Phase 6 (the
  event-driven engine) becomes mandatory. Board test over JTAG not yet run:
  the JTAG cable was in use by a running relay.

### 14. Phase 3 ablations (complete)

- Each variant trained through both stages with the baseline's schedule and
  scored on validation only under one rule (`results/stream_ablation.json`,
  `ablation_report.py`). Points against the baseline, live / stream / MSWC
  recall (baseline 54.7% / 58.7% / 42.5%):

  | Variant | Live | Stream | MSWC |
  |---|---|---|---|
  | LIF (no threshold adaptation) | −20.6 | −36.2 | −20.5 |
  | No learnable delays | −21.2 | −19.6 | −13.2 |
  | No recurrence | −10.3 | −25.4 | −11.5 |
  | + distillation from the teacher | −10.1 | −10.9 | −14.8 |
  | No TTS negatives | −5.5 | +1.4 | −3.3 |
  | PCEN front end | +2.0 | −5.8 | −5.1 |
  | 64 + 64 neurons | +4.5 | −5.8 | +7.8 |

- **Outcome.** Adaptation, delays and recurrence each carry 10-36 points:
  the temporal mechanisms the report chose are what the model uses.
  Distillation hurts here (the teacher judges 1 s windows at 40 bands; the
  student is scored on its last 0.4 s at 24 bands). PCEN and 64 neurons
  trade stream recall for live or MSWC recall; by the combined rule
  (live + stream) the baseline stays (113.4 against 112.1 for 64 neurons,
  the closest). The 64-neuron model is a candidate if memory or activity
  must shrink.

### 15. Phase 4 complete: QAT and the integer model

- QAT fine-tune of the baseline (8 × 200 steps, int8 fake quantization,
  rounded shifts; `runs_stream/s2_nokd_qat`), exported with
  `quantize_stream_model.py`; thresholds chosen on validation with the
  integer oracle (`results/stream_selection_qat.json`):

  | Model | Live | Stream | MSWC |
  |---|---|---|---|
  | Float baseline (before QAT) | 54.7% | 58.7% | 42.5% |
  | Float QAT, `last.pt` | 59.7% | 61.6% | 41.9% |
  | **Integer, `int_last.npz`** | **59.7%** | **63.0%** | **43.4%** |
  | Integer, `int_model.npz` | 41.3% | 45.7% | 25.6% |

- The selected integer model equals its float model (within 1.5 points,
  no loss): Phase 4's accuracy criterion holds. `int_model.npz` tracks
  its float model as closely (score correlation 0.97, same firing rates);
  its gap comes from the operating point, which the rule fixes with only 2
  allowed false accepts in the one-hour validation stream. Validation
  numbers therefore carry a few points of noise.
- Native C equals the oracle on all 1,080,332 test frames (3,419 live
  clips and the full one-hour stream; `results/verification_stream.json`).
- int4 was not tried: int8 fits comfortably (36 KB of weights) and the
  confusable rejection is already at its floor.

### 16. Phase 6: event-driven neuron engine

- **Design** (`rtl/neuron_engine.v`, MMIO at 0x1000_4000, ABI bit 1): the
  CPU computes layer 1 with `kdot` and writes the indices of the layer-1
  neurons that spiked; the engine streams the delayed synapses of the last
  32 frames from a CSR table (one synaptic event per clock, 16 KB BRAM),
  adds the recurrent columns 16 lanes wide (128-bit rows), updates the 128
  ALIF neurons in a 5-stage pipeline (one per clock; the plan's 16 lanes
  would buy 120 cycles per frame against ~1,600 event cycles) and the
  leaky readout. Firmware: `make -C firmware stream` builds
  `keyword_stream_engine.bin` (`USE_ENGINE`).
- **Verification.**
  - `sim/engine_tb.cpp`: the engine alone against a C++ model with random
    tables and spike patterns from silent to every delay slot full, 3,000
    frames × 4 seeds: 0 mismatches. (Score saturation was not reached.)
  - Full SoC with firmware against the oracle: 40 test streams, 358 hops,
    bit-exact, LED pulse exactly 1 s.
  - `tb_axi`, `tb_led` pass with the engine in the SoC.
- **Timing closure.** The first version was bit-exact but failed timing
  (WNS −7.9 ns, 34k LUTs): tables and neuron state were flip-flops behind
  128:1 multiplexers, and the score took 37 logic levels. Tables and state
  moved to synchronous-read RAMs (LUTs 14.8k, WNS −2.7 ns); then the class
  maximum became a comparison tree and the synapse word got a register
  after the BRAM. Final: **WNS +0.293 ns at 100 MHz**, 12,940 LUTs (engine
  about 10k), 6,504 FFs, 88 BRAM36 (engine 24), 12 DSPs (engine 4).
- **Measurements** (final model, RTL, `results/hardware_engine.json`):

  | Per 250 ms hop | Worst | Mean |
  |---|---|---|
  | Software only (RV32IM) | 113 ms | 90 ms |
  | `kdot` + software | 77 ms | 53 ms |
  | `kdot` + engine | **10.2 ms** | **9.9 ms** |

  Synaptic operations per hop: 76.8k dense layer-1 MACs; 40.6k layer-2
  and readout accumulates against 829k for a dense equivalent (20×).
  Arithmetic energy proxy [31]: 27 nJ against 272 nJ per hop (memory
  access excluded; not a power measurement).
- Board run over JTAG: not done (cable in use by a running relay).

### 17. Phase 7: evaluation against the acceptance targets — not promoted

- The selected integer model, scored once on the test split
  (`results/robust_final.json`), against the release:

  | Metric | Release | Streaming SNN (int8) | Target | Met |
  |---|---|---|---|---|
  | Live "yes" detected | 67.3% | 57.8% | ≥ 85% | no |
  | Live other words accepted | 0.13% | 0.27% | ≤ 0.2% | no |
  | Synthesized /ts/ words (yeets…, pizza, eats) | 11-53% | 0-5.6% | ≤ 10% | **yes** |
  | Synthesized "yes" | 77% | 41% | ≥ 85% | no |
  | Microphone / room drop | 17.7 / 9.1 | 19.6 / 4.1 | ≤ 3 | no |
  | MSWC drop at equal false-accept rate | 26.5 | −9.9 (in-corpus) | ≤ 5 | not a leave-one-out test |
  | Stream false accepts per hour | 52 (76% recall) | 5 (60% recall) | ≤ 2 | no |
  | Stream recall at ≤ 1 / ≤ 2 false accepts per hour | 42% / 42% | 49% / 54% | — | better |
  | Compute per hop on the SoC | 2.7 ms | 10.2 ms (engine) | ≤ 25 ms | **yes** |
  | Bit-exact oracle / C / RTL / board | yes | yes / yes / yes / not run | yes | partly |

- Also on test: MSWC near-miss words 0-4% accepted (release 3-15%); LibriSpeech
  read speech causes 4 false accepts per hour (release 45); median latency
  0.10 s after the word (release 0.19 s).
- **Decision.** The new detector does not meet the acceptance targets, so
  per the plan it is **not promoted**: `deploy/` keeps the release, and the
  streaming SNN stays a documented candidate. It is clearly better at what
  failed before (confusable words, running speech, false alarms per hour,
  latency) and fits the hardware with margin, but it detects fewer "yes"
  in clean clips than the release, and it is not microphone-robust.
- **Where the gap is.** Recall is limited by the operating point the
  ≤ 2 false accepts per hour rule imposes and by the small model (58k
  parameters, 30 + 20 short epochs). Held-out microphones remain the
  largest weakness for every model so far (window or streaming, with or
  without the augmentation); the training microphones' EQ is applied to the
  power spectrum, which may not model real devices well enough. A
  leave-one-source-out test (training without MSWC) is still owed for the
  new route.

## 2026-09-27

### 18. Board test of the streaming firmware and the neuron engine

`build/keyword_engine.bit` (engine + `kdot` + bus fix) on the physical
PYNQ-Z2, loaded over JTAG. Vectors: `jtag/make_stream_vectors.py`, the same
40 live test streams as `verify_stream_rtl.py` (358 hops of 25 frames). Test:
`jtag/stream_board_test.tcl`, which resets per stream and checks best and last
score, spikes, detection bits, detection frame and frame counter against the
integer oracle. It also checks rejected malformed requests and the LED0 pulse.
Results in `results/board_stream.json` and `results/board_stream_*.csv`.

| Firmware | Board = oracle | Board = RTL (all fields, cycles) | Worst ms/hop | Mean |
|---|---|---|---|---|
| engine | 358/358 hops | 358/358 | 10.22 | 9.87 |
| `kdot` | 358/358 | 72/72 (RTL ran 8 streams) | 88.77 | 54.66 |
| RV32IM | 358/358 | 72/72 | 125.37 | 91.27 |

The LED pulse was 0.99-1.00 s. On the same bitstream, the release firmware
(`deploy/keyword_kdot.bin`) passes its 40 vectors in 144,642 cycles worst,
against 267,393 on the release bitstream. That is the bus fix (`READ_WAIT`
8 → 1) on hardware.

Findings:
- **The SD card boots now.** With the board freshly powered, PYNQ Linux runs
  with the base overlay (four MicroBlaze IOPs visible over JTAG), and the PS
  already has FCLK0 at 100 MHz and the level shifters on. `jtag/bringup.tcl`
  detects this and only reprograms the PL; memory is then accessed physically
  through the APU debug port. The ps7_init path remains for a board without
  Linux. `board_test.tcl`, `stream_board_test.tcl` and `jtag_server.tcl` fall
  back to the APU target when the core's MMU is on.
- Once, after the fourth PL reprogramming under running Linux, the PS hung.
  The DAP reported an AHB-AP transaction error and the Cortex-A9 targets
  disappeared. `rst -system` over JTAG rebooted the board (PYNQ back in 60 s),
  and the same sequence then passed. The cause is not known; likely candidates
  are Linux touching a base-overlay address, or an access during
  reconfiguration. Load overlays through PYNQ once the Ethernet path is in use.
- The host-side ABI word (`ps_if.v`, syscon 0x1c) still read `0x00020001` in
  this bitstream; only the SoC-side register had bit 1. `ps_if.v` now returns
  `0x00020003`, which needs a rebuild.

Afterwards the release (`deploy/keyword_kdot.bit` / `.bin`) was reloaded
and the user's `jtag_server.py` relay restarted. An INFO request plus vector 0
through the relay returned the oracle scores.

### 18b. (Pedro, second board) Standard Ethernet path on PYNQ Linux; requests processed twice

*Merged from branch `Pedro` (his entry 8). The fix below is in `board_server.py`;
it applies to the v3 streaming requests as well.*

- **Motivation.** Open item 1: run the intended deployment (`board_server.py`
  on PYNQ Linux) instead of the JTAG relay.
- **Board.** A second PYNQ-Z2 boots from SD (BootROM status `0x00400000`, no
  error) into PYNQ Linux 3.0.1, user `student`, passwordless sudo. It is
  cabled directly to the PC: board 10.43.0.1, PC 10.43.0.2. No rebuild was
  needed. The Vitis 2026.1 `riscv64-unknown-elf-gcc` (GCC 13.4) reproduces
  both released firmware images byte for byte.
- **Change 1: start-up scripts.** Under `sudo`, PYNQ reported "No Devices
  Found" because `sudo` drops `XILINX_XRT`. `start_board.sh` (runs on the
  board) sets it, loads the bitstream and starts the server in the background.
  `run_board.ps1` (PC) copies the files over SSH and runs it.
- **Finding.** 40/40 vectors were bit-exact at once, but a stream probe with
  known vectors showed every positive window as *confirmed*, even a single one
  after two negatives (expected: +,+,−,+ → 2,3,0,3; measured 3,3,0,3).
  Evidence:
  - The disassembled firmware implements "2 of 3" correctly.
  - The history words on the stack showed a stream window that ended 747
    cycles before the current one started, with no request sent in between.
    Every request ran twice, so its second pass confirmed its first.
  - A debug firmware recorded the sequence number it accepted. It was 0.
    The core then acknowledged 0 and ran the real request again.
  - Cause: `struct.pack_into('<I', mmap, …)` in `board_server.py`. CPython
    zeroes the four bytes and then writes them one byte at a time, so the
    polling core briefly sees `seq_in = 0`. The JTAG relay writes whole words
    (`mwr`), so it never showed this. The same pattern also briefly released
    the CPU reset whenever the server tried to hold it.
- **Change 2: `board_server.py`.** All register and mailbox words go through
  32-bit `memoryview`s (one bus access each). Firmware and bitstreams are
  unchanged. A firmware-side double read of `seq_in` was tried first; it
  reduced but did not remove the problem, because the zero is a real
  intermediate value, not a read glitch. It was discarded.
- **Outcome** (both `keyword.bin` and `keyword_kdot.bin`, over Ethernet):
  - 40/40 vectors bit-exact (3 × 40 for RV32IM). Cycles equal RTL; worst
    `kdot` case 267,393 cycles = 2.67 ms.
  - Stream sequences as specified, 3 repetitions each: + → 2;
    −,−,+ → 0,0,2; +,+,−,+ → 2,3,0,3; −,+,−,+ → 0,2,0,3.
  - LED0 went off 999 ms after a detection (read back from syscon `LED`).
  - PC round trip about 7 ms per window (JTAG relay: 30–45 ms). The live
    microphone demo streams one window per 250 ms.
  - Windows voices (David, Zira): "yes" detected in both single and stream
    mode (first window unconfirmed, then confirmed); "no", "pizza" and
    "hello" rejected in both modes.
- **Unaffected earlier results.** `confusables.json`, `tune_stream.py` and
  the manual microphone test (over JTAG) did not use `board_server.py`.

### 19. Iteration loop, round 1: what limits the streaming SNN

All numbers below are validation data. The test split is untouched.

**The microphone drop is extrapolation.** `mic_diagnose.py` splits the
held-out microphone chain for the selected model:

| Condition | Live recall |
|---|---|
| clean | 59.7% |
| full chain | 42.6% |
| EQ only | 45.1% |
| gain + clipping only | 59.4% |

The worst profiles all have low-pass corners at 4.6-5.4 kHz, which remove
the /s/ of "yes". The training microphones use even third-octave bands of
4.5-7.6 kHz, so their only low-pass band is b = 8 (5.66-7.13 kHz). Held-out
b = 7 was never bracketed. `channels.RANGES['wide']` (training only,
`--mic-ranges wide`) draws low-pass 3.5-7.9 kHz, high-pass 40-500 Hz and
peaks 100-7000 Hz. Every held-out band then has even bands on both sides; the
held-out profiles are unchanged (checked). Run `s1_wide` (50 epochs instead of
30) is training.

**The 1 h stream cannot set a ≤ 2 FA/h threshold.** At the operating point
it holds 2 false accepts (Poisson 95%: 0.24-7.2 per hour); the threshold is
the third-highest LibriSpeech peak.

`robust_eval.negative_stream` is a negatives-only stream of 17.85 h: all of
LibriSpeech dev-clean + dev-other without "yes" in the transcript, and every
non-"yes" validation word. `stream_select.py` now uses it
(`--fa-source negatives`), with exact recall steps as candidates.

For the selected model, the honest operating point is **55.9%** live recall
at 1.90 FA/h (threshold 9955), not 59.7%. Its deployed threshold 9662 gives
more than 2 FA/h.

**Where it fails** (`diagnose_stream.py`, `results/diagnose_s2_nokd_qat.json`):
- False accepts are almost all LibriSpeech read speech. At 65/70/75% live
  recall the 1 h stream has 4/6/10 FA/h, of which 3/5/8 are LibriSpeech.
- Misses depend only weakly on SNR (53 → 65% across terciles) and level
  (56 → 66%). Clips whose /s/ is cut off (11% of "yes") reach 33%, against
  63% for the rest. Missed clips sit close to the threshold (median −0.6 SD).

**Decision rule (no retraining):** detect on the integer sum of the last W
frame scores, using the long negatives and the same rule.

| W | Live recall | FA/h | MSWC recall |
|---|---|---|---|
| 1 | 55.9% | 1.90 | 38.3% |
| 5 | 58.7% | 1.96 | 41.9% |
| 10 | 60.2% | 1.96 | 39.2% |
| 20 | **61.0%** | 1.96 | 38.9% |

With W = 20 the median latency is still 0.19 s. `model.decision_scores` is
the reference; it is implemented in `snn_stream` detectors,
`firmware/stream_main.c` (int64 running sum, MB[19] = W), `sim/soc_stream.cpp`,
`board_server.SimulatedBoard` and `jtag/make_stream_vectors.py`. It is stored
as `decision_window` in the model.

**Unfair comparison in the targets.** Under the same rule, the release
window model reaches 38.3% live recall (1 h stream) against the SNN's 59.7%.
The target "live recall ≥ 67.3%" puts the SNN at ≤ 2 FA/h against the
release at its own operating point (about 52 FA/h on test). The targets are
unchanged here; results are reported both ways from now on.

**Cascade** (`cascade_eval.py`: SNN proposes, release window model confirms
within −0.25/+0.5 s). The best pair gives 63.0% against 59.7% live recall
and 54.8% against 43.4% MSWC recall. Costs: live false accepts 0.17% against
0.10%, and other MSWC words 1.26% against 0.54%. The gain comes from one grid
point on the 1 h stream, so it is not trusted until re-scored on the long
negatives. It would also put two models on the board.

**Running:**
- `s2_mine`: stage 2 from the same stage 1 as the selected model. A quarter
  of each batch is LibriSpeech-only 3 s streams, half of them mined every 50
  steps as the 256 highest-"yes" segments out of 2048
  (`train_stream.py --speech-streams --mine-every`).
- The leave-MSWC-out run was stopped after 9 epochs. It will be rerun on the
  final recipe, where it says more.

**Hardware:** `build/keyword_engine_abi3.bit` (`ps_if.v` ABI `0x00020003`)
first missed timing by 0.023 ns. The path is `acc[idx] += w`: a 128-way
18-bit read multiplexer, adder and write decode in one cycle, 11 logic
levels, 64% routing. Re-implemented with Performance_ExplorePostRoutePhysOpt:
WNS +0.348 ns, 12,894 LUTs (`results/engine_abi3_*.rpt`). The design sits at
the edge. The durable fix is pipelining the read-modify-write with
forwarding. Not yet loaded on the board.

**Ethernet path:** the board is at 192.168.2.106 on the direct link (SSH,
Jupyter). Copying files to it was not permitted in this session.
`net_board_test.py` (v2 and v3 over TCP against the integer oracle) passes
against the v3 simulator and, through the JTAG relay, against the release on
the board (40/40).

### 20. Iteration loop, round 2: mining, label conflicts, and the front end

All numbers are validation data, selected with `stream_select.py` on the
long negatives. Live recall rests on 397 positives (SE about 2.5 points).

**Float models from the same stage 1** (live recall at ≤ 2 FA/h and ≤ 0.2%
live other-word accepts):

| Stage 2 | W=1 | W=10 | Other words accepted (live / MSWC) |
|---|---|---|---|
| baseline `s2_nokd_seed0` | 48.9% | 54.7% | 0.07% / 0.45% |
| speech mining `s2_mine` | 55.2% | 53.9% | 0.20% / 0.88% |
| wide mics `s2_wide` (from `s1_wide`) | 49.9% | 57.2% | 0.17% / 0.65% |

- **Speech mining.** It removes the LibriSpeech false accepts; with W=1 it
  gains 6 points. The word limit then binds instead: other words
  accepted hit 0.2%, and MSWC doubles. With W=10 there is no gain. Smoothing
  and mining remove the same false accepts.
- **`s1_wide`** (wide mics, 50 epochs) reaches 79.4% against 77.6% 35-word
  accuracy. Its stage 2 is within noise of the baseline.

**Label conflict found by word mining.** The hardest non-"yes" training words
(`s2_wide_mine2`, mined margin stuck at about 7 against a threshold of about
8) are mostly TTS pseudo-words that contain a whole "yes" followed by one
letter: yesd, yest, yesf, yesv, yesq, ... (`make_tts_negatives.py`,
GraphemeAug insertions). A causal detector that must fire within 0.35 s
cannot tell "yes" from "yes"+d without waiting for the burst. Labelling them
negative pushes every real "yes" down, and no user says them.
`--exclude-words 'yes[a-z]'` removes the 22 words (3,018 clips, 1.9%) from
training only; the evaluation sets are unchanged. Run `s2_wide_mine3`
(identical to `mine2` apart from the exclusion) is training.

**Microphone hypothesis rejected.** Wide training mics do not reduce the
held-out drop:

| Model | Clean | Full chain |
|---|---|---|
| `s2_wide` | 57.2% | 38.8% |
| baseline | 54.7% | 38.0% |

Per-profile recall in `mic_diagnose.py` rests on about 25 clips, and each
profile has different clips, so the earlier per-profile reading ("low-pass
profiles are the worst") was weak.

`band_sensitivity.py` is the controlled version: the same clips, one filter
at a time.
- Every model needs the 5-7 kHz band (/s/). A 5 kHz low-pass takes recall
  from about 57% to 32-38%; wide training buys about 5 points under low-pass.
  This is structural: without the /s/, "yes" is "ye(ah)".
- Low cuts do not hurt.
- **Level does:** −10 dB gain costs 2-12 points, −20 dB costs 13-27 points,
  for every model.

**Cause: the log-mel window.** The streaming front end maps −80…0 dBFS to
uint8. At normal level, 35% of all values sit on the floor and none come near
the top: the loudest speech bands have a median of 92/255, about −51 dB.
At −20 dB input, 46% of the > 3.5 kHz values of the loudest word frames are
on the floor, so the /s/ is erased before the network sees it.

`features.LOGMEL_RANGE['logmel_w']` = −120…−20 dB (0.39 dB per step) moves
the window down. numpy and torch agree bit for bit, and `logmel` is
unchanged. Stage 1 `s1_logw` (as `s1_wide`, `--frontend logmel_w`) is
training. The front end runs on the PC, so this needs no hardware change.

**Engineering:**
- `stream_select.py` scores raw traces, so a stored window is not applied
  twice.
- `tests/test_pipeline.py` has a moving-sum test across 13-frame requests
  and a reset.
- `verify_stream_rtl.py --tag` keeps each model's results. The W=20 run had
  overwritten the W=1 engine CSV; its files are now `*_w20`.

### 21. Rule: words that begin with "yes" are "don't care" (agreed with the user)

A causal detector cannot tell "yes" from "yesterday", "yessir" or "yes" +
one letter before the rest of the word arrives. Waiting would add about
100-200 ms to every detection. From now on these words are neither
negatives nor false accepts:
- Evaluation: `robust_eval.yes_prefixed`.
  - LibriSpeech utterances with any YES… word are left out of both streams
    (validation: 6 utterances).
  - MSWC "yesterday" moves from the ye- group to its own `YES_PREFIXED`
    group, reported but not counted. It was never in "random other".
  - TTS "cheese/yesterday" is split.
  - `stream_select`/`cascade_eval` other-word accepts leave these words out.
- Training: `--exclude-words 'yes.+'` (yes + letter, yesterday: 3,424
  clips). The packed LibriSpeech training stream still holds about 60
  YESTERDAY utterances (0.2%); it is repacked when no run reads it.

Numbers before this entry count "yesterday" detections as false accepts,
which is slightly pessimistic (6 of about 5,600 negative utterances).

**Word mining with the conflict in place fails badly.** `s2_wide_mine2`
(speech + word mining, pseudo-words included) reaches 39.8% live recall
(W=10) against 57.2% for `s2_wide`, and MSWC 29.2% against 43.4%. The mined
pool was mostly yes + letter, so 30% of word slots taught "yes is negative".
`s2_wide_mine3` differs only by excluding `yes[a-z]`.

### 22. Round 3: mining without the conflict, the front-end test, keyword change

**`s2_wide_mine3`**: `mine2` without the yes + letter pseudo-words. Live
recall 46.1% at W=10, against 39.8% for `mine2` and 57.2% for `s2_wide`.
Removing the conflicting words recovers most of the loss. Mining as set up
(30% of word slots from a mined pool, a quarter of the batch speech-only)
still costs recall. Mining is not used further.

**`logmel_w` is not adopted.** Same stage-2 recipe as `s2_wide`, on
`s1_logw`. `band_sensitivity.py`, same clips:

| Condition | `s2_wide` | `s2_logw` |
|---|---|---|
| Clean | 57.2% | 48.1% |
| −20 dB | 30.0% | 53.9% |
| −10 dB | 44.6% | 53.1% |
| +10 dB | 52.6% | 35.5% |
| Low-pass 4.5 kHz | 30.0% | 37.3% |
| Live recall at ≤ 2 FA/h (W=10) | 57.2% | 48.1% |

Moving the floor fixes quiet input, but the model is still level-dependent;
its best level moved and loud input now fails. Any absolute log scale
teaches the network a preferred level. Next: level invariance by
construction, with a causal gain control relative to a running peak
(`logmel_agc`).

**Keyword changed to "sheila"** (user request).
- `keyword_config.py` (env `KWS_KEYWORD`, default "yes") drives data
  selection, training, evaluation, vectors and TTS.
  - Models record their keyword, and the detectors refuse a mismatch.
  - "yes" reproduces exactly: the derived labels equal `features.npz`'s y.
- Speech Commands v2 has 2,022 "sheila" clips (1,606 / 204 / 212); the
  validation live set holds 204 positives (SE about 3.4 points).
- MSWC has almost none (16 train, 2 dev, 2 test; "shiela" is treated as an
  alias, never a negative). It supplies near-miss negatives only; the
  corpus-shift test must come from held-out TTS voices.
  - `data/mswc/*_selected_sheila.csv`: 7,523 train, 3,634 dev and 3,634
    test clips over 463 words.
  - MSWC dev and test share 770 speakers (the official splits are not
    speaker-disjoint), so thresholds chosen on dev can slightly favour
    test speakers.
- TTS generation for "sheila" is ready (`data/tts_sheila`, 348 negative
  words, none containing a whole "sheila") and waits for the user's choice
  of data sources.
- Stage 1 is keyword-agnostic (35 words), so `s1_wide` is reused.
  `sheila_s2_wide` trains on the existing multi-corpus data through a
  junction `data/multi_sheila` → `data/multi`.

**Verifier track** (branch `explore-verifier`):
- v2 had keyword AUC 0.39 (it ranked other words above "yes").
- v3, with `logmel_w` and a clean-audio warm-up, reached AUC 0.976.
- RTL: 10.5M cycles per verification with kdot, bit-exact; 121 KB of the
  144 KB model region with stage 1.
- Its keyword target changes to SH IY L AH.

### 23. "sheila": first baseline, the teammate's clip model under our rule, AGC stage 1

**First "sheila" baseline** (`sheila_s2_wide`, stage 2 on `s1_wide`,
validation, ≤ 2 FA/h on 17.85 h of negatives):

| W | Live recall | Complete recordings | Other words accepted | 1 h stream recall |
|---|---|---|---|---|
| 1 | 62.7% | | 0.00% | 72.1% |
| 10 | 63.7% | 70.3% | 0.00% | 73.5% |
| 20 | 61.3% | | 0.00% | 70.6% |

- The word limit never binds: false accepts come from continuous speech,
  and "zero" becomes the main word false accept above 70% recall.
- 22.5% of the "sheila" positives (46/204) are cut off by the 1 s
  recording window. Their recall is 41%, against 70% for complete ones;
  stream_select now reports `live_recall_complete`.
- `sheila_s2_clip` (`--clipped-dontcare`: truncated positives get no
  target) is trained and being selected.

**Teammate's model (`upstream/damien-dicking-around`, "sheila v2")** scored
with our harness (`damien_detector.py`, `results/sheila_damien_ourrule.json`):
- The port is bit-exact: all 40 of his board vectors, and his features
  computed from the WAVs.
- It runs a 1 s window every 250 ms, as in his live demo; the score is his
  spike margin.
- At his margin 2: 83.3% live recall, but 2.8% other words accepted and
  **324 FA/h** on the negatives.
- At ≤ 2 FA/h: **8.3%** live recall.
- Our streaming SNN reaches 63.7% at the same rate. His 98% clip accuracy
  (with 1.2% word false accepts on the test clips) does not carry over to
  continuous audio.
- His finding worth keeping: for "sheila" the information sits below about
  5.1 kHz (no gain from a 6.5 kHz cutoff), unlike the /s/ of "yes".
  `band_sensitivity.py` for "sheila" is queued to check this for our models.

**Alex's `snn_layer.v`** (`upstream/Alex-parallel`):
- 64 parallel LIF neurons fed by a row-wide weight BRAM, the same idea as
  the neuron engine.
- The file itself says "SKETCH. NOT compiled, NOT simulated, NOT
  synthesized".
- It removes kdot and the hardware LED timer from `spike_soc.v`, and uses
  0x1000_4000, where the engine sits. Merging would conflict.

**`s1_agc`** (AGC front end) reaches 75.5% 35-word accuracy, against 79.4%
for `s1_wide`: level invariance costs 4 points on clean clips.
`sheila_s2_agc` is training.

**Resources:**
- My own TTS generation (6 Piper workers of 1.9 GB) exhausted memory and
  killed `s1_agc` at epoch 31; it was rerun, and TTS now runs with 2
  workers.
- A `pytest` from 2026-09-27 had hung for two days (1.6 GB); it was
  stopped.

### 24. "sheila": truncated positives, band sensitivity, dedicated data

**Ignoring truncated positives hurts.** `sheila_s2_clip` (W=10) reaches
51.0% live recall and 60.1% on complete recordings, against 63.7% and 70.3%
for the baseline. With only 1,606 real positives, dropping 22% costs more
than the truncation noise does. Positive data is the bottleneck.

**Band sensitivity** (`results/sheila_band_sensitivity.json`, same clips,
baseline model):
- Low-pass at 4.5 kHz: 70.1% against 63.7% clean; at 3.5 kHz: 61.8%.
- For "yes", a 5 kHz low-pass halved recall. "sheila" does not need the high
  band, which confirms the teammate's finding; the held-out microphone
  problem should be much milder.
- The high band seems to add distracting variation. A front end limited to
  about 5 kHz, with the 24 bands placed where the information is, is a
  candidate.
- Level still matters: −20 dB costs 10 points and +10 dB costs 5. The AGC
  run is in progress.

**Dedicated data:**
- TTS for "sheila" is done: 55,000 utterances in `data/tts_sheila`, 8,000
  training positives, plus a held-out voice-model test set.
- `data/multi_sheila_full` (`KWS_MULTI`) is being built from Speech
  Commands v2, the "sheila" MSWC selection and this TTS.
- `keyword_config.TTS` and `KWS_MULTI` make these paths per keyword; "yes"
  keeps `data/multi` and `data/tts`.

### 25. "sheila": four attempts fail to beat the baseline; seed variance first

Validation, ≤ 2 FA/h on 17.85 h of negatives, best window per model:

| Model | Live recall | Complete recordings | Other words accepted |
|---|---|---|---|
| baseline `sheila_s2_wide` (W=10) | **63.7%** | 70.3% | 0.00% |
| `--clipped-dontcare` (W=10) | 51.0% | 60.1% | 0.00% |
| AGC front end `sheila_s2_agc` (W=20) | 57.8% | 67.7% | 0.13% |
| + MSWC + TTS data `sheila_s2_tts` (W=10) | 57.8% | 62.0% | 0.03% |
| baseline + PC low-pass 4.5 / 5.5 kHz | no threshold meets the rule | | |

- **AGC:** level-invariant as designed (−20 and −10 dB give identical
  recall), but it costs about 6 points at normal level and breaks under
  low-pass filtering (48.5% against 70.1%): its reference is the loudest
  band. Across `logmel_w`, AGC and earlier PCEN, the plain log-mel stays best.
- **Low-pass:** the band sweep showed 70.1% recall under a 4.5 kHz low-pass
  at the model's own threshold. Under the full rule no threshold is
  feasible: the filtered negatives score above every positive. A
  recall-only robustness sweep must not drive a decision; only the full
  rule (both false-accept limits) counts.
- **TTS + MSWC data:** worse on real speech (the live set is Speech
  Commands). Synthetic positives and extra near-misses shift the model away
  from the target domain. Recall on the held-out TTS voices was not measured.

**Method check before a fifth attempt:**
- 204 positives mean an SE of about 3.4 points per model. Seed-to-seed
  training variance is unknown, so "63.7% against 57.8%" may be partly
  noise.
- Two more seeds of the baseline recipe (`sheila_s2_wide_seed1`,
  `_seed2`) are training. Differences smaller than their spread will not be
  called.

### 26. Seed variance of the "sheila" baseline, and the revised verdicts

Same recipe as `sheila_s2_wide`, seeds 0/1/2 (best window per seed):

| Seed | Live recall | Complete recordings |
|---|---|---|
| 0 | 63.7% | 70.3% |
| 1 | 57.4% | 62.0% |
| 2 | 63.7% | 72.8% |

Mean 61.6% ± 3.6 (SD); complete recordings 68.4% ± 5.6.

Revised verdicts from entry 25:
- AGC and TTS data (57.8%) are within one SD of the recipe's own seed mean.
  The verdict is "no measurable gain", not "worse".
- Ignoring truncated clips (51.0%) and the low-pass (no feasible
  threshold) are genuinely worse.

Rule from here: a single-run difference under about 7 points (2 SD) is not
called. Changes need several seeds or a paired comparison.

**Next:** QAT of seeds 0 and 2 (`sheila_qat_seed0/2`, 8 epochs, as the
"yes" QAT). Then integer export, the long-negatives selection on the integer
model, and the RTL check.

### 27. "sheila" candidate: QAT, RTL, and the one-time test run

**QAT and integer export** (`sheila_qat_seed0/2`, 8 epochs from the float
seeds). Validation, integer models, full rule:

| Model | Live recall | Complete recordings | FA/h |
|---|---|---|---|
| **seed 2, W=1** | **66.7%** | **75.9%** | 1.44 |
| seed 0, W=1 | 59.3% | 67.1% | 1.99 |

- As for "yes", QAT costs nothing: seed 2 went from 63.7% as a float model
  to 66.7% as an integer model.
- The selection picks W=1 (W=10 ties).
- Candidate: `runs_stream/sheila_qat_seed2/int_model.npz`
  (stream_threshold 18462, decision_window 1, logmel).

**RTL** (`verify_stream_rtl.py --engine --tag sheila`): 40 streams (20
"sheila" and 20 other test clips), 355 hops, bit-exact with the oracle,
including detections, malformed requests, reset, duplicate sequence and the
1 s LED pulse. Worst hop 1,029,278 cycles (10.29 ms at 100 MHz), mean 9.89 ms.

**Test split, once** (`final_eval.py`, `results/final_sheila_test.json`).
Thresholds were fixed on validation; the negatives are 18.97 h of
LibriSpeech test-clean + test-other and every non-"sheila" Speech Commands
test word.

| Model | Live recall | Complete recordings | Other words accepted | FA/h |
|---|---|---|---|---|
| ours, clean | 63.7% | 70.7% | 0.00% | **1.58** |
| ours, held-out mics | 66.5% | 70.1% | 0.03% | |
| ours, real rooms | 60.9% | 69.5% | 0.20% | |
| ours, mics + rooms | 58.0% | 64.9% | 0.40% | |
| teammate's clip model, threshold for ≤ 2 FA/h (validation) | 7.1% | 8.1% | 0.00% | 2.11 |
| teammate's clip model, his margin 2 | 80.2% | 85.1% | 3.53% | 366 |

- Validation predicted the test well: recall 66.7% → 63.7% (SE about
  3.3), false accepts 1.44 → 1.58 per hour, within the ≤ 2/h target.
- No microphone drop for "sheila" (+2.8 points, noise), where "yes" lost
  17-20 points: the word does not depend on the band above 5 kHz. Rooms cost
  about 3 points, mics + rooms about 6.
- 1 h test stream: 69.3% recall at 1 FA/h, median latency 0.04 s after
  the end of the word.
- At an equal false-accept rate the streaming SNN detects 9× more
  "sheila"s than the clip classifier.

**Not done yet:**
- The board test of this model (JTAG or the Ethernet path, needs the user's
  go-ahead).
- Native C on the full test set (`verify_stream.py`).
- The phoneme verifier retargeted to SH IY L AH.

### 28. The phoneme verifier for "sheila": stage 1 proposes, the verifier confirms

Branch `explore-verifier` (Rework merged into it). The verifier of entry 22 was
retargeted from Y EH S to SH IY L AH and put behind the "sheila" candidate
(`runs_stream/sheila_qat_seed2/int_model.npz`).

**Retargeting.**
- `verifier_model.keyword_score` and `firmware/verifier.c` now score any
  phoneme sequence. A CTC blank is required between two equal phonemes.
  On random logits the result is identical to the old Y EH S code (numpy
  and torch). Native C is bit-exact.
- Training (`train_verifier.py`, `KWS_KEYWORD=sheila`):
  - LibriSpeech train-clean-100 plus the keyword clips of `data/multi_sheila`,
    with "sheila" as 25% of each keyword sub-batch;
  - the stage-1 front end (`logmel`), so the board needs no second frame stream;
  - 4000 steps of clean warm-up, 40 epochs, 47 min.
- Validation clips: AUC 0.993; 95.6% of "sheila" clips score above the 99th
  percentile of other words. For "yes" the verifier reached AUC 0.976, and the
  99th percentile of other words was already a perfect path. Four phonemes
  are easier to verify than three.
- The last epoch is used (no checkpoint selection): the verifier's
  validation clips are the same Speech Commands speakers as the live set.

**Where stage 1 alone stands** (`verifier_cascade.py sources`, validation,
18.1 h of negatives, integer model):

| Stage-1 live recall | FA/h | Other words accepted |
|---|---|---|
| 65% | 1.2 | 0.00% |
| 75% | 8.1 | 0.20% |
| 85% | 30.3 | 1.07% |
| 90% | 59.5 | 2.67% |

The false accepts are LibriSpeech "she ..." phrases and Speech Commands
clips labelled "zero", "four" and "two".

**The cascade.** When stage 1 reaches t1 in request k or k-1, the verifier
scores the last 1.5 s (150 frames) at the end of request k. The selection
rule is unchanged: ≤ 0.2% of live other words and ≤ 2 FA/h on the negatives.
Results are on validation, with t1 and t2 chosen there
(`results/sheila_cascade_logmel.json`):

| | Live recall | Complete recordings | Other words | FA/h | 1 h stream recall | Latency (median) |
|---|---|---|---|---|---|---|
| stage 1 alone (t = 18462) | 66.7% | 76.0% | 0.03% | 1.44 | 58.1% | 0.07 s |
| **cascade** (t1 = 13744, t2 = −11) | **82.8%** | **90.5%** | 0.00% | 1.99 | 82.4% | 0.27 s |
| stage 1 alone, held-out mics | 61.8% | 70.3% | 0.03% | | | |
| cascade, held-out mics | 85.3% | 91.8% | 0.07% | | | |
| stage 1 alone, mics + rooms | 59.3% | 65.8% | 0.20% | | | |
| cascade, mics + rooms | 86.8% | 93.7% | 0.23% | | | |

- **Gain:** +16.2 points of live recall. 35 clips are found only by the
  cascade and 2 only by stage 1 (exact McNemar p < 0.001). That is well
  outside the seed noise of stage 1 (SD 3.6 points).
- **Microphones and rooms:** under held-out mics and rooms the cascade does
  not lose recall. These conditions flatten stage 1's scores: fewer clips
  exceed 18462, more exceed 13744. The verifier passes 89-91% of keyword
  clips in every condition.
- **Cost:** 53 verifier calls per hour of negatives. The latency grows by
  the 107 ms verifier run and one 250 ms request.
- **MSWC near-miss words:** the /ʃiː/-onset group (she, sheep, shield, ...)
  goes from 13/183 to 26/183 accepted. Random MSWC words are unchanged at
  2/3000.

**What remains** (the 36 false accepts in 18.1 h at the operating point):
- 25 are LibriSpeech phrases where "she", "sure" or "shore" is followed by a
  vowel, L, W or N ("she added", "she laughed", "she will", "sure thing").
  These are true acoustic neighbours of SH IY L AH.
- 11 are Speech Commands clips labelled "zero", "two" or "four". The
  verifier's greedy decoding of every one of them is SH IY L AH, and stage 1
  also fires on them. They are probably mislabelled "sheila" recordings; one
  speaker (e11fbc6e) has two. Across all 9,777 negative Speech Commands
  clips, 30 get a perfect verifier path (18 "zero"). The list is in
  `results/sheila_suspected_label_errors_validation.json`. They stay in the
  evaluation until someone listens to them.

**Firmware and RTL.** A `-DCASCADE` build of the stream firmware runs the
cascade on the PicoRV32 itself: the LED follows confirmed detections only.
- `verifier_cascade.cascade_requests` is its oracle. It equals the
  evaluation code on 240 validation clips (269 verifier runs).
- `verify_verifier_rtl.py --cascade 13744 -11` on the SoC RTL: 30 records
  (keyword clips, other words, 30 s pieces of the negatives stream), 934
  requests, bit-exact, 41 verifier runs, 13 detections with the LED on.
- Worst request (stage 1 plus the verifier): 11.72M cycles, 117 ms at
  100 MHz, within the 250 ms request.
- Both models use 121 KB of the 144 KB model region.

**Critique.**
- t1 and t2 were chosen on the same validation data, from about 31
  verifier thresholds times all stage-1 thresholds, so the numbers above
  are optimistic. The test split has to confirm them.
- One verifier seed so far.
- The remaining acoustic false accepts ("she l...") need a verifier that
  separates L AH from L AE / W / N after SH IY. Near-miss words in training
  (MSWC "she", "sheep", "shell", ...: `data/multi_sheila_full`) are the
  next experiment.

### 29. The cascade candidate, and its one-time test run

**Three more verifier variants, validation only** (stage 1 unchanged, same rule):

| Verifier | Live recall | Complete recordings | FA/h | MSWC "she-" words accepted | MSWC other words |
|---|---|---|---|---|---|
| seed 0 (entry 28) | 82.8% | 90.5% | 1.99 | 26/183 | 1.02% |
| seed 1 | 82.4% | 89.9% | 1.99 | 22/183 | 0.94% |
| near-miss words 15% (`data/multi_sheila_full`, real keyword clips only) | 79.4% | 88.6% | 1.99 | **4/183** | **0.30%** |
| capped-margin score, cap 0.5 logit (seed 0) | 72.1% | 81.0% | 1.99 | 13/183 | 0.63% |

- **Seeds:** both select the same t1 and t2, and recall differs by
  0.5 points. The verifier adds little seed variance.
- **Near-miss training** almost stops isolated "she", "sheep" and "shield"
  from being accepted: 4/183, where stage 1 alone accepts 13/183. It costs
  3.4 points of live recall (8 clips found only by seed 0, 1 only by it;
  p = 0.04). The rule ranks by live recall, so seed 0 stays the candidate.
  The near-miss verifier is the option if isolated near-miss words matter
  more than recall.
- **Capped margin** (`verifier_model.step_costs`): a step whose label wins
  adds up to `cap` instead of 0. Every cap loses recall: 72%, 65%, 58% and
  39% for 0.5, 1, 2 and 4 logits. Clean read speech ("she laughed") earns
  larger margins than keywords in noise. The score that saturates at 0 is
  the better one here.

**Candidate:** stage 1 `sheila_qat_seed2` (W = 1) with verifier
`runs_verifier/sheila_logmel/last.pt`, t1 = 13744, t2 = −11, policy (a).

**Test split, once** (`verifier_cascade.py final`,
`results/final_sheila_cascade_test.json`). Thresholds come from
validation; the negatives are 18.97 h of LibriSpeech test-clean and
test-other plus every non-"sheila" Speech Commands test word.

| | Stage 1 alone | Cascade |
|---|---|---|
| Live recall | 63.7% | **84.0%** |
| Complete recordings | 70.7% | **90.2%** |
| Held-out mics | 66.5% | 82.6% |
| Real rooms | 60.9% | 83.0% |
| Mics + rooms | 58.0% | 82.6% |
| Live other words accepted | 0.00% | 0.10% |
| **FA/h on the negatives** | 1.58 | **2.37** |
| 1 h stream: recall, false accepts | 69.3%, 1 | 84.3%, 0 |
| Latency after the word (median, p90) | 0.04 s, 0.11 s | 0.25 s, 0.38 s |
| MSWC other words | 0.83% | 1.13% |

- The stage-1 column equals entry 27's `final_eval.py` report, so the
  cache pipeline and the old evaluation agree.
- **Recall:** +20.3 points (44 clips only by the cascade, 1 only by stage
  1). The gain holds under held-out mics and rooms (+16 to +25 points).
- **False accepts: the target is missed.** 2.37 FA/h against ≤ 2.
  Validation chose the threshold at the edge of the budget (1.99). Stage 1
  alone went from 1.44 to 1.58 between validation and test (+10%); the
  cascade went from 1.99 to 2.37 (+19%).
- **What the 45 test false accepts are:**
  - 33 are LibriSpeech "she"/"sh-" phrases.
  - 12 are Speech Commands clips labelled nine, two, zero, seven, six or
    four. The verifier decodes every one as SH IY L AH, and two speakers
    have two clips each. Validation had the same pattern (entry 28).
  - If these clips are mislabelled "sheila" recordings, the real rate is
    33 / 18.97 h = 1.74 FA/h. This is unconfirmed: nobody has listened to
    them yet (`results/sheila_suspected_label_errors_{validation,test}.json`).
- For the next candidates, validation needs a safety margin on the FA
  budget, for example ≤ 1.6 FA/h. This rule is introduced after seeing one
  test result and is recorded as such.

### 30. A keyword head on the verifier: 91% live recall on test

The first cascade missed the false-accept target on test (entry 29). The
remaining false accepts were "she ..." phrases that the phoneme verifier
passes. All results below are on validation with the stricter budget
(≤ 1.6 FA/h, entry 29) unless marked as test.

**Hard negatives from stage 1** (`mine_verifier_negatives.py`). The
stage-1 float model runs over the verifier's own training speech
(LibriSpeech train-clean-100, 17,698 utterances, 17 min on the GPU).
The 8000 highest proposal peaks are exactly the confusions seen on
validation: "she had three", "she held up", "sure ned land", "cheer up",
"down to the shore".

**Three ways to use them, and policy (b):**

| Verifier | Live recall | Complete recordings | FA/h |
|---|---|---|---|
| phoneme path, seed 0 (entry 28) | 76.5% | 86.1% | 1.49 |
| policy (b): no new phoneme within 200 ms after AH | 76.5% | 84.2% | 1.55 |
| path + hinge fine-tuning on the mined windows (10 epochs) | 77.0% | 86.7% | 1.49 |
| **keyword head**, seed 0 | **89.2%** | **95.6%** | 1.55 |
| keyword head, seed 1 | 87.8% | 94.9% | 1.38 |

- **Policy (b)** does not help. A CTC model is "peaky": blank wins most
  frames, so a phrase that goes on after "she l..." rarely breaks the
  boundary. It costs 0.15 s of latency.
- **Hinge fine-tuning** (`train_verifier.py --init --hard`): the mined
  windows' path score is pushed below −2 logits, and keyword clips above
  −0.5. There was no gain. The path score only asks whether the best
  phoneme path runs through SH IY L AH, so it leaves little room to rank
  "she laughed" below "sheila". The GPU version of the path DP made this
  run CPU-bound (14 min per epoch). The DP now handles all states as one
  tensor: 2.5× faster on the GPU, 9× on the CPU, with identical values
  and gradients.
- **Keyword head** (`--head-weight 1`): a linear readout of GRU 2 per step,
  whose maximum over steps is the decision. It is trained with binary
  cross-entropy: keyword clips against the other words of the keyword
  sub-batch and the mined windows. The CTC losses stay, starting from
  verifier seed 0, 10 epochs, 20 min.
  - On clips: AUC 0.996 against other words and 0.985 against near-miss
    words (path score: 0.93).
  - The cascade can then lower the stage-1 threshold from 15226 to 7080,
    and the verifier rejects what stage 1 lets through.
  - Two seeds: 89.2% and 87.8%.

**Firmware.** `verifier.c` computes the head per step: 64 MACs through
kdot, and a running maximum. The cascade build decides on the head when the
verifier has one (MB[32]); MB[26..29] stay the path scores.
- Native C is bit-exact: logits, path scores, head.
- SoC RTL (`verify_verifier_rtl.py --cascade 7080 1588`): 934 requests
  bit-exact, 75 verifier runs; 10/12 keyword clips detected, 0/12 other
  words. Worst request 117 ms at 100 MHz; 121 KB of the 144 KB model region.
- The board vectors (`jtag/make_stream_vectors.py --cascade`) include the head.

**Test split, once** (`results/final_sheila_head_cascade_test.json`).
Head seed 0, t1 = 7080, t2 = 1588, chosen on validation. Negatives are
18.97 h.

| | Stage 1 alone | Path cascade (entry 29) | **Head cascade** |
|---|---|---|---|
| Live recall | 63.7% | 84.0% | **91.0%** |
| Complete recordings | 70.7% | 90.2% | **96.6%** |
| Held-out mics | 66.5% | 82.6% | 88.7% |
| Real rooms | 60.9% | 83.0% | 90.6% |
| Mics + rooms | 58.0% | 82.6% | 87.7% |
| Live other words accepted | 0.00% | 0.10% | 0.20% |
| **FA/h** | 1.58 | 2.37 | **2.11** |
| 1 h stream: recall, false accepts | 69.3%, 1 | 84.3%, 0 | 91.4%, 2 |
| Latency after the word (median, p90) | 0.04 s, 0.11 s | 0.25 s, 0.38 s | 0.18 s, 0.33 s |
| Verifier calls per hour of negatives | | 56 | 663 (107 ms each, 2% of the time) |

- **Recall:** +27.3 points over stage 1 alone (58 clips only by the
  cascade, 0 only by stage 1). The gain holds under held-out mics and
  rooms (+22 to +30 points).
- **False accepts: the target is missed narrowly**, 2.11 against ≤ 2.
  Validation said 1.55: the stricter budget absorbed less than this
  model's validation-to-test increase (+36%).
- **What the 40 test false accepts are:** 24 LibriSpeech phrases (1.27
  FA/h) and 16 Speech Commands clips labelled zero (7), two (5), four (2),
  seven or left. The verifier's phoneme decoding of all 16 ends in IY L AH:
  8 × SH IY L AH, 4 × IY L AH, 2 × S IY L AH, EH L AH, IY L AH N. None
  decodes as its label. They are probably mislabelled or cut-off "sheila"
  recordings (`results/final_sheila_head_cascade_test_sc_false_accepts.json`),
  unconfirmed until someone listens to them.
- **Other words accepted:** exactly at the 0.2% limit on clean audio, and
  0.27-0.50% under rooms and mics. Stage 1 alone was at 0.00-0.40%.

**Critique.**
- The FA budget was missed on both test runs, each time by less than the
  Speech Commands clips that are probably mislabelled. Listening to them
  settles it (about 30 files). If they are "sheila", both candidates are
  inside the target. If not, the next candidate needs a larger validation
  margin.
- The head was trained on one kind of hard negative (LibriSpeech read
  speech). Conversational speech, TV and music were not mined and are not
  in the negatives.
- The mined windows come from the float stage-1 model and the training
  split only; validation and test speech were never mined.
### 31. Neuron engine: the accumulator pipelined, timing met with the default strategy

Branch `engine-timing`. Entry 19 found the engine's critical path:
`acc[idx] += w` in one clock, i.e. a 128-way 18-bit read multiplexer, the
adder and the write decode, 11 logic levels. The rebuilt bitstream met
100 MHz only with Performance_ExplorePostRoutePhysOpt.

**Change** (`rtl/neuron_engine.v`): two stages after the synapse register.
- Stage A reads `acc[post]`. If stage B writes the same neuron in this
  clock, stage A takes B's sum instead (forwarding), so back-to-back events
  to one neuron stay exact.
- Stage B adds the weight and writes.
- A frame waits for both stages before the recurrence. This costs one
  engine clock per frame.

**Checks:**
- Streaming RTL check (`verify_stream_rtl.py --engine`, "sheila"
  candidate): 40 streams, 355 hops bit-exact. Worst hop 1,029,278 cycles
  (unchanged); mean 988,705 (was 988,684).
- Vivado 2025.2, default strategy (`vivado/build.tcl`):

  | | WNS | Hold (WHS) | LUTs | BRAM36 | DSP |
  |---|---|---|---|---|---|
  | before, default strategy | −0.023 ns | | | | |
  | before, ExplorePostRoutePhysOpt | +0.348 ns | | 12,894 | 88 | |
  | **pipelined, default strategy** | **+0.406 ns** | +0.040 ns | 14,627 (27%) | 88 | 12 |

  All constraints are met, with 0 failing endpoints
  (`results/engine_pipe_timing_summary.rpt`, `_utilization.rpt`).
- The worst path is now in the Poisson encoder (a DSP multiply,
  3 logic levels), not in the engine.

The bitstream is `build/keyword_engine_pipe.bit`. It has not been on a board.

### 32. Near-miss words as head negatives: the first cascade inside both false-accept targets

The keyword head of entry 30 was trained again from verifier seed 0, now on
`data/multi_sheila_full`: its real MSWC near-miss words (she, sheep, shell,
shield, ...) are 15% of each keyword sub-batch, and the synthetic "sheila"
clips are left out (`--near-share .15 --kw-real-only`). 10 epochs, 23 min.

**Validation** (≤ 1.6 FA/h):

| Keyword head | Live recall | Complete | FA/h | "she-" words accepted | MSWC other | Other words, rooms / mics + rooms |
|---|---|---|---|---|---|---|
| seed 0 (entry 30) | 89.2% | 95.6% | 1.55 | 23/183 | 1.16% | 0.33 / 0.50% |
| seed 1 | 87.8% | 94.9% | 1.38 | 20/183 | 0.85% | 0.27 / 0.33% |
| **near-miss negatives** | 87.8% | 94.3% | **1.16** | **1/183** | **0.22%** | **0.13 / 0.27%** |

**Choosing the candidate.** The recall of the near-miss head is 1.5 points
below seed 0, the same as the spread between the two plain seeds. A
tie-break was fixed before its test run: when validation recall differs by
no more than the seed spread, the larger false-accept margin wins. That
makes the near-miss head the candidate (t1 = 3785, t2 = 1524). Its stage-1
threshold is low: 2,200 verifier calls per hour of speech, 117 ms each,
within every 250 ms request.

**Test split, once** (`results/final_sheila_head_near_cascade_test.json`):

| | Stage 1 alone | Head, seed 0 (entry 30) | **Head, near-miss negatives** |
|---|---|---|---|
| Live recall | 63.7% | 91.0% | **86.8%** |
| Complete recordings | 70.7% | 96.6% | **92.0%** |
| Held-out mics / rooms / both | 66.5 / 60.9 / 58.0% | 88.7 / 90.6 / 87.7% | 86.3 / 85.9 / 84.9% |
| Live other words accepted | 0.00% | 0.20% | **0.07%** |
| **FA/h (18.97 h)** | 1.58 | 2.11 | **1.42** |
| MSWC "she-" words / other words | 23/183 / 0.83% | 26/183 / 1.18% | **6/183 / 0.36%** |
| 1 h stream: recall, false accepts | 69.3%, 1 | 91.4%, 2 | 87.9%, 1 |
| Latency after the word (median, p90) | 0.04 s, 0.11 s | 0.18 s, 0.33 s | 0.18 s, 0.32 s |

- **This is the first cascade inside both false-accept targets on test**
  (≤ 2 FA/h, ≤ 0.2% other words), at +23.1 points of live recall (50 clips
  only by the cascade, 1 only by stage 1).
- Under rooms, other words stay at 0.27%, just over the clean-audio 0.2%
  (stage 1 alone: 0.20-0.40%).
- Candidates tested so far, one test run each: the path cascade (entry 29),
  head seed 0 (entry 30) and this one. Each was chosen on validation before
  its test; the test results were not used to choose between them.

## Open items

Status of `IMPLEMENTATION_PLAN.md`: Phases 0-6 done; the streaming firmware and
engine are verified on the board over JTAG (entry 18); Phase 7 evaluated for "yes",
not promoted (entry 17). The keyword is now "sheila" (entry 22). The best system
is the cascade of the stage-1 SNN and the verifier with a keyword head trained
against near-miss words (entry 32): 86.8% live recall on test at 1.42 FA/h and
0.07% other words, inside both false-accept targets; RTL bit-exact with the
pipelined engine (entry 31); not yet on a board.
See `MERGE_SUMMARY.md` at the repository root for all branches.

1. **Board test of the cascade** (`firmware -DCASCADE` with the head,
   `build/keyword_stream_cascade_engine.bin` on `build/keyword_engine_pipe.bit`,
   entries 31-32): `verify_verifier_rtl.py
   results/models/sheila_verifier_head_near.pt --cascade 3785 1524` builds the
   firmware, `jtag/make_stream_vectors.py --cascade
   results/models/sheila_verifier_head_near.pt 3785 1524` the vectors, then
   `jtag/stream_board_test.tcl`. Needs a board and the
   team's go-ahead. The stage-1-only candidate (entry 27) is also not yet
   board-tested.
2. **Suspected Speech Commands label errors** (entries 28-30). About 30
   negative clips labelled zero, two, four, and so on decode as "...IY L AH".
   Listening to them decides whether both cascade test runs are inside the
   ≤ 2 FA/h target (`results/sheila_suspected_label_errors_*.json`).
3. **Other words accepted under rooms**: 0.27% for the candidate (entry 32;
   target 0.2% on clean audio). More varied hard negatives are needed:
   rooms, conversational speech, TV.
4. **v3 streaming over Ethernet.** Pedro's run (entry 18b) covers the v2
   release. `board_server.py` serves v3 and has his fix; `net_board_test.py`
   checks stage 1 against the oracle (the cascade's fields are not in the
   TCP protocol yet). Not yet run on a board.
5. ~~Engine timing margin~~: done (entry 31). The pipelined engine meets 100 MHz
   with the default strategy (WNS +0.406 ns); use `keyword_engine_pipe.bit`
   for the board test.
6. Recordings of the actual user and microphone, for evaluation only.
7. From Pedro's list:
   - "release or trial model": superseded by the streaming SNN (entries 17, 27);
   - time-convolutional first layer: the streaming SNN's learnable delays
     cover this;
   - lower `READ_WAIT`: done (entry 13);
   - LIF unit: the neuron engine (entry 16).
