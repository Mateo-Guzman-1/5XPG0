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

## Open items

1. Reflash the SD card and test the standard Ethernet path (`board_server.py`).
2. Choose between the release and trial models (recall against confusable
   rejection).
3. A time-convolutional first layer, the likely real fix for /ts/ and
   onset-less words. `kdot` already computes its dot products, and the
   latency budget allows it.
4. Recordings of the actual user and microphone for training and evaluation.
5. Optional speedups: lower `READ_WAIT`, add a LIF unit.
