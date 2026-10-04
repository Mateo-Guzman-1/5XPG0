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

## 2026-09-27

### 8. Standard Ethernet path on a second PYNQ-Z2; requests processed twice

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

## 2026-09-30

### 9. Keyword changed from "yes" to "Sheila"

- **Motivation.** User decision: the demo and the research programme (E0–E8)
  continue with the keyword **"Sheila"**.
- **Data.** "sheila" is one of the 35 Speech Commands v0.02 words: 2,022
  clips, official speaker-disjoint split, 1,606 / 204 / 212 positives in
  train / validation / test. Archive (SHA256 checked), frontend and features
  are the same as for "yes"; only the labels change.
- **Changes.**
  - `prepare_data.py --keyword` (default `sheila`) labels the data and stores
    the keyword in `features.npz` and `manifest.json`; an existing cache of the
    same clips is only relabelled. The keyword travels into `model.npz`,
    `training.json` and the result files. `tune_stream.py`, `confusables.py` and
    `evaluate.py` refuse a model trained for another word. The live-window cache
    name now contains the keyword; otherwise windows labelled for "yes" would
    be reused silently.
  - `confusables.py`: TTS groups per keyword, result keys `<keyword>_detected`
    ("yes" keeps `yes_detected`), augmented section only for "yes".
    `augment.py` refuses other keywords, and `train_keyword_snn.py` refuses
    `--aug-fraction` above 0 for them.
  - `evaluate.py --results --expect`, `robustness.py --model --out`,
    `make_tts_probe.ps1 -Keyword`. `pc_keyword_demo.py` prints the keyword
    named in `deploy/manifest.json`.
  - New `research/tools/build_firmware.py` (this PC has no make or WSL); it
    rebuilds the "yes" release byte for byte. New `check_board.py`: the running
    board against the integer oracle over Ethernet (vectors, cycles, "2 of 3"
    patterns, streamed WAV clips). Before the switch it passed on the "yes"
    release: 120/120 bit-exact, max 267,393 cycles.
  - `deploy/`: new `model.npz`, `keyword.bin`, `keyword_kdot.bin`. The manifest
    names the keyword and the evidence folder `results/sheila/`. Bitstreams,
    firmware C code, RTL, board server and protocol are unchanged.
  - Tests: the deployed model is paired with its own verification vectors
    (manifest `evidence`); a new test checks the manifest hashes and keyword.
    Two board unit tests had failed since entry 8 because their fake `Board`
    lacked the new 32-bit views; fixed.
- **Training.** Release recipe (current encoding, 64 hidden, 12 steps, 35
  epochs, last 10 QAT), seeds 0/1/2, on the CPU (Ryzen 5 7600X, PyTorch
  2.11.0+cpu): 9.8–10.4 s per run; the GPU runs for "yes" took 18 s. Integer
  validation F1 0.8955 / 0.9054 / 0.8900, so seed 1 is deployed: thresholds
  3002 (window) and 1080 (stream, validation-tuned at 0.3% confirmed false
  accepts, validation confirmed recall 68.1%).
- **Outcome** (`results/sheila/`):
  - Test, seed 1: precision 85.64%, recall 81.60%, F1 0.8357, FPR 0.269%
    (TP 173, FP 29, FN 39, TN 10,764); quantized and float checkpoints never
    disagree. Three seeds: F1 0.835 ± 0.020, below "yes" (0.867 ± 0.022) with
    half the positives. Seed 2 had the best test F1 (0.855); only validation
    decides.
  - Live windows (test clips in noise, 250 ms hop): "Sheila" detected 74.1%
    single, 68.9% with 2 of 3; other words accepted 1.07% and 0.40% ("yes":
    74.5 / 67.3% and 1.07 / 0.13%).
  - Background noise: 2 detections in 398 windows (doing_the_dishes,
    dude_miaowing; "yes": 0).
  - Synthesized words, 2 of 3: "Sheila" 68%. Words that start like it are often
    accepted: sheena 71%, shayla 68%, shield 64%, cheetah 60%, sheep 55%, she
    54%, sheet 48%, shelter 45%. Words sharing only the ending are not: keela
    28%, tequila 27%, feeler 19%, healer 14%, dealer 3%, leila 3%, delilah 0%.
    Unrelated words (yes, no, hello, seven, marvin): 0%. The dense network keys
    on the "shee-" onset, the mirror image of the "yes" ending problem.
  - Board over Ethernet (`check_board.py`): 40 vectors ×3 bit-exact for both
    images; max 5,980,921 cycles (RV32IM, 59.81 ms) and 266,793 (`kdot`,
    2.67 ms); "2 of 3" patterns as specified; round trip 3.9 ms with `kdot`.
    12 synthesized clips streamed at the real hop, every window bit-exact:
    4 of 5 "Sheila" confirmed (the fast, high-pitched one had its two positive
    windows 750 ms apart); she, sheet and shield fired one window each but were
    not confirmed; leila, hello, yes and seven had no positive window.

## 2026-10-01

### 10. Sheila near-misses: hard negatives from real recordings

- **Motivation.** User's live microphone test of the entry-9 release: a plain
  "sh" or "shi" was detected as "Sheila". Speech Commands has no other word
  with /ʃ/, so "sh" (+ "ee") alone separated the keyword from all 34 negatives.
- **Change.** `augment.py` gains Sheila variants cut from real "sheila" clips,
  keeping the official split. `sheila_marks` finds the /ʃ/ per clip as ≥ 50 ms
  of frication (energy mostly ≥ 2 kHz) directly followed by ≥ 150 ms of voicing
  (1,537 of 2,022 clips). Negatives: `sh` (the /ʃ/ alone, sometimes stretched),
  `shee` ("-la" cut off), `sheet` ("shee" + closure + burst), `swap` (/ʃ/ + another
  word without its unvoiced onset, ×2), `ending` ("shee" + the start of another
  word), `onset` (another word's unvoiced onset + "eila"), `head` ("eila"),
  `noise` (white to brown, −60 to −10 dBFS); positive: `shift` (whole word at a
  random position). Half of the variants get colored background noise at 5–30 dB
  SNR. Tools: `train_keyword_snn.py` accepts `augment.npz` of the same keyword;
  `confusables.py --split validation` and variant rates at the stream threshold;
  `evaluate.py --deploy` stages a candidate outside `deploy/`; the TTS probe adds
  a pure "shh" (IPA phoneme; the voices read "shh" as letter names), shush, shy,
  show, shoe and shelly.
- **Round 1** (sh, shee, sheet, swap, head, shift; commit 8506d0e,
  `results/sheila_hardneg/round1/`). Test: "sh" windows passing the live
  threshold fell from 48% to 0%, but pure-noise windows then passed 18.7%
  (validation) and the pink-noise recording triggered 5 of 60 windows. Not used.
- **Round 2** (+ ending, onset, noise, background noise). Configurations, 3 seeds
  each: 64 hidden at fraction 0.3 and 0.5, 95 hidden at 0.3, 60 epochs at 0.3.
  Chosen by validation F1 including the variants: **95 hidden, fraction 0.3,
  seed 2** (0.7440; mean over seeds 0.730 against 0.725 with 64 hidden, within
  the seed spread; also the best live validation recall, 63.7%).
- **Outcome** (test data, `results/sheila_hardneg/`; entry-9 model → candidate):
  - Per window at the live threshold: "sh" 55.8% → 1.2%, "shee" 55.2% → 7.6%,
    "sheet" 52.9% → 8.1%, /ʃ/ + other word 25.3% → 6.4%, "shee" + other word
    64.5% → 33.1%, other onset + "eila" 34.0% → 35.2%, noise 0% → 0%.
  - Synthesized words, 2 of 3: sh sounds 42% → 6%, she- words 58% → 31%, other
    sh- words 49% → 27%, "Sheila" 68% → 63%, unrelated words 0%.
  - Live windows, 2 of 3: "Sheila" 68.9% → 60.9%, other words 0.40% → 0.33%.
    Background noise: 2 → 0 of 398 windows.
  - Clips: precision 0.856 → 0.812, recall 0.816 → 0.755, F1 0.836 → 0.782
    (95 hidden, 3 seeds: 0.781 ± 0.003).
  - Board (candidate firmware copied to the board, `deploy/` unchanged): 40
    vectors ×3 bit-exact for both images; max 8,877,865 cycles (88.78 ms, RV32IM)
    and 395,945 (3.96 ms, `kdot`); "2 of 3" patterns as specified; model 146,680
    of 147,456 bytes. Streamed synthesized clips: 3 of 4 "Sheila" confirmed;
    shh (2 voices), she, sheet, sheep, shy, leila, hello and yes gave no positive
    window; show one window, not confirmed; shield and sheena confirmed.
- **Limits.** Words as close as shield, sheena or shayla, and the "shee" + other
  word splices, still pass about a third of the time: a dense layer cannot follow
  the order of sounds (E3/E4). The price is about 8 points of live Sheila recall.
- **Status.** Candidate loaded on the board for the user's live test; `deploy/`
  still holds the entry-9 model.

## 2026-10-04

### 11. Hard-negative model deployed after the user's live test

- **Live test.** The entry-10 candidate ran on the board (`kdot`, live
  microphone, "2 of 3"). Before the test the board was checked against the
  candidate's oracle: 120/120 requests bit-exact, max 395,945 cycles. The user
  found it clearly better than the entry-9 model: it detected a clearly spoken
  "Sheila" almost every time. This was an informal test with one speaker and
  one microphone, not a measurement.
- **Decision (user).** Deploy the candidate "for now". Research experiment E2
  (network size) uses its recipe; E8 may still replace it.
- **Change.** `deploy/model.npz`, `keyword.bin` and `keyword_kdot.bin` are the
  files staged in `build/sheila_hardneg/` (95 hidden, hard negatives at
  fraction 0.3, seed 2). A fresh `build_firmware.py` build matched them byte for
  byte. `deploy/manifest.json` names the evidence folder
  `results/sheila_hardneg/` and the previous release (64 hidden, git 3dbfdb6).
  Bitstreams, firmware C code, RTL, board server and protocol are unchanged.
- **Checks.** `pytest`: 12 passed. Board from `deploy/` (`run_board.ps1`,
  `check_board.py`): 40 vectors ×3 bit-exact for both images, max 8,877,865
  (RV32IM) and 395,945 (`kdot`) cycles, identical to entry 10; "2 of 3"
  patterns as specified; round trip 90.1 ms (RV32IM) and 5.0 ms (`kdot`).

## Open items

1. ~~Reflash the SD card and test the standard Ethernet path~~: done on a
   second board (entry 8). The first board still needs a reflashed SD card.
2. Choose between the release and trial models (recall against confusable
   rejection).
3. A time-convolutional first layer, the likely real fix for /ts/ and
   onset-less words. `kdot` already computes its dot products, and the
   latency budget allows it.
4. Recordings of the actual user and microphone for training and evaluation.
5. Optional speedups: lower `READ_WAIT`, add a LIF unit.
6. Sheila (entry 9): words starting with "shee-" are often accepted. Entry 10
   removes most of it with hard negatives (deployed since entry 11); shield,
   sheena, shayla and "shee" + another word still pass about a third of the
   time. E1 (near-miss recordings) and E5 (hard negatives) study it properly.
   Items 2 and 3 above were written for "yes"; the time-convolutional layer is E4.
