# Research plan: encoding and network size vs detection quality and latency

**Research goal.** Understand how *input encoding* and *network size* trade off against
*detection quality* and *latency* for a keyword detector running in software on a
RISC-V core.

**Starting question.** Which co-design choices between the network (input encoding and
topology) and the resource-constrained RISC-V platform let a keyword spotter stay
accurate while fitting the platform's memory and real-time limits?

The work is split into nine experiments (E0–E8). Each one is done in its own Claude Code
session and ends with a short, detailed report containing measured data. Together the
reports form the evidence for the conclusion of the main report. How to start a session:
see [README.md](README.md). Progress: [STATUS.md](STATUS.md).

---

## Part A — Primer: the ideas behind the plan

### A1. What the system does
1. The PC records audio at 16 kHz. Every **250 ms** it takes the last **1 second** and
   turns it into a small "sound picture", a *mel spectrogram*: **24 frequency bands ×
   32 time slices** (each slice ≈ 31 ms), each value one byte (0–255). That is 768 bytes.
2. The PC sends those 768 bytes over Ethernet to the PYNQ-Z2 board.
3. On the FPGA, a small **RISC-V processor (PicoRV32)** runs the spiking neural network
   from on-chip memory (BRAM). It returns two scores ("yes" and "not yes"), a spike
   count and the exact number of **clock cycles** the inference took.
4. If the "yes" score wins by more than a threshold, LED0 lights for 1 s. In live mode,
   2 of 3 consecutive windows must agree ("2 of 3 confirmation").

### A2. Spiking neurons in one paragraph
Each hidden neuron has a *membrane voltage*. At every time step it leaks a bit (×7/8),
adds its input current, and when it passes a threshold it emits a **spike** and drops back.
The output layer counts the spikes of the hidden neurons and weighs them into the two
scores. Everything runs in integer arithmetic, so the RISC-V core never needs floating
point, and the board results are **bit-exact** with a Python reference (the "integer oracle").

### A3. Input encoding — how the sound picture becomes neuron activity
This is one of the two main knobs of the research question.

| Encoding | How it works | What it costs | What it can "see" |
|---|---|---|---|
| **Current** (release) | The whole picture is multiplied once by the weights; the result is a *constant* input current for 12 time steps. | One big multiply (768 × neurons), then cheap steps. | The whole second at once, at fixed positions. It knows *where* energy is, not *in which order* sounds happened. |
| **Rate** | Each pixel becomes a spike train: bright pixels spike often, dark ones rarely. The input layer is recomputed at every step. | About 10× more cycles than current (Mateo measured 59.8 ms vs 715 ms). | Same information as current, plus noise-like spike timing. |
| **Temporal** (new, E3) | The picture is fed **column by column**: time slice 1 at step 1, slice 2 at step 2, and so on. The same small weight set is reused at every step. | ~30× fewer weights; work spread over 32 steps. | The **order** of sounds: "y → e → s" is different from "l → e → s". |

Why this matters for your problem: the release network uses current encoding with one
dense layer. It learned roughly "an *e*-vowel followed by a hissing *s* somewhere in the
second". "les", "mes", "pes" and "des" contain exactly that, so they fire. It never had
to learn the short "y" glide at the start. "yep", "yeah", "ye" and "yech" share the start
and vowel, and they fire when the hiss-like part is noisy or when the window cuts the word.

### A4. Network size and topology — the second knob
- **Size** = number of hidden neurons. More neurons can store finer distinctions, but each
  neuron in the release design needs 768 weights × 2 bytes ≈ 1.5 KB of memory and ~768
  multiply-adds per inference.
- **Topology** = how neurons are connected. The release network is *dense*: every neuron
  sees every pixel at a fixed position. A *temporal-convolution* network (E4) uses small
  filters that slide over time, so the same detector works wherever the word sits in the
  window. A second layer then combines "what happened when", which is how you detect a
  sequence such as y-e-s.

### A5. The platform limits — what "resource-constrained" means here
| Limit | Value | Where it comes from |
|---|---|---|
| Model memory | **144 KiB** (0x18000–0x3bfff) | 256 KiB BRAM shared with code, mailbox and stack. The release model uses 98,824 bytes. |
| Real-time budget | **25,000,000 cycles** per window | The 250 ms hop at 100 MHz. Inference must finish before the next window arrives. |
| Compute | RV32IM, ~120 cycles per multiply-add from BRAM | PicoRV32 is small and memory-bound; every load waits on BRAM. |
| Custom instruction `kdot` | ~0.75 cycles per multiply-add | Mateo's coprocessor: int16 weight × uint8 input dot products, 22× faster input layer. |

Largest dense network that fits with int16 weights: bytes ≈ 1544 × neurons + 8, so
**95 neurons** maximum. This "memory wall" is itself a finding (E2, E6).

### A6. How we measure quality
- **Clip metrics** on the official test set: 11,005 one-second clips from speakers never
  used in training. *Recall* = share of real "yes" detected. *Precision* = share of
  detections that were really "yes". *F1* combines both. *FPR* = share of other words
  that fire.
- **Live-stream metrics**: clips placed in background noise, cut into 250 ms-hop windows
  exactly like the live demo. "yes" detected (%) and other words accepted (%), single
  window and 2 of 3.
- **Confusable false-accept rate** (new): share of near-miss words (les, mes, pes, des,
  yep, yeah, ye, yech, …) detected as "yes". Measured on your recordings (E1) and on
  computer voices.

### A7. How we measure cost
- **Latency** = cycle count reported by the RISC-V core on the real board (maximum over
  the 40 verification vectors), for plain RV32IM and with `kdot`. We never estimate
  latency from operation counts. Before trusting a cycle count, the board scores must be
  bit-exact with the integer oracle.
- **Memory** = bytes of the `.model` section, compared with the 144 KiB limit.
- **Operation counts** (multiply-adds, spikes) as explanations, not as measurements.

### A8. What "trade-off" and "Pareto front" mean
Every design choice moves several numbers at once: more neurons raise F1 but also cycles
and bytes. A design is **Pareto-optimal** if no other design is better on one number
without being worse on another. The answer to the research question is:
1. the Pareto front (quality vs latency vs memory);
2. which platform limit (memory or real time) cuts it off for each design family;
3. which co-design choice moves that limit (encoding, topology, precision, `kdot`).

---

## Part B — Method rules (every experiment; this is what makes it scientific)

1. **Data split.** Speech Commands v0.02, official speaker-disjoint split: 84,843
   train / 9,981 validation / 11,005 test clips. Choose checkpoints, thresholds and
   "best configuration" on **validation only**. Score the **test** set only to report.
   Your own recordings: session 1 for training/validation, session 2+ **test only**.
2. **Repetition.** 3 seeds (0, 1, 2) per configuration; report mean ± sample sd. Say
   clearly when differences are smaller than the seed spread.
3. **One factor at a time.** Each report lists what changed and what was held constant.
   Default controls = the release recipe: 35 epochs, last 10 quantisation-aware, AdamW,
   lr 0.002, batch 256, 16,384 samples/epoch, beta 7/8, Q10 int16 weights.
4. **Latency on hardware only.** Use `research/tools/bench_board.py` (built in E0). It
   compiles the model's firmware, runs the 40 verification vectors on the board, checks
   bit-exactness against the oracle, and returns cycles. No bit-exact result, no reported
   cycles.
5. **Store everything.** Raw results go to `research/results/E<n>/*.json` (plus plots as
   PNG). Trained integer models go to `research/results/E<n>/models/<name>.npz` (small).
   Datasets, recordings and PyTorch checkpoints stay in ignored folders (`data/`,
   `runs*/`, `build/`).
6. **Threats to validity.** Every report names them: e.g. only two computer voices, a
   single human speaker, clip metrics are not false alarms per hour, cycles depend on
   BRAM wait states.
7. **Do not break the release.** `deploy/` and the release firmware stay untouched.
   Experimental firmware lives in `build/` and is only loaded on the board for
   measurements. At the end of a session, restore the demo with `run_board.ps1`.
8. **Report** with [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) into
   `research/reports/E<n>_<short-name>.md`. Then update [STATUS.md](STATUS.md) and commit
   locally on branch `Pedro`. Ask before pushing.

---

## Part C — The experiments

Order matters: E0 builds the tools everything else uses, and E1 builds the test data
that E2–E7 are scored on. E2 → E4 go from "size" to "encoding" to "topology". E5 and E6
test the other co-design levers (data, precision). E7 looks at the whole live system.
E8 draws the conclusion. After each experiment, the report's "next step" section may
adjust the parameters of the following one (for example, which sizes E3 uses).

---

### E0 — Setup, baseline reproduction and shared tools

**What.** Make this PC able to train networks, and build the measuring tools every other
experiment uses.

**How.** Install PyTorch, download the dataset, and re-score the existing release model
to check we get Mateo's numbers. Time one training run. Write three helper scripts:
1. build firmware for any model;
2. measure it on the board;
3. score it on all quality metrics.

**Why.** Every later claim depends on the measurement being correct and identical across
experiments. Reproducing a known number first (release test F1 = 0.8761) proves the setup
is sound. The training-time measurement tells us how many runs later sessions can afford.

**Expected outcome.** Release metrics reproduced exactly; cycles 5,981,521 (RV32IM) /
267,393 (`kdot`) reproduced on the board through the new tool; a time per training run.

**Steps for the session.**
1. Environment (explain the downloads to the user first: ~200 MB PyTorch, 2.4 GB dataset):
   - `.venv/Scripts/python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu`,
     then `-r code/snn_keyword/requirements.txt` plus `matplotlib`.
     If no wheel exists for Python 3.14, recreate `.venv` with `py -3.13 -m venv .venv`
     and reinstall `numpy sounddevice pytest matplotlib torch`.
   - `python code/snn_keyword/prepare_data.py` (downloads, checks SHA256, caches
     `data/features.npz`). Then `python code/snn_keyword/augment.py` (Mateo's variants,
     needed by `confusables.py`).
   - `python -m pytest code/snn_keyword/tests -q` must pass. Tests needing Verilator or
     native gcc may be skipped; list them.
2. Baseline: score `deploy/model.npz` on the test set with `model.integer_forward` and
   compare with `results/evaluation.json` / REPORT.md. Must match exactly.
3. Time one training run on the CPU:
   `train_keyword_snn.py --encodings current --seeds 0 --out runs_e0`, with the wall time
   recorded. If it takes more than ~30 min, note it and propose reduced settings (fewer
   seeds or samples) for the sweeps in STATUS.md, with the user's agreement.
4. Build `research/tools/`:
   - `build_firmware.py <model.npz> --out build/<name>`: runs `export_model.py`, then
     compiles `keyword.bin` and `keyword_kdot.bin` with the Vitis gcc and the flags in
     `firmware/Makefile` (+ `-Wl,--no-warn-rwx-segments`). It reports the `.model`
     section size (`riscv64-unknown-elf-size -A`) and refuses models over 144 KiB. It must
     rebuild `deploy/model.npz` into binaries byte-identical to `deploy/keyword*.bin`.
   - Extend `start_board.sh` with an optional firmware path argument:
     `start_board.sh kdot path/to/fw.bin`.
   - `bench_board.py <model.npz> [--variant base|kdot|both]`:
     - build the firmware;
     - scp it to the board; restart the server with it;
     - send the 40 vectors in `results/verification_vectors.npz` (key `x`);
     - compare scores and spikes with `integer_forward` for *this* model;
     - write JSON: bit_exact, cycles mean/max per variant, model bytes, bin size.

     New model kinds (E3/E4) add their own oracle via a `forward_int(x, q)` dispatcher in
     `model.py`, keyed on `q['kind']`.
   - `eval_model.py name=model.npz … --out research/results/E<n>/quality.json`:
     - clip metrics on validation and test (threshold from the model file);
     - live-stream metrics and TTS confusables (reusing `confusables.py`);
     - user-recording metrics once E1 exists.
5. Report `reports/E0_setup.md`: versions, dataset hashes, reproduced numbers, training
   time, tool descriptions and usage.

---

### E1 — Evaluation sets for the near-miss problem

**What.** Record your own voice saying "yes" and the words that fool the detector, and
measure how often the current models are fooled.

**How.** A small program shows one word at a time on screen and records 1.5 s after you
press Enter (or automatically after a countdown). You say:
- "yes" 40 times in different ways (normal, fast, slow, quiet, loud, question-like,
  near and far from the mic);
- each near-miss word 10 times: les, mes, pes, des, tes, ness, guess, less, bless, yep,
  yeah, yea, ye, yech, yet, yell, jess, yesterday;
- some ordinary words (no, go, hello, test, seven…).

Do this **twice, on different days**: session 1 may later be used for training; session
2 is **only ever used for testing**. Teammates can record extra sessions (more speakers =
better conclusions). We also regenerate the computer-voice probe with the new words.
Then we score the release model and Mateo's augmented trial model on everything.

**Why.** "You cannot improve what you do not measure." The Speech Commands test set has
none of these near-miss words, so its F1 of 0.876 hides your problem completely. The
second session is kept apart so that later improvements are tested on recordings the
network has never heard. Otherwise we would be fooling ourselves.

**Expected outcome.** A table per word: % detected as "yes" (single window and 2 of 3)
for both models. We expect high rates for les/mes/pes/des and yeah/yep, and Mateo's trial
model to be better on endings but not on onsets.

**Steps for the session.**
1. Write `research/tools/record_words.py`:
   - `--speaker pedro --session 1`;
   - 16 kHz mono PCM16, 1.5 s per take, randomised word order;
   - shows a level meter and warns if the peak is < 0.02 (the mic is quiet; ask the user
     to speak closer);
   - saves `data/recordings/<speaker>_s<session>/<word>_<nn>.wav` (ignored by git: voice
     recordings stay private unless the user decides otherwise);
   - allows redoing the last take.
2. Guide the user through session 1 (and session 2 if they have time; otherwise note in
   STATUS.md that session 2 is still needed before E5's final evaluation).
3. Extend `make_tts_probe.ps1` with the new near-miss words (keep the old list) and
   regenerate `build/tts_probe/`.
4. Extend `eval_model.py` with a recordings section:
   - per word, % detected over 250 ms-hop windows (clip placed in background noise, as in
     `confusables.py`), single and 2 of 3;
   - group summaries: onset confusables (les/mes/pes/des/tes/ness/guess/less/bless/jess),
     ending confusables (yep/yeah/yea/ye/yech/yet/yell), longer words (yesterday),
     neutral words, and "yes" recall.
5. Score `deploy/model.npz` (release) and `results/models/augmented_current_seed2.npz`
   (trial). Report `reports/E1_eval_sets.md` with the per-word table and group summaries.

---

### E2 — Network size sweep (dense network, current encoding)

**What.** Train the release design with different numbers of hidden neurons and measure
quality, latency and memory for each.

**How.** Sizes **8, 16, 32, 64, 95** neurons (95 is the most that fits the 144 KiB
model memory with int16 weights), 3 seeds each = 15 training runs. Everything else is
identical to the release recipe. For every trained network, record:
- validation/test F1, recall and FPR;
- live-stream and near-miss metrics (E1 sets);
- model bytes;
- board cycles for plain RV32IM and for `kdot`.

**Why.** Size is the most basic co-design knob. We expect:
1. quality rises quickly and then **saturates**;
2. memory and plain-RV32IM cycles grow **linearly** with size, while `kdot` cuts the
   input-layer cost so strongly that the leak/spike steps become the bottleneck;
3. the dense design hits the **memory wall** before it hits the real-time wall.

If that holds, "just make it bigger" is not the path to fixing near-miss words. That
motivates E3/E4 (different encoding and topology) and E6 (smaller weights).

**Expected outcome.** Three plots (F1 vs neurons, cycles vs neurons, F1 vs cycles, with
the 144 KiB and 25 M-cycle limits drawn in) and a table. The smallest size within one sd
of the best F1 is the "efficient" dense point.

**Steps for the session.**
1. `train_keyword_snn.py --encodings current --hidden <H> --seeds 0 1 2 --out runs_e2/h<H>`
   for each size. For 95, first confirm the model fits (`build_firmware.py` size check).
2. `eval_model.py` on all 15 models, then `bench_board.py --variant both` on each (cycles
   do not depend on seed much: measure all three, report mean/max).
3. Copy the integer models to `research/results/E2/models/`. Write plots with matplotlib.
4. Report `reports/E2_size_sweep.md`. Name the efficient size(s) that E3 should use.

---

### E3 — Input encoding comparison at matched size

**What.** Compare the three ways of feeding the sound picture into the network, and the
number of simulation time steps, at the same network size.

**How.**
- **Current vs rate**: at two sizes chosen in E2 (default 16 and 64), with 6, 12 and 24
  time steps. Rate encoding at 24 steps may exceed the real-time budget. That is a
  result, not a failure: record it.
- **Temporal (new)**: the 32 time slices of the spectrogram become the 32 time steps. At
  step *t*, each neuron receives the weighted sum of slices *t−k+1 … t* (24 mels × *k*
  context slices, *k* ∈ {1, 3}). The same weights are reused at every step. The neurons
  leak and integrate over time, so recent sounds count more than old ones. Readout =
  accumulated spikes, like the release model. Sizes: 16, 32, 64, 128. Temporal weights
  are tiny, so larger sizes fit in memory.
- This needs:
  - a new PyTorch model and an integer oracle (`kind = temporal`) in `model.py`;
  - export support;
  - a new firmware path in `inference.c` selected by `MODEL_KIND`. The C code first
    transposes the 768-byte input from mel-major (as sent by the PC) to time-major, so
    that *k* consecutive slices are contiguous and `kdot` can compute each weighted sum.
    Both are word-aligned: 24·*k* elements, a multiple of 4.

**Why.** Encoding decides both **how much work** happens per step and **what information**
the network can use. Current encoding is cheap but blind to order; rate encoding repeats
work at every step; temporal encoding spreads the work over time, needs ~30× fewer
weights, and lets the network see the order of sounds. This is the heart of the research
question: the same quality might be reachable with far less memory, or much better
near-miss rejection at similar latency.

**Expected outcome.**
- A table and a plot of F1 and near-miss false accepts vs cycles vs bytes for every
  encoding × size × steps combination.
- Expectation: rate ≈ current in F1 but ~10× the cycles. Temporal matches or beats current
  on onset near-misses (les/mes/…) at a fraction of the memory.

**Steps for the session.**
1. Implement the temporal model (PyTorch + integer oracle + export + C path).
2. **Bit-exactness first**: run `bench_board.py` on a trained temporal model before any
   sweep. Scores and spikes must match the oracle on all 40 vectors, for base and `kdot`.
3. Add unit tests to `tests/test_pipeline.py` for the new oracle (hand-computed tiny case).
4. Run the sweeps (current/rate/temporal), `eval_model.py`, `bench_board.py`.
5. Report `reports/E3_encoding.md`. State which encoding(s) and sizes go on to E4/E5.

---

### E4 — Time-aware topology: a temporal-convolution SNN

**What.** A two-layer spiking network designed to recognise *sequences* of sounds.

**How.**
- Layer 1 has **filters** that each look at a few consecutive time slices (kernel 3 or 5)
  and slide over the whole second. The same filter detects "a y-glide" or "an s-hiss"
  wherever it happens. Its outputs are spikes over time.
- Layer 2 neurons each look at layer-1 spikes over a short span of time (another small
  kernel). So they can learn "y-glide, then e-vowel, then hiss", and ignore
  "l-onset, then e-vowel, then hiss".
- Because layer-2 inputs are spikes (0/1), layer 2 only adds weights where a spike
  happened. That is the classic SNN efficiency argument, and we can measure it (spike
  counts and cycles).
- Sweep: filters 16 / 32 / 64 × kernel 3 / 5, a layer-2 size chosen from E3's results,
  3 seeds each.
- Firmware: the same transposed input and `kdot` for layer 1, plus an event-driven loop
  for layer 2. No FPGA change.

**Why.** E2 and E3 change *how big* and *how fed*. E4 changes *how connected*, which is
the third co-design lever. It tests whether the right topology fixes near-miss words at
low memory and within the latency budget. Mateo's journal already identified this as
the likely real fix.

**Expected outcome.** The temporal-conv SNN on the Pareto plot next to the best dense
and temporal models. Expectation: clearly lower onset near-miss false accepts than dense
at similar F1, a few ms of latency with `kdot`, and well under the memory limit.

**Steps for the session.** Same pattern as E3:
1. model + oracle + export + C path;
2. unit test;
3. bit-exact board check;
4. sweep, evaluation and bench;
5. `reports/E4_temporal_conv.md`.

---

### E5 — Hard-negative training data (teaching what "not yes" sounds like)

**What.** Train the networks with near-miss words labelled as "not yes".

**How.** Three sources of near-miss examples:
1. **Onset swaps (new).** Take a real "yes" from the dataset (thousands of speakers),
   cut off the "y" glide, and glue on the beginning of another word from the dataset
   that starts with a different consonant:
   - l from "left" or "learn";
   - m from "marvin";
   - d from "dog" or "down";
   - b from "bed";
   - t from "tree";
   - g from "go";
   - k from "cat";
   - n from "no".

   The result sounds like "les", "mes", "des", "bes", "tes", "ges", "kes", "nes", spoken
   by many different people. Each variant keeps the train/validation/test split of its
   source clips, so test speakers never leak into training.
2. **Ending variants.** "yep" (the hiss replaced by a p closure and burst) and "yeah"
   (longer vowel, no hiss), next to Mateo's existing "yets", "yech", "yeh" and cut
   variants (`augment.py`).
3. **Your session-1 recordings** (never session 2, which stays the test set).

We retrain the best dense model (E2) and the best time-aware model (E3/E4) **with and
without** the new data, 3 seeds each, and tune thresholds on validation data that
includes hard negatives.

**Why.** This separates two explanations for the near-miss problem:
- the network **cannot** tell the words apart (a limit of topology or encoding), or
- the network was simply **never shown** the difference (a data limit).

If the dense model improves a lot with data, data was the limit. If only the time-aware
model reaches low false accepts *without losing recall*, the topology matters. This
interaction is itself a co-design result.

**Expected outcome.**
- A 2 × 2 table (dense / time-aware × without / with hard negatives): "yes" recall,
  onset near-miss FA, ending near-miss FA, clip F1, cycles, bytes.
- Tested on session 2 of your voice, the computer voices, and the dataset test split.

**Steps for the session.**
1. Extend `augment.py` with `onset` (per consonant source), `p` and `yeah` kinds. Onset
   splicing:
   - find the vowel start in "yes" (reuse `word_span`/`s_region`, e.g. the first
     low-frication voiced frame after the onset);
   - find the consonant+transition segment in the source word (the first ~60–120 ms up
     to the vowel);
   - cross-fade 5–10 ms.

   Listen to a few examples: save 10 WAVs to `build/aug_examples/` and tell the user where
   they are.
2. Add recordings (session 1) as an extra training pool (`--recordings` option,
   speaker/session list).
3. Train, tune (validation), evaluate (session 2 + TTS + test), bench.
4. Report `reports/E5_hard_negatives.md`.

---

### E6 — Weight precision vs memory (int16 vs int8)

**What.** Store weights in 1 byte instead of 2, and see what that costs in accuracy and
speed and what it gains in memory.

**How.**
- Quantisation-aware training with int8 weights (Q-format chosen per layer from the
  weight range) for:
  - the dense network, including sizes that only fit with int8 (128, 180);
  - the best time-aware network.
- On the RISC-V side, int8 weights run through the plain C loop, since `kdot` reads int16
  weights. Measure the cycle cost.
- Optionally (only if the user wants hardware work), a `kdot8` variant of the coprocessor
  in `rtl/kdot_pcpi.v`. That needs a Vivado 2026.1 rebuild of the bitstream (~20 min) and
  new timing/utilisation reports.

**Why.** Memory is one of the two platform limits. Precision is the lever that moves the
memory wall found in E2. The interesting question is whether the accuracy lost to
coarser weights is smaller than the accuracy gained by a bigger network, and what that
does to latency when `kdot` cannot be used.

**Expected outcome.** F1 vs bytes for int16 and int8 at each size, cycles with and
without `kdot`, and a statement on whether int8 moves the Pareto front.

**Steps for the session.** Quantisation changes in `model.py` (`quantize`, oracle), export
of `int8_t` arrays, C loop variant, bit-exact check, sweep, `reports/E6_precision.md`.

---

### E7 — Decision rule and end-to-end latency on the live system

**What.** Look at the whole live system: how the rule that turns window scores into a
"yes" affects missed words vs false alarms, and where the time goes from your mouth to
the LED.

**How.**
- Build long test streams: minutes of background noise with test-split words, near-miss
  words and "yes" inserted at known times. Play them through the real board at the real
  250 ms hop (`bench_board.py` stream mode).
- Compare rules: single window, 2 of 3, 3 of 4, at several thresholds. Measure:
  - "yes" detected;
  - false alarms **per minute**;
  - detection delay (time from end of word to LED).
- Measure the latency split:
  - PC feature computation (time in Python);
  - network round trip;
  - inference cycles;
  - the window/hop delay itself.
- A live session with the user (and teammates if possible) using the best 2–3 models.

**Why.** "Real-time" is decided end to end, not just by the RISC-V inference time. Mateo
measured inference at 2.7 ms with `kdot`, but a word needs up to 1 s of window plus hops
before it is confirmed. This experiment shows which part of the system really limits the
user experience, and puts the network's latency into perspective.

**Expected outcome.** A ROC-style plot (detection vs false alarms/min) per rule and model,
and a latency breakdown bar chart.

**Steps for the session.** Stream builder + replay tool, measurements,
`reports/E7_live_system.md`.

---

### E8 — Synthesis and conclusion

**What.** Put all results together into the answer to the research question, and deploy
the best design.

**How.**
- Merge all `research/results/E*/` data into one table.
- Plot the Pareto fronts:
  - F1 vs cycles;
  - near-miss FA vs cycles;
  - F1 vs bytes.
- Mark the platform limits on those plots. Write the conclusion:
  - which choices matter most;
  - which limit binds each design family;
  - which co-design point is recommended, and why.
- With the user's approval, promote the chosen model:
  - update `deploy/` and the manifest;
  - bit-exact board check;
  - live demo.

  Update `README.md`, `REPORT.md` and `JOURNAL.md`.

**Why.** The individual experiments each answer part of the question; the synthesis
connects them into one argument that can go into the main report, with figures.

**Expected outcome.** `reports/E8_synthesis.md` with final figures, a conclusion section
ready to adapt for the main report, and the upgraded demo on the board.

---

## Part D — Rough effort

| Exp | Needs the user | Training runs | Main new code |
|---|---|---|---|
| E0 | No | 1 | tools (build, bench, eval) |
| E1 | **Yes, ~15 min per recording session** | 0 | recorder, recordings metrics |
| E2 | No | 15 | plots |
| E3 | No | ~30 | temporal model + firmware path |
| E4 | No | ~18 | conv SNN + firmware path |
| E5 | Listening check (optional) | ~12 | onset-splice augmentation |
| E6 | Decide on optional RTL work | ~12 | int8 quantisation + C loop |
| E7 | **Yes, live test** | 0 | stream builder/replay |
| E8 | Approve deployment | 0 | figures, docs |

Training time per run on this CPU is measured in E0. If sweeps are too slow, reduce
samples per epoch or seeds *consistently across compared configurations* and state it in
the report.
