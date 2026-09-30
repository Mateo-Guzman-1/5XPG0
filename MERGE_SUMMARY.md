# Group 2 keyword spotter: everything since `714db5c`, and the merge of all branches

Covers all work after commit `714db5c` ("Complete Group 2 keyword SNN training
and verified FPGA deployment", 2026-09-24) on every branch, up to 2026-09-30:
what was tried, what worked and what did not, what is still open, and what to
keep, change or delete. It also describes the local branch `merge-all`, which
combines all of them. Numbers come from the files named in each row. JOURNAL
means `code/snn_keyword/JOURNAL.md`.

## In short

- **Best system: a two-stage detector for "sheila" on the PicoRV32.** A
  streaming SNN (neuron-engine accelerator) proposes. A small GRU verifier
  with a keyword head confirms, running only when the SNN proposes. Test
  split (thresholds fixed on validation, JOURNAL 30):
  - 91.0% of live "sheila" detected, 96.6% of complete recordings (the SNN
    alone: 63.7% and 70.7%);
  - 87.7-90.6% under held-out microphones and real rooms (SNN alone:
    58.0-66.5%);
  - 0.20% of other words accepted;
  - **2.11 false alarms per hour** on 19 h of unseen speech: just above the
    ≤ 2 target. 16 of the 40 are Speech Commands clips labelled "zero",
    "two", ... that the verifier decodes as "...IY L AH" (probably
    mislabelled; nobody has listened yet);
  - median detection 0.18 s after the word;
  - worst request 117 ms of compute per 250 ms of audio.

  Both stages are bit-exact from the Python oracle to the RTL. The system has
  **not yet run on a board** (the engine and the stage-1 firmware have, over
  JTAG, with the earlier "yes" model). The SNN alone (1.58 FA/h, 63.7%)
  remains the fallback that meets the false-alarm target.
- **Clip accuracy did not predict live behaviour.** The clip classifiers (the
  "yes" release, Damien's "sheila v2") score 98% on isolated clips but fire
  52-366 times per hour on ordinary speech. At an equal false-alarm rate
  (≤ 2/h), Damien's model finds 7% of "sheila", the streaming SNN 64%.
- **Hardware speed is solved.** On the RTL, compared with the start (≈ 60 ms
  per 1 s window, plain RV32IM):
  - `kdot` custom instruction: 22× faster (on the board as well);
  - one-cycle BRAM reads: a further 1.8× (board);
  - event-driven neuron engine: 10.2 ms per 250 ms hop for a 58k-parameter
    recurrent network with delays.
- **Evaluation had to be fixed before the numbers meant anything.** A 1 h
  stream holds too few false alarms to set a ≤ 2/h threshold; recall-only
  robustness sweeps mislead; one training seed varies by ±3.6 points. All three
  are handled now (see "Method").
- **The merge (`merge-all`) combines 7 branches.** It builds, the 15 Python
  tests pass, the RTL unit tests pass, the release firmware rebuilds
  byte-identical, and the streaming RTL check is bit-exact. Two things are for
  the team to decide: the duplicate board scripts and the two pipelines' file
  layout.

## Branches

| Branch | Who | Commits after `714db5c` | What it holds | In `merge-all` |
|---|---|---|---|---|
| `Rework` | Mateo | 8 | JTAG bring-up, `kdot`, "2 of 3", the full streaming-SNN plan (phases 0-7), neuron engine, iteration loop, keyword → "sheila" | base |
| `Mateo_AI` (upstream) | Mateo | 4 | the first 4 commits of `Rework` | contained |
| `explore-yes-boundary` | Mateo (worktree) | +1 on `Rework` | later decision point, "reject yesterday" policy | merged |
| `explore-verifier` | Mateo (worktree, agent) | +35 on `Rework` | verifier on the PicoRV32 (CTC GRU + keyword head, int8, C, RTL), the cascade firmware, JOURNAL 28-30 | merged (twice) |
| `Pedro` (upstream) | Pedro | 3 (on `Mateo_AI`) | Ethernet path on PYNQ Linux, `board_server.py` double-request fix, E0-E8 research plan | merged |
| `Alex-parallel` (upstream) | Alex | 2 (on `Mateo_AI`) | one-click demo scripts, 64-neuron parallel LIF layer (sketch) | merged, sketch relocated |
| `damien-dicking-around` (upstream) | Damien | 4 (from `c2009be`, before `714db5c`) | independent clip pipeline (8×16 features, ZeroMQ), front-end sweeps, "sheila v2" on his board | merged, 3 files renamed |
| `main`, `Group2`, `Khalid`, `Alex's` | — | 0 | course material only | — |

Nothing was pushed. `merge-all` and the explore branches exist only locally.

## What was tried, by theme

Status: ✅ worked and kept · ⚠️ partly / open · ❌ did not work.

### A. Getting it onto the board

| Item | Result | Status | Source |
|---|---|---|---|
| First board would not boot from SD (BootROM error `0x200A`) | JTAG bring-up replaces the boot chain (`jtag/bringup.tcl`); 40/40 bit-exact, LED 1.0 s | ✅ | JOURNAL 1 |
| Live demo through a JTAG relay (`jtag_server.py`) | Works. Three bugs fixed: a Tcl regex limit, an orphaned `xsdb.exe`, cable contention with Hardware Manager. Round trip 30-45 ms. | ✅ | JOURNAL 2 |
| **Standard Ethernet path on PYNQ Linux** (Pedro, second board) | Works. 40/40 bit-exact (both firmware images), LED 999 ms, 7 ms round trip. `start_board.sh`, `run_board.ps1`. | ✅ | JOURNAL 18b |
| **Bug: every request ran twice** (Pedro) | `struct.pack_into` wrote mailbox words byte by byte, so the core saw sequence 0 and re-ran the request; "2 of 3" confirmed itself. Fixed with 32-bit `memoryview` access. Also covers the v3 streaming requests. | ✅ | JOURNAL 18b |
| First board later booted PYNQ Linux | `bringup.tcl` detects a running PS and only reprograms the PL. Once, repeated PL reloads under Linux hung the PS; `rst -system` recovered it. | ⚠️ | JOURNAL 18 |
| Damien's ZeroMQ deployment (`keyword_bridge.py` on the board, 192.168.2.99) | Works for his clip model, on the stock `spike_top.bit` | ✅ (separate pipeline) | `README_clip_pipeline.md` |
| One-click demo (Alex: `run_board_demo.bat/.ps1`, `board_run.sh`) | Same job as Pedro's scripts | ⚠️ duplicate | — |

### B. Hardware acceleration

| Item | Result | Status | Source |
|---|---|---|---|
| `kdot` custom instruction (PCPI dot-product coprocessor) | Release window model 5.98 M → 267k cycles (**22×**); bit-exact, board = RTL; +180 LUT, +2 DSP | ✅ | JOURNAL 3 |
| Bus fix `READ_WAIT` 8 → 1 | Release: RV32IM 59.8 → 24.7 ms, `kdot` 2.67 → 1.45 ms (on the board: 144,642 cycles) | ✅ | JOURNAL 13, 18 |
| Event-driven **neuron engine** (`rtl/neuron_engine.v`) | Streaming SNN: 113 ms (software) → 77 ms (`kdot`) → **10.2 ms per hop**. Bit-exact on 40 streams in RTL and on the board over JTAG. 12.9k LUT, 88 BRAM36. 20× fewer synaptic operations than a dense equivalent. | ✅ | JOURNAL 16, 18 |
| Engine timing margin | Rebuild with the host ABI fix: WNS −0.023 ns with the default strategy, +0.348 ns with Performance_ExplorePostRoutePhysOpt. The `acc[idx] += w` path is at the edge of 10 ns. | ⚠️ pipeline it | JOURNAL 19 |
| 64 parallel LIF neurons, row-wide weight BRAM (Alex) | Sketch; its header says "not compiled, simulated or synthesized" (it lints). Removes `kdot` and the LED timer from `spike_soc.v`, and maps to the engine's address. | ⚠️ idea only | `rtl_sketches/snn_layer.v` |

### C. The "yes" window model (the current release)

| Item | Result | Status | Source |
|---|---|---|---|
| Augmentation against /ts/, /tʃ/ words (ending-only negatives, window position, closures, capacity 88-256) | Confusables down (yeets 75% → 22%), but recall down (F1 0.876 → 0.820). The trial model was not promoted. | ⚠️ | JOURNAL 4 |
| "2 of 3" window confirmation | Other words accepted 1.07% → 0.13% at 74.5% → 67.3% "yes" | ✅ (in release) | JOURNAL 5 |
| 64 time bins | Worse (F1 0.819 against 0.837) | ❌ | JOURNAL 6 |
| Phase 1 data (multi-corpus, augmentation) on the same dense model | Device drops ÷3, stream false alarms 52 → 10/h, but −10 points of clean recall: one dense layer over a 1 s snapshot cannot hold the data | ⚠️ (motivated the SNN) | JOURNAL 9 |

### D. Streaming SNN (IMPLEMENTATION_PLAN phases 0-7, keyword "yes")

Architecture:
- 24-band log-mel frame every 10 ms;
- 128 ALIF neurons (`kdot` layer);
- learnable delays of 0-31 frames;
- 128 recurrent ALIF neurons;
- leaky readout;
- 58k parameters, int8.

| Item | Result | Status | Source |
|---|---|---|---|
| Robustness harness: held-out mics and real rooms, MSWC corpus, 1 h stream | In use for every model since. Found MSWC's official splits share about 10k speakers; fixed by re-splitting. | ✅ | JOURNAL 8 |
| BC-ResNet-8 teacher and distillation | Teacher 96.5%; distillation *hurt* the student (−10 to −15 points) | ❌ | JOURNAL 10, 14 |
| Ablations | Adaptation, delays and recurrence each worth 10-36 points; PCEN and 64 neurons mixed | ✅ (design confirmed) | JOURNAL 14 |
| QAT → int8 | No loss (the integer model even better); native C bit-exact on 1.08 M frames | ✅ | JOURNAL 15 |
| Streaming firmware, ABI v3 (frames in, state on the core) | Bit-exact in RTL and on the board; protocol v2 and v3 in one server | ✅ | JOURNAL 13, 18 |
| Phase 7 against the targets ("yes") | Better on confusables (/ts/ 0-5.6% against 11-53%), read speech (4 against 45 FA/h) and latency; worse clean recall (57.8% against 67.3%) and mic drop. **Not promoted.** | ❌ for "yes" | JOURNAL 17 |

### E. Method: what was wrong with the evaluation, and the fixes

| Problem found | Fix | Source |
|---|---|---|
| A 1 h validation stream holds about 2 false alarms at the operating point (95% interval 0.2-7/h), so it cannot set a ≤ 2/h threshold | `robust_eval.negative_stream`: 17.85 h (validation) / 18.97 h (test) of negatives-only speech; `stream_select.py` uses it | JOURNAL 19 |
| "Recall ≥ release" compared the SNN at 2 FA/h with the release at 52 FA/h | Report at equal false-alarm rates; under one rule the release reaches 38% against the SNN's 60% | JOURNAL 19 |
| Recall-only robustness sweeps mislead: a 4.5 kHz low-pass "raised" recall to 70%, but under the full rule no threshold was feasible | Decisions only from the full rule (both false-alarm limits) | JOURNAL 25 |
| One training run varies by ±3.6 points (3 seeds) | Differences under about 7 points are not called; compare several seeds | JOURNAL 26 |
| Label conflict: TTS pseudo-words "yes" + letter (yesd, yest…) were negatives, although they contain a whole "yes" | Words that *begin* with the keyword are "don't care" (agreed); `--exclude-words` | JOURNAL 20-21 |
| 22% of Speech Commands "sheila" clips are cut off by the 1 s recording | Report recall on complete recordings too | JOURNAL 23 |

### F. Iteration loop: ideas tested after phase 7

Validation data, full rule.

| Idea | Result | Status | Source |
|---|---|---|---|
| Decide on the sum of the last W frame scores | "yes": 55.9% → 61.0% (W=20); firmware and RTL bit-exact. For "sheila" W=1 is enough. | ✅ | JOURNAL 19 |
| Cascade with the release window model | +3 points, from one noisy grid point; two models on board | ⚠️ not trusted | JOURNAL 19 |
| Wider training microphones (bracket the held-out bands) | No reduction of the "yes" mic drop: the drop is the /s/ band, which a low-pass removes | ❌ | JOURNAL 20 |
| Hard-negative mining (speech, then words) | Speech: moves false alarms from speech to words. Words: −17 points while the pseudo-word conflict existed, −11 after. | ❌ | JOURNAL 20-22 |
| Front ends `logmel_w` (−120…−20 dB), `logmel_agc` (level-normalised), PCEN | Each fixes one level range and loses at normal level; plain log-mel stays | ❌ | JOURNAL 22, 25 |
| Later decision point to reject "yesterday" (`--pos-delay`) | Run stopped when the keyword changed; not evaluated | ⚠️ | branch `explore-yes-boundary` |
| Phoneme verifier on the PicoRV32 (stage 2) | 49k-parameter CTC GRU, 107 ms per check with `kdot`, 121 KB with stage 1, bit-exact C/RTL. For "sheila" see G. | ✅ | JOURNAL 28 |

### G. Keyword "sheila"

| Item | Result | Status | Source |
|---|---|---|---|
| `KWS_KEYWORD` setting across data, training, evaluation, vectors, TTS | "yes" reproduces exactly; models record their keyword | ✅ | JOURNAL 22 |
| Data | Speech Commands v2: 2,022 "sheila" (1,606 train). MSWC: only 16/2/2, so it is used for near-miss negatives only. TTS: 55k utterances generated. | ⚠️ positives are scarce | JOURNAL 22, 24 |
| Baseline (stage 2 on the shared stage 1) | 63.7% / 61.6 ± 3.6% over 3 seeds | ✅ | JOURNAL 23, 26 |
| Ignoring cut-off positives; AGC; TTS + MSWC data; low-pass | −12.7 points; within noise; within noise; infeasible | ❌ | JOURNAL 24-26 |
| QAT seed 2 → int8, W=1 | Validation 66.7% (75.9% complete) at 1.44 FA/h; RTL bit-exact, 10.3 ms per hop | ✅ | JOURNAL 27 |
| **Test split, once** | Clean 63.7% (70.7%) · mics 66.5% · rooms 60.9% · both 58.0% · 0.00% other words · **1.58 FA/h** (19 h) · latency 0.04 s | ✅ | JOURNAL 27, `results/final_sheila_test.json` |
| Damien's "sheila v2" under the same test rule | 7.1% at ≤ 2 FA/h; at his own margin 80.2% but **366 FA/h** and 3.5% other words | — | JOURNAL 27 |
| Verifier retargeted to SH IY L AH (`logmel`, same frames as stage 1) | Clip AUC 0.993 (the "yes" verifier: 0.976) | ✅ | JOURNAL 28 |
| Cascade, phoneme-path score (stage 1 at a lower threshold, the verifier confirms) | Validation +16 points; **test, once:** 84.0% (90.2% complete) at **2.37 FA/h** | ⚠️ over the FA target | JOURNAL 28-29, `results/final_sheila_cascade_test.json` |
| Capped-margin path score; policy (b) after the keyword; hinge fine-tuning on mined proposals | −11 points; no gain; no gain | ❌ | JOURNAL 29-30 |
| Near-miss words in verifier training | "she-" words 26/183 → 4/183 accepted, but −3.4 points of recall | ⚠️ trade-off | JOURNAL 29 |
| **Keyword head** on the verifier, trained on stage-1 false proposals mined from training speech | Validation 89.2% / 87.8% (2 seeds) at ≤ 1.6 FA/h. **Test, once:** 91.0% (96.6% complete), mics/rooms 87.7-90.6%, 0.20% other words, **2.11 FA/h**. Firmware `-DCASCADE`, RTL bit-exact, 117 ms worst request. | ✅ (FA/h see In short) | JOURNAL 30, `results/final_sheila_head_cascade_test.json` |
| Suspected Speech Commands label errors | About 30 negative clips labelled with digits decode as "...IY L AH"; both cascade test runs are inside the FA target without them | ⚠️ listen to them | `results/sheila_suspected_label_errors_*.json` |

### H. Damien's front-end findings

From `research/SHEILA_V2_REPORT.md` and `FRONTEND_SWEEP_REPORT.md`, clip accuracy on Speech Commands:
- Feature precision: 3-8 bits is a plateau; 2 bits hurts. 4 bits would only
  pay off with packed transport.
- Band count: accuracy is flat from 8 to 32 bands for "sheila" (8 is best,
  6.4 KB of parameters). For "yes", 20 bands over 150-7600 Hz.
- Frequency range: "sheila" information lies below about 5.1 kHz, while
  "yes" needs the high band (/s/). Our microphone results agree: "sheila"
  loses nothing on held-out microphones, "yes" loses 17-20 points.
- He fixed a train/live mismatch in his pooling. Our `features.py` uses one
  function for both paths, and the demo test checks live framing against
  offline features.

## Still to do (in priority order)

1. **Listen to the suspected label errors** (about 30 short clips,
   `results/sheila_suspected_label_errors_*.json`). This decides whether the
   cascade meets the ≤ 2 FA/h target on test.
2. **Board test of the cascade** with `build/keyword_engine_abi3.bit` and the
   `-DCASCADE` firmware (`verify_verifier_rtl.py --cascade 7080 1588` builds
   it; `jtag/make_stream_vectors.py --cascade results/models/sheila_verifier_head.pt
   7080 1588` and `jtag/stream_board_test.tcl` check it). The stage-1 firmware
   alone is the fallback. The engine bitstream with the ABI fix has not been
   loaded on any board.
3. **Decide the release.** Replace `deploy/` ("yes" window model) with the
   "sheila" system after item 2, and write "sheila" acceptance targets (the
   IMPLEMENTATION_PLAN targets were written for "yes").
4. **Other words under rooms:** 0.27-0.50% accepted by the cascade (target
   0.2% on clean audio). Mine hard negatives in reverberant and
   conversational speech, not only read speech.
5. **One deployment script** for v2 and v3, the engine bitstream and both
   boards (merge Alex's and Pedro's). The Ethernet protocol does not carry
   the cascade's fields yet.
6. **Engine timing margin:** pipeline the accumulator read-modify-write.
7. Recordings of the actual user and microphone (evaluation).

## Keep / change / delete

**Keep**
- The verifier and cascade: `verifier_*.py`, `train_verifier.py` (with
  `--head-weight`, `--hard`), `mine_verifier_negatives.py`,
  `firmware/verifier.c`, the `-DCASCADE` build, `sim/soc_cascade.cpp`,
  `verify_verifier_rtl.py`.
- The streaming-SNN pipeline:
  - `snn_stream.py`, `train_stream.py`, `stream_select.py`;
  - `robust_eval.py` with the negatives stream, `final_eval.py`,
    `diagnose_stream.py`, `band_sensitivity.py`;
  - `keyword_config.py`;
  - `firmware/stream_*`;
  - `rtl/neuron_engine.v`, `rtl/kdot_pcpi.v`, `READ_WAIT = 1`.
- The release in `deploy/` until item 2.
- `board_server.py` with Pedro's fix, protocol v2/v3, the JTAG scripts.
- The evaluation rules in "Method" (long negatives, full rule, seeds, complete
  recordings).
- Damien's reports and sweep data: the front-end knowledge carries over.
- His clip pipeline as the documented baseline it now is (not the product
  path).

**Change**
- `code/snn_keyword/README.md` and `IMPLEMENTATION_PLAN.md` still describe
  "yes" as the keyword; update them with the release decision.
- The two pipelines share one folder with similar names
  (`features.py`/`audio_features.py`,
  `train_keyword_snn.py`/`train_keyword_snn_clip.py`). Once Damien agrees,
  move the clip pipeline into its own subfolder (his scripts use relative
  paths, so this needs him).
- Deployment scripts: one script (item 3).
- `rtl_sketches/snn_layer.v`: finish and verify it, or retire it. As it
  stands it cannot be combined with the SoC (address clash, removes `kdot`).

**Delete or archive** (after the report is written; the journal keeps the results)
- Flag-gated code of experiments that did not work:
  - hard-negative mining (`HardNegatives`, `HardWords`);
  - `--clipped-dontcare`;
  - front ends `logmel_w`, `logmel_agc`, `logmel_lp*`;
  - `teacher.py` and distillation, if the report does not need to reproduce
    them.
- `mic_diagnose.py` (superseded by `band_sensitivity.py`) and `cascade_eval.py`
  (superseded by the verifier cascade).
- Verifier options that did not work: `score --cap` (capped margin), the hinge
  losses of `train_verifier.py` (`--hard-weight`, `--pos-weight`; the head
  keeps `--hard`), policy (b) reporting.
- `research/pedro/` (the E0-E8 plan was not started, and most of its questions
  are answered above), unless Pedro wants to continue it.
- Alex's demo scripts, once item 3 merges their features.

## The merge (`merge-all`, local only)

Built in a separate worktree from `Rework` (`55807a2`), one merge commit per
branch:

| Step | Commit | Conflicts and resolution |
|---|---|---|
| `explore-yes-boundary` | `ae7b326` | 3 files. Both features kept (`--pos-delay` and `--clipped-dontcare`); "yes"-only code generalised to the keyword setting. |
| `explore-verifier` | `0eab9a1` | none (adds files; `stream_main.c` command 6) |
| `Pedro` | `c874a77` | Demo: his one-line output extended to v3. Journal: his entry 8 kept in full as 18b, one merged open-items list. His research plan → `research/pedro/`. His root `CLAUDE.md` (a personal brief: his machine, "work only on branch Pedro") → `research/pedro/SESSION_BRIEF.md`, so it no longer instructs every session in the repo. |
| `Alex-parallel` | `fc5b78c` | `spike_soc.v`: ours (engine, `kdot`, LED timer). `snn_layer.v` → `rtl_sketches/`, because the Vivado build adds every `rtl/*.v`. His rebuilt `spike_top.bit` reverted (built from his SoC). `code/ann_vs_snn_mnist` deleted (he and Damien both deleted it). |
| `damien-dicking-around` | `6d2e54c` | Separate lineage. Our `train_keyword_snn.py`, `pc_keyword_demo.py` and `README.md` keep their names; his become `train_keyword_snn_clip.py`, `pc_keyword_demo_zmq.py` and `README_clip_pipeline.md` (importer, install script and README updated). Root README describes both. |
| Follow-up | `df165ad` | Tests fixed for Pedro's views (his branch did not update them; they failed there too). The two candidate models added to `results/models/`. This summary. |
| `explore-verifier` again | (this commit) | Journal merged automatically (entries 28-30), open items rewritten. The verifier with keyword head added as `results/models/sheila_verifier_head.pt`. |

**Verified on `merge-all`:**
- all Python byte-compiles;
- `pytest tests/test_pipeline.py`: 15/15 (after the fix above);
- `make -C firmware all stream` with `-Werror`;
- `keyword.bin` and `keyword_kdot.bin` byte-identical to `deploy/`;
- `sim/unit.sh` (AXI, LED): pass;
- streaming RTL check of `sheila_stream_int8.npz` with the neuron engine:
  **40 streams, 355 hops bit-exact**, worst 1,029,278 cycles per hop, identical
  to `Rework`, so the verifier and cascade additions change nothing for the
  stage-1 firmware (`results/verification_stream_rtl_engine_rw1_merge.json`,
  re-run after the second verifier merge);
- the cascade firmware with the keyword head (on `explore-verifier`, same
  files): 934 requests bit-exact on the SoC RTL
  (`results/verification_cascade_rtl_engine_sheila_head_cascade.json`).

**Not in git** (regenerate or copy):
- `data/`: the corpora; `fetch_corpora.py`, `prepare_multicorpus.py`,
  `make_tts_negatives.py` with `KWS_KEYWORD=sheila`.
- `runs_*`: checkpoints. The two deployable integer models are in
  `results/models/`.
- `build/`: bitstreams and firmware. `keyword_engine_abi3.bit` and the Vivado
  reports are in `results/engine_abi3_*.rpt`.
- Verifier checkpoints other than the candidate (`runs_verifier/`, on
  Mateo's PC), the verifier caches and the mined windows (`data_verifier/`).

**Known leftovers:**
- Some of Pedro's upstream commits carry a Claude co-author line. Rewriting
  them would split `merge-all` from his branch's history, so they stay.
- Two sets of board scripts, and two pipelines in one folder (see
  Keep / change / delete).
