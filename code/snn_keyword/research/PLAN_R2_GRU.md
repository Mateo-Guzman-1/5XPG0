# Plan R2: our pruned first stage + Mateo's GRU verifier (cascade)

Written 2026-10-19, before any R2 result, for Pedro and for a fresh Claude Code session. Follows R1 ([PLAN_R1.md](PLAN_R1.md), [reports/R1_requirements.md](reports/R1_requirements.md)).

## 0. How to use this plan (new session: read this first)
**Prompt to start:**

> "Read 5XPG0/CLAUDE.md, then 5XPG0/code/snn_keyword/research/PLAN_R2_GRU.md, and carry out R2 from Phase A. Stop at every gate and report to me."

**Read, in this order:**
1. this file;
2. `research/STATUS.md` (sections R1 and S0–S6);
3. `research/reports/R1_requirements.md`.

Read other files only when a step needs them.

**Rules** (from `5XPG0/CLAUDE.md`):
- Work only on branch `Pedro`.
- Read other branches only with `git show <branch>:<path>`. Never check them out.
- Commit locally at the end of each phase. Never push without asking.
- Leave `deploy/` untouched. No board session is needed in R2.

**Pedro:**
- He is a beginner: use plain language and say what a long command does before running it.
- Report real numbers only.

**Gates:** stop at gates **G1** and **G2** and wait for Pedro. Also stop before installing system software (WSL, drivers).

## 1. Where we are (context for the new session)
**System:**
- The PC records audio and computes a 24×32 log-mel window (768 uint8 values, 1 s) every 250 ms.
- It sends this over TCP to the PYNQ-Z2. The ARM there only relays it.
- In the FPGA fabric, a PicoRV32 RV32IM core at 100 MHz runs the network from 256 KiB of BRAM. The model region is 144 KiB.
- **Fabrics:** plain RV32IM; `kdot` (dense dot-product coprocessor); `ksdot` (Pedro's sparse extension: `kxload` and `ksdot`, experimental bitstream `results/S6/keyword_ksdot.bit`).
- Diagrams: R1 report figures 1–3.

**Network:**
- 768 inputs → H LIF neurons → non-spiking linear readout of spike counts → margin. Only one layer spikes.
- Neurons: 12 steps, current encoding (constant drive), β = 7/8, threshold 1024, soft reset, Q10 integer arithmetic.
- The 12-step loop can be replaced by an exact table of 12 thresholds ("LIF table").
- Detection: margin ≥ threshold in 2 of 3 consecutive windows.

**Training** (`code/snn_keyword/train_keyword_snn.py`):
- Data: Speech Commands v2 train split (84,843 clips) plus 30 % synthetic near misses (`augment.py`).
- Regularisation: AdamW (weight decay 1e-3) and a 1e-4 spike penalty. There is no explicit L1/L2 term.
- Schedule: 35 epochs; QAT in epochs 26–35; pruning to K per neuron (cubic schedule) in epochs 4–20.
- Retraining is bit-identical.

**S0–S6 (2026-10-17):**
- Energy method: Vivado power × board cycles (`research/ENERGY.md`).
- Size, bits per weight, pruning, LIF table and sparse `ksdot` were measured. All cycle counts are bit-exact board measurements.

**R1 (2026-10-19):** our designs were measured against the group requirements with Mateo's harness, vendored in `research/tools/mateo_eval/` (from `origin/ALIF-Layer1` a856a03). Thresholds were set on validation (≤ 0.2 % other words, ≤ 2 FA/h); each test was run once.

| Requirement | Target | Our pruned choice H48 K64, `ksdot` + table (R1) |
|---|---|---|
| Live "Sheila" detected | ≥ 85 % | 36.3 ± 6.1 % ✗ |
| False accepts per hour | ≤ 2 | 2.86 ✗ |
| Other words accepted | ≤ 0.2 % | 0.03 % ✓ |
| Word end to decision (median) | ≤ 500 ms | 134 ms ✓ |
| Worst compute per 250 ms request | ≤ 200 ms | 0.35 ms ✓ |
| Model memory | ≤ 144 KiB | 12.4 KiB ✓ |
| Average PL power while listening | ≤ 1 W | 0.367 W ✓ (96.5 µJ per inference) |

- **The limit is false accepts on running speech:** 86 % of them come from LibriSpeech. A one-window network reaches at most 77 % at any threshold.
- **As a proposer** (H48 K64 seed 1, test), the pruned network gives:

  | Live recall | Other words | Proposals per hour of speech |
  |---|---|---|
  | 85 % | 7.4 % | 252 |
  | 90 % | 11.8 % | 391 |
  | 95 % | 32.6 % | 1,012 |

**Mateo's designs** (same test sets; `origin/ALIF-Layer1` `results/final_sheila_head_near_cascade_test.json`):
- **Stage 1 alone (streaming ALIF SNN):** 63.7 % live, 1.58 FA/h, 0.00 %.
- **Cascade (stage 1 + GRU verifier):**
  - live 86.8 % (92.0 % on complete recordings), 0.07 % other words, 1.42 FA/h on 18.97 h;
  - median latency 175 ms; worst request 117 ms; verifier call 107.3 ms (RTL, plain `kdot`);
  - 2,221.5 verifier calls per hour;
  - thresholds t1 = 3785 (stage 1) and t2 = 1524 (verifier head), chosen on validation with `--max-fa-hour 1.6`.
- **Verifier** `results/models/sheila_verifier_head_near.pt` (in git):
  - Input: 24-band log-mel per 10 ms frame (`features.frame_features(audio,'logmel')`, same FFT and mel bank as our window features), the last 150 frames = 1.5 s, stacked by 2.
  - Layers: GRU 48→64 → GRU 64→64 → 40 CTC classes plus a keyword head. 49,513 parameters; int8 export of 53,320 B.
  - Scoring uses the head.
- The verifier runs on CPU unchanged (about 7.7k windows/s).
- No cascade has run on a board yet.

## 2. Question and hypotheses (written before any result)
**Question.**
- Does the cascade "our pruned window SNN → Mateo's unchanged GRU verifier" meet the requirements?
- At what cost, compared with our stage 1 alone and with Mateo's cascade?
- Does fine-tuning the verifier on our stage 1's false proposals improve it?

**Hypotheses:**
- **R2-H1 (unchanged verifier):** live recall rises far above our stage 1 alone (36 %), to 70–85 %, at ≤ 2 FA/h and ≤ 0.2 % on test. It may fall short of 85 %: our proposer passes more "shee-" near misses than Mateo's stage 1, and the verifier never saw ours as hard negatives.
- **R2-H2 (fine-tuned verifier):** fine-tuning on our false proposals raises live recall at the same limits by ≥ 3 points (mean over seeds, paired), or cuts verifier calls per hour by ≥ 20 %.
- **R2-H3 (cost):** every cost requirement still holds:
  - compute ≈ 107.3 + 0.35 ms ≤ 200 ms;
  - memory ≈ 12.4 + 52.1 KiB ≤ 144 KiB;
  - average PL power ≪ 1 W.
  The verifier calls dominate the dynamic energy, but the per-hour energy stays below Mateo's cascade because our stage 1 costs about 0.1 mJ instead of 7.4 mJ per request.
- **R2-H4:** the median from word end to decision is 0.15–0.35 s, within 500 ms.

## 3. The design under test
**Every 250 ms request k:**
1. Our window SNN (H48 K64, seeds 0–2) scores the last 1 s window.
2. If the stage-1 score is ≥ t1 in request k or k−1, the verifier runs on the last 150 frames (24 bands per 10 ms).
3. Detect if the verifier head score is ≥ t2, with a 1 s hold-off.

This is Mateo's `Cascade` logic, unchanged.

**Stage-1 score variants**, chosen on validation:
- (m) single-window margin (main, because the verifier confirms);
- (c) 2-of-3 confirmed score.

**Alignment:**
- Window j ends at 1 + 0.25 j s, i.e. frame 98 + 25 j (the first frame ending after the window).
- That frame lies in request j + 3.
- The stage-1 per-frame trace is NEG = int64.min//2 everywhere except that frame.

**Cost assumptions, stated in the report:**
- The verifier runs on our `ksdot` fabric with plain `kdot` dot products.
- T_V = 107.3 ms worst case; mean per call 105.83 ms (Alex, FINAL REPORT Parallel section).
- E_V = P_PL,dyn(ksdot) 0.282 W × T_V ≈ 30 mJ per call.
- Stage-1 cost from `results/R1/designs.csv` (`h48_k64/ksdot_lut`).

## 4. Phase A: unchanged verifier (software only, ~1 session)
### A0. Setup
- `.venv/Scripts/python.exe -m pip install psutil` (needed by Mateo's `memguard.py`). `soundfile` and `scipy` are already installed.
- **Vendor from `origin/ALIF-Layer1`** (a856a03) into `research/tools/mateo_eval/`:
  - files: `verifier_cascade.py`, `verifier_model.py`, `verifier_data.py`, `verifier_export.py`, `memguard.py`, plus his `features.py` and `model.py` (they hold `frame_features`, `integer_forward_stream`, `decision_scores`);
  - set `ROOT = parents[3]` as in R1;
  - update the folder README with each change.
- **Copy into `research/results/R2/mateo/`** (small, tracked): `results/models/sheila_verifier_head_near.pt` and `results/models/sheila_stream_int8.npz`.
- **Data:** `KWS_KEYWORD=sheila python fetch_corpora.py mswc --split test --needed-only`, and the same with `--split dev`. `cache` always builds the MSWC set. LibriSpeech dev/test are already in `data/librispeech/`.
- **Patch** `verifier_cascade.prefixed_extras` to return empty sets when `data/multi_sheila/manifest.csv` is missing. For "sheila" all prefixed sets are "0/0". Record the patch in the README.

### A0 gates
- **G-A1:** Mateo's `features()` and `integer_forward` equal Pedro's, bit-exact, on the R1 caches (`build/r1_cache/live_test.npz`) for the 3 H48 K64 seeds.
- **G-A2 (reproduce Mateo):**
  1. `verifier_cascade.py cache --stage1 research/results/R2/mateo/sheila_stream_int8.npz` for validation and for test (about 30 min each).
  2. `score research/results/R2/mateo/sheila_verifier_head_near.pt --score head` for both splits.
  3. `final --t1 3785 --t2 1524 --stage1-threshold 18462 --verify-ms 107.3 --policy a`.

  This must give live 0.8679, complete 0.9195, other words 0.0007, 1.424 FA/h over 18.97 h, stream 0.8786 and median latency 0.175 s. MSWC numbers may differ slightly if the MSWC selection differs; note it. If live or FA/h differ, stop and investigate.

### A1. Stage-1 adapter: `research/tools/r2_cascade.py` (new)
- Rebuild the cascade cache for our models with exactly the audio and item order of `verifier_cascade.cache`. Reuse its code path: replace only its stage-1 scorer.
- Assert that the frames equal the G-A2 cache, so the G-A2 verifier scores (vscores) can be reused unchanged.
- From the audio, compute our window margins with the R1 functions (`req_eval.windows` / `fast_margin`, checked against `integer_forward`). Write `{g}_raw` (int64) per seed × variant.
- Output: `data_verifier/cache_<split>_sheila_r2_<seed>_<variant>.npz` (git-ignored).

### A2–A4. Select, test, cost
- **A2. Selection on validation:** `select --verify-ms 107.3 --max-fa-hour 1.6` (Mateo's final setting) per seed × variant. Also give 2.0 as a sensitivity check. Choose the variant by mean validation live recall. Record t1 and t2 per seed.
- **A3. Test once:** `final` with the chosen thresholds → `results/R2/A/h48_k64_seed<s>.json`. It contains the stage-1-alone blocks (`stage1_frame`, `stage1_hop`) under the same rule, and the cascade block.
- **A4. Cost per seed:** `research/tools/r2_synthesis.py` (new; pattern of `r1_synthesis.py`).
  - **Worst compute per request:** t_max(stage 1) + 107.3 ms.
  - **Energy per hour:**
    - Stage 1: 14,400 × E_inf.
    - Verifier: calls/h × E_V.
    - Static.
    - Scenarios as in `research/tools/energy_model.py`: busy-polling unchanged (0.367 W); clock-gated = floor + (stage-1 + verifier core energy)/3600 s.
  - **Memory:** 12.4 KiB + 53,320 B.
  - **Word end to decision:** stream median latency (already includes `verify_ms`) + 1.9 ms PC and link.

**Gate G1: stop and report to Pedro.** Show a table: stage 1 alone vs cascade (unchanged verifier) vs Mateo's cascade, with ✓/✗ for every requirement and the cost. Pedro decides on Phase B, which brings about 25 GB of downloads. Suggested rule: continue if the cascade's test live recall is ≥ 60 %, or ≥ 15 points above stage 1 alone. If it already meets every requirement, Phase B is still useful (fewer calls, robustness) but optional.

### A5. Call-limited rule (added 2026-10-19 after G1, Pedro's choice; written before any A5 result)
**Why:** at G1, Mateo's rule (highest live recall) set t1 below zero. Our stage 1 then passed 57–85 % of all requests, giving 10,244 ± 2,065 verifier calls per hour against Mateo's 2,222.

**Rule:**
- Mateo's `operating` rule unchanged (live other words ≤ 0.2 %, ≤ 1.6 FA/h on the validation negatives, t2 on `select`'s grid, best by live recall then stream recall), plus **≤ 2,150 verifier calls per hour on the validation negatives**. 2,150 is Mateo's validation value (2,149.6).
- Policy a, verify-ms 107.3.
- The variant (m or c) is chosen by mean validation live recall (tie: fewer calls).
- The test is run once per seed → `results/R2/A/budget2150/`.

**Hypothesis R2-H5:**
- Test live recall falls to 83–86 % (validation hint: 84.3–86.3 % from `A/call_budget_validation.json`), so 85 % may be missed.
- Test calls stay near 2,150 per hour.
- FA/h stays ≤ 2.
- The dynamic energy per hour falls about 5× (from 307 J/h to about 65 J/h).

**Then:** Pedro decides on Phase B. Under this rule, Phase B could raise recall at the same number of calls.

## 5. Phase B: fine-tune the verifier on our false proposals
**Data choice (Pedro, 2026-10-19):** Mateo's recipe, with the parts his fine-tune never reads skipped.

**Decision after G1 and A5 (Pedro, 2026-10-19; before any Phase B result):**
- Phase B is judged **mainly under the A5 call-limited rule** (≤ 2,150 verifier calls per hour on validation), against cascade A5 (test live 84.4 ± 0.9 %).
- The highest-recall rule (A2) is reported as a second comparison, against cascade A.
- The question is whether the fine-tuned verifier lifts live recall above 85 % at the same number of calls (R2-H2, first branch).
- After R2, the next step is clock gating (section 7b): fewer calls save energy only when the chip can idle.

### B0. GPU (time-box 1.5 h)
1. Check AMD's current official support list for PyTorch on Windows (ROCm) for the RX 7800 XT, with a web search of AMD's documentation.
2. If it is supported: create a **separate** venv `.venv-rocm` (never change `.venv`) and install as AMD documents.
3. Smoke test on the GPU: `torch.cuda.is_available()`, the device name, a matmul, `torch.fft.rfft`, `F.ctc_loss`, and a GRU forward and backward pass.
4. Fallbacks, in order:
   - (a) WSL2 + ROCm (ask Pedro first);
   - (b) CPU: measure `train_verifier.py --epochs 1 --steps-per-epoch 20`;
   - (c) ask Pedro to have Mateo run the fine-tune on his GPU.
5. Mateo's reference: RTX 4060 Laptop, 8 GB; about 0.25 s per step; 10 × 500 steps in 20–23 min.

### B1. Data
Start the downloads first; they run unattended and in parallel. Check that about 30 GB of disk is free.
- **LibriSpeech train-clean-100** (about 6.3 GB): `fetch_corpora.py librispeech --subset train-clean-100`. dev-clean is already present.
- **MUSAN** (about 11 GB): `fetch_corpora.py musan`.
- **RIRs** (about 1.2 GB): `fetch_corpora.py rirs`. `Augmenter` requires them.
- **MSWC train selection:** `KWS_KEYWORD=sheila fetch_corpora.py mswc --split train --needed-only`. The dev selection comes from A0.
- **CMU dictionary:** `cmudict.dict` from github.com/cmusphinx/cmudict into `data_verifier/cmudict.dict`.
- **Piper TTS:** `pip install piper-tts` (the newer API: `PiperVoice`, `SynthesisConfig`, `EspeakPhonemizer`).
  - Run `KWS_KEYWORD=sheila make_tts_negatives.py`, patched to generate only the **train and validation negatives**: 32,000 + 4,000 clips. Skip the keyword clips (excluded by `--kw-real-only`) and the test splits. Record the patch.
  - Use `--workers` ≤ 8: about 1.9 GB RAM each, and the PC has 31 GB.
- The dataset server ran at about 7 Mbit/s on 2026-10-19, so expect 4–8 h for all downloads.

### B2. Build
- **Vendor** into `research/tools/mateo_eval/`: `train_verifier.py`, `prepare_multicorpus.py`, `make_tts_negatives.py`, `augment_online.py`, `mine_verifier_negatives.py`, and any module they import.
- **Build the training data:**
  - `KWS_MULTI=data/multi_sheila_full python prepare_multicorpus.py --only clips noise`;
  - `python verifier_data.py`, which packs `libri100_*` and the keyword targets.
- **Noise path:** `train_verifier.py:367` reads `data/multi/noise_train.npy`. Point it to `K.MULTI`.
- **Device patch:** `--device` defaults to cuda if available, else cpu. Guard `set_per_process_memory_fraction`, `cufft_plan_cache`, `empty_cache` and `max_memory_allocated` with `dev.type == 'cuda'`.
- **memguard limits:** on 16 GB use `--gpu-fraction .14`, or raise `MAX_GPU` and the private-bytes limit.

### B3. Mine our hard negatives: `research/tools/r2_mine.py` (adapted from `mine_verifier_negatives.py`)
- Score every LibriSpeech train-clean-100 utterance with our H48 K64 seeds 0–2 (the variant chosen in A2): 1 s windows every 250 ms, integer model.
- Find peaks with a 1 s gap (4 windows), at most 3 per utterance, top 8000 per seed. Take the union of the seeds, removing duplicates within ±0.5 s.
- Offset = utterance start + 16000 + 4000 k (the window end). Use the same npz format; `HardWindows` reads only `offset`.
- Skip utterances shorter than 1 s, and clear the feature cache per batch.
- Output: `data_verifier/hard_sheila_r2.npz`.

### B4. Fine-tune (3 seeds)
Mateo's recipe; only the hard negatives and the starting point differ. His CTC base `sheila_logmel` is not in git, so start from `sheila_verifier_head_near.pt`.

```
KWS_KEYWORD=sheila KWS_MULTI=data/multi_sheila_full python train_verifier.py \
  --init research/results/R2/mateo/sheila_verifier_head_near.pt --hard data_verifier/hard_sheila_r2.npz \
  --head-weight 1 --hard-weight 0 --pos-weight 0 --near-share .15 --kw-real-only \
  --epochs 10 --lr 1e-3 --force-start --seed <0|1|2> --name sheila_head_r2_s<seed>
```

Keep `last.pt` of each run (as Mateo did) in `research/results/R2/verifier/` (about 200 KB each, tracked).

### B5. Evaluate
- Score each new verifier on the A1 caches for validation and test.
- Re-select with the A5 rule (`r2_cascade.py select-budget --max-calls 2150`, main) and with the A2 rule (second), then run the test once per rule with `final`.
- Compute cost as in A4.
- 3 verifier seeds × 3 stage-1 seeds = 9 cascades. Report mean ± sd, with a paired comparison against Phase A (McNemar per clip, as `verifier_cascade` does).

**Gate G2: stop and report to Pedro before writing the report.**

## 6. Phase C: mini report and record
**Decision at G2 (Pedro, 2026-10-20):** the report shows **both Phase B results side by side**, as a trade-off between recall and verifier calls, with no single recommendation:
- B under the call limit: 85.8 % live (mean 85.85 %), 1,908 calls/h;
- B under the highest-recall rule: 87.8 % live, 3,044 calls/h.

Phase A, A5 and Mateo's cascade are the comparison points. Energy is labelled by scenario: busy-polling (today's board) is the main number, and clock-gated is marked as a calculation.

**Data and tools:**
- Results in `results/R2/{A,B}/` and `results/R2/{designs,compliance}.csv`, produced by `r2_synthesis.py`.
- Teammates' reference points: reuse `results/R1/references.json`.

**LaTeX:**
- Location: `research/reports/latex/cascade_gru/`, built with `latexmk -pdf main.tex`.
- Class and packages: IEEEtran, with the same packages as `FINAL REPORT/main.tex`.
- Generator: `research/tools/make_r2_latex.py`, following the patterns of `make_r1_latex.py`. Copy `figures/r1_style.tex`.
- Each figure and table is its own `.tex` file, so it can be `\input` into the group report.
- Style: dashed requirement lines and a shaded pass region, as in R1.

**Figures:**
1. Cascade architecture (TikZ, full width):
   - PC sends the window features and the 10 ms frames;
   - pruned SNN on `ksdot`, every 250 ms → t1 trigger → 1.5 s frame buffer → GRU verifier → t2 → LED0;
   - a timing strip: requests, proposal, 107 ms verification.
2. Verifier network: 24 × 150 frames → stack 2 → GRU 48→64 → GRU 64→64 → keyword head (and CTC 40).
3. Requirements scatter: live vs other %, and live vs FA/h. Points: stage 1 alone (R1), cascade A, cascade B (mean ± sd), Mateo's stage 1 and cascade.
4. Operating curves on test: live vs FA/h for stage 1 alone, cascade A, cascade B, plus Mateo's operating point.
5. Cost:
   - (a) worst compute per request, with a 200 ms line;
   - (b) energy per hour stacked (static, stage 1, verifier), with the 1 W line (3,600 J/h);
   - (c) memory stacked against 144 KiB.
6. Margin chart (as R1 Fig. 9) for the final cascade.
7. Verifier calls per hour, and false accepts by source.

**Tables:**
- requirements (reuse R1 `tab_requirements`);
- results with ✓/✗: stage 1 alone vs cascade A vs cascade B vs Mateo;
- cost breakdown;
- fine-tune settings next to Mateo's.

**Report sections**, short as in R1: Aim; Architecture; Method; Results; Trade-offs; Later work (section 7).

**Record:**
- `research/reports/R2_cascade_gru.md`, with hypotheses and outcomes and threats to validity;
- an update to `STATUS.md`;
- a local commit. PDFs only: the root `.gitignore` excludes `*.tex`, so the sources stay local.

## 7b. Next step after R2: clock gating (R3, outline only; Pedro, 2026-10-19)
**Why:**
- With today's busy-polling firmware the core never idles. Every design draws its full PL power: 0.367 W for ours, 0.727 W for Mateo's.
- Fewer verifier calls (A5, Phase B) only save energy when the chip can stop between requests. Clock-gated, the cascade estimates are 0.177 W (A) and 0.110 W (A5), but that is only a calculation (ENERGY.md §5).

**Outline** (to be planned in detail in its own file):
1. **Quick first number, no RTL change:** clock scaling. Set FCLK0 from Python on PYNQ, and relax the server's 100 MHz check. Measure our stage 1 alone, busy vs scaled, with the USB meter (ENERGY.md §3).
2. **Clock gating in RTL:**
   - stop the core clock (BUFGCE) and the BRAM enables between requests;
   - wake on the ARM's mailbox write;
   - the firmware sleeps after it answers.
   Rebuild in Vivado and check timing.
3. **Board check:** bit-exact results (`check_board.py`), then the meter: busy vs gated, stage 1 alone.
4. **The cascade, gated:** after the hardware merge below (verifier firmware, 10 ms frames). Compare A vs A5 vs Phase B by measured energy per hour.

## 7. Later work, not part of R2: the hardware merge
- **Alex's verifier instructions:** the KX unit lives only on `origin/dual-core-alif`.
  - `code/pynqz2_riscv_flow/rtl/kdot_pcpi.v`, commit 076dba1.
  - `firmware/verifier.c` with `-DUSE_KX`, merge commit f93b9c8.
  - KX4 cuts a verifier call from 105.83 to 41.16 ms. That is about 11.6 mJ at 0.282 W.
- **Clashes with Pedro's `ksdot`** (branch Pedro, e0609e4):
  - Same opcode and funct3: `kxload` = `kdotc` (funct3 2) and `ksdot` = `kload` (funct3 3).
  - ABI bit 1 means "sparse kdot" on Pedro but "neuron engine" on Alex's branch.
  - Both define module `kdot_pcpi`.
  - Fix: move `ksdot`/`kxload` to funct7 = 1, or to custom-1, and add a new ABI bit 4.
- **Build risk:** Alex's final build has WNS +0.001 ns, 94.5 RAMB36 and 37,264 LUTs. Our cascade drops his neuron engine, which frees BRAM and timing margin.
- **Firmware:**
  - the PC must also send the 10 ms frames (25 × 24 B = 600 B per hop);
  - the core keeps a 150-frame buffer (3.6 KB);
  - add the t1 trigger to `inference.c`.
- **Then the first board test of any cascade.**

## 8. Practical notes for the new session (learned in R1)
- **Bash heredocs mangle backslashes.** Write Python, LaTeX and patch scripts with the Write tool, then run them.
- **Checking PDFs:** `.venv/Scripts/python.exe -m pip install --target <scratchpad>/pylib pymupdf`, then render pages to PNG with `fitz` and view them. The scratchpad is per session, so this has to be redone.
- **pgfplots and TikZ:**
  - use `\addplot[...]`, not `\addplot+`, which adds the cycle markers;
  - TikZ style names must not clash with keys (`out`, `fill`);
  - put `\draw`/`\fill` styles in `\tikzset`.
- **Unicode:** set `PYTHONIOENCODING=utf-8` when printing `git show` output that contains unicode.
- **R1 caches** in `build/r1_cache/`: live sets, negatives (18.97 h test / 18.11 h validation), 1 h stream. They come from the same seeded audio as Mateo's sets, which makes them useful for equality checks.
- **Timings:**
  - R1 evaluation of 39 models: about 6 min after the caches;
  - verifier scoring: about 1 min per split on CPU;
  - Mateo's stage-1 cache: about 30 min.

## 9. Verification
- Gates G-A1 (bit-exact frontends and models) and G-A2 (Mateo's cascade numbers reproduced).
- The A1 frames equal the G-A2 frames.
- Every threshold comes from validation; each test is run once.
- Every number in the report traces to `results/R2/*.json` / `*.csv`.
- `latexmk` builds cleanly, and every figure and table also compiles inside a copy of the `FINAL REPORT/main.tex` preamble (plus `\input{figures/r1_style}`).

## 10. Rough schedule
| Phase | Estimate |
|---|---|
| A | 1 session, about 3–5 h, including about 1 h of caching for the Mateo reproduction |
| B | 2–4 h of setup; 4–8 h of unattended downloads and TTS (overnight); fine-tunes 3 × about 20–60 min |
| C | about half a day |
