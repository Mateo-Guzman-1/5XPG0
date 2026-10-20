# R2: our pruned window SNN as the first stage of Mateo's GRU-verified cascade

*Dates:* 2026-10-19 to 2026-10-20
*Branch:* Pedro
*Plan:* [PLAN_R2_GRU.md](../PLAN_R2_GRU.md)
*Data:* `results/R2/`
*LaTeX report:* `reports/latex/cascade_gru/` (`main.pdf`; every figure and table is its own `.tex` file for the group report)

## Design codes (used in the LaTeX report, which names no people)
| Code | Design | Names used in this record and in the result files |
|---|---|---|
| W0 | window SNN alone (our pruned H48 K64) | "stage 1 alone (R1)", `stage1_R1` |
| S0 | streaming SNN alone | "Mateo's stage 1", `mateo_stage1` |
| S1 | streaming SNN + original GRU verifier | "Mateo's cascade", `mateo_cascade` |
| A1 | window SNN + original verifier, highest-recall rule | "A", `cascade_A` |
| A2 | window SNN + original verifier, ≤ 2,150 calls/h | "A5", `cascade_A_calls2150` |
| B1 | window SNN + fine-tuned verifier, highest-recall rule | "B, highest recall", `cascade_B` |
| B2 | window SNN + fine-tuned verifier, ≤ 2,150 calls/h | "B, call limit", `cascade_B_calls2150` |

The number gives the threshold rule: 0 no verifier, 1 highest live recall, 2 at most 2,150 verifier calls per hour. In the report, Alex's work appears as "the parallel-processing design" and Mateo's as "the streaming design".

## Summary
| | |
|---|---|
| **Question** | Can our cheapest design (H48 K64 on `ksdot` + table: 0.35 ms, 12.4 KiB, 96.5 µJ per window) be the first stage of Mateo's cascade, with his GRU verifier unchanged (A) or fine-tuned on our false proposals (B)? Does it meet the group requirements, and at what cost? |
| **Method** | Mateo's cascade harness (`verifier_cascade.py`, vendored from `origin/ALIF-Layer1` a856a03) with only the stage-1 scorer replaced. Thresholds chosen on validation (≤ 0.2 % other words, ≤ 1.6 FA/h), test run once. Two rules: highest live recall (Mateo's), and the same with ≤ 2,150 verifier calls per hour (added after the first test, fixed before its own test). B: 3 fine-tuned verifiers × 3 stage-1 seeds. |
| **Detection** | A: 87.6 ± 0.3 % live, 1.83 FA/h, but 10,244 verifier calls per hour. A5 (call limit): 84.4 ± 0.9 %, misses by 0.6 points. **B, call limit: 85.8 ± 0.8 % (mean 85.85 %, 7 of 9 ≥ 85 %), 1,908 calls/h. B, highest recall: 87.8 ± 1.0 % (all 9 ≥ 85 %), 3,044 calls/h.** Both B results meet all seven requirements. |
| **Cost** | Worst compute 107.6 ms; memory 64.5 KiB; word end to decision 162–178 ms; PL power 0.367 W as built (busy-polling), half of Mateo's 0.727 W. With clock gating (a calculation): 0.110 W (call limit) to 0.177 W (A). |
| **Choice** | Pedro (G2): show both B results side by side, as a recall-vs-calls trade-off; no single recommendation. |

## Hypotheses (PLAN_R2_GRU.md §2, A5) and outcome
| Hypothesis | Outcome |
|---|---|
| R2-H1: with the unchanged verifier, live recall rises to 70–85 % at ≤ 2 FA/h and ≤ 0.2 %; it may fall short of 85 % | **Exceeded:** 87.6 %, but only because the highest-recall rule set t1 below zero. Our stage 1 then passes 57–85 % of all requests (validation) and the cascade reaches the verifier's own limit (87.75 % on validation, the same as Mateo's cascade). |
| R2-H2: fine-tuning on our false proposals raises live recall by ≥ 3 points at the same limits, or cuts verifier calls by ≥ 20 % | **Second branch met:** at the same recall, 70 % fewer calls under the highest-recall rule (3,044 vs 10,244 per hour). First branch not met: +1.4 points under the call limit, +0.3 points under the highest-recall rule. |
| R2-H3: compute ≈ 107.3 + 0.35 ms ≤ 200 ms; memory ≈ 64.5 KiB ≤ 144 KiB; power ≪ 1 W; energy per hour below Mateo's cascade | **Mostly confirmed:** 107.6 ms, 64.5 KiB, 0.367 W as built (Mateo 0.727 W). Work energy (Alex's method) is 77 J/h (A5), 76 J/h (B, call limit) and 120 J/h (B, highest recall) against Mateo's 277 J/h. Cascade A, with 400 J/h, is above it. |
| R2-H4: median word end to decision 0.15–0.35 s | **Confirmed:** 162–178 ms. |
| R2-H5 (A5): with ≤ 2,150 calls/h, live 83–86 %, calls near 2,150, FA/h ≤ 2, energy about 5× lower | **Confirmed:** 84.4 %, 1,937 calls/h, 1.09 FA/h, 400 → 77 J/h (work energy; 307 → 59 J/h dynamic only). |

## Checks
- **G-A1:** Mateo's window `features()` and `integer_forward` equal Pedro's bit for bit (32,081 live windows; 3 seeds of H48 K64): `results/R2/gate_A1.json`.
- **G-A2:** Mateo's cascade rerun here reproduces his test report exactly: 0.8679 live, 0.9195 complete, 0.0007 other words, 1.424 FA/h over 18.97 h, stream 0.8786, 0.175 s, every MSWC row.
  - His validation selection is identical too (t1 = 3785, t2 = 1524).
  - Two small differences: verifier calls are 2,222.0 vs 2,221.5 per hour, and his stage-1-alone stream recall is 0.664 vs 0.693 in his stored file (4 of 140 keywords; unexplained, cascade numbers unaffected).
  - His held-out microphone and room rows were not rerun.
- **Our caches:** every non-stage-1 array equals Mateo's cache (same audio, frames, items). The fast margin equals the integer model on 20,000 windows per split: `A/cache_check_*.json`.
- **Call-limited rule:** with Mateo's own stage 1 (2,149.6 calls/h) it returns his thresholds unchanged.
- **Figure data:** every operating point marked in the curves reproduces its reported test numbers, including R1's stage-1-alone result (0.3962 live, 3.638 FA/h, seed 1) through a different code path. False accepts by source sum to the reported FA/h (asserted).
- **Paired** (`B/paired_vs_A.txt`): under the call limit, B gains 32 test keywords over A5 and loses 5 across the 9 cascades. No single cascade differs significantly (exact McNemar, p ≥ 0.06).

## Observations
- **Our stage 1 is a weak proposer.** It ranks some real "Sheila" windows below much running speech. On test, at ≤ 0.2 % and ≤ 2 FA/h, 85 % needs 2,129 calls per hour with the fine-tuned verifier and 2,735 with Mateo's. Mateo's stage 1 gives 84 % with ≤ 120 calls and 85 % from 971.
- **Fine-tuning moves the curve up at a fixed number of calls.** It does not raise the verifier's own limit measurably.
- **Paired with Mateo's stage 1, the fine-tuned verifiers reach his validation ceiling (87.75 %) with 177–714 calls/h** instead of 2,150. Not tested; a side result of the selection.
- **Other words rise** from 0.03–0.10 % (A, A5) to 0.15 % (B), still inside 0.2 %. One B cascade sits exactly at 0.20 %.
- **LibriSpeech running speech is 74–83 % of the false accepts in every cascade**, Mateo's included (74 %; our stage 1 alone: 87 %).
- **With busy-polling every verifier call is free.** Fewer calls save energy only with an idle mode (R3).

## Threats to validity
- **Seeds:** 3 stage-1 seeds and 3 verifier seeds. One test keyword is 0.47 points (212 keywords). The call-limited B mean (85.85 %) is within one sd of 85 %.
- **FA/h counts:** 2 per hour over 19 h is about 38 events (Poisson ± 6).
- **Rule order:** the call limit was added after cascade A's test result was seen. It was fixed before its own test, and its value (2,150) is Mateo's, not tuned.
- **Verifier calls** are counted on the 19 h negatives stream, which is dense speech. A quiet room would need fewer calls.
- **Energy:** a Vivado estimate (S0 method), not a board measurement. Clock gating is a calculation. Mateo's figures use Alex's dual-core fabric (0.727 W), ours the `ksdot` fabric (0.367 W).
- **No board run:** no cascade has run on a board. The verifier time (107.3 ms) is Alex's RTL simulation on plain `kdot`.
- **Training:**
  - GPU training (PyTorch 2.14.0+rocm10.1.0, MIOpen off) is not bit-reproducible.
  - Mateo's recipe keeps only the 4,218 of 32,000 synthetic training words that have dictionary pronunciations.
  - Our fine-tune starts from his final verifier, not from his CTC base (not in git).
- **Rounding:** the Phase B commit message (342c301) quotes 85.9 ± 0.9 % for B under the call limit, rounded up. The exact value is 85.85 %, shown as 85.8 ± 0.8 % everywhere else.

## Reproduce
Phase A (`.venv`), from `code/snn_keyword`, with `KWS_KEYWORD=sheila`:
```bash
python research/tools/r2_gate_a1.py
python research/tools/mateo_eval/verifier_cascade.py cache --stage1 research/results/R2/mateo/sheila_stream_int8.npz --split validation   # and --split test
python research/tools/mateo_eval/verifier_cascade.py score research/results/R2/mateo/sheila_verifier_head_near.pt --score head --split validation --out data_verifier/vscores_mateo_head_near_validation.npz   # and test
python research/tools/r2_cascade.py cache --split validation      # and test; our 3 seeds x 2 variants
python research/tools/r2_cascade.py select --seed <s> --variant <m|c> <scores> --verify-ms 107.3 --max-fa-hour 1.6 --out ...
python research/tools/r2_cascade.py choose
python research/tools/r2_cascade.py select-budget <scores> --max-calls 2150
python research/tools/r2_cascade.py final --seed <s> --variant m <test scores> --t1 .. --t2 .. --stage1-threshold .. --verify-ms 107.3 --out ...
```
Phase B (`.venv-rocm` for data and training):
- data: `fetch_corpora.py` librispeech train-clean-100, musan, rirs, mswc train; `make_tts_negatives.py`; `prepare_multicorpus.py --only clips noise` with `KWS_MULTI=data/multi_sheila_full`; `verifier_data.py`;
- mining and training: `research/tools/r2_mine.py`, then `train_verifier.py` with the command in PLAN_R2_GRU.md B4 plus `--gpu-fraction .14 --max-instances 3`;
- evaluation: `score`, `select-budget --variants m --out-dir results/R2/B/v<v>/budget2150`, `select`, `final`, `paired`.

Synthesis and report:
```bash
python research/tools/r2_synthesis.py                  # designs.csv, models.csv, compliance.csv
python research/tools/r2_cascade.py figdata            # figdata_test.json (test curves for figures only)
python research/tools/make_r2_latex.py                 # figures/ and tables/
cd research/reports/latex/cascade_gru; latexmk -pdf main.tex
```
