# R1: synapse-study designs measured against the group requirements

*Date:* 2026-10-19
*Branch:* Pedro
*Plan:* [PLAN_R1.md](../PLAN_R1.md)
*Data:* `results/R1/`
*LaTeX report:* `reports/latex/requirements_eval/` (`main.pdf`; every figure and table is its own `.tex` file for the group report)

## Summary
| | |
|---|---|
| **Question** | Do the S-study designs meet the group requirements on the same test sets as the teammates' designs? |
| **Method** | Mateo's harness, vendored unchanged from `origin/ALIF-Layer1` a856a03 in `tools/mateo_eval/`. Thresholds are chosen on validation with his rule (≤ 0.2 % other words, ≤ 2 FA/h); then each test is run once. Cost comes from the S-study bit-exact board runs. 13 configs × 3 seeds = 39 models. |
| **Detection** | No design reaches 85 % live. The best mean is 42.9 ± 6.9 % (H48 8-bit), and no single model exceeds 50 %. Every design keeps other words at ≤ 0.07 %. FA/h is ≤ 2 on validation but 1.6–4.1 on test (mean 2.8); only 2 of 39 models are ≤ 2. 86 % of false accepts are LibriSpeech running speech. |
| **Cost** | Every design meets compute (≤ 89 ms), memory (≤ 143 KiB), latency (134–258 ms) and power (PL 0.36–0.37 W busy-polling, budget 1 W = 250 mJ per window). |
| **Choice** | Dense champion: H48 8-bit, kdot + table (the deployed model): 42.9 %, 165 µJ, 72.4 KiB. Pruned champion: H48 K64, ksdot + table: 36.3 %, 96.5 µJ, 12.4 KiB. The 6.6-point gap is less than sd(D) = 6.9, so the **pruned design is recommended** (−42 % energy, −83 % memory). The choice is the same with the pooled sd and with test F1. |

## Hypotheses (PLAN_R1.md) and outcome
| Hypothesis | Outcome |
|---|---|
| R-H1: no design reaches 85 % live | **Confirmed.** 26–43 % (design means). |
| R-H2: the FA/h limit sets the threshold | **Confirmed.** At ≤ 0.2 % alone the dense champion would detect 59 % (8.2 FA/h). With ≤ 2 FA/h it detects 40 %. |
| R-H3: kdot/ksdot designs meet compute, memory and power with wide margins; RV32IM meets compute and power | **Confirmed.** All 26 designs (config × hardware) pass all five cost and latency requirements. |
| R-H4: median latency far below 500 ms | **Confirmed.** The 1 h stream median is 106–249 ms; with compute and link, 134–258 ms. |

## Checks
- **Determinism:** retraining h48_b8 (3 seeds) reproduces Saturday's weights exactly. Saturday's models are reused, and their sha256 is checked against both the quality and the board JSONs.
- **Harness agreement:** at Saturday's stream thresholds, Mateo's live set reproduces `test.live.two_of_three` exactly for all 39 models.
- **Fast forward pass:** the fast exact float64 forward equals `model.integer_forward` on 3,000 windows per model.
- **Sets:**
  - live: 212 + 3,000 test clips (204 + 3,000 validation);
  - negatives: 18.97 h test (Mateo: 18.97 h) and 18.11 h validation;
  - 1 h stream: 140 / 455 / 284 clips, as in Mateo's config;
  - Damien's subset: 9,606 / 2,704 / 3,212 clips (1,606 / 204 / 212 positives).

## Observations
- **The old rule versus the new one.** At the S-study threshold (≤ 0.3 % on validation windows), the same models detect 49–67 % live but make 8–95 FA/h. The requirement threshold costs about 20 points of live detection.
- **No threshold reaches 85 %.** Along the test operating curves, even 2 % other words and about 70 FA/h give at most 77 % (H48 8b) and 72 % (H48 K64).
- **Clip F1 does not change** (same models, same window threshold): 0.76–0.79 on the full test split and 0.81–0.85 on Damien's subset, where fewer negatives raise the precision.
- **The levers move cost, not detection.** Size (≥ 32 neurons), bits and K all stay within the seed spread; H16 is lower (26 %).
- **Teammates on the same tests:**
  - Mateo's ALIF SNN + GRU cascade: 86.8 %, 1.42 FA/h, 0.07 %, latency 175 ms, worst compute 117 ms (meets all detection requirements);
  - his streaming SNN alone: 63.7 %, 1.58 FA/h;
  - Damien's 8×16 model: 7.1 % under Mateo's rule, or 80.2 % / 366 FA/h at its own margin.

## Threats to validity
- **Seeds:** 3 per design; the live sd is 2–12 points, so the dense versus pruned difference is not resolved.
- **FA/h counts:** 2 per hour over 19 h is about 38 events (Poisson ± 6). The validation-to-test gap (≤ 2 → mean 2.8) is partly this noise and partly a domain gap (running speech).
- **Training data:** Speech Commands plus our near misses only. Mateo's extra corpora were not used (deferred, Pedro's choice).
- **Energy:** a Vivado estimate (S0 method). ksdot is an experimental bitstream.

## Reproduce
```powershell
.venv/Scripts/python.exe code/snn_keyword/research/tools/mateo_eval/fetch_corpora.py librispeech --subset test-clean   # also test-other, dev-clean, dev-other
.venv/Scripts/python.exe code/snn_keyword/research/tools/req_eval.py          # about 6 min after the feature caches (build/r1_cache/)
.venv/Scripts/python.exe code/snn_keyword/research/tools/r1_synthesis.py      # designs.csv, compliance.csv, choice.json, references.json
.venv/Scripts/python.exe code/snn_keyword/research/tools/make_r1_latex.py     # figures/ and tables/ of the LaTeX report
cd code/snn_keyword/research/reports/latex/requirements_eval; latexmk -pdf main.tex
```
