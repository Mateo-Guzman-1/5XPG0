# Plan R1: the synapse-study designs measured against the group requirements

Written 2026-10-19, before any R1 result. This follows S0–S6 ([PLAN_SYNAPSE.md](PLAN_SYNAPSE.md)).

## Why
S0–S6 judged quality with our own rules. Those were test F1 on the full split and live 2-of-3 at a stream threshold with at most 0.3 % validation false accepts. The group report needs every design judged against the same requirements and the same evaluation data as the other designs (Mateo's ALIF + GRU cascade, Damien's encoding model).

## Requirements (Pedro, 2026-10-19)

| Requirement | Target | How it is measured here |
|---|---|---|
| Live "Sheila" detected | ≥ 85 % | Mateo's live set: 212 test clips in 2.25 s of noise, 250 ms hop, 2-of-3 confirmed score |
| False accepts per hour | ≤ 2 | Mateo's `negative_stream('test')`: every non-keyword SC test word plus LibriSpeech test-clean and test-other, about 19 h, 1 s hold-off |
| Other words accepted | ≤ 0.2 % | 3,000 SC test words in the same live set |
| Word end to decision (median) | ≤ 500 ms | Mateo's 1 h test stream (median latency), plus the worst compute and the PC↔board round trip |
| Worst compute per 250 ms request | ≤ 200 ms | t_max from Saturday's bit-exact board runs |
| Model memory | ≤ 144 KiB | firmware model section |
| Average power while listening | ≤ 1 W (PL only) | ENERGY.md scenarios: busy-polling, clock-gated, clock-scaled; equivalently ≤ 250 mJ per 250 ms window |

## Data
- **Training (unchanged):** the full official Speech Commands v2 train split plus our hard negatives.
- **Thresholds, on validation only, Mateo's rule (`stream_select.py`):** the highest live recall such that live other words ≤ 0.2 % and FA/h ≤ 2 on `negative_stream('validation')` (dev-clean, dev-other and SC validation words).
- **Tests:** every requirement is tested once, at that fixed threshold.
- **Clip F1:** on the full test split (11,005 clips), and on Damien's 3,212-clip subset for comparison with the Encoding and Quantization sections.
- **Not used here:** Mateo's extra training corpora (MSWC, Piper TTS, MUSAN, rooms, microphones). They are a follow-up.

## Designs (13 configs × 3 seeds; the worst S-designs dropped)
- **Dense:** h16, h32, h48, h95 (16-bit); h48_b8, h48_b4.
- **Pruned:** h48_k64/128/256/512; h48_b8_k64/128/256.
- **Hardware variants and cycles:** as measured on 2026-10-17 (same models, same sha256).

## Hypotheses
- **R-H1:** no design reaches 85 % live recall under the 0.2 % + 2 FA/h rule. Saturday's best was 65.6 % at the looser 0.3 % rule.
- **R-H2:** the FA/h limit, not the 0.2 % limit, sets the threshold. Running speech (LibriSpeech) causes most false accepts, as for Mateo's window baseline (52 FA/h on his 1 h stream).
- **R-H3:** every kdot/ksdot design meets compute (≤ 200 ms), memory and power (≤ 1 W) with a wide margin. RV32IM designs also meet compute and power.
- **R-H4:** median word-end-to-decision is far below 500 ms (window model: about 0.1–0.3 s).

## Choice rule (fixed before the results; Pedro, 2026-10-19)
- **Accuracy measure:** mean live recall at the validation-chosen threshold.
- **Candidates:** only designs that pass every cost requirement. A design meeting *all* requirements wins outright; if several do, the lowest E_inf wins.
- **Dense champion D:** the best dense design. Designs within one seed sd of it are ties; among ties the lowest E_inf wins.
- **Pruned champion P:** chosen the same way among the pruned designs.
- **Verdict:**
  - If P ≥ D − sd(D): recommend P alone.
  - Otherwise: present both, with the recall-versus-energy trade-off.
- **Sensitivity checks:** the same rule with the pooled sd, and with test F1 as the accuracy measure.

## Outputs
- `results/R1/`: per-model JSONs, designs and compliance CSVs.
- `reports/R1_requirements.md`.
- `reports/latex/requirements_eval/`: IEEEtran report; each figure and table in its own `.tex` file.
