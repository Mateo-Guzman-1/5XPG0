# Research status

Update this file at the end of every session: tick the experiment, add its key numbers,
note open issues and the next step.

## Progress

| Exp | Title | Status | Report |
|---|---|---|---|
| E0 | Setup, baseline reproduction, shared tools | **next** | — |
| E1 | Evaluation sets for near-miss words (user recordings) | to do | — |
| E2 | Network size sweep (dense, current encoding) | to do | — |
| E3 | Input encoding comparison (current / rate / temporal) | to do | — |
| E4 | Temporal-convolution SNN topology | to do | — |
| E5 | Hard-negative training data | to do | — |
| E6 | Weight precision int16 vs int8 | to do | — |
| E7 | Decision rule and end-to-end latency | to do | — |
| E8 | Synthesis and conclusion | to do | — |

## Starting point (before the programme, 2026-09-27)

Release model `deploy/model.npz` (dense 768→64→2 LIF, current encoding, 12 steps,
int16 Q10), from Mateo's REPORT.md and our board runs:

| Quantity | Value | Source |
|---|---|---|
| Test precision / recall / F1 | 90.15% / 85.20% / 0.8761 | REPORT.md |
| Negative-clip FPR | 0.368% | REPORT.md |
| Max cycles, RV32IM | 5,981,521 (59.82 ms @ 100 MHz) | board, Ethernet path |
| Max cycles, `kdot` | 267,393 (2.67 ms) | board, Ethernet path |
| Model bytes | 98,824 of 147,456 (144 KiB) | REPORT.md |
| Live stream, 2 of 3: "yes" / other words | 67.3% / 0.13% | results/confusables.json |
| Rate encoding (64 neurons), max latency | 715.52 ms, misses the 250 ms budget | REPORT.md |
| PC ↔ board round trip per window | ~7 ms | JOURNAL entry 8 |

Known quality problem (user, live mic): near-miss words fire as "yes": onset
confusables "les, mes, pes, des" and ending confusables "yech, yep, ye, yea/yeah".
Mateo's augmented trial model (`results/models/augmented_current_seed2.npz`) reduced
ending confusions but lost recall.

## Environment notes
- PyTorch is **not installed yet** (E0). The PC has an AMD GPU, so training runs on
  the CPU (Ryzen 5 7600X, 31 GB RAM).
- The dataset is **not downloaded yet** (E0; 2.4 GB into `code/snn_keyword/data/`).
- The firmware builds with the Vitis gcc; board access via `ssh pynq`.

## Open issues / decisions
- (none yet)

## Next step
Run **E0**.
