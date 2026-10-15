# Sheila encoding verification

The five evaluation levels complement detector accuracy: reference-grid fidelity,
fixed landmark proxies, class separation and nearest-centroid probing, deployed
model occlusion, and ten-seed paired SNN evaluation. Direct diagnostics cover
61 unique configurations from both encoding sweeps, including the passband /
resolution controls. Fidelity and landmark calculations use 128 positive and
128 negative validation clips (landmarks use the positives). Probes fit on the
training split, choose their threshold on validation, and report test metrics
with 500 speaker-bootstrap replicates. These measures are not information-
theoretic upper bounds or phoneme-aligned measurements.

## Reproduction

From `code/snn_keyword`, using the project Python environment:

```powershell
python research/verify_encoding_reproducibility.py `
  --dataset data/speech_commands_v2_sheila_subset `
  --primary-cache data/frontend_sweep_cache_sheila_v2 `
  --extended-cache data/extended_encoding_cache_sheila_v2 `
  --deployed-checkpoint research/results/extended_encoding_sweep_sheila_v2_models/b8_f8_tb16_hz300-3400_t1000_g3000_s20000_mapequal_w25_h10_qlinear_mabsolute_confirm_seed0.pt `
  --models research/results/encoding_reproducibility_models_b128 `
  --seeds 10 --epochs 12 --samples-per-epoch 4096 --batch-size 128

python research/benchmark_reproducibility_board.py `
  --board 192.168.2.99 `
  --report research/results/encoding_reproducibility.json `
  --dataset data/speech_commands_v2_sheila_subset `
  --cache-dir data/reproducibility_cache `
  --models research/results/encoding_reproducibility_models_b128 `
  --exports data/encoding_reproducibility_board_exports_b128
```

Use a new model directory when changing training settings: existing checkpoint
files are reused. The board script temporarily deploys each seed-zero model and
restores the baseline and its enabled startup service in a `finally` block.
The reference dataset must use the same deterministic path order as cached
features; remove/rebuild caches for a changed dataset.

## Results (2026-10-15)

| Encoding | Accuracy % | F1 mean ± seed SD % | Paired F1 difference, 95% CI (points) | FPGA ms | Parameter bytes | Input bits |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 8 bands, 300–3400 Hz | 98.35 | 87.46 ± 1.24 | — | 10.40 | 6440 | 1024 |
| 20 bands, 300–3400 Hz | 98.38 | 87.52 ± 1.34 | +0.06 [-1.26, 1.37] | 21.19 | 15656 | 2560 |
| 16 bands, 300–6500 Hz | 98.14 | 85.85 ± 1.39 | -1.61 [-2.69, -0.54] | 17.57 | 12584 | 2048 |
| 8 bands, 40–5120 Hz | 98.27 | 86.85 ± 1.31 | -0.61 [-1.81, 0.59] | 10.40 | 6440 | 1024 |
| 8 bands, 4-bit logarithmic | 98.28 | 86.68 ± 1.55 | -0.78 [-1.76, 0.21] | 10.40 | 6440 | 512 |

The paired intervals are unadjusted t intervals across ten matched seeds,
conditional on this fixed test set. Exact McNemar tests are saved per seed;
Holm-adjusted majority-vote McNemar p-values are 1, 1, 1 and 0.474. Majority
votes represent ten-model ensembles, not the deployed single model, and test
correctness rather than F1. No candidate establishes a clear improvement over
the compact baseline. The wide 16-band control has lower seed-averaged F1.

All 200 FPGA reference vectors matched Python integer spike counts. The restored
live WAV path detected Sheila with spikes [4,10] and 1,039,911 cycles; the board
service was active and enabled after the run.

The 40–5120 Hz encoding has higher reference correlation (0.87 versus 0.75)
and descriptive nearest-centroid F1 (43.51% versus 38.24%), but lower mean SNN
F1. This is consistent with capacity/optimization limitations; it does not
prove their cause. Widening to 6500 Hz retains more frication proxy contrast
(0.96 versus 0.72) without improving task performance. Four-bit logarithmic
coding lowers quantization SNR from 45.52 to 24.68 dB while halving input bits.
Occlusion identifies both upper-band and low/mid-frequency contributions.

False accepts per hour were not measured. They require a continuous background
corpus and a specified streaming/detection-merging protocol. Speech Commands v2
does not provide the suggested near-miss words she, sheep or shield. The
landmark regions have no phoneme annotations, and fidelity can penalize useful
noise rejection. The report therefore treats the compact baseline as a
supported deployment trade-off, not a uniquely optimal encoding.
