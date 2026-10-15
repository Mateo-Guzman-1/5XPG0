# Training experiments — "sheila" keyword SNN

Owner: Khalid (training). Script: `train_sheila.py`. Raw numbers for every run
are logged automatically in `runs/experiments.csv`; this file records the
**reasoning**: what was asked, what was expected, what happened, what was
decided.

## Fixed during all training experiments (not my variables)

| Setting | Value | Owner |
|---|---|---|
| Data | Speech Commands v0.02, "sheila" vs other words, `NEG_RATIO=3`, `DATA_SEED=0`, official speaker-disjoint split | data |
| Features | 16×16 spectrogram, `features.py` (`MIN_PEAK=1.0`) | data / encoding |
| Encoding | rate (same input every step), T = 25 | encoding |
| Network | 256-64-2 LIF, β = 0.9, threshold 1.0 | size |

Metrics (validation set): accuracy, **recall** (share of "sheila" caught),
**false-alarm rate** (share of other words fired on). Test set used once, at
the end (`--final`).

## Variables I study

epochs · learning rate · LR schedule · surrogate gradient (shape, slope) ·
batch size · training seed (repeatability)

---

## Experiment template (copy for each new experiment)

### E? — <title>
- **Question:**
- **Hypothesis (before running):**
- **Runs:** `<names>` — command(s):
- **Result:** (numbers from `runs/experiments.csv` / `compare_runs.py` plot)
- **Conclusion + design decision:**

---

## E0 — Baseline: skeleton settings on real data
- **Question:** how well does the unchanged skeleton (1 epoch, lr 1e-3,
  batch 16, fast-sigmoid slope 25) do on real audio?
- **Runs:** `baseline` — `python train_sheila.py --note "skeleton settings on real data"`
- **Result (validation):** acc 0.917, recall 0.686, false alarm 0.007.
  32 of 816 clips produced **no output spikes at all** (counted as "not sheila").
- **Conclusion:** cautious detector: few false alarms but misses ~31 % of
  "sheila". Silent outputs suggest under-training.

## E1 — Does more training reduce misses? (first check)
- **Question:** are the misses caused by under-training?
- **Hypothesis:** more epochs → higher recall and fewer silent-output clips.
- **Runs:** `epochs3` — `python train_sheila.py --name epochs3 --epochs 3`
- **Result (validation):** recall 0.686 → 0.819, silent clips 32 → 11,
  false alarm 0.007 → 0.010, train loss 0.287 → 0.134.
- **Conclusion:** supported so far. Next: longer runs (e.g. 10, 20 epochs) to
  find where validation stops improving (overfitting), and repeat with 2-3
  seeds to check the differences are larger than seed-to-seed noise.

## E2 — Epoch sweep: where does validation stop improving?
- **Question:** how many epochs until validation recall plateaus, and does the
  false-alarm rate exceed the 10 % limit (smart-home wake word) on the way?
- **Hypothesis:** recall keeps rising for a few epochs, then flattens while
  train loss keeps falling (start of overfitting); false alarms creep up as
  the net becomes more willing to fire.
- **Runs:** `epochs20` — `python train_sheila.py --name epochs20 --epochs 20`
  (validation after every epoch in `runs/epochs20/history.csv`).
  Plot: `python plot_tradeoff.py epochs20 --epoch 10` → `runs/epochs20/tradeoff.png`.
- **Result (validation, seed 0):** recall 68.1 % (ep 1) → 87.3 % (ep 3) →
  92.6 % (ep 10) → 93.6 % (ep 20); accuracy flat at 0.95–0.96 from epoch 3,
  recall jumps ±4 points between epochs; train loss 0.293 → 0.064 (ep 10) →
  0.034 (ep 20). False alarms 0.8 % → 2.9 % (ep 10) → 3.9 % (ep 20), never
  near the 10 % limit. Epoch 1 matches `baseline_mac` exactly (same machine,
  same seed).
- **Conclusion:** most of the gain is in the first 3 epochs; beyond ~10,
  falling train loss with flat validation = early overfitting. Tentative
  choice: 10 epochs. The 10 vs 20 difference (2 more hits, 6 more false
  alarms) is the size of the epoch-to-epoch noise → check with seeds (E3).

## E3 — Seed repeat: is the epoch 10 vs 20 difference real?
- **Question:** are the differences in E2 larger than seed-to-seed variation?
- **Runs:** `epochs20_seed1`, `epochs20_seed2` — same as E2 with `--seed 1/2`.
- **Result (validation, mean ± sd over seeds 0, 1, 2):**

  | Epoch | Train loss | Recall | False alarm | Accuracy |
  |---|---|---|---|---|
  | 1 | 0.291 ± 0.012 | 74.3 ± 10.3 % | 1.4 ± 1.0 % | 0.925 ± 0.018 |
  | 3 | 0.128 ± 0.010 | 85.8 ± 2.1 % | 1.9 ± 1.0 % | 0.951 ± 0.006 |
  | 10 | 0.064 ± 0.006 | 89.9 ± 2.5 % | 2.1 ± 0.9 % | 0.959 ± 0.006 |
  | 20 | 0.034 ± 0.004 | 91.3 ± 2.8 % | 2.5 ± 1.3 % | 0.960 ± 0.004 |

  False alarms epoch 10 → 20 per seed: 18 → 24, 13 → 8, 7 → 14 (of 612).
- **Conclusion:** the plateau is real for every seed. The "more false alarms
  at epoch 20" seen for seed 0 is NOT: seed 1 goes the other way, and the
  mean differences (1.5 points recall, 0.4 points false alarm) are smaller
  than the seed spread. **Decision: 10 epochs**, because training longer gives
  no measurable gain (not because 20 is measurably worse). Training cost is
  paid once on the PC; inference cost on the board is unchanged.
  Next: train the final `epochs10` model (only final weights are saved) and
  check float vs. quantized accuracy with `export_weights.py`.
