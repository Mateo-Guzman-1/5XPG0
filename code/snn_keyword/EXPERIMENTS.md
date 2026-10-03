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
