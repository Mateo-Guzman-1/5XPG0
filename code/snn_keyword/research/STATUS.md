# Research status

Update this file at the end of every session: tick the experiment, add its key numbers,
note open issues and the next step.

## Progress

| Exp | Title | Status | Report |
|---|---|---|---|
| E0 | Setup, baseline reproduction, shared tools | tools done (built during E2); report to do | — |
| E1 | Evaluation sets for near-miss words (user recordings) | **next** (needs the user) | — |
| E2 | Network size sweep (dense, current encoding) | **done** 2026-10-04 | [E2_size_sweep.md](reports/E2_size_sweep.md), background: [E2_network_size_explained.md](reports/E2_network_size_explained.md) |
| E3 | Input encoding comparison (current / rate / temporal) | to do | — |
| E4 | Temporal-convolution SNN topology | to do | — |
| E5 | Hard-negative training data | to do | — |
| E6 | Weight precision int16 vs int8 | to do | — |
| E7 | Decision rule and end-to-end latency | to do | — |
| E8 | Synthesis and conclusion | to do | — |

## Keyword change to "Sheila" (2026-09-30, before E0)

The keyword is now **"Sheila"** (Speech Commands word `sheila`); see `JOURNAL.md` entry 9
and the "Keyword change" note at the top of PLAN.md. The first Sheila release (evidence in
`code/snn_keyword/results/sheila/`) was replaced on 2026-10-04 by the hard-negative model
(evidence in `results/sheila_hardneg/`, section below). The "yes" release stays in git
(commit 05ad009) and in the files directly in `results/`.

Already done for E0 during the switch:
- PyTorch 2.11.0+cpu and matplotlib installed in `.venv` (Python 3.14.7).
- Dataset downloaded and SHA256-checked; `data/features.npz` labelled for "sheila"
  (`prepare_data.py --keyword sheila`; `--keyword yes` relabels in seconds).
- CPU training time measured: **about 10 s per run** (9.8–10.4 s, release recipe,
  Ryzen 5 7600X). Full 3-seed sweeps are cheap; no reduced settings are needed.
- `research/tools/build_firmware.py` (rebuilds the "yes" release byte for byte) and
  `code/snn_keyword/check_board.py` (board against the oracle: 40 vectors, cycles,
  "2 of 3" patterns, streamed WAVs).
- Sheila baseline trained, evaluated and verified on the board (numbers below).

Built during E2 (2026-10-04):
- `research/tools/bench_board.py`: build, upload, restart the server with that firmware
  (`start_board.sh <variant> <fw.bin>`), then the `check_board.py` checks.
- `research/tools/eval_model.py`: one quality JSON per model, reusing `confusables.py`,
  `tune_stream.py` and `robustness.py`.
- `benchmark_inputs.py`: input fingerprint.
- `size_sweep.py`.

Both measuring tools were calibrated on the deployed model: 73 of 73 stored quality values
and the stored cycle counts were reproduced exactly. Still to do in E0: the E0 report.

## E2 results: network size (2026-10-04)

Sizes 8, 16, 32, 48, 64, 80, 95 × seeds 0–2, deployed recipe (hard negatives 0.3). All 21
models are bit-exact on the board. Report: `reports/E2_size_sweep.md`; data: `results/E2/`.

| Quantity | Value |
|---|---|
| Model memory | 1,544 bytes per neuron + 8 (measured within 72 B); 95 neurons = 99.5% of 144 KiB |
| RV32IM cycles | 93,340 per neuron + 7,900 (R² 0.999999); 95 neurons = 8.88 M (88.8 ms, 35.5% of the budget) |
| `kdot` cycles | 4,056 per neuron + 7,950 from 32 neurons on; 2,700–2,800 per neuron at 8 and 16 (loop unrolled by GCC, state in registers); 95 neurons = 0.39 M (3.9 ms) |
| Round trip | inference + 0.6–1.5 ms (`kdot`: 1.7–5.0 ms, steps of ~1.1 ms from the server's polling); PC features 0.40 ms |
| Test F1 (mean ± sd) | 8: 0.690 ± 0.074 · 16: 0.764 ± 0.036 · 32: 0.760 ± 0.036 · **48: 0.790 ± 0.006** · 64: 0.764 ± 0.015 · 80: 0.775 ± 0.018 · 95: 0.781 ± 0.003 |
| Live 2 of 3, "Sheila" detected | 8: 51.1% · 16: 61.2% · 32: 59.8% · 48: 63.7% · 64: 54.4% · 80: 60.9% · 95: 60.5% |
| Near misses at the live threshold | flat across sizes: "shee" + word 33–41%, onset + "eila" 27–35%, synthesized she- words 29–34%, "sh" 1–2% |
| Efficient size (validation rule) | **48** (validation F1 0.740 ± 0.003, the best mean) |

Slides: `reports/E2_network_size_slides.pptx`, five slides for about one minute of a larger
talk (speaker notes contain the script), plus a backup slide for questions (confusion matrix
of the deployed model: how F1 and live detection are measured). They are built by `tools/make_e2_slides.py`
(python-pptx, installed in `.venv`).

Conclusions:
- Memory, not time, limits the dense design.
- Quality saturates at about 16 neurons, so the memory wall costs no measurable quality.
- More neurons do not reduce near misses: that needs a time-aware design (E3/E4).

## Deployed model: hard negatives, 95 hidden (since 2026-10-04)

`deploy/model.npz` = `results/sheila_hardneg/models/r2_aug30_h95_seed2.npz`: dense
768→95→2 LIF, current encoding, 12 steps, int16 Q10, release recipe plus Sheila hard
negatives (`augment.py`, fraction 0.3), seed 2 selected by validation F1 (with the
validation-split variants). Deployed after the user's live test (JOURNAL entry 11).

| Quantity | Value | Source |
|---|---|---|
| Validation F1 with variants, seeds 0 / 1 / 2 | 0.7174 / 0.7297 / **0.7440** | `results/sheila_hardneg/evaluation.json` |
| Test precision / recall / F1 (seed 2) | 81.22% / 75.47% / 0.7824 | same |
| Test F1, 3 seeds | 0.781 ± 0.003 | same |
| Negative-clip FPR (seed 2) | 0.343% | same |
| Thresholds: window / stream | 1594 / 403 | `deploy/model.npz` |
| Live stream, single: Sheila / other words | 73.6% / 1.47% | `results/sheila_hardneg/confusables.json` |
| Live stream, 2 of 3: Sheila / other words | 60.9% / 0.33% | same |
| Hard-negative windows passing the live threshold: sh / shee / sheet / "shee"+word / onset+"eila" | 1.2% / 7.6% / 8.1% / 33.1% / 35.2% | same |
| Synthesized words, 2 of 3: Sheila / sh sounds / she- words / other sh- words | 63% / 6% / 31% / 27% | same |
| Background-noise windows detected | 0 / 398 | `results/sheila_hardneg/background_noise.json` |
| Max cycles, RV32IM (board, bit-exact) | 8,877,865 (88.78 ms) | `results/sheila_hardneg/board_base.json`, re-checked 2026-10-04 |
| Max cycles, `kdot` (board, bit-exact) | 395,945 (3.96 ms) | `results/sheila_hardneg/board_kdot.json`, re-checked 2026-10-04 |
| Model bytes | 146,680 of 147,456 | `results/sheila_hardneg/firmware_build.json` |
| PC ↔ board round trip, RV32IM / `kdot` | 90.1 ms / 5.0 ms | `check_board.py`, 2026-10-04 |

## First Sheila release (2026-09-30 to 2026-10-04; E0's baseline)

`results/sheila/models/current_seed1.npz` (git 3dbfdb6): dense 768→64→2 LIF,
current encoding, 12 steps, int16 Q10, release recipe, selected by validation F1.

| Quantity | Value | Source |
|---|---|---|
| Positives train / validation / test | 1,606 / 204 / 212 | `results/sheila/dataset.json` |
| Validation F1, seeds 0 / 1 / 2 | 0.8955 / **0.9054** / 0.8900 | `results/sheila/evaluation.json` |
| Test precision / recall / F1 (seed 1) | 85.64% / 81.60% / 0.8357 | same |
| Test F1, 3 seeds | 0.835 ± 0.020 | same |
| Negative-clip FPR (seed 1) | 0.269% | same |
| Thresholds: window / stream | 3002 / 1080 | `results/sheila/stream_seed1.json` |
| Live stream, single: Sheila / other words | 74.1% / 1.07% | `results/sheila/confusables.json` |
| Live stream, 2 of 3: Sheila / other words | 68.9% / 0.40% | same |
| Background-noise windows detected | 2 / 398 | `results/sheila/background_noise.json` |
| Max cycles, RV32IM (board, 40 vectors ×3, bit-exact) | 5,980,921 (59.81 ms) | `results/sheila/board_base.json` |
| Max cycles, `kdot` (board, bit-exact) | 266,793 (2.67 ms) | `results/sheila/board_kdot.json` |
| Model bytes | 98,824 of 147,456 | `results/sheila/firmware_build.json` |
| PC ↔ board round trip (`kdot`) | 3.9 ms | `results/sheila/board_kdot.json` |

Known quality problem (first TTS probe, 2 voices, 2 of 3): the model keys on the
**"shee-" onset**. Words starting like "Sheila" are accepted often (sheena 71%, shayla
68%, shield 64%, cheetah 60%, sheep 55%, she 54%, sheet 48%, shelter 45%; "Sheila" 68%),
words sharing only the ending rarely (leila 3%, delilah 0%), unrelated words never (0%).

For reference, the previous "yes" release: test F1 0.8761 (seeds 0.867 ± 0.022), live
2 of 3 67.3% / 0.13%, max cycles 5,981,521 / 267,393 (`results/evaluation.json`,
`results/confusables.json`).

## Environment notes
- Training runs on the CPU (AMD GPU, no CUDA): ~10 s per run for the release recipe.
- Firmware: `research/tools/build_firmware.py <model.npz> --out build/<name>`.
- Board: `run_board.ps1` (after every power cycle), then `check_board.py 10.43.0.1`.
- The Sheila TTS probe is in `build/tts_probe_sheila/` (`make_tts_probe.ps1 -Keyword sheila`).
- Benchmark of one model:
  - quality: `research/tools/eval_model.py name=<model.npz> --out <json>` (~12 s);
  - board: `research/tools/bench_board.py <model.npz> --out <json>` (~75 s, both variants,
    leaves experimental firmware running: restore with `run_board.ps1`).
- Whole size sweep: `research/tools/size_sweep.py --hidden …`, then `plot_size_sweep.py`.

## Open issues / decisions
- **Resolved 2026-10-04:** the hard-negative candidate (JOURNAL entry 10) is deployed after
  the user's live test (entry 11: a clearly spoken "Sheila" was detected almost every time).
  Its price: live 2 of 3 Sheila 68.9% → 60.9%, clip F1 0.836 → 0.782; its gain: "sh"
  windows 55.8% → 1.2%, synthesized sh/she-/sh- words 42/58/49% → 6/31/27%.
- E1 and E5 must use Sheila near-miss words (PLAN.md note): mainly "she-" words; still
  accepted by the deployed model: shield, sheena, shayla, "shee" + other word.
- First Sheila model: test F1 below validation F1 (0.836 vs 0.905 for seed 1). Part of it
  is selection on validation, part sampling: with 212 test positives, one standard error of
  recall alone is about 2.7 percentage points. (For the hard-negative model, validation F1
  includes the hard variants and is lower than test F1: 0.744 vs 0.782.)

- E2 → later experiments:
  - E3 uses sizes **16 and 48** (not 16 and 64: 64 was a dip);
  - E3/E4 must beat the dense near-miss rates;
  - E6 (int8) should measure memory saved at equal quality, since bigger dense networks
    are not expected to help.
- E2's 64-neuron dip (live 54% vs 61–64% at 48/80/95, all three seeds) is unexplained; with
  three seeds it may be chance.

## Next step
Run **E1** (two recording sessions on different days, Sheila near-miss words; needs the
user, about 15 min each), then E3 with sizes 16 and 48. The short E0 report can be written
in any session.
