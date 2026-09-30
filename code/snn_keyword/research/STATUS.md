# Research status

Update this file at the end of every session: tick the experiment, add its key numbers,
note open issues and the next step.

## Progress

| Exp | Title | Status | Report |
|---|---|---|---|
| E0 | Setup, baseline reproduction, shared tools | **next** (partly done, see below) | — |
| E1 | Evaluation sets for near-miss words (user recordings) | to do | — |
| E2 | Network size sweep (dense, current encoding) | to do | — |
| E3 | Input encoding comparison (current / rate / temporal) | to do | — |
| E4 | Temporal-convolution SNN topology | to do | — |
| E5 | Hard-negative training data | to do | — |
| E6 | Weight precision int16 vs int8 | to do | — |
| E7 | Decision rule and end-to-end latency | to do | — |
| E8 | Synthesis and conclusion | to do | — |

## Keyword change to "Sheila" (2026-09-30, before E0)

The keyword is now **"Sheila"** (Speech Commands word `sheila`); see `JOURNAL.md` entry 9
and the "Keyword change" note at the top of PLAN.md. `deploy/` holds the Sheila release;
its evidence is in `code/snn_keyword/results/sheila/`. The "yes" release stays in git
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

Still to do in E0: `bench_board.py` (build → upload → restart the server with that
firmware → `check_board.py`; needs the `start_board.sh <variant> <fw.bin>` extension),
`eval_model.py` (one quality JSON per model; can reuse `confusables.py`), and the E0 report.

## Starting point: Sheila release (2026-09-30)

`deploy/model.npz` = `results/sheila/models/current_seed1.npz`: dense 768→64→2 LIF,
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

## Open issues / decisions
- **Hard-negative candidate (JOURNAL entry 10, 2026-10-01)**, awaiting the user's live test
  and deploy decision: 95 hidden, trained with Sheila hard negatives
  (`augment.py`, fraction 0.3), staged in `build/sheila_hardneg/`, evidence in
  `results/sheila_hardneg/`. Test: "sh" windows passing the live threshold 55.8% → 1.2%,
  synthesized sh/she-/sh- words 42/58/49% → 6/31/27%, background noise 2 → 0 of 398, but
  live 2 of 3 Sheila 68.9% → 60.9% and clip F1 0.836 → 0.782. Board: bit-exact, max
  8,877,865 / 395,945 cycles, model 146,680 of 147,456 bytes.
- E1 and E5 must use Sheila near-miss words (PLAN.md note): mainly "she-" words; still
  accepted by the candidate: shield, sheena, shayla, "shee" + other word.
- The user found by live test that the entry-9 release fires on a plain "sh".
- Test F1 is below validation F1 (0.836 vs 0.905 for seed 1). Part of it is selection on
  validation, part sampling: with 212 test positives, one standard error of recall alone
  is about 2.7 percentage points.

## Next step
Run **E0** (remaining parts above), then E1 with Sheila near-miss words.
