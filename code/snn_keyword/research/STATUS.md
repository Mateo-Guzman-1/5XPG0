# Research status

Update this file at the end of every session: tick the experiment, add its key numbers,
note open issues and the next step.

## Progress

Since 2026-10-17 the programme follows **part 2, [PLAN_SYNAPSE.md](PLAN_SYNAPSE.md)** (energy-aware
size and synapse study for the report). All energy numbers are in [ENERGY.md](ENERGY.md).

| Exp | Title | Status | Results |
|---|---|---|---|
| S0 | Energy method (Vivado power of both fabrics, energy model) | **done** 2026-10-17 | `results/S0/`, [ENERGY.md](ENERGY.md) |
| S1 | Network size with energy (E2 re-analysis) | **done** 2026-10-17 | `results/S1/`, [ENERGY.md](ENERGY.md) §4 |
| S2 | Synapse precision (bits per weight, vs Mark) | **done** 2026-10-17 | `results/S2/`, [S_energy_synapse.md](reports/S_energy_synapse.md) |
| S3 | Synapse count (fixed fan-in pruning) | **done** 2026-10-17 | `results/S3/`, same |
| S4 | Exact LIF lookup (neuron side) | **done** 2026-10-17 | `results/S4/`, same |
| S5 | Synthesis, choice, report section (LaTeX) | **done** 2026-10-17 (draft for review) | `results/S5/`, `reports/latex/size_synapse/` |
| S6 | Sparse kdot (kxload/ksdot hardware) vs the deployed dense-kdot model | **done** 2026-10-17 | `results/S6/`, [S_energy_synapse.md](reports/S_energy_synapse.md) §S6 |
| R1 | The S designs against the group requirements, with Mateo's test harness ([PLAN_R1.md](PLAN_R1.md)) | **done** 2026-10-19 | `results/R1/`, [R1_requirements.md](reports/R1_requirements.md), `reports/latex/requirements_eval/` |

## R1 result: group requirements (2026-10-19)
**Requirements (Pedro):**
- ≥ 85 % live "Sheila" detected;
- ≤ 2 false accepts per hour;
- ≤ 0.2 % other words accepted;
- ≤ 500 ms median from word end to decision;
- ≤ 200 ms worst compute per request;
- ≤ 144 KiB memory;
- ≤ 1 W average PL power.

**How it was measured:**
- Mateo's harness (`tools/mateo_eval/`, from `origin/ALIF-Layer1` a856a03).
- The threshold is chosen on validation with his rule; then the test is run once.
- 13 configs × 3 seeds, the same models as in S2–S6 (retraining is bit-identical).

**Results:**
- **Detection:** no design meets it. The best mean is 42.9 ± 6.9 % (H48 8-bit). FA/h is 1.6–4.1 on test (mean 2.8); 86 % of the false accepts are LibriSpeech running speech. Other words are ≤ 0.07 %.
- **Cost:** every design meets compute, memory, latency (134–258 ms) and power (0.37 W busy-polling).
- **Choice (two-champion rule):** the pruned H48 K64 on ksdot + table, 36.3 % live, 96.5 µJ, 12.4 KiB. The dense champion is H48 8-bit kdot + table (deployed), 42.9 %, 165 µJ, 72.4 KiB. The 6.6-point difference is within the seed spread.
- **Comparison:** Mateo's ALIF + GRU cascade meets all detection requirements on the same tests (86.8 %, 1.42 FA/h, 0.07 %).
- **Next:** retrain on Mateo's full training data (MSWC, Piper TTS, LibriSpeech negatives, rooms, microphones). This is the deferred follow-up, Pedro's choice.

Part 1 ([PLAN.md](PLAN.md)):

| Exp | Title | Status | Report |
|---|---|---|---|
| E0 | Setup, baseline reproduction, shared tools | tools done (built during E2); report to do | — |
| E1 | Evaluation sets for near-miss words (user recordings) | deferred (after the report) | — |
| E2 | Network size sweep (dense, current encoding) | **done** 2026-10-04 | [E2_size_sweep.md](reports/E2_size_sweep.md), background: [E2_network_size_explained.md](reports/E2_network_size_explained.md) |
| E3 | Input encoding comparison (current / rate / temporal) | deferred (Damien measured current vs rate for "yes") | — |
| E4 | Temporal-convolution SNN topology | deferred | — |
| E5 | Hard-negative training data | deferred | — |
| E6 | Weight precision int16 vs int8 | replaced by S2 | — |
| E7 | Decision rule and end-to-end latency | deferred | — |
| E8 | Synthesis and conclusion | replaced by S5 | — |

## S0–S1 results: energy (2026-10-17)

The source is Vivado 2026.1 `report_power` on the routed release RTL of both fabrics (vectorless,
25 °C, confidence "Medium"). Raw values, formulas and the worked example are in ENERGY.md.

| Quantity | RV32IM | RV32IM + kdot |
|---|---|---|
| PL dynamic power (dynamic − PS7) | 0.278 W | 0.282 W |
| of which block RAM (256 KiB) | 0.251 W | 0.251 W |
| PL static power (PL rails) | 0.085 W | 0.085 W |
| Energy per inference, deployed model (95 neurons) | 24.7 mJ | 1.10 mJ |
| Energy per neuron (linear fit, H ≥ 32) | 259 µJ | 11.4 µJ |
| Average PL power, busy-polling (deployed firmware), any size | 363 mW | 367 mW |
| Average PL power, clock-gated idle, H = 8 → 95 | 101 → 189 mW | 93 → 97 mW |
| Average PL power, clock scaled to the deadline, H = 8 → 95 | 94 → 194 mW | 85 → 90 mW |

- **The BRAM is ~90 % of the PL dynamic power,** read on both ports in every cycle.
- **With today's busy-polling firmware the network size does not change the energy.**
- **Savings need an idle mode.** With kdot, even then the network adds only 0.2–4.4 mW to a 93 mW
  floor (static power plus the AXI interface).
- **Robust to Vivado's activity assumption:** toggle rates of 6.25 % and 25 % change P_PL,dyn by
  ≤ 2 mW.
- **Meter validation (ENERGY.md §3):** on Alex's dual-core design `8ed0b32`, Vivado's PL estimate
  (0.773 W) is 7 % below the measured board increase (0.833 W).

## S2–S5 results: synapses and the design choice (2026-10-17)

Full record: [reports/S_energy_synapse.md](reports/S_energy_synapse.md). Every cycle count is from a
bit-exact board run (120/120).

| Lever | Quality | Cost |
|---|---|---|
| Bits per weight (QAT, H 16/48) | Unchanged within the seed spread down to 4 bits; 3 bits slightly lower; 2 bits fail; rounding only is worse | Memory 4× smaller at 4 bits; on RV32IM the packed loop costs *more* energy (14.1 / 18.3 / 28.1 mJ at 8 / 4 / 2 bits vs 12.5 mJ) |
| Fixed fan-in pruning (H 48/95) | Large-sparse > small-dense at equal synapses (6,144: 0.727 vs 0.643); K = 64 keeps 0.727 | RV32IM energy ∝ non-zeros (7.4× less at K = 64); kdot gains nothing |
| Exact LIF table | Bit-exact | kdot 2.7–3.4× fewer cycles; RV32IM 2–3 % |

**Choice** (pre-registered rule; same with the pooled sd): 48 neurons, 8-bit weights,
kdot + LIF table.
- **Cost:** 165 µJ and 0.59 ms per window, 72.4 KiB.
- **Quality:** test F1 0.789 ± 0.010, live 65.6 %.
- **Against the deployed model:** 6.7× less energy per inference, half the memory, equal quality.
- **Average PL power barely changes:** 93.6 vs 97.3 mW clock-gated, because of the static floor.
- **Deployed 2026-10-17** at Pedro's request: seed 1 (best validation F1). Evidence in
  `results/sheila_h48_b8/`, JOURNAL entry 14. Bit-exact on the board from `deploy/`, both fabrics.

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

## S6 result: sparse kdot (2026-10-17)
- **Hardware:**
  - two new PCPI instructions: `kxload` copies the input into a 1 KiB buffer; `ksdot` streams the
    kept synapses at one per cycle;
  - experimental bitstream `results/S6/keyword_ksdot.bit` (ABI 0x0002_0003);
  - xsim unit test 48/48; timing +0.545 ns; +362 LUTs; same estimated power.
- **Gates passed:** the deployed firmware gives identical cycles on the new bitstream, and all 48
  board runs are bit-exact.
- **Prediction H6 confirmed:** within 170 cycles for all 16 configurations.
- **Main result:** H48 pruned to K = 64 takes 34,221 cycles and 96.5 µJ (−42 % vs the deployed
  58,667 cycles and 165 µJ), with 12.4 KiB memory. The cost is quality: val F1 0.727 vs 0.745,
  live 57.5 vs 65.6 %. The floor is ≈ 32k cycles.
- **Selection rule (unchanged):** it still chooses the deployed model. The pooled-sd sensitivity
  check picks H48 K64 sparse, by a 0.00006 margin.
- **Not deployed:** experimental bitstream only.

## Previous release: hard negatives, 95 hidden (2026-10-04 to 2026-10-17)

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
The experiments of part 2 and R1 are complete. Open:
- Pedro reviews the R1 report `reports/latex/requirements_eval/main.pdf`. Its figures and tables can be `\input`
  into the group report; it needs `\input{figures/r1_style}` in the preamble.
- Follow-up: retrain the two champions on Mateo's full training data (`origin/ALIF-Layer1`
  `prepare_multicorpus.py`, `train_dense_online.py`; needs a CPU port of its CUDA-only augmenter).
- Pedro reviews the LaTeX section `reports/latex/size_synapse/section.tex`: wording, which figures to
  keep, and the meter details for the method text.
- Live microphone test of the newly deployed model (`pc_keyword_demo.py 10.43.0.1`).
- Optional: stretch items in PLAN_SYNAPSE.md (BRAM-size what-if, clock-scaling demo with the USB meter).
- Part 1's E1 (recordings) is deferred until after the report.
