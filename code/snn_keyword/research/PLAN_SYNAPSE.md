# Plan, part 2: energy-aware network size and synapse optimisation (S0–S5)

Written 2026-10-17 for Pedro's report section ("network size", extended with synapse optimisation),
about one week before the report deadline. It replaces the not-yet-started experiments E1 and E3–E8
of [PLAN.md](PLAN.md); those are deferred until after the report. Progress is tracked in
[STATUS.md](STATUS.md), and every energy number lives in [ENERGY.md](ENERGY.md).

## Why
- **The teacher's advice:** optimise the **synapse** (the weighted connection between two neurons),
  instead of only varying the number of neurons. Bits per weight is one property of a synapse.
- **The application:** an **always-on smart-home wake-word device** (Alexa-like). It must use the
  least energy, keep good accuracy, finish each inference within 250 ms (one window per 250 ms),
  and fit in the 144 KiB model memory.
- **What E2 is missing:** E2 (network size) has no energy numbers. It is completed with energy and
  extended on the same dense 768→H→2 network, which is the controlled testbed.
- **Other topologies are cited, not re-swept.** Mateo's time-aware network needs a GPU and about
  60 GB of evaluation data, and its "sheila" model never ran on a board. His "yes" ablations:
  removing synaptic delays cost 21 points of live recall; LIF instead of ALIF also cost 21 points.

## Decisions (Pedro, 2026-10-17)
- **Energy source:** Vivado `report_power`. The report gives the raw values *and* the
  calculation. Vivado's estimates were validated earlier against a USB power meter (ENERGY.md §3).
- **Precision:** our own bits-per-weight sweep, compared with Mark's results on the 128-input network
  (branch `Quantization`). His QAT test F1 was 87.8 / 87.9 / 88.4 / 87.0 / 83.3 / 33.0 at
  8 / 6 / 5 / 4 / 3 / 2 bits, and he chose int4.
- **Quality bar, fixed before any result:**
  - A design qualifies if its mean validation F1 ≥ the best mean − sd(best).
  - The lowest-energy qualifying design is recommended.
  - Sensitivity check: the same rule with the pooled sd.
- **No hardware (RTL) changes.** kdot stays int16-dense. The rebuilt bitstreams are used only for
  power analysis.
- **Report:** a LaTeX section for Overleaf, using template `pedronavarro17/report-template`
  (`mlmodern`, A4, 2.5 cm margins, booktabs).

## Synapse optimisation in this study
- **What a synapse costs:**
  - One synapse = one weight. There are 768·H + 2·H of them: 99.5 % of the model memory.
  - On plain RV32IM, each costs one BRAM read and one multiply-add per inference, about 99 % of
    the cycles.
- **Levers:**
  1. **Precision:** bits per synapse.
  2. **Count:** pruning to a fixed fan-in K per neuron.
  3. **Synaptic operations actually executed.**
- **Facts found while planning (checked):**
  - **The release weights use only 10 of their 16 bits:** max |w1| = 274–497 in Q10 over the 21
    E2 models (366 in the deployed model).
  - **With kdot, synapses are only about 14 % of the cycles;** the 12-step LIF loop is about 72 %.
    Synapse levers can therefore save at most about 14 % there.
  - **The LIF loop has an exact shortcut.** Under current encoding the drive d is constant, so the
    12-step spike count is a monotone staircase of d: 0 spikes for d ≤ 160, 12 spikes for d ≥ 1024.
    The thresholds for ≥ k spikes are 161, 233, 310, 388, 489, 547, 660, 728, 818, 906, 988 and
    1024. A table of these 12 thresholds is bit-exact. It also supports a discussion point: with
    current encoding, this "SNN" behaves like a quantised ANN with 13 activation levels.

## Hypotheses (written before the experiments)
- **H1:** Within one fabric, energy per inference ∝ cycles. Synaptic multiply-adds are ≥ 95 % of
  the RV32IM cycles but ≤ about 15 % with kdot.
- **H2:**
  - QAT keeps validation F1 within the seed spread down to 4 bits and collapses at 2 bits.
  - Memory falls 2.5× (vs 10-bit packed) and 4× (vs int16).
  - RV32IM cycles change by less than 10 %, because the loop is limited by instruction fetch.
- **H3:**
  - At an equal number of non-zero synapses, large-sparse ≥ small-dense.
  - RV32IM cycles ∝ non-zero synapses.
  - kdot gains nothing.
- **H4:** The exact LIF lookup cuts kdot cycles by ≥ 2× and RV32IM cycles by < 5 %.
- **H5:**
  - With today's busy-polling firmware, the window energy does not depend on the network.
  - Savings appear only with clock gating or with the clock scaled to the deadline.
  - The lowest-energy qualifying design is a small dense network on kdot with the LIF lookup.

## Experiments
- **S0 — Energy method (done 2026-10-17).**
  - Vivado 2026.1 power reports of both fabrics: base = RTL of `714db5c`, kdot = `HEAD`.
  - Parser and energy model.
  - Three operating scenarios: (a) busy-polling, (b) clock-gated idle, (c) clock scaled to the
    deadline.
  - ENERGY.md.
- **S1 — Size with energy (done 2026-10-17).** The 21 E2 models get energy columns from their
  board cycles. No training and no board time.
- **S2 — Synapse precision:**
  - **Quantizer:** Mark's power-of-two rule in Q10, fixed at the first QAT epoch, with clamping
    and STE.
  - **Runs:** w1 at 8/6/4/3/2 bits × H ∈ {16, 48} × seeds 0–2 (30 trainings). The 16-bit point is
    the E2 models.
  - **Storage:** the model file keeps the full-scale int16 weights, plus `w1_bits` and `w1_shift`.
  - **Controls:** a post-training rounding control (no training), and a per-seed comparison paired
    with E2.
  - **Firmware:** a packed RV32IM loop (`USE_PACKED`) for 16 (control) / 8 / 4 / 2 bits, measured
    on the board at H = 48.
  - **Comparison with Mark:** ΔF1 relative to each network's own 8-bit result.
- **S3 — Synapse count:**
  - **Pruning:** top-K magnitude pruning per neuron, with a cubic schedule over epochs 3→20, then
    a frozen mask (5 recovery epochs + 10 QAT epochs).
  - **Runs:** K ∈ {16, 32, 64, 128, 256, 512} × H ∈ {48, 95} × 3 seeds. This matches E2's dense
    synapse counts: H48·K128 = H8, H48·K256 = H16, H48·K512 = H32, H95·K64 ≈ H8.
  - **Firmware:** a sparse RV32IM loop (`USE_SPARSE`), one word per non-zero, `(w<<16)|idx`.
  - **Combined:** bits × fan-in runs at H = 48.
  - **kdot check:** kdot runs on pruned models show that kdot cannot skip zeros.
- **S4 — Exact LIF lookup:**
  - Firmware `USE_LIF_TABLE`, on dense H ∈ {16, 48, 95} × 3 seeds, on both fabrics.
  - It also gives the cycle split: synapse layer / LIF loop / fixed part.
- **S5 — Synthesis:**
  - all designs in one table, Pareto front of quality vs energy;
  - the pre-registered choice, bit-exact on the board;
  - figures and tables;
  - the research record `reports/S_energy_synapse.md` and the LaTeX section `reports/latex/size_synapse/`.

## S6 — Sparse kdot (added 2026-10-17, written before building)
- **Why.** S3 and S5 showed that pruning only pays off on plain RV32IM: kdot streams every weight,
  zeros included.
- **Design.** Two new instructions in the kdot unit:
  - `kxload` copies the input once into a local buffer;
  - `ksdot` streams only the kept synapses (S3 format, `(int16 w << 16) | index`), one per cycle.
- **ABI** 0x0002_0003; experimental bitstream only.
- **Hypothesis H6.** For the same model, only the dot-product time changes:
  C_ksdot − C_kdot_lut ≈ −H·(576 − K) + ~200 cycles (± H·15).

  | Model | Predicted cycles | Predicted energy |
  |---|---|---|
  | H48 K64 | ≈ 34.4k | ≈ 98 µJ (−41 % vs the deployed 58.7k) |
  | H48 K128 | ≈ 37.4k | |
  | H48 K256 | ≈ 43.6k | |
  | H95 K64 | ≈ 66k | |

  Floor ≈ 31k (non-dot-product work).
- **The selection rule is unchanged.** Expected: sparse kdot moves the Pareto front and the
  pooled-sd sensitivity choice; the strict choice stays the deployed model unless a pruned 8-bit
  model reaches 0.7405.
- **Models:** the S3 models, plus new `h48_b8_k{64,128,256}` (the deployed recipe + pruning).

## Rules (as in PLAN.md part B)
- 3 seeds; validation for choices, test only for reporting.
- Cycles only from bit-exact board runs.
- `deploy/` untouched; experiment firmware in `build/`.
- Unique model names (`h48_b4_seed0`, `h95_k64_seed1`) so that `build/bench/` never overwrites E2.
- New options are off by default, so `build_firmware.py deploy/model.npz --compare deploy` stays
  byte-identical and E2 stays reproducible.
- A local commit per experiment; no push without asking; `run_board.ps1` at the end of each session.

## Figures (journal style, `tools/plot_style.py`)
Computer Modern to match `mlmodern`, 9 pt, sizes 16 × 6 cm / 7.8 × 5.5 cm, white background, ticks
inward, no titles, error bars = ± 1 sample sd over 3 seeds, grayscale-safe colours and markers,
PDF + PNG.

| Figure | Content |
|---|---|
| F1 | Energy and quality vs H |
| F2 | Average power under (a)/(b)/(c) |
| F3 | F1 and memory vs bits |
| F4 | F1 vs non-zero synapses, small-dense vs large-sparse |
| F5 | Cycle/energy breakdown with and without the LIF lookup |
| F6 | Pareto front, choice marked |

| Table | Content |
|---|---|
| T1 | Raw Vivado power |
| T2 | Worked calculation |
| T3 | Design summary |
| T4 | Choice vs deployed model |

## Schedule (if late, cut in this order)
1. combined bits × fan-in runs;
2. pruning at H = 95;
3. the toggle-rate sensitivity;
4. the packed firmware (keep accuracy and memory);
5. kdot runs on pruned models.

Stretch goals: SAIF activity from xsim, a frequency-scaling demo with the USB meter, a BRAM-size
what-if build, and an ASIC operation-count projection.
