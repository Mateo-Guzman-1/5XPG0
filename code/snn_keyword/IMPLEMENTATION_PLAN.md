# Implementation plan: streaming delay-SNN keyword detector

This plan implements the route chosen in
[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) §6. The target model is a
streaming two-layer adaptive-LIF SNN with learnable delays. It is trained
on 35 classes with distillation and user- and microphone-agnostic
augmentation, quantized to int8/int4, and run first in RV32 software, then
on an event-driven neuron engine.

**Principles carried over from the release**

- Checkpoints and thresholds are selected on validation data only. The test
  split and the synthesized-word probe are used once per candidate, for
  reporting.
- One integer reference model (Python) defines the deployed arithmetic.
  Native C, RTL simulation and the board must match it bit-exactly.
- Every phase ends with results in `results/*.json` and an entry in
  `JOURNAL.md` covering motivation, change and outcome, including negative
  results.
- The current release (`deploy/`, dense model plus `kdot`) stays usable
  until the new detector beats it on the acceptance criteria.

**Out of scope for this plan**

- On-board audio capture and front end (the PYNQ-Z2 codec).
- Any training or tuning on the developer's own voice.
- The per-user enrollment mode (documented in OPTIMIZATION_REPORT.md §3.7).

---

## Acceptance targets

The new detector replaces the release only if, on the held-out measures
produced by `confusables.py` (extended in Phase 0), it meets all of these:

| Metric | Release today | Target |
|---|---|---|
| Live "yes" detected (test clips in noise, 250 ms hop, confirmed decision) | 67.3% | **≥ 85%** |
| Live other words accepted | 0.13% | **≤ 0.2%** |
| Synthesized /ts/ words (yeets/yets/yetz, pizza, eats) detected | 11–53% | **≤ 10%** |
| Synthesized "yes" detected | 77% | **≥ 85%** |
| Drop under held-out microphone/room conditions (Phase 0) | not measured | **≤ 3 points** recall |
| Drop on a held-out *corpus* (leave-one-source-out, Phase 0) | not measured | **≤ 5 points** recall at equal false-accept rate |
| False accepts per hour, continuous-stream proxy (Phase 0) | not measured | report; target set after Phase 0 |
| Compute per 250 ms hop on the board | 2.7 ms | **≤ 25 ms** (10% duty) |
| Bit-exact oracle / C / RTL / board | yes | yes |

The targets are ambitious but grounded in the SNN results in the report
(§2). Phase gates allow the plan to stop early with a documented result if
they are not reachable.

---

## Phase 0: Evaluation harness (≈ 2 days)

- **Goal.** Measure user and microphone agnosticism, and continuous-stream
  behaviour, before changing the model.
- **Work.**
  - `robust_eval.py` (or an extension of `confusables.py`) adds:
    1. **Device-shift test.** Test clips passed through held-out microphone
       filters: random parametric EQ, band limits and gain drawn from a
       fixed seed and disjoint from the training augmentation. Also
       held-out room impulse responses from the RIR database [17].
    2. **Continuous-stream proxy.** An hour-scale stream of concatenated
       test-split non-"yes" words and background noise, with test "yes"
       inserted at known times. It reports false accepts per hour and
       latency from the end of the word to detection.
    3. **Streaming interface.** The harness accepts both window models (the
       current release) and frame-streaming models (Phase 3 onwards).
    4. **Leave-one-source-out test.** Recall and false accepts on "yes"
       and non-"yes" clips from a corpus the model was not trained on
       (Common Voice single words, MSWC). This measures dependence on one
       recording setting. For the current release, every added corpus is
       held out.
  - Re-score the release and trial models to create the baseline in
    `results/robust_baseline.json`.
- **Done when** the baseline numbers exist for every metric in the targets
  table.

## Phase 1: Data and front end (≈ 6 days)

- **Goal.** A training data path that is user- and microphone-agnostic
  and not tied to one recording setting.
- **Work.**
  0. **Diverse real audio from several corpora** (report §3.3):
     - Download and index Common Voice Single Word Target Segment ("yes",
       "no", digits, "hey") [37][38], and MSWC English [39] for "yes" in
       running speech and thousands of real near-miss words.
     - Use LibriSpeech [40] for continuous negatives and the false-accept
       stream.
     - `prepare_multicorpus.py` builds one manifest with a corpus tag,
       speaker id, split and licence per clip.
     - Balance sampling by corpus. Filter MSWC clips by alignment
       confidence and edge energy.
     - Keep splits speaker-disjoint within each corpus, and leave one
       corpus per Phase 0 test out of training.
  1. **On-the-fly waveform augmentation** (PyTorch, GPU), in
     `augment_online.py`:
     - microphone simulation: random EQ, band-limiting, gain and soft
       clipping [22];
     - rooms: RIR convolution [17];
     - noise and music from MUSAN [18] and the Speech Commands background
       noise;
     - speed perturbation and VTLP [19];
     - SpecAugment with short time masks [16].
  2. **Multi-speaker TTS hard negatives** [14][15], in `make_tts_negatives.py`:
     - Grapheme edits of "yes" (insert, delete, substitute: "yeets",
       "yesh", "mes", "tes", "yeah", "guess", …) plus a fixed list of
       real /s/-, /ts/- and /tʃ/-final words.
     - Synthesized with an open multi-speaker TTS (for example a Piper
       multi-speaker English model; check each voice's licence).
     - Voices split into train, validation and test sets. The current
       two Windows voices stay as an additional held-out probe.
  3. **PCEN front end** [20] as an option in `features.py`, integer-friendly
     (fixed-point smoothing and power). Compare with log-mel on validation.
  4. **Frame-level features** for streaming: one 24-band vector per 10 ms
     frame (the existing FFT settings), instead of 32 pooled bins.
- **Gate.** Retrain the *current* dense model with the new augmentation and
  front end, as a pure data ablation. Record the Phase 0 metrics. This
  separates data gains from architecture gains.

## Phase 2: Teacher (≈ 2 days)

- **Goal.** A strong non-spiking teacher for distillation.
- **Work.**
  - Train BC-ResNet-8 [8] on the 35 GSC classes plus "silence", with the
    Phase 1 augmentation, using the published reference implementation as
    a starting point.
  - Export soft labels for training clips and augmented streams (logits at
    temperature T), in `teacher/`.
- **Done when** the teacher reaches ≥ 95% on the 35-class validation split.
  It is used for training only and never deployed.

## Phase 3: Streaming SNN student (float) (≈ 7 days)

- **Goal.** The architecture of report §6, in floating point.
- **Work.**
  - `snn_stream.py` defines the model:
    - input: frame features;
    - layer 1: 128 ALIF neurons [3], feedforward, with DCLS learnable
      delays up to 32 frames [4];
    - layer 2: 128 ALIF neurons, recurrent;
    - readout: leaky integrator per class.
  - Train with surrogate gradients through time [2]: learnable τ_m and
    τ_adapt [3][36], and a spike-count regularizer [34].
  - Curriculum:
    1. 35-class pretraining with a KD loss [13][7];
    2. streaming fine-tuning for "yes" against unknown against silence, on
       long augmented streams with an end-of-word label [11];
    3. decision threshold and hold-off chosen on validation.
- **Ablations** on validation, recorded in `results/stream_ablation.json`:
  - no delays;
  - no recurrence;
  - LIF instead of ALIF;
  - no KD;
  - no TTS negatives;
  - no PCEN;
  - 64 against 128 neurons.
- **Gate.** If the best float model does not clearly beat the release on
  the Phase 0 validation proxies, stop. Record the result and reconsider
  (for example a temporal-conv SNN front layer) before investing in
  hardware.

## Phase 4: Quantization and integer oracle (≈ 4 days)

- **Goal.** A deployable integer model with its exact reference.
- **Work.**
  - QAT to int8 weights (per-channel scale folded into thresholds). Neuron
    state is int16 with saturating arithmetic; accumulators are int32.
    Delays are integer frame counts.
  - Try int4 next, and keep it if validation F1 drops by at most 1 point
    [23].
  - `integer_forward_stream()` in `model.py`: the frame-by-frame integer
    oracle (exact shifts for the leak and adaptation, as in the current LIF
    arithmetic).
  - Extend `export_model.py` with packed int8/int4 weights, delay tables
    and neuron parameters.
  - A native C kernel (`firmware/stream_infer.c`), and `sim/native.c`
    extended to check that it is bit-exact with the oracle on the full test
    set.
- **Done when** native C matches the oracle exactly on all test streams,
  and quantized validation metrics are within 1 point of float.

## Phase 5: Firmware and protocol, RV32 software path (≈ 4 days)

- **Goal.** Real time on the existing SoC, without new RTL beyond a bus
  fix.
- **Work.**
  1. **Protocol and ABI v3.** A request carries the 25 new frames of each
     250 ms hop (600 bytes at 24 bands). Neuron state persists on the core
     between requests. A reset command clears it. The mailbox publishes
     the model kind (window or stream), frame size and threshold.
     `board_server.py`, `jtag_server.py` and `pc_keyword_demo.py` support
     both ABI v2 (the current window models) and v3.
  2. Layer 1 uses `kdot` (a 24-input dot product per neuron per frame).
     Layers 2 and the readout are event-driven in C: iterate over spiking
     neurons and add weight rows.
  3. **Bus fix (report E2).** Reduce `READ_WAIT` in `spike_soc.v` to the
     true BRAM latency. Re-verify with `tb_axi` and `tb_led`, the full RTL
     harness and Vivado timing.
  4. Extend the RTL harness (`sim/soc_main.cpp`) with streamed requests,
     state reset, and bit-exactness against the oracle on 40 streams. Then
     run the board test over JTAG (`jtag/`).
- **Done when** the design is bit-exact in RTL and on the board, and the
  measured worst-case cycles per hop are within the target (≤ 25 ms). If
  not, Phase 6 becomes mandatory rather than an enhancement.

## Phase 6: Event-driven neuron engine (≈ 10 days)

- **Goal.** The neuromorphic co-design contribution (report E4). Sparse,
  parallel synaptic processing in the fabric.
- **Work.**
  - `rtl/neuron_engine.v` contains:
    - a spike FIFO in address-event form;
    - weight and delay BRAM (int4/int8 packed);
    - a spike-history ring buffer for delays (32 frames × 256 bits);
    - a 16-lane membrane-update datapath (accumulate, leak by shift,
      adaptive threshold, reset by subtraction);
    - output-spike write-back.
  - Control from the PicoRV32 through MMIO or PCPI (start frame, status,
    readout values). `kdot` stays for the dense input layer.
  - Unit testbenches: random spike patterns against a C model of the
    engine, plus the full SoC harness bit-exact with the oracle.
  - Vivado build: timing met at 100 MHz, resources reported in
    `results/hardware_engine.json`.
- **Measurements for the report.** Cycles per hop, and synaptic operations
  per hop with and without sparsity, for three versions:
  - software only;
  - `kdot` plus software;
  - `kdot` plus the engine.

  Resources (LUT, FF, BRAM, DSP) and an activity-based energy proxy
  (synaptic operations times energy per operation, as argued in [3][31]).
  Report it as a proxy, not a power measurement.
- **Done when** the board run over JTAG is bit-exact, with a measured
  speedup and resource cost.

## Phase 7: Evaluation, selection and documentation (≈ 3 days)

- Score the final model with all Phase 0 metrics against the release and
  trial models.
- Promote it to `deploy/` only if it meets the acceptance targets.
  Otherwise keep it as a documented candidate.
- Update `README.md`, `JOURNAL.md`, `make_report.py` (REPORT.md and
  presentation) and `package_release.py` (hashes, ABI).
- Live demo check over JTAG (or Ethernet once the SD card is reflashed)
  using the stream protocol.

---

## Schedule and dependencies

```
Phase 0 ─► Phase 1 ─┬─► Phase 2 ─┐
                    └────────────┴─► Phase 3 ─► Phase 4 ─► Phase 5 ─► Phase 6 ─► Phase 7
                         (gate after 1)   (gate after 3)          (gate after 5)
```

About 38 working days in total for one person, including 2 for the
multi-corpus data. Phases 1–2 and the RTL
groundwork of Phase 6 can overlap if work is split between group members,
for example one on training (Phases 1–4) and one on hardware (Phases 5–6,
starting from the bus fix and the engine's C model).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Delay learning (DCLS) unstable or slow to tune | Ablation without delays (Phase 3); ALIF recurrence alone already gives order sensitivity [3] |
| TTS negatives teach TTS artifacts, not phonetics | Keep real speech dominant; evaluate on held-out *voices*; the device-shift and real-speaker metrics decide |
| int4 hurts rejection of confusables more than overall accuracy | Evaluate quantized models on the confusable probe, not only on F1; fall back to int8 (it fits) |
| Neuron state drifts in long streams (no reset between words) | Test on hour-long streams; adaptation and leak bound the state; saturating int16 |
| Engine RTL takes longer than planned | Phase 5 already delivers real time in software; the engine is the enhancement and measurement |
| ABI change breaks the working demo | ABI v2 and v3 side by side; the server detects the model kind from the mailbox |
| Added corpora bring label noise (MSWC cut points) or language mix | Confidence and edge-energy filtering; English only for positives; corpus-balanced sampling; test metrics on curated sets |
| Dataset size and licences | Record the licence per clip in the manifest (CC0, CC BY 4.0); download English subsets only |

## Deferred (recorded, not planned)

- **On-board audio.** PYNQ-Z2 codec input and front end on the ARM or in
  fabric. It becomes relevant once the SD card boots PYNQ Linux.
- **Per-user enrollment ("dictation-style") mode.** See report §3.7. A
  prompted recording session fine-tunes only the readout and threshold on
  the PC. It is excluded to keep the system user-agnostic.
- **A NVFP4 or MX-format path.** Rejected in report §3.4. No hardware
  support, and no benefit for binary-activation SNNs.

References [n] refer to the list in [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md#references).
