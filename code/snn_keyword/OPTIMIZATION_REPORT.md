# Optimization routes for spiking keyword detection on the PYNQ-Z2

Group 2, 5XPG0. This report compares ways to improve the "yes" detector
beyond the current release. It weighs model, training, data, quantization
and hardware options, and ends with the route chosen and why.
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) turns that choice into
work packages. Measured numbers refer to files in `results/`; history is in
[JOURNAL.md](JOURNAL.md).

**Constraints set for this work**

- **User- and microphone-agnostic.** The detector must work for unseen
  speakers and microphones. Training on the developer's own voice is
  excluded. A per-user enrollment mode is discussed (§3.7) but not planned.
- **No on-board audio yet.** Audio capture and the mel front end stay on
  the PC. The PYNQ-Z2 audio codec path is deferred.
- **Keep the verification discipline.** An integer reference model must be
  bit-exact with native C, RTL simulation and the board.
- **Target hardware.** PYNQ-Z2 (XC7Z020: 53,200 LUTs, 220 DSP slices,
  140 BRAM36 ≈ 630 KB, 100 MHz fabric). The PicoRV32 SoC uses 64 BRAM36
  (256 KB); about 76 remain free.

---

## 1. Where the current system stands

| Measure (held-out, `results/confusables.json`) | Release model |
|---|---|
| Clip-level precision / recall / F1 | 0.902 / 0.852 / 0.876 |
| Live windows, single decision: "yes" detected / other words accepted | 74.5% / 1.07% |
| Live windows, "2 of 3" confirmation | 67.3% / 0.13% |
| Synthesized "yeets"/"yets"/"yetz", single window (250 ms hop) | 75% |
| Worst-case inference with the `kdot` instruction | 267 k cycles = 2.67 ms |

**Root causes** (JOURNAL.md, entries 4–6):

1. **No temporal structure.** The network is one dense layer over a 1 s
   spectrogram snapshot, followed by LIF neurons driven by a constant
   current. It cannot represent "vowel, then *no* gap, then /s/" at an
   arbitrary position. Larger hidden layers (up to 256) and finer time
   bins (64) did not help.
2. **A narrow training task.** It was trained only as "yes" against the
   rest, so it never needed features that separate "yes" from "yeah",
   "guess" or "yeets". Speech Commands [1] contains no such near-misses.
3. **Data mismatch with streaming use.** Training clips are centred, while
   the demo slides a window. Augmentation fixed part of this, at a
   5–10 point recall cost.
4. **Dependence on one recording setting.** All real training audio comes
   from one collection: Speech Commands [1]. That means isolated 1 s words,
   read on command, close to a consumer microphone, in mostly quiet
   rooms. A model trained on one source learns that source's microphones,
   rooms, speaking style and level as part of "yes".
5. **Compute is not the bottleneck.** After `kdot`, inference uses about
   1% of the 250 ms hop. Of the remaining cycles, 72% go to the LIF time
   loop.

The route chosen therefore has to fix the model and the data first. The
hardware has room to spare and should be shaped around the chosen model.

---

## 2. What the literature says is achievable

| System | Type | Task | Accuracy | Ref |
|---|---|---|---|---|
| Adaptive spiking recurrent network (SRNN, ALIF neurons) | SNN | GSC | 92.1% | [3] |
| Same SRNN | SNN | SHD / SSC (spiking audio benchmarks [6]) | 90.4% / 74.2% | [3] |
| Feedforward SNN with learnable delays (DCLS) | SNN | GSC v2, 35 classes | 95.35% | [4] |
| Surrogate-gradient SNN baseline | SNN | Speech commands | Competitive with ANNs | [5] |
| SpikeSCR (SNN, curriculum knowledge distillation) | SNN | SHD, SSC, GSC v2 | State of the art at publication; 60% fewer time steps, 54.8% less energy | [7] |
| BC-ResNet-1 / BC-ResNet-8 | ANN (CNN) | GSC, 12 commands | 96.6% (<10k params) / 98.0% | [8] |

Two findings matter most for this project.

- **Temporal SNNs are within a few points of the best ANNs.** Recurrent
  adaptive neurons [3] and learnable synaptic delays [4] give that result.
  The current design uses neither.
- **Sparsity is the efficiency lever.** In [3], typical per-neuron firing
  probability was below 0.1 per time step. A spike only needs an
  accumulate (AC) instead of a multiply-accumulate (MAC). [3] cites about
  a 31× energy advantage of AC over MAC in 45 nm CMOS [31]. Commercial
  neuromorphic audio shows the same effect: Xylo classifies ambient audio
  in streaming mode at under 100 µW of dynamic power [30].

---

## 3. Optimization routes

Each route is scored on expected accuracy gain, cost to implement, fit to
the PYNQ, and fit to the group's neuromorphic hardware/software co-design
question.

### 3.1 Model architecture

| Option | Idea | For | Against |
|---|---|---|---|
| **A1. Dense snapshot SNN (current)** | 1 s window → dense layer → constant-current LIF | Simple; already verified | Cannot model temporal order (§1); has plateaued |
| **A2. Streaming recurrent ALIF SNN** [3] | Process one 10 ms frame at a time; recurrent adaptive-threshold neurons keep state | Order-aware by construction; no overlapping windows; sparse; proven on GSC | BPTT training is slower; state must persist across frames in hardware |
| **A3. A2 plus learnable delays (DCLS)** [4] | Each synapse learns a delay; a Gaussian kernel shrinks to one tap during training | Best reported SNN accuracy on GSC-35; delays directly model a "gap then /s/" pattern | Delays need a spike history buffer in hardware; more training hyperparameters |
| A4. Spiking transformer or conv SNN [7] | Attention or conv blocks with spiking activations | Highest reported SNN accuracy | Far larger; attention is dense and poorly suited to a 64 k-word BRAM and event-driven hardware |
| A5. Small non-spiking CNN or transformer (BC-ResNet [8], KWT [9]) | Run a tiny CNN on the RISC-V or in fabric | Best accuracy per parameter | Abandons the SNN research question; dense MACs; no sparsity benefit |
| A6. ANN-to-SNN conversion [12] | Train an ANN, map activations to spike rates | Reuses ANN tooling | Needs many time steps to approximate rates; poor fit for streaming audio; worse than direct training in [5] |

**Assessment.** A2 and A3 address the root cause. A3 adds most where the
problem lies: timing between phonemes. A5 is the strongest *baseline* and
is used as the teacher (§3.2), not as the deployed model.

### 3.2 Training and transfer from large to small models

| Option | Idea | Expected effect | Cost |
|---|---|---|---|
| **B1. Multi-class pretraining** | Train on all 35 GSC words (plus silence and unknown), then fine-tune yes against the rest | Forces features that separate "yes" from its neighbours; the most direct fix for cause 2 | Low; same data |
| **B2. Knowledge distillation (KD)** [13], on top of surrogate-gradient training [2] | Student matches a teacher's softened outputs; curriculum KD has been shown for spiking speech-command models [7] | Transfers "how much each word resembles yes"; typical gains of a few points; better near-miss ranking | Teacher training once (BC-ResNet-8 [8], hours on the RTX 4060) |
| B3. Large pretrained speech encoder as teacher (wav2vec 2.0, Whisper, …) | A stronger teacher | Slightly better soft labels than B2 | Heavy; little benefit over a 98% BC-ResNet teacher for 35 words |
| **B4. Streaming training** [11] | Long synthetic streams with keywords at random positions; loss at the end of the word | Matches deployment and removes the centring mismatch (cause 3) | Moderate; data loader work |
| **B5. Sparsity regularization** [3][34] | Penalize the spike count | Fewer events, so less energy and faster event-driven hardware | Trivial; tune against accuracy |
| B6. Learnable time constants and heterogeneity [3][36] | Train per-neuron τ_m and τ_adapt | Better multi-timescale memory; reported to help in [3] | Low; ALIF already has it |

**Assessment.** B1, B2, B4 and B5 are cheap and complementary. B3 is not
worth its cost at this task size.

### 3.3 Data and robustness without the user's voice

The goal is robustness to *any* speaker and microphone, reached through
data diversity and front-end normalization rather than user recordings.

**More diverse real audio.** Augmentation (C2–C4 below) varies a
recording after the fact. It cannot add speakers, accents, speaking
styles or genuine microphones that were never recorded. The training set
should therefore draw real audio from several independent collections, so
that no single recording setting defines what "yes" sounds like:

| Source | What it adds | "yes" positives | Negatives | Licence |
|---|---|---|---|---|
| Speech Commands v0.02 [1] (current) | Isolated words from thousands of crowd-sourced speakers on consumer mics | ≈4,000 | 34 other words | CC BY 4.0 |
| **Common Voice, Single Word Target Segment** [37][38] | Digits, "yes", "no", "hey", "Firefox" from ≈11,000 contributors in 18 languages: new speakers, accents, devices | Yes (English, plus accented English from non-native speakers) | "no", digits, "hey" | CC0 |
| **Multilingual Spoken Words Corpus** (MSWC) [39] | 1 s keyword clips cut by forced alignment from Common Voice *sentences*: words in running speech, different prosody and coarticulation | Yes ("yes" inside sentences) | Thousands of English words, including real /s/-, /ts/- and /tʃ/-final words ("its", "gets", "each", "guess") | CC BY 4.0 |
| **LibriSpeech** [40] | 1,000 h of read audiobook speech from many microphones and rooms | Few; not used for positives | Continuous speech for negatives and a false-accepts-per-hour stream | CC BY 4.0 |
| Background and noise sets (MUSAN [18], RIR database [17]) | Music, babble and real rooms | — | Non-speech negatives, augmentation | Per set |

The rules for combining them:
- **Balance by source** during sampling, so the largest corpus does not
  dominate.
- **Keep splits speaker-disjoint within each corpus.** Corpus splits do not
  share speaker identifiers, so a speaker overlap *across* corpora cannot
  be fully excluded. Keep the official test splits and add the check below.
- **Measure setting dependence directly with leave-one-source-out
  evaluation**: train without one corpus, test on it. A small gap means
  the model does not depend on one setting.
- **Treat MSWC labels as noisy.** Forced-alignment cut points can clip word
  edges, which matters for the final /s/. Filter clips by alignment
  confidence and energy at the edges, and never use MSWC alone for test
  metrics on the /s/ cue.

| Option | Idea | For | Against |
|---|---|---|---|
| **C1. Multi-speaker TTS hard negatives** [14][15] | Synthesize near-misses ("yeets", "yeah", "guess", "mes", "tes", …) with a multi-speaker TTS (hundreds of voices), for example by grapheme edits of the keyword as in GraphemeAug [15] | Covers exactly the observed failure mode; [15] reports a 61% improvement on synthetic hard negatives with positives preserved; [14] shows synthetic speech can replace hundreds of real examples | TTS voices differ from real speech; keep real data dominant and evaluate on held-out voices |
| **C2. Channel/microphone augmentation** [22] | Random microphone-like filtering: parametric EQ, band limits, gain, clipping, codec artifacts, and learned microphone styles as in MicAugment [22] | Directly targets unknown microphones; no recordings needed | Parametric filters only approximate real devices |
| **C3. Room and noise augmentation** [17][18] | Convolve with room impulse responses (RIR database of [17]); add MUSAN noise and music [18] | Standard, large gains in far-field and noisy use | Larger dataset download |
| **C4. SpecAugment, speed and vocal-tract perturbation** [16][19] | Time/frequency masking; speed perturbation; VTLP | Speaker-independence; cheap | Too much masking can remove the /s/ cue; keep time masks short |
| **C5. PCEN front end** [20] | Per-channel energy normalization instead of static log compression; trainable | Removes gain and slow spectral tilt (microphone and room colouring); shown to improve far-field KWS | Front-end change on the PC and later on the board; must stay integer-friendly |
| C6. Inference-time mic mapping (Mic2Mic) [21] | CycleGAN that maps an unknown microphone to the training one | Recovers 66–89% of the accuracy lost to microphone mismatch in [21] | Needs unlabeled audio from the target mic and a GAN at inference; too heavy here |
| C7. Test-time adaptation (TENT) [35] | Update normalization parameters online from unlabeled input | No labels needed | Risky for a detector (can drift towards always "no"); hard to verify bit-exactly |
| C8. Device-shift evaluation | Test on microphone filters and room impulse responses held out from training | Measures agnosticism without a user recording | Proxy only |

**Assessment.** More diverse real sources plus C1–C5 and C8 are
practical, all offline, and together cover speaker, accent, speaking-style,
microphone and room variation. Real multi-corpus data comes first, because
augmentation and synthesis can only vary what was recorded. C6 and C7 add
runtime complexity for uncertain gain.

### 3.4 Quantization

Spike activations are already 1 bit, so quantization concerns weights and
neuron state only.

| Option | Weights for a ~45 k-synapse model | Expected loss (with QAT) | Hardware |
|---|---|---|---|
| int16 (current) | 90 KB | none | done |
| **int8 per-channel** [23]; standard for microcontroller KWS [10] | 45 KB | ≈ none | trivial |
| **int4 per-channel** | 23 KB | usually <1 point for small KWS models | 8 weights per word; shifts and adds |
| 4-bit codebook (16 learned levels, NF4-style) [26] | 23 KB | slightly below int4 | 16-entry lookup table |
| Ternary {−1, 0, +1} | ≈ 11 KB | a few points | no multiplier at all |
| **NVFP4** [25] / MXFP4 [24] | 23 KB + scales | fine on GPUs | E2M1 elements with an FP8 (E4M3) scale per 16 values and an FP32 tensor scale. Built for Blackwell tensor cores. It solves the dynamic range of large LLM activations, a problem this model does not have. No FP4 arithmetic in PicoRV32 or the fabric: decoding costs more than integer weights |

**Assessment.** Keep neuron state as int16 and accumulators as int32. Use
int8 QAT first, then int4 if the §5 targets still hold. NVFP4 is rejected:
its advantage (wide dynamic range at 4 bits for large-model activations)
does not apply to a small SNN with binary activations and integer
hardware.

### 3.5 Hardware execution

| Option | Idea | Speedup / benefit | Cost |
|---|---|---|---|
| E1. Current `kdot` PCPI | Streaming dense dot product | 22.4×, done | — |
| E2. Faster BRAM bus (`READ_WAIT` 8 → 1–2) | Remove unnecessary wait states | About 3× on all RV32 code | Very low; re-verify timing |
| E3. LIF update instruction | One instruction per neuron update | About 3.5× on the current loop | Low; specific to the current model |
| **E4. Event-driven neuron engine** [27][28] | PL block: spike FIFO → weight row fetch (int4/int8) → parallel membrane update (for example 16 lanes) → ALIF threshold/adaptation → output spikes; RISC-V as controller | Cost scales with spikes rather than weights; the neuromorphic principle made concrete; plenty of headroom (§4) | Largest RTL effort; new verification |
| E5. Run on the ARM Cortex-A9 (NEON) | Software on the PS | Fast to develop | No neuromorphic contribution; ARM needs PYNQ Linux or bare metal |
| E6. Dedicated neuromorphic chips (Loihi [29], Xylo [30]) | External hardware | Real sub-mW numbers | Not the course platform |

**Assessment.** E2 is a cheap win for any model. E4 is the co-design
contribution: it supports the chosen model (§3.1) and allows a measured
comparison of dense (`kdot`), event-driven and software execution. E3 is
superseded by E4.

### 3.6 Decision layer

The current "2 of 3" confirmation (hand-tuned, validation-selected
threshold) cut false accepts about 8×. With a streaming recurrent model,
evidence integrates in the leaky readout itself. The decision becomes a
threshold on the readout with a short hold-off. Confirmation stays
available as a fallback and as an ablation.

### 3.7 Per-user enrollment mode (documented only, not planned)

Classic dictation software adapted to one speaker through an enrollment
session. The user read prompted text, and the acoustic model was adapted,
for example with MLLR [32]. Modern keyword spotting does the same with a
few recorded utterances (few-shot on-device customization [33]).

**A possible design.** An "enroll" command in `pc_keyword_demo.py` would:
1. Prompt the user for about 10 × "yes" and about 10 × listed near-misses
   ("yeah", "yeets", "guess", …) plus 30 s of room noise.
2. Fine-tune only the readout (or a prototype/threshold per user) on the
   PC against the frozen network, keeping a held-out check against the
   general validation set.
3. Export a new readout and threshold to the firmware (a small blob) and
   keep the general model as the default.

**Why it is not planned.** It makes the system user-specific, which this
project explicitly avoids. It needs a re-enrollment per user and per
microphone. It adds a data-handling path. Adaptation from about 20 clips
can also overfit and degrade rejection of words not recorded. It stays
listed as a future option. The general route (§3.3) should first show how
far a user-agnostic model gets.

---

## 4. Does the chosen model fit the PYNQ-Z2?

Sizing for a representative model: 24 input bands every 10 ms →
layer 1 with 128 ALIF neurons (feedforward, learnable delays) →
layer 2 with 128 ALIF neurons (recurrent) → leaky readout. Proposed
target, to be confirmed by the ablations in the plan.

| Quantity | Estimate | Budget | Margin |
|---|---|---|---|
| Synapses | 24×128 + 128×128 (in) + 128×128 (rec) + 128×3 ≈ 36 k | 45 KB of weights at int8 | Fits in about 12 free BRAM36 at int8, or 6 at int4 |
| Delays | ≤ 32 frames (320 ms) per synapse, 5 bits | ≈ 23 KB | Stored beside the weights |
| Neuron state | 256 membranes + 256 adaptations, int16 | 1 KB | Trivial |
| Dense input layer per frame | 3 k MAC (reuses `kdot`) | 25 frames per 250 ms hop → 77 k MAC | ≈ 60 k cycles |
| Spike-driven synapse updates per frame (5% firing) | ≈ 1.6 k accumulates | 41 k per hop | Software ≈ 4–6 M cycles (40–60 ms, feasible); engine at 16 lanes ≈ 3 k cycles |
| Worst case (every neuron fires) | 33 k accumulates per frame | Engine ≈ 2 k cycles per frame (20 µs per 10 ms) | ≈ 500× headroom |

Two consequences follow:
- **The model fits** in BRAM with room to spare, even at int8.
- **Real time is reachable in stages.** Before any new RTL, the model is
  likely feasible in software on the PicoRV32 (with `kdot` for layer 1 and
  a faster bus). The event-driven engine then turns "feasible" into
  "about 100× headroom and sparse by construction" and is the measured
  co-design result.

---

## 5. Tradeoff summary

| Route | Accuracy on confusables | Effort | Fits PYNQ | Neuromorphic value | Risk |
|---|---|---|---|---|---|
| Keep A1, tune data only | + | low | yes | low | plateau already observed |
| **A2 + A3, with B1, B2, B4, B5, C1–C5** | +++ | medium–high | yes (§4) | high | BPTT/delay tuning; hardware delays |
| A5 BC-ResNet on the PYNQ | +++ | medium | yes | none | abandons the research question |
| A4 spiking transformer | +++ | high | marginal | medium | too large and dense for BRAM |
| NVFP4 quantization | 0 | medium | poor | none | no hardware support |
| int8 → int4 QAT | 0 / − | low | yes | medium | small accuracy loss |
| E4 event-driven engine | 0 (speed and energy) | high | yes | high | RTL verification effort |
| Enrollment mode | + (for one user) | medium | yes | low | not user-agnostic (excluded) |

---

## 6. Conclusion: chosen route

**Chosen:** a **streaming, two-layer adaptive-LIF spiking network with
learnable synaptic delays** [3][4]. It is:
- **pretrained on all 35 GSC words and distilled from a BC-ResNet-8
  teacher** [8][13], then fine-tuned for "yes" against everything else;
- **trained on diverse real audio from several corpora** (Speech Commands,
  Common Voice single words, MSWC, LibriSpeech negatives [1][37]–[40]),
  so that no single recording setting defines the keyword;
- **trained on streaming data** with **user- and microphone-agnostic
  augmentation**: multi-speaker TTS hard negatives, microphone and room
  simulation, noise, SpecAugment and speed/VTLP [14]–[19][22];
- fed by a **PCEN front end** [20];
- **quantized with QAT to int8, then int4** [23];
- executed on the PYNQ **first in RV32 software** (with `kdot` for the
  dense input layer and a faster BRAM bus), then on an **event-driven
  neuron engine in the fabric** [27][28] with the PicoRV32 as controller.

**Why this route.**

1. **It fixes the root cause.** Recurrence, adaptation and delays give the
   network temporal order. The current architecture lacks it, and more
   data, bigger layers and finer time bins did not compensate (JOURNAL.md,
   entries 4 and 6). Delays specifically capture inter-phoneme timing such
   as /s/ versus /t/+/s/.
2. **The evidence is strong and recent.** SNNs of this kind reach 92–95%
   on GSC [3][4], within a few points of the best ANNs [8], at sparse
   activity [3].
3. **It is user- and microphone-agnostic by design.** Robustness comes from
   diverse real recordings across several corpora, synthetic and augmented
   data, and front-end normalization (§3.3), not from any one person's
   recordings or one collection setting. Leave-one-source-out testing
   verifies it.
4. **It fits the hardware with large margins** (§4) and keeps the existing
   verification discipline (integer oracle, bit-exact C/RTL/board).
5. **It maximizes the co-design contribution.** The event-driven engine
   exploits sparsity, the property that distinguishes neuromorphic
   computation. Comparing it against dense `kdot` and against software
   gives a clear, measurable answer to the group's research question.

**Rejected or deferred.**
- NVFP4: no hardware support and no benefit for binary-activation SNNs.
- Spiking transformers: too large and dense.
- A pure CNN: kept as teacher and baseline, not deployed.
- ANN-to-SNN conversion: a poor fit for streaming.
- Mic2Mic and test-time adaptation: runtime complexity and verification
  risk.
- Per-user enrollment: documented in §3.7 only.
- On-board audio: deferred by scope.

---

## References

[1] P. Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition," arXiv:1804.03209, 2018.
[2] E. O. Neftci, H. Mostafa, F. Zenke, "Surrogate Gradient Learning in Spiking Neural Networks," *IEEE Signal Processing Magazine*, 36(6):51–63, 2019.
[3] B. Yin, F. Corradi, S. M. Bohté, "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks," *Nature Machine Intelligence*, 3:905–913, 2021. arXiv:2103.12593.
[4] I. Hammouamri, I. Khalfaoui-Hassani, T. Masquelier, "Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings," *ICLR*, 2024.
[5] A. Bittar, P. N. Garner, "A surrogate gradient spiking baseline for speech command recognition," *Frontiers in Neuroscience*, 16:865897, 2022.
[6] B. Cramer, Y. Stradmann, J. Schemmel, F. Zenke, "The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks," *IEEE Trans. Neural Networks and Learning Systems*, 33(7):2744–2757, 2022.
[7] J. Wang et al., "Efficient Speech Command Recognition Leveraging Spiking Neural Network and Curriculum Learning-based Knowledge Distillation," arXiv:2412.12858, 2024.
[8] B. Kim, S. Chang, J. Lee, D. Sung, "Broadcasted Residual Learning for Efficient Keyword Spotting," *Interspeech*, 4538–4542, 2021.
[9] A. Berg, M. O'Connor, M. T. Cruz, "Keyword Transformer: A Self-Attention Model for Keyword Spotting," *Interspeech*, 2021.
[10] Y. Zhang, N. Suda, L. Lai, V. Chandra, "Hello Edge: Keyword Spotting on Microcontrollers," arXiv:1711.07128, 2017.
[11] O. Rybakov, N. Kononenko, N. Subrahmanya, M. Visontai, S. Laurenzo, "Streaming Keyword Spotting on Mobile Devices," *Interspeech*, 2277–2281, 2020.
[12] B. Rueckauer, I.-A. Lungu, Y. Hu, M. Pfeiffer, S.-C. Liu, "Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification," *Frontiers in Neuroscience*, 11:682, 2017.
[13] G. Hinton, O. Vinyals, J. Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531, 2015.
[14] J. Lin, K. Kilgour, D. Roblek, M. Sharifi, "Training Keyword Spotters with Limited and Synthesized Speech Data," *ICASSP*, 7474–7478, 2020.
[15] H. Zhang, K. Partridge, P. Zhu, N. Chen, H. J. Park, D. Agarwal, Q. Wang, "GraphemeAug: A Systematic Approach to Synthesized Hard Negative Keyword Spotting Examples," *Interspeech*, 2025. arXiv:2505.14814.
[16] D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," *Interspeech*, 2019.
[17] T. Ko, V. Peddinti, D. Povey, M. L. Seltzer, S. Khudanpur, "A study on data augmentation of reverberant speech for robust speech recognition," *ICASSP*, 2017.
[18] D. Snyder, G. Chen, D. Povey, "MUSAN: A Music, Speech, and Noise Corpus," arXiv:1510.08484, 2015.
[19] N. Jaitly, G. E. Hinton, "Vocal Tract Length Perturbation (VTLP) improves speech recognition," *ICML Workshop on Deep Learning for Audio, Speech and Language*, 2013.
[20] Y. Wang, P. Getreuer, T. Hughes, R. F. Lyon, R. A. Saurous, "Trainable frontend for robust and far-field keyword spotting," *ICASSP*, 5670–5674, 2017.
[21] A. Mathur, A. Isopoussu, F. Kawsar, N. Berthouze, N. D. Lane, "Mic2Mic: Using Cycle-Consistent Generative Adversarial Networks to Overcome Microphone Variability in Speech Systems," *IPSN*, 2019.
[22] Z. Borsos, Y. Li, B. Gfeller, M. Tagliasacchi, "MicAugment: One-shot Microphone Style Transfer," arXiv:2010.09658, 2020.
[23] B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," *CVPR*, 2018.
[24] B. D. Rouhani et al., "Microscaling Data Formats for Deep Learning," arXiv:2310.10537, 2023.
[25] NVIDIA, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference," NVIDIA Technical Blog, 2025. https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
[26] T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," *NeurIPS*, 2023.
[27] C. Frenkel, M. Lefebvre, J.-D. Legat, D. Bol, "A 0.086-mm² 12.7-pJ/SOP 64k-Synapse 256-Neuron Online-Learning Digital Spiking Neuromorphic Processor in 28-nm CMOS," *IEEE Trans. Biomedical Circuits and Systems*, 13(1):145–158, 2019.
[28] A. Carpegna, A. Savino, S. Di Carlo, "Spiker: an FPGA-optimized Hardware Accelerator for Spiking Neural Networks," *IEEE ISVLSI*, 14–19, 2022.
[29] M. Davies et al., "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning," *IEEE Micro*, 38(1):82–99, 2018.
[30] H. Bos, D. R. Muir, "Sub-mW Neuromorphic SNN audio processing applications with Rockpool and Xylo," in *Embedded Artificial Intelligence*, River Publishers, 2023, 94–103. arXiv:2208.12991.
[31] M. Horowitz, "Computing's energy problem (and what we can do about it)," *IEEE ISSCC*, 10–14, 2014.
[32] C. J. Leggetter, P. C. Woodland, "Maximum likelihood linear regression for speaker adaptation of continuous density hidden Markov models," *Computer Speech & Language*, 9(2):171–185, 1995.
[33] M. Rusci, T. Tuytelaars, "Few-Shot Open-Set Learning for On-Device Customization of KeyWord Spotting Systems," *Interspeech*, 2768–2772, 2023.
[34] F. Zenke, T. P. Vogels, "The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks," *Neural Computation*, 33(4):899–925, 2021.
[35] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, T. Darrell, "Tent: Fully Test-Time Adaptation by Entropy Minimization," *ICLR*, 2021.
[36] N. Perez-Nieves, V. C. H. Leung, P. L. Dragotti, D. F. M. Goodman, "Neural heterogeneity promotes robust learning," *Nature Communications*, 12:5791, 2021.
[37] R. Ardila et al., "Common Voice: A Massively-Multilingual Speech Corpus," *LREC*, 2020.
[38] Mozilla Common Voice, "Single Word Target Segment" (digits, yes, no, hey, Firefox; 2020). https://github.com/mozilla/common-voice/blob/main/docs/taxonomies/singleword-benchmark.md
[39] M. Mazumder et al., "Multilingual Spoken Words Corpus," *NeurIPS Datasets and Benchmarks Track*, 2021.
[40] V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: An ASR corpus based on public domain audio books," *ICASSP*, 5206–5210, 2015.
