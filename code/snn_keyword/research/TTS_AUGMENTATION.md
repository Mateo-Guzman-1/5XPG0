# TTS hard-negative training

This experiment adds synthetic Sheila positives and nine confusing negatives:
she, sheep, shield, she'll, Shelley, she likes, shilling, shelter, and sealer.
The same voices and processing produce both labels, so the label cannot be
inferred from whether an example is synthetic. The original real Speech
Commands v2 subset and its official splits are unchanged.

## Corpus

Kokoro v1.0 runs locally through `kokoro-onnx` in an isolated environment.
Eight voices are training-only, two validation-only, and four test-only.
Every word is generated at speeds 0.85, 1.0 and 1.15. Each source has three
deterministic gain/position/noise variants: RMS 0.012, 0.035, 0.090 and clean,
30 dB, 20 dB white-noise SNR respectively. The result is 720 training, 180
validation and 360 test clips. Of the test clips, 36 are Sheila and 324 are
confusing negatives. The 360 clips share 120 source utterances and four voices;
they must not be treated as 360 independent speakers.

The 24 kHz source is low-pass filtered before resampling to mono 16 kHz PCM16.
Leading/trailing silence is trimmed, with margins. Speech longer than 880 ms
is compressed to fit in the one-second input rather than losing the ending.
The raw output is retained. These duration adjustments and the use of only
one TTS engine limit how closely the challenge represents real speech.

`data/tts_confusables/manifest.json` records voice, split, text, speed, processing,
audio hash, engine version and model hashes. Files and derived versions are
kept together by voice. Test words overlap training words deliberately; test
voices and source utterances do not. This tests new voices saying known
confusable vocabulary, not unseen negative vocabulary or independent engines.

## Controlled training and selection

All configurations use the existing 8x16, 8-bit, 300–3400 Hz encoding and
48-hidden-neuron / two-output SNN. The optimizer, 12 epochs, 4096 samples per
epoch and batch size 128 are fixed. Class-balanced sampling allocates 0%, 5%, 10%,
25% or 50% of each class's expected samples to TTS; the remaining examples
come from the real training set. Ten matched seeds are used for every fraction.
The confusion vocabulary is uniformly represented within the synthetic
negative class. TTS replaces part of the sample budget rather than adding
training steps.

Each seed's threshold maximizes real validation F1. The selected fraction
maximizes the sum of mean real and synthetic validation F1, provided mean real
validation F1 is within 0.5 percentage points of the baseline. The deployment
candidate seed maximizes the same validation sum. This constraint is a
selection rule, not a statistical non-inferiority guarantee. Test results are
computed after selection and cannot influence fraction or seed selection.

The original real test set and separate held-out-voice TTS test set are both
evaluated. Report real accuracy/F1, synthetic F1, Sheila recall, and negative
detection rate by word, with integer recurrence checks. Seed variability is
conditional on a small fixed synthetic corpus. It does not estimate variation
across independent real speakers, rooms or TTS engines.

## Reproduction

From `code/snn_keyword`, install `research/requirements-tts.txt` in an isolated
environment that can also access the existing SNN dependencies. Download
`kokoro-v1.0.onnx` and `voices-v1.0.bin` from
<https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1>.
Model weights are Apache-2.0; the ONNX wrapper is MIT. Preserve their notices
when redistributing assets.

```powershell
python research/generate_tts_confusables.py --assets data/tts_assets --output data/tts_confusables
python research/train_tts_confusables.py `
  --dataset data/speech_commands_v2_sheila_subset `
  --feature-cache data/extended_encoding_cache_sheila_v2/b8_f8_tb16_hz300-3400_t1000_g3000_s20000_mapequal_w25_h10_qlinear_mabsolute.npy `
  --tts data/tts_confusables
```

Use a new model directory after changing corpus or training settings. Existing
checkpoints are reused. The dataset/audio/model directories are ignored by Git;
scripts, manifest, per-clip predictions and measurements are research artifacts.

## Measured outcome

Validation selected 5% TTS sampling and seed 6 for the hardware candidate.

| Ten-seed floating-point test mean | Baseline | 5% TTS |
|---|---:|---:|
| Real accuracy | 98.35% | 98.29% |
| Real F1 | 87.46% | 86.83% |
| Synthetic F1 | 27.15% | 22.11% |
| Synthetic Sheila recall | 25.00% | 14.72% |
| Synthetic negative false-positive rate | 6.11% | 1.05% |

The real F1 paired mean difference is -0.63 points, with a conditional 95%
seed-paired interval of [-1.40, 0.13]. The false-positive improvement is
accompanied by lower keyword recall and is not an overall quality improvement.
The original detector therefore remains the live reference.

The selected integer candidate also reached 76.80% F1 at 10 dB SNR and
68.39% at 5 dB SNR, compared with the reference detector's 77.12% F1 at
5 dB. It produced 0/400 generated-background false positives, including
0/100 exact-silence examples. These tests reinforce the decision to retain
the live reference; they are not real-room streaming measurements.

The candidate's integer real accuracy/F1 is 98.41%/88.00%, with 6440 bytes of
parameters, 1024 input bits and an 8192-byte firmware binary. Both integer
models matched all 400 reference vectors each on the PYNQ: 40 real validation
vectors and every one of the 360 held-out synthetic test clips. Candidate
inference latency was 10.377 ms at 100 MHz (baseline 10.385 ms on the same
vectors), so there is no material hardware cost change.

Results, per-word detections, validation curves across all five mixtures,
per-clip margins and corpus provenance are saved under `research/results/tts_*`.
The manifest is copied to `research/results/tts_corpus_manifest.json`; audio
and trained PyTorch models remain in `data/tts_confusables` and `data/tts_models`.
The selected integer model and its firmware remain under `data/tts_board_exports`.

To reproduce the hardware checks after training, materialize the real feature
cache under `data/tts_models` using its full encoding-key filename, then run:

```powershell
python research/benchmark_reproducibility_board.py `
  --report research/results/tts_augmentation_board_input.json `
  --dataset data/speech_commands_v2_sheila_subset `
  --cache-dir data/tts_models --models data/tts_models `
  --extra-vectors data/tts_models/tts_test_vectors.npz `
  --output research/results/tts_augmentation_board.json `
  --exports data/tts_board_exports
```

The failed generalization to held-out synthetic voices is a finding, not a
reason to tune on this test set. A new independent challenge set is needed
before further changes can support an unbiased generalization claim.
