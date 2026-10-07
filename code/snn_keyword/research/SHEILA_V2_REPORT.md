# Sheila Speech Commands v2 encoding report

## Outcome

The deployed Group-2 detector now recognizes `sheila` using an 8 frequency
band by 16 time-bin frontend, 40-5120 Hz coverage, a centered 1.0 s window,
and 8-bit feature levels. This is the best measured deployable trade-off among
accuracy, latency, and memory.

The selected floating model averaged 98.27% test accuracy over three seeds.
The deployed seed-0 integer model reached 98.07% accuracy on the 3,212-clip
official test split. On the PYNQ-Z2 it uses 6,440 parameter bytes and averages
10.39 ms inference latency at 100 MHz.

## Data and method

The source is Google Speech Commands v2 (`speech_commands_v0.02`). Preparation
keeps all 2,022 `sheila` clips and deterministically samples negatives across
the other 34 labels. The official validation and testing manifests are
preserved, producing this speaker-disjoint split:

| Split | Total | Sheila | Other | Speakers |
|---|---:|---:|---:|---:|
| Train | 9,606 | 1,606 | 8,000 | 1,894 |
| Validation | 2,704 | 204 | 2,500 | 249 |
| Test | 3,212 | 212 | 3,000 | 241 |

Each one-factor value was screened with a fixed seed. Selected and nearby
Pareto candidates were then retrained with seeds 0, 1, and 2. Thresholds were
chosen only on validation F1; the test split was used only for confirmation.

## Encoding relationships

### Feature precision and number of levels

At 16 bands, moving from 2 bits (4 levels) to 3-8 bits substantially improved
the screen result; 2 bits reached only 96.12% validation accuracy. Three-seed
test accuracy was 98.04% at 3 bits, 98.14% at 4 bits, 97.95% at 6 bits, and
98.08% at 8 bits. The differences above 3 bits are small and seed-dependent.

The relevant interaction test compared 8-band frontends directly:

| Encoding | Mean test accuracy | Mean test F1 | Logical packed input | Current ABI input |
|---|---:|---:|---:|---:|
| 8 bit, 8 bands | **98.27%** | **86.98%** | 128 B | 128 B |
| 4 bit, 8 bands | 98.17% | 86.13% | **64 B** | 128 B |

Relationship: precision below 3 bits is clearly harmful; 3-8 bits form a
plateau. Four bits is attractive only with a packed protocol. Because the
current Ethernet/BRAM ABI stores one byte per feature, 4-bit quantization saves
neither physical memory nor latency and slightly reduced mean accuracy.

### Frequency-band count

| Bands | Mean test accuracy | Mean test F1 | Parameter bytes | Input MACs |
|---:|---:|---:|---:|---:|
| 8 | **98.27%** | **86.98%** | **6,440** | **6,144** |
| 16 | 98.08% | 85.61% | 12,584 | 12,288 |
| 20 | 98.01% | 85.10% | 15,656 | 15,360 |
| 24 | 98.19% | 86.28% | 18,728 | 18,432 |
| 32 | 98.17% | 86.32% | 24,872 | 24,576 |

Relationship: model storage and first-layer MACs grow linearly with band
count, while accuracy plateaus. Eight bands dominates the larger frontends in
this experiment, halving model size and MACs relative to 16 bands while
slightly improving accuracy.

### Filtered frequency range

The one-seed screen peaked at 98.19% validation accuracy for 300-5120 Hz.
Across three seeds at 16 bands, 40-5120 Hz and 300-5120 Hz were statistically
close: 98.08% versus 98.01% mean test accuracy. Extending the upper cutoff to
6500 Hz reduced mean test accuracy to 97.86%; 7600 Hz recovered accuracy but
had worse balanced accuracy. Low cutoffs of 500 Hz also underperformed in the
screen.

Relationship: Sheila's useful information is concentrated below about 5.1
kHz, with a broad plateau for a 40-300 Hz lower boundary. The deployed 40 Hz
lower cutoff retains low-frequency evidence and was the directly confirmed
8-band configuration. The practical optimum is 40-5120 Hz.

### Represented time range

| Duration | Mean test accuracy | Mean test F1 |
|---:|---:|---:|
| 600 ms | 97.85% | 83.89% |
| 800 ms | 97.86% validation screen | 84.97% validation screen |
| 1000 ms | **98.08%** | **85.61%** |
| 1200 ms | 97.97% | 84.63% |

Relationship: 600-800 ms truncates useful context. With time bins fixed at
16, 1200 ms reduces temporal resolution and adds padding. One second is the
best measured duration and matches the dataset clip format.

## Final accuracy, latency, and memory

| Metric | Result |
|---|---:|
| Floating test accuracy, three-seed mean | 98.27% |
| Deployed integer test accuracy | 98.07% |
| Deployed integer test precision / recall / F1 | 83.78% / 87.74% / 85.71% |
| Deployed integer balanced accuracy | 93.27% |
| PicoRV32 latency, mean / maximum | 10.39 / 10.46 ms |
| Ethernet request/reply, mean / maximum | 13.72 / 26.37 ms |
| Host feature extraction during noise tests | about 1.0 ms/clip |
| Quantized model parameters | 6,440 B |
| Input frame | 128 B |
| Classifier local work arrays | 440 B |
| Complete firmware binary | 8,192 B |

The prior 16-band firmware measured about 17.55 ms on the same 100 MHz core.
The 8-band system therefore reduces physical inference latency by about 40.8%.

## Robustness rerun

The exact exported integer recurrence was tested after adding deterministic
white, low-frequency, and mains-like noise to every official test clip:

| Test | Accuracy | F1 | Balanced accuracy |
|---|---:|---:|---:|
| Clean | 98.07% | 85.71% | 93.27% |
| 10 dB SNR | 97.38% | 79.90% | 88.74% |
| 5 dB SNR | 96.73% | 73.01% | 82.91% |

Generated background/silence produced 0 false positives in 400 trials,
including 0/100 exact-silence frames. These are synthetic-noise results, not a
substitute for hours of recorded room audio.

## Board verification and changes

- Regenerated `keyword_model.h` with `KW_KEYWORD="sheila"`, 128 inputs, the
  selected weights, and a validation-chosen decision margin of two spikes.
- Changed the shared frontend and wire contract from 16x16 to 8x16.
- Corrected a pre-existing train/live mismatch: training used `array_split`
  for 98 FFT frames while live extraction used integer `linspace` boundaries.
  The two paths now quantize identically on checked WAVs.
- Made firmware input validation use generated `KW_INPUTS`, allowing band-count
  experiments without changing the global memory-map constant.
- Rebuilt and deployed the 8,192-byte firmware to `192.168.2.99`.
- Replayed 40 positive/negative golden vectors; all board spike counts matched
  Python exactly. A real Sheila WAV was detected and a real `backward` WAV was
  rejected.

## Next steps

1. Record hours of local office, fan, keyboard, music, and overlapping-speech
   audio and report false accepts per hour, not only per-frame accuracy.
2. Add local speakers saying Sheila at multiple distances and microphones to
   quantify domain shift from Speech Commands.
3. Implement packed 4-bit feature transport, then repeat board latency and
   memory measurements; otherwise 4-bit precision has no physical benefit.
4. Sweep hidden width and SNN timesteps jointly with the 8-band frontend to
   find additional model-memory and latency reductions.
