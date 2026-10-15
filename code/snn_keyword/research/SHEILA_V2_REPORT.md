# Sheila Speech Commands v2 encoding report

## Outcome

The deployed Group-2 detector recognizes `sheila` with an 8-frequency-band by
16-time-bin frontend, 300-3400 Hz coverage, a centered 1.0 s window, and 8-bit
feature values. Its RMS silence gate is 0.003 and its selected full-scale RMS
is 0.020.

Across three floating-model seeds, the selected encoding averaged 98.36% test
accuracy and 87.60% F1. The deployed seed-0 integer model reached 98.57%
accuracy and 88.67% F1 on the 3,212-clip official test split. On the 100 MHz
PYNQ-Z2 PicoRV32 it uses 6,440 parameter bytes and averages 10.40 ms per
inference.

## Data and method

The source is Google Speech Commands v2 (`speech_commands_v0.02`). Preparation
keeps all 2,022 `sheila` clips and deterministically samples negatives across
the other 34 labels. The official speaker-disjoint split is:

| Split | Total | Sheila | Other | Speakers |
|---|---:|---:|---:|---:|
| Train | 9,606 | 1,606 | 8,000 | 1,894 |
| Validation | 2,704 | 204 | 2,500 | 249 |
| Test | 3,212 | 212 | 3,000 | 241 |

Every one-factor value was screened with fixed seed 0. Validation F1 selected
the candidates for three-seed confirmation, and the held-out test split was
used only for confirmation. All ten sweeps used one common final baseline:
8 bands, 16 time bins, 8-bit values, 300-3400 Hz, a 1000 ms window, a 0.003
RMS silence gate, and 0.020 RMS full scale.

All 62 sweep candidates were compiled for RV32IM and replayed on the physical
PYNQ using 40 reference vectors each. All 2,480 board score comparisons matched
the Python integer recurrence.

## Primary sweep relationships

| Sweep | Tested values | Main result |
|---|---|---|
| Feature precision | 2, 3, 4, 6, 8 bits | 2 bits clearly lost quality; 3-8 bits formed a broad confirmed plateau |
| Frequency bands | 8, 12, 16, 20, 24, 32 | 20 bands gained 1.11 F1 points but required about 2.5x the input/model size and 2.04x the latency |
| Frequency range | 13 ranges from 40-3400 to 500-7600 Hz | 300-3400 Hz produced the strongest screened result at unchanged deployment cost |
| Audio duration | 400-1600 ms in 100 ms steps | 1000 ms won validation; the confirmed test curve plateaued from 700 to 1300 ms |

The selected encoding reached 98.36% mean test accuracy and 87.60% mean F1.
The wider 300-6500 and 300-7600 Hz candidates fell to 83.73% and 83.24%
confirmed F1, respectively. Range changes do not alter input size, parameter
memory, or board inference work.

Band count determines the main deployment cost:

| Bands | Logical input | Parameter bytes | Mean PYNQ latency |
|---:|---:|---:|---:|
| 8 | 1,024 bit | 6,440 | 10.40 ms |
| 12 | 1,536 bit | 9,512 | 13.99 ms |
| 16 | 2,048 bit | 12,584 | 17.56 ms |
| 20 | 2,560 bit | 15,656 | 21.19 ms |
| 24 | 3,072 bit | 18,728 | 24.80 ms |
| 32 | 4,096 bit | 24,872 | 31.98 ms |

The relationship is approximately linear because each added input contributes
one weight to every hidden neuron and one first-layer multiply-accumulate.

## Secondary sweep relationships

| Sweep | Tested values | Result |
|---|---|---|
| Time bins | 8, 12, 16, 24, 32 | 16 bins gave the highest screened F1; cost scaled approximately linearly |
| RMS gate / full scale | 4 gates x 3 full-scale values | 0.003 / 0.020 gave the highest screened F1 at unchanged deployment cost |
| Band mapping | Equal-Hz, logarithmic, mel | All were tested; equal-Hz remained selected |
| FFT window / hop | 8 pairs from 15/5 to 40/20 ms | All were tested; confirmed differences were marginal |
| Four-bit quantizer | Linear, logarithmic | Logarithmic was the stronger 4-bit screen candidate but remained below the 8-bit baseline after confirmation |
| Temporal features | Absolute, absolute plus delta | Delta did not improve screened F1 and doubled input, model, and latency |

Logarithmic 4-bit features reached 98.22% mean test accuracy and 86.42% F1.
They remain a useful packed-representation target, but the current ABI stores
every feature in one byte. It therefore still transfers 1,024 physical bits
and provides no physical RAM or latency saving yet.

Time-bin count shows the same near-linear deployment relationship:

| Time bins | Logical input | Parameter bytes | Mean PYNQ latency |
|---:|---:|---:|---:|
| 8 | 512 bit | 3,368 | 6.79 ms |
| 12 | 768 bit | 4,904 | 8.58 ms |
| 16 | 1,024 bit | 6,440 | 10.40 ms |
| 24 | 1,536 bit | 9,512 | 13.99 ms |
| 32 | 2,048 bit | 12,584 | 17.56 ms |

## Final deployment

| Metric | Result |
|---|---:|
| Floating test accuracy, three-seed mean | 98.36% |
| Floating test F1, three-seed mean | 87.60% |
| Deployed integer test accuracy | 98.57% |
| Deployed integer precision / recall / F1 | 92.78% / 84.91% / 88.67% |
| Deployed integer balanced accuracy | 92.22% |
| PicoRV32 latency, mean / maximum | 10.40 / 10.49 ms |
| Steady-state Ethernet latency, mean / maximum | 13.68 / 13.98 ms |
| Quantized model parameters | 6,440 B |
| Logical / physical input | 1,024 bit / 128 B |
| Complete firmware binary | 8,192 B |
| Deployment reference agreement | 40/40 vectors |

The persistent `snn-keyword.service` is enabled and active on `192.168.2.99`.
The live and training frontends produced identical uint8 features for all 128
sampled WAV files after the deployment update.

## Robustness rerun

The deployed integer recurrence was retested after adding deterministic white,
low-frequency, and mains-like noise to every official test clip:

| Test | Accuracy | F1 | Balanced accuracy |
|---|---:|---:|---:|
| Clean | 98.57% | 88.67% | 92.22% |
| 10 dB SNR | 98.04% | 84.67% | 90.62% |
| 5 dB SNR | 97.23% | 77.12% | 84.93% |

Generated background and silence produced 0 false positives in 400 trials,
including 0/100 exact-silence frames. These generated tests are not a
substitute for continuous recordings from the intended room and microphone.

## Next steps

1. Record hours of office, fan, keyboard, music, and overlapping-speech audio
   and report false accepts per hour plus recall at a fixed false-accept rate.
2. Implement the packed 4-bit ABI and remeasure physical input size, BRAM use,
   Ethernet latency, and power.
3. Measure end-to-end detection delay from word onset, not only classifier
   execution time.
4. Sweep hidden width and SNN time steps jointly with the selected frontend.
