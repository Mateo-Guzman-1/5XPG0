# Spectrogram-encoding sweep report

> Historical Mini Speech Commands `yes` result, superseded for the deployed
> system by `SHEILA_V2_REPORT.md`.

## Conclusion

The best stable frontend is **20 frequency bands, 150–7600 Hz, 16 time bins,
and a 1.0-second window**. Use 8-bit features when prioritizing quality with
the current byte-per-feature interface. Use 4-bit features as the compact
Pareto point only after implementing packed transport/storage.

The selected 8-bit integer model reached **96.74% test accuracy** and
**86.60% F1** for its validation-best seed. It ran bit-exactly on the PYNQ-Z2
in **21.19 ms mean / 21.24 ms maximum**, well inside the 250 ms hop.

## Dataset and method

The sweep uses all 8,000 Mini Speech Commands clips. Speakers are disjoint:

| Split | Clips | “yes” | Other | Speakers |
|---|---:|---:|---:|---:|
| Train | 5,673 | 692 | 4,981 | 1,225 |
| Validation | 1,130 | 162 | 968 | 262 |
| Test | 1,197 | 146 | 1,051 | 263 |

Twenty-five one-factor configurations were screened with one fixed seed.
Validation-selected values and nearby trade-off points were then trained with
three seeds. The test split was used only in this confirmation stage.

## Feature precision

All rows use 16 bands, 40–5120 Hz and one second.

| Bits | Packed frame | Test accuracy | Test F1 |
|---:|---:|---:|---:|
| 3 | 96 B | 94.15% ± 0.22 pp | 76.65% ± 0.90 pp |
| 4 | 128 B | 94.99% ± 0.29 pp | 78.72% ± 0.75 pp |
| 6 | 192 B | 95.43% ± 0.41 pp | 80.30% ± 1.27 pp |
| 8 | 256 B | 95.15% ± 0.38 pp | 79.80% ± 0.88 pp |

Two-bit features were rejected during screening: validation accuracy was
90.53% and F1 was 70.36%. Quality rises quickly through four bits and then
saturates; six and eight bits are indistinguishable at this experiment’s
scale.

**Ideal:** 8 bits for the existing ABI and quality-first configuration. Four
bits is the practical packed-memory knee: it halves the feature frame, while
the confirmed accuracy loss at the baseline frontend is only 0.17 percentage
points relative to 8 bits. Precision does not change MAC count or weight
memory.

## Number of frequency bands

All rows use 8-bit features, 40–5120 Hz and one second.

| Bands | Inputs | int8 model parameters | Input MACs | Test accuracy | Test F1 |
|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 6,440 B | 6,144 | 93.98% ± 0.22 pp | 75.04% ± 1.11 pp |
| 16 | 256 | 12,584 B | 12,288 | 95.15% ± 0.38 pp | 79.80% ± 0.88 pp |
| 20 | 320 | 15,656 B | 15,360 | 95.24% ± 0.25 pp | 80.76% ± 0.65 pp |
| 32 | 512 | 24,872 B | 24,576 | 94.96% ± 0.19 pp | 79.89% ± 0.66 pp |

Twelve and 24 bands were also screened. Neither displaced the confirmed
20-band knee. More bands help from 8 to 20, then stop helping while memory and
latency continue to rise.

**Ideal:** 20 bands for quality; 16 bands remains the smaller Pareto point.

For `B` bands, 16 time bins and 48 hidden neurons:

```text
inputs                  = 16 B
first-layer MACs        = 768 B
learned parameter bytes = 768 B + 296
byte-aligned frame      = 16 B bytes
packed q-bit frame      = 2 q B bytes
```

Physical measurements at 16 and 20 bands give the local latency relation:

```text
mean cycles ≈ 300,110 + 90,940 B
mean ms     ≈ 3.001 + 0.9094 B       (100 MHz)
```

This relation is specific to the current firmware, compiler and 48-neuron
topology, but it confirms the expected linear dependence on band count.

## Filtered frequency range

With 16 bands and a one-second window, ranges ending at 3.4–4.0 kHz produced
only 72.8–73.9% validation F1. Extending the high cutoff preserved the
high-frequency `/s/` ending of “yes”:

| Range | Confirmed validation F1 | Test accuracy | Test F1 |
|---|---:|---:|---:|
| 40–5120 Hz | 79.84% | 95.15% ± 0.38 pp | 79.80% ± 0.88 pp |
| 40–6500 Hz | 83.25% | 96.18% ± 0.27 pp | 84.28% ± 0.78 pp |
| 40–7600 Hz | 82.26% | 96.32% ± 0.22 pp | 84.42% ± 0.96 pp |
| 150–7600 Hz | **83.47%** | 96.05% ± 0.51 pp | 84.13% ± 1.87 pp |

The expanded single-seed grid showed that raising the low cutoff from 40 Hz
to 150 Hz helped slightly, while 300–500 Hz began removing useful speech
content. The differences among the three wide ranges are modest, but
150–7600 Hz also interacted best with 20 bands.

**Ideal:** 150–7600 Hz. Frequency range changes only PC feature extraction;
with a fixed band count it does not change model memory or PicoRV32 latency.

## Audio time range

All rows retain 16 time bins, so shorter windows have finer time resolution.

| Duration | Resolution per time bin | Test accuracy | Test F1 |
|---:|---:|---:|---:|
| 600 ms | 37.5 ms | 94.49% ± 0.36 pp | 77.05% ± 0.59 pp |
| 1000 ms | 62.5 ms | **95.15% ± 0.38 pp** | **79.80% ± 0.88 pp** |
| 1200 ms | 75.0 ms | 94.63% ± 0.75 pp | 77.91% ± 1.71 pp |

The 800 ms screening point was also below one second. The result is an
inverted-U relationship: 600–800 ms can crop contextual parts of an isolated
word, while 1.2 seconds adds silence and coarsens each fixed time bin.

**Ideal:** 1,000 ms. Duration does not change board memory or inference
latency while the time-bin count stays at 16.

## Combined configurations

| Configuration | Three-seed test accuracy | Three-seed test F1 | Model bytes | Deployed input |
|---|---:|---:|---:|---:|
| Baseline: 8 bit, 16 bands, 40–5120 Hz, 1.0 s | 95.15% ± 0.38 pp | 79.80% ± 0.88 pp | 12,584 | 256 B |
| Quality: 8 bit, 20 bands, 150–7600 Hz, 1.0 s | 96.24% ± 0.74 pp | 84.92% ± 2.38 pp | 15,656 | 320 B |
| Compact: 4 bit, 20 bands, 150–7600 Hz, 1.0 s | **96.60% ± 0.21 pp** | **85.16% ± 0.79 pp** | 15,656 | 320 B (160 B packed) |
| Oversized interaction: 8 bit, 32 bands, 150–7600 Hz, 1.2 s | 96.66% ± 0.22 pp | 85.66% ± 1.26 pp | 24,872 | 512 B |

The oversized interaction has similar test quality but lower confirmed
validation F1 than the 20-band quality model, and it is about 59% larger in
parameters. It is therefore not selected. The compact model’s slightly
higher test mean was not used to select it; its validation F1 was 1.01
percentage points below the quality model.

After int8 weight export and validation-only threshold retuning, the selected
seed-1 models scored:

| Export | Test accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Quality 8-bit | 96.74% | 86.90% | 86.30% | 86.60% |
| Compact 4-bit | 96.83% | 94.26% | 78.77% | 85.82% |

The compact model trades recall for precision. This matters for a keyword
detector even though its raw accuracy is slightly higher.

## Physical PYNQ results

| Firmware | Vectors | Mean cycles | Maximum cycles | Mean latency | Maximum latency | Mismatches |
|---|---:|---:|---:|---:|---:|---:|
| Active 16-band baseline | 320 | 1,755,150 | 1,760,898 | 17.551 ms | 17.609 ms | 0 |
| Quality 8-bit / 20-band | 40 | 2,118,909 | 2,124,045 | 21.189 ms | 21.240 ms | 0 |
| Compact 4-bit / 20-band | 40 | 2,119,530 | 2,126,100 | 21.195 ms | 21.261 ms | 0 |

All measurements were made over Ethernet on the PYNQ-Z2 at `192.168.2.99`
with the PicoRV32 at 100 MHz. The candidate firmware text grew from 14,324 to
17,400 bytes, almost entirely due to the extra first-layer weights. Both
candidates remain far inside the 250 ms hop.

Four-bit precision does not reduce measured latency because the deployed ABI
expands every feature to uint8 before inference. Packing would reduce the
320-byte input frame to 160 bytes but requires an unpacking implementation and
a new board timing measurement.

## Limitations

- This is an eight-word mini-dataset result, not the earlier full
  Speech Commands v0.02 result.
- The audio consists of centred isolated clips, not continuous live speech.
- Background-noise and near-silence robustness were not swept here.
- The broad sweep used one seed; only selected and nearby configurations were
  confirmed with three seeds.
- The local latency formula has only two measured band-count points.
- Packed-input memory is a design projection, not the current implementation.

Raw results are in `results/frontend_sweep.json` and
`results/frontend_board_2026-09-28.json`.
