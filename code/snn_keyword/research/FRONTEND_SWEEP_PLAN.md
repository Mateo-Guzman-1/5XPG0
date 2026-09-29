# Spectrogram-encoding sweep plan

> Historical Mini Speech Commands `yes` sweep. The v2 Sheila replacement is
> specified in `SHEILA_V2_PLAN.md` and reported in `SHEILA_V2_REPORT.md`.

## Question

Which spectrogram representation gives the best accuracy, PicoRV32 latency,
and memory trade-off for the Group-2 “yes” detector?

In this study, “amount of levels” means the number of frequency bands. The
time range means the audio duration represented by one spectrogram. The
number of time bins remains fixed at 16 so the effect of coverage versus
temporal resolution can be isolated.

## Variables

| Variable | Values screened |
|---|---|
| Feature precision | 2, 3, 4, 6, 8 bits |
| Frequency bands | 8, 12, 16, 20, 24, 32 |
| Low frequency cutoff | 40, 150, 300, 500 Hz |
| High frequency cutoff | 3,400, 4,000, 5,120, 6,500, 7,600 Hz |
| Audio duration | 600, 800, 1,000, 1,200 ms |

The frequency sweep is not a complete Cartesian product. It first covers
representative ranges and then expands the promising 6.5/7.6 kHz region
across the low cutoffs.

## Fixed controls

- Mini Speech Commands: all 8,000 clips from eight commands.
- Positive class “yes”; all other words are negative.
- Speaker-disjoint 70/15/15 split: 5,673 train, 1,130 validation and 1,197
  test clips.
- Current encoding; 48 hidden LIF neurons; 16 LIF steps.
- Sixteen spectrogram time bins.
- 25 ms FFT window and 10 ms hop.
- Identical optimizer, balanced training sampler and training budget.
- Validation F1 selects values; test results are not used for selection.

## Procedure

1. Screen every value using seed 0 for eight epochs.
2. Confirm the baseline, validation winners and nearby Pareto candidates using
   seeds 0, 1 and 2 for 12 epochs.
3. Confirm interaction candidates rather than assuming independent winners
   combine cleanly.
4. Export the validation-best seed to the exact int8/int32 recurrence.
5. Tune its integer decision threshold on validation data.
6. Evaluate the exported model once on the test split.
7. Replay 40 held-out vectors on the Ethernet-connected PYNQ-Z2 and require
   exact equality with the integer oracle.

## Metrics

- Accuracy is the requested primary quality metric. F1, precision and recall
  are retained because the binary dataset is imbalanced.
- Latency is measured with the PicoRV32 cycle counter at 100 MHz.
- Memory includes learned parameters, linked firmware text, the byte-aligned
  input buffer, and the theoretical packed input size.

## Selection rule

The quality configuration maximizes mean validation F1 after three-seed
confirmation. A compact configuration may be recommended when its quality
loss is small and it reduces realizable memory or transport cost.

Lower feature precision only reduces memory if features are packed. The
current board ABI uses one byte per feature, so it is reported both as
deployed memory and as a packed lower bound.

## Expected relationships

- Precision should show diminishing returns once the quantization step is
  smaller than relevant spectrogram differences.
- Parameter memory and first-layer latency should grow linearly with the
  number of bands.
- Frequency range and audio duration should affect accuracy but not board
  inference cost when band/time-bin counts stay fixed.
- Duration should have an intermediate optimum: too short clips the word;
  too long reduces time resolution and introduces more silence.
