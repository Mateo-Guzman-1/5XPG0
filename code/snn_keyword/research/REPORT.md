# Data-encoding research report

The subsequent spectrogram-parameter sweep is reported in
`FRONTEND_SWEEP_REPORT.md`. Its recommended frontend is 20 bands over
150–7600 Hz and one second, with 8-bit quality and 4-bit compact variants.
That experiment uses Mini Speech Commands and is therefore kept separate
from the full-dataset encoding results below.

## Outcome

Direct-current encoding is the deployable choice for this PicoRV32 system.
Deterministic rate encoding improved mean test accuracy by only **0.118
percentage points**, while taking **9.99 times** as many cycles on average,
using a 2,320-byte larger inference stack frame, and exceeding the 250 ms
streaming budget. Both encodings fit in BRAM and were bit-exact on the
physical PYNQ-Z2.

## Method

The comparison uses Speech Commands v0.02 with its official
speaker-disjoint split: 84,843 training, 9,981 validation and 11,005 test
clips. The network and frontend are fixed at 768 inputs, 64 LIF neurons, 12
steps, a two-value readout, int16 Q10 weights, and int32 state. Each encoding
was trained with seeds 0, 1 and 2. Validation data selected checkpoints and
decision thresholds; test results were not used for selection.

Direct current computes the input projection once and reuses it for all 12
steps. Deterministic rate encoding maintains a phase accumulator for each
input and recomputes a sparse first-layer projection at every step.

Physical timing was repeated on 28 September 2026 using the PYNQ-Z2 at
`192.168.2.99`, the baseline RV32IM fabric at 100 MHz, and GCC 14.2.0. The
same 40 verification vectors were used for both encodings. All returned
scores and hidden-spike counts matched the independent integer oracle.

## Accuracy

| Encoding | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Direct current | 99.128% | 98.801% | 99.082% | **99.003% ± 0.177 pp** |
| Deterministic rate | 99.109% | 99.146% | 99.109% | **99.122% ± 0.021 pp** |

The test set is strongly imbalanced, so accuracy alone looks better than the
keyword detector actually is. As a supporting measure, mean F1 was 86.67% ±
2.19 pp for current and 88.05% ± 0.33 pp for rate. Rate therefore has a small
quality advantage in this experiment, but not enough to offset its platform
cost.

## PicoRV32 latency

| Encoding / implementation | Mean cycles | Maximum cycles | Mean latency | Maximum latency | Oracle mismatches |
|---|---:|---:|---:|---:|---:|
| Direct current, RV32IM | 5,989,563 | 5,992,157 | **59.896 ms** | **59.922 ms** | 0/40 |
| Deterministic rate, RV32IM | 59,815,133 | 71,873,640 | **598.151 ms** | **718.736 ms** | 0/40 |
| Direct current, `kdot` accelerator | 264,799 | 267,393 | **2.648 ms** | **2.674 ms** | 0/40 |

The fair software-only comparison is the first two rows. Rate is 9.99× slower
on mean cycles and about 12.0× slower at the observed maximum. Its measured
maximum is almost three times the 250 ms hop, so it cannot keep up with the
live stream.

The `kdot` row is a separate hardware/software co-design result. It shows
that direct current is especially accelerator-friendly because its dense
projection is performed once. It must not be used to exaggerate the
encoding-only speed difference.

## Memory use

| Quantity | Direct current | Deterministic rate | Difference |
|---|---:|---:|---:|
| Mathematical learned parameters | 98,824 B | 98,824 B | 0 B |
| Linked `.model` section | 98,816 B | 98,816 B | 0 B |
| Input buffer | 768 B | 768 B | 0 B |
| Firmware `.text` | 1,124 B | 1,244 B | +120 B |
| Firmware `.bss` | 4 B | 4 B | 0 B |
| Inference stack frame | 656 B | 2,976 B | **+2,320 B** |
| Approx. active total of rows above | 101,368 B | 103,808 B | **+2,440 B** |

The eight-byte difference between mathematical parameter bytes and the
linked model section is caused by the compiler embedding the two readout
bias constants in code. Both variants use the same 98,816-byte linked model
section.

Rate encoding needs per-input phase state and event storage. This accounts
for nearly all of its additional stack use. Both versions still fit the fixed
16 KiB stack reservation and 256 KiB BRAM, so memory does not disqualify
either variant in the current topology.

## Interpretation

The hypothesis is supported. Rate encoding is marginally more accurate, but
its extra temporal representation is expensive on this scalar processor:
the 768-input projection is revisited during all 12 steps. Direct current
retains essentially the same detection quality, easily meets the real-time
constraint, uses less working memory, and maps cleanly to the existing
accelerator.

This result applies to this frontend, network, timestep count and processor.
It does not prove that rate encoding is generally inferior or that SNNs are
more efficient than nonspiking networks.

## Recommended next experiment

Implement **time-to-first-spike encoding** next, not another rate variant.
It is the most informative third point because it carries timing information
with at most one event per input. Use the same three seeds and controls, then
measure it on the board before examining test results.

If TTFS loses too much accuracy, test a signed ON/OFF delta encoding aimed at
the known problem of distinguishing the endings of “yes”, “yeets” and
similar words. Delta should be evaluated at both equal hidden width and equal
total model bytes, because doubling input channels otherwise gives it an
unfair memory advantage.

## Evidence provenance

- Training and held-out results: branch `Pedro`, commit
  `05ad009ddf1796f7d94973a158fdaee6391f2fe2`.
- Fresh physical measurements: `results/board_encoding_2026-09-28.json`.
- The active `damien-dicking-around` checkout is a separate 16-by-16, int8
  mini-dataset demo. Its model and frontend are not the source of the
  24-by-32, int16 research results above.
- The PYNQ measurements cover inference only. PC feature extraction,
  Ethernet transfer, audio-window collection and power were not measured.
