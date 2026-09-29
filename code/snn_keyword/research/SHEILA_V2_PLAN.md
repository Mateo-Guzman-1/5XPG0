# Sheila Speech Commands v2 encoding plan

## Question

Which spectrogram encoding gives the best deployable trade-off among held-out
accuracy, measured PYNQ latency, and memory for detecting `sheila`?

## Dataset and controls

- Google Speech Commands v2 (`speech_commands_v0.02`).
- Keep all Sheila clips and deterministically sample negatives approximately
  evenly from the other 34 labels.
- Preserve the official speaker-disjoint validation and testing manifests.
- Never use test results for training, threshold selection, or candidate
  selection.
- Hold the SNN at 48 hidden neurons, two outputs, 16 LIF steps, Q8 leak, int8
  weights, and int32 state.
- Use a fixed training budget for all one-factor comparisons and confirm
  finalists with seeds 0, 1, and 2.

## Variables

1. Feature precision: 2, 3, 4, 6, and 8 bits, corresponding to 4, 8, 16, 64,
   and 256 amplitude levels.
2. Frequency bands: 8, 12, 16, 20, 24, and 32.
3. Frequency range: lower cutoffs 40-500 Hz and upper cutoffs 3.4-7.6 kHz,
   including the existing 40-5120 Hz baseline.
4. Represented time: 600, 800, 1000, and 1200 ms, always pooled into 16 time
   bins.

After the one-factor sweep, measure relevant interactions, especially 4-bit
precision with the lowest-memory band count.

## Metrics

### Accuracy

Record accuracy, precision, recall, F1, balanced accuracy, and the confusion
matrix. Select decision thresholds on validation F1. Report mean test results
over the three confirmation seeds and separately report the exact exported
integer model selected for deployment.

### Latency

For every candidate, record first-layer MAC count as a hardware-independent
scaling proxy. For the deployed candidate, replay 40 golden vectors on the
100 MHz PYNQ-Z2 and record minimum, mean, and maximum firmware cycles plus
Ethernet request/reply latency. Time host feature extraction separately.

### Memory

Record quantized parameter bytes, byte-aligned input bytes, theoretical packed
input bytes, classifier work arrays, and complete linked firmware size. Keep
logical packed memory separate from physical memory: bit depth provides no
real saving until the transport and firmware unpack packed features.

## Decision rule

A candidate is preferred when no alternative is more accurate, no slower,
and no larger, with at least one strict improvement. Accuracy differences
below about 0.1 percentage points are treated cautiously unless consistent
across seeds. The production choice must fit the current byte-aligned ABI and
must reproduce Python integer spike counts on the board.

## Robustness and verification

- Evaluate every official test clip after deterministic 10 dB and 5 dB SNR
  additive noise.
- Test at least 400 generated background/silence frames, including exact
  silence, and report false positives.
- Confirm live preprocessing quantizes identically to cached training
  preprocessing on representative WAVs.
- Send real positive and negative WAVs through the complete Ethernet/BRAM/SNN
  chain after deployment.

## Limitations

Synthetic noise is not a substitute for hours of recorded local audio. The
subset has all Sheila clips but not every negative clip. Isolated one-second
classification does not measure false accepts per hour, microphone capture
hop latency, power, or generalization to local speakers and rooms.
