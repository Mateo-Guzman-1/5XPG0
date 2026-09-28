# Data-encoding research plan

> Historical current-versus-rate plan for the earlier `yes` detector. The
> active Sheila frontend experiment is specified in `SHEILA_V2_PLAN.md` and
> reported in `SHEILA_V2_REPORT.md`.

The follow-up sweep of feature precision, frequency-band count, frequency
range and audio duration is specified in `FRONTEND_SWEEP_PLAN.md`. Its results
are reported separately in `FRONTEND_SWEEP_REPORT.md` so the mini-dataset
frontend study is not mixed with the full-dataset current/rate comparison.

## Research question

For a fixed keyword-spotting SNN on the 100 MHz PicoRV32, which input
encoding gives the best trade-off between held-out classification accuracy,
inference latency, and memory use?

The target is the word **yes**. An encoding is considered deployable only if
its maximum observed inference latency is below the 250 ms audio hop and its
complete firmware/model fits the 256 KiB RISC-V BRAM.

## What can be changed

The 24-by-32 uint8 log-mel representation is held fixed. The experiment
changes only the way those 768 values drive the first LIF layer:

1. **Direct current:** compute `W*x + b` once and apply the constant current
   at every LIF step. This preserves the feature magnitude directly and lets
   the processor cache the expensive input projection.
2. **Deterministic rate:** each input has a phase accumulator. It adds the
   uint8 feature value every step and emits a spike when the phase reaches
   255. The first-layer projection must be evaluated at every step.
3. **Time to first spike (future experiment):** a stronger feature emits one
   spike earlier. This needs at most one event per input, but represents zero
   and weak inputs carefully to avoid artificial events.
4. **Signed delta / ON-OFF events (future experiment):** encode changes
   between adjacent spectrogram time bins on separate positive and negative
   channels. This may emphasize phoneme transitions but doubles the logical
   input channels unless a packed representation is used.
5. **Hybrid current plus delta (optional):** keep absolute energy as current
   and add a sparse transition channel. This is relevant only if delta alone
   loses stationary vowel information.

Population coding and stochastic Poisson rate coding are excluded from the
first study. They increase input state or add sampling variance without a
clear benefit on this small processor.

## Hypotheses

- Rate encoding will have similar held-out accuracy to direct current, but
  substantially higher latency and working-memory use.
- Time-to-first-spike encoding may reduce event count relative to rate, but
  may lose accuracy because each input contributes at most once.
- Delta encoding may improve rejection of similar-sounding endings, but will
  use more model memory unless its input layer is made sparse or shared.

## Controls

The following stay fixed when encodings are compared:

- Speech Commands v0.02 official speaker-disjoint train/validation/test split;
- 24 mel bands, 32 time bins, one-second windows and uint8 transport;
- 64 hidden LIF neurons, 12 simulation steps, beta 7/8 and subtractive reset;
- two-class accumulated nonspiking readout;
- int16 Q10 weights and int32 accumulators;
- optimizer, training budget, sampling policy and augmentation;
- three random seeds per encoding;
- threshold selection using validation data only;
- the unaccelerated RV32IM bitstream for the primary latency comparison.

The `kdot` current-encoding result is reported separately as a co-design
result, because using a hardware accelerator in only one condition would not
be a fair encoding comparison.

## Measurements

### Accuracy

Report test accuracy for the three independently trained seeds and their mean
and sample standard deviation. Because only 419 of the 11,005 test clips are
positive, also retain precision, recall, F1 and the confusion matrix so that
the high raw accuracy is not misinterpreted.

### Latency

Run the same 40-vector verification set on the physical PYNQ-Z2. Record
minimum, mean and maximum PicoRV32 cycle counts from the firmware timer and
convert them at the measured 100 MHz fabric clock. The set includes positive,
negative, threshold-boundary, repeated, zero, saturated and random inputs.
These are observed extrema, not a formal worst-case execution-time proof.

### Memory

For every encoding record:

- learned parameter bytes;
- input-buffer bytes;
- linked `.text`, `.model` and `.bss` section sizes;
- compiler-generated stack-frame size of the inference function;
- whether the fixed linker regions and 16 KiB reserved stack still fit.

The flat binary file size is not used as the memory metric because it contains
padding between linked address regions.

## Execution sequence

1. Train direct-current and deterministic-rate models for seeds 0, 1 and 2.
2. Select thresholds and checkpoints using validation F1 only.
3. Evaluate each selected checkpoint once on the official test set.
4. Export integer models and verify NumPy against native C on all test clips.
5. Compile both encodings with the same RV32IM compiler and optimization flags.
6. Verify score and spike equality on the connected PYNQ-Z2.
7. Record cycle and memory measurements and apply the 250 ms feasibility gate.
8. Implement time-to-first-spike next. Continue to signed delta only if TTFS
   either improves the Pareto frontier or provides a useful failure result.

## Decision rule

An encoding is Pareto-optimal if no other encoding is at least as accurate,
no slower, and no larger, with a strict improvement in at least one metric.
For the live system, encodings above 250 ms maximum observed latency are
infeasible even if they have slightly higher accuracy.

## Reproducibility and limitations

The completed current/rate training evidence is retained on branch `Pedro`,
commit `05ad009ddf1796f7d94973a158fdaee6391f2fe2`. Physical measurements are
recorded in `results/board_encoding_2026-09-28.json` and summarized in
`REPORT.md`.

The active `damien-dicking-around` checkout contains a different 16-by-16,
int8 mini-dataset demo. It is useful as a live-demo baseline but must not be
used to regenerate the 24-by-32, int16 full-dataset numbers in this report.
The research implementation and retained model artifacts should be merged or
checked out from the commit above before extending the encoding experiment.

Three seeds estimate run-to-run variation but do not establish statistical
significance. Isolated-clip accuracy does not measure false accepts per hour,
end-to-end microphone latency, power, or robustness to a new room and
microphone. Those require separate experiments.
