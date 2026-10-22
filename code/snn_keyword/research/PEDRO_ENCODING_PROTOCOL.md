# Encoding evaluation with Pedro's protocol

## Scope fixed before results

Evaluate the union of the primary and extended encoding sweeps and the two
passband controls: 61 configurations, three matched seeds each. Keep the
48-hidden-neuron, two-output SNN, 16 simulation steps, original real Speech
Commands training subset, and 12 epochs of 4,096 class-balanced samples.
Recreate missing checkpoints with this same recipe. No synthetic-word data.

The source of the event protocol is Pedro commit
`5b37c59` (`research/tools/req_eval.py` and its vendored `mateo_eval/robust_eval.py`).
Use the original waveform generation and event scoring code without edits.
The encoding and integer model remain those of the compact encoding study.

## Evaluation

- Full official validation/test clip lists, ordered as Pedro's preparation.
- All 204/212 Sheila clips and 3,000 sampled other words in 2.25 seconds of
  background, with the same random placement seeds (validation 2, test 1).
- One-second requests every 250 ms. Current-window acceptance plus at least
  one of the two preceding windows; one-second hold-off between events.
- Validation negatives: all non-keyword validation words and LibriSpeech
  dev-clean/dev-other. Test negatives use test words and test-clean/test-other.
- Select maximum validation live recall subject to <=0.2% other-word accepts
  and <=2 false accepts/hour. Ties use the highest threshold, as Pedro does.
  Threshold 17 explicitly represents abstention if no detecting threshold
  satisfies the limits; report that outcome rather than dropping the model.
- Freeze thresholds before test scoring. Also evaluate the same checkpoints
  at their original validation clip-F1 thresholds to isolate the change in
  decision rule from the change in test audio.
- Use Pedro's one-hour stream for word-end-to-detection latency and recall.
  This is an offline acoustic latency, not a microphone-to-LED measurement.
- Report mean and sample SD across seeds, input bits, parameter bytes, and
  separate physical FPGA timings. Select encodings using validation only.

The report's cascade uses a 1.6 FA/h validation margin. This experiment uses
Pedro R1's standalone-detector limit of 2 FA/h; it does not evaluate the GRU.
Duration retains the original centre-crop/zero-pad operation inside a
one-second request. Windows above one second therefore add padding rather
than new acoustic context, an existing limitation of this particular sweep.

## Implementation checks

Shared FFT calculations must produce identical quantized feature values to
the original frontend for all 61 configurations. The fast integer simulation
must reproduce the original simulator on real, zero, full-scale and random
inputs. Preserve waveform seeds, checkpoint hashes, protocol revision and
per-window margins. Cache completed negative-stream chunks for restart.

## Commands

Run with the project Python environment. Dependencies additionally include
scipy, soundfile, numba and psutil.

```powershell
python research/prepare_pedro_evaluation.py --data E:/URE/5XPG0-encoding-eval/code/snn_keyword/data
python research/pedro_encoding_eval.py --pedro E:/URE/5XPG0-encoding-eval
```

The validation and test waveform partitions may also run concurrently with
`--split validation` and `--split test`; both retain the same model hashes,
waveform seeds and configuration order. After they finish, use
`--summarize-only` and run `research/summarize_pedro_encoding.py`.
`research/pedro_encoding_board.py` measures seed zero for all configurations,
backs up the running board installation, and restores its startup service.

Results are written beneath `research/results/pedro_encoding/`. The protocol
JSON records the full Pedro commit and configuration order. Model files and
raw margin caches are reproducibility artefacts; the concise report uses
the final summary, not intermediate partial results.
