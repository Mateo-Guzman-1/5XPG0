# Hand-off: alternative-methods track (second PC)

> **State when written: keyword "yes".** The project's keyword is now **"sheila"**
> (JOURNAL entry 22); the tracks below apply to it unchanged (`KWS_KEYWORD` is
> "sheila" by default), but their "yes" wording and the /s/ arguments refer to the
> earlier keyword. Compare on the same rule (JOURNAL 19, 26).

For a Claude Code session on the second PC (RX 6700 XT 12 GB, i5-12400F,
32 GB). It runs in parallel with the main session on the RTX 4060 PC, which
keeps the streaming SNN line and all board and JTAG work.

## Goal

Find out whether the **spiking architecture** or the **data and front end**
limits the "yes" detector, using methods the main line does not try.
Report results so that the main session can reproduce them.

1. **Non-spiking streaming reference.** A causal streaming model on the same
   24-band log-mel frames, the same data (`multicorpus.ClipSampler`,
   `train_stream.build_streams`) and the same stage-2 loss (`stream_loss`).
   Candidates: a small GRU (1-2 × 64-128 units) or a causal depthwise CNN.
   Keep it under about 60k parameters, like the SNN, and add one larger
   version as a ceiling. Output per 10 ms frame: the same score, yes minus
   the largest other class.
2. **Front end for the /s/.** "Yes" is mostly recognised by its /s/
   (4-8 kHz). Try more bands above 4 kHz, or a high/low band energy ratio.
   `features.frame_features` and `augment_online.FrontEnd` must stay equal
   (see `tests/test_pipeline.py`).
3. **Cut-off /s/ in training data.** 11% of Speech Commands "yes" clips end
   inside the /s/ (`robust_eval.edge_clipped`). Test whether dropping them
   from the positives, or not counting them as positives, helps.

## Rules (do not break)

- **Validation data only** for every choice. Do not run anything on the
  test split (`robust_eval.py --split test`, the default). The main session
  runs the test once per final candidate.
- **Operating point:** `stream_select.py` with the defaults: live other-word
  accepts ≤ 0.2% and ≤ 2 false accepts per hour on the 17.85 h
  negatives-only stream (`--fa-source negatives`). Report `--window 1 10 20`.
  The 1 h stream alone is too short (JOURNAL entry 19).
- **Statistics:** live recall uses 397 positive clips (SE about 2.5 points).
  Compare models on the same clips with a paired test, and do not claim a
  difference under about 5 points without one.
- A non-spiking model needs a detector class with the `traces(audios)`
  interface of `snn_stream.StreamDetector` (returns times, decision score,
  raw score) so that `stream_select.py` and `robust_eval.py` can use it.
- **Do not change**, only add: `model.py` (integer oracle), `firmware/`,
  `sim/`, `jtag/`, `rtl/`, `deploy/`, `channels.py` held-out profiles, or
  the evaluation sets in `robust_eval.py`.
- Work on branch `alt-methods`. Write results to
  `results/alt_*.json` and notes as JOURNAL entries titled
  "Alt track: ...". Commit often; the main session reviews and merges.

## Setup

- Linux (Ubuntu 22.04/24.04). PyTorch with ROCm. The 6700 XT (gfx1031) needs
  `export HSA_OVERRIDE_GFX_VERSION=10.3.0`. Windows/WSL ROCm does not
  support this card; DirectML is too slow.
- `pip install -r requirements.txt` (without the CUDA torch build). The code
  uses `torch.device('cuda')`, which ROCm PyTorch provides.
- Copy `code/snn_keyword/data/` from the main PC (not in git): Speech
  Commands, `multi/`, `mswc/`, `librispeech/` (dev-clean, dev-other,
  test-clean, test-other, train-clean-100), `musan/`, `rirs/`, `tts/`, and
  `features.npz`.
- Smoke test first:
  `python train_stream.py --stage 2 --init <stage-1 .pt> --epochs 1 --steps-per-epoch 4 --batch 32 --name smoke --out build/smoke`
  Copy `runs_stream/s1_wide/model.pt` from the main PC for this.

## Current state (main line, validation)

| Model | Rule | Live recall at ≤ 2 FA/h |
|---|---|---|
| `s2_nokd_qat/int_last.npz` | raw | 55.9% |
| same | sum of 20 frames | 61.0% |

- Release window model, same rule: about 38%.
- False accepts: mostly continuous speech. Mining speech negatives moves them
  to isolated words (JOURNAL entries 19-20).
