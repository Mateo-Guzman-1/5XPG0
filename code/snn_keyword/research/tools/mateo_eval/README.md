# Mateo's evaluation harness (vendored copy)

These are copies of Mateo's requirement tests from branch `origin/ALIF-Layer1`, commit `a856a03`, taken with `git show`. Using them means our window models are scored with exactly the same live set, false-accept stream and latency stream as Mateo's designs.

| File | Role | Added |
|---|---|---|
| `robust_eval.py` | live set, 1 h stream, about 19 h negatives stream, scoring | R1 |
| `final_eval.py`, `stream_select.py` | Mateo's one-time test report and validation threshold rule (kept for reference) | R1 |
| `keyword_config.py`, `channels.py` | keyword `sheila`, held-out microphone profiles | R1 |
| `fetch_corpora.py` | LibriSpeech and MSWC download (later: MUSAN, RIRs) | R1 |
| `verifier_cascade.py` | cascade evaluation: `cache`, `score`, `select`, `final` (stage 1 proposes, GRU verifier confirms) | R2 |
| `verifier_model.py`, `verifier_export.py`, `verifier_data.py` | GRU verifier, its integer model and int8 export; phoneme tables | R2 |
| `memguard.py` | memory guard used by the cascade tools (needs `psutil`) | R2 |
| `features.py`, `model.py` | Mateo's versions: they add the 10 ms frame front end (`frame_features`) and his streaming ALIF SNN (`integer_forward_stream`, `decision_scores`) | R2 |

**Changes to Mateo's files:**
1. `ROOT` points to `code/snn_keyword/` (`parents[3]`) instead of the file's own folder, so data is read from `code/snn_keyword/data/` and the cascade caches go to `code/snn_keyword/data_verifier/` (git-ignored). Applied to `robust_eval.py`, `final_eval.py`, `stream_select.py`, `keyword_config.py`, `channels.py`, `fetch_corpora.py`, `verifier_cascade.py`, `verifier_data.py`, `verifier_export.py`.
2. `verifier_cascade.prefixed_extras` (R2): when `data/multi_sheila/manifest.csv` does not exist (no multi-corpus build), the keyword-prefixed clips are an empty set. The LibriSpeech part is still searched. For "sheila" both sets are empty anyway ("0/0" in Mateo's results).

**Which `features` / `model` is used:**
- R1's `req_eval.py` puts `code/snn_keyword/` first on `sys.path`, so it uses Pedro's `features.py` and `model.py` (board-exact `integer_forward`, LIF table, n-bit and pruned models).
- Scripts run from this folder (`verifier_cascade.py`) and R2's `r2_cascade.py` put this folder first, so they use Mateo's versions.
- Both are safe: Mateo's window `features()` and `integer_forward` equal Pedro's bit for bit on the R1 live caches and the H48 K64 models (gate G-A1, `research/results/R2/gate_A1.json`).

**Data** (all git-ignored, under `code/snn_keyword/`):
- LibriSpeech test-clean, test-other, dev-clean and dev-other in `data/librispeech/`: `python fetch_corpora.py librispeech --subset test-clean` (and the other three).
- MSWC test and dev selections in `data/mswc/`: `KWS_KEYWORD=sheila python fetch_corpora.py mswc --split test --needed-only` (and `--split dev`). 3,634 clips each.
- `soundfile`, `scipy` and `psutil` must be installed in `.venv`.

**Used by:** `research/tools/req_eval.py` (R1, `research/PLAN_R1.md`); `research/tools/r2_cascade.py` (R2, `research/PLAN_R2_GRU.md`).
