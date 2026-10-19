# Mateo's evaluation harness (vendored copy)

These are copies of Mateo's requirement tests from branch `origin/ALIF-Layer1`, commit `a856a03`, taken with `git show`. Using them means our window models are scored with exactly the same live set, false-accept stream and latency stream as Mateo's designs.

| File | Role |
|---|---|
| `robust_eval.py` | live set, 1 h stream, about 19 h negatives stream, scoring |
| `final_eval.py`, `stream_select.py` | Mateo's one-time test report and validation threshold rule (kept for reference) |
| `keyword_config.py`, `channels.py` | keyword `sheila`, held-out microphone profiles |
| `fetch_corpora.py` | LibriSpeech download |

**Only change:** `ROOT` points to `code/snn_keyword/` (`parents[3]`) instead of the file's own folder. This way the data is read from `code/snn_keyword/data/`. Callers put `code/snn_keyword/` on `sys.path` so that `features` and `model` resolve to Pedro's versions:
- window `features()` is identical on both branches;
- Pedro's `integer_forward` is the board-exact one, and also handles the LIF table, n-bit and pruned models.

**Data:**
- LibriSpeech test-clean, test-other, dev-clean and dev-other go into `code/snn_keyword/data/librispeech/` (git-ignored):
  `python fetch_corpora.py librispeech --subset test-clean` (and the same for the other three).
- `soundfile` must be installed in `.venv`.

**Used by:** `research/tools/req_eval.py` (experiment R1, `research/PLAN_R1.md`).
