# E<n> — <title>

*Date:* YYYY-MM-DD · *Branch/commit:* Pedro @ <hash> · *Session wall time:* <h>

## 1. Question and hypothesis
What this experiment asks, in one or two sentences, and what we expected beforehand
(written before looking at the results).

## 2. What was changed and what was held constant
| Changed | Values |
|---|---|
| … | … |

Held constant: data split, training recipe (epochs, QAT, optimiser, samples/epoch),
frontend (24×32 mel), threshold selection on validation, seeds 0/1/2, … (list the
exceptions explicitly).

## 3. Setup
Exact commands, number of runs, training time per run, which board/firmware variants
were benchmarked, and any deviation from PLAN.md with the reason.

## 4. Results
Tables with the real numbers (mean ± sd over seeds where applicable):

| Config | Val F1 | Test F1 | Recall | FPR | Near-miss FA (onset / ending) | Bytes | Max cycles RV32IM | Max cycles kdot | Bit-exact |
|---|---|---|---|---|---|---|---|---|---|

Plots (saved in `results/E<n>/`, linked here).

## 5. Interpretation
What the numbers mean for the research question: which trade-off appears, which
platform limit is reached, what is surprising. Be explicit where differences are within
the seed spread.

## 6. Threats to validity
E.g. few seeds, single human speaker, two TTS voices, clip metrics ≠ false alarms per
hour, cycles depend on BRAM wait states, anything that was not measured.

## 7. Artefacts
Paths to the JSON results, integer models, plots and scripts used.

## 8. Consequences for the plan
What the next experiment should change or keep (e.g. "E3 uses sizes 16 and 64").
