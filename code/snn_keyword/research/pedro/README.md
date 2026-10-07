# Research programme — how to run a session

> **Keyword.** Written for the keyword **"yes"** and the dense window model of the first
> release. The project's keyword is now **"sheila"**, detected against everything else by a
> streaming SNN (`../../README.md`, `../../JOURNAL.md` entries 22 onwards). The experiments
> below compare encodings and sizes of the window model; for "sheila" they would have to be
> repeated with `KWS_KEYWORD=sheila` (the default) and the streaming model.

This folder holds the Group 2 research study: how input encoding and network size trade
off against detection quality and latency on the PicoRV32 RISC-V core.

| File | What it is |
|---|---|
| [PLAN.md](PLAN.md) | The full plan: background primer, method rules, and experiments E0–E8 (each with What / How / Why / Expected outcome / steps). |
| [STATUS.md](STATUS.md) | Progress log: what is done, key numbers so far, open issues, next experiment. |
| [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) | The structure every experiment report follows. |
| `reports/` | One report per experiment (`E2_size_sweep.md`, …) with measured data. |
| `results/E<n>/` | Raw data of each experiment (JSON, CSV, plots, small integer models). |
| `tools/` | Shared scripts (built in E0): firmware build, board benchmark, model evaluation, recorder. |

## Starting a new Claude Code session

1. In VS Code, open the **`5XPG0`** folder (File → Open Folder…), so that the project
   brief `5XPG0/CLAUDE.md` is loaded automatically. Open a new Claude Code chat.
2. Make sure the PYNQ-Z2 is switched on and the Ethernet cable is connected.
3. Type, for example:

   > Do experiment **E2** from `code/snn_keyword/research/pedro/PLAN.md`. Read `STATUS.md` first.
   > Explain what you are going to do before you start.

That is all the context a session needs. It will:
- read the plan and the status;
- explain the experiment;
- do the work;
- write `reports/E2_*.md` and `results/E2/`;
- update `STATUS.md`;
- commit locally on branch `Pedro`.

It will not push to GitHub unless you say so.

If you want only part of an experiment ("only train sizes 8 and 16 today"), say so; the
session records what is left in `STATUS.md`.

## Rules every session follows
- Work on branch `Pedro` only; `Mateo_AI` is never touched; no push without asking.
- Validation data for choices, test data only for reporting, 3 seeds.
- Latency only from the real board, and only after a bit-exact check.
- The release demo (`deploy/`) stays as it is; restore it on the board at the end
  (`code/snn_keyword/run_board.ps1`).
