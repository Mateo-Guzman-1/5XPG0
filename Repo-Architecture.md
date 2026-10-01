# Repository architecture

This document sets the target organisation of the Group 2 repository:
- where each block of `system_description.md` lives;
- which interfaces must stay consistent;
- what goes into git;
- how the code is checked, branched and released;
- how to get from today's layout to the target without losing the verified
  results.

**Scope.** Group 2's "sheila" keyword detector on the PYNQ-Z2. Group 4's MNIST
benchmark (`code/ann_vs_snn_mnist`) is not part of Group 2's branches; its skeleton
stays on `main`, the course template.

## 1. Goals

The layout serves the goals of the system description:

| Goal | Meaning for the repository |
|---|---|
| G1 Traceable blocks | Each block B0-B12 has one home. A reader finds the RTL, firmware, host and model code of a block without searching. |
| G2 One arithmetic | The Python integer oracle, native C, RTL and board stay bit-exact. Every interface between blocks has one definition and one check (§4). |
| G3 Reproducible numbers | Every reported number comes from a tracked script and a tracked results file. Data and checkpoints are rebuilt from scripts, not stored. |
| G4 A shippable release | The bitstream, firmware, model and thresholds that run on the board form one versioned bundle (§5). |
| G5 Product apart from history | The product path (the "sheila" cascade) is separate from the baselines (the "yes" window release, the clip pipeline) and from research. |
| G6 Course deliverables | Report, code and presentation; the code is submitted on branch `group2_2026` (course rule). |

## 2. Today's layout and its problems

```
README.md, MERGE_SUMMARY.md, system_description.md, Repo-Architecture.md
board_run.sh, run_board_demo.bat            one-click demo (board and PC side)
PresentationInstruction/, ProjectPitch/     course material
code/
  pynqz2_riscv_flow/    the course's bring-up flow, extended by Group 2
    rtl/                SoC RTL, including Group 2's kdot and neuron engine
    rtl_sketches/       the 64-neuron layer sketch
    firmware/, host/    course demo, the clip pipeline's board files, the layer smoke test
    vivado/             build script, constraints, board files, course bitstream
  snn_keyword/          everything else of the keyword detector
    54 Python scripts and 8 Markdown documents in one folder
    firmware/, sim/, jtag/, tests/, deploy/, research/, results/
```

| Problem | Effect |
|---|---|
| `code/snn_keyword/` holds 54 scripts at one level. They mix data, training, evaluation, export, verification and deployment, of two pipelines and of the frozen "yes" release, plus flag-gated experiments that did not work. | Hard to find the product path. Near-identical names: `features.py` / `audio_features.py`, `train_keyword_snn.py` / `train_keyword_snn_clip.py`. |
| The hardware is split across two folders. The SoC RTL is in the course folder; the firmware, simulation and JTAG scripts are in `code/snn_keyword/`. The keyword firmware builds `start.S` from the course folder, and the RTL scripts point at `../pynqz2_riscv_flow/rtl`. | Blocks B5-B8 have no single home. Paths cross folders. |
| The board release in `code/snn_keyword/deploy/` is the "yes" window model. The "sheila" bitstreams and firmware exist only in the ignored `build/` folder on one PC. | The current system cannot be redeployed from git. |
| There are two sets of board scripts: Alex's `run_board_demo.*` and `board_run.sh`, and Pedro's `run_board.ps1` and `start_board.sh`. | Two ways to do one job. Neither deploys the "sheila" cascade. |
| The documentation is spread over 8 documents in `code/snn_keyword/` plus the root. Five were written for "yes" and carry a banner. | Readers have to know the history to know what is current. |
| The checks are one pytest file (16 tests) and shell scripts in `sim/`. Nothing runs automatically. | Regressions are found only when someone runs the scripts by hand. |

## 3. Target layout

Each top-level folder is one layer of the system. The blocks it holds are in
brackets.

```
README.md                  entry point: task, status, documentation map, quick start
system_description.md      the system block by block, with alternatives
Repo-Architecture.md       this document
pyproject.toml             the kws package and its pinned dependencies (pip install -e .)
docs/
  design/                  OPTIMIZATION_REPORT.md, IMPLEMENTATION_PLAN.md, decisions/NNN-title.md
  journal/JOURNAL.md       dated log, append-only
  history/                 MERGE_SUMMARY.md, HANDOFF_ALT_METHODS.md, EXPERIMENT_PLAN.md, pedro/
  report/                  the "sheila" report and presentation (course deliverables)
  course/                  kick-off slides and project brief
hw/                        [B5-B8, B11] everything in the fabric
  rtl/                     spike_top.v, ps_if.v, spike_soc.v, kdot_pcpi.v, neuron_engine.v, poisson.v
  vendor/picorv32.v        unmodified, with its upstream version noted
  sketches/                snn_layer.v and its smoke test (not built)
  vivado/                  build.tcl, spike_top.xdc, board_files/
  sim/                     Verilator and Icarus benches with their scripts
firmware/                  [B6, B7, B9, B10] PicoRV32 C: start.S, linker.ld, runtime.c,
                           stream_main.c, stream_infer.c, verifier.c, one Makefile
host/                      [B1-B4] everything outside the fabric at run time
  pc/                      pc_keyword_demo.py
  board/                   board_server.py, start_board.sh
  jtag/                    bringup.tcl, stream_board_test.tcl, jtag_server.py, jtag_server.tcl
  deploy.ps1, deploy.sh    one deployment script for every board and image
kws/                       [B0, B2, B3, B12] the Python package
  config.py, frontend.py, protocol.py, oracle.py
  data/, models/, train/, eval/, export/, verify/
tests/                     pytest, mirroring kws/; also starts the simulation unit benches
results/                   evidence: *.json, *.rpt, models/ (deployable int8 models)
release/sheila-v1/         the shipped bundle (§5)
baselines/
  yes-window/              the first release (today's deploy/), its pipeline, report and firmware; frozen
  clip-pipeline/           the 1 s clip classifier over ZeroMQ, with its research and board files
  bringup-demo/            the course's one-neuron demo
research/                  open explorations: plans, sweeps, their results
.github/workflows/ci.yml   the automatic checks (§6)
```

Why this split:
- **`hw/`, `firmware/`, `host/` and `kws/` follow the system boundaries.** A block's
  interface is a folder boundary (§4), which keeps the dependencies visible.
- **`kws/` becomes an installable package.** Imports such as
  `from kws.frontend import frame_features` work from every folder: tests, host,
  research. Scripts run as modules: `python -m kws.train.stream`.
- **`baselines/` keeps finished systems runnable but out of the way.** Each one is
  frozen with its own README. The product path no longer depends on them.
- **`release/` holds only what has passed the board test.** `build/` stays local
  and disposable.

### Where today's files go

All paths below are relative to `code/snn_keyword/` unless they start with
`code/` or are at the root.

| Today | Target |
|---|---|
| `code/pynqz2_riscv_flow/rtl/*.v` (not `picorv32.v`) | `hw/rtl/` |
| `code/pynqz2_riscv_flow/rtl/picorv32.v` | `hw/vendor/` |
| `code/pynqz2_riscv_flow/rtl_sketches/snn_layer.v`; `code/pynqz2_riscv_flow/firmware/snn_smoke.c`, `snn_vectors.h`, `build_smoke.sh`; `code/pynqz2_riscv_flow/host/snn_smoke_test.sh` | `hw/sketches/` |
| `code/pynqz2_riscv_flow/vivado/` (not `spike_top.bit`) | `hw/vivado/` |
| `sim/` | `hw/sim/` |
| `firmware/stream_*`, `verifier.*`, `runtime.c`, `linker.ld`, `Makefile`; `code/pynqz2_riscv_flow/firmware/start.S` | `firmware/` |
| `pc_keyword_demo.py` | `host/pc/` |
| `board_server.py`, `start_board.sh` | `host/board/` |
| `jtag/`, `jtag_server.py`, `jtag_server.tcl` | `host/jtag/` |
| `run_board.ps1`, `run_board_demo.ps1`; root `run_board_demo.bat`, `board_run.sh` | `host/deploy.ps1`, `host/deploy.sh` (one script) |
| `keyword_config.py`, `features.py`, `protocol.py`, `model.py`, `memguard.py` | `kws/config.py`, `kws/frontend.py`, `kws/protocol.py`, `kws/oracle.py`, `kws/memguard.py` |
| `fetch_corpora.py`, `prepare_multicorpus.py`, `multicorpus.py`, `augment_online.py`, `channels.py`, `make_tts_negatives.py`, `verifier_data.py`, `mine_verifier_negatives.py` | `kws/data/` |
| `snn_stream.py`, `verifier_model.py` | `kws/models/` |
| `train_stream.py`, `train_verifier.py`, `quantize_stream_model.py` | `kws/train/` |
| `robust_eval.py`, `stream_select.py`, `final_eval.py`, `verifier_cascade.py`, `tts_sentence_probe.py`, `diagnose_stream.py`, `band_sensitivity.py`, `ablation_report.py` | `kws/eval/` |
| `export_model.py`, `verifier_export.py`, `package_release.py` (generalised to the "sheila" bundle) | `kws/export/` |
| `verify_stream.py`, `verify_stream_rtl.py`, `verify_verifier_rtl.py`, `net_board_test.py` | `kws/verify/` |
| `tests/` | `tests/` |
| `results/` | `results/` |
| `deploy/`; `firmware/main.c`, `inference.*`; `prepare_data.py`, `train_keyword_snn.py`, `evaluate.py`, `select_deployment.py`, `verify.py`, `tune_stream.py`, `confusables.py`, `robustness.py`, `make_report.py`, `augment.py`, `make_tts_probe.ps1`, `train_dense_online.py`; `REPORT.md`, `presentation.pdf` | `baselines/yes-window/` |
| `audio_features.py`, `train_keyword_snn_clip.py`, `pc_keyword_demo_zmq.py`, `install_keyword_demo.sh`, `README_clip_pipeline.md`, `damien_detector.py`, `research/` (not `pedro/`); `code/pynqz2_riscv_flow/firmware/keyword_*`, `code/pynqz2_riscv_flow/host/keyword_*` | `baselines/clip-pipeline/` |
| `code/pynqz2_riscv_flow/firmware/main.c`, `board.h`, `hal.h`, `spike.bin`; `code/pynqz2_riscv_flow/host/spike_pynq.py`, `demo.sh`, `smoke_test.sh`; `install.sh`, `run_demo.sh`, `Makefile`; `vivado/spike_top.bit` | `baselines/bringup-demo/` |
| `JOURNAL.md` | `docs/journal/` |
| `OPTIMIZATION_REPORT.md`, `IMPLEMENTATION_PLAN.md` | `docs/design/` |
| `MERGE_SUMMARY.md`, `HANDOFF_ALT_METHODS.md`, `EXPERIMENT_PLAN.md`, `research/pedro/` | `docs/history/` |
| `PresentationInstruction/`, `ProjectPitch/` | `docs/course/` |
| `teacher.py` and distillation, `mic_diagnose.py`, `cascade_eval.py`, `wait_and_train.sh`, and the flag-gated experiments listed under "Delete or archive" in `MERGE_SUMMARY.md` | deleted after the report; the journal keeps their results |

## 4. Interfaces that must stay consistent

Each interface between blocks has one defining file and one check. A change to
an interface changes every implementation in the same commit and runs the
check.

| Interface | Defined in | Also implemented in | Checked by |
|---|---|---|---|
| Feature frame: 24 uint8 per 10 ms, window, mel bank, dB range | `features.py` | the training front end (`augment_online.FrontEnd`), firmware (`STREAM_BANDS`), the PC client | `tests/test_pipeline.py`: offline frames, and live framing equal to offline |
| Stage-1 integer arithmetic | `model.py` (`integer_forward_stream`) | `firmware/stream_infer.c`, `rtl/neuron_engine.v` | `verify_stream.py` (C), `verify_stream_rtl.py --engine` (RTL), `jtag/stream_board_test.tcl` (board) |
| Verifier integer arithmetic | `verifier_model.py` (`integer_forward`) | `firmware/verifier.c` | `verify_verifier_rtl.py --cascade` |
| Host protocol, ABI v2 and v3 | `protocol.py` | `board_server.py`, `jtag_server.py` and `.tcl`, the mailbox of `firmware/stream_main.c` | pytest (framing, limits, refused ABI); the RTL harness (malformed requests) |
| Mailbox words | the header comment of `firmware/stream_main.c` | `board_server.py`, the JTAG scripts, `sim/soc_*.cpp` | the RTL harness; pytest (`test_board_mailbox_layout_and_sequence_wrap`) |
| Memory map, syscon, ABI version | `rtl/spike_soc.v`, `rtl/ps_if.v` | `firmware/linker.ld`, `board_server.py`, the JTAG scripts | the magic and ABI version checked at load; `sim/tb_axi.sv` |
| Engine registers and table formats | the header comment of `rtl/neuron_engine.v` | `firmware/stream_infer.c` (`USE_ENGINE`), `export_model.py` | `sim/engine_tb.cpp`; the RTL streams |
| Model file: keyword, thresholds, arithmetic constants | `quantize_stream_model.py`, `verifier_export.py` | the detectors, which refuse a keyword mismatch | pytest (`test_the_project_keyword_is_sheila_and_yes_models_are_guarded`) |

**Target:** generate the hand-copied constants from one description, for example
`hw/regmap.yaml`, turned into a C header, a Python module and a Verilog include.
This covers the memory map, the mailbox words and the engine registers. The model
headers are already generated this way (`export_model.py`).

## 5. What goes into git

| In git | Not in git (rebuilt) | In a release bundle |
|---|---|---|
| Sources: Python, C, Verilog, Tcl, scripts | `data/`: the corpora, about 60 GB (`fetch_corpora.py`, `prepare_multicorpus.py`) | the bitstream |
| Documents | `runs_*/`: training checkpoints (`*.pt`) | the firmware image |
| Evidence: `results/*.json`, `*.rpt`, `*.csv` | `build/`: firmware, bitstreams, Vivado projects, simulator objects | the deployed models |
| Deployed models only, `results/models/`: the int8 stage-1 models and the verifier checkpoint, each under 0.25 MB | caches, virtual environments | `manifest.json`: keyword, thresholds t1 and t2, ABI version, git commit, tool versions, SHA-256 of every file |
| Released bundles, `release/` | tool output: `dfx_runtime.txt`, Vivado `.jou` and `.log`, `.Xil/` | timing and utilisation reports, and the board test log |

Rules:
- A results file records the command line and the git commit that produced it.
- A number in a document cites its results file or its JOURNAL entry.
- A release is tagged in git (`release-yes-v1` for today's `deploy/`,
  `release-sheila-v1` after the board test). It is never changed after the
  tag; a fix is a new release.

## 6. Checks and continuous integration

| Level | What it checks | Command today | When |
|---|---|---|---|
| Unit | front end, protocol, decision rules, keyword guard, board server | `pytest code/snn_keyword/tests` | CI, every push and pull request |
| Build | the firmware compiles with `-Werror`; the release firmware stays byte-identical | `make -C code/snn_keyword/firmware all stream` | CI |
| RTL unit | the AXI interface, the LED timer, the engine against its C++ model | `sim/unit.sh`, `sim/engine_tb.cpp` | CI |
| Oracle = C | native C on the test streams | `verify_stream.py` | before a merge into the integration branch |
| Oracle = RTL | 40 streams (stage 1); the cascade requests | `verify_stream_rtl.py --engine`, `verify_verifier_rtl.py --cascade` | before a release |
| Board | the same vectors on the PYNQ-Z2; LED pulse of 1 s | `jtag/stream_board_test.tcl`, `net_board_test.py` | before a release; the log goes into the bundle |
| Accuracy | the test split, once per candidate | `final_eval.py`, `verifier_cascade.py final` | once per candidate; the result goes to `results/` and JOURNAL |

CI runs the first three levels on GitHub Actions (Ubuntu, CPU PyTorch,
`gcc-riscv64-unknown-elf`, Icarus Verilog, Verilator). The slower levels stay
manual, and their logs are committed as evidence.

## 7. Branches, reviews and releases

| Branch | Role |
|---|---|
| `main` | The course template: both groups' skeletons and the course material. It stays the common base. Group 2 adds only `system_description.md` there. |
| `merge-all` | Group 2's integration branch. Every change reaches it through a pull request that passes CI. |
| Topic branches | One topic, branched from `merge-all`, short-lived, merged back by pull request. The merge of 8 long-lived branches (`MERGE_SUMMARY.md`) is the cost this avoids. |
| `group2_2026` | The submission branch (course rule), cut from `merge-all` at the end. |
| Tags `release-*` | The released bundles (§5). |

Shared branches are never force-pushed, and merged history is not rewritten.

## 8. Documentation

| Document | Answers | Rule |
|---|---|---|
| `README.md` | What is this, what state is it in, where do I start? | Short; links to the rest |
| `system_description.md` | What is the system, and why is each block built this way? | Updated when a block or a choice changes |
| `Repo-Architecture.md` | Where is everything, and what are the rules? | Updated with the layout |
| A README per top-level folder | How do I build, run and reproduce this part? | Commands that work as written |
| `JOURNAL.md` | What was tried, when, and with what result? | Append-only; negative results too |
| `IMPLEMENTATION_PLAN.md` | What are the targets and gates? | Current targets only |
| `docs/design/decisions/NNN-title.md` | One decision: its alternatives and why it was taken | One file per decision (for example "GRU verifier", "event-driven engine") |
| Report and presentation | The course deliverables | Numbers only from `results/` |

A fact lives in one place and the others link to it. A document written for an
earlier state moves to `docs/history/` instead of collecting banners.

## 9. Conventions

- **Keyword-neutral names in new code** (`keyword`, `kw_class`). `KWS_KEYWORD`
  selects the keyword. The historical `yes_*` names stay only where saved models
  need them.
- **Python:** package imports. Command-line tools are modules with `--help`. Seeds
  are explicit. Thresholds are selected on validation data only.
- **C:** `-Werror`. Generated headers go to `build/` and are never edited by hand.
- **Verilog:** one module per file. A header comment holds the register map and
  the reference model. Vendor code stays unmodified.
- **Results:** `results/<what>_<keyword>_<variant>.json`.
- **Line endings:** as set in `.gitattributes`.

## 10. Migration plan

The files move last. Every command in the journal and the READMEs uses today's
paths, and the report needs them stable. A move counts as done only after the
same checks as the merge (`MERGE_SUMMARY.md`, "Verified on merge-all") pass
again.

| Phase | When | Work | Done when |
|---|---|---|---|
| 0 | done (2026-10-01) | `system_description.md` and this document; the root README reordered around the "sheila" system; Group 4 removed from Group 2's documents; tracked Vivado tool output (`dfx_runtime.txt`) removed and ignored | — |
| 1 | now; no file moves | One deployment script (merges Alex's and Pedro's; it also deploys the engine bitstream and the cascade firmware; "Still to do", item 6). CI for unit, build and RTL unit. Tag `release-yes-v1`. | CI is green on `merge-all`; one script deploys both releases to both boards |
| 2 | after the "sheila" board test | `release/sheila-v1/` with its manifest and board log. `deploy/` and its pipeline move to `baselines/yes-window/`. `board_server.py` defaults to the "sheila" release. The clip pipeline moves to `baselines/clip-pipeline/`, together with Damien, because his scripts use relative paths. | The bundle redeploys from git on a clean PC and passes the board test |
| 3 | after the report, before `group2_2026` | The moves of §3, one `git mv` commit per area with no content changes, so history follows the files; then the imports, Makefiles, Vivado and simulation paths, and documents. The generated register map. The deletions of §3. The journal's next entry gives the path mapping. | pytest passes; the release firmware is byte-identical; RTL bit-exact on 40 streams (355 hops) and on the cascade (934 requests); the Vivado build meets timing |
