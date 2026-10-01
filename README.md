# 5XPG0 Group 2 — detecting "sheila" with a spiking neural network on a RISC-V soft core

The Group 2 repository of the 5XPG0 Electronic Systems project. A **spiking
neural network** on a **PicoRV32 RISC-V soft core** on a **PYNQ-Z2** FPGA detects
one keyword, **"sheila"**, against everything else: all other words, running
speech, silence and noise. The PC computes the mel features from its microphone
and sends them to the board over Ethernet. LED0 lights for one second per
detection.

Kick-off slides: [`PresentationInstruction/main.pdf`](PresentationInstruction/main.pdf).
Project brief: [`ProjectPitch/5XPG0_Projects_ES_Group.pdf`](ProjectPitch/5XPG0_Projects_ES_Group.pdf).

---

## Status

- **Two stages.**
  - A streaming SNN proposes. Its second layer runs on an event-driven neuron
    engine in the fabric: 10.3 ms of compute per 250 ms of audio.
  - A small GRU verifier on the same core confirms. It runs only after a proposal,
    117 ms in the worst case.
- **Test split**, with thresholds fixed on validation data:
  - 86.8% of live "sheila" detected (92.0% of complete recordings), target ≥ 85%;
  - 1.42 false alarms per hour of speech, target ≤ 2;
  - 0.07% of other words accepted, target ≤ 0.2%.
- **Verified:** bit-exact from the Python integer model to the RTL; timing met at
  100 MHz.
- **Not yet run on a board for "sheila".** The board release in
  `code/snn_keyword/deploy/` is the first keyword, "yes".
- **Main open problem:** "sheila" said inside a sentence; 6% are detected in the
  middle of one.

## Documentation

| Read | For |
|---|---|
| [system_description.md](system_description.md) | The system block by block: components, inputs, outputs, alternatives |
| [Repo-Architecture.md](Repo-Architecture.md) | How this repository is organised, its rules, and the target layout |
| [code/snn_keyword/README.md](code/snn_keyword/README.md) | The results, and the commands that reproduce them |
| [code/snn_keyword/JOURNAL.md](code/snn_keyword/JOURNAL.md) | Every step, its motivation and its outcome (entries 1-21 are for "yes", from 22 for "sheila") |
| [code/snn_keyword/IMPLEMENTATION_PLAN.md](code/snn_keyword/IMPLEMENTATION_PLAN.md) | Phases and acceptance targets |
| [MERGE_SUMMARY.md](MERGE_SUMMARY.md) | The work on every branch, their merge into `merge-all`, and what is still to do |
| [code/pynqz2_riscv_flow/README.md](code/pynqz2_riscv_flow/README.md) | The hardware flow and the memory map |

---

## Repository layout

```
system_description.md     the system design: blocks and alternatives
Repo-Architecture.md      repository organisation and target layout
MERGE_SUMMARY.md          branch history and open items
board_run.sh              one-click demo, board side (copied by run_board_demo.ps1)
run_board_demo.bat        one-click demo, PC side (calls code/snn_keyword/run_board_demo.ps1)
code/
  pynqz2_riscv_flow/      hardware: the PicoRV32 SoC RTL (with kdot and the neuron
                          engine), the Vivado build, the course's bring-up demo
  snn_keyword/            the detector: training, evaluation, firmware, simulation,
                          JTAG and board tools, results
    deploy/               the frozen first release (the "yes" window model)
PresentationInstruction/  kick-off presentation (PDF)
ProjectPitch/             project descriptions (PDF)
```

[Repo-Architecture.md](Repo-Architecture.md) gives the layout this repository is
moving to, and the order of the moves.

---

## Prerequisites (once per board)

* A **PYNQ-Z2** with the stock **PYNQ 3.1** image and an Ethernet cable.
* **SSH key auth** for user `xilinx`: `ssh xilinx@<board-ip>` must work with
  no password.
* **Passwordless sudo** on the board. The stock credentials are
  **user `xilinx`, password `xilinx`**.
* PC and board on the same network; the board can reach the internet.

> The `xilinx` / `xilinx` default is fine on an isolated lab network. Do not
> expose the board to the open internet.

---

## Quick start

### Python environment

Windows (PowerShell):

```powershell
Set-Location code\snn_keyword
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks local scripts, run `Set-ExecutionPolicy -Scope Process
Bypass` once in that terminal; the change lasts only for that PowerShell
process. Pass `-Recreate` to the setup script to rebuild the environment.

Linux, WSL or Git Bash:

```bash
cd code/snn_keyword && ./setup_venv.sh && source .venv/bin/activate
```

GPU training needs CUDA PyTorch installed first; see "Local environment" in
[code/snn_keyword/README.md](code/snn_keyword/README.md).

### The "sheila" detector without a board

From `code/snn_keyword`, with the environment active, in two terminals:

```powershell
python board_server.py --simulate --model results/models/sheila_stream_int8.npz
python pc_keyword_demo.py 127.0.0.1                  # microphone; or --wav path/to/sheila.wav
```

The simulated board runs stage 1, the streaming SNN.

### On the board

* **The board release ("yes")**, over Ethernet: run
  `code/snn_keyword/run_board.ps1 -Board <board-ip>`, then
  `python code/snn_keyword/pc_keyword_demo.py <board-ip>`. Alternatively, use the
  one-click `run_board_demo.bat`. Details: "Run on the PYNQ board" in
  [code/snn_keyword/README.md](code/snn_keyword/README.md).
* **The "sheila" system** has its board test over JTAG. The commands are in
  "Stage 2: a verifier confirms the SNN's proposals" in the same README.

### The course's bring-up flow

```bash
cd code/pynqz2_riscv_flow
./install.sh <board-ip>     # deploy bitstream, firmware and host tools
./run_demo.sh <board-ip>    # load and run the one-neuron demo
```

The prebuilt `vivado/spike_top.bit` is the course's demo image. The keyword
system needs its own bitstream, built with Vivado 2025.2 (see "Build the FPGA
image" in [code/snn_keyword/README.md](code/snn_keyword/README.md)).

The FPGA deployment scripts and Makefiles need a Unix-like shell; on Windows,
use WSL or Git Bash. The Python tools run directly in PowerShell.

### A second pipeline

`code/snn_keyword` also holds an independent clip pipeline: 1 s windows of 8×16
features, deployed over ZeroMQ (`README_clip_pipeline.md`).
[MERGE_SUMMARY.md](MERGE_SUMMARY.md) compares the two pipelines under one rule.

---

## Formalities

* **You may use AI tools for everything**, and group coding is allowed, but
  **you are responsible for the results, not the AI**.
* Weekly meetings: **Monday 08:30–09:00** and **Thursday 13:30–14:00**.
* Deliverables: **report, code, presentation**.
* **Code submission:** at the end of the project, push the code to this repo
  on the branch **`group2_2026`**.
* Evaluation: **33%** research-question formulation, **33%** scientific
  execution (no logical gaps), **33%** engineering solutions (results).

Contact: Federico Corradi `f.corradi@tue.nl` · Ajeya Naithani `a.naithani@tue.nl`
