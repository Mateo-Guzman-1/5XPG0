# 5XPG0 — RISC-V & Spiking Neural Networks on the PYNQ-Z2

The 5XPG0 repo for the Electronic Systems project.
Two groups, one platform: a **PicoRV32 RISC-V soft core** on a **PYNQ-Z2**
FPGA, used to run **spiking neural networks**.

Kick-off slides: [`PresentationInstruction/main.pdf`](PresentationInstruction/main.pdf)

---

## The two projects

### Group 2 — single-keyword detection
Build a spiking neural network that detects **one keyword**, deploy it on the
PYNQ-Z2 RISC-V core, and light an LED for one second when the keyword is
spoken into the PC microphone. The PC computes a spectrogram and sends it to
the board over Ethernet.

### Group 4 — benchmark ANN vs SNN on MNIST
Build a small **ANN** and a small **SNN** for MNIST, deploy **both** on the
RISC-V core, and benchmark which one wins and why. Extra: extend the RISC-V
instruction set to make the SNN more efficient.

Both projects have many open **design choices**: input encoding, network
topology, training method, and demo/benchmark setup. Those choices, and their
justification, are the project.

---

## Repository layout

```
code/
  pynqz2_riscv_flow/     # the hardware/software bring-up flow (both groups)
    rtl/                 #   PicoRV32, Poisson spike generator, SoC, PS interface
    firmware/            #   the demo C code: one LIF neuron
    vivado/              #   build scripts + the released bitstream
    host/                #   spike_pynq.py, demo/smoke scripts
    install.sh           #   deploy everything to the board
  snn_keyword/           # Group 2: training + deployable PC/board keyword demo
  ann_vs_snn_mnist/      # Group 4: ANN/SNN training + benchmark skeleton
PresentationInstruction/  # kick-off presentation (PDF)
ProjectPitch/             # original project description (PDF)
```

---

## Prerequisites (once per board)

* A **PYNQ-Z2** with the stock **PYNQ 3.1** image and an Ethernet cable.
* **SSH key auth** for user `xilinx` — `ssh xilinx@<board-ip>` must work with
  no password.
* **Passwordless sudo** on the board. The stock credentials are
  **user `xilinx`, password `xilinx`**.
* PC and board on the same network; the board can reach the internet.

> The `xilinx` / `xilinx` default is fine on an isolated lab network — do not
> expose the board to the open internet.

---

## Quick start — hardware flow

```bash
cd code/pynqz2_riscv_flow
./install.sh <board-ip>     # deploy bitstream, firmware and host tools
./run_demo.sh <board-ip>    # load + run the demo on the board
```

The flow ships a **prebuilt bitstream** (`vivado/spike_top.bit`); rebuilding
it from the RTL needs **Vivado 2024.1** (`make bitstream`).

See `code/pynqz2_riscv_flow/README.md` for the full details and memory map.

---

## Quick start — training skeletons

### Windows (PowerShell)

```powershell
# Group 2
Set-Location code\snn_keyword
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
python train_keyword_snn.py

# Deploy the exported model (Git Bash / MSYS2)
# ./install_keyword_demo.sh 192.168.2.99
# Then, in PowerShell: python pc_keyword_demo.py 192.168.2.99

# Group 4
Set-Location ..\ann_vs_snn_mnist
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
python train_mnist.py
python benchmark.py
```

If PowerShell blocks local scripts, run `Set-ExecutionPolicy -Scope Process
Bypass` once in that terminal. The policy change lasts only for the current
PowerShell process. Pass `-Recreate` to a setup script to rebuild its venv.

### Linux, WSL, or Git Bash

```bash
# Group 2
cd code/snn_keyword && ./setup_venv.sh && source .venv/bin/activate
python train_keyword_snn.py

# Group 4
cd code/ann_vs_snn_mnist && ./setup_venv.sh && source .venv/bin/activate
python train_mnist.py && python benchmark.py
```

The Group 2 trainer now provides a complete real-audio reference path and
defaults to 12 epochs; the Group 4 trainer remains a one-epoch starting
point. See the README in each folder.

The FPGA deployment scripts and Makefiles require a Unix-like shell. On a
Windows PC, use WSL or Git Bash for that hardware flow; the Python training
and benchmark tools run directly in PowerShell.

---

## Formalities

* **You may use AI tools for everything**, and group coding is allowed — but
  **you are responsible for the results, not the AI**.
* Weekly meetings: **Monday 08:30–09:00** and **Thursday 13:30–14:00**
  (both groups).
* Deliverables: **report, code, presentation**.
* **Code submission:** at the end of the project, push your code to this repo
  on a branch named `group<id>_<year>` (e.g. `group2_2026`, `group4_2026`).
* Evaluation: **33%** research-question formulation, **33%** scientific
  execution (no logical gaps), **33%** engineering solutions (results).

Contact: Federico Corradi `f.corradi@tue.nl` · Ajeya Naithani `a.naithani@tue.nl`
