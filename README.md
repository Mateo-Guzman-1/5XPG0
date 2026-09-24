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
  snn_keyword/           # Group 2: snnTorch training + PC demo skeleton
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

```bash
# Group 2
cd code/snn_keyword && ./setup_venv.sh && source .venv/bin/activate
python train_keyword_snn.py

# Group 4
cd code/ann_vs_snn_mnist && ./setup_venv.sh && source .venv/bin/activate
python train_mnist.py && python benchmark.py
```

Both training scripts run for **one epoch** on purpose: they are starting
points, not solutions. See the README in each folder.

---

## Formalities

* **You may use AI tools for everything**, and group coding is allowed — but
  **you are responsible for the results, not the AI**.
* Weekly meetings: **Monday 08:30–09:00** and **Thursday 13:30–14:00**
  (both groups).
* Deliverables: **report, code, presentation**.
* Evaluation: **33%** research-question formulation, **33%** scientific
  execution (no logical gaps), **33%** engineering solutions (results).

Contact: Federico Corradi `f.corradi@tue.nl` · Ajeya Naithani `a.naithani@tue.nl`
