# Group 4 — benchmark ANN vs SNN on MNIST with RISC-V

Train a small **ANN** and a small **SNN** for MNIST, deploy **both** on the
PYNQ-Z2 RISC-V core, and benchmark which one wins (and why). Extra: can the
RISC-V instruction set be extended to make the SNN more efficient?

## What is here

| File                 | Purpose                                                  |
|----------------------|----------------------------------------------------------|
| `train_mnist.py`     | ANN + SNN, 2-layer MLPs, ONE epoch each, saves weights   |
| `benchmark.py`       | parameters, PC-CPU latency, SNN spike-count proxy        |
| `requirements.txt`   | Python deps for training                                 |
| `setup_venv.sh` / `setup_venv.ps1` | create `.venv` and install the deps       |

Both networks are **skeletons**: tiny, trained one epoch, deliberately not a
fair comparison yet. Building a *scientific* benchmark is the project.

## Setup & run the training example

Windows PowerShell:

```powershell
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
python train_mnist.py
python benchmark.py
```

Linux, WSL, or Git Bash:

```bash
./setup_venv.sh
source .venv/bin/activate
python train_mnist.py
python benchmark.py
```

## Suggested plan

1. **Research question (starting point — refine it).**
   > For edge inference on a resource-constrained processor, is a spiking
   > network a better design choice than a classical ANN — and what hardware
   > support (e.g. instruction-set extensions) is worth adding to make it win?

   Make it measurable: compare ANN vs SNN under the same budget, measure
   accuracy and a cost proxy (energy / synaptic operations / latency), and keep
   topology, parameter count, training and seeds fixed. State a hypothesis
   before you build.
2. **Training.** Make the comparison fair: same width, same budget, several
   seeds, enough epochs to actually learn. Report error bars.
3. **Deployment.** Port both inference paths to `firmware/main.c` in
   `../pynqz2_riscv_flow/`. Measure **on the board**, not just on the PC:
   latency, cycles (use the timer), and memory.
4. **Energy proxy.** For spiking hardware, count spikes / synaptic operations,
   not just FLOPs. Compare MACs (ANN) against SOPs (SNN).
5. **Extra: ISA extension.** PicoRV32 offers a coprocessor (PCPI) hook. A
   custom instruction could accelerate the LIF update (e.g. leak + add +
   compare + reset in one instruction) or a dot product. Implement it, measure
   the speed-up, and discuss the area cost.

## Deliverables

Report, code, presentation. You are responsible for the results — including
anything an AI tool helped produce.
