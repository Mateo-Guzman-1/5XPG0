# Group 2 — single-keyword detection SNN (starting skeleton)

Build a spiking neural network that detects **one keyword**, deploy it on the
PYNQ-Z2 RISC-V core, and light an LED for one second when the keyword is
spoken into the PC microphone.

## What is here

| File                    | Purpose                                                    |
|-------------------------|------------------------------------------------------------|
| `train_keyword_snn.py`  | snnTorch 2-layer SNN, "sheila" vs other words/noise/silence|
| `features.py`           | shared 1 s → 16x16 spectrogram front-end (train + demo)    |
| `export_weights.py`     | quantize → `firmware/weights.h` + bit-exact integer check  |
| `pc_keyword_demo.py`    | PC side: mic → spectrogram → ZeroMQ (`--local`: no board)  |
| `requirements.txt`      | Python deps for training                                   |
| `setup_venv.sh`         | create `.venv` and install the deps                        |

Board side: `../pynqz2_riscv_flow/firmware/` (`snn.c`, `main.c`) and
`../pynqz2_riscv_flow/host/keyword_bridge.py`.

## Full pipeline (Windows PowerShell; on Linux use `setup_venv.sh`)

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt ziglang pyzmq sounddevice

python train_keyword_snn.py        # downloads Speech Commands (~2.3 GB) once
python export_weights.py           # -> firmware/weights.h, float vs int accuracy
python ..\pynqz2_riscv_flow\firmware\build_zig.py   # -> firmware/spike.bin
python pc_keyword_demo.py --local  # try it on the PC mic, no board needed
```

Changing `features.py` invalidates the cached features: delete
`data/sheila_*_features.pt` and retrain.

On the board (bash, e.g. Git Bash), from `../pynqz2_riscv_flow/`:

```bash
./install.sh <board-ip>                  # copies bitstream, spike.bin, bridge
./run_demo.sh <board-ip>                 # terminal 1: loads + runs the bridge
python ../snn_keyword/pc_keyword_demo.py <board-ip>   # terminal 2 (PC mic)
```

Board login is `xilinx` by default; for a lab board with another account
prefix the scripts with e.g. `PYNQ_USER=student`.

## Suggested plan

1. **Research question (starting point — refine it).**
   > What co-design choices between the network (input encoding and topology)
   > and the resource-constrained RISC-V platform let a keyword spotter stay
   > accurate while fitting the platform's memory and real-time limits?

   Make it measurable: pick the design knobs you vary (e.g. encoding scheme,
   network size), the outcomes you measure (accuracy, latency), and the
   platform limits you hold fixed. State a hypothesis before you build.
2. **Data.** Replace `SyntheticSpectrograms` with real audio (TorchAudio
   `SPEECHCOMMANDS`, or your own recordings). Compute mel spectrograms on the
   PC — the board should receive spectrograms, not raw audio.
3. **Encoding.** Try rate vs latency (time-to-first-spike) vs delta encoding.
4. **Network.** Tune width/depth, recurrence, beta, threshold. Remember the
   target is a tiny RV32 core with no hardware accelerator (yet).
5. **Deployment.** Export the weights to integers, port the forward pass to
   `firmware/main.c` in `../pynqz2_riscv_flow/`, and decide how the PC sends
   spectrogram frames to the board (hint: a reserved BRAM buffer + a mailbox
   flag, or a small command protocol).
6. **Demo.** PC captures the mic, sends frames over ETH; the board classifies
   and flashes the LED for 1 s on the keyword.

## Deliverables

Report, code, presentation. You are responsible for the results — including
anything an AI tool helped produce.
