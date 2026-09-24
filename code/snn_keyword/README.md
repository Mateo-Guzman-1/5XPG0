# Group 2 — single-keyword detection SNN (starting skeleton)

Build a spiking neural network that detects **one keyword**, deploy it on the
PYNQ-Z2 RISC-V core, and light an LED for one second when the keyword is
spoken into the PC microphone.

## What is here

| File                    | Purpose                                                    |
|-------------------------|------------------------------------------------------------|
| `train_keyword_snn.py`  | snnTorch 2-layer SNN, ONE epoch, synthetic spectrograms    |
| `pc_keyword_demo.py`    | PC side: mic → mel spectrogram → ZeroMQ publisher          |
| `requirements.txt`      | Python deps for training                                   |
| `setup_venv.sh`         | create `.venv` and install the deps                        |

The training script is a **skeleton**: it runs end-to-end but is
intentionally under-trained and uses fake data. The board side (receive the
spectrogram, run the SNN, drive the LED) is left to you — see the PYNQ flow in
`../pynqz2_riscv_flow/`.

## Setup & run the training example

```bash
./setup_venv.sh
source .venv/bin/activate
python train_keyword_snn.py
```

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
