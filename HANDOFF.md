# Hand-off notes (moving from the Windows PC to the Mac)

Context for continuing this project, including in a new Claude Code session:
ask Claude to "read HANDOFF.md first".

## Project

TU/e 5XPG0, Group 2: a spiking neural network (SNN) that detects the keyword
**"sheila"**, running in C on a PicoRV32 RISC-V soft core on a PYNQ-Z2 FPGA
board; the PC streams microphone spectrograms to the board, the board lights
its LEDs for 1 s on "sheila". Code lives on branch `Khalid` of
`https://github.com/Mateo-Guzman-1/5XPG0` (remote `team`; `origin` is the
read-only course repo).

## My role: TRAINING

The group splits the work: training (me), custom instruction, parallel,
encoding, quantization, size. I keep encoding (rate, T=25), network
(256-64-2 LIF) and data choices fixed while I study training settings.
I'm learning the concepts as I go, so explanations should be from first
principles and I must be able to defend every choice ("you are responsible
for the results, not the AI").

## Where things stand

- **Full demo works end to end** (verified on the board): trained model ->
  `export_weights.py` (integer weights) -> firmware `snn.c` -> board. Uses
  `code/snn_keyword/train_keyword_snn.py` (Claude's version: real data +
  noise/silence negatives).
- **My experiment file:** `code/snn_keyword/train_sheila.py` — the ORIGINAL
  course skeleton with only the dataset swapped to Google Speech Commands
  ("sheila" vs other words, NEG_RATIO=3, official speaker-disjoint split) and
  a test bench added. Training settings come from command-line options
  (`--epochs --lr --batch --surrogate --slope --scheduler --seed --note`);
  defaults = skeleton. Every run logs a row to `runs/experiments.csv` and
  writes `runs/<name>/`. Validation is used to compare; test only with
  `--final`.
- `compare_runs.py <names>` plots learning curves -> `runs/compare.png`.
- `EXPERIMENTS.md` = my lab notebook (question, hypothesis, result, decision).
  Done: E0 baseline (1 epoch: val recall 0.686, false alarm 0.007, 32 clips
  with no output spikes), E1 epochs3 (recall 0.819, 11 silent clips).
- **Next:** epochs 5/10/20 -> learning rate -> cosine schedule -> surrogate
  shape/slope -> repeat best with seeds 1, 2 -> one `--final` run.

## Setting up on the Mac

```bash
git clone https://github.com/Mateo-Guzman-1/5XPG0.git Semiconductorproject
cd Semiconductorproject && git checkout Khalid
git remote rename origin team     # match the Windows naming (optional)
cd code/snn_keyword
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt matplotlib ziglang pyzmq sounddevice
python train_sheila.py            # first run re-downloads the 2.3 GB dataset
```
`data/` and `.venv/` are not in git (too big / machine-specific).

## The PYNQ board (lab board, not the stock image)

- Reached over a USB-Ethernet adapter at **10.43.0.1**, account
  **`student`** (lab password; passwordless sudo), no internet on the
  board. The Windows PC had SSH alias `pynq` (`~/.ssh/config`: HostName
  10.43.0.1, User student). On the Mac: set the adapter to the 10.43.0.x
  network if needed, add the same alias, and run `ssh-copy-id
  student@10.43.0.1` once.
- Deploy: `cd code/pynqz2_riscv_flow && PYNQ_USER=student ./install.sh
  10.43.0.1 --smoke-test`. Demo: `ssh pynq "cd ~/snn && ./demo.sh"`, then on
  the laptop `python code/snn_keyword/pc_keyword_demo.py 10.43.0.1`.
- Firmware is built without a RISC-V GCC: `python
  code/pynqz2_riscv_flow/firmware/build_zig.py` (pip `ziglang`).
- Hardware quirk (fixed in firmware/main.c): CPU reads of the frame counter
  can return 0 while the ARM writes frame data, so the firmware only accepts
  frame number seq_done + 1.
- Shut down before power-off: `ssh pynq "sudo shutdown -h now"`, wait 30 s.

## Known issues

- The Windows laptop mic delivered near-silent audio (peak ~0.0002): check
  with `python code/snn_keyword/mic_check.py`. The Mac mic may be fine.
