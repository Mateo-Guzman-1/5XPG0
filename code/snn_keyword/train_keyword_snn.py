#!/usr/bin/env python3
"""train_keyword_snn.py — SKELETON for GROUP 2 (single-keyword detection).

This is a starting point, NOT a solution. It shows the whole pipeline:

    spectrogram frame  ->  spike encoding  ->  2-layer SNN (snnTorch)
    ->  training loop (ONE epoch, on purpose)  ->  saved weights

It deliberately:
  * uses a tiny, SYNTHETIC spectrogram dataset so the script runs anywhere
    with no downloads (see SyntheticSpectrograms below),
  * trains for exactly ONE epoch, with a very small 2-layer MLP,
  * does NOT check accuracy. The network is intentionally under-trained;
    deciding the topology / encoding / training is your job.

What you should change (design choices!):
  1. INPUT ENCODING   — try rate, latency (time-to-first-spike), or delta.
  2. TOPOLOGY         — number of hidden neurons, recurrent layers, ...
  3. TRAINING         — more epochs, better LR schedule, surrogate gradients.
  4. DATASET          — replace SyntheticSpectrograms with real audio:
                        e.g. torchaudio.datasets.SPEECHCOMMANDS, then compute
                        a mel spectrogram per 1 s window.

Run:  python train_keyword_snn.py
Output: runs/keyword_snn.pt  (state_dict) and prints the training loss.
"""

import os

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# ----------------------------------------------------------------------
# Configuration (sensible defaults for a quick smoke run)
# ----------------------------------------------------------------------
N_FREQ = 16          # spectrogram "frequency bins"
N_TIME = 16          # spectrogram time frames per sample
N_INPUT = N_FREQ * N_TIME
N_HIDDEN = 64
N_CLASSES = 2        # e.g. {keyword, not-keyword}
N_TIMESTEPS = 25     # simulation steps per sample (rate encoding)
BATCH = 16
N_TRAIN = 512
EPOCHS = 1           # <-- on purpose: the skeleton is under-trained
LR = 1e-3
SEED = 0
OUT_DIR = "runs"


# ----------------------------------------------------------------------
# Synthetic spectrogram dataset
# ----------------------------------------------------------------------
class SyntheticSpectrograms(torch.utils.data.Dataset):
    """Random 16x16 "spectrograms" with a class-dependent bump.

    Class 0 has an energy bump in the lower half of the frequency axis,
    class 1 in the upper half. This is only a stand-in so the script runs;
    real data will look very different. Do not read accuracy into this.
    """

    def __init__(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, N_FREQ, N_TIME, generator=g)
        self.y = torch.randint(0, N_CLASSES, (n,), generator=g)
        for i in range(n):
            if self.y[i] == 0:
                self.x[i, : N_FREQ // 2, :] += 1.0
            else:
                self.x[i, N_FREQ // 2:, :] += 1.0
        self.x = self.x.clamp(0, 2) / 2.0        # -> [0, 1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def rate_encode(x, n_steps):
    """Rate encoding: repeat the (analog) input for n_steps steps.

    Each input value is treated as a constant drive for every time step.
    A Bernoulli/rate encoder or latency encoder would be a design change.
    """
    return x.unsqueeze(0).repeat(n_steps, 1, 1)   # [T, B, N_INPUT]


# ----------------------------------------------------------------------
# 2-layer spiking MLP
# ----------------------------------------------------------------------
class SpikeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(N_INPUT, N_HIDDEN)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc2 = nn.Linear(N_HIDDEN, N_CLASSES)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        # x: [T, B, N_INPUT]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for step in range(x.shape[0]):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec, dim=0)      # [T, B, N_CLASSES]


def main():
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = SyntheticSpectrograms(N_TRAIN, seed=SEED)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)

    net = SpikeMLP()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    net.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in loader:
            x = x.reshape(x.shape[0], -1)             # [B, N_INPUT]
            spikes = net(rate_encode(x, N_TIMESTEPS))  # [T, B, N_CLASSES]
            # decode: use the spike count (firing rate) over time
            out = spikes.sum(dim=0)                    # [B, N_CLASSES]
            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * y.shape[0]
        print(f"epoch {epoch+1}/{EPOCHS}  loss={total/len(ds):.4f}")

    path = os.path.join(OUT_DIR, "keyword_snn.pt")
    torch.save(net.state_dict(), path)
    print(f"saved {path}")
    print("NOTE: one epoch only - this model is NOT expected to work well.")


if __name__ == "__main__":
    main()
