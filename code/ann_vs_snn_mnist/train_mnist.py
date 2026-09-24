#!/usr/bin/env python3
"""train_mnist.py — SKELETON for GROUP 4 (ANN vs SNN on MNIST).

Trains TWO tiny 2-layer MLPs on MNIST and saves their weights:
  * an ANN  (784 -> 128 -> 10, ReLU)
  * an SNN  (784 -> 128 LIF -> 10 LIF, rate-encoded input, snnTorch)

Both are trained for exactly ONE epoch and are deliberately small and
under-trained. Do NOT read the printed accuracy as a result: it is there only
to show the pipeline works. Reproducing a fair ANN-vs-SNN benchmark (same
topology / budget / seeds, repeated runs) is the actual project.

Key design choices left to you:
  1. INPUT ENCODING   — rate, latency, or delta encoding for the SNN.
  2. TOPOLOGY         — width/depth, recurrence, shared vs separate tuning.
  3. TRAINING         — epochs, optimiser, learning-rate schedule.
  4. BENCHMARK        — configure benchmark.py and analyse the results.

Run:  python train_mnist.py
Output: runs/ann_mnist.pt, runs/snn_mnist.pt
"""

import os
import time

import torch
import torch.nn as nn
import torchvision
import snntorch as snn
from snntorch import surrogate

EPOCHS = 1            # <-- on purpose
BATCH = 64
LR = 1e-3
N_HIDDEN = 128
N_TIMESTEPS = 25      # SNN simulation steps
SEED = 0
OUT_DIR = "runs"


def get_mnist():
    tf = torchvision.transforms.ToTensor()
    train = torchvision.datasets.MNIST("./data", train=True, download=True,
                                       transform=tf)
    test = torchvision.datasets.MNIST("./data", train=False, download=True,
                                      transform=tf)
    return train, test


# ----------------------------------------------------------------------
def rate_encode(x, n_steps):
    """Binary rate encoding of a normalized image -> [T, B, 784] spikes.

    Each pixel (in [0,1]) becomes a Bernoulli spike train whose expected
    number of spikes over T steps is proportional to the pixel value.
    """
    x = x.reshape(x.shape[0], -1)          # [B, 784]
    rnd = torch.rand(n_steps, *x.shape, device=x.device, dtype=x.dtype)
    return (rnd < x).float()


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, N_HIDDEN), nn.ReLU(),
            nn.Linear(N_HIDDEN, 10),
        )

    def forward(self, x):
        return self.net(x)


class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(784, N_HIDDEN)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc2 = nn.Linear(N_HIDDEN, 10)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        # x: [T, B, 784]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        rec = []
        for step in range(x.shape[0]):
            spk1, mem1 = self.lif1(self.fc1(x[step]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            rec.append(spk2)
        return torch.stack(rec, dim=0)     # [T, B, 10]


def train(model, loader, is_snn):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    total = 0.0
    for x, y in loader:
        if is_snn:
            out = model(rate_encode(x, N_TIMESTEPS)).sum(dim=0)  # firing rate
        else:
            out = model(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * y.shape[0]
    return total / len(loader.dataset), time.time() - t0


@torch.no_grad()
def quick_acc(model, loader, is_snn, limit=1000):
    model.eval()
    correct = n = 0
    for x, y in loader:
        out = model(rate_encode(x, N_TIMESTEPS)).sum(0) if is_snn else model(x)
        correct += (out.argmax(1) == y).sum().item()
        n += y.shape[0]
        if n >= limit:
            break
    return correct / n


def main():
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    train_ds, test_ds = get_mnist()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH,
                                              shuffle=False, num_workers=2)

    for name, model, is_snn in (("ANN", ANN(), False), ("SNN", SNN(), True)):
        loss, secs = train(model, train_loader, is_snn)
        acc = quick_acc(model, test_loader, is_snn)
        path = os.path.join(OUT_DIR, f"{name.lower()}_mnist.pt")
        torch.save(model.state_dict(), path)
        print(f"{name}: loss={loss:.4f}  train_time={secs:.1f}s  "
              f"quick_acc={acc*100:.1f}%  -> {path}")

    print("NOTE: one epoch each - these numbers are NOT a benchmark yet.")


if __name__ == "__main__":
    main()
