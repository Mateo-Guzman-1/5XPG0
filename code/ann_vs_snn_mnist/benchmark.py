#!/usr/bin/env python3
"""benchmark.py — SKELETON for GROUP 4: a first cut at comparing ANN vs SNN.

Loads the weights from train_mnist.py and reports, for each model:
  * parameter count and model size in bytes,
  * mean inference latency on the PC CPU (a proxy only — the real target is
    the RISC-V core on the PYNQ-Z2!),
  * for the SNN, the mean number of spikes per sample (an energy proxy:
    event-driven hardware only "pays" when spikes move).

This is intentionally basic. A scientific benchmark needs: fixed random
seeds, repeated runs, the same topology/training budget, on-board timing,
and a discussion of what is being held constant. Those are your choices.

Run:  python train_mnist.py && python benchmark.py
"""

import time

import torch
import torchvision

from train_mnist import ANN, SNN, rate_encode, N_TIMESTEPS

N_SAMPLES = 200


@torch.no_grad()
def latency(model, x, is_snn, repeat=10):
    model.eval()
    # warm-up
    for _ in range(3):
        _ = model(rate_encode(x, N_TIMESTEPS)).sum(0) if is_snn else model(x)
    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = model(rate_encode(x, N_TIMESTEPS)).sum(0) if is_snn else model(x)
    return (time.perf_counter() - t0) / repeat * 1000.0   # ms/batch


@torch.no_grad()
def snn_spikes(model, x):
    spikes = model(rate_encode(x, N_TIMESTEPS))
    return spikes.sum().item() / x.shape[0]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    tf = torchvision.transforms.ToTensor()
    ds = torchvision.datasets.MNIST("./data", train=False, download=True,
                                    transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=N_SAMPLES, shuffle=False)
    x, _ = next(iter(loader))

    for name, model, is_snn in (("ANN", ANN(), False), ("SNN", SNN(), True)):
        try:
            model.load_state_dict(torch.load(f"runs/{name.lower()}_mnist.pt"))
        except FileNotFoundError:
            print(f"{name}: no runs/{name.lower()}_mnist.pt - run train_mnist.py first")
            continue
        ms = latency(model, x, is_snn)
        line = (f"{name}: params={n_params(model)}  "
                f"latency={ms:.1f} ms per batch of {N_SAMPLES}")
        if is_snn:
            line += f"  spikes/sample={snn_spikes(model, x):.1f}"
        print(line)

    print("\nRemember: measure the real thing on the RISC-V core too.")


if __name__ == "__main__":
    main()
