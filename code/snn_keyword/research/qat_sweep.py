#!/usr/bin/env python3
"""Quantization-aware fine-tuning sweep for the Sheila SNN (stage 1, step B).

Starts from the same float checkpoints as quant_sweep.py and fine-tunes them
for a few epochs while the forward pass uses rounded weights (fake
quantization with a straight-through estimator):

    w_q = w + (clamp(round(w * s), -qmax, qmax) / s - w).detach()

The scale s of each layer is fixed before fine-tuning (power-of-two rule of
quant_sweep.py) and reused at export, so training sees exactly the integer
grid that goes into BRAM.

Control: the same fine-tuning WITHOUT fake quantization, followed by the same
post-training rounding. QAT only counts as better if it beats this control,
otherwise the gain could just come from the extra training epochs.

Every model is scored with quant_sweep.evaluate (bit-exact firmware arithmetic,
threshold chosen on validation F1, test scored once).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import frontend_sweep as sweep
import quant_sweep as qs


HERE = Path(__file__).resolve().parent


def fake_quant(w: torch.Tensor, scale: int, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    rounded = torch.clamp(torch.round(w * scale), -qmax, qmax) / scale
    return w + (rounded - w).detach()


class QuantSpikeMLP(sweep.SpikeMLP):
    """Damien's SpikeMLP with optional fake-quantized weights in the forward pass."""

    def __init__(self, inputs: int):
        super().__init__(inputs)
        self.quant = None          # (s1, bits1, s2, bits2) or None for plain float

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1, w2 = self.fc1.weight, self.fc2.weight
        if self.quant is not None:
            s1, bits1, s2, bits2 = self.quant
            w1, w2 = fake_quant(w1, s1, bits1), fake_quant(w2, s2, bits2)
        mem1 = x.new_zeros((len(x), sweep.HIDDEN))
        mem2 = x.new_zeros((len(x), 2))
        counts = torch.zeros_like(mem2)
        current1 = F.linear(x, w1, self.fc1.bias)
        for _ in range(sweep.TIMESTEPS):
            mem1 = 0.9 * mem1 + current1
            spike1 = self.spike(mem1 - 1.0)
            mem1 = mem1 - spike1
            mem2 = 0.9 * mem2 + F.linear(spike1, w2, self.fc2.bias)
            spike2 = self.spike(mem2 - 1.0)
            mem2 = mem2 - spike2
            counts = counts + spike2
        return counts


def fine_tune(state, x_train, y_train, quant, seed, args) -> dict[str, torch.Tensor]:
    """Same sampler, loss and optimizer as frontend_sweep.train_one."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = QuantSpikeMLP(x_train.shape[1])
    net.load_state_dict(state)
    net.quant = quant
    counts = np.bincount(y_train, minlength=2)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(1.0 / counts[y_train]).double(),
        num_samples=args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch, sampler=sampler,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        net.train()
        for xb, yb in loader:
            loss = loss_fn(net(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return {key: value.detach().clone() for key, value in net.state_dict().items()}


def main() -> int:
    snn = HERE.parent
    key = "b8_f8_hz40-5120_t1000"
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path,
                        default=HERE / "results/repro/frontend_sweep_sheila_v2_repro_models")
    parser.add_argument("--key", default=key)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6, 5, 4, 3, 2])
    parser.add_argument("--w2-bits", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4, help="fine-tuning epochs")
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="lower than the 2e-3 of the original training (fine-tuning)")
    parser.add_argument("--dataset", type=Path, default=snn / "data/speech_commands_v2_sheila_subset")
    parser.add_argument("--keyword", default="sheila")
    parser.add_argument("--feature-cache", type=Path,
                        default=snn / f"data/frontend_sweep_cache_sheila_v2/{key}.npy")
    parser.add_argument("--output", type=Path, default=HERE / "results/quant/qat.json")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    x = np.load(args.feature_cache, mmap_mode="r")
    paths, y, speakers = sweep.discover_dataset(args.dataset, args.keyword)
    split = sweep.dataset_split(args.dataset, paths, speakers)
    train_index = np.flatnonzero(split == "train")
    val_index, test_index = np.flatnonzero(split == "val"), np.flatnonzero(split == "test")
    x_train, y_train = np.asarray(x[train_index]), y[train_index]
    model_dir = args.output.parent / f"{args.output.stem}_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in args.seeds:
        checkpoint = torch.load(args.models_dir / f"{args.key}_seed{seed}.pt",
                                map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        net = sweep.SpikeMLP(x.shape[1])
        net.load_state_dict(state)
        float_decision = sweep.margins(net, x[test_index], 256) >= checkpoint["run"]["threshold"]

        # Control: fine-tune in float, then round after training.
        control = fine_tune(state, x_train, y_train, None, seed, args)
        torch.save(control, model_dir / f"{args.key}_seed{seed}_control.pt")

        for bits in args.bits:
            # Scales are fixed from the original float weights and shared by both methods.
            s1 = qs.power_of_two_scale(state["fc1.weight"].numpy(), bits)
            s2 = qs.power_of_two_scale(state["fc2.weight"].numpy(), args.w2_bits)

            q = qs.quantize(control, bits, args.w2_bits, s1=s1, s2=s2)
            row = qs.evaluate(q, x, y, val_index, test_index, float_decision)
            runs.append({"seed": seed, "method": "control_ft_then_ptq", **row})

            tuned = fine_tune(state, x_train, y_train, (s1, bits, s2, args.w2_bits), seed, args)
            torch.save(tuned, model_dir / f"{args.key}_seed{seed}_qat{bits}.pt")
            q = qs.quantize(tuned, bits, args.w2_bits, s1=s1, s2=s2)
            row = qs.evaluate(q, x, y, val_index, test_index, float_decision)
            runs.append({"seed": seed, "method": "qat", **row})
            print(f"seed {seed} w1 {bits} bit (scale {s1}): "
                  f"control {100 * runs[-2]['test']['accuracy']:.2f} %, "
                  f"QAT {100 * row['test']['accuracy']:.2f} %", flush=True)

    print(f"\n{'method':<22}{'w1 bits':>8}  {'accuracy %':>14}  {'balanced %':>14}  "
          f"{'Sheila det. %':>14}  {'false alarms':>13}  {'weight B':>8}  int32 ok")
    groups: dict[tuple, list] = {}
    for run in runs:
        groups.setdefault((run["method"], run["bits_w1"]), []).append(run)
    summary = []
    for (method, bits), group in sorted(groups.items(), key=lambda item: (-item[0][1], item[0][0])):
        test = [run["test"] for run in group]
        entry = {
            "method": method, "bits_w1": bits, "seeds": [run["seed"] for run in group],
            "accuracy": qs.mean_std([t["accuracy"] for t in test]),
            "balanced_accuracy": qs.mean_std([t["balanced_accuracy"] for t in test]),
            "recall": qs.mean_std([t["recall"] for t in test]),
            "false_alarms": [t["fp"] for t in test],
            "clipped_weights": [run["clipped_weights"] for run in group],
            "weight_bytes": group[0]["bytes"]["total"],
            "int32_safe_on_data": all(run["int32_safe_on_data"] for run in group),
        }
        summary.append(entry)
        print(f"{method:<22}{bits:>8}  {entry['accuracy']:>14}  {entry['balanced_accuracy']:>14}  "
              f"{entry['recall']:>14}  {str(entry['false_alarms']):>13}  "
              f"{entry['weight_bytes']:>8}  {entry['int32_safe_on_data']}")

    report = {
        "description": "Quantization-aware fine-tuning vs float fine-tuning + post-training rounding",
        "checkpoints": str(args.models_dir.resolve()),
        "key": args.key,
        "w2_bits": args.w2_bits,
        "fine_tuning": {"epochs": args.epochs, "samples_per_epoch": args.samples_per_epoch,
                        "batch": args.batch, "lr": args.lr, "optimizer": "Adam"},
        "threshold_selection": "validation F1 (Damien's choose_threshold)",
        "summary": summary,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
