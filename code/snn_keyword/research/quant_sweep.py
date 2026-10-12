#!/usr/bin/env python3
"""Post-training weight quantization sweep for the Sheila SNN (stage 1, step A).

Takes the trained float checkpoints of one frontend configuration, rounds the
layer-1 weights (the 6,144 weights that dominate BRAM) to fewer bits, and scores
the resulting integer model exactly as the firmware computes it.

Quantization per layer:
    qmax  = 2**(bits - 1) - 1              (127 for 8 bit, 7 for 4 bit, 1 for 2 bit)
    scale = 2**k, largest k >= 0 with max|w| * scale <= qmax   (no clipping)
    w_int = round(w * scale)
The scale is a power of two so it can be applied with a shift. Biases and
thresholds follow the scale exactly as in export_frontend_candidate.py:
    b1 = round(bias1 * s1 * 255), THRESHOLD1 = s1 * 255
    b2 = round(bias2 * s2),       THRESHOLD2 = s2

The decision threshold is chosen on validation F1 (as in Damien's pipeline) and
the test split is scored once. A reference row with Damien's fixed scale of 64
at int8 must reproduce his integer exporter exactly; the script stops if not.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import export_frontend_candidate as damien_export
import frontend_sweep as sweep


HERE = Path(__file__).resolve().parent
BETA_Q8 = damien_export.BETA_Q8          # 230, i.e. beta = 0.898
INT32_MAX = 2**31 - 1


def power_of_two_scale(weights: np.ndarray, bits: int) -> int:
    qmax = 2 ** (bits - 1) - 1
    k = math.floor(math.log2(qmax / np.abs(weights).max()))
    return 2 ** max(0, k)


def quantize(state: dict[str, torch.Tensor], bits1: int, bits2: int,
             s1: int | None = None, s2: int | None = None) -> dict:
    w1 = state["fc1.weight"].numpy().astype(np.float64)
    w2 = state["fc2.weight"].numpy().astype(np.float64)
    s1 = s1 or power_of_two_scale(w1, bits1)
    s2 = s2 or power_of_two_scale(w2, bits2)
    q1, q2 = 2 ** (bits1 - 1) - 1, 2 ** (bits2 - 1) - 1
    w1q, w2q = np.rint(w1 * s1), np.rint(w2 * s2)
    clipped = int((np.abs(w1q) > q1).sum() + (np.abs(w2q) > q2).sum())
    return {
        "bits1": bits1, "bits2": bits2, "s1": s1, "s2": s2,
        "w1": np.clip(w1q, -q1, q1).astype(np.int64),
        "w2": np.clip(w2q, -q2, q2).astype(np.int64),
        "b1": np.rint(state["fc1.bias"].numpy().astype(np.float64) * s1 * 255).astype(np.int64),
        "b2": np.rint(state["fc2.bias"].numpy().astype(np.float64) * s2).astype(np.int64),
        "clipped_weights": clipped,
    }


def integer_counts(q: dict, x: np.ndarray) -> tuple[np.ndarray, int]:
    """Firmware arithmetic (keyword_main.c classify) in int64.

    Also returns the largest |membrane * BETA_Q8| seen, because the firmware
    computes that product in 32 bits."""
    qx = np.rint(np.clip(np.asarray(x), 0.0, 1.0) * 255).astype(np.int64)
    current1 = qx @ q["w1"].T + q["b1"]
    thr1, thr2 = q["s1"] * 255, q["s2"]
    mem1 = np.zeros_like(current1)
    mem2 = np.zeros((len(qx), 2), dtype=np.int64)
    counts = np.zeros_like(mem2)
    peak = 0
    for _ in range(sweep.TIMESTEPS):
        peak = max(peak, int(np.abs(mem1).max()) * BETA_Q8, int(np.abs(mem2).max()) * BETA_Q8)
        mem1 = ((mem1 * BETA_Q8) >> 8) + current1
        spike1 = mem1 >= thr1
        mem1 -= spike1 * thr1
        mem2 = ((mem2 * BETA_Q8) >> 8) + spike1.astype(np.int64) @ q["w2"].T + q["b2"]
        spike2 = mem2 >= thr2
        mem2 -= spike2 * thr2
        counts += spike2
    return counts, peak


def weight_bytes(q: dict) -> dict[str, int]:
    w1 = math.ceil(q["w1"].size * q["bits1"] / 8)
    w2 = math.ceil(q["w2"].size * q["bits2"] / 8)
    bias = 4 * (q["b1"].size + q["b2"].size)
    return {"w1": w1, "w2": w2, "bias": bias, "total": w1 + w2 + bias}


def evaluate(q: dict, x, y, val_index, test_index, float_test_decision) -> dict:
    val_counts, val_peak = integer_counts(q, x[val_index])
    test_counts, test_peak = integer_counts(q, x[test_index])
    val_margin = val_counts[:, 1] - val_counts[:, 0]
    test_margin = test_counts[:, 1] - test_counts[:, 0]
    threshold = sweep.choose_threshold(y[val_index], val_margin)
    decision = test_margin >= threshold
    peak = max(val_peak, test_peak)
    return {
        "bits_w1": q["bits1"], "bits_w2": q["bits2"],
        "scale_w1": q["s1"], "scale_w2": q["s2"],
        "w1_range": [int(q["w1"].min()), int(q["w1"].max())],
        "w1_zero_fraction": float((q["w1"] == 0).mean()),
        "clipped_weights": q["clipped_weights"],
        "threshold": threshold,
        "validation": sweep.metrics(y[val_index], val_margin >= threshold),
        "test": sweep.metrics(y[test_index], decision),
        "test_disagreements_vs_float": int((decision != float_test_decision).sum()),
        "bytes": weight_bytes(q),
        "peak_membrane_product": peak,
        "int32_safe_on_data": peak <= INT32_MAX,
    }


def mean_std(values: list[float]) -> str:
    v = 100 * np.asarray(values)
    return f"{v.mean():6.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:4.2f}"


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
    parser.add_argument("--w2-bits", type=int, default=8,
                        help="layer-2 weights (96 of 6,240) stay int8 unless set")
    parser.add_argument("--dataset", type=Path, default=snn / "data/speech_commands_v2_sheila_subset")
    parser.add_argument("--keyword", default="sheila")
    parser.add_argument("--feature-cache", type=Path,
                        default=snn / f"data/frontend_sweep_cache_sheila_v2/{key}.npy")
    parser.add_argument("--output", type=Path, default=HERE / "results/quant/ptq.json")
    args = parser.parse_args()

    x = np.load(args.feature_cache, mmap_mode="r")
    paths, y, speakers = sweep.discover_dataset(args.dataset, args.keyword)
    split = sweep.dataset_split(args.dataset, paths, speakers)
    val_index, test_index = np.flatnonzero(split == "val"), np.flatnonzero(split == "test")

    runs = []
    for seed in args.seeds:
        checkpoint = torch.load(args.models_dir / f"{args.key}_seed{seed}.pt",
                                map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        net = sweep.SpikeMLP(x.shape[1])
        net.load_state_dict(state)
        float_margin = sweep.margins(net, x[test_index], 256)
        float_decision = float_margin >= checkpoint["run"]["threshold"]

        # Reference: Damien's fixed scale 64 must match his exporter bit for bit.
        reference = quantize(state, 8, 8, s1=64, s2=64)
        damien = damien_export.quantize(state)
        for index in (val_index, test_index):
            ours, _ = integer_counts(reference, x[index])
            theirs = damien_export.integer_counts(damien, np.asarray(x[index]))
            if not np.array_equal(ours, theirs):
                raise SystemExit(f"seed {seed}: reference does not match Damien's exporter")
        row = evaluate(reference, x, y, val_index, test_index, float_decision)
        runs.append({"seed": seed, "method": "fixed64_int8_reference", **row})
        print(f"seed {seed}: reference matches Damien's exporter", flush=True)

        for bits in args.bits:
            q = quantize(state, bits, args.w2_bits)
            row = evaluate(q, x, y, val_index, test_index, float_decision)
            runs.append({"seed": seed, "method": "ptq_pow2", **row})
            print(f"seed {seed} w1 {bits} bit: scale {q['s1']}, "
                  f"test acc {100 * row['test']['accuracy']:.2f} %", flush=True)

    print(f"\n{'method':<24}{'w1 bits':>8}  {'accuracy %':>14}  {'balanced %':>14}  "
          f"{'Sheila det. %':>14}  {'false alarms':>12}  {'weight B':>8}  int32 ok")
    groups: dict[tuple, list] = {}
    for run in runs:
        groups.setdefault((run["method"], run["bits_w1"]), []).append(run)
    summary = []
    for (method, bits), group in groups.items():
        test = [run["test"] for run in group]
        entry = {
            "method": method, "bits_w1": bits, "seeds": [run["seed"] for run in group],
            "accuracy": mean_std([t["accuracy"] for t in test]),
            "balanced_accuracy": mean_std([t["balanced_accuracy"] for t in test]),
            "recall": mean_std([t["recall"] for t in test]),
            "false_alarms": [t["fp"] for t in test],
            "weight_bytes": group[0]["bytes"]["total"],
            "int32_safe_on_data": all(run["int32_safe_on_data"] for run in group),
        }
        summary.append(entry)
        print(f"{method:<24}{bits:>8}  {entry['accuracy']:>14}  {entry['balanced_accuracy']:>14}  "
              f"{entry['recall']:>14}  {str(entry['false_alarms']):>12}  "
              f"{entry['weight_bytes']:>8}  {entry['int32_safe_on_data']}")

    report = {
        "description": "Post-training power-of-two weight quantization of layer 1",
        "checkpoints": str(args.models_dir.resolve()),
        "key": args.key,
        "w2_bits": args.w2_bits,
        "beta_q8": BETA_Q8,
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
