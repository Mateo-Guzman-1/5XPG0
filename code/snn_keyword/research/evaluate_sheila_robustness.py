#!/usr/bin/env python3
"""Evaluate the deployed Sheila model on noisy speech and background audio."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import frontend_sweep as sweep
from export_frontend_candidate import integer_counts, quantize

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_features import N_INPUTS, extract_features, read_wav  # noqa: E402
from train_keyword_snn_clip import add_noise_at_snr, generated_noise  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--keyword", default="sheila")
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--background-examples", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    params = quantize(checkpoint["state_dict"])
    paths, labels, speakers = sweep.discover_dataset(args.dataset, args.keyword)
    split = sweep.dataset_split(args.dataset, paths, speakers)
    indices = np.flatnonzero(split == "test")

    results: dict[str, object] = {
        "keyword": args.keyword,
        "test_clips": int(len(indices)),
        "threshold": args.threshold,
    }
    for snr_db in (10, 5):
        rng = np.random.default_rng(1000 + snr_db)
        features = np.empty((len(indices), N_INPUTS), dtype=np.float32)
        started = time.perf_counter()
        for out_index, source_index in enumerate(indices):
            audio = read_wav(paths[source_index])
            noisy = add_noise_at_snr(audio, rng, float(snr_db))
            features[out_index] = extract_features(noisy).reshape(-1)
        feature_seconds = time.perf_counter() - started
        counts = integer_counts(params, features)
        margin = counts[:, 1] - counts[:, 0]
        results[f"noise_{snr_db}db"] = {
            **sweep.metrics(labels[indices], margin >= args.threshold),
            "frontend_ms_per_clip": 1000.0 * feature_seconds / len(indices),
        }

    rng = np.random.default_rng(2000)
    background = np.empty((args.background_examples, N_INPUTS), dtype=np.float32)
    for index in range(args.background_examples):
        if index % 4 == 0:
            audio = np.zeros(16_000, dtype=np.float32)
        else:
            rms = float(10.0 ** rng.uniform(-4.0, -1.25))
            audio = generated_noise(rng, rms)
        background[index] = extract_features(audio).reshape(-1)
    counts = integer_counts(params, background)
    margin = counts[:, 1] - counts[:, 0]
    decisions = margin >= args.threshold
    silence = np.arange(args.background_examples) % 4 == 0
    results["background"] = {
        "examples": args.background_examples,
        "false_positives": int(decisions.sum()),
        "false_positive_rate": float(decisions.mean()),
        "exact_silence_examples": int(silence.sum()),
        "exact_silence_false_positives": int(decisions[silence].sum()),
    }

    rendered = json.dumps(results, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
