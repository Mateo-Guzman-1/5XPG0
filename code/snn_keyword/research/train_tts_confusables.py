"""Controlled TTS hard-negative augmentation with fixed real test data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

import extended_encoding_sweep as encoding
import frontend_sweep as base
from export_extended_candidate import integer_counts, quantize_weights
from verify_encoding_reproducibility import exact_mcnemar, mean_ci


def train(x, y, split, synthetic, sy, fraction, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    real = split == "train"
    tx = np.concatenate([x[real], synthetic]) if fraction else np.asarray(x[real])
    ty = np.r_[y[real], sy] if fraction else y[real]
    weights = (1.0 - fraction) / np.bincount(y[real], minlength=2)[y[real]]
    if fraction:
        weights = np.r_[weights, fraction / np.bincount(sy, minlength=2)[sy]]
    sampler = torch.utils.data.WeightedRandomSampler(torch.from_numpy(weights).double(), 4096, replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)), batch_size=128, sampler=sampler)
    net = base.SpikeMLP(encoding.BASELINE.inputs)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
    started = time.perf_counter()
    for _ in range(12):
        net.train()
        for xb, yb in loader:
            loss = torch.nn.functional.cross_entropy(net(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return net, time.perf_counter() - started


def breakdown(labels, margins, threshold, rows):
    result = {"overall": base.metrics(labels, margins >= threshold), "by_word": {}}
    for word in sorted({row["text"] for row in rows}):
        selected = np.array([row["text"] == word for row in rows])
        result["by_word"][word] = {"clips": int(selected.sum()), "detections": int((margins[selected] >= threshold).sum()), "detection_rate": float(np.mean(margins[selected] >= threshold))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--tts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("research/results/tts_augmentation.json"))
    parser.add_argument("--models", type=Path, default=Path("data/tts_models"))
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(8)
    args.models.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.tts / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["clips"]
    voice_sets = [{row["voice"] for row in rows if row["split"] == split} for split in ("train", "val", "test")]
    assert not any(voice_sets[i] & voice_sets[j] for i, j in [(0, 1), (0, 2), (1, 2)])
    assert len({r["path"] for r in rows}) == len(rows)
    paths, y, speakers = base.discover_dataset(args.dataset, "sheila")
    split = base.dataset_split(args.dataset, paths, speakers)
    x = np.load(args.feature_cache)
    if x.shape != (len(paths), 128):
        raise ValueError("real feature cache has wrong shape")
    check = np.linspace(0, len(paths) - 1, 32, dtype=int)
    rebuilt = encoding.feature_batch(np.stack([encoding.read_pcm16(paths[i]) for i in check]), encoding.BASELINE)
    if not np.allclose(x[check], rebuilt, atol=1e-6):
        raise ValueError("real cache does not match current dataset and frontend")
    synth = encoding.get_features([args.tts / r["path"] for r in rows], [encoding.BASELINE], args.tts / "feature_cache", 64)[encoding.BASELINE.key]
    sy = np.array([r["label"] for r in rows], np.int64)
    ss = np.array([r["split"] for r in rows])
    tv, stv = split == "val", ss == "val"
    # The screen uses validation only. All ten seeds are fixed before testing.
    screen = {}
    for fraction in [0.0, 0.05, 0.1, 0.25, 0.5]:
        label = f"tts_{round(100*fraction):02d}pct"
        runs = []
        for seed in range(args.seeds):
            checkpoint = args.models / f"{label}_seed{seed}.pt"
            if checkpoint.exists():
                saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
                net = base.SpikeMLP(128)
                net.load_state_dict(saved["state_dict"])
                seconds = saved["run"]["training_seconds"]
            else:
                net, seconds = train(x, y, split, np.asarray(synth[ss == "train"]), sy[ss == "train"], fraction, seed)
            val_margin = base.margins(net, x[tv], 128)
            threshold = base.choose_threshold(y[tv], val_margin)
            syn_margin = base.margins(net, np.asarray(synth[stv]), 128)
            run = {"config": asdict(encoding.BASELINE), "key": encoding.BASELINE.key, "seed": seed, "epochs": 12, "samples_per_epoch": 4096, "batch_size": 128, "tts_fraction_per_class": fraction, "training_seconds": seconds, "threshold": threshold, "validation": base.metrics(y[tv], val_margin >= threshold), "tts_validation": breakdown(sy[stv], syn_margin, threshold, [r for r in rows if r["split"] == "val"])}
            torch.save({"run": run, "state_dict": net.state_dict()}, checkpoint)
            runs.append(run)
            print(f"{label} seed={seed} val F1={run['validation']['f1']:.4f} TTS val F1={run['tts_validation']['overall']['f1']:.4f}", flush=True)
        screen[label] = {"fraction": fraction, "runs": runs, "real_validation_f1": float(np.mean([r["validation"]["f1"] for r in runs])), "tts_validation_f1": float(np.mean([r["tts_validation"]["overall"]["f1"] for r in runs]))}
        args.output.with_name("tts_augmentation_validation.json").write_text(json.dumps(screen, indent=2), encoding="utf-8")
    floor = screen["tts_00pct"]["real_validation_f1"] - 0.005
    eligible = {k: v for k, v in screen.items() if v["real_validation_f1"] >= floor}
    selected = max(eligible, key=lambda k: (eligible[k]["real_validation_f1"] + eligible[k]["tts_validation_f1"], -eligible[k]["fraction"]))
    # Pick the exported seed from validation performance, never test scores.
    deployment_seed = max(range(args.seeds), key=lambda seed: (screen[selected]["runs"][seed]["validation"]["f1"] + screen[selected]["runs"][seed]["tts_validation"]["overall"]["f1"], -seed))
    print(f"validation selected {selected} seed={deployment_seed}", flush=True)
    result = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": asdict(encoding.BASELINE), "seeds": args.seeds, "training": {"epochs": 12, "samples_per_epoch": 4096, "batch_size": 128, "learning_rate": 0.002}, "selection": {"selected": selected, "deployment_seed": deployment_seed, "rule": "Maximize mean(real-validation F1 + TTS-validation F1) subject to real-validation F1 within 0.5 percentage points of baseline. Pick deployment seed by the same summed validation F1.", "real_validation_floor": floor}, "real_dataset_digest": base.source_digest(paths), "tts_manifest_sha256": hashlib.sha256((args.tts / "manifest.json").read_bytes()).hexdigest(), "voice_splits": manifest["voice_splits"], "tts_counts": {s: int(np.sum(ss == s)) for s in ("train", "val", "test")}, "screen": screen, "test": {}}
    t, st = split == "test", ss == "test"
    predictions = {"truth_real": y[t], "paths_real": np.array([str(p.relative_to(args.dataset)) for p in np.array(paths)[t]]), "truth_tts": sy[st], "paths_tts": np.array([r["path"] for r in rows if r["split"] == "test"])}
    for label in dict.fromkeys(["tts_00pct", selected]):
        tests = []
        margins_real, margins_tts = [], []
        for seed in range(args.seeds):
            saved = torch.load(args.models / f"{label}_seed{seed}.pt", weights_only=True, map_location="cpu")
            net = base.SpikeMLP(128)
            net.load_state_dict(saved["state_dict"])
            threshold = saved["run"]["threshold"]
            rm = base.margins(net, x[t], 128)
            tm = base.margins(net, np.asarray(synth[st]), 128)
            margins_real.append(rm)
            margins_tts.append(tm)
            params = quantize_weights(saved["state_dict"])
            icv = integer_counts(params, x[tv])
            ithreshold = base.choose_threshold(y[tv], icv[:, 1] - icv[:, 0])
            ir = integer_counts(params, x[t])
            it = integer_counts(params, np.asarray(synth[st]))
            tests.append({"seed": seed, "threshold": threshold, "integer_threshold": ithreshold, "real": base.metrics(y[t], rm >= threshold), "tts": breakdown(sy[st], tm, threshold, [r for r in rows if r["split"] == "test"]), "integer_real": base.metrics(y[t], ir[:, 1] - ir[:, 0] >= ithreshold), "integer_tts": breakdown(sy[st], it[:, 1] - it[:, 0], ithreshold, [r for r in rows if r["split"] == "test"])})
        result["test"][label] = {"runs": tests, "real_accuracy": mean_ci(np.array([r["real"]["accuracy"] for r in tests])), "real_f1": mean_ci(np.array([r["real"]["f1"] for r in tests])), "tts_f1": mean_ci(np.array([r["tts"]["overall"]["f1"] for r in tests])), "tts_false_positive_rate": mean_ci(np.array([r["tts"]["overall"]["fp"] / (r["tts"]["overall"]["fp"] + r["tts"]["overall"]["tn"]) for r in tests]))}
        predictions[f"{label}_real_margins"] = np.array(margins_real, np.int8)
        predictions[f"{label}_tts_margins"] = np.array(margins_tts, np.int8)
        predictions[f"{label}_thresholds"] = np.array([r["threshold"] for r in tests], np.int8)
    if selected != "tts_00pct":
        a, b = result["test"]["tts_00pct"]["runs"], result["test"][selected]["runs"]
        result["paired_real_f1_difference"] = mean_ci(np.array([rb["real"]["f1"] - ra["real"]["f1"] for ra, rb in zip(a, b)]))
        result["mcnemar_real_per_seed"] = [exact_mcnemar((predictions["tts_00pct_real_margins"][s] >= a[s]["threshold"]) == y[t], (predictions[f"{selected}_real_margins"][s] >= b[s]["threshold"]) == y[t]) for s in range(args.seeds)]
    # Explicitly keep raw margins and identifiers for independent paired analysis.
    np.savez_compressed(args.output.with_suffix(".npz"), **predictions)
    np.savez_compressed(args.models / "tts_test_vectors.npz", x=np.asarray(synth[st]), y=sy[st], paths=np.array([str(args.tts / r["path"]) for r in rows if r["split"] == "test"]))
    # Reuse the physical verifier, with explicit validation-selected checkpoints.
    board_summaries = {}
    for label, seed in [("tts_00pct", 0), (selected, deployment_seed)]:
        board_summaries[label] = {"config": asdict(encoding.BASELINE), "checkpoint": str((args.models / f"{label}_seed{seed}.pt").resolve()), "accuracy": result["test"][label]["real_accuracy"], "f1": result["test"][label]["real_f1"]}
    args.output.with_name("tts_augmentation_board_input.json").write_text(json.dumps({"paired_ten_seed": {"summaries": board_summaries}}, indent=2), encoding="utf-8")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
