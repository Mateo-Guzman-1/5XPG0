#!/usr/bin/env python3
"""Controlled frontend/encoding sweep for the Group-2 keyword SNN.

The script varies feature precision, frequency-band count, frequency range,
and represented audio duration.  It uses the official Speech Commands split
manifests when present (or a speaker-disjoint fallback), screens every value
with one fixed seed, selects values on validation F1, and confirms the selected
configurations with three seeds.  The test split is evaluated only for the
confirmation runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from snntorch import surrogate


SAMPLE_RATE = 16_000
FFT_WINDOW = 400
FFT_HOP = 160
TIME_BINS = 16
RMS_GATE = 0.003
RMS_FULL_SCALE = 0.040
HIDDEN = 48
TIMESTEPS = 16


@dataclass(frozen=True)
class FeatureConfig:
    bits: int = 8
    bands: int = 16
    low_hz: int = 40
    high_hz: int = 5120
    duration_ms: int = 1000

    @property
    def key(self) -> str:
        return (
            f"b{self.bits}_f{self.bands}_hz{self.low_hz}-{self.high_hz}_"
            f"t{self.duration_ms}"
        )

    @property
    def inputs(self) -> int:
        return self.bands * TIME_BINS


BASELINE = FeatureConfig()
BITS = (2, 3, 4, 6, 8)
BANDS = (8, 12, 16, 20, 24, 32)
RANGES = (
    (40, 3400),
    (40, 4000),
    (40, 5120),
    (40, 6500),
    (40, 7600),
    (300, 3400),
    (300, 5120),
    (150, 6500),
    (300, 6500),
    (500, 6500),
    (150, 7600),
    (300, 7600),
    (500, 7600),
)
DURATIONS_MS = (600, 800, 1000, 1200)


class SpikeMLP(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.fc1 = nn.Linear(inputs, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 2)
        self.spike = surrogate.fast_sigmoid(slope=25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem1 = x.new_zeros((len(x), HIDDEN))
        mem2 = x.new_zeros((len(x), 2))
        counts = torch.zeros_like(mem2)
        current1 = self.fc1(x)
        for _ in range(TIMESTEPS):
            mem1 = 0.9 * mem1 + current1
            spike1 = self.spike(mem1 - 1.0)
            mem1 = mem1 - spike1
            mem2 = 0.9 * mem2 + self.fc2(spike1)
            spike2 = self.spike(mem2 - 1.0)
            mem2 = mem2 - spike2
            counts = counts + spike2
        return counts


def read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16_000, 1, 2):
            raise ValueError(f"expected mono 16 kHz PCM16: {path}")
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    result = np.zeros(SAMPLE_RATE, dtype=np.float32)
    source = audio.astype(np.float32) / 32768.0
    if len(source) >= SAMPLE_RATE:
        result[:] = source[:SAMPLE_RATE]
    else:
        start = (SAMPLE_RATE - len(source)) // 2
        result[start:start + len(source)] = source
    return result


def fit_duration(batch: np.ndarray, duration_ms: int) -> np.ndarray:
    samples = duration_ms * SAMPLE_RATE // 1000
    if samples == SAMPLE_RATE:
        return batch
    if samples < SAMPLE_RATE:
        start = (SAMPLE_RATE - samples) // 2
        return batch[:, start:start + samples]
    pad = samples - SAMPLE_RATE
    left = pad // 2
    return np.pad(batch, ((0, 0), (left, pad - left)))


def feature_batch(audio: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    fitted = fit_duration(audio, cfg.duration_ms)
    rms = np.sqrt(np.mean(fitted * fitted, axis=1))
    activity = np.zeros_like(rms)
    active = rms > RMS_GATE
    activity[active] = np.clip(
        np.log(rms[active] / RMS_GATE) / np.log(RMS_FULL_SCALE / RMS_GATE),
        0.0,
        1.0,
    )

    frames = np.lib.stride_tricks.sliding_window_view(
        fitted, FFT_WINDOW, axis=1
    )[:, ::FFT_HOP, :]
    window = np.hanning(FFT_WINDOW).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=2)).astype(np.float32)
    frequencies = np.fft.rfftfreq(FFT_WINDOW, 1.0 / SAMPLE_RATE)
    selected = np.flatnonzero(
        (frequencies >= cfg.low_hz) & (frequencies <= cfg.high_hz)
    )
    if len(selected) < cfg.bands:
        raise ValueError(f"{cfg.key}: fewer FFT bins than requested bands")

    groups = np.array_split(selected, cfg.bands)
    band_values = np.stack(
        [spectrum[:, :, group].mean(axis=2) for group in groups], axis=1
    )
    band_values = np.log1p(band_values)
    frame_groups = np.array_split(np.arange(band_values.shape[2]), TIME_BINS)
    pooled = np.stack(
        [band_values[:, :, group].mean(axis=2) for group in frame_groups], axis=2
    )
    peak = pooled.max(axis=(1, 2), keepdims=True)
    pooled = np.divide(pooled, peak, out=np.zeros_like(pooled), where=peak > 0)
    pooled *= activity[:, None, None]
    levels = (1 << cfg.bits) - 1
    pooled = np.rint(np.clip(pooled, 0.0, 1.0) * levels) / levels
    return pooled.reshape(len(audio), -1).astype(np.float32)


def all_configs() -> tuple[list[FeatureConfig], dict[str, list[str]]]:
    groups = {
        "bits": [replace(BASELINE, bits=value).key for value in BITS],
        "bands": [replace(BASELINE, bands=value).key for value in BANDS],
        "range": [
            replace(BASELINE, low_hz=low, high_hz=high).key
            for low, high in RANGES
        ],
        "duration": [
            replace(BASELINE, duration_ms=value).key for value in DURATIONS_MS
        ],
    }
    configs: dict[str, FeatureConfig] = {BASELINE.key: BASELINE}
    for bits in BITS:
        cfg = replace(BASELINE, bits=bits)
        configs[cfg.key] = cfg
    for bands in BANDS:
        cfg = replace(BASELINE, bands=bands)
        configs[cfg.key] = cfg
    for low, high in RANGES:
        cfg = replace(BASELINE, low_hz=low, high_hz=high)
        configs[cfg.key] = cfg
    for duration in DURATIONS_MS:
        cfg = replace(BASELINE, duration_ms=duration)
        configs[cfg.key] = cfg
    return list(configs.values()), groups


def discover_dataset(
    root: Path, keyword: str
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    paths = sorted(root.glob("*/*.wav"))
    if not paths:
        raise RuntimeError(f"no WAV files found under {root}")
    labels = np.array([int(path.parent.name == keyword) for path in paths], np.int64)
    speakers = np.array([path.name.split("_nohash_")[0] for path in paths])
    return paths, labels, speakers


def speaker_split(speakers: np.ndarray, seed: int = 2026) -> np.ndarray:
    unique = sorted(set(speakers.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = round(0.70 * len(unique))
    n_val = round(0.15 * len(unique))
    assignment = {
        speaker: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
        for i, speaker in enumerate(unique)
    }
    return np.array([assignment[speaker] for speaker in speakers])


def dataset_split(root: Path, paths: list[Path], speakers: np.ndarray) -> np.ndarray:
    validation_manifest = root / "validation_list.txt"
    test_manifest = root / "testing_list.txt"
    if not (validation_manifest.is_file() and test_manifest.is_file()):
        return speaker_split(speakers)

    def entries(path: Path) -> set[str]:
        return {
            line.strip().replace("\\", "/")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    validation = entries(validation_manifest)
    test = entries(test_manifest)
    if validation & test:
        raise RuntimeError("validation and test manifests overlap")
    relative = [path.relative_to(root).as_posix() for path in paths]
    split = np.array(
        ["val" if name in validation else "test" if name in test else "train"
         for name in relative]
    )
    speaker_sets = {
        name: set(speakers[split == name].tolist())
        for name in ("train", "val", "test")
    }
    if any(
        speaker_sets[left] & speaker_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("official manifests are not speaker-disjoint")
    return split


def generate_features(
    paths: list[Path], configs: list[FeatureConfig], batch_size: int
) -> dict[str, np.ndarray]:
    result = {
        cfg.key: np.empty((len(paths), cfg.inputs), dtype=np.float32)
        for cfg in configs
    }
    for start in range(0, len(paths), batch_size):
        end = min(len(paths), start + batch_size)
        audio = np.stack([read_pcm16(path) for path in paths[start:end]])
        # Computing the FFT separately for every configuration is deliberate:
        # it keeps duration and frequency changes exact and avoids a cache whose
        # preprocessing assumptions differ between sweep points.
        for cfg in configs:
            result[cfg.key][start:end] = feature_batch(audio, cfg)
        print(f"features {end}/{len(paths)}", flush=True)
    return result


def cache_path(cache_dir: Path, cfg: FeatureConfig) -> Path:
    return cache_dir / f"{cfg.key}.npy"


def get_features(
    paths: list[Path], configs: list[FeatureConfig], cache_dir: Path, batch_size: int
) -> dict[str, np.ndarray]:
    cached: dict[str, np.ndarray] = {}
    missing = []
    for cfg in configs:
        path = cache_path(cache_dir, cfg)
        if path.is_file():
            value = np.load(path, mmap_mode="r")
            if value.shape == (len(paths), cfg.inputs):
                cached[cfg.key] = value
                continue
        missing.append(cfg)
    if missing:
        generated = generate_features(paths, missing, batch_size)
        for cfg in missing:
            path = cache_path(cache_dir, cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, generated[cfg.key])
            cached[cfg.key] = np.load(path, mmap_mode="r")
    return cached


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, bool)
    prediction = np.asarray(prediction, bool)
    tp = int((truth & prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    tn = int((~truth & ~prediction).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(truth),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "balanced_accuracy": 0.5
        * (tp / max(1, tp + fn) + tn / max(1, tn + fp)),
    }


def choose_threshold(truth: np.ndarray, margin: np.ndarray) -> int:
    candidates = np.unique(np.r_[margin, margin.max() + 1])
    scored = [(metrics(truth, margin >= threshold)["f1"], int(threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[0], item[1]))[1]


@torch.no_grad()
def margins(net: SpikeMLP, x: np.ndarray, batch_size: int) -> np.ndarray:
    net.eval()
    values = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.asarray(x[start:start + batch_size]))
        counts = net(xb).cpu().numpy()
        values.append(counts[:, 1] - counts[:, 0])
    return np.concatenate(values)


def memory_model(cfg: FeatureConfig) -> dict[str, int]:
    parameters = HIDDEN * cfg.inputs + 4 * HIDDEN + 2 * HIDDEN + 8
    return {
        "learned_parameter_bytes_int8": parameters,
        "byte_aligned_input_bytes": cfg.inputs,
        "packed_input_bytes": math.ceil(cfg.inputs * cfg.bits / 8),
        "input_mac_count": HIDDEN * cfg.inputs,
    }


def train_one(
    x: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    cfg: FeatureConfig,
    seed: int,
    epochs: int,
    samples_per_epoch: int,
    batch_size: int,
    evaluate_test: bool,
) -> tuple[dict, dict[str, torch.Tensor]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_index = np.flatnonzero(split == "train")
    val_index = np.flatnonzero(split == "val")
    test_index = np.flatnonzero(split == "test")
    x_train = np.asarray(x[train_index])
    y_train = y[train_index]
    class_counts = np.bincount(y_train, minlength=2)
    sample_weights = 1.0 / class_counts[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(sample_weights).double(),
        num_samples=samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler
    )
    net = SpikeMLP(cfg.inputs)
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    started = time.perf_counter()
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            loss = loss_fn(net(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    seconds = time.perf_counter() - started

    val_margin = margins(net, x[val_index], batch_size)
    threshold = choose_threshold(y[val_index], val_margin)
    report = {
        "config": asdict(cfg),
        "key": cfg.key,
        "seed": seed,
        "epochs": epochs,
        "samples_per_epoch": samples_per_epoch,
        "training_seconds": seconds,
        "threshold": threshold,
        "validation": metrics(y[val_index], val_margin >= threshold),
        "memory": memory_model(cfg),
    }
    if evaluate_test:
        test_margin = margins(net, x[test_index], batch_size)
        report["test"] = metrics(y[test_index], test_margin >= threshold)
    return report, {key: value.detach().cpu() for key, value in net.state_dict().items()}


def split_summary(y: np.ndarray, split: np.ndarray, speakers: np.ndarray) -> dict:
    result = {}
    for name in ("train", "val", "test"):
        selected = split == name
        result[name] = {
            "clips": int(selected.sum()),
            "positive": int(y[selected].sum()),
            "negative": int(selected.sum() - y[selected].sum()),
            "speakers": len(set(speakers[selected].tolist())),
        }
    return result


def source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=here / "data" / "mini_speech_commands",
    )
    parser.add_argument("--keyword", default="yes")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=here / "data" / "frontend_sweep_cache",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "frontend_sweep.json")
    parser.add_argument("--feature-batch", type=int, default=64)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--screen-epochs", type=int, default=8)
    parser.add_argument("--confirm-epochs", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    paths, labels, speakers = discover_dataset(args.dataset, args.keyword)
    split = dataset_split(args.dataset, paths, speakers)
    configs, groups = all_configs()
    features = get_features(paths, configs, args.cache_dir, args.feature_batch)

    screen = []
    for index, cfg in enumerate(configs, 1):
        print(f"screen {index}/{len(configs)} {cfg.key}", flush=True)
        run, _ = train_one(
            features[cfg.key], labels, split, cfg, 0,
            args.screen_epochs, args.samples_per_epoch, args.train_batch, False,
        )
        screen.append(run)

    by_key = {run["key"]: run for run in screen}
    selected = {}
    for group, keys in groups.items():
        candidates = [by_key[key] for key in keys]
        winner = max(
            candidates,
            key=lambda run: (
                run["validation"]["f1"],
                run["validation"]["accuracy"],
                -run["memory"]["learned_parameter_bytes_int8"],
                -run["memory"]["packed_input_bytes"],
            ),
        )
        selected[group] = winner["key"]

    config_by_key = {cfg.key: cfg for cfg in configs}
    selected_configs = {BASELINE.key: BASELINE}
    for key in selected.values():
        selected_configs[key] = config_by_key[key]
    # Confirm nearby Pareto candidates as well as the single-seed winners.
    # This prevents a noisy screening maximum from being reported as an
    # optimum and provides enough points to describe each relationship.
    confirmation_candidates = [
        replace(BASELINE, bits=value) for value in (3, 4, 6)
    ] + [
        replace(BASELINE, bands=value) for value in (8, 20, 32)
    ] + [
        replace(BASELINE, high_hz=value) for value in (6500, 7600)
    ] + [
        replace(BASELINE, duration_ms=value) for value in (600, 1200)
    ]
    for cfg in confirmation_candidates:
        selected_configs[cfg.key] = cfg
    combined = FeatureConfig(
        bits=config_by_key[selected["bits"]].bits,
        bands=config_by_key[selected["bands"]].bands,
        low_hz=config_by_key[selected["range"]].low_hz,
        high_hz=config_by_key[selected["range"]].high_hz,
        duration_ms=config_by_key[selected["duration"]].duration_ms,
    )
    selected_configs[combined.key] = combined
    selected_range = config_by_key[selected["range"]]
    quality_combined = FeatureConfig(
        bits=8,
        bands=20,
        low_hz=selected_range.low_hz,
        high_hz=selected_range.high_hz,
        duration_ms=1000,
    )
    compact_combined = replace(quality_combined, bits=4)
    deployment_compact = FeatureConfig(
        bits=4,
        bands=8,
        low_hz=BASELINE.low_hz,
        high_hz=BASELINE.high_hz,
        duration_ms=BASELINE.duration_ms,
    )
    selected_configs[quality_combined.key] = quality_combined
    selected_configs[compact_combined.key] = compact_combined
    selected_configs[deployment_compact.key] = deployment_compact
    missing_confirmation_features = [
        cfg for cfg in selected_configs.values() if cfg.key not in features
    ]
    if missing_confirmation_features:
        features.update(
            get_features(
                paths,
                missing_confirmation_features,
                args.cache_dir,
                args.feature_batch,
            )
        )

    confirmation = []
    checkpoint_dir = args.output.parent / f"{args.output.stem}_models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for cfg in selected_configs.values():
        for seed in (0, 1, 2):
            print(f"confirm {cfg.key} seed={seed}", flush=True)
            run, state = train_one(
                features[cfg.key], labels, split, cfg, seed,
                args.confirm_epochs, args.samples_per_epoch, args.train_batch, True,
            )
            confirmation.append(run)
            torch.save(
                {"state_dict": state, "run": run},
                checkpoint_dir / f"{cfg.key}_seed{seed}.pt",
            )

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": str(args.dataset.resolve()),
        "dataset_manifest_sha256": source_digest(paths),
        "split": split_summary(labels, split, speakers),
        "fixed": {
            "keyword": args.keyword,
            "time_bins": TIME_BINS,
            "hidden": HIDDEN,
            "timesteps": TIMESTEPS,
            "rms_gate": RMS_GATE,
            "rms_full_scale": RMS_FULL_SCALE,
            "selection_metric": "validation_f1",
        },
        "sweep_groups": groups,
        "screening": screen,
        "selected": selected,
        "confirmation_candidates": list(selected_configs),
        "combined": combined.key,
        "quality_combined": quality_combined.key,
        "compact_combined": compact_combined.key,
        "deployment_compact": deployment_compact.key,
        "confirmation": confirmation,
        "limitations": [
            "The experiment uses all keyword clips but a deterministic subset of non-keyword clips.",
            "Screening uses one seed; three-seed confirmation is limited to validation-selected candidates.",
            "Packed input bytes are a logical lower bound; the current Ethernet/BRAM ABI stores one byte per feature.",
            "Latency scales are based on input MAC count until finalists are measured on the physical board.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
