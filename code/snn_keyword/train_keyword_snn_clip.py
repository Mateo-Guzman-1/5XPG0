#!/usr/bin/env python3
"""Train and export the Group-2 single-keyword SNN.

By default this downloads TensorFlow's small Speech Commands subset, trains a
balanced ``yes`` versus ``not-yes`` classifier, validates both floating-point
and the exact integer recurrence used by the PicoRV32 firmware, and writes a C
header containing quantized weights.

Use ``--synthetic`` for a quick dependency/test smoke run without a download.
"""

from __future__ import annotations

import argparse
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from snntorch import surrogate

from audio_features import (
    N_INPUTS,
    N_MELS,
    N_TIME,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    extract_features,
    read_wav,
)


DATA_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "mini_speech_commands.zip"
)
LABELS = ("down", "go", "left", "no", "right", "stop", "up", "yes")
BETA = 0.9
BETA_Q8 = round(BETA * 256)
WEIGHT_SCALE = 64
DEFAULT_DECISION_MARGIN = 2
DEFAULT_MIN_KEYWORD_SPIKES = 2


class SpikeMLP(nn.Module):
    """Two-layer SNN with an explicit soft-reset LIF recurrence."""

    def __init__(self, hidden: int, timesteps: int):
        super().__init__()
        self.hidden = hidden
        self.timesteps = timesteps
        self.fc1 = nn.Linear(N_INPUTS, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.spike = surrogate.fast_sigmoid(slope=25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem1 = torch.zeros((x.shape[0], self.hidden), device=x.device)
        mem2 = torch.zeros((x.shape[0], 2), device=x.device)
        count = torch.zeros_like(mem2)
        cur1 = self.fc1(x)
        for _ in range(self.timesteps):
            mem1 = BETA * mem1 + cur1
            spk1 = self.spike(mem1 - 1.0)
            mem1 = mem1 - spk1
            mem2 = BETA * mem2 + self.fc2(spk1)
            spk2 = self.spike(mem2 - 1.0)
            mem2 = mem2 - spk2
            count = count + spk2
        return count


def download_dataset(data_dir: Path) -> Path:
    root = data_dir / "mini_speech_commands"
    if root.is_dir():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "mini_speech_commands.zip"
    if not archive.is_file():
        print(f"downloading {DATA_URL}")

        def progress(blocks, block_size, total):
            if total > 0 and blocks % 128 == 0:
                print(f"  {min(100.0, blocks * block_size * 100 / total):5.1f}%", end="\r")

        urllib.request.urlretrieve(DATA_URL, archive, reporthook=progress)
        print("  100.0%")
    print(f"extracting {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    return root


def load_feature_cache(root: Path, data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # v2 records the energy-aware frontend.  Keeping a version in the name
    # prevents an old peak-normalised cache from silently training a new model.
    cache = data_dir / "mini_speech_commands_8x16_v3.npz"
    if cache.is_file():
        d = np.load(cache, allow_pickle=False)
        return d["x"], d["labels"], d["paths"]

    paths = [p for label in LABELS for p in sorted((root / label).glob("*.wav"))]
    if not paths:
        raise RuntimeError(f"no WAV files found under {root}")
    x = np.empty((len(paths), N_INPUTS), dtype=np.float32)
    labels = np.empty(len(paths), dtype="<U8")
    relative = np.empty(len(paths), dtype="<U96")
    for i, path in enumerate(paths):
        x[i] = extract_features(read_wav(path)).reshape(-1)
        labels[i] = path.parent.name
        relative[i] = str(path.relative_to(root))
        if (i + 1) % 500 == 0 or i + 1 == len(paths):
            print(f"features {i + 1}/{len(paths)}")
    np.savez_compressed(cache, x=x, labels=labels, paths=relative)
    return x, labels, relative


def real_dataset(
    data_dir: Path,
    keyword: str,
    max_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    root = download_dataset(data_dir)
    x, labels, paths = load_feature_cache(root, data_dir)
    if keyword not in set(labels.tolist()):
        raise ValueError(f"keyword {keyword!r} not in dataset labels")

    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == keyword)
    neg = np.flatnonzero(labels != keyword)
    n = min(max_per_class, len(pos), len(neg))
    selected = np.concatenate(
        [rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]
    )
    rng.shuffle(selected)
    return x[selected], (labels[selected] == keyword).astype(np.int64), paths[selected], root


def synthetic_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n, dtype=np.int64)
    x = rng.random((n, N_MELS, N_TIME), dtype=np.float32) * 0.35
    for i, label in enumerate(y):
        if label:
            x[i, 1:4, 5:12] += 0.65
        else:
            x[i, 4:7, 2:9] += 0.65
    return np.clip(x, 0, 1).reshape(n, -1), y, np.full(n, "synthetic", dtype="<U16")


def split_dataset(x, y, paths, seed: int, val_fraction: float = 0.2):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return (
        x[train_idx], y[train_idx], paths[train_idx],
        x[val_idx], y[val_idx], paths[val_idx],
    )


def fit_audio_window(wav: np.ndarray) -> np.ndarray:
    """Match the live client's one-second window before augmentation."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size >= WINDOW_SAMPLES:
        start = (wav.size - WINDOW_SAMPLES) // 2
        return wav[start:start + WINDOW_SAMPLES].copy()
    padding = WINDOW_SAMPLES - wav.size
    left = padding // 2
    return np.pad(wav, (left, padding - left))


def generated_noise(rng: np.random.Generator, target_rms: float) -> np.ndarray:
    """Generate white, low-frequency, or mains-like background noise."""
    kind = int(rng.integers(0, 3))
    white = rng.standard_normal(WINDOW_SAMPLES).astype(np.float32)
    if kind == 0:
        noise = white
    elif kind == 1:
        # A vectorised moving average gives fan/room-like low-frequency noise.
        width = int(rng.integers(16, 65))
        csum = np.concatenate(([0.0], np.cumsum(white, dtype=np.float64)))
        smooth = (csum[width:] - csum[:-width]) / width
        noise = np.pad(smooth, (width - 1, 0), mode="edge").astype(np.float32)
    else:
        t = np.arange(WINDOW_SAMPLES, dtype=np.float32) / SAMPLE_RATE
        fundamental = float(rng.choice((50.0, 60.0)))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        noise = (
            np.sin(2.0 * np.pi * fundamental * t + phase)
            + 0.35 * np.sin(4.0 * np.pi * fundamental * t + phase)
            + 0.20 * white
        ).astype(np.float32)
    rms = float(np.sqrt(np.mean(noise * noise)))
    return noise * (target_rms / max(rms, 1e-8))


def add_noise_at_snr(
    wav: np.ndarray,
    rng: np.random.Generator,
    snr_db: float,
) -> np.ndarray:
    wav = fit_audio_window(wav)
    signal_rms = float(np.sqrt(np.mean(wav * wav)))
    noise_rms = np.clip(signal_rms / (10.0 ** (snr_db / 20.0)), 0.0005, 0.05)
    return np.clip(wav + generated_noise(rng, float(noise_rms)), -1.0, 1.0)


def augment_waveform(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply gain, time shift, and 8-25 dB background noise."""
    wav = fit_audio_window(wav)
    gain_db = float(rng.uniform(-6.0, 4.0))
    wav = wav * (10.0 ** (gain_db / 20.0))
    shift = int(rng.integers(-1600, 1601))  # up to 100 ms, no wrap-around
    if shift > 0:
        wav = np.pad(wav[:-shift], (shift, 0))
    elif shift < 0:
        wav = np.pad(wav[-shift:], (0, -shift))
    return add_noise_at_snr(wav, rng, float(rng.uniform(8.0, 25.0)))


def add_training_robustness(
    x: np.ndarray,
    y: np.ndarray,
    paths: np.ndarray,
    root: Path,
    seed: int,
    augmented_copies: int,
    background_examples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add noisy utterances plus explicit silence/background negatives."""
    rng = np.random.default_rng(seed + 101)
    feature_sets = [x]
    label_sets = [y]
    for copy in range(augmented_copies):
        augmented = np.empty_like(x)
        for i, relative in enumerate(paths):
            wav = read_wav(root / str(relative))
            augmented[i] = extract_features(augment_waveform(wav, rng)).reshape(-1)
        feature_sets.append(augmented)
        label_sets.append(y.copy())
        print(f"noise augmentation copy {copy + 1}/{augmented_copies}")

    # Half are exact/near silence; half span quiet-room through loud fan noise.
    n_silence = background_examples // 2
    n_noise = background_examples - n_silence
    backgrounds = np.zeros((background_examples, N_INPUTS), dtype=np.float32)
    for i in range(n_silence):
        rms = float(10.0 ** rng.uniform(-5.0, -2.6))
        backgrounds[i] = extract_features(generated_noise(rng, rms)).reshape(-1)
    for i in range(n_silence, n_silence + n_noise):
        rms = float(10.0 ** rng.uniform(-2.5, -1.25))
        backgrounds[i] = extract_features(generated_noise(rng, rms)).reshape(-1)
    feature_sets.append(backgrounds)
    label_sets.append(np.zeros(background_examples, dtype=np.int64))
    return np.concatenate(feature_sets), np.concatenate(label_sets)


def robustness_sets(
    root: Path,
    paths: np.ndarray,
    seed: int,
    background_examples: int = 400,
) -> dict[str, np.ndarray]:
    """Build deterministic noisy-speech and background-only test inputs."""
    result = {}
    for snr_db in (10, 5):
        rng = np.random.default_rng(seed + 1000 + snr_db)
        noisy = np.empty((len(paths), N_INPUTS), dtype=np.float32)
        for i, relative in enumerate(paths):
            wav = read_wav(root / str(relative))
            noisy[i] = extract_features(add_noise_at_snr(wav, rng, snr_db)).reshape(-1)
        result[f"noise_{snr_db}db"] = noisy

    rng = np.random.default_rng(seed + 2000)
    background = np.zeros((background_examples, N_INPUTS), dtype=np.float32)
    for i in range(background_examples):
        # Include exact silence and a log-uniform range of realistic noise.
        if i % 4:
            rms = float(10.0 ** rng.uniform(-4.0, -1.25))
            background[i] = extract_features(generated_noise(rng, rms)).reshape(-1)
    result["background"] = background
    return result


def decisions_from_counts(
    counts: np.ndarray,
    decision_margin: int,
    min_keyword_spikes: int,
) -> np.ndarray:
    return (
        (counts[:, 1] >= min_keyword_spikes)
        & (counts[:, 1] >= counts[:, 0] + decision_margin)
    ).astype(np.int64)


@torch.no_grad()
def float_counts(net: SpikeMLP, x: np.ndarray, batch: int) -> np.ndarray:
    net.eval()
    out = []
    for start in range(0, len(x), batch):
        t = torch.from_numpy(x[start:start + batch])
        out.append(net(t).cpu().numpy())
    return np.concatenate(out)


def quantized_parameters(net: SpikeMLP) -> dict[str, np.ndarray]:
    w1 = np.rint(net.fc1.weight.detach().cpu().numpy() * WEIGHT_SCALE)
    w2 = np.rint(net.fc2.weight.detach().cpu().numpy() * WEIGHT_SCALE)
    if np.abs(w1).max() > 127 or np.abs(w2).max() > 127:
        raise RuntimeError("weight exceeds int8 range; reduce WEIGHT_SCALE")
    return {
        "w1": w1.astype(np.int8),
        "b1": np.rint(
            net.fc1.bias.detach().cpu().numpy() * WEIGHT_SCALE * 255
        ).astype(np.int32),
        "w2": w2.astype(np.int8),
        "b2": np.rint(
            net.fc2.bias.detach().cpu().numpy() * WEIGHT_SCALE
        ).astype(np.int32),
    }


def quantized_predictions(
    params: dict[str, np.ndarray],
    x: np.ndarray,
    timesteps: int,
    decision_margin: int,
    min_keyword_spikes: int,
) -> tuple[np.ndarray, np.ndarray]:
    qx = np.rint(np.clip(x, 0, 1) * 255).astype(np.int32)
    cur1 = qx @ params["w1"].astype(np.int32).T + params["b1"]
    mem1 = np.zeros_like(cur1)
    mem2 = np.zeros((len(x), 2), dtype=np.int32)
    counts = np.zeros_like(mem2)
    th1 = WEIGHT_SCALE * 255
    th2 = WEIGHT_SCALE
    for _ in range(timesteps):
        mem1 = ((mem1 * BETA_Q8) >> 8) + cur1
        spk1 = mem1 >= th1
        mem1 -= spk1.astype(np.int32) * th1
        cur2 = spk1.astype(np.int32) @ params["w2"].astype(np.int32).T + params["b2"]
        mem2 = ((mem2 * BETA_Q8) >> 8) + cur2
        spk2 = mem2 >= th2
        mem2 -= spk2.astype(np.int32) * th2
        counts += spk2.astype(np.int32)
    return decisions_from_counts(counts, decision_margin, min_keyword_spikes), counts


def c_values(a: np.ndarray, per_line: int = 16) -> str:
    values = [str(int(v)) for v in a.reshape(-1)]
    return "\n".join(
        "    " + ", ".join(values[i:i + per_line]) + ","
        for i in range(0, len(values), per_line)
    )


def export_header(
    path: Path,
    params: dict[str, np.ndarray],
    keyword: str,
    hidden: int,
    timesteps: int,
    decision_margin: int,
    min_keyword_spikes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""// Generated by train_keyword_snn_clip.py. Do not edit by hand.
#ifndef KEYWORD_MODEL_H
#define KEYWORD_MODEL_H

#define KW_KEYWORD \"{keyword}\"
#define KW_INPUTS {N_INPUTS}
#define KW_HIDDEN {hidden}
#define KW_OUTPUTS 2
#define KW_TIMESTEPS {timesteps}
#define KW_BETA_Q8 {BETA_Q8}
#define KW_WEIGHT_SCALE {WEIGHT_SCALE}
#define KW_THRESHOLD1 {WEIGHT_SCALE * 255}
#define KW_THRESHOLD2 {WEIGHT_SCALE}
#define KW_DECISION_MARGIN {decision_margin}
#define KW_MIN_KEYWORD_SPIKES {min_keyword_spikes}

static const int8_t kw_w1[KW_HIDDEN * KW_INPUTS] = {{
{c_values(params['w1'])}
}};

static const int32_t kw_b1[KW_HIDDEN] = {{
{c_values(params['b1'], 8)}
}};

static const int8_t kw_w2[KW_OUTPUTS * KW_HIDDEN] = {{
{c_values(params['w2'])}
}};

static const int32_t kw_b2[KW_OUTPUTS] = {{
{c_values(params['b2'], 8)}
}};

#endif
"""
    path.write_text(text, encoding="ascii", newline="\n")


def accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float((pred == truth).mean())


def main() -> int:
    base = Path(__file__).resolve().parent
    default_header = base.parent / "pynqz2_riscv_flow" / "firmware" / "keyword_model.h"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keyword", default="yes", choices=LABELS)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--timesteps", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-per-class", type=int, default=800)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--augmented-copies", type=int, default=1)
    p.add_argument("--background-examples", type=int, default=600)
    p.add_argument("--decision-margin", type=int, default=DEFAULT_DECISION_MARGIN)
    p.add_argument(
        "--min-keyword-spikes", type=int, default=DEFAULT_MIN_KEYWORD_SPIKES
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=Path, default=base / "data")
    p.add_argument("--out-dir", type=Path, default=base / "runs")
    p.add_argument("--header", type=Path, default=default_header)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.synthetic:
        x, y, paths = synthetic_dataset(1024, args.seed)
        dataset_root = None
    else:
        x, y, paths, dataset_root = real_dataset(
            args.data_dir, args.keyword, args.max_per_class, args.seed
        )
    x_train, y_train, train_paths, x_val, y_val, val_paths = split_dataset(
        x, y, paths, args.seed
    )
    robustness = {}
    clean_train_count = len(x_train)
    if dataset_root:
        x_train, y_train = add_training_robustness(
            x_train,
            y_train,
            train_paths,
            dataset_root,
            args.seed,
            args.augmented_copies,
            args.background_examples,
        )
        robustness = robustness_sets(dataset_root, val_paths, args.seed)
    print(
        f"train={len(x_train)} (clean={clean_train_count}) val={len(x_val)} "
        f"keyword={args.keyword!r}"
    )

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    net = SpikeMLP(args.hidden, args.timesteps)
    opt = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * class_counts)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        for xb, yb in loader:
            scores = net(xb)
            loss = loss_fn(scores, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yb)
        val_counts = float_counts(net, x_val, args.batch_size)
        pred = decisions_from_counts(
            val_counts, args.decision_margin, args.min_keyword_spikes
        )
        print(
            f"epoch {epoch + 1:02d}/{args.epochs}  "
            f"loss={total / len(ds):.4f}  val_acc={accuracy(pred, y_val):.3f}"
        )

    params = quantized_parameters(net)
    float_val_counts = float_counts(net, x_val, args.batch_size)
    float_pred = decisions_from_counts(
        float_val_counts, args.decision_margin, args.min_keyword_spikes
    )
    int_pred, counts = quantized_predictions(
        params,
        x_val,
        args.timesteps,
        args.decision_margin,
        args.min_keyword_spikes,
    )
    float_acc = accuracy(float_pred, y_val)
    int_acc = accuracy(int_pred, y_val)
    agreement = accuracy(int_pred, float_pred)
    print(
        f"validation: float={float_acc:.3f} integer={int_acc:.3f} "
        f"float/int agreement={agreement:.3f}"
    )

    robustness_metrics = {}
    for name, robust_x in robustness.items():
        robust_pred, _ = quantized_predictions(
            params,
            robust_x,
            args.timesteps,
            args.decision_margin,
            args.min_keyword_spikes,
        )
        if name == "background":
            metric = float(robust_pred.mean())
            robustness_metrics["background_false_positive_rate"] = metric
            print(f"robustness: background false-positive rate={metric:.4f}")
        else:
            metric = accuracy(robust_pred, y_val)
            robustness_metrics[f"{name}_accuracy"] = metric
            print(f"robustness: {name} integer accuracy={metric:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "keyword_snn.pt"
    torch.save(
        {
            "state_dict": net.state_dict(),
            "keyword": args.keyword,
            "hidden": args.hidden,
            "timesteps": args.timesteps,
            "feature_shape": (N_MELS, N_TIME),
            "float_val_accuracy": float_acc,
            "integer_val_accuracy": int_acc,
            "decision_margin": args.decision_margin,
            "min_keyword_spikes": args.min_keyword_spikes,
            "robustness": robustness_metrics,
        },
        model_path,
    )
    export_header(
        args.header,
        params,
        args.keyword,
        args.hidden,
        args.timesteps,
        args.decision_margin,
        args.min_keyword_spikes,
    )
    np.savez_compressed(
        args.out_dir / "validation_examples.npz",
        x=np.rint(x_val * 255).astype(np.uint8),
        y=y_val,
        counts=counts,
        paths=val_paths,
    )
    print(f"saved {model_path}")
    print(f"exported {args.header}")
    if dataset_root:
        print(f"dataset root: {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
