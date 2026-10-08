#!/usr/bin/env python3
"""Six additional Sheila frontend sweeps requested after the main study.

The experiment keeps the dataset split and SNN controls from frontend_sweep.py
while varying time bins, activity scaling, frequency-band mapping, STFT
resolution, four-bit quantization, and temporal-delta features.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

import frontend_sweep as base


SAMPLE_RATE = 16_000
HIDDEN = base.HIDDEN
TIMESTEPS = base.TIMESTEPS


@dataclass(frozen=True)
class EncodingConfig:
    bits: int = 8
    bands: int = 8
    time_bins: int = 16
    low_hz: int = 40
    high_hz: int = 5120
    duration_ms: int = 1000
    rms_gate_u6: int = 3000
    rms_full_scale_u6: int = 40_000
    band_mapping: str = "equal"
    fft_window_ms: int = 25
    fft_hop_ms: int = 10
    quantizer: str = "linear"
    feature_mode: str = "absolute"

    @property
    def rms_gate(self) -> float:
        return self.rms_gate_u6 / 1_000_000.0

    @property
    def rms_full_scale(self) -> float:
        return self.rms_full_scale_u6 / 1_000_000.0

    @property
    def channels(self) -> int:
        return 2 if self.feature_mode == "absolute_delta" else 1

    @property
    def board_rows(self) -> int:
        return self.bands * self.channels

    @property
    def inputs(self) -> int:
        return self.board_rows * self.time_bins

    @property
    def key(self) -> str:
        return (
            f"b{self.bits}_f{self.bands}_tb{self.time_bins}_"
            f"g{self.rms_gate_u6}_s{self.rms_full_scale_u6}_"
            f"map{self.band_mapping}_w{self.fft_window_ms}_h{self.fft_hop_ms}_"
            f"q{self.quantizer}_m{self.feature_mode}"
        )


BASELINE = EncodingConfig()
TIME_BINS = (8, 12, 16, 24, 32)
RMS_GATES_U6 = (1500, 3000, 5000, 8000)
RMS_FULL_SCALES_U6 = (20_000, 40_000, 80_000)
BAND_MAPPINGS = ("equal", "log", "mel")
FFT_PAIRS_MS = (
    (15, 5), (15, 10),
    (25, 5), (25, 10), (25, 20),
    (40, 5), (40, 10), (40, 20),
)
QUANTIZERS = ("linear", "log")
FEATURE_MODES = ("absolute", "absolute_delta")


def all_configs() -> tuple[list[EncodingConfig], dict[str, list[str]]]:
    groups: dict[str, list[EncodingConfig]] = {
        "time_bins": [replace(BASELINE, time_bins=value) for value in TIME_BINS],
        "activity": [
            replace(
                BASELINE,
                rms_gate_u6=gate,
                rms_full_scale_u6=full_scale,
            )
            for gate in RMS_GATES_U6
            for full_scale in RMS_FULL_SCALES_U6
        ],
        "band_mapping": [
            replace(BASELINE, band_mapping=value) for value in BAND_MAPPINGS
        ],
        "fft": [
            replace(BASELINE, fft_window_ms=window, fft_hop_ms=hop)
            for window, hop in FFT_PAIRS_MS
        ],
        "quantizer": [
            replace(BASELINE, bits=4, quantizer=value) for value in QUANTIZERS
        ],
        "feature_mode": [
            replace(BASELINE, feature_mode=value) for value in FEATURE_MODES
        ],
    }
    unique: dict[str, EncodingConfig] = {BASELINE.key: BASELINE}
    for values in groups.values():
        for cfg in values:
            unique[cfg.key] = cfg
    return list(unique.values()), {
        name: [cfg.key for cfg in values] for name, values in groups.items()
    }


def read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getframerate(),
            stream.getnchannels(),
            stream.getsampwidth(),
        ) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"expected mono 16 kHz PCM16: {path}")
        source = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    source = source.astype(np.float32) / 32768.0
    result = np.zeros(SAMPLE_RATE, dtype=np.float32)
    if source.size >= SAMPLE_RATE:
        start = (source.size - SAMPLE_RATE) // 2
        result[:] = source[start : start + SAMPLE_RATE]
    else:
        start = (SAMPLE_RATE - source.size) // 2
        result[start : start + source.size] = source
    return result


def fit_duration(batch: np.ndarray, duration_ms: int) -> np.ndarray:
    samples = duration_ms * SAMPLE_RATE // 1000
    if samples == SAMPLE_RATE:
        return batch
    if samples < SAMPLE_RATE:
        start = (SAMPLE_RATE - samples) // 2
        return batch[:, start : start + samples]
    padding = samples - SAMPLE_RATE
    left = padding // 2
    return np.pad(batch, ((0, 0), (left, padding - left)))


def hz_to_mel(value: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def mel_to_hz(value: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def band_weights(frequencies: np.ndarray, cfg: EncodingConfig) -> np.ndarray:
    """Return normalized rectangular or triangular band weights."""
    weights = np.zeros((cfg.bands, frequencies.size), dtype=np.float32)
    selected = np.flatnonzero(
        (frequencies >= cfg.low_hz) & (frequencies <= cfg.high_hz)
    )
    if selected.size < cfg.bands:
        raise ValueError(f"{cfg.key}: fewer FFT bins than bands")

    if cfg.band_mapping == "equal":
        for row, indices in enumerate(np.array_split(selected, cfg.bands)):
            weights[row, indices] = 1.0 / len(indices)
        return weights

    if cfg.band_mapping == "log":
        edges = np.geomspace(max(1.0, cfg.low_hz), cfg.high_hz, cfg.bands + 1)
        for row in range(cfg.bands):
            if row + 1 == cfg.bands:
                indices = np.flatnonzero(
                    (frequencies >= edges[row]) & (frequencies <= edges[row + 1])
                )
            else:
                indices = np.flatnonzero(
                    (frequencies >= edges[row]) & (frequencies < edges[row + 1])
                )
            if indices.size == 0:
                indices = np.array([np.abs(frequencies - np.sqrt(edges[row] * edges[row + 1])).argmin()])
            weights[row, indices] = 1.0 / len(indices)
        return weights

    if cfg.band_mapping != "mel":
        raise ValueError(f"unknown band mapping: {cfg.band_mapping}")
    mel_edges = np.linspace(hz_to_mel(cfg.low_hz), hz_to_mel(cfg.high_hz), cfg.bands + 2)
    hz_edges = mel_to_hz(mel_edges)
    for row in range(cfg.bands):
        left, center, right = hz_edges[row : row + 3]
        rising = (frequencies - left) / max(center - left, 1e-12)
        falling = (right - frequencies) / max(right - center, 1e-12)
        triangular = np.maximum(0.0, np.minimum(rising, falling))
        total = triangular.sum()
        if total <= 0.0:
            raise ValueError(f"{cfg.key}: empty mel filter {row}")
        weights[row] = triangular / total
    return weights


def quantize(values: np.ndarray, cfg: EncodingConfig) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    levels = (1 << cfg.bits) - 1
    if cfg.quantizer == "linear":
        return np.rint(values * levels) / levels
    if cfg.quantizer != "log":
        raise ValueError(f"unknown quantizer: {cfg.quantizer}")
    mu = float(levels)
    compressed = np.log1p(mu * values) / np.log1p(mu)
    indices = np.rint(compressed * levels)
    return np.expm1(indices / levels * np.log1p(mu)) / mu


def feature_batch(audio: np.ndarray, cfg: EncodingConfig) -> np.ndarray:
    fitted = fit_duration(audio, cfg.duration_ms)
    rms = np.sqrt(np.mean(fitted * fitted, axis=1))
    activity = np.zeros_like(rms)
    active = rms > cfg.rms_gate
    activity[active] = np.clip(
        np.log(rms[active] / cfg.rms_gate)
        / np.log(cfg.rms_full_scale / cfg.rms_gate),
        0.0,
        1.0,
    )

    window_samples = cfg.fft_window_ms * SAMPLE_RATE // 1000
    hop_samples = cfg.fft_hop_ms * SAMPLE_RATE // 1000
    frames = np.lib.stride_tricks.sliding_window_view(
        fitted, window_samples, axis=1
    )[:, ::hop_samples, :]
    window = np.hanning(window_samples).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=2)).astype(np.float32)
    frequencies = np.fft.rfftfreq(window_samples, 1.0 / SAMPLE_RATE)
    weights = band_weights(frequencies, cfg)
    band_values = np.einsum("bf,ntf->nbt", weights, spectrum, optimize=True)
    band_values = np.log1p(band_values)

    frame_groups = np.array_split(np.arange(band_values.shape[2]), cfg.time_bins)
    pooled = np.stack(
        [band_values[:, :, group].mean(axis=2) for group in frame_groups],
        axis=2,
    )
    peak = pooled.max(axis=(1, 2), keepdims=True)
    pooled = np.divide(pooled, peak, out=np.zeros_like(pooled), where=peak > 0)
    pooled *= activity[:, None, None]

    if cfg.feature_mode == "absolute_delta":
        delta = np.diff(pooled, axis=2, prepend=pooled[:, :, :1])
        pooled = np.concatenate((pooled, np.abs(delta)), axis=1)
    elif cfg.feature_mode != "absolute":
        raise ValueError(f"unknown feature mode: {cfg.feature_mode}")

    pooled = quantize(pooled, cfg)
    return pooled.reshape(len(audio), -1).astype(np.float32)


def cache_path(cache_dir: Path, cfg: EncodingConfig) -> Path:
    return cache_dir / f"{cfg.key}.npy"


def get_features(
    paths: list[Path],
    configs: list[EncodingConfig],
    cache_dir: Path,
    batch_size: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    missing: list[EncodingConfig] = []
    for cfg in configs:
        path = cache_path(cache_dir, cfg)
        if path.is_file():
            values = np.load(path, mmap_mode="r")
            if values.shape == (len(paths), cfg.inputs):
                result[cfg.key] = values
                continue
        missing.append(cfg)
    if not missing:
        return result

    generated = {
        cfg.key: np.empty((len(paths), cfg.inputs), dtype=np.float32)
        for cfg in missing
    }
    for start in range(0, len(paths), batch_size):
        end = min(len(paths), start + batch_size)
        audio = np.stack([read_pcm16(path) for path in paths[start:end]])
        for cfg in missing:
            generated[cfg.key][start:end] = feature_batch(audio, cfg)
        print(f"features {end}/{len(paths)}", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for cfg in missing:
        path = cache_path(cache_dir, cfg)
        np.save(path, generated[cfg.key])
        result[cfg.key] = np.load(path, mmap_mode="r")
    return result


def memory_model(cfg: EncodingConfig) -> dict[str, int]:
    parameters = HIDDEN * cfg.inputs + 4 * HIDDEN + 2 * HIDDEN + 8
    return {
        "learned_parameter_bytes_int8": parameters,
        "byte_aligned_input_bytes": cfg.inputs,
        "packed_input_bytes": math.ceil(cfg.inputs * cfg.bits / 8),
        "logical_input_bits": cfg.inputs * cfg.bits,
        "input_mac_count": HIDDEN * cfg.inputs,
    }


def train_one(
    features: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    cfg: EncodingConfig,
    seed: int,
    epochs: int,
    samples_per_epoch: int,
    batch_size: int,
    evaluate_test: bool,
) -> tuple[dict, dict[str, torch.Tensor]]:
    run, state = base.train_one(
        features,
        labels,
        split,
        cfg,
        seed,
        epochs,
        samples_per_epoch,
        batch_size,
        evaluate_test,
    )
    run["memory"] = memory_model(cfg)
    return run, state


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=here / "data" / "speech_commands_v2_sheila_subset")
    parser.add_argument("--keyword", default="sheila")
    parser.add_argument("--cache-dir", type=Path, default=here / "data" / "extended_encoding_cache_sheila_v2")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "extended_encoding_sweep_sheila_v2.json")
    parser.add_argument("--feature-batch", type=int, default=64)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--screen-epochs", type=int, default=8)
    parser.add_argument("--confirm-epochs", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    paths, labels, speakers = base.discover_dataset(args.dataset, args.keyword)
    split = base.dataset_split(args.dataset, paths, speakers)
    configs, groups = all_configs()
    by_key = {cfg.key: cfg for cfg in configs}
    features = get_features(paths, configs, args.cache_dir, args.feature_batch)

    checkpoint_dir = args.output.parent / f"{args.output.stem}_models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    screening = []
    for index, cfg in enumerate(configs, 1):
        print(f"screen {index}/{len(configs)} {cfg.key}", flush=True)
        run, state = train_one(
            features[cfg.key], labels, split, cfg, 0,
            args.screen_epochs, args.samples_per_epoch, args.train_batch, False,
        )
        screening.append(run)
        torch.save(
            {"state_dict": state, "run": run},
            checkpoint_dir / f"{cfg.key}_screen_seed0.pt",
        )

    screen_by_key = {run["key"]: run for run in screening}
    selected: dict[str, str] = {}
    for group, keys in groups.items():
        selected[group] = max(
            keys,
            key=lambda key: (
                screen_by_key[key]["validation"]["f1"],
                screen_by_key[key]["validation"]["accuracy"],
                -screen_by_key[key]["memory"]["learned_parameter_bytes_int8"],
                -screen_by_key[key]["memory"]["logical_input_bits"],
            ),
        )

    confirmation_keys = list(dict.fromkeys([BASELINE.key, *selected.values()]))
    confirmation = []
    for key in confirmation_keys:
        cfg = by_key[key]
        for seed in (0, 1, 2):
            print(f"confirm {key} seed={seed}", flush=True)
            run, state = train_one(
                features[key], labels, split, cfg, seed,
                args.confirm_epochs, args.samples_per_epoch, args.train_batch, True,
            )
            confirmation.append(run)
            torch.save(
                {"state_dict": state, "run": run},
                checkpoint_dir / f"{key}_confirm_seed{seed}.pt",
            )

    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": str(args.dataset.resolve()),
        "split": base.split_summary(labels, split, speakers),
        "fixed": {
            "keyword": args.keyword,
            "hidden": HIDDEN,
            "timesteps": TIMESTEPS,
            "low_hz": BASELINE.low_hz,
            "high_hz": BASELINE.high_hz,
            "duration_ms": BASELINE.duration_ms,
            "selection_metric": "validation_f1",
            "delta_definition": "absolute magnitude of first temporal difference",
            "log_quantizer_mu": 15,
        },
        "baseline": BASELINE.key,
        "sweep_groups": groups,
        "screening": screening,
        "selected": selected,
        "confirmation_candidates": confirmation_keys,
        "confirmation": confirmation,
        "limitations": [
            "Screening uses one seed; only validation-selected winners are confirmed over three seeds.",
            "The board receives precomputed features, so FFT settings affect host extraction rather than PicoRV32 inference cycles when input dimensions are unchanged.",
            "Four-bit inputs remain byte-aligned on the physical interface until packed transport is implemented.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
