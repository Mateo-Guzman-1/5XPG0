#!/usr/bin/env python3
"""Model-free and paired-model verification of the Sheila encoding sweeps.

The original sweep retrains an SNN for every encoding.  This script separates
encoding information from training variation with five complementary levels:

1. fidelity to one common high-resolution spectral reference;
2. retention of fixed time-frequency landmark proxies for ``sheila``;
3. deterministic class separability and a nearest-centroid probe;
4. cell-wise occlusion of the deployed SNN;
5. a ten-seed paired shortlist with saved predictions and McNemar tests.

The direct metrics use the exact feature implementations from the sweep
scripts.  No scikit-learn or scipy dependency is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

import extended_encoding_sweep as extended
import frontend_sweep as primary


REFERENCE = extended.EncodingConfig(
    bits=16,
    bands=64,
    time_bins=64,
    low_hz=40,
    high_hz=7600,
    duration_ms=1600,
)

LANDMARKS = {
    "sh_frication": (2300.0, 7600.0, 0.00, 0.30),
    "ee_f1": (200.0, 500.0, 0.20, 0.62),
    "ee_f2": (2300.0, 2800.0, 0.20, 0.62),
    "l_transition": (500.0, 2000.0, 0.55, 0.84),
    "final_schwa": (300.0, 1800.0, 0.74, 1.00),
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def primary_to_extended(cfg: primary.FeatureConfig) -> extended.EncodingConfig:
    return extended.EncodingConfig(
        bits=cfg.bits,
        bands=cfg.bands,
        low_hz=cfg.low_hz,
        high_hz=cfg.high_hz,
        duration_ms=cfg.duration_ms,
    )


def configuration_catalog() -> tuple[list[extended.EncodingConfig], dict[str, list[str]]]:
    """Return the union of both published sweeps plus the two confound checks."""
    values: dict[str, extended.EncodingConfig] = {}
    groups: dict[str, list[str]] = {}

    primary_configs, primary_groups = primary.all_configs()
    primary_by_key = {cfg.key: primary_to_extended(cfg) for cfg in primary_configs}
    for cfg in primary_by_key.values():
        values[cfg.key] = cfg
    for name, keys in primary_groups.items():
        groups[f"primary_{name}"] = [primary_by_key[key].key for key in keys]

    extended_configs, extended_groups = extended.all_configs()
    for cfg in extended_configs:
        values[cfg.key] = cfg
    for name, keys in extended_groups.items():
        groups[f"extended_{name}"] = keys

    confounds = [
        replace(extended.BASELINE, bands=16, high_hz=6500),
        replace(extended.BASELINE, low_hz=40, high_hz=5120),
    ]
    for cfg in confounds:
        values[cfg.key] = cfg
    groups["confound_checks"] = [cfg.key for cfg in confounds]
    return list(values.values()), groups


def primary_compatible(cfg: extended.EncodingConfig) -> bool:
    return (
        cfg.time_bins == 16
        and cfg.rms_gate_u6 == 3000
        and cfg.rms_full_scale_u6 == 20_000
        and cfg.band_mapping == "equal"
        and cfg.fft_window_ms == 25
        and cfg.fft_hop_ms == 10
        and cfg.quantizer == "linear"
        and cfg.feature_mode == "absolute"
    )


class FeatureStore:
    def __init__(
        self,
        paths: list[Path],
        primary_cache: Path,
        extended_cache: Path,
        local_cache: Path,
        batch_size: int,
    ) -> None:
        self.paths = paths
        self.primary_cache = primary_cache
        self.extended_cache = extended_cache
        self.local_cache = local_cache
        self.batch_size = batch_size

    def _valid(self, path: Path, cfg: extended.EncodingConfig) -> np.ndarray | None:
        if not path.is_file():
            return None
        value = np.load(path, mmap_mode="r")
        if value.shape != (len(self.paths), cfg.inputs):
            return None
        return value

    def get(self, cfg: extended.EncodingConfig) -> np.ndarray:
        local = self._valid(self.local_cache / f"{cfg.key}.npy", cfg)
        if local is not None:
            return local

        existing = self._valid(self.extended_cache / f"{cfg.key}.npy", cfg)
        if existing is not None:
            return existing

        if primary_compatible(cfg):
            old = primary.FeatureConfig(
                bits=cfg.bits,
                bands=cfg.bands,
                low_hz=cfg.low_hz,
                high_hz=cfg.high_hz,
                duration_ms=cfg.duration_ms,
            )
            existing = self._valid(self.primary_cache / f"{old.key}.npy", cfg)
            if existing is not None:
                return existing

        return extended.get_features(
            self.paths, [cfg], self.local_cache, self.batch_size
        )[cfg.key]

    def materialize(self, cfg: extended.EncodingConfig) -> Path:
        """Put a unified-key cache in the local directory for board export."""
        destination = self.local_cache / f"{cfg.key}.npy"
        existing = self._valid(destination, cfg)
        if existing is None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, np.asarray(self.get(cfg)))
        return destination


def feature_axes(cfg: extended.EncodingConfig) -> tuple[np.ndarray, np.ndarray]:
    window = cfg.fft_window_ms * extended.SAMPLE_RATE // 1000
    hop = cfg.fft_hop_ms * extended.SAMPLE_RATE // 1000
    frequencies = np.fft.rfftfreq(window, 1.0 / extended.SAMPLE_RATE)
    weights = extended.band_weights(frequencies, cfg)
    frequency_centres = weights @ frequencies

    samples = cfg.duration_ms * extended.SAMPLE_RATE // 1000
    starts = np.arange(0, samples - window + 1, hop)
    groups = np.array_split(np.arange(len(starts)), cfg.time_bins)
    time_centres = np.array(
        [
            np.mean((starts[group] + window / 2) * 1000 / extended.SAMPLE_RATE)
            - cfg.duration_ms / 2
            for group in groups
        ],
        dtype=np.float64,
    )
    return frequency_centres, time_centres


def interpolate_last(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    clamped = np.clip(target, source[0], source[-1])
    upper = np.searchsorted(source, clamped, side="left")
    upper = np.clip(upper, 1, len(source) - 1)
    lower = upper - 1
    span = source[upper] - source[lower]
    weight = np.divide(
        clamped - source[lower], span, out=np.zeros_like(target), where=span > 0
    )
    result = values[..., lower] * (1.0 - weight) + values[..., upper] * weight
    outside = (target < lower_bound) | (target > upper_bound)
    result[..., outside] = 0.0
    return result


def reconstruct(
    features: np.ndarray,
    cfg: extended.EncodingConfig,
    reference_frequency: np.ndarray,
    reference_time: np.ndarray,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).reshape(
        len(features), cfg.board_rows, cfg.time_bins
    )
    values = values[:, : cfg.bands, :]
    frequency, time_axis = feature_axes(cfg)
    time_resampled = interpolate_last(
        values,
        time_axis,
        reference_time,
        -cfg.duration_ms / 2,
        cfg.duration_ms / 2,
    )
    frequency_resampled = interpolate_last(
        np.swapaxes(time_resampled, 1, 2),
        frequency,
        reference_frequency,
        cfg.low_hz,
        cfg.high_hz,
    )
    return np.swapaxes(frequency_resampled, 1, 2)


def mean_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    signal = np.sum(reference.astype(np.float64) ** 2, axis=tuple(range(1, reference.ndim)))
    error = np.sum(
        (reference.astype(np.float64) - estimate.astype(np.float64)) ** 2,
        axis=tuple(range(1, reference.ndim)),
    )
    valid = signal > 1e-12
    return float(np.mean(10.0 * np.log10((signal[valid] + 1e-12) / (error[valid] + 1e-12))))


def fidelity_metrics(
    encoded: np.ndarray,
    unquantized: np.ndarray,
    reconstructed: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    floor = 1.0 / 255.0
    ref_db = 20.0 * np.log10(np.clip(reference, floor, 1.0))
    rec_db = 20.0 * np.log10(np.clip(reconstructed, floor, 1.0))
    lsd = np.sqrt(np.mean((ref_db - rec_db) ** 2, axis=(1, 2)))

    ref_flat = reference.reshape(len(reference), -1).astype(np.float64)
    rec_flat = reconstructed.reshape(len(reconstructed), -1).astype(np.float64)
    ref_flat -= ref_flat.mean(axis=1, keepdims=True)
    rec_flat -= rec_flat.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(ref_flat, axis=1) * np.linalg.norm(rec_flat, axis=1)
    correlation = np.divide(
        np.sum(ref_flat * rec_flat, axis=1),
        denom,
        out=np.zeros(len(reference), dtype=np.float64),
        where=denom > 1e-12,
    )
    return {
        "reference_snr_db": mean_snr(reference, reconstructed),
        "log_spectral_distance_db": float(np.mean(lsd)),
        "reference_correlation": float(np.mean(correlation)),
        "quantization_snr_db": mean_snr(unquantized, encoded),
        "zero_cell_fraction": float(np.mean(encoded == 0.0)),
    }


def weighted_centroid(values: np.ndarray, axis: np.ndarray) -> float | None:
    total = float(values.sum())
    if total <= 1e-12:
        return None
    return float(np.sum(values * axis) / total)


def landmark_metrics(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    frequency: np.ndarray,
    time_axis: np.ndarray,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    envelopes = reference.mean(axis=1)
    spans = []
    for envelope in envelopes:
        active = np.flatnonzero(envelope >= 0.2 * envelope.max())
        spans.append((int(active[0]), int(active[-1])) if active.size else (0, len(time_axis) - 1))

    for name, (low_hz, high_hz, start_fraction, end_fraction) in LANDMARKS.items():
        contrast_similarity = []
        energy_ratios = []
        frequency_errors = []
        time_errors = []
        frequency_mask = (frequency >= low_hz) & (frequency <= high_hz)
        for index, (start, end) in enumerate(spans):
            length = max(1, end - start)
            left = start + round(start_fraction * length)
            right = start + round(end_fraction * length)
            right = max(left + 1, min(len(time_axis), right + 1))
            time_mask = np.zeros(len(time_axis), dtype=bool)
            time_mask[max(0, left) : right] = True

            ref_window = reference[index][np.ix_(frequency_mask, time_mask)]
            rec_window = reconstructed[index][np.ix_(frequency_mask, time_mask)]
            ref_signal = float(ref_window.mean()) if ref_window.size else 0.0
            rec_signal = float(rec_window.mean()) if rec_window.size else 0.0
            ref_context = float(reference[index][:, time_mask].mean())
            rec_context = float(reconstructed[index][:, time_mask].mean())
            ref_contrast = (ref_signal + 1e-6) / (ref_context + 1e-6)
            rec_contrast = (rec_signal + 1e-6) / (rec_context + 1e-6)
            contrast_similarity.append(
                math.exp(-abs(math.log(rec_contrast) - math.log(ref_contrast)))
            )
            energy_ratios.append(rec_signal / max(ref_signal, 1e-6))

            ref_frequency_energy = reference[index][:, time_mask].sum(axis=1)
            rec_frequency_energy = reconstructed[index][:, time_mask].sum(axis=1)
            ref_fc = weighted_centroid(ref_frequency_energy[frequency_mask], frequency[frequency_mask])
            rec_fc = weighted_centroid(rec_frequency_energy[frequency_mask], frequency[frequency_mask])
            frequency_errors.append(
                abs(ref_fc - rec_fc) if ref_fc is not None and rec_fc is not None else high_hz - low_hz
            )

            ref_time_energy = reference[index][frequency_mask, :].sum(axis=0)
            rec_time_energy = reconstructed[index][frequency_mask, :].sum(axis=0)
            ref_tc = weighted_centroid(ref_time_energy[time_mask], time_axis[time_mask])
            rec_tc = weighted_centroid(rec_time_energy[time_mask], time_axis[time_mask])
            time_errors.append(
                abs(ref_tc - rec_tc) if ref_tc is not None and rec_tc is not None else float(time_axis[-1] - time_axis[0])
            )

        results[name] = {
            "contrast_similarity": float(np.mean(contrast_similarity)),
            "median_energy_ratio": float(np.median(energy_ratios)),
            "mean_frequency_centroid_error_hz": float(np.mean(frequency_errors)),
            "mean_time_centroid_error_ms": float(np.mean(time_errors)),
        }
    return results


def choose_float_threshold(truth: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    ordered_truth = np.asarray(truth, dtype=bool)[order]
    ordered_score = np.asarray(score, dtype=np.float64)[order]
    tp = np.cumsum(ordered_truth)
    fp = np.cumsum(~ordered_truth)
    positives = int(ordered_truth.sum())
    last_of_tie = np.r_[ordered_score[1:] != ordered_score[:-1], True]
    indices = np.flatnonzero(last_of_tie)
    fn = positives - tp[indices]
    f1 = 2 * tp[indices] / np.maximum(1, 2 * tp[indices] + fp[indices] + fn)
    return float(ordered_score[indices[int(np.argmax(f1))]])


def speaker_bootstrap(
    truth: np.ndarray,
    prediction: np.ndarray,
    speakers: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, list[float]]:
    unique = np.array(sorted(set(speakers.tolist())))
    groups = {speaker: np.flatnonzero(speakers == speaker) for speaker in unique}
    rng = np.random.default_rng(seed)
    scores = {name: [] for name in ("accuracy", "f1", "balanced_accuracy")}
    for _ in range(repeats):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([groups[speaker] for speaker in chosen])
        measured = primary.metrics(truth[index], prediction[index])
        for name in scores:
            scores[name].append(float(measured[name]))
    return {
        name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for name, values in scores.items()
    }


def separability_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    speakers: np.ndarray,
    paths: list[Path],
    bootstrap_repeats: int,
) -> dict:
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "val")
    test = np.flatnonzero(split == "test")
    x_train = np.asarray(features[train], dtype=np.float64)
    y_train = labels[train]
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    useful = scale > 1e-6
    scale[~useful] = 1.0
    z_train = (x_train[:, useful] - mean[useful]) / scale[useful]
    negative = z_train[y_train == 0]
    positive = z_train[y_train == 1]
    mean_negative = negative.mean(axis=0)
    mean_positive = positive.mean(axis=0)
    var_negative = negative.var(axis=0) + 1e-6
    var_positive = positive.var(axis=0) + 1e-6
    pooled = 0.5 * (var_negative + var_positive)
    fisher = np.mean((mean_positive - mean_negative) ** 2 / (var_positive + var_negative))
    bhattacharyya = np.mean(
        0.125 * (mean_positive - mean_negative) ** 2 / pooled
        + 0.5 * np.log(pooled / np.sqrt(var_positive * var_negative))
    )

    def probe(index: np.ndarray) -> np.ndarray:
        values = (np.asarray(features[index], dtype=np.float64)[:, useful] - mean[useful]) / scale[useful]
        distance_negative = np.mean((values - mean_negative) ** 2, axis=1)
        distance_positive = np.mean((values - mean_positive) ** 2, axis=1)
        return distance_negative - distance_positive

    validation_score = probe(validation)
    threshold = choose_float_threshold(labels[validation], validation_score)
    test_score = probe(test)
    prediction = test_score >= threshold
    test_metrics = primary.metrics(labels[test], prediction)
    bootstrap = speaker_bootstrap(
        labels[test], prediction, speakers[test], bootstrap_repeats, 20261015
    )

    names = np.array([path.parent.name for path in paths], dtype=object)[test]
    confusable = []
    for name in sorted(set(names[labels[test] == 0].tolist())):
        selected = (names == name) & (labels[test] == 0)
        confusable.append(
            {
                "label": name,
                "clips": int(selected.sum()),
                "mean_probe_score": float(test_score[selected].mean()),
                "false_positive_rate": float(prediction[selected].mean()),
            }
        )
    confusable.sort(key=lambda item: (item["mean_probe_score"], item["false_positive_rate"]), reverse=True)
    return {
        "active_dimensions": int(useful.sum()),
        "fisher_ratio_mean": float(fisher),
        "bhattacharyya_distance_mean": float(bhattacharyya),
        "probe_threshold": threshold,
        "nearest_centroid_test": test_metrics,
        "speaker_bootstrap_95ci": bootstrap,
        "highest_scoring_negative_labels": confusable[:5],
    }


def exact_mcnemar(reference_correct: np.ndarray, candidate_correct: np.ndarray) -> dict:
    reference_only = int(np.sum(reference_correct & ~candidate_correct))
    candidate_only = int(np.sum(~reference_correct & candidate_correct))
    discordant = reference_only + candidate_only
    if discordant == 0:
        probability = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(reference_only, candidate_only) + 1)
        ) / (2 ** discordant)
        probability = min(1.0, 2.0 * tail)
    return {
        "baseline_only_correct": reference_only,
        "candidate_only_correct": candidate_only,
        "discordant": discordant,
        "exact_two_sided_p": float(probability),
    }


def mean_ci(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = 2.262 * std / math.sqrt(len(values)) if len(values) == 10 else 1.96 * std / math.sqrt(max(1, len(values)))
    mean = float(values.mean())
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def ten_seed_shortlist(
    shortlist: list[tuple[str, extended.EncodingConfig]],
    store: FeatureStore,
    labels: np.ndarray,
    split: np.ndarray,
    paths: list[Path],
    dataset: Path,
    seeds: int,
    epochs: int,
    samples_per_epoch: int,
    batch_size: int,
    model_dir: Path,
    predictions_path: Path,
    progress_path: Path,
) -> dict:
    validation = np.flatnonzero(split == "val")
    test = np.flatnonzero(split == "test")
    all_margins = np.empty((len(shortlist), seeds, len(test)), dtype=np.int8)
    all_thresholds = np.empty((len(shortlist), seeds), dtype=np.int8)
    runs: dict[str, list[dict]] = {label: [] for label, _ in shortlist}
    model_dir.mkdir(parents=True, exist_ok=True)

    for config_index, (label, cfg) in enumerate(shortlist):
        features = store.get(cfg)
        store.materialize(cfg)
        for seed in range(seeds):
            checkpoint_path = model_dir / f"{cfg.key}_seed{seed}.pt"
            if checkpoint_path.is_file():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                run = checkpoint["run"]
                state = checkpoint["state_dict"]
                print(f"reuse {label} seed={seed}", flush=True)
            else:
                print(f"train {label} seed={seed}", flush=True)
                run, state = extended.train_one(
                    features,
                    labels,
                    split,
                    cfg,
                    seed,
                    epochs,
                    samples_per_epoch,
                    batch_size,
                    True,
                )
                torch.save({"run": run, "state_dict": state}, checkpoint_path)

            network = primary.SpikeMLP(cfg.inputs)
            network.load_state_dict(state)
            validation_margin = primary.margins(network, np.asarray(features[validation]), batch_size)
            threshold = primary.choose_threshold(labels[validation], validation_margin)
            test_margin = primary.margins(network, np.asarray(features[test]), batch_size)
            metrics = primary.metrics(labels[test], test_margin >= threshold)
            all_margins[config_index, seed] = np.rint(test_margin).astype(np.int8)
            all_thresholds[config_index, seed] = threshold
            runs[label].append(
                {
                    "seed": seed,
                    "threshold": int(threshold),
                    "test": metrics,
                    "training_seconds": float(run["training_seconds"]),
                    "checkpoint": str(checkpoint_path.resolve()),
                }
            )
            write_json(progress_path, {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": runs})

    np.savez_compressed(
        predictions_path,
        config_labels=np.array([label for label, _ in shortlist]),
        config_keys=np.array([cfg.key for _, cfg in shortlist]),
        seeds=np.arange(seeds),
        test_paths=np.array([path.relative_to(dataset).as_posix() for path in np.array(paths)[test]]),
        truth=labels[test].astype(np.uint8),
        margins=all_margins,
        thresholds=all_thresholds,
    )

    summaries = {}
    for config_index, (label, cfg) in enumerate(shortlist):
        summaries[label] = {
            "config": asdict(cfg),
            "key": cfg.key,
            "logical_input_bits": cfg.inputs * cfg.bits,
            "byte_aligned_input_bytes": cfg.inputs,
            "model_parameter_bytes": extended.memory_model(cfg)["learned_parameter_bytes_int8"],
            "accuracy": mean_ci(np.array([run["test"]["accuracy"] for run in runs[label]])),
            "f1": mean_ci(np.array([run["test"]["f1"] for run in runs[label]])),
            "thresholds": all_thresholds[config_index].astype(int).tolist(),
            "runs": runs[label],
        }

    baseline_label = shortlist[0][0]
    baseline_correct_by_seed = (
        all_margins[0] >= all_thresholds[0, :, None]
    ) == labels[test][None, :]
    comparisons = {}
    majority_p_values = []
    for config_index, (label, _) in enumerate(shortlist[1:], 1):
        candidate_correct_by_seed = (
            all_margins[config_index] >= all_thresholds[config_index, :, None]
        ) == labels[test][None, :]
        per_seed = [
            exact_mcnemar(baseline_correct_by_seed[seed], candidate_correct_by_seed[seed])
            for seed in range(seeds)
        ]
        baseline_vote = np.sum(
            all_margins[0] >= all_thresholds[0, :, None], axis=0
        ) > seeds / 2
        candidate_vote = np.sum(
            all_margins[config_index] >= all_thresholds[config_index, :, None], axis=0
        ) > seeds / 2
        majority = exact_mcnemar(
            baseline_vote == labels[test], candidate_vote == labels[test]
        )
        majority_p_values.append((label, majority["exact_two_sided_p"]))
        f1_difference = np.array([run["test"]["f1"] for run in runs[label]]) - np.array(
            [run["test"]["f1"] for run in runs[baseline_label]]
        )
        comparisons[label] = {
            "paired_f1_difference": mean_ci(f1_difference),
            "mcnemar_per_seed": per_seed,
            "mcnemar_majority_vote": majority,
        }

    ordered = sorted(majority_p_values, key=lambda item: item[1])
    running = 0.0
    adjusted = {}
    for rank, (label, probability) in enumerate(ordered):
        value = min(1.0, probability * (len(ordered) - rank))
        running = max(running, value)
        adjusted[label] = running
    for label in comparisons:
        comparisons[label]["mcnemar_majority_holm_p"] = float(adjusted[label])

    return {
        "seeds": seeds,
        "epochs": epochs,
        "samples_per_epoch": samples_per_epoch,
        "batch_size": batch_size,
        "predictions_file": str(predictions_path.resolve()),
        "summaries": summaries,
        "comparisons_to_baseline": comparisons,
    }


def occlusion_analysis(
    checkpoint_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    batch_size: int,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    run = checkpoint["run"]
    cfg = extended.EncodingConfig(**run["config"])
    network = primary.SpikeMLP(cfg.inputs)
    network.load_state_dict(checkpoint["state_dict"])
    test = np.flatnonzero(split == "test")
    x_test = np.asarray(features[test])
    truth = labels[test]
    baseline_margin = primary.margins(network, x_test, batch_size)
    threshold = int(run["threshold"])
    baseline_metrics = primary.metrics(truth, baseline_margin >= threshold)
    cells = []
    for cell in range(cfg.inputs):
        occluded = np.array(x_test, copy=True)
        occluded[:, cell] = 0.0
        margin = primary.margins(network, occluded, batch_size)
        measured = primary.metrics(truth, margin >= threshold)
        band = cell // cfg.time_bins
        time_bin = cell % cfg.time_bins
        cells.append(
            {
                "cell": cell,
                "band": band,
                "time_bin": time_bin,
                "f1_drop": float(baseline_metrics["f1"] - measured["f1"]),
                "accuracy_drop": float(baseline_metrics["accuracy"] - measured["accuracy"]),
                "positive_margin_drop": float(
                    baseline_margin[truth == 1].mean() - margin[truth == 1].mean()
                ),
                "negative_margin_drop": float(
                    baseline_margin[truth == 0].mean() - margin[truth == 0].mean()
                ),
            }
        )
        if (cell + 1) % 16 == 0:
            print(f"occlusion {cell + 1}/{cfg.inputs}", flush=True)

    frequency, time_axis = feature_axes(cfg)
    for cell in cells:
        cell["frequency_hz"] = float(frequency[cell["band"]])
        cell["time_ms"] = float(time_axis[cell["time_bin"]])
    importance = np.array([cell["positive_margin_drop"] for cell in cells]).reshape(
        cfg.board_rows, cfg.time_bins
    )
    top = sorted(cells, key=lambda item: item["positive_margin_drop"], reverse=True)[:12]
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "config": asdict(cfg),
        "threshold": threshold,
        "baseline_test": baseline_metrics,
        "band_positive_margin_importance": importance.sum(axis=1).tolist(),
        "time_positive_margin_importance": importance.sum(axis=0).tolist(),
        "top_cells": top,
        "cells": cells,
    }


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--primary-cache", type=Path, required=True)
    parser.add_argument("--extended-cache", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=project / "data" / "reproducibility_cache")
    parser.add_argument("--deployed-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "encoding_reproducibility.json")
    parser.add_argument("--predictions", type=Path, default=Path(__file__).parent / "results" / "encoding_reproducibility_predictions.npz")
    parser.add_argument("--models", type=Path, default=Path(__file__).parent / "results" / "encoding_reproducibility_models")
    parser.add_argument("--sample-clips", type=int, default=256)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch", type=int, default=64)
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-occlusion", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="preserve completed sections already present in --output",
    )
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    paths, labels, speakers = primary.discover_dataset(args.dataset, "sheila")
    split = primary.dataset_split(args.dataset, paths, speakers)
    configs, groups = configuration_catalog()
    by_key = {cfg.key: cfg for cfg in configs}
    store = FeatureStore(
        paths,
        args.primary_cache,
        args.extended_cache,
        args.cache_dir,
        args.feature_batch,
    )
    if args.resume and args.output.is_file():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        result["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    else:
        result = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "dataset": str(args.dataset.resolve()),
            "dataset_digest": primary.source_digest(paths),
            "split": primary.split_summary(labels, split, speakers),
            "configuration_count": len(configs),
            "sweep_groups": groups,
        }

    if not args.skip_direct:
        validation = np.flatnonzero(split == "val")
        positives = validation[labels[validation] == 1]
        negatives = validation[labels[validation] == 0]
        half = args.sample_clips // 2
        sample = np.r_[positives[:half], negatives[: args.sample_clips - min(half, len(positives))]]
        sample_audio = np.stack([extended.read_pcm16(paths[index]) for index in sample])
        reference_flat = extended.feature_batch(sample_audio, REFERENCE)
        reference = reference_flat.reshape(len(sample), REFERENCE.bands, REFERENCE.time_bins)
        reference_frequency, reference_time = feature_axes(REFERENCE)
        direct = {}
        for number, cfg in enumerate(configs, 1):
            print(f"direct {number}/{len(configs)} {cfg.key}", flush=True)
            encoded = np.asarray(store.get(cfg)[sample])
            unquantized_cfg = replace(cfg, bits=16, quantizer="linear")
            unquantized = extended.feature_batch(sample_audio, unquantized_cfg)
            reconstructed = reconstruct(
                encoded, cfg, reference_frequency, reference_time
            )
            positive_reconstructed = reconstructed[labels[sample] == 1]
            positive_reference = reference[labels[sample] == 1]
            direct[cfg.key] = {
                "config": asdict(cfg),
                "fidelity": fidelity_metrics(
                    encoded, unquantized, reconstructed, reference
                ),
                "landmarks": landmark_metrics(
                    positive_reconstructed,
                    positive_reference,
                    reference_frequency,
                    reference_time,
                ),
                "separability": separability_metrics(
                    store.get(cfg),
                    labels,
                    split,
                    speakers,
                    paths,
                    args.bootstrap_repeats,
                ),
            }
            result["direct_metrics"] = direct
            write_json(args.output, result)
        result["direct_metric_sample"] = {
            "clips": len(sample),
            "positive": int(labels[sample].sum()),
            "negative": int(len(sample) - labels[sample].sum()),
            "split": "validation",
            "reference": asdict(REFERENCE),
            "landmarks_are_fixed_proxies_without_phone_alignment": True,
        }

    baseline = extended.BASELINE
    shortlist = [
        ("baseline_8band_300_3400", baseline),
        ("quality_20band_300_3400", replace(baseline, bands=20)),
        ("resolution_control_16band_300_6500", replace(baseline, bands=16, high_hz=6500)),
        ("wide_8band_40_5120", replace(baseline, low_hz=40, high_hz=5120)),
        ("packed_candidate_4bit_log", replace(baseline, bits=4, quantizer="log")),
    ]
    result["shortlist"] = {label: asdict(cfg) for label, cfg in shortlist}

    if not args.skip_occlusion:
        result["occlusion"] = occlusion_analysis(
            args.deployed_checkpoint,
            store.get(baseline),
            labels,
            split,
            args.batch_size,
        )
        write_json(args.output, result)

    if not args.skip_training:
        result["paired_ten_seed"] = ten_seed_shortlist(
            shortlist,
            store,
            labels,
            split,
            paths,
            args.dataset,
            args.seeds,
            args.epochs,
            args.samples_per_epoch,
            args.batch_size,
            args.models,
            args.predictions,
            args.output.with_name(args.output.stem + "_training_progress.json"),
        )

    result["limitations"] = [
        "The landmark scores are fixed time-frequency proxies, not forced phoneme alignments.",
        "Speech Commands v2 contains zero but not the suggested near-miss words she, sheep, or shield.",
        "False accepts per hour require a streaming evaluation corpus and are not estimated from isolated clips.",
        "Fidelity is measured against one 64-by-64, 40--7600 Hz, 1600 ms reference and can penalize useful noise removal.",
    ]
    write_json(args.output, result)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
