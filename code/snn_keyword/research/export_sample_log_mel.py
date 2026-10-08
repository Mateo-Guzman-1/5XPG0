#!/usr/bin/env python3
"""Export a Sheila log-mel example beside the deployed feature encoding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from audio_features import (  # noqa: E402
    FFT_HOP,
    FFT_WINDOW,
    HIGH_HZ,
    LOW_HZ,
    SAMPLE_RATE,
    extract_features,
    read_wav,
)


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def mel_to_hz(frequency_mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(frequency_mel) / 2595.0) - 1.0)


def log_mel_spectrogram(wav: np.ndarray, n_mels: int = 64) -> np.ndarray:
    """Return a normalized log-power mel spectrogram for visualization."""
    if wav.size < SAMPLE_RATE:
        missing = SAMPLE_RATE - wav.size
        wav = np.pad(wav, (missing // 2, missing - missing // 2))
    elif wav.size > SAMPLE_RATE:
        start = (wav.size - SAMPLE_RATE) // 2
        wav = wav[start : start + SAMPLE_RATE]

    starts = np.arange(0, SAMPLE_RATE - FFT_WINDOW + 1, FFT_HOP)
    frames = np.stack([wav[start : start + FFT_WINDOW] for start in starts])
    spectrum = np.fft.rfft(frames * np.hanning(FFT_WINDOW), axis=1)
    power = np.abs(spectrum) ** 2
    fft_frequencies = np.fft.rfftfreq(FFT_WINDOW, 1.0 / SAMPLE_RATE)

    mel_edges = np.linspace(hz_to_mel(LOW_HZ), hz_to_mel(HIGH_HZ), n_mels + 2)
    hz_edges = mel_to_hz(mel_edges)
    filters = np.zeros((n_mels, fft_frequencies.size), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = hz_edges[index : index + 3]
        rising = (fft_frequencies - left) / max(center - left, 1e-12)
        falling = (right - fft_frequencies) / max(right - center, 1e-12)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))

    mel_power = filters @ power.T
    log_mel = 10.0 * np.log10(np.maximum(mel_power, 1e-12))
    log_mel -= log_mel.max()
    return np.maximum(log_mel, -80.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    wav = read_wav(args.input)
    detailed = log_mel_spectrogram(wav)
    deployed = extract_features(wav)

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)

    left = axes[0].imshow(
        detailed,
        origin="lower",
        aspect="auto",
        extent=(0.0, 1.0, LOW_HZ / 1000.0, HIGH_HZ / 1000.0),
        cmap="magma",
        vmin=-80.0,
        vmax=0.0,
    )
    axes[0].set_title("Conventional log-mel spectrogram (64 bands)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (kHz)")
    figure.colorbar(left, ax=axes[0], label="Relative power (dB)")

    right = axes[1].imshow(
        deployed,
        origin="lower",
        aspect="auto",
        extent=(0.0, 1.0, LOW_HZ / 1000.0, HIGH_HZ / 1000.0),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[1].set_title("Deployed detector input (8 bands × 16 bins)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency range (kHz)")
    figure.colorbar(right, ax=axes[1], label="Normalized feature value")

    figure.suptitle(f'Sheila sample: {args.input.name}', fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
