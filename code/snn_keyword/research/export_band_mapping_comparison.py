#!/usr/bin/env python3
"""Export equal-Hz, logarithmic, and mel encodings of one Sheila clip."""

from __future__ import annotations

import argparse
from pathlib import Path
import wave

import matplotlib.pyplot as plt
import numpy as np

MAPPINGS = (
    ("equal", "Equal-Hz grouped bins"),
    ("log", "Logarithmic band edges"),
    ("mel", "Mel triangular filters"),
)

SAMPLE_RATE = 16_000
FFT_WINDOW = 400
FFT_HOP = 160
N_BANDS = 8
N_TIME = 16
LOW_HZ = 40
HIGH_HZ = 5120
RMS_GATE = 0.003
RMS_FULL_SCALE = 0.040


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
    if source.size >= SAMPLE_RATE:
        start = (source.size - SAMPLE_RATE) // 2
        return source[start : start + SAMPLE_RATE]
    padding = SAMPLE_RATE - source.size
    return np.pad(source, (padding // 2, padding - padding // 2))


def hz_to_mel(value: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def mel_to_hz(value: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def band_weights(frequencies: np.ndarray, mapping: str) -> np.ndarray:
    weights = np.zeros((N_BANDS, frequencies.size), dtype=np.float32)
    selected = np.flatnonzero(
        (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)
    )
    if mapping == "equal":
        for row, indices in enumerate(np.array_split(selected, N_BANDS)):
            weights[row, indices] = 1.0 / len(indices)
        return weights
    if mapping == "log":
        edges = np.geomspace(LOW_HZ, HIGH_HZ, N_BANDS + 1)
        for row in range(N_BANDS):
            upper = frequencies <= edges[row + 1] if row == N_BANDS - 1 else frequencies < edges[row + 1]
            indices = np.flatnonzero((frequencies >= edges[row]) & upper)
            if indices.size == 0:
                centre = np.sqrt(edges[row] * edges[row + 1])
                indices = np.array([np.abs(frequencies - centre).argmin()])
            weights[row, indices] = 1.0 / len(indices)
        return weights
    if mapping != "mel":
        raise ValueError(f"unknown mapping: {mapping}")
    edges = mel_to_hz(np.linspace(hz_to_mel(LOW_HZ), hz_to_mel(HIGH_HZ), N_BANDS + 2))
    for row in range(N_BANDS):
        left, centre, right = edges[row : row + 3]
        rising = (frequencies - left) / (centre - left)
        falling = (right - frequencies) / (right - centre)
        triangular = np.maximum(0.0, np.minimum(rising, falling))
        weights[row] = triangular / triangular.sum()
    return weights


def band_centres_hz(mapping: str) -> np.ndarray:
    frequencies = np.fft.rfftfreq(FFT_WINDOW, 1.0 / SAMPLE_RATE)
    weights = band_weights(frequencies, mapping)
    return weights @ frequencies


def extract_features(wav: np.ndarray, mapping: str) -> np.ndarray:
    rms = float(np.sqrt(np.mean(wav * wav)))
    if rms <= RMS_GATE:
        return np.zeros((N_BANDS, N_TIME), dtype=np.float32)
    activity = np.log(rms / RMS_GATE) / np.log(RMS_FULL_SCALE / RMS_GATE)
    activity = float(np.clip(activity, 0.0, 1.0))
    starts = range(0, SAMPLE_RATE - FFT_WINDOW + 1, FFT_HOP)
    frames = np.stack([wav[start : start + FFT_WINDOW] for start in starts])
    spectrum = np.abs(
        np.fft.rfft(frames * np.hanning(FFT_WINDOW), axis=1)
    ).astype(np.float32)
    frequencies = np.fft.rfftfreq(FFT_WINDOW, 1.0 / SAMPLE_RATE)
    bands = band_weights(frequencies, mapping) @ spectrum.T
    bands = np.log1p(bands)
    groups = np.array_split(np.arange(bands.shape[1]), N_TIME)
    pooled = np.stack([bands[:, group].mean(axis=1) for group in groups], axis=1)
    peak = float(pooled.max())
    if peak > 0.0:
        pooled /= peak
    pooled *= activity
    return np.rint(np.clip(pooled, 0.0, 1.0) * 255.0) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    wav = read_pcm16(args.input)
    features = [extract_features(wav, mapping) for mapping, _ in MAPPINGS]

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 4.8),
        sharex=True,
        constrained_layout=True,
    )
    images = []
    for axis, values, (mapping, title) in zip(
        axes, features, MAPPINGS, strict=True
    ):
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=(0.0, 1.0, -0.5, N_BANDS - 0.5),
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        images.append(image)
        centres = band_centres_hz(mapping) / 1000.0
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Band centre (kHz)")
        axis.set_yticks(np.arange(N_BANDS))
        axis.set_yticklabels([f"{value:.2f}" for value in centres])
        axis.set_xticks(np.linspace(0.0, 1.0, 6))

    colorbar = figure.colorbar(
        images[0],
        ax=axes,
        location="right",
        shrink=0.92,
        pad=0.02,
    )
    colorbar.set_label("Normalized 8-bit feature value")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
