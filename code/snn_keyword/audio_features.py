#!/usr/bin/env python3
"""Shared audio preprocessing and wire format for the keyword demo.

The PC and the trainer both use this module.  Keeping one implementation is
important: a model trained with different FFT bins or time pooling than the
live demo is not a useful model, even if both sides call their result a
"spectrogram".
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = SAMPLE_RATE
FFT_WINDOW = 400          # 25 ms
FFT_HOP = 160             # 10 ms
N_MELS = 16               # compact, mel-ish frequency bands
N_TIME = 16               # fixed board input length
N_INPUTS = N_MELS * N_TIME

PACKET_MAGIC = b"KWS1"
PACKET_HEADER = struct.Struct("<4sIHH")


def _fit_one_second(wav: np.ndarray) -> np.ndarray:
    """Return exactly one second of mono float32 audio."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size >= WINDOW_SAMPLES:
        return wav[-WINDOW_SAMPLES:]
    return np.pad(wav, (WINDOW_SAMPLES - wav.size, 0))


def resample_linear(wav: np.ndarray, source_rate: int) -> np.ndarray:
    """Small dependency-free resampler, sufficient for the demo input."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if source_rate == SAMPLE_RATE or wav.size == 0:
        return wav
    n_out = max(1, round(wav.size * SAMPLE_RATE / source_rate))
    old_x = np.linspace(0.0, 1.0, wav.size, endpoint=False)
    new_x = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(new_x, old_x, wav).astype(np.float32)


def read_wav(path: str | Path) -> np.ndarray:
    """Read PCM WAV, mix channels to mono, resample, and return float32."""
    with wave.open(str(path), "rb") as f:
        channels = f.getnchannels()
        width = f.getsampwidth()
        rate = f.getframerate()
        raw = f.readframes(f.getnframes())

    if width == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        pcm = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")

    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return resample_linear(pcm, rate)


def extract_features(wav: np.ndarray) -> np.ndarray:
    """Create a fixed 16x16 log-frequency/time map in the range [0, 1].

    This deliberately uses grouped FFT bins rather than a large mel-filterbank
    dependency.  Sixteen groups of eight bins cover roughly 40 Hz to 5.1 kHz,
    and 98 short-time frames are mean-pooled into sixteen time bins.
    """
    wav = _fit_one_second(wav)
    starts = range(0, WINDOW_SAMPLES - FFT_WINDOW + 1, FFT_HOP)
    frames = np.stack([wav[i:i + FFT_WINDOW] for i in starts])
    spectrum = np.abs(
        np.fft.rfft(frames * np.hanning(FFT_WINDOW), axis=1)
    )

    # Drop DC, retain 128 frequency bins, and group them 8-at-a-time.
    bands = spectrum[:, 1:1 + N_MELS * 8]
    bands = bands.reshape(bands.shape[0], N_MELS, 8).mean(axis=2)
    bands = np.log1p(bands).T  # [frequency, short-time frame]

    edges = np.linspace(0, bands.shape[1], N_TIME + 1, dtype=np.int32)
    pooled = np.stack(
        [bands[:, edges[i]:edges[i + 1]].mean(axis=1) for i in range(N_TIME)],
        axis=1,
    )
    peak = float(pooled.max())
    if peak > 0.0:
        pooled /= peak
    return pooled.astype(np.float32)


def quantize_features(features: np.ndarray) -> np.ndarray:
    """Quantize a feature map to the board's uint8 input representation."""
    a = np.asarray(features, dtype=np.float32)
    if a.shape != (N_MELS, N_TIME):
        raise ValueError(f"expected {(N_MELS, N_TIME)}, got {a.shape}")
    return np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


def pack_frame(features: np.ndarray, sequence: int) -> bytes:
    quantized = quantize_features(features)
    return (
        PACKET_HEADER.pack(PACKET_MAGIC, sequence & 0xFFFFFFFF, N_MELS, N_TIME)
        + quantized.tobytes(order="C")
    )


def unpack_frame(packet: bytes) -> tuple[int, np.ndarray]:
    if len(packet) < PACKET_HEADER.size:
        raise ValueError("packet is shorter than the keyword header")
    magic, sequence, rows, cols = PACKET_HEADER.unpack_from(packet)
    if magic != PACKET_MAGIC:
        raise ValueError(f"bad packet magic {magic!r}")
    if (rows, cols) != (N_MELS, N_TIME):
        raise ValueError(f"unsupported feature shape {(rows, cols)}")
    payload = packet[PACKET_HEADER.size:]
    if len(payload) != N_INPUTS:
        raise ValueError(f"expected {N_INPUTS} payload bytes, got {len(payload)}")
    return sequence, np.frombuffer(payload, dtype=np.uint8).reshape(rows, cols)
