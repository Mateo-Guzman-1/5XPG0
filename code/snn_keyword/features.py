"""Single frontend used by dataset preparation, WAV replay, and microphone demo.

16 kHz mono, 1 s, 400-sample periodic Hann / 160 hop / 512 FFT,
24 triangular HTK mel bands (80--7600 Hz), 32 pooled time bins.
Absolute log-power mapping [-80, 0] dB to uint8; no per-clip normalization.
"""
from functools import lru_cache
import os
import wave
import numpy as np

SAMPLE_RATE = 16000
N_MELS = 24
# KWS_TIME_BINS exists only for the time-resolution experiment (see JOURNAL.md).
# Firmware, host, and model must all use the same value; the release uses 32.
N_TIME = int(os.environ.get('KWS_TIME_BINS', 32))
N_INPUT = N_MELS * N_TIME


def read_wav(path):
    with wave.open(str(path), 'rb') as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
            raise ValueError('Expected mono 16 kHz PCM16 WAV')
        return np.frombuffer(f.readframes(f.getnframes()), '<i2').astype(np.float32) / 32768


@lru_cache(maxsize=1)
def mel_bank():
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    points = 700 * (10 ** (np.linspace(mel(80), mel(7600), N_MELS + 2) / 2595) - 1)
    freqs = np.fft.rfftfreq(512, 1 / SAMPLE_RATE)
    bank = np.maximum(0, np.minimum((freqs[None] - points[:-2, None]) / (points[1:-1, None] - points[:-2, None]),
                                    (points[2:, None] - freqs[None]) / (points[2:, None] - points[1:-1, None])))
    return (bank / bank.sum(axis=1, keepdims=True)).astype(np.float32)


def features(audio):
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError('Audio contains non-finite values')
    a = np.pad(a[:SAMPLE_RATE], (0, max(0, SAMPLE_RATE - len(a))))
    frames = np.lib.stride_tricks.sliding_window_view(a, 400)[::160]
    window = np.hanning(401)[:-1].astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, n=512, axis=1) / window.sum()) ** 2
    db = 10 * np.log10(np.maximum(power @ mel_bank().T, 1e-8))
    # Pool in time only; mel-major flattening is the firmware ABI.
    pooled = np.stack([part.mean(axis=0) for part in np.array_split(db, N_TIME)], axis=1)
    return np.rint(np.clip((pooled + 80) / 80, 0, 1) * 255).astype(np.uint8).reshape(-1)
