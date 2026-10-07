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


# ----------------------------------------------------------------------------
# Frame-level front end for streaming models (IMPLEMENTATION_PLAN.md, Phase 1).
# One 24-band uint8 vector per 10 ms frame, same FFT and mel bank as above;
# frame k covers samples [160 k, 160 k + 400) of the stream.

PCEN = dict(s=0.04, alpha=0.96, delta=2.0, r=0.5, scale=2.0 ** 31, top=6.0)


def frame_power(audio):
    """Mel power per frame, float32 (n_frames, 24). Streams shorter than one frame give no frames."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(a) < 400:
        return np.zeros((0, N_MELS), np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(a, 400)[::160]
    window = np.hanning(401)[:-1].astype(np.float32)
    out = np.empty((len(frames), N_MELS), np.float32)
    # Blocks of frames bound the memory of hour-long streams (each frame is independent).
    for i in range(0, len(frames), 20000):
        power = np.abs(np.fft.rfft(frames[i:i + 20000] * window, n=512, axis=1) / window.sum()) ** 2
        out[i:i + 20000] = power @ mel_bank().T
    return out


# Log-mel ranges (dB re full scale) mapped onto uint8. 'logmel' is the window
# front end's; speech in the live sets never gets above about -40 dB, and 35% of
# its values sit on the -80 dB floor (46% of the /s/ bands at -20 dB input), so
# 'logmel_w' moves the window down and widens it (JOURNAL entry 20).
LOGMEL_RANGE = {'logmel': (-80., 0.), 'logmel_w': (-120., -20.)}
FRONTENDS = ('logmel', 'logmel_w', 'logmel_agc', 'pcen', 'logmel_lp<Hz>')
# logmel_lp<Hz> (e.g. logmel_lp4500): the logmel front end after an 8th-order
# Butterworth low-pass on the PC (JOURNAL entry 24: "sheila" needs no high band).
# logmel_agc: log-mel relative to a causal peak-level tracker (JOURNAL entry 22).
# The level of a frame is its loudest band (dB); the tracker follows it up at
# once and falls by `release` dB per frame, never below `floor`. Features are
# band dB minus the tracker over `range` dB, so loudness drops out while the
# spectral shape stays. It runs on the PC; the state carries across chunks.
AGC = dict(release=.05, floor=-95., range=80., min_db=-140.)


def agc_track(level, state=None, p=AGC):
    """Peak tracker over per-frame levels (dB); returns (tracker per frame, last value)."""
    y = np.empty(len(level), np.float64)
    prev = -np.inf if state is None else state
    for t, v in enumerate(level):
        prev = max(v, prev - p['release'], p['floor'])
        y[t] = prev
    return y, (prev if len(level) else state)


def logmel_agc_frames(power, state=None, p=AGC):
    """Level-normalised log-mel to uint8; returns (frames, tracker state) for streaming."""
    db = 10 * np.log10(np.maximum(power, 10 ** (p['min_db'] / 10)))
    y, state = agc_track(db.max(1), state, p)
    rel = db - y[:, None]
    return np.rint(np.clip((rel + p['range']) / p['range'], 0, 1) * 255).astype(np.uint8), state


def logmel_frames(power, frontend='logmel'):
    """Absolute log power over LOGMEL_RANGE[frontend] dB to uint8."""
    lo, hi = LOGMEL_RANGE[frontend]
    db = 10 * np.log10(np.maximum(power, 10 ** (lo / 10)))
    return np.rint(np.clip((db - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def pcen_frames(power, state=None, p=PCEN):
    """Per-channel energy normalization [20] to uint8; returns (frames, smoother state).

    M[t] = (1 - s) M[t-1] + s E[t];  y = (E / (eps + M)^alpha + delta)^r - delta^r.
    The smoother starts at the first frame unless a state is passed (streaming).
    The gain normalization removes static level and slow spectral tilt, i.e.
    most of a microphone's and room's colouring.
    """
    from scipy.signal import lfilter
    e = power.astype(np.float64) * p['scale']
    if len(e) == 0:
        return np.zeros((0, N_MELS), np.uint8), state
    prev = e[0] if state is None else state
    m, _ = lfilter([p['s']], [1, p['s'] - 1], e, axis=0, zi=((1 - p['s']) * prev)[None])
    y = (e / (1e-6 + m) ** p['alpha'] + p['delta']) ** p['r'] - p['delta'] ** p['r']
    return np.rint(np.clip(y / p['top'], 0, 1) * 255).astype(np.uint8), m[-1]


def frame_features(audio, frontend='logmel'):
    if frontend.startswith('logmel_lp'):
        from scipy.signal import butter, sosfilt
        audio = sosfilt(butter(8, float(frontend[9:]), 'lowpass', fs=SAMPLE_RATE, output='sos'), audio).astype(np.float32)
        frontend = 'logmel'
    power = frame_power(audio)
    if frontend == 'pcen':
        return pcen_frames(power)[0]
    if frontend == 'logmel_agc':
        return logmel_agc_frames(power)[0]
    return logmel_frames(power, frontend)
