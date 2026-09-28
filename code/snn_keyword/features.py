#!/usr/bin/env python3
"""features.py — the spectrogram front-end, shared by training and the PC demo.

Training (train_keyword_snn.py) and the live demo (pc_keyword_demo.py) MUST
compute exactly the same features, otherwise the network sees different
inputs at run time than it was trained on. Keep this the single source.

1 s of 16 kHz audio  ->  [N_MELS, N_TIME] float32 in [0, 1]
"""

import numpy as np

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE      # 1 s window (Speech Commands clips are 1 s)
N_MELS = 16                     # frequency bands
N_TIME = 16                     # time bins after pooling
# The spectrogram is scaled by its own peak, but never by less than this:
# otherwise silence / faint room noise is blown up to full scale and looks
# like speech. ~1st percentile of the Speech Commands clip peaks.
MIN_PEAK = 1.0


def mel_spectrogram(wav):
    """Very small hand-rolled mel-ish spectrogram -> [N_MELS, n_frames], [0,1]."""
    win = int(SAMPLE_RATE * 0.025)
    hop = int(SAMPLE_RATE * 0.010)
    frames = [wav[i:i + win] for i in range(0, len(wav) - win, hop)]
    if not frames:
        return np.zeros((N_MELS, 1), dtype=np.float32)
    spec = np.abs(np.fft.rfft(np.stack(frames) * np.hanning(win), axis=1))
    # crude band grouping instead of a real mel filterbank (a design choice!)
    spec = spec[:, :N_MELS * 8].reshape(len(frames), N_MELS, 8).mean(axis=2)
    spec = np.log1p(spec).T                     # [N_MELS, n_frames]
    return (spec / max(spec.max(), MIN_PEAK)).astype(np.float32)


def clip_features(wav):
    """1 s clip -> fixed-size [N_MELS, N_TIME] input for the SNN.

    Pads/crops to CLIP_SAMPLES, then average-pools the ~98 spectrogram frames
    down to N_TIME bins so the network input stays small for the RISC-V core.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < CLIP_SAMPLES:
        wav = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
    wav = wav[:CLIP_SAMPLES]
    spec = mel_spectrogram(wav)                            # [N_MELS, ~98]
    bins = np.array_split(np.arange(spec.shape[1]), N_TIME)
    return np.stack([spec[:, b].mean(axis=1) for b in bins], axis=1)
