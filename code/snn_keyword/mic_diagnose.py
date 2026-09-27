"""Which part of the held-out microphone chain costs recall? Validation data only.

Live validation clips (robust_eval.place_clips, 'mic' condition) with the
held-out profiles applied in full, as EQ only (filters), as gain + tanh
clipping only, and as gain only; then recall per profile under the full chain
with the profile's corners. The model's own threshold is used.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

import channels
import robust_eval as R

ROOT = Path(__file__).resolve().parent


def eq_only(a, m):
    sos = np.concatenate([butter(2, m['highpass'], 'highpass', fs=16000, output='sos'),
                          butter(4, m['lowpass'], 'lowpass', fs=16000, output='sos')]
                         + [channels.peaking(*p) for p in m['peaks']])
    return sosfilt(sos, a).astype(np.float32)


def gain_clip(a, m):
    k = m['drive']
    return (np.tanh(k * a * 10 ** (m['gain_db'] / 20)) / k).astype(np.float32)


def gain_only(a, m):
    return (a * 10 ** (m['gain_db'] / 20)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    a = p.parse_args()
    det = R.load_detector(a.model)
    cfg = R.SPLITS['validation']
    clips, y, rng = R.sc_live_clips(a.data, cfg['sc'], seed=cfg['seed'])
    noise = R.bg_noise(a.data)
    full = channels.apply_mic
    for name, fn in [('clean', None), ('full', full), ('eq_only', eq_only), ('gain_clip', gain_clip), ('gain_only', gain_only)]:
        channels.apply_mic = fn or full
        audio = R.place_clips(clips, noise, rng, None if fn is None else 'mic')
        conf = np.array([c.max() for _, c, _ in det.traces(audio)])
        print(f'{name:10s} recall {100 * (conf[y == 1] >= det.threshold).mean():5.1f}%  '
              f'other accepted {100 * (conf[y == 0] >= det.threshold).mean():.2f}%', flush=True)
    channels.apply_mic = full
    conf = np.array([c.max() for _, c, _ in det.traces(R.place_clips(clips, noise, rng, 'mic'))])
    k = np.arange(len(y)) % channels.N_PROFILES
    for i, m in enumerate(channels.heldout_mics()):
        s = (k == i) & (y == 1)
        print(f'profile {i:2d} recall {100 * (conf[s] >= det.threshold).mean():5.1f}%  gain {m["gain_db"]:+5.1f} dB '
              f'drive {m["drive"]:.2f}  hp {m["highpass"]:4.0f} Hz  lp {m["lowpass"]:5.0f} Hz  peaks '
              + ', '.join(f'{f:.0f} Hz {g:+.1f} dB' for f, g, _ in m['peaks']))


if __name__ == '__main__':
    main()
