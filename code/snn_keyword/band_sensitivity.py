"""Which frequency bands does a detector depend on? Validation data only.

Controlled version of mic_diagnose.py: the same live validation clips (clean
placement, robust_eval.place_clips) pass through ONE filter at a time, applied
to the whole buffer (speech + noise) as a microphone would:
  low-pass   4th-order Butterworth at 3.5 ... 7.5 kHz
  high-pass  2nd-order Butterworth at 100 ... 500 Hz
  peak       +-9 dB peaking EQ (Q 1.4) at 250 Hz ... 6 kHz
  gain       -20 ... +10 dB (level only; log-mel has an absolute floor)
Recall of the keyword and other-word accepts at the model's own threshold, so every
row differs from the clean row only by that filter.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

import channels
import robust_eval as R

ROOT = Path(__file__).resolve().parent
SR = 16000


def conditions():
    out = [('clean', None)]
    for f in (3500, 4000, 4500, 5000, 5500, 6000, 7000, 7500):
        out.append((f'lowpass {f} Hz', butter(4, f, 'lowpass', fs=SR, output='sos')))
    for f in (100, 200, 300, 400, 500):
        out.append((f'highpass {f} Hz', butter(2, f, 'highpass', fs=SR, output='sos')))
    for f in (250, 500, 1000, 2000, 3000, 4000, 5000, 6000):
        for g in (-9, 9):
            out.append((f'peak {f} Hz {g:+d} dB', channels.peaking(f, g, 1.4)))
    for g in (-20, -10, 10):
        out.append((f'gain {g:+d} dB', 10 ** (g / 20)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('models', nargs='+', type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--out', type=Path, default=ROOT / 'results/band_sensitivity.json')
    a = p.parse_args()
    cfg = R.SPLITS['validation']
    clips, y, rng = R.sc_live_clips(a.data, cfg['sc'], seed=cfg['seed'])
    base = R.place_clips(clips, R.bg_noise(a.data), rng)
    dets = {str(m): R.load_detector(m) for m in a.models}
    report = {}
    for name, filt in conditions():
        if filt is None:
            audio = base
        elif isinstance(filt, float):
            audio = [np.clip(b * filt, -1, 1).astype(np.float32) for b in base]
        else:
            audio = [sosfilt(filt, b).astype(np.float32) for b in base]
        row = {}
        for m, det in dets.items():
            conf = np.array([c.max() for _, c, _ in det.traces(audio)])
            row[m] = {'recall': round(float((conf[y == 1] >= det.threshold).mean()), 4),
                      'other_accepted': round(float((conf[y == 0] >= det.threshold).mean()), 4)}
        report[name] = row
        print(f'{name:22s} ' + '  '.join(f"{Path(m).parent.name}/{Path(m).stem}: {r['recall'] * 100:5.1f}% "
                                         f"({r['other_accepted'] * 100:.2f}%)" for m, r in row.items()), flush=True)
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
