"""Choose the per-window threshold for "2 of 3" stream confirmation.

The live demo sends a 1 s window every 250 ms; firmware command 3 reports a
detection only when a window and one of the two previous windows both reach
the stream threshold. Confirmation removes most single-window false accepts,
so the per-window threshold can sit below the single-clip F1 threshold.

Validation clips only (never test): each clip is placed at a random position
in 2.25 s of background noise and cut into consecutive 250 ms-hop windows,
as the demo would see it. The chosen threshold is the lowest one whose
confirmed false-accept rate on validation negatives stays within --max-fa.
It is stored in the model file as `stream_threshold` for export_model.py.
"""
import argparse
import glob
import json
from pathlib import Path
import numpy as np
from features import SAMPLE_RATE, features, read_wav
from model import integer_forward

ROOT = Path(__file__).resolve().parent
HOP = SAMPLE_RATE // 4


def live_windows(data, negatives, seed=2):
    cache = ROOT / 'build' / f'live_validation_{negatives}_{seed}.npz'
    if cache.exists():
        z = np.load(cache)
        return z['x'], z['clip'], z['y']
    d = np.load(data / 'features.npz')
    raw = data / 'speech_commands_v0.02'
    names, y = d['names'][d['split'] == 1], d['y'][d['split'] == 1]
    noise = np.concatenate([read_wav(p) for p in sorted(glob.glob(str(raw / '_background_noise_' / '*.wav')))])
    rng = np.random.default_rng(seed)
    ids = np.r_[np.flatnonzero(y == 1), rng.choice(np.flatnonzero(y == 0), negatives, replace=False)]
    x, clip = [], []
    for k, i in enumerate(ids):
        a = read_wav(raw / names[i])[:SAMPLE_RATE]
        buf = np.zeros(int(2.25 * SAMPLE_RATE), np.float32)
        off = rng.integers(0, len(buf) - len(a)); buf[off:off + len(a)] = a
        n0 = rng.integers(0, len(noise) - len(buf)); buf += noise[n0:n0 + len(buf)] * rng.uniform(.02, .1)
        for start in range(int(rng.integers(0, HOP)), len(buf) - SAMPLE_RATE + 1, HOP):
            x.append(features(buf[start:start + SAMPLE_RATE])); clip.append(k)
    x, clip, labels = np.stack(x), np.array(clip), y[ids]
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, x=x, clip=clip, y=labels)
    return x, clip, labels


def confirmed_margin(margin, clip, n_clips):
    """Per clip, the highest threshold at which some window and one of its two predecessors both pass."""
    best = np.full(n_clips, np.iinfo(np.int64).min)
    for c in range(n_clips):
        v = margin[clip == c]
        for i in range(1, len(v)):
            best[c] = max(best[c], min(v[i], v[max(0, i - 2):i].max()))
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', type=Path, help='model .npz; stream_threshold is written into it')
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--max-fa', type=float, default=.003, help='confirmed false-accept budget per negative clip')
    p.add_argument('--negatives', type=int, default=3000)
    a = p.parse_args()
    q = dict(np.load(a.model))
    x, clip, y = live_windows(a.data, a.negatives)
    scores, _ = integer_forward(x, q)
    best = confirmed_margin((scores[:, 1] - scores[:, 0]).astype(np.int64), clip, len(y))
    candidates = np.unique(best)
    allowed = [t for t in candidates if (best[y == 0] >= t).mean() <= a.max_fa]
    threshold = int(min(allowed))
    q['stream_threshold'] = np.array(threshold)
    np.savez_compressed(a.model, **q)
    print(json.dumps({'model': str(a.model), 'decision_threshold': int(q['decision_threshold']),
                      'stream_threshold': threshold, 'max_fa': a.max_fa,
                      'validation_confirmed_recall': float((best[y == 1] >= threshold).mean()),
                      'validation_confirmed_fa': float((best[y == 0] >= threshold).mean())}, indent=2))


if __name__ == '__main__':
    main()
