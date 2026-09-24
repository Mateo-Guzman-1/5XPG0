"""Held-out evaluation of confusable words and live-stream behaviour.

For each model (integer inference, identical to the board):
  clips      official test clips at the decision threshold (precision/recall/F1)
  live       test clips placed in 2.25 s of background noise, consecutive 250 ms-hop
             windows as the demo sends them: "yes" detected / other word accepted,
             for a single window and for "2 of 3" confirmation at stream_threshold
  augmented  false accepts on augment.py test-split variants (shift = recall)
  tts        per word, share of synthesized utterances (make_tts_probe.ps1, never
             trained on) detected over 250 ms-hop windows, averaged over 4 hop phases
Writes one JSON file; run with KWS_TIME_BINS=64 --data data/b64 for 64-bin models.
"""
import argparse
import collections
import glob
import json
import os
from pathlib import Path
import numpy as np
from features import N_TIME, SAMPLE_RATE, features, read_wav
from model import integer_forward, metrics
from tune_stream import HOP, live_windows

ROOT = Path(__file__).resolve().parent
TTS_GROUPS = {'yes': ['yes'], 'yeets/yets/yetz': ['yeets', 'yets', 'yetz'], 'pizza(s)': ['pizza', 'pizzas'],
              'eats/its': ['eats', 'its'], 'other -ts': ['jets', 'gets', 'bets', 'lets', 'sets'],
              'ch words': ['yech', 'yetch', 'each', 'peach'], 'yeah': ['yeah'], 'yeet': ['yeet'],
              'guess/less': ['guess', 'less'], 'cheese/yesterday': ['cheese', 'yesterday']}


def tts_sequences(data, probe):
    files = sorted(glob.glob(str(probe / '*.wav')))
    if not files:
        return None
    raw = data / 'speech_commands_v0.02' / '_background_noise_'
    noise = np.concatenate([read_wav(p) for p in sorted(glob.glob(str(raw / '*.wav')))[:3]])
    rng = np.random.default_rng(0)
    words, x, bounds = [], [], [0]
    for p in files:
        a = read_wav(p)[:int(1.4 * SAMPLE_RATE)]
        buf = np.zeros(int(2.5 * SAMPLE_RATE), np.float32)
        off = rng.integers(0, len(buf) - len(a)); buf[off:off + len(a)] += a
        n0 = rng.integers(0, len(noise) - len(buf)); buf += noise[n0:n0 + len(buf)] * .05
        for phase in range(0, HOP, HOP // 4):
            windows = [features(buf[s:s + SAMPLE_RATE]) for s in range(phase, len(buf) - SAMPLE_RATE + 1, HOP)]
            words.append(os.path.basename(p).split('_')[0]); x.extend(windows); bounds.append(len(x))
    return words, np.stack(x), np.array(bounds)


def decide(margin, bounds, single, stream):
    """Per sequence: any window >= single; any window and one of its two predecessors >= stream."""
    one, two = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        v = margin[a:b]
        one.append(bool((v >= single).any()))
        two.append(any(v[i] >= stream and (v[max(0, i - 2):i] >= stream).any() for i in range(len(v))))
    return np.array(one), np.array(two)


def evaluate(q, clips, live, aug, tts):
    th = int(q['decision_threshold']); sth = int(q.get('stream_threshold', th))
    margin = lambda x: (lambda s: s[:, 1] - s[:, 0])(integer_forward(x, q)[0])
    out = {'decision_threshold': th, 'stream_threshold': sth, 'hidden': int(len(q['w1']))}
    m = metrics(clips[1], margin(clips[0]) >= th)
    out['clips'] = {k: round(float(m[k]), 4) for k in ('precision', 'recall', 'f1')}
    x, clip, y = live
    bounds = np.r_[0, np.cumsum(np.bincount(clip))]
    one, two = decide(margin(x), bounds, th, sth)
    out['live'] = {'single_window': {'yes_detected': round(float(one[y == 1].mean()), 4),
                                     'other_accepted': round(float(one[y == 0].mean()), 4)},
                   'two_of_three': {'yes_detected': round(float(two[y == 1].mean()), 4),
                                    'other_accepted': round(float(two[y == 0].mean()), 4)}}
    ax, ak = aug
    hit = margin(ax) >= th
    out['augmented_test_detected'] = {k: round(float(hit[ak == k].mean()), 4) for k in dict.fromkeys(ak)}
    if tts:
        words, tx, tb = tts
        one, two = decide(margin(tx), tb, th, sth)
        words = np.array(words)
        out['tts'] = {mode: {g: round(float(r[np.isin(words, ws)].mean()), 4) for g, ws in TTS_GROUPS.items()}
                      for mode, r in (('single_window', one), ('two_of_three', two))}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('models', nargs='+', help='name=path/to/model.npz')
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--probe', type=Path, default=ROOT / 'build/tts_probe')
    p.add_argument('--out', type=Path, default=ROOT / 'results/confusables.json')
    a = p.parse_args()
    d = np.load(a.data / 'features.npz')
    clips = (d['x'][d['split'] == 2], d['y'][d['split'] == 2])
    aug = np.load(a.data / 'augment.npz')
    aug = (aug['x'][aug['split'] == 2], aug['kind'][aug['split'] == 2])
    live = live_windows(a.data, 3000, seed=1, split=2)
    tts = tts_sequences(a.data, a.probe)
    report = {'time_bins': N_TIME, 'live_test': {'yes_clips': int((live[2] == 1).sum()), 'other_clips': int((live[2] == 0).sum())},
              'tts_utterances': (len(tts[0]) // 4) if tts else 0, 'models': {}}
    for spec in a.models:
        name, path = spec.split('=', 1)
        report['models'][name] = evaluate(dict(np.load(path)), clips, live, aug, tts)
        report['models'][name]['file'] = Path(path).as_posix()
        print(name, json.dumps(report['models'][name]['live']), flush=True)
    a.out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
