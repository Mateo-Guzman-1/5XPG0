"""Where does a streaming model fail? Validation data only (robust_eval.py sets).

1. Operating point: stream_select.py's rule (live other-word accepts <= 0.2%,
   stream false accepts <= 2/h), and which constraint binds.
2. False accepts on the validation stream at that threshold and at looser
   ones: source kind and word (Speech Commands label or LibriSpeech).
3. Missed "yes" clips in the live set against signal-to-noise ratio, speech
   level, and whether the /s/ is cut off (robust_eval.edge_clipped).
4. Decision rules on the same per-frame scores (no retraining): the raw score,
   a moving average over w frames, and a sustained score (minimum over the
   last k frames). Each gets its own threshold under the same rule.
"""
import argparse
import collections
import copy
import json
from pathlib import Path

import numpy as np

import robust_eval as R

ROOT = Path(__file__).resolve().parent


def smooth(score, rule):
    kind, w = rule
    if kind == 'raw' or w <= 1:
        return score
    s = score.astype(np.float64)
    if kind == 'mean':
        c = np.cumsum(np.r_[0., s])
        out = np.empty_like(s)
        out[w - 1:] = (c[w:] - c[:-w]) / w
        out[:w - 1] = c[1:w] / np.arange(1, w)
        return out
    # 'min': the score was at least this for the last w frames
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.r_[np.full(w - 1, s.min()), s]
    return sliding_window_view(pad, w).min(1)


def operating_point(live, y, stream, marks, hours, max_live_fa=.002, max_fa_hour=2.):
    times, score = stream
    cands = np.unique(np.quantile(np.r_[live, score[::10]], np.linspace(.5, 1, 400)))
    best = None
    for t in cands:
        fa = (live[y == 0] >= t).mean()
        if fa > max_live_fa:
            continue
        st = R.score_stream(times, score, t, marks, hours)
        if st['fa_per_hour'] > max_fa_hour:
            continue
        rec = (live[y == 1] >= t).mean()
        if best is None or rec > best['live_recall']:
            best = dict(threshold=float(t), live_recall=round(float(rec), 4), live_fa=round(float(fa), 4),
                        stream_recall=st['recall'], fa_per_hour=st['fa_per_hour'])
    # Which constraint binds: the recall each one alone would allow.
    only_live = max(((live[y == 1] >= t).mean() for t in cands if (live[y == 0] >= t).mean() <= max_live_fa), default=0)
    only_stream = max(((live[y == 1] >= t).mean() for t in cands
                       if R.score_stream(times, score, t, marks, hours)['fa_per_hour'] <= max_fa_hour), default=0)
    best['recall_if_only_live_fa'] = round(float(only_live), 4)
    best['recall_if_only_stream_fa'] = round(float(only_stream), 4)
    return best


def false_accepts(times, score, th, marks, segments):
    ev = R.detections(times, score, th)
    out = []
    for t in ev:
        if any(s <= t <= e + R.TOLERANCE for s, e in marks):
            continue
        best, what = 0., ('noise only', '')
        for s, e, k, *name in segments:
            o = min(e, t) - max(s, t - 1.0)
            if o > best:
                best, what = o, (k, name[0] if name else '')
        kind, name = what
        word = name.split('/')[0] if kind in ('word', 'yes') else ('libri' if kind == 'libri' else kind)
        out.append((round(float(t), 2), kind, word, Path(name).name))
    return out


def live_conditions(clips, rng, noise_len):
    """Replays place_clips' draws: per clip the noise gain and the clip's active speech level (dBFS)."""
    rng = copy.deepcopy(rng)
    n = int(R.LIVE_SECONDS * R.SAMPLE_RATE)
    snr, level = [], []
    for a in clips:
        a = a[:R.SAMPLE_RATE]
        rng.integers(0, n - len(a)); rng.integers(0, noise_len - n)  # offset, noise start
        gain = rng.uniform(.02, .1); rng.integers(0, R.HOP)
        s, e = R.word_span(a)
        act = np.sqrt(np.mean(a[s:e] ** 2)) + 1e-9
        level.append(20 * np.log10(act)); snr.append(20 * np.log10(act / gain))
    return np.array(snr), np.array(level)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    det = R.load_detector(a.model)
    sets, info = R.build_sets(a.data, 'validation', None, R.STREAM_SECONDS, {'device', 'tts'})
    y = sets['live_y']
    live_tr = det.traces(sets['live']['clean'])
    audio, marks, sinfo = sets['stream']
    (times, score, _), = det.traces([audio])
    hours = sinfo['seconds'] / 3600
    mswc_audio, words, clipped = sets['mswc']
    mswc_tr = det.traces(mswc_audio)
    report = {'model': a.model.as_posix(), 'rules': {}}
    rules = [('raw', 1), ('mean', 3), ('mean', 5), ('mean', 10), ('min', 2), ('min', 3), ('min', 5)]
    for rule in rules:
        live = np.array([smooth(c, rule).max() for _, c, _ in live_tr])
        st = (times, smooth(score, rule))
        op = operating_point(live, y, st, marks, hours)
        m = np.array([smooth(c, rule).max() for _, c, _ in mswc_tr])
        op['mswc_recall'] = round(float((m[words == 'yes'] >= op['threshold']).mean()), 4)
        report['rules'][f'{rule[0]}{rule[1]}'] = op
        print(rule, json.dumps(op), flush=True)
    # Failure analysis on the raw score.
    op = report['rules']['raw1']
    th = op['threshold']
    fas = false_accepts(times, score, th, marks, sinfo['segments'])
    report['false_accepts_at_threshold'] = fas
    # Looser thresholds: what enters first as the threshold drops.
    live = np.array([c.max() for _, c, _ in live_tr])
    looser = {}
    for target in (.65, .70, .75):
        t = np.quantile(live[y == 1], 1 - target)
        f = false_accepts(times, score, t, marks, sinfo['segments'])
        looser[f'live_recall_{target}'] = {'threshold': float(t), 'fa_per_hour': round(len(f) / hours, 1),
                                           'live_fa': round(float((live[y == 0] >= t).mean()), 4),
                                           'by_kind': dict(collections.Counter(k for _, k, _, _ in f)),
                                           'top_words': collections.Counter(w for _, _, w, _ in f).most_common(12)}
    report['looser'] = looser
    # Misses.
    clips, _, rng = R.sc_live_clips(a.data, R.SPLITS['validation']['sc'], seed=R.SPLITS['validation']['seed'])
    snr, level = live_conditions(clips, rng, len(R.bg_noise(a.data)))
    edge = np.array([R.edge_clipped(c) for c in clips])
    pos = y == 1
    hit = live >= th
    miss = {}
    for name, v in (('snr_db', snr), ('level_dbfs', level)):
        q = np.quantile(v[pos], [0, 1 / 3, 2 / 3, 1])
        miss[name] = [{'range': [round(float(q[i]), 1), round(float(q[i + 1]), 1)],
                       'recall': round(float(hit[pos & (v >= q[i]) & (v <= q[i + 1])].mean()), 3)} for i in range(3)]
    miss['edge_clipped'] = {'n': int((pos & edge).sum()), 'recall': round(float(hit[pos & edge].mean()), 3) if (pos & edge).any() else None,
                            'recall_not_clipped': round(float(hit[pos & ~edge].mean()), 3)}
    gap = (live[pos] - th) / max(1., np.std(live[pos]))
    miss['missed_margin_sd'] = np.round(np.quantile(gap[gap < 0], [.1, .5, .9]), 2).tolist() if (gap < 0).any() else None
    report['misses'] = miss
    print(json.dumps({k: v for k, v in report.items() if k not in ('rules', 'false_accepts_at_threshold')}, indent=1))
    print('false accepts at threshold:', collections.Counter(w for _, _, w, _ in fas).most_common(20))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1, default=str))


if __name__ == '__main__':
    main()
