"""Cascade: the streaming SNN proposes, the release window model confirms (validation only).

A detection at SNN frame i needs  snn[i] >= t_snn  and a window-model margin
(single 1 s window, features.py) >= t_win for some window ending within
[t_i - before, t_i + after]. On the board the window model only runs when the
SNN fires, so its cost (2.7 ms per window with kdot) is paid rarely.

For every t_win on a grid, t_snn is chosen by stream_select.py's rule (live
other-word accepts <= 0.2%, stream false accepts <= 2/h); the pair with the
highest live recall wins. Reported next to each detector alone.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import robust_eval as R
from diagnose_stream import operating_point

ROOT = Path(__file__).resolve().parent
NEG = np.iinfo(np.int64).min // 4


def window_max(t_snn, t_win, margin, before, after):
    """Per SNN frame: max window margin over windows ending in [t - before, t + after]."""
    lo = np.searchsorted(t_win, t_snn - before, 'left')
    hi = np.searchsorted(t_win, t_snn + after, 'right')
    # Sparse-table-free: windows are 250 ms apart, so each range holds few of them.
    out = np.full(len(t_snn), NEG, np.int64)
    width = int((hi - lo).max()) if len(t_snn) else 0
    for k in range(width):
        idx = lo + k
        ok = idx < hi
        out[ok] = np.maximum(out[ok], margin[np.minimum(idx, len(margin) - 1)][ok])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('snn', type=Path)
    p.add_argument('--window', type=Path, default=ROOT / 'deploy/model.npz')
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--before', type=float, default=.25)
    p.add_argument('--after', type=float, default=.5)
    p.add_argument('--out', type=Path, default=ROOT / 'results/cascade_validation.json')
    a = p.parse_args()
    snn, win = R.load_detector(a.snn), R.load_detector(a.window)
    sets, _ = R.build_sets(a.data, 'validation', None, R.STREAM_SECONDS, {'device', 'tts'})
    y = sets['live_y']
    audio, marks, info = sets['stream']
    hours = info['seconds'] / 3600
    mswc_audio, words, _ = sets['mswc']
    groups = {'live': sets['live']['clean'], 'stream': [audio], 'mswc': mswc_audio}
    tr = {g: (snn.traces(v), win.traces(v)) for g, v in groups.items()}
    wmax = {g: [window_max(ts, tw, m, a.before, a.after) for (ts, _, _), (tw, _, m) in zip(*tr[g])] for g in groups}

    def gated(g, t_w):
        return [np.where(w >= t_w, s, NEG) for (_, s, _), w in zip(tr[g][0], wmax[g])]

    def evaluate(t_w):
        live = np.array([s.max() for s in gated('live', t_w)])
        (ts, _, _), = tr['stream'][0]
        op = operating_point(live, y, (ts, gated('stream', t_w)[0]), marks, hours)
        if op is None:
            return None
        m = np.array([s.max() for s in gated('mswc', t_w)])
        kw = R.K.KEYWORD
        op['mswc_recall'] = round(float((m[words == kw] >= op['threshold']).mean()), 4)
        op['mswc_other_fa'] = round(float((m[(words != kw) & ~np.vectorize(R.yes_prefixed)(words)] >= op['threshold']).mean()), 4)
        op['t_win'] = None if t_w == NEG else int(t_w)
        return op

    report = {'snn': a.snn.as_posix(), 'window': a.window.as_posix(), 'before_s': a.before, 'after_s': a.after}
    report['snn_alone'] = evaluate(NEG)
    print('snn alone', json.dumps(report['snn_alone']), flush=True)
    # Window model alone, its own rule ("2 of 3" confirmed score as in the release).
    live_w = np.array([c.max() for _, c, _ in tr['live'][1]])
    (tw, cw, _), = tr['stream'][1]
    report['window_alone'] = operating_point(live_w, y, (tw, cw), marks, hours)
    print('window alone', json.dumps(report['window_alone']), flush=True)
    margins = np.concatenate([m for _, _, m in tr['live'][1]])
    grid = np.unique(np.quantile(margins, np.linspace(.5, .995, 30)).astype(np.int64))
    rows = []
    for t_w in grid:
        op = evaluate(int(t_w))
        if op:
            rows.append(op)
            print(json.dumps(op), flush=True)
    report['grid'] = rows
    report['best'] = max(rows, key=lambda r: (r['live_recall'], r['stream_recall']))
    print('best', json.dumps(report['best']))
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
