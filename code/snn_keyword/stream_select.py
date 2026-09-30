"""Decision threshold for streaming models, on validation data only (Phase 3, curriculum step 3).

Builds the robust_eval.py sets from validation data (Speech Commands
validation clips in noise, MSWC dev, a one-hour stream of Speech Commands
validation words and LibriSpeech dev-clean) and picks the threshold with the
highest live recall such that, on validation,
  live other words accepted <= --max-live-fa   and   false accepts/hour <= --max-fa-hour.
False accepts per hour come from robust_eval.negative_stream (--fa-source
negatives, default: about 15 h of LibriSpeech dev-clean + dev-other and every
non-keyword validation word). The original 1 h stream (--fa-source stream) holds
only about 2 false accepts at the operating point: a Poisson 95% interval of
0.2-7 per hour, too wide to set a threshold or to rank models (JOURNAL entry 19).
Candidate thresholds are the exact recall steps (each positive clip's peak).

--window w scores the integer sum of the last w frame scores (w = 1: the raw
score). The threshold is then in sum units; the firmware compares the same sum.
The held-out microphones and rooms are not used here: they stay a test.
The threshold (and window) is written into each checkpoint; a JSON summary ranks them.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import keyword_config as K
from model import decision_scores
from robust_eval import build_sets, detections, load_detector, negative_traces, score_stream, yes_prefixed

ROOT = Path(__file__).resolve().parent


def moving_sum(score, w):
    """model.decision_scores on a raw trace; float scores (.pt models) stay float."""
    score = np.asarray(score)
    if np.issubdtype(score.dtype, np.integer):
        return decision_scores(score, w)
    c = np.cumsum(np.r_[0., score.astype(np.float64)])
    idx = np.arange(len(score))
    return c[idx + 1] - c[np.maximum(idx + 1 - w, 0)]


def fa_per_hour(traces, hours, t):
    return sum(len(detections(times, s, t)) for times, s in traces) / hours


def select(det, sets, neg, max_live_fa, max_fa_hour, w=1):
    y = sets['live_y']
    live = np.array([moving_sum(r, w).max() for _, _, r in det.traces(sets['live']['clean'])])
    audio, marks, info = sets['stream']
    (times, _, score), = det.traces([audio])
    score = moving_sum(score, w)
    hours = info['seconds'] / 3600
    if neg is None:   # the 1 h stream as the false-accept source
        neg_tr, neg_hours = [(times, score)], hours
    else:
        neg_tr, neg_hours = [(t, moving_sum(s, w)) for t, s, _ in neg[0]], neg[1]
    mswc_audio, words, _ = sets['mswc']
    mswc = np.array([moving_sum(r, w).max() for _, _, r in det.traces(mswc_audio)])
    other = (words != K.KEYWORD) & ~np.vectorize(yes_prefixed)(words)   # prefixed words are don't care
    best = None
    for t in np.unique(live[y == 1])[::-1]:            # highest threshold first: recall rises
        fa = (live[y == 0] >= t).mean()
        if fa > max_live_fa:
            break
        fph = fa_per_hour(neg_tr, neg_hours, t)
        if fph > max_fa_hour:
            if fph > 5 * max_fa_hour:
                break
            continue
        rec = (live[y == 1] >= t).mean()
        if best is None or rec > best['live_recall']:
            complete = (y == 1) & ~sets['live_clipped']
            best = dict(threshold=float(t), window=w, live_recall=float(rec), live_fa=float(fa),
                        live_recall_complete=float((live[complete] >= t).mean()),
                        fa_per_hour=round(fph, 3), fa_hours=round(neg_hours, 2),
                        stream=score_stream(times, score, t, marks, hours),
                        mswc_recall=float((mswc[words == K.KEYWORD] >= t).mean()),
                        mswc_other_fa=float((mswc[other] >= t).mean()))
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoints', nargs='+', type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--max-live-fa', type=float, default=.002)
    p.add_argument('--max-fa-hour', type=float, default=2.)
    p.add_argument('--fa-source', choices=['negatives', 'stream'], default='negatives')
    p.add_argument('--window', type=int, nargs='+', default=[1], help='moving-sum lengths to try (frames)')
    p.add_argument('--no-write', action='store_true', help='only report; leave the checkpoints unchanged')
    p.add_argument('--out', type=Path, default=ROOT / 'results/stream_selection.json')
    a = p.parse_args()
    sets, info = build_sets(a.data, 'validation', None, 3600, {'device', 'tts'})
    report = {'keyword': K.KEYWORD, 'config': info, 'rule': vars(a) | {'checkpoints': [str(c) for c in a.checkpoints]}, 'models': {}}
    for path in a.checkpoints:
        det = load_detector(path)
        neg = negative_traces(det, a.data, 'validation') if a.fa_source == 'negatives' else None
        rows = [select(det, sets, neg, a.max_live_fa, a.max_fa_hour, w) for w in a.window]
        rows = [r for r in rows if r is not None]
        r = max(rows, key=lambda r: r['live_recall']) if rows else None
        report['models'][str(path)] = {'best': r, 'by_window': rows}
        print(path, json.dumps({'best': r, 'by_window': [{k: v for k, v in x.items() if k != 'stream'} for x in rows]}), flush=True)
        if r is None or a.no_write:
            continue
        if path.suffix == '.pt':
            ck = torch.load(path, weights_only=False)
            ck['threshold'] = r['threshold']
            ck['window'] = r['window']
            ck['selection'] = r
            torch.save(ck, path)
        else:  # integer stream models get the threshold; window models are only scored
            q = dict(np.load(path))
            if str(q.get('kind', 'window')) == 'stream':
                q['stream_threshold'] = np.array(int(np.ceil(r['threshold'])))
                q['decision_window'] = np.array(int(r['window']))
                np.savez_compressed(path, **q)
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
