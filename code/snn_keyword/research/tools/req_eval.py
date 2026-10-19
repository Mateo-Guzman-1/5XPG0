"""R1: requirement tests of the synapse-study designs with Mateo's harness (research/PLAN_R1.md).

Audio sets come from the vendored robust_eval.py (tools/mateo_eval/, origin/ALIF-Layer1 a856a03):
  live      212 + 3,000 Speech Commands clips in 2.25 s of noise (test: seed 1; validation: 204 + 3,000, seed 2)
  negatives every non-keyword SC word of the split + LibriSpeech (test: test-clean/-other; validation: dev-clean/-other)
  stream    the 1 h test stream (keyword at known times) for recall and latency
The window features of every set are computed once and cached in build/r1_cache/ (git-ignored).

Per model:
  1. stream threshold on validation with Mateo's rule (stream_select.select): the highest live recall with
     live other words <= 0.2 % and false accepts per hour <= 2 on the validation negatives;
  2. the test, once, at that threshold: live recall (all / complete recordings), other words accepted,
     false accepts per hour (about 19 h), the 1 h stream (recall, latency), clip F1 on the full test split
     and on Damien's 3,212-clip subset;
  3. the same test at the S-study stream threshold (<= 0.3 % on validation windows) for comparison,
     and the test operating curves (threshold sweep) for the figures.
Gates: the fast exact forward equals model.integer_forward; at the stored S-study threshold the live test
equals the stored quality JSON (Mateo's live set = tune_stream.live_windows).

Usage: python research/tools/req_eval.py [--configs h48_b8 ...] [--out research/results/R1]
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

KWS = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(KWS), str(Path(__file__).resolve().parent / 'mateo_eval')]
import robust_eval as R  # noqa: E402
from features import SAMPLE_RATE, features  # noqa: E402
from model import Q, integer_forward, metrics, trunc_div  # noqa: E402

RES = KWS / 'research/results'
DATA = KWS / 'data'
CACHE = KWS / 'build/r1_cache'
HOP = SAMPLE_RATE // 4
MAX_LIVE_FA, MAX_FA_HOUR = .002, 2.
# config -> experiment folder holding <config>_seed<s>.npz and quality/<config>.json
CONFIGS = {'h16': 'E2', 'h32': 'E2', 'h48': 'E2', 'h95': 'E2', 'h48_b8': 'S2', 'h48_b4': 'S2',
           'h48_k64': 'S3', 'h48_k128': 'S3', 'h48_k256': 'S3', 'h48_k512': 'S3',
           'h48_b8_k64': 'S6', 'h48_b8_k128': 'S6', 'h48_b8_k256': 'S6'}
SEEDS = (0, 1, 2)


# ----------------------------------------------------------------------------- features

def windows(audio):
    """1 s windows every 250 ms from the start, as robust_eval.WindowDetector.window_features."""
    starts = range(0, len(audio) - SAMPLE_RATE + 1, HOP)
    return np.stack([features(audio[s:s + SAMPLE_RATE]) for s in starts]) if len(starts) else np.zeros((0, 768), np.uint8)


def windows_parallel(audio, pool, piece=400):
    starts = list(range(0, len(audio) - SAMPLE_RATE + 1, HOP))
    parts = [starts[i:i + piece] for i in range(0, len(starts), piece)]
    out = pool.map(lambda ss: np.stack([features(audio[s:s + SAMPLE_RATE]) for s in ss]), parts)
    return np.concatenate(list(out))


def packed(feats):
    """List of (n_i, 768) arrays -> one array and the start offsets."""
    lens = np.array([len(f) for f in feats])
    return np.concatenate(feats), np.r_[0, np.cumsum(lens)]


def live_set(split):
    path = CACHE / f'live_{split}.npz'
    if path.exists():
        return dict(np.load(path))
    cfg = R.SPLITS[split]
    clips, y, rng = R.sc_live_clips(DATA, cfg['sc'], seed=cfg['seed'])
    audios = R.place_clips(clips, R.bg_noise(DATA), rng, None)
    with ThreadPoolExecutor(12) as pool:
        feats = list(pool.map(windows, audios))
    x, off = packed(feats)
    out = dict(x=x, off=off, y=y, clipped=np.array([R.edge_clipped(c) for c in clips]))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def negative_set(split):
    path = CACHE / f'negatives_{split}.npz'
    if path.exists():
        d = dict(np.load(path))
        d['segments'] = json.loads((CACHE / f'negatives_{split}_segments.json').read_text())
        return d
    feats, segs, seconds = [], [], 0.
    with ThreadPoolExecutor(12) as pool:
        for audio, seg in R.negative_stream(DATA, split):
            seconds += len(audio) / SAMPLE_RATE
            feats.append(windows_parallel(audio, pool))
            segs.append([(s, e, k, Path(n).name) for s, e, k, n in seg])
            print(f'negatives {split}: {seconds / 3600:.2f} h', flush=True)
    x, off = packed(feats)
    np.savez(path, x=x, off=off, hours=np.array(seconds / 3600))
    (CACHE / f'negatives_{split}_segments.json').write_text(json.dumps(segs))
    return negative_set(split)


def stream_set():
    path = CACHE / 'stream_test.npz'
    if path.exists():
        d = dict(np.load(path))
        d['info'] = json.loads((CACHE / 'stream_test_info.json').read_text())
        return d
    audio, marks, info = R.build_stream(DATA, 'test')
    with ThreadPoolExecutor(12) as pool:
        x = windows_parallel(audio, pool)
    np.savez(path, x=x, marks=marks)
    info['segments'] = [(s, e, k, Path(n).name) for s, e, k, n in info['segments']]
    (CACHE / 'stream_test_info.json').write_text(json.dumps(info))
    return stream_set()


def damien_subset():
    """Damien's prepare_speech_commands_v2.py selection (origin/damien-dicking-around): every keyword clip,
    negatives balanced over the 34 other words by sha256('group2-sheila-v2:' + path), per official split."""
    from collections import defaultdict
    raw = DATA / 'speech_commands_v0.02'
    val = set((raw / 'validation_list.txt').read_text().split())
    test = set((raw / 'testing_list.txt').read_text().split())
    d = np.load(DATA / 'features.npz')
    names = [str(n) for n in d['names']]
    split_of = lambda p: 'validation' if p in val else 'test' if p in test else 'train'
    limits = {'train': 8000, 'validation': 2500, 'test': 3000}
    key = lambda p: hashlib.sha256(('group2-sheila-v2:' + p).encode()).hexdigest()
    chosen = {p for p in names if p.split('/')[0] == 'sheila'}
    for split, count in limits.items():
        grouped = defaultdict(list)
        for p in names:
            if p.split('/')[0] != 'sheila' and split_of(p) == split:
                grouped[p.split('/')[0]].append(p)
        for v in grouped.values():
            v.sort(key=key)
        labels, sel, cursor = sorted(grouped), [], 0
        while len(sel) < count:
            progress = False
            for lab in labels:
                if cursor < len(grouped[lab]) and len(sel) < count:
                    sel.append(grouped[lab][cursor]); progress = True
            if not progress:
                break
            cursor += 1
        chosen.update(sel)
    mask = np.array([p in chosen for p in names])
    counts = {s: [int((mask & (d['split'] == i)).sum()), int((mask & (d['split'] == i) & (d['y'] == 1)).sum())]
              for i, s in enumerate(('train', 'validation', 'test'))}
    return mask, counts


# ----------------------------------------------------------------------------- scoring

def fast_margin(x, q, batch=32768):
    """integer_forward's margin with an exact float64 matrix product (|sum| < 2^53), current encoding."""
    assert not int(q['encoding']), 'current encoding only'
    w1 = q['w1'].astype(np.float64)
    w2 = q['w2'].astype(np.int64)
    out = np.empty(len(x), np.int64)
    for s in range(0, len(x), batch):
        drive = trunc_div(np.rint(x[s:s + batch].astype(np.float64) @ w1.T).astype(np.int64), 255) + q['b1']
        mem = np.zeros_like(drive)
        count = np.zeros_like(drive)
        for _ in range(int(q['steps'])):
            mem = trunc_div(mem * 7, 8) + drive
            spk = mem >= Q
            mem -= spk * Q
            count += spk
        sc = count @ w2.T + q['b2'] * int(q['steps'])
        out[s:s + batch] = sc[:, 1] - sc[:, 0]
    return out


def confirmed(m):
    """robust_eval.WindowDetector: c[k] = min(m[k], max(m[k-2:k])); c[0] never detects."""
    c = np.full(len(m), np.iinfo(np.int64).min // 2)
    if len(m) > 1:
        prev = m[:-1].copy()
        prev[1:] = np.maximum(m[1:-1], m[:-2])
        c[1:] = np.minimum(m[1:], prev)
    return c


def traces(margin, off):
    """Per audio: (window end times in s, confirmed score)."""
    out = []
    for a, b in zip(off[:-1], off[1:]):
        m = margin[a:b]
        out.append(((np.arange(len(m)) * HOP + SAMPLE_RATE) / SAMPLE_RATE, confirmed(m)))
    return out


class Negatives:
    """Detections per threshold on the long negative stream, with robust_eval's 1 s hold-off."""

    def __init__(self, margin, off, hours, segments=None):
        self.tr = traces(margin, off)
        self.hours = float(hours)
        self.segments = segments

    def detections(self, t):
        return [R.detections(times, c, t) for times, c in self.tr]

    def fa_per_hour(self, t):
        return sum(len(d) for d in self.detections(t)) / self.hours

    def by_source(self, t):
        import collections
        cnt = collections.Counter()
        for det, seg in zip(self.detections(t), self.segments):
            cnt.update(R.source_of(x, seg) for x in det)
        return dict(cnt)


def live_max(margin, off):
    return np.array([c.max() for _, c in traces(margin, off)])


def select(live, y, neg):
    """stream_select.select for window models (w = 1), the validation threshold."""
    best = None
    for t in np.unique(live[y == 1])[::-1]:
        fa = (live[y == 0] >= t).mean()
        if fa > MAX_LIVE_FA:
            break
        fph = neg.fa_per_hour(t)
        if fph > MAX_FA_HOUR:
            if fph > 5 * MAX_FA_HOUR:
                break
            continue
        rec = (live[y == 1] >= t).mean()
        if best is None or rec > best['live_recall']:
            best = dict(threshold=int(t), live_recall=float(rec), live_other=float(fa), fa_per_hour=round(fph, 3))
    return best


def test_at(t, live, y, clipped, neg, stream, stream_marks, stream_hours, segments):
    complete = (y == 1) & ~clipped
    st = R.score_stream(*stream, t, stream_marks, stream_hours, segments)
    return dict(threshold=int(t), live_recall=round(float((live[y == 1] >= t).mean()), 4),
                live_recall_complete=round(float((live[complete] >= t).mean()), 4),
                live_other=round(float((live[y == 0] >= t).mean()), 5),
                fa_per_hour=round(neg.fa_per_hour(t), 3), fa_by_source=neg.by_source(t),
                stream=st)


def curve(live, y, neg, n=60):
    """Test operating curve: thresholds from the keyword clips' peaks, up to 2 % other words accepted."""
    cand = np.unique(live[y == 1])
    cand = cand[[(live[y == 0] >= t).mean() <= .02 for t in cand]]
    if len(cand) > n:
        cand = np.unique(cand[np.linspace(0, len(cand) - 1, n).round().astype(int)])
    return [dict(threshold=int(t), live_recall=round(float((live[y == 1] >= t).mean()), 4),
                 live_other=round(float((live[y == 0] >= t).mean()), 5), fa_per_hour=round(neg.fa_per_hour(t), 3))
            for t in cand]


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--configs', nargs='+', default=list(CONFIGS))
    p.add_argument('--out', type=Path, default=RES / 'R1')
    p.add_argument('--sets-only', action='store_true', help='build the feature caches and stop')
    a = p.parse_args()
    t0 = time.time()
    lv, lt = live_set('validation'), live_set('test')
    nv, nt, st = negative_set('validation'), negative_set('test'), stream_set()
    sub_mask, sub_counts = damien_subset()
    d = np.load(DATA / 'features.npz')
    xt, yt = d['x'][d['split'] == 2], d['y'][d['split'] == 2]
    sub_t = sub_mask[d['split'] == 2]
    info = dict(live_validation=dict(keyword=int(lv['y'].sum()), other=int((1 - lv['y']).sum())),
                live_test=dict(keyword=int(lt['y'].sum()), other=int((1 - lt['y']).sum()),
                               keyword_complete=int(((lt['y'] == 1) & ~lt['clipped']).sum())),
                negatives_hours=dict(validation=round(float(nv['hours']), 3), test=round(float(nt['hours']), 3)),
                negatives_windows=dict(validation=len(nv['x']), test=len(nt['x'])),
                stream_test={k: v for k, v in st['info'].items() if k != 'segments'} | {'keyword_inserted': len(st['marks'])},
                damien_subset_clips=sub_counts, full_test_clips=[len(yt), int(yt.sum())],
                rule=dict(max_live_other=MAX_LIVE_FA, max_fa_per_hour=MAX_FA_HOUR, source='mateo_eval/stream_select.py'),
                harness='origin/ALIF-Layer1 a856a03 (research/tools/mateo_eval)')
    print(json.dumps(info), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'sets.json').write_text(json.dumps(info, indent=2))
    if a.sets_only:
        return
    (a.out / 'requirements').mkdir(exist_ok=True)
    stream_hours = st['info']['seconds'] / 3600
    for config in a.configs:
        exp = CONFIGS[config]
        quality = json.loads((RES / exp / 'quality' / f'{config}.json').read_text())['models']
        rows = {}
        for s in SEEDS:
            name = f'{config}_seed{s}'
            path = RES / exp / 'models' / f'{name}.npz'
            q = dict(np.load(path))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            old = quality[name]
            assert old['sha256'] == sha, f'{name}: model differs from its quality JSON'
            # Gate 1: the fast forward equals the integer oracle.
            probe = lt['x'][::7][:3000]
            ref, _ = integer_forward(probe, q)
            assert np.array_equal(fast_margin(probe, q), ref[:, 1] - ref[:, 0]), f'{name}: fast forward differs'
            mlv, mlt = fast_margin(lv['x'], q), fast_margin(lt['x'], q)
            live_v, live_t = live_max(mlv, lv['off']), live_max(mlt, lt['off'])
            negv = Negatives(fast_margin(nv['x'], q), nv['off'], nv['hours'])
            negt = Negatives(fast_margin(nt['x'], q), nt['off'], nt['hours'], nt['segments'])
            stream = traces(fast_margin(st['x'], q), np.array([0, len(st['x'])]))[0]
            # Gate 2: Mateo's live set at the S-study threshold reproduces the stored live result.
            old_t = int(q['stream_threshold'])
            gate = dict(sheila_detected=round(float((live_t[lt['y'] == 1] >= old_t).mean()), 4),
                        other_accepted=round(float((live_t[lt['y'] == 0] >= old_t).mean()), 4))
            assert gate == old['test']['live']['two_of_three'], f'{name}: live gate {gate} != {old["test"]["live"]["two_of_three"]}'
            sel = select(live_v, lv['y'], negv)
            test = test_at(sel['threshold'], live_t, lt['y'], lt['clipped'], negt, stream, st['marks'], stream_hours,
                           st['info']['segments']) if sel else None
            old_rule = test_at(old_t, live_t, lt['y'], lt['clipped'], negt, stream, st['marks'], stream_hours,
                               st['info']['segments'])
            clip = integer_forward(xt, q)[0]
            pred = (clip[:, 1] - clip[:, 0]) >= int(q['decision_threshold'])
            rows[name] = dict(
                file=path.relative_to(KWS.parents[1]).as_posix(), sha256=sha,
                decision_threshold=int(q['decision_threshold']),
                validation=dict(selection=sel, s_rule_threshold=old_t,
                                at_s_rule=dict(live_recall=round(float((live_v[lv['y'] == 1] >= old_t).mean()), 4),
                                               live_other=round(float((live_v[lv['y'] == 0] >= old_t).mean()), 5),
                                               fa_per_hour=round(negv.fa_per_hour(old_t), 3))),
                test=test, test_at_s_rule=old_rule, test_curve=curve(live_t, lt['y'], negt),
                clips_full=metrics(yt, pred), clips_damien_subset=metrics(yt[sub_t], pred[sub_t]),
                gates=dict(fast_forward='equal', live_at_s_rule=gate))
            msg = {k: v for k, v in (test or {}).items() if k in ('threshold', 'live_recall', 'live_other', 'fa_per_hour')}
            print(name, 'val', sel, 'test', msg, 'latency', (test or {}).get('stream', {}).get('latency_median_s'),
                  f'{time.time() - t0:.0f} s', flush=True)
        (a.out / 'requirements' / f'{config}.json').write_text(json.dumps(
            dict(config=config, experiment=exp, sets='sets.json', models=rows), indent=2))


if __name__ == '__main__':
    main()
