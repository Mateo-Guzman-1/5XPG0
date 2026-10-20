"""R2 step A1 (research/PLAN_R2_GRU.md): our window SNN as stage 1 of Mateo's cascade.

Mateo's verifier_cascade.py (vendored in tools/mateo_eval/) evaluates "stage 1 proposes, the GRU verifier
confirms" on cached 10 ms frames and per-frame stage-1 scores. This tool reuses that code unchanged and swaps
only the stage-1 scorer:

  cache   runs verifier_cascade.cache (same audio, same items, same order) with two wrappers:
          frame_features also computes our 1 s windows (every 250 ms) of the same audio and their margins
          (req_eval.fast_margin) for the H48 K64 seeds; stage1_raw returns our per-frame trace instead of
          Mateo's streaming SNN. Trace: NEG = int64.min // 2 everywhere except frame 98 + 25 j (the first
          frame ending after window j, i.e. in request j + 3), which holds the window score:
            m  the single-window margin;
            c  the 2-of-3 confirmed score (req_eval.confirmed, robust_eval.WindowDetector).
          One pass writes data_verifier/cache_<split>_sheila_r2_<seed>_<variant>.npz for every seed x variant.
          Gates: every non-stage-1 array equals Mateo's cache (data_verifier/cache_<split>_sheila.npz), so the
          verifier scores of G-A2 are reused unchanged; fast_margin equals model.integer_forward.
  select  verifier_cascade.select on one of our caches;
  choose  A2: the variant (m or c) with the higher mean validation live recall, thresholds per seed
          (results/R2/A/selection.json);
  final   verifier_cascade.final on one of our caches (the one-time test).

Usage (from code/snn_keyword, KWS_KEYWORD=sheila is set here):
  python research/tools/r2_cascade.py cache --split validation
  python research/tools/r2_cascade.py select --seed 0 --variant m <scores.npz> --verify-ms 107.3 --max-fa-hour 1.6 --out x.json
  python research/tools/r2_cascade.py final --seed 0 --variant m <scores.npz> --t1 .. --t2 .. --stage1-threshold .. --out y.json
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

os.environ['KWS_KEYWORD'] = 'sheila'
TOOLS = Path(__file__).resolve().parent
KWS = TOOLS.parents[1]
# Mateo's folder first: verifier_cascade needs his features.frame_features and model.integer_forward_stream.
# His window features() and integer_forward equal Pedro's bit for bit (gate G-A1, results/R2/gate_A1.json).
sys.path[:0] = [str(TOOLS / 'mateo_eval'), str(TOOLS)]
import verifier_cascade as VC  # noqa: E402
import req_eval as RE  # noqa: E402
from model import integer_forward  # noqa: E402

MODELS = KWS / 'research/results/S3/models'
SEEDS = (0, 1, 2)
VARIANTS = ('m', 'c')
NEG = np.iinfo(np.int64).min // 2
FIRST, HOP_FRAMES = 98, 25      # window j ends at sample 16000 + 4000 j: frame 98 + 25 j is the first ending after it
STAGE1_KEYS = ('stage1', 'decision_window')


def r2_file(split, seed, variant):
    return VC.DV / f'cache_{split}_sheila_r2_{seed}_{variant}.npz'


def model_path(seed):
    return MODELS / f'h48_k64_seed{seed}.npz'


def place(score, n):
    """Per-frame trace of n frames with window j's score at frame 98 + 25 j (the last frame if the stream ends first)."""
    raw = np.full(n, NEG, np.int64)
    if len(score):
        np.maximum.at(raw, np.minimum(FIRST + HOP_FRAMES * np.arange(len(score)), n - 1), score)
    return raw


def cache(a):
    qs = {s: dict(np.load(model_path(s))) for s in SEEDS}
    pool = ThreadPoolExecutor(12)
    stash, calls, check_x = {}, [], []
    frame_features = VC.frame_features

    def features_and_margins(audio, front='logmel'):
        f = frame_features(audio, front)
        x = RE.windows_parallel(audio, pool) if len(audio) > 60 * 16000 else RE.windows(audio)
        if sum(len(c) for c in check_x) < 20000:
            check_x.append(x)
        stash[id(f)] = (f, {s: RE.fast_margin(x, q) if len(x) else np.zeros(0, np.int64) for s, q in qs.items()})
        return f

    def our_raw(q, feats):
        per = []
        for f in feats:
            _, margins = stash.pop(id(f))
            per.append({(s, v): place(m if v == 'm' else RE.confirmed(m), len(f))
                        for s, m in margins.items() for v in VARIANTS})
        calls.append(per)
        return [p[(SEEDS[0], 'm')] for p in per]

    tmp = VC.DV / f'cache_{a.split}_sheila_r2_tmp.npz'
    VC.frame_features, VC.stage1_raw, VC.cache_file = features_and_margins, our_raw, lambda split: tmp
    t0 = time.perf_counter()
    VC.cache(type('A', (), dict(stage1=model_path(SEEDS[0]), data=KWS / 'data', split=a.split,
                                device=False, add_device=False))())
    pool.shutdown()
    out = dict(np.load(tmp))
    # Which group each stage-1 call was: match the first config's trace with the saved one.
    groups = [g for g in VC.GROUPS if f'{g}_off' in out]
    by_group = {}
    for per in calls:
        if not per:
            continue
        first = np.concatenate([p[(SEEDS[0], 'm')] for p in per])
        match = [g for g in groups if len(out[f'{g}_off']) - 1 == len(per) and np.array_equal(out[f'{g}_raw'], first)]
        assert len(match) == 1, match
        by_group[match[0]] = per
    assert stash == {}, 'frames without a stage-1 call'
    # Gate: everything except the stage-1 scores equals Mateo's cache (same audio, frames and items).
    ref_file = VC.DV / f'cache_{a.split}_sheila.npz'
    while not ref_file.exists():     # Mateo's cache may still be running in parallel
        time.sleep(30)
    time.sleep(5)
    ref = np.load(ref_file)
    keys = sorted(k for k in ref.files if not k.endswith('_raw') and k not in STAGE1_KEYS)
    assert sorted(k for k in out if not k.endswith('_raw') and k not in STAGE1_KEYS) == keys
    same = {k: bool(np.array_equal(out[k], ref[k])) for k in keys}
    # Gate: fast_margin equals the integer oracle on the first ~20k windows (live clips).
    x = np.concatenate(check_x)
    exact = {s: bool(np.array_equal(RE.fast_margin(x, q), np.diff(integer_forward(x, q)[0], axis=1)[:, 0]))
             for s, q in qs.items()}
    check = {'split': a.split, 'items': {g: len(out[f'{g}_off']) - 1 for g in groups},
             'frames_equal_mateo_cache': all(same.values()), 'arrays_compared': same,
             'fast_margin_equals_integer_forward': {'windows': len(x), **{f'seed{s}': v for s, v in exact.items()}},
             'seconds': round(time.perf_counter() - t0)}
    (KWS / 'research/results/R2/A').mkdir(parents=True, exist_ok=True)
    (KWS / f'research/results/R2/A/cache_check_{a.split}.json').write_text(json.dumps(check, indent=1))
    print(json.dumps({k: v for k, v in check.items() if k != 'arrays_compared'}), flush=True)
    assert check['frames_equal_mateo_cache'] and all(exact.values()), 'gate failed: see cache_check json'
    for s in SEEDS:
        for v in VARIANTS:
            o = dict(out)
            for g in groups:
                if g in by_group:
                    o[f'{g}_raw'] = np.concatenate([p[(s, v)] for p in by_group[g]])
                else:
                    assert len(out[f'{g}_raw']) == 0
                    o[f'{g}_raw'] = np.zeros(0, np.int64)
            o.update(stage1=np.array(str(model_path(s))), decision_window=np.array(1),
                     r2_seed=np.array(s), r2_variant=np.array(v))
            np.savez(r2_file(a.split, s, v), **o)
    tmp.unlink()
    print('written', [r2_file(a.split, s, v).name for s in SEEDS for v in VARIANTS], flush=True)


def choose(a):
    """A2: the variant with the higher mean validation live recall of the cascade (policy a, main FA/h budget);
    thresholds per seed for the one-time test. Reads results/R2/A/select_<seed>_<variant>_fa<budget>.json."""
    sel = KWS / 'research/results/R2/A'
    out = {'rule': f'policy a; live other words <= 0.2 %, <= {a.max_fa_hour} FA/h on the validation negatives; '
                   'variant by the mean validation live recall of the cascade over the seeds '
                   '(tie: fewer mean verifier calls per hour)',
           'budgets': {}}
    for budget in [a.max_fa_hour] + a.sensitivity:
        rows = {}
        for v in VARIANTS:
            for s in SEEDS:
                rep = json.loads((sel / f'select_{s}_{v}_fa{budget}.json').read_text())
                run, = [r for r in rep['runs'] if r['policy'] == 'a']
                rows[(v, s)] = {k: run[k] and {x: run[k][x] for x in ('t1', 't2', 'live_recall', 'live_recall_complete',
                                                                       'live_fa', 'fa_per_hour', 'stream_recall',
                                                                       'latency_median_s', 'verifier_calls_per_hour')
                                               if x in run[k]}
                                for k in ('stage1_frame', 'cascade')}
        mean = {v: float(np.mean([rows[(v, s)]['cascade']['live_recall'] if rows[(v, s)]['cascade'] else 0.
                                  for s in SEEDS])) for v in VARIANTS}
        calls = {v: float(np.mean([rows[(v, s)]['cascade']['verifier_calls_per_hour'] if rows[(v, s)]['cascade'] else 0.
                                   for s in SEEDS])) for v in VARIANTS}
        best = max(VARIANTS, key=lambda v: (mean[v], -calls[v]))     # tie: fewer verifier calls
        out['budgets'][str(budget)] = {'mean_cascade_live_recall': mean, 'mean_verifier_calls_per_hour': calls,
                                       'chosen_variant': best,
                                       'per_seed': {f'{v}_seed{s}': r for (v, s), r in rows.items()}}
    main_b = out['budgets'][str(a.max_fa_hour)]
    v = main_b['chosen_variant']
    out['chosen'] = {'variant': v, 'thresholds': {
        f'seed{s}': {'t1': main_b['per_seed'][f'{v}_seed{s}']['cascade']['t1'],
                     't2': main_b['per_seed'][f'{v}_seed{s}']['cascade']['t2'],
                     'stage1_threshold': main_b['per_seed'][f'{v}_seed{s}']['stage1_frame']['t1']} for s in SEEDS}}
    (sel / 'selection.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out['chosen'], indent=1), json.dumps({b: x['mean_cascade_live_recall'] for b, x in out['budgets'].items()}))


BUDGETS = (50, 120, 250, 500, 1000, 2150, 4000, 8000, 14400)


def budget(a):
    """Extra, validation only (not pre-registered): the best cascade live recall under a verifier-call budget.
    For t2 on select's grid and t1 on 80 quantiles of the live keyword scores: live recall, other words,
    FA/h (negatives, 1 s hold-off) and verifier calls per hour; then, per budget, the highest live recall with
    other words <= 0.2 %, <= max_fa_hour FA/h and calls/h <= budget."""
    configs = [('mateo', VC.DV / 'cache_validation_sheila.npz')] + \
        [(f'r2_seed{s}_{v}', r2_file('validation', s, v)) for v in VARIANTS for s in SEEDS]
    v = dict(np.load(a.scores))
    report = {'note': 'extra analysis on validation only, not part of the pre-registered selection',
              'max_fa_hour': a.max_fa_hour, 'budgets': BUDGETS, 'configs': {}}
    all_points = {}
    for name, path in configs:
        c = dict(np.load(path))
        cas = VC.Cascade(c, v, 1, a.verify_ms)
        y = c['live_y']
        hours = float(c['neg_seconds']) / 3600
        vv, voff = v['live_va'], v['live_voff']
        vmax = np.array([vv[voff[i]:voff[i + 1]].max() for i in range(len(voff) - 1)])
        grid = np.unique(np.quantile(vmax[y == 1], np.linspace(0, .6, 31)).astype(np.int64))
        prop = np.sort(np.concatenate([np.maximum(hm, np.r_[VC.NEG, hm[:-1]])
                                       for hm in (VC.hop_max(d) for d in cas.d1['neg'])]))
        points = []
        for t2 in grid[grid > VC.NEG // 2]:
            live = np.array([s.max() for _, s in cas.traces('live', 'cascade', int(t2))])
            neg = cas.traces('neg', 'cascade', int(t2))
            pos = live[y == 1]
            cands = np.unique(np.quantile(pos[pos > VC.NEG], np.linspace(0, 1, 81)).astype(np.int64))
            neg = [(t[s >= cands[0]], s[s >= cands[0]]) for t, s in neg]
            for t1 in cands:
                other = float((live[y == 0] >= t1).mean())
                if other > .002:
                    continue
                points.append(dict(t1=int(t1), t2=int(t2), live_recall=round(float((pos >= t1).mean()), 4),
                                   live_other=round(other, 4), fa_per_hour=round(VC.fa_hour(neg, hours, t1), 3),
                                   calls_per_hour=round(float((len(prop) - np.searchsorted(prop, t1)) / hours), 1)))
        best = {}
        for b in BUDGETS:
            ok = [p for p in points if p['fa_per_hour'] <= a.max_fa_hour and p['calls_per_hour'] <= b]
            best[str(b)] = max(ok, key=lambda p: (p['live_recall'], -p['calls_per_hour'])) if ok else None
        report['configs'][name] = {'best_per_budget': best}
        all_points[name] = points
        print(name, {b: x and (x['live_recall'], x['calls_per_hour'], x['t1'], x['t2']) for b, x in best.items()}, flush=True)
    a.out.write_text(json.dumps(report, indent=1))
    (VC.DV / 'call_budget_points_validation.json').write_text(json.dumps(all_points))   # every grid point (git-ignored)


def proposals(cas):
    """Sorted per-request proposal scores on the negatives: the verifier runs at request k when this is >= t1
    (verifier_cascade.metrics' verifier_calls_per_hour)."""
    return np.sort(np.concatenate([np.maximum(hm, np.r_[VC.NEG, hm[:-1]]) for hm in (VC.hop_max(d) for d in cas.d1['neg'])]))


def operating_budget(cas, t2, max_fa_hour, max_calls, prop, max_live_fa=.002):
    """verifier_cascade.operating ('cascade', policy a) plus verifier calls per hour <= max_calls on the negatives."""
    c = cas.c
    y = c['live_y']
    hours = float(c['neg_seconds']) / 3600
    live = np.array([s.max() for _, s in cas.traces('live', 'cascade', t2)])
    neg = cas.traces('neg', 'cascade', t2)
    lo = np.unique(live[y == 1])
    lo = lo[lo > VC.NEG]
    if not len(lo):
        return None
    neg = [(t[s >= lo[0]], s[s >= lo[0]]) for t, s in neg]
    best = None
    for t1 in lo[::-1]:
        if (len(prop) - np.searchsorted(prop, t1)) / hours > max_calls:
            break                    # calls only grow as t1 falls
        if (live[y == 0] >= t1).mean() > max_live_fa:
            break
        fph = VC.fa_hour(neg, hours, t1)
        if fph > max_fa_hour:
            if fph > 5 * max_fa_hour:
                break
            continue
        rec = (live[y == 1] >= t1).mean()
        if best is None or rec > best[1]:
            best = (int(t1), rec)
    if best is None:
        return None
    r = VC.metrics(cas, 'cascade', best[0], t2, 'a')
    r.pop('_live_hits', None)
    return r


def select_budget(a):
    """A5: thresholds with the call limit for every seed x variant (validation), the variant choice, and the same
    rule on Mateo's cascade for reference. Writes results/R2/A/budget<calls>/selection.json."""
    v = dict(np.load(a.scores))
    out_dir = a.out_dir or KWS / f'research/results/R2/A/budget{a.max_calls:g}'
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = a.variants or VARIANTS
    report = {'rule': f'policy a; live other words <= 0.2 %, <= {a.max_fa_hour} FA/h and <= {a.max_calls:g} verifier '
                      'calls per hour on the validation negatives; t2 on select\'s grid; best by live recall, then '
                      'stream recall; variant by mean live recall (tie: fewer calls)', 'configs': {}}
    configs = [('mateo', VC.DV / 'cache_validation_sheila.npz')] + \
        [(f'{vr}_seed{s}', r2_file('validation', s, vr)) for vr in variants for s in SEEDS]
    for name, path in configs:
        cas = VC.Cascade(dict(np.load(path)), v, 1, a.verify_ms)
        y = cas.c['live_y']
        vv, voff = v['live_va'], v['live_voff']
        vmax = np.array([vv[voff[i]:voff[i + 1]].max() for i in range(len(voff) - 1)])
        grid = np.unique(np.quantile(vmax[y == 1], np.linspace(0, .6, 31)).astype(np.int64))
        prop = proposals(cas)
        cands = [r for t2 in grid[grid > VC.NEG // 2]
                 if (r := operating_budget(cas, int(t2), a.max_fa_hour, a.max_calls, prop))]
        best = max(cands, key=lambda r: (r['live_recall'], r['stream_recall'])) if cands else None
        report['configs'][name] = {'cascade': best, 'grid': [{k: r[k] for k in ('t1', 't2', 'live_recall', 'live_fa',
                                   'fa_per_hour', 'verifier_calls_per_hour')} for r in cands]}
        print(name, best and {k: best[k] for k in ('t1', 't2', 'live_recall', 'live_fa', 'fa_per_hour',
                                                   'stream_recall', 'verifier_calls_per_hour')}, flush=True)
    rec = lambda vr, k: float(np.mean([report['configs'][f'{vr}_seed{s}']['cascade'][k] for s in SEEDS]))
    variant = max(variants, key=lambda vr: (rec(vr, 'live_recall'), -rec(vr, 'verifier_calls_per_hour')))
    sel = json.loads((KWS / 'research/results/R2/A/selection.json').read_text())['chosen']['thresholds']
    report['scores'] = str(a.scores)
    report['mean_live_recall'] = {vr: rec(vr, 'live_recall') for vr in variants}
    report['chosen'] = {'variant': variant, 'thresholds': {
        f'seed{s}': {'t1': report['configs'][f'{variant}_seed{s}']['cascade']['t1'],
                     't2': report['configs'][f'{variant}_seed{s}']['cascade']['t2'],
                     'stage1_threshold': sel[f'seed{s}']['stage1_threshold']} for s in SEEDS}}
    (out_dir / 'selection.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('mean_live_recall', 'chosen')}, indent=1))


def paired(a):
    """B5: per-clip live hits of two cascades on the same stage-1 cache (e.g. Phase B vs Phase A verifier, each at
    its own validation thresholds) and the exact McNemar test (verifier_cascade.mcnemar). Prints one JSON line."""
    c = dict(np.load(r2_file(a.split, a.seed, a.variant)))
    y = c['live_y']
    hits = []
    for scores, t1, t2 in ((a.scores_a, a.t1_a, a.t2_a), (a.scores_b, a.t1_b, a.t2_b)):
        cas = VC.Cascade(c, dict(np.load(scores)), 1, a.verify_ms)
        live = np.array([s.max() for _, s in cas.traces('live', 'cascade', t2)])
        hits.append(live[y == 1] >= t1)
    r = VC.mcnemar(hits[0], hits[1])
    out = {'split': a.split, 'seed': a.seed, 'variant': a.variant, 'recall_a': round(float(hits[0].mean()), 4),
           'recall_b': round(float(hits[1].mean()), 4), 'only_b': r['only_cascade'], 'only_a': r['only_stage1'],
           'p_exact': r['p_exact']}
    print(json.dumps(out), flush=True)


def main():
    import argparse
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cache')
    c.add_argument('--split', choices=['validation', 'test'], required=True)
    for name in ('select', 'final'):
        e = sub.add_parser(name)
        e.add_argument('--seed', type=int, choices=SEEDS, required=True)
        e.add_argument('--variant', choices=VARIANTS, required=True)
        e.add_argument('scores', type=Path)
        e.add_argument('--verify-ms', type=float, default=0.)
        e.add_argument('--out', type=Path, required=True)
    sub.choices['select'].add_argument('--split', choices=['validation'], default='validation')
    sub.choices['select'].add_argument('--max-fa-hour', type=float, default=2.)
    sub.choices['select'].add_argument('--windows', type=int, nargs='+', default=[1])
    f = sub.choices['final']
    f.add_argument('--split', choices=['validation', 'test'], default='test')
    f.add_argument('--t1', type=int, required=True)
    f.add_argument('--t2', type=int, required=True)
    f.add_argument('--stage1-threshold', type=int, required=True)
    f.add_argument('--window', type=int, default=1)
    f.add_argument('--policy', choices=['a', 'b'], default='a')
    f.add_argument('--checkpoint-name')
    h = sub.add_parser('choose')
    h.add_argument('--max-fa-hour', type=float, default=1.6)
    h.add_argument('--sensitivity', type=float, nargs='*', default=[2.0])
    g = sub.add_parser('budget')
    g.add_argument('scores', type=Path)
    g.add_argument('--verify-ms', type=float, default=107.3)
    g.add_argument('--max-fa-hour', type=float, default=1.6)
    g.add_argument('--out', type=Path, required=True)
    sb = sub.add_parser('select-budget')
    sb.add_argument('scores', type=Path)
    sb.add_argument('--max-calls', type=float, default=2150.)
    sb.add_argument('--verify-ms', type=float, default=107.3)
    sb.add_argument('--max-fa-hour', type=float, default=1.6)
    sb.add_argument('--variants', nargs='+', choices=VARIANTS, help='default: both (Phase B: m, chosen in Phase A)')
    sb.add_argument('--out-dir', type=Path, help='default: results/R2/A/budget<calls>')
    pr = sub.add_parser('paired')
    pr.add_argument('--split', choices=['validation', 'test'], default='test')
    pr.add_argument('--seed', type=int, choices=SEEDS, required=True)
    pr.add_argument('--variant', choices=VARIANTS, default='m')
    pr.add_argument('--verify-ms', type=float, default=107.3)
    for x in ('a', 'b'):
        pr.add_argument(f'--scores-{x}', type=Path, required=True)
        pr.add_argument(f'--t1-{x}', type=int, required=True)
        pr.add_argument(f'--t2-{x}', type=int, required=True)
    a = p.parse_args()
    if a.cmd == 'paired':
        return paired(a)
    if a.cmd == 'cache':
        return cache(a)
    if a.cmd == 'choose':
        return choose(a)
    if a.cmd == 'budget':
        return budget(a)
    if a.cmd == 'select-budget':
        return select_budget(a)
    VC.cache_file = lambda split: r2_file(split, a.seed, a.variant)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    {'select': VC.select, 'final': VC.final}[a.cmd](a)


if __name__ == '__main__':
    main()
