"""R1 synthesis: requirement results (req_eval.py) joined with the S-study board and energy data.

Inputs: results/R1/requirements/<config>.json, results/S5/designs.csv (cycles, memory, energy per
model x hardware variant, from bit-exact board runs; the model sha256 is checked against R1).
Outputs (results/R1/):
  models.csv      one row per model x variant
  designs.csv     mean and sample sd over the 3 seeds per design (config / variant)
  compliance.csv  pass / fail per requirement per design
  choice.json     dense and pruned champions (PLAN_R1.md choice rule) and the sensitivity checks
"""
import csv
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / 'results'
OUT = RES / 'R1'
T_HOP = 0.25
OVERHEAD_MS = 1.9   # PC features 0.40 ms + network and server polling 1.5 ms (E2: round trip = inference + 0.6-1.5 ms)
REQ = dict(live_recall=('>=', .85), fa_per_hour=('<=', 2.), live_other=('<=', .002), latency_ms=('<=', 500.),
           t_max_ms=('<=', 200.), memory_kib=('<=', 144.), p_avg_busy_W=('<=', 1.))
COST = ('latency_ms', 't_max_ms', 'memory_kib', 'p_avg_busy_W')
# Hardware variants kept per config family (S5/designs.csv names).
VARIANTS = {'dense': ('base', 'kdot', 'kdot_lut'), 'bits': ('packed', 'kdot_lut'),
            'pruned': ('sparse', 'ksdot_lut'), 'pruned_bits': ('ksdot_lut',)}
LABEL = {'base': 'RV32IM', 'kdot': 'kdot', 'kdot_lut': 'kdot+LUT', 'packed': 'RV32IM packed',
         'sparse': 'RV32IM sparse', 'ksdot_lut': 'ksdot+LUT'}


def family(config):
    if '_k' in config:
        return 'pruned_bits' if '_b' in config else 'pruned'
    return 'bits' if '_b' in config else 'dense'


def ok(key, value):
    op, target = REQ[key]
    return value is not None and (value >= target if op == '>=' else value <= target)


def references():
    """Teammates' designs on the same test sets (Mateo's final_eval), read from git (origin/ALIF-Layer1)."""
    import subprocess
    show = lambda f: json.loads(subprocess.run(['git', 'show', f'origin/ALIF-Layer1:code/snn_keyword/results/{f}'],
                                               capture_output=True, check=True, cwd=RES).stdout)
    final, casc = show('final_sheila_test.json'), show('final_sheila_head_near_cascade_test.json')
    s1, c = casc['stage1_frame'], casc['cascade']
    refs = [dict(name='Mateo: ALIF streaming SNN (stage 1)', live_recall=s1['live_recall'], live_other=s1['live_fa'],
                 fa_per_hour=s1['fa_per_hour'], latency_ms=1000 * s1['latency_median_s'], t_max_ms=10.3,
                 source='final_sheila_head_near_cascade_test.json stage1_frame; compute: README (RTL, 10.3 ms)'),
            dict(name='Mateo: ALIF SNN + GRU verifier (cascade)', live_recall=c['live_recall'], live_other=c['live_fa'],
                 fa_per_hour=c['fa_per_hour'], latency_ms=1000 * c['latency_median_s'], t_max_ms=117.,
                 source='final_sheila_head_near_cascade_test.json cascade; worst compute: README (117 ms)')]
    for key, label in (('damien_ourrule', "Damien: 8x16 clip SNN, Mateo's rule"), ('damien_own', 'Damien: 8x16 clip SNN, own margin')):
        m = final['models'][key]
        refs.append(dict(name=label, live_recall=m['live_clean']['recall'], live_other=m['live_clean']['other_accepted'],
                         fa_per_hour=m['negatives']['fa_per_hour'], latency_ms=1000 * m['stream_1h']['latency_median_s'],
                         t_max_ms=10.4, source=f'final_sheila_test.json {key}; compute: FINAL REPORT Table "Final Sheila detector results"'))
    (OUT / 'references.json').write_text(json.dumps(dict(harness='same test sets (robust_eval / final_eval)', designs=refs), indent=2))
    return refs


def main():
    references()
    board = {}
    for r in csv.DictReader(open(RES / 'S5/designs.csv', encoding='utf-8')):
        board[(r['name'], r['variant'])] = r
    rows = []
    for path in sorted((OUT / 'requirements').glob('*.json')):
        rep = json.loads(path.read_text())
        config = rep['config']
        for name, m in rep['models'].items():
            t, old = m['test'], m['test_at_s_rule']
            for variant in VARIANTS[family(config)]:
                b = board.get((name, variant))
                if b is None:   # not measured on the board (S4 ran the LIF table at H = 16, 48, 95 only)
                    continue
                bj = json.loads((RES.parent / b['board_file']).read_text())
                assert bj.get('model_sha256', m['sha256']) == m['sha256'], f'{name}: board run used other weights'
                t_max = float(b['t_max_ms'])
                lat = None if t is None or t['stream']['latency_median_s'] is None else \
                    1000 * t['stream']['latency_median_s'] + t_max + OVERHEAD_MS
                rows.append(dict(
                    name=name, config=config, seed=int(name.rsplit('seed', 1)[1]), family=family(config),
                    variant=variant, hardware=LABEL[variant], design=f'{config}/{variant}',
                    hidden=int(b['hidden']), bits=int(b['bits']), fan_in=int(b['fan_in']),
                    threshold=None if t is None else t['threshold'],
                    live_recall=None if t is None else t['live_recall'],
                    live_recall_complete=None if t is None else t['live_recall_complete'],
                    live_other=None if t is None else t['live_other'],
                    fa_per_hour=None if t is None else t['fa_per_hour'],
                    fa_libri=None if t is None else t['fa_by_source'].get('librispeech', 0),
                    stream_recall=None if t is None else t['stream']['recall'],
                    latency_stream_ms=None if t is None or t['stream']['latency_median_s'] is None else 1000 * t['stream']['latency_median_s'],
                    latency_ms=lat,
                    s_rule_live_recall=old['live_recall'], s_rule_live_other=old['live_other'],
                    s_rule_fa_per_hour=old['fa_per_hour'],
                    val_selection_recall=None if m['validation']['selection'] is None else m['validation']['selection']['live_recall'],
                    test_f1=m['clips_full']['f1'], test_f1_subset=m['clips_damien_subset']['f1'],
                    precision_subset=m['clips_damien_subset']['precision'], recall_clip=m['clips_full']['recall'],
                    t_max_ms=t_max, memory_kib=int(b['memory']) / 1024, cycles_mean=float(b['cycles_mean']),
                    e_inf_uJ=float(b['e_inf_uJ']), e_inf_all_uJ=float(b['e_inf_all_uJ']),
                    p_avg_busy_W=float(b['p_avg_busy_mW']) / 1000, p_avg_gated_W=float(b['p_avg_gated_mW']) / 1000,
                    p_avg_scaled_W=float(b['p_avg_scaled_mW']) / 1000,
                    e_win_busy_mJ=float(b['p_avg_busy_mW']) * T_HOP, e_win_gated_mJ=float(b['p_avg_gated_mW']) * T_HOP,
                    e_win_scaled_mJ=float(b['p_avg_scaled_mW']) * T_HOP))
    OUT.mkdir(exist_ok=True)
    keys = list(rows[0])
    with open(OUT / 'models.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

    # Mean and sample sd over seeds per design.
    num = [k for k in keys if isinstance(rows[0][k], (int, float)) and k not in ('seed', 'hidden', 'bits', 'fan_in')]
    designs = []
    for d in dict.fromkeys(r['design'] for r in rows):
        rs = [r for r in rows if r['design'] == d]
        out = {k: rs[0][k] for k in ('design', 'config', 'family', 'variant', 'hardware', 'hidden', 'bits', 'fan_in')}
        out['seeds'] = len(rs)
        for k in num:
            v = [r[k] for r in rs if r[k] is not None]
            out[k] = float(np.mean(v)) if len(v) == len(rs) else None
            out[k + '_sd'] = float(np.std(v, ddof=1)) if len(v) == len(rs) and len(v) > 1 else None
        designs.append(out)
    dkeys = list(designs[0])
    with open(OUT / 'designs.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, dkeys); w.writeheader(); w.writerows(designs)

    # Compliance on the design means.
    comp = []
    for d in designs:
        c = {'design': d['design']}
        for k in REQ:
            c[k] = ok(k, d[k])
        c['all'] = all(c[k] for k in REQ)
        c['cost'] = all(c[k] for k in COST)
        comp.append(c)
    with open(OUT / 'compliance.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(comp[0])); w.writeheader(); w.writerows(comp)
    passes = {c['design']: c for c in comp}

    def champion(pool, metric, sd):
        best = max(pool, key=lambda d: d[metric])
        bar = best[metric] - sd(best)
        tied = [d for d in pool if d[metric] >= bar]
        return min(tied, key=lambda d: d['e_inf_uJ']), best, bar, tied

    def rule(metric, pooled):
        cand = [d for d in designs if passes[d['design']]['cost'] and d[metric] is not None]
        pooled_sd = float(np.sqrt(np.mean([d[metric + '_sd'] ** 2 for d in cand])))
        sd = (lambda d: pooled_sd) if pooled else (lambda d: d[metric + '_sd'])
        full = [d for d in cand if passes[d['design']]['all']]
        res = {'metric': metric, 'sd': 'pooled' if pooled else 'own', 'pooled_sd': pooled_sd,
               'meets_all_requirements': [d['design'] for d in full]}
        if full:
            res['winner'] = min(full, key=lambda d: d['e_inf_uJ'])['design']
            return res
        out = {}
        for fam, members in (('dense', ('dense', 'bits')), ('pruned', ('pruned', 'pruned_bits'))):
            pool = [d for d in cand if d['family'] in members]
            ch, best, bar, tied = champion(pool, metric, sd)
            out[fam] = dict(champion=ch['design'], value=ch[metric], sd=ch[metric + '_sd'], e_inf_uJ=ch['e_inf_uJ'],
                            best=best['design'], best_value=best[metric], bar=bar, tied=[d['design'] for d in tied])
        D, P = out['dense'], out['pruned']
        dd = next(d for d in designs if d['design'] == D['champion'])
        threshold = D['value'] - sd(dd)
        res |= out
        res['pruned_not_worse'] = bool(P['value'] >= threshold)
        res['verdict'] = 'pruned champion alone' if res['pruned_not_worse'] else 'both champions'
        res['tradeoff'] = dict(points=100 * (D['value'] - P['value']) if metric != 'test_f1' else D['value'] - P['value'],
                               energy_saving_pct=100 * (1 - P['e_inf_uJ'] / D['e_inf_uJ']))
        return res

    choice = {'rule': 'PLAN_R1.md: two champions (dense, pruned); live recall at the validation threshold',
              'main': rule('live_recall', False),
              'sensitivity_pooled_sd': rule('live_recall', True),
              'sensitivity_test_f1': rule('test_f1', False)}
    (OUT / 'choice.json').write_text(json.dumps(choice, indent=2))
    print(json.dumps(choice['main'], indent=1))
    for d in sorted(designs, key=lambda d: -(d['live_recall'] or 0)):
        print(f"{d['design']:22s} live {d['live_recall']:.3f}±{d['live_recall_sd']:.3f} other {100*d['live_other']:.2f}% "
              f"FA/h {d['fa_per_hour']:.2f} lat {d['latency_ms']:.0f} ms F1 {d['test_f1']:.3f}/{d['test_f1_subset']:.3f} "
              f"t {d['t_max_ms']:.2f} ms {d['memory_kib']:.1f} KiB E {d['e_inf_uJ']:.0f} uJ P {d['p_avg_busy_W']:.3f} W "
              f"{'ALL' if passes[d['design']]['all'] else ''}")


if __name__ == '__main__':
    main()
