"""S5: one table of all measured designs, the Pareto front, and the pre-registered choice.

A design is a model configuration (H, bits, fan-in) run with one firmware variant; its numbers are
the means over the seeds (collect_designs.py, research/results/S5/designs.csv).

Rule (research/PLAN_SYNAPSE.md, fixed before the experiments): a design qualifies if its mean
validation F1 is at least the best design's mean minus that design's sample sd, its worst-case
latency is below 250 ms and its model fits 144 KiB. The qualifying design with the lowest energy
per inference (E_inf) is recommended. Sensitivity: the same rule with the pooled sd of all designs.

    research/results/S5/design_summary.csv   one row per design (means and sds)
    research/results/S5/choice.json          rule, thresholds, qualifying designs, choice

    python synthesis.py
"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
S5 = ROOT / 'results/S5'
LATENCY_MS, MEMORY_B = 250.0, 147456
NUMERIC = ['val_f1', 'test_f1', 'live_2of3_detected', 'live_2of3_other', 'e_inf_uJ', 'e_inf_all_uJ', 't_max_ms',
           'cycles_mean', 'memory', 'macs', 'p_avg_gated_mW', 'p_avg_scaled_mW', 'p_avg_busy_mW']


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['config'], r['variant']), []).append(r)
    out = []
    for (config, variant), g in sorted(groups.items()):
        s = dict(design=f'{config}/{variant}', config=config, variant=variant, fabric=g[0]['fabric'],
                 hidden=int(g[0]['hidden']), bits=int(g[0]['bits']), fan_in=int(g[0]['fan_in']), seeds=len(g))
        for k in NUMERIC:
            v = np.array([float(r[k]) for r in g])
            s[k], s[k + '_sd'] = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
        out.append(s)
    return out


def pareto(designs, quality='val_f1', cost='e_inf_uJ'):
    """Designs that no other design beats on both lower energy and higher quality."""
    front = []
    for d in designs:
        if not any(o[cost] <= d[cost] and o[quality] >= d[quality] and (o[cost] < d[cost] or o[quality] > d[quality])
                   for o in designs):
            front.append(d['design'])
    return front


def choose(designs, sd_key):
    best = max(designs, key=lambda d: d['val_f1'])
    bar = best['val_f1'] - (best['val_f1_sd'] if sd_key == 'best' else sd_key)
    ok = [d for d in designs if d['val_f1'] >= bar and d['t_max_ms'] <= LATENCY_MS and d['memory'] <= MEMORY_B]
    ok.sort(key=lambda d: (d['e_inf_uJ'], d['memory']))
    return dict(best=best['design'], best_val_f1=best['val_f1'], best_val_f1_sd=best['val_f1_sd'], threshold=bar,
                qualifying=[d['design'] for d in ok], choice=ok[0]['design'] if ok else None)


def main():
    with open(S5 / 'designs.csv', newline='') as f:
        # Controls are not candidate designs: rounding-only models (S2) and the one-seed paired runs of
        # pruned models on dense kdot (S6, they only show that dense kdot cannot skip zeros).
        rows = [r for r in csv.DictReader(f) if r['method'] == 'qat' and r['variant'] != 'kdot_lut_on_ksdot']
    designs = summarize(rows)
    with open(S5 / 'design_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(designs[0]))
        w.writeheader()
        w.writerows(designs)
    # One sd per configuration (validation F1 does not depend on the firmware), pooled over configurations.
    per_config = {d['config']: d['val_f1_sd'] for d in designs}
    pooled = float(np.sqrt(np.mean(np.square(list(per_config.values())))))
    result = dict(rule='mean val F1 >= best - sd, t_max <= 250 ms, memory <= 144 KiB; lowest E_inf',
                  primary=choose(designs, 'best'), sensitivity_pooled_sd=dict(pooled_sd=pooled, **choose(designs, pooled)),
                  pareto_val_f1_vs_e_inf=pareto(designs))
    (S5 / 'choice.json').write_text(json.dumps(result, indent=2))
    for name in ('primary', 'sensitivity_pooled_sd'):
        r = result[name]
        print(f"{name}: best {r['best']} {r['best_val_f1']:.4f} (sd {r['best_val_f1_sd']:.4f}); threshold {r['threshold']:.4f}; "
              f"{len(r['qualifying'])} qualify; choice {r['choice']}")
    print('Pareto front:', result['pareto_val_f1_vs_e_inf'])


if __name__ == '__main__':
    main()
