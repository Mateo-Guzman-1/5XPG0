"""S1: energy of the E2 size-sweep models, from their board cycle counts and Vivado's power estimate.

No training and no board time: E2 measured every model bit-exactly on both fabrics. This script adds
the energy columns (energy_model.py) and writes

    research/results/S1/energy_models.csv    one row per model and fabric
    research/results/S1/energy_summary.csv   mean and sample sd over the seeds of each size
    research/results/S1/energy_fits.json     energy per neuron (linear fit) per fabric

    python analyze_energy.py
"""
import csv
import json
from pathlib import Path
import numpy as np
from energy_model import load_power, energy, F_CLK

ROOT = Path(__file__).resolve().parent.parent
E2 = ROOT / 'results/E2/models.csv'
OUT = ROOT / 'results/S1'
N_INPUT = 768
QUALITY = ['val_f1', 'test_f1', 'test_precision', 'test_recall', 'test_fpr', 'live_2of3_detected',
           'live_2of3_other', 'hidden_spikes_mean', 'model_bytes']
SENSITIVITY = {'tr6': 'power_toggle6.25', 'tr25': 'power_toggle25'}


def energy_row(name, hidden, seed, fabric, firmware, cycles_mean, cycles_max, macs, power, extra):
    """One output row: identification, cycles, energy (convenient units) and quality columns."""
    e = energy(cycles_mean, cycles_max, power['power'][fabric], macs)
    row = dict(name=name, hidden=hidden, seed=seed, fabric=fabric, firmware=firmware, macs=macs,
               cycles_mean=cycles_mean, cycles_max=cycles_max,
               t_mean_ms=e['t_mean'] * 1e3, t_max_ms=e['t_max'] * 1e3,
               e_inf_uJ=e['e_inf'] * 1e6, e_inf_all_uJ=e['e_inf_all'] * 1e6,
               e_win_busy_mJ=e['e_win_busy'] * 1e3, e_win_gated_mJ=e['e_win_gated'] * 1e3,
               e_win_scaled_mJ=e['e_win_scaled'] * 1e3, p_avg_busy_mW=e['p_avg_busy'] * 1e3,
               p_avg_gated_mW=e['p_avg_gated'] * 1e3, p_avg_scaled_mW=e['p_avg_scaled'] * 1e3,
               f_min_MHz=e['f_min'] / 1e6, e_per_mac_nJ=e['e_per_mac'] * 1e9)
    for tag, report in SENSITIVITY.items():
        row[f'e_inf_uJ_{tag}'] = energy(cycles_mean, cycles_max, power[report][fabric])['e_inf'] * 1e6
    row.update(extra)
    return row


def summarize(rows, keys=('hidden', 'fabric', 'firmware')):
    """Mean and sample sd of every numeric column, per group."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: tuple(str(k).zfill(4) for k in kv[0])):
        s = dict(zip(keys, key), seeds=len(members))
        for col, value in members[0].items():
            if col in keys or col in ('name', 'seed') or isinstance(value, str):
                continue
            v = np.array([m[col] for m in members], dtype=float)
            s[col] = v.mean()
            s[col + '_sd'] = v.std(ddof=1) if len(v) > 1 else 0.0
        out.append(s)
    return out


def fit(rows, fabric, min_hidden=0):
    """Least-squares line e_inf = slope * H + intercept over all seeds of one fabric."""
    pts = [(r['hidden'], r['e_inf_uJ']) for r in rows if r['fabric'] == fabric and r['hidden'] >= min_hidden]
    h, e = np.array(pts, dtype=float).T
    slope, intercept = np.polyfit(h, e, 1)
    pred = slope * h + intercept
    r2 = 1 - ((e - pred) ** 2).sum() / ((e - e.mean()) ** 2).sum()
    return dict(fabric=fabric, min_hidden=min_hidden, uJ_per_neuron=slope, uJ_fixed=intercept, r2=r2,
                points=len(h))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    power = {name: load_power(report=name) for name in ['power', *SENSITIVITY.values()]}
    rows = []
    with open(E2, newline='') as f:
        for m in csv.DictReader(f):
            if m['bit_exact'] != 'True':
                raise SystemExit(f"{m['name']}: not bit-exact on the board; its cycles cannot be used")
            hidden = int(m['hidden'])
            macs = N_INPUT * hidden + 2 * hidden     # dense input layer + dense output loop
            extra = {k: float(m[k]) for k in QUALITY}
            for fabric, firmware in (('base', 'plain'), ('kdot', 'kdot')):
                rows.append(energy_row(m['name'], hidden, int(m['seed']), fabric, firmware,
                                       float(m[f'{fabric}_cycles_mean']), float(m[f'{fabric}_cycles_max']),
                                       macs, power, extra))
    write_csv(OUT / 'energy_models.csv', rows)
    summary = summarize(rows)
    write_csv(OUT / 'energy_summary.csv', summary)
    fits = [fit(rows, fabric, min_hidden) for fabric in ('base', 'kdot') for min_hidden in (0, 32)]
    terms = {name: {fab: {k: v for k, v in p.items() if k != 'static_by_rail'} for fab, p in pw.items()}
             for name, pw in power.items()}
    (OUT / 'energy_fits.json').write_text(json.dumps(dict(f_clk_hz=F_CLK, fits=fits, power_terms_w=terms), indent=2))
    for s in summary:
        print(f"H={s['hidden']:>3} {s['fabric']:>4}: E_inf {s['e_inf_uJ']:9.2f} uJ  (all-in {s['e_inf_all_uJ']:9.2f})  "
              f"t_max {s['t_max_ms']:6.2f} ms  P_avg busy/gated/scaled {s['p_avg_busy_mW']:6.1f} / "
              f"{s['p_avg_gated_mW']:6.1f} / {s['p_avg_scaled_mW']:6.1f} mW  nJ/MAC {s['e_per_mac_nJ']:.2f}")
    for f_ in fits:
        print(f"fit {f_['fabric']} H>={f_['min_hidden']}: {f_['uJ_per_neuron']:.3f} uJ/neuron + {f_['uJ_fixed']:.2f} uJ (R2 {f_['r2']:.6f})")


if __name__ == '__main__':
    main()
