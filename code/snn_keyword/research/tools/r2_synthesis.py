"""R2 synthesis (research/PLAN_R2_GRU.md A4): cascade results joined with cost.

Inputs:
  results/R2/A/h48_k64_seed<s>.json   verifier_cascade final on our stage 1 (r2_cascade.py final): blocks
                                      stage1_frame, stage1_hop (stage 1 alone) and cascade
  results/R2/A/mateo_cascade_test.json  Mateo's cascade, reproduced (gate G-A2)
  results/R1/models.csv               our stage 1 on ksdot + LIF table: worst compute, memory, energy per inference
  results/S6/power.json (energy_model.load_power)  Vivado power terms of the ksdot fabric
Outputs (results/R2/):
  models.csv      one row per seed x design
  designs.csv     mean and sample sd over seeds, plus Mateo's references
  compliance.csv  pass / fail per requirement per design (on the means)

Cost model (stated in the report):
  worst compute per request = t_max(stage 1) + T_V,max (107.3 ms, verifier on plain kdot);
  word end to decision = stream median latency (includes T_V,max) + t_max(stage 1) + 1.9 ms PC and link;
  memory = stage-1 model + 53,320 B (int8 verifier export);
  energy per hour = static + 14,400 x E_inf(stage 1) + calls/h x E_V, E_V = P_PL,dyn(ksdot) x T_V,mean (105.83 ms);
  average PL power: busy-polling = P_PL,dyn + P_PL,stat (unchanged, the core never idles);
                    clock-gated = P_PL,stat + P_infra,dyn + (14,400 x E_core,stage1 + calls/h x P_core,dyn x T_V,mean) / 3600 s.
  Sensitivity: E_V at Alex's dual-core PL power (0.727 W, FINAL REPORT Table "GRU verifier by custom-instruction level").
Verifier calls per hour are counted on the about 19 h negatives stream (Speech Commands words and LibriSpeech).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from energy_model import F_CLK, load_power  # noqa: E402

RES = TOOLS.parent / 'results'
OUT = RES / 'R2'
SEEDS = (0, 1, 2)
T_V_MAX, T_V_MEAN = 107.3e-3, 105.83e-3     # s; FINAL REPORT Parallel section (Verilator SoC timer, plain kdot)
VERIFIER_BYTES = 53320                       # int8 export of sheila_verifier_head_near.pt
P_PL_ALEX = 0.727                            # W, Alex's final dual-core build (sensitivity)
OVERHEAD_MS = 1.9                            # as R1: PC features + network and server polling
REQUESTS_H = 3600 / 0.25
REQ = dict(live_recall=('>=', .85), fa_per_hour=('<=', 2.), live_other=('<=', .002), latency_ms=('<=', 500.),
           t_max_ms=('<=', 200.), memory_kib=('<=', 144.), p_avg_busy_W=('<=', 1.))
COST = ('latency_ms', 't_max_ms', 'memory_kib', 'p_avg_busy_W')


def ok(key, value):
    op, target = REQ[key]
    return value is not None and (value >= target if op == '>=' else value <= target)


def detection(block):
    return dict(t1=block['t1'], t2=block['t2'], live_recall=block['live_recall'],
                live_recall_complete=block['live_recall_complete'], live_other=block['live_fa'],
                fa_per_hour=block['fa_per_hour'], stream_recall=block['stream_recall'],
                latency_stream_ms=None if block['latency_median_s'] is None else 1000 * block['latency_median_s'],
                mswc_recall=block['mswc_recall'], mswc_other_fa=block['mswc_other_fa'],
                calls_per_hour=block.get('verifier_calls_per_hour', 0.))


def cost(det, s1, p, verifier):
    """s1: R1 models.csv row of the stage-1 model on ksdot + LUT; verifier: whether the GRU runs."""
    t1_ms, e_inf = float(s1['t_max_ms']), float(s1['e_inf_uJ']) * 1e-6
    calls = det['calls_per_hour'] if verifier else 0.
    e_core_s1 = p['core_dyn'] * float(s1['cycles_mean']) / F_CLK
    out = dict(t_max_ms=t1_ms + (1000 * T_V_MAX if verifier else 0.),
               memory_kib=float(s1['memory_kib']) + (VERIFIER_BYTES / 1024 if verifier else 0.),
               latency_ms=None if det['latency_stream_ms'] is None else det['latency_stream_ms'] + t1_ms + OVERHEAD_MS,
               e_h_static_J=p['pl_static'] * 3600, e_h_stage1_J=REQUESTS_H * e_inf,
               e_h_verifier_J=calls * p['pl_dyn'] * T_V_MEAN, e_v_mJ=1000 * p['pl_dyn'] * T_V_MEAN if verifier else 0.,
               e_h_verifier_alex_J=calls * P_PL_ALEX * T_V_MEAN,
               p_avg_busy_W=p['pl_dyn'] + p['pl_static'],
               p_avg_gated_W=p['pl_static'] + p['infra_dyn'] +
               (REQUESTS_H * e_core_s1 + calls * p['core_dyn'] * T_V_MEAN) / 3600)
    out['e_h_dynamic_J'] = out['e_h_stage1_J'] + out['e_h_verifier_J']
    # Like for like with Alex's figures (PL power incl. static x compute time) and with busy-polling.
    out['e_h_work_J'] = REQUESTS_H * float(s1['e_inf_all_uJ']) * 1e-6 + calls * (p['pl_dyn'] + p['pl_static']) * T_V_MEAN
    out['e_h_busy_J'] = out['p_avg_busy_W'] * 3600
    out['verifier_busy_share'] = calls * T_V_MEAN / 3600
    return out


def mean_sd(rows, design):
    rs = [r for r in rows if r['design'] == design]
    out = {'design': design, 'seeds': len(rs)}
    for k, v in rs[0].items():
        if k in ('design', 'seed', 'name') or not isinstance(v, (int, float)) or v is None:
            continue
        vals = [r[k] for r in rs if r[k] is not None]
        out[k] = float(np.mean(vals)) if len(vals) == len(rs) else None
        out[k + '_sd'] = float(np.std(vals, ddof=1)) if len(vals) == len(rs) and len(vals) > 1 else None
    return out


def main():
    p = load_power()['ksdot']
    r1 = {r['name']: r for r in csv.DictReader(open(RES / 'R1/models.csv', encoding='utf-8'))
          if r['config'] == 'h48_k64' and r['variant'] == 'ksdot_lut'}
    rows = []
    for s in SEEDS:
        path = OUT / f'A/h48_k64_seed{s}.json'
        if not path.exists():
            continue
        rep = json.loads(path.read_text())
        s1 = r1[f'h48_k64_seed{s}']
        # Stage 1 alone as measured in R1 (req_eval, <= 2 FA/h on validation).
        r1_det = dict(t1=int(s1['threshold']), t2=None, live_recall=float(s1['live_recall']),
                      live_recall_complete=float(s1['live_recall_complete']), live_other=float(s1['live_other']),
                      fa_per_hour=float(s1['fa_per_hour']), stream_recall=float(s1['stream_recall']),
                      latency_stream_ms=float(s1['latency_stream_ms']), mswc_recall=None, mswc_other_fa=None,
                      calls_per_hour=0.)
        blocks = [('stage1_R1', r1_det, False, None),
                  ('stage1_cascade_rule', detection(rep['stage1_frame']), False, None),
                  ('cascade_A', detection(rep['cascade']), True, rep['cascade'])]
        for budget in sorted((OUT / 'A').glob('budget*')):     # A5: the call-limited rule
            path_b = budget / f'h48_k64_seed{s}.json'
            if path_b.exists():
                cb = json.loads(path_b.read_text())['cascade']
                blocks.append((f'cascade_A_calls{budget.name[6:]}', detection(cb), True, cb))
        for design, det, ver, block in blocks:
            row = dict(name=f'h48_k64_seed{s}', design=design, seed=s, **det, **cost(det, s1, p, ver))
            if block:
                row['paired_only_cascade'] = block['paired_vs_stage1_frame']['only_cascade']
                row['paired_only_stage1'] = block['paired_vs_stage1_frame']['only_stage1']
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / 'models.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

    designs = [mean_sd(rows, d) for d in dict.fromkeys(r['design'] for r in rows)]
    # Mateo's cascade on the same tests (reproduced in G-A2), with his own cost figures.
    m = json.loads((OUT / 'A/mateo_cascade_test.json').read_text())
    for design, block, t_max, e_req_mJ, e_v_mJ in (
            ('mateo_stage1', m['stage1_frame'], 11.42, 7.40, 0.),
            ('mateo_cascade', m['cascade'], 117.3, 7.40, 76.9)):
        det = detection(block)
        calls = det['calls_per_hour'] if e_v_mJ else 0.
        designs.append(dict(design=design, seeds=1, **det, t_max_ms=t_max,
                            latency_ms=det['latency_stream_ms'] + OVERHEAD_MS,
                            memory_kib=None, e_h_stage1_J=REQUESTS_H * e_req_mJ / 1000,
                            e_h_verifier_J=calls * e_v_mJ / 1000, e_v_mJ=e_v_mJ, p_avg_busy_W=P_PL_ALEX,
                            e_h_work_J=REQUESTS_H * e_req_mJ / 1000 + calls * e_v_mJ / 1000,
                            e_h_busy_J=P_PL_ALEX * 3600,
                            source='reproduced final (G-A2); cost: FINAL REPORT Parallel section (stage 1 10.17 ms mean, '
                                   '11.42 ms max, 7.40 mJ per request; verifier 105.83 ms, 76.9 mJ per call; PL 0.727 W)'))
    dkeys = list(dict.fromkeys(k for d in designs for k in d))
    with open(OUT / 'designs.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, dkeys); w.writeheader(); w.writerows(designs)
    comp = []
    for d in designs:
        c = {'design': d['design']}
        for k in REQ:
            c[k] = ok(k, d.get(k)) if d.get(k) is not None else None
        c['all'] = all(c[k] for k in REQ if c[k] is not None)     # over the known requirements
        c['unknown'] = ' '.join(k for k in REQ if c[k] is None)
        comp.append(c)
    with open(OUT / 'compliance.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(comp[0])); w.writeheader(); w.writerows(comp)
    for d, c in zip(designs, comp):
        print(d['design'], {k: (round(d[k], 4) if isinstance(d.get(k), float) else d.get(k), c[k]) for k in REQ})


if __name__ == '__main__':
    main()
