"""R3 energy (research/PLAN_R3_GATING.md 2E): average PL power of B1 and B2 with clock gating, as built and with the
address-decoded BRAM (option B).

Inputs:
  results/R3/power.json        per-state Vivado reports of the routed designs (r3_power_states.tcl, parse_power.py):
                               folders asbuilt_rw8, asbuilt_rw1, decomp_rw1; states busy (CPU clock always on = awake),
                               idle (CPU asleep), day_B1, day_B2 (time-averaged clock), each at 12.5 / 6.25 / 25 % toggle
  results/R3/phase2/activity.json   r3_verify.py duties: awake cycles per request without a verifier call (A_S1),
                               verifier cycles (C_V, MB[21] of command 7), the layout's BRAM enable probabilities
  results/R3/phase2/verification.json, results/R3/fabric.json   worst compute, cost
  results/R2/designs.csv       verifier calls per hour (mean, sd over 9 cascades); B JSONs: representative configuration
  results/S6/power.json        busy-polling reference (energy_model.load_power()['ksdot']: 0.367 W)
  results/R3/phase1/hypothesis.json, phase2/hypothesis_rw1.json, phase2/hypothesis_decomp.json   predictions
Model (f = 100 MHz, 14,400 requests per hour; P = PL dynamic + PL static of a state report):
  E_h = 3600 P_idle + (P_awake - P_idle) (14,400 A_S1 + N_V C_V) / f;  P_avg = E_h / 3600;  saving = 1 - P_avg / P_busy.
  With the BRAM enables fixed by the layout, awake power is the same in stage 1 and in the verifier (one 'busy' state).
  Cross-check: the day_B1 / day_B2 reports (CPU clock at the time-averaged duty).
Every number is "Vivado estimate + RTL-simulated cycles" (no board measurement).
Writes results/R3/energy.json and energy.csv.
"""
import csv
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from energy_model import F_CLK, load_power, split  # noqa: E402

RES = TOOLS.parent / 'results'
R3 = RES / 'R3'
REQ_H = 14400
T_V_R2_MAX_MS = 107.3
RATES = ('12.5', '6.25', '25')
VARIANTS = ('asbuilt_rw8', 'asbuilt_rw1', 'decomp_rw1')
STATES = ('busy', 'idle', 'day_B1', 'day_B2')
DESIGNS = {'B1': ('cascade_B', RES / 'R2/B/v1/h48_k64_seed2.json'),
           'B2': ('cascade_B_calls2150', RES / 'R2/B/v1/budget2150/h48_k64_seed2.json')}


def rnd(x, n=4):
    return None if x is None else round(float(x), n)


def state_power(report):
    s = split(report)
    return dict(p_pl=s['pl_dyn'] + s['pl_static'], pl_dyn=s['pl_dyn'], pl_static=s['pl_static'], core_dyn=s['core_dyn'],
                infra_dyn=s['infra_dyn'])


def components(report):
    """On-chip rows and hierarchy of one parsed report (for the stacked figure)."""
    num = lambda v: 0.0 if isinstance(v, str) else float(v)
    chip = {r['On-Chip']: num(r['Power (W)']) for r in report['on_chip']}
    hier = {r['Name']: float(r['Power (W)']) for r in report['hierarchy']}
    s = split(report)
    return dict(pl_static=s['pl_static'], clocks=chip.get('Clocks'), signals=chip.get('Signals'),
                logic=chip.get('Slice Logic'), bram=chip.get('Block RAM'), dsp=chip.get('DSPs'),
                u_soc=hier.get('u_soc'), u_cpu=hier.get('u_cpu'), u_kdot=hier.get('u_kdot'),
                u_poisson=hier.get('u_poisson'), u_ps_if=hier.get('u_ps_if'), pc0=hier.get('pc0'))


def designs():
    rows = {r['design']: r for r in csv.DictReader(open(RES / 'R2/designs.csv', encoding='utf-8'))}
    out = {}
    for code, (name, rep) in DESIGNS.items():
        c = json.loads(Path(rep).read_text(encoding='utf-8'))['cascade']
        r = rows[name]
        out[code] = dict(calls=float(r['calls_per_hour']), calls_sd=float(r['calls_per_hour_sd']),
                         calls_rep=float(c['verifier_calls_per_hour']), latency_ms_r2=float(r['latency_ms']),
                         live=float(r['live_recall']), fa_h=float(r['fa_per_hour']), other=float(r['live_other']),
                         memory_kib=float(r['memory_kib']))
    return out


def energy(p_idle, p_awake, a_s1, c_v, calls):
    awake_cycles = REQ_H * a_s1 + calls * c_v
    e_h = 3600 * p_idle + (p_awake - p_idle) * awake_cycles / F_CLK
    return e_h / 3600, e_h, awake_cycles / F_CLK / 3600


def main():
    power = json.loads((R3 / 'power.json').read_text())
    act = json.loads((R3 / 'phase2/activity.json').read_text())
    ver = json.loads((R3 / 'phase2/verification.json').read_text())
    fab = json.loads((R3 / 'fabric.json').read_text())
    busy_ref = load_power()['ksdot']
    p_busy = busy_ref['pl_dyn'] + busy_ref['pl_static']
    D = designs()
    out = dict(method='Vivado estimate (vectorless defaults; gated CPU clock at the simulated duty; BRAM enable '
                      'probability of the built layout) + RTL-simulated cycles; no board measurement',
               p_busy_W=rnd(p_busy), variants={})
    rows = []
    for var in VARIANTS:
        if var not in power:
            continue
        a = act[var]['activity']
        rw = act[var]['read_wait']
        st = {s: {r: state_power(power[var][f'power_{s}_tr{r}']) for r in RATES} for s in STATES}
        a_s1, c_v = a['awake_s1_cycles'], a['verifier_cycles']['mean']
        v = dict(layout=act[var]['layout'], read_wait=rw,
                 states={s: {r: {k: rnd(x) for k, x in d.items()} for r, d in sd.items()} for s, sd in st.items()},
                 components={s: components(power[var][f'power_{s}_tr12.5']) for s in ('busy', 'idle')},
                 a_s1_cycles=a_s1, c_v_cycles=c_v, c_v_max=a['verifier_cycles']['max'],
                 worst_compute_ms=ver['sets'][f'rw{rw}']['worst_compute_ms'], designs={})
        for code, d in D.items():
            res = {}
            for rate in RATES:
                pi, pw = st['idle'][rate]['p_pl'], st['busy'][rate]['p_pl']
                for label, calls in (('mean', d['calls']), ('mean-sd', d['calls'] - d['calls_sd']),
                                     ('mean+sd', d['calls'] + d['calls_sd']), ('rep', d['calls_rep'])):
                    pa, eh, aw = energy(pi, pw, a_s1, c_v, calls)
                    res.setdefault(rate, {})[label] = dict(calls=calls, p_avg_W=rnd(pa), e_h_J=rnd(eh, 1),
                                                           saving=rnd(1 - pa / p_busy), awake_share=rnd(aw, 5))
                res[rate]['day_report_W'] = rnd(st[f'day_{code}'][rate]['p_pl'])
            res['latency_ms'] = rnd(d['latency_ms_r2'] - T_V_R2_MAX_MS + a['verifier_cycles']['max'] / 1e5, 1)
            v['designs'][code] = res
            m = res['12.5']['mean']
            rows.append(dict(variant=var, layout=v['layout'], read_wait=rw, design=code, calls_per_hour=d['calls'],
                             p_idle_W=rnd(st['idle']['12.5']['p_pl']), p_awake_W=rnd(st['busy']['12.5']['p_pl']),
                             a_s1_cycles=a_s1, c_v_cycles=c_v, awake_share=m['awake_share'], p_avg_W=m['p_avg_W'],
                             p_avg_lo_W=res['12.5']['mean-sd']['p_avg_W'], p_avg_hi_W=res['12.5']['mean+sd']['p_avg_W'],
                             p_avg_rep_W=res['12.5']['rep']['p_avg_W'], e_h_J=m['e_h_J'], saving=m['saving'],
                             day_report_W=res['12.5']['day_report_W'], p_avg_tr6_W=res['6.25']['mean']['p_avg_W'],
                             p_avg_tr25_W=res['25']['mean']['p_avg_W'], worst_compute_ms=v['worst_compute_ms'],
                             latency_ms=res['latency_ms']))
        out['variants'][var] = v
    out['outcomes'] = outcomes(out, act, ver, fab, D)
    (R3 / 'energy.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    with open(R3 / 'energy.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: r[k] for k in ('variant', 'design', 'p_idle_W', 'p_awake_W', 'p_avg_W', 'saving', 'day_report_W',
                                 'p_avg_tr6_W', 'p_avg_tr25_W', 'worst_compute_ms', 'latency_ms')})
    print(json.dumps(out['outcomes'], indent=1))


def outcomes(out, act, ver, fab, D):
    V = out['variants']
    p = lambda var, c: V[var]['designs'][c]['12.5']['mean']['p_avg_W']
    idle = lambda var: V[var]['states']['idle']['12.5']['p_pl']
    awake = lambda var: V[var]['states']['busy']['12.5']['p_pl']
    o = {}
    a8, a1 = V['asbuilt_rw8'], V['asbuilt_rw1']
    o['H1'] = dict(c_v=a8['c_v_cycles'], worst_compute_ms=a8['worst_compute_ms'], in_range=19e6 <= a8['c_v_cycles'] <= 25e6,
                   fails_200ms=a8['worst_compute_ms'] > 200, below_hop=a8['worst_compute_ms'] < 250)
    o['H1_rw1'] = dict(c_v=a1['c_v_cycles'], worst_compute_ms=a1['worst_compute_ms'],
                       in_range=10.3e6 <= a1['c_v_cycles'] <= 11.2e6, meets_200ms=a1['worst_compute_ms'] <= 200)
    o['H3'] = {var: dict(p_idle_W=idle(var), in_range=0.085 <= idle(var) <= 0.105) for var in V}
    for var in V:
        b1, b2 = p(var, 'B1'), p(var, 'B2')
        sav = [V[var]['designs'][c]['12.5']['mean']['saving'] for c in ('B1', 'B2')]
        o[f'H4_{var}'] = dict(B1_W=b1, B2_W=b2, savings=sav, confirmed=b1 <= 0.150 and b2 <= 0.130, refuted=min(sav) < 0.5)
    for var in ('asbuilt_rw8', 'asbuilt_rw1'):
        gate = out['p_busy_W'] - idle(var)
        o[f'H5_{var}'] = dict(gate_only_saving_W=rnd(gate), gate_only_share=rnd(gate / out['p_busy_W']),
                              first_part_confirmed=gate <= 0.03,
                              read_enables_effective=False,
                              note='the read enables are not in the built netlist (results/R3/vivado/bram_enable/); '
                                   'second part (read enables give >= 85 % of the saving) not met')
    for key, var, lim, asleep in (('H6', 'asbuilt_rw8', (40e3, 80e3), {'B1': .79, 'B2': .86}),
                                  ('H6_rw1', 'asbuilt_rw1', (15e3, 40e3), {'B1': .90, 'B2': .93})):
        sl = {c: rnd(1 - V[var]['designs'][c]['12.5']['mean']['awake_share'], 4) for c in ('B1', 'B2')}
        o[key] = dict(awake_cycles=V[var]['a_s1_cycles'], awake_in_range=lim[0] <= V[var]['a_s1_cycles'] <= lim[1],
                      asleep_share=sl, asleep_ok=all(sl[c] >= asleep[c] for c in sl))
    o['H10'] = dict(rw1_minus_rw8_W={c: rnd(p('asbuilt_rw1', c) - p('asbuilt_rw8', c), 5) for c in ('B1', 'B2')},
                    in_range={c: 0.001 <= p('asbuilt_rw8', c) - p('asbuilt_rw1', c) <= 0.005 for c in ('B1', 'B2')})
    s1 = ver['sets']['rw1']['stage1_cycles']
    o['S1_rw1'] = dict(stage1_cycles=s1, in_range=11e3 <= s1['mean'] <= 18e3)
    if 'decomp_rw1' in V:
        dv = V['decomp_rw1']
        bram = lambda var, s: V[var]['components'][s]['bram']
        e = fab['builds'].get('dec', {})
        rw1 = fab['builds'].get('rw1', {})
        o['H9'] = dict(bram_awake_W=dict(asbuilt=bram('asbuilt_rw1', 'busy'), decomp=bram('decomp_rw1', 'busy')),
                       bram_awake_reduction=rnd(1 - bram('decomp_rw1', 'busy') / bram('asbuilt_rw1', 'busy')),
                       confirmed=bram('decomp_rw1', 'busy') <= 0.2 * bram('asbuilt_rw1', 'busy'))
        dl = dict(luts=e.get('luts', 0) - rw1.get('luts', 0), registers=e.get('registers', 0) - rw1.get('registers', 0))
        o['H9_cost'] = dict(delta_vs_asbuilt_rw1=dl, wns_ns=e.get('wns_ns'), bram36=e.get('bram36'),
                            confirmed=1000 <= dl['luts'] <= 2200 and 0 <= dl['registers'] <= 150 and
                            e.get('bram36') == 64 and (e.get('wns_ns') or -1) >= 0)
        o['H9_idle'] = dict(p_idle_W=idle('decomp_rw1'), in_range=0.10 <= idle('decomp_rw1') <= 0.14)
        o['H9_designs'] = {c: dict(p_avg_W=p('decomp_rw1', c), saving=dv['designs'][c]['12.5']['mean']['saving'],
                                   in_range=0.10 <= p('decomp_rw1', c) <= 0.15) for c in ('B1', 'B2')}
        o['H9_layout_alone'] = dict(p_awake_W=awake('decomp_rw1'), below_0_17=awake('decomp_rw1') < 0.17)
    main = 'decomp_rw1' if 'decomp_rw1' in V else 'asbuilt_rw1'
    o['requirements_main'] = {c: dict(variant=main, live=D[c]['live'], fa_h=D[c]['fa_h'], other=D[c]['other'],
                                      latency_ms=V[main]['designs'][c]['latency_ms'],
                                      worst_compute_ms=V[main]['worst_compute_ms'], memory_kib=D[c]['memory_kib'],
                                      p_avg_W=p(main, c),
                                      all_met=D[c]['live'] >= .85 and D[c]['fa_h'] <= 2 and D[c]['other'] <= .002 and
                                      V[main]['designs'][c]['latency_ms'] <= 500 and V[main]['worst_compute_ms'] <= 200 and
                                      D[c]['memory_kib'] <= 144 and p(main, c) <= 1)
                              for c in ('B1', 'B2')}
    return o


if __name__ == '__main__':
    main()
