"""R3 pre-registration (research/PLAN_R3_GATING.md section 2): the hypotheses H1-H9 with their derivation.

Run before any R3 measurement; writes research/results/R3/phase1/hypothesis.json.

Inputs (all from earlier experiments, nothing measured in R3):
  results/R2/designs.csv                    B1 = cascade_B, B2 = cascade_B_calls2150: calls/h (mean, sd),
                                            R2 clock-gated power, latency, worst compute
  results/R2/B/v1/h48_k64_seed2.json        representative B1 configuration (cascade.t1, t2, calls/h)
  results/R2/B/v1/budget2150/h48_k64_seed2.json   representative B2 configuration
  results/R1/models.csv                     stage 1 (h48_k64 seed 2, ksdot + LUT): board cycles, t_max
  results/S6/power.json (energy_model.load_power()['ksdot'])   Vivado power terms of the ksdot fabric
  results/S6/fabric.json, results/S6/board/h48_k64_seed2.json  utilization/timing and the cycle anchor

Energy model (section 4, 2E of the plan), f = 100 MHz, 14,400 requests per hour:
  E_h = 3600 P_idle + 14,400 (P_S1 - P_idle) C_S1 / f + N_V (P_V - P_idle) C_V / f,   P_avg = E_h / 3600
  P_idle = P_PL,stat + P_infra,dyn (clock stopped, BRAM ports not enabled).
Three predictions for the awake increments P - P_idle:
  (a) R2 model: P_core,dyn while awake, T_V = 105.83 ms (the R2 numbers, unchanged);
  (b) R3 central prior: T_V = 21.93 M cycles (READ_WAIT = 8 estimate), increment 0.079 W (plan prior);
  (c) pessimistic: T_V = 21.93 M cycles, increment P_core,dyn = 0.273 W (the whole core while awake).
"""
import csv
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from energy_model import F_CLK, load_power  # noqa: E402

RES = TOOLS.parent / 'results'
OUT = RES / 'R3' / 'phase1' / 'hypothesis.json'
REQUESTS_H = 3600 / 0.25
DESIGNS = {'B1': ('cascade_B', RES / 'R2/B/v1/h48_k64_seed2.json'),
           'B2': ('cascade_B_calls2150', RES / 'R2/B/v1/budget2150/h48_k64_seed2.json')}
STAGE1 = 'h48_k64_seed2'                      # R1 models.csv name, hardware ksdot+LUT
P_BUSY = 0.367                                # W, busy-polling reference (S6 report, unchanged)

# Verifier time. R2 used the teammate SoC (READ_WAIT = 1, plain kdot, Verilator timer).
T_V_R2_MAX, T_V_R2_MEAN = 107.3e-3, 105.83e-3
READ_WAIT_OURS, READ_WAIT_TEAM = 8, 1
CPU_READS_V = 1.6e6                           # plan estimate: CPU bus reads (fetches + loads) per verifier call
DP_V_CENTRAL = 0.079                          # W, plan prior for the awake increment during the verifier

# Verifier shape (results/R2/verifier/sheila_head_r2_s1.pt config) for the cross-check of the prior.
H1, H2, STACK, CLASSES, FRAMES = 64, 64, 2, 40, 150


def rnd(x, n=4):
    return None if x is None else round(float(x), n)


def stage1_row():
    for r in csv.DictReader(open(RES / 'R1/models.csv', encoding='utf-8')):
        if r['name'] == STAGE1 and r['hardware'] == 'ksdot+LUT':
            return r
    raise SystemExit(f'{STAGE1} ksdot+LUT not in R1 models.csv')


def design_rows():
    rows = {r['design']: r for r in csv.DictReader(open(RES / 'R2/designs.csv', encoding='utf-8'))}
    out = {}
    for code, (name, rep) in DESIGNS.items():
        r, c = rows[name], json.loads(Path(rep).read_text(encoding='utf-8'))['cascade']
        out[code] = dict(design=name, calls_h=float(r['calls_per_hour']), calls_h_sd=float(r['calls_per_hour_sd']),
                         busy_share_r2=float(r['verifier_busy_share']), p_gated_r2=float(r['p_avg_gated_W']),
                         latency_ms_r2=float(r['latency_ms']), t_max_ms_r2=float(r['t_max_ms']),
                         memory_kib=float(r['memory_kib']), live_recall=float(r['live_recall']),
                         rep=dict(file=Path(rep).relative_to(TOOLS.parent).as_posix(), t1=int(c['t1']), t2=int(c['t2']),
                                  calls_h=float(c['verifier_calls_per_hour']), live_recall=float(c['live_recall']),
                                  latency_median_s=float(c['latency_median_s'])))
    return out


def power_terms():
    p = load_power()['ksdot']
    rep = json.loads((RES / 'S6/power.json').read_text())['ksdot']['power']
    hier = {r['Name']: float(r['Power (W)']) for r in rep['hierarchy']}
    chip = {r['On-Chip']: r['Power (W)'] for r in rep['on_chip']}
    num = lambda v: 0.0 if isinstance(v, str) else float(v)     # '<0.001' rows count as 0
    return dict(pl_dyn=p['pl_dyn'], pl_static=p['pl_static'], core_dyn=p['core_dyn'], infra_dyn=p['infra_dyn'],
                bram=num(chip['Block RAM']), clocks=num(chip['Clocks']), signals=num(chip['Signals']),
                slice_logic=num(chip['Slice Logic']), dsp=num(chip['DSPs']),
                u_cpu=hier['u_cpu'], u_kdot=hier['u_kdot'], u_poisson=hier['u_poisson'])


def p_avg(p_idle, calls, dp_s1, c_s1, dp_v, c_v):
    e_h = 3600 * p_idle + REQUESTS_H * dp_s1 * c_s1 / F_CLK + calls * dp_v * c_v / F_CLK
    return e_h / 3600, e_h


def verifier_kdot_cycles():
    """kdot cycles of one verifier call (3 cycles per 4 MACs) and the number of kdot instructions."""
    per_step = 3 * H1 * (24 * STACK + H1) + 3 * H2 * (H1 + H2) + CLASSES * H2 + H2   # two GRU layers, logits, head
    steps = FRAMES // STACK
    calls = steps * (4 * H1 + 4 * H2 + CLASSES + 1)
    return dict(macs=per_step * steps, kdot_cycles=per_step * steps * 3 // 4, kdot_instructions=calls)


def main():
    P = power_terms()
    D = design_rows()
    s1 = stage1_row()
    fabric = json.loads((RES / 'S6/fabric.json').read_text())
    board = json.loads((RES / 'S6/board/h48_k64_seed2.json').read_text())
    anchor = board['variants']['ksdot_lut']['single_window']
    p_idle = P['pl_static'] + P['infra_dyn']

    # H1: verifier time at READ_WAIT = 8 = R2 time + 7 extra cycles per CPU read.
    c_v_rw1 = T_V_R2_MAX * F_CLK
    c_v_rw8 = c_v_rw1 + CPU_READS_V * (READ_WAIT_OURS - READ_WAIT_TEAM)
    t_s1_max = float(s1['t_max_ms'])
    h1 = dict(prediction='T_V about 21.9 M cycles (19-25 M) = 190-250 ms at READ_WAIT 8; 10.3-11.2 M at READ_WAIT 1; '
                         'worst compute about 219 ms > 200 ms (predicted fail), < 250 ms',
              derivation=dict(c_v_read_wait1=c_v_rw1, cpu_reads_per_call_estimate=CPU_READS_V,
                              extra_cycles_per_read=READ_WAIT_OURS - READ_WAIT_TEAM, c_v_read_wait8=c_v_rw8,
                              t_v_read_wait8_ms=rnd(1000 * c_v_rw8 / F_CLK, 2), stage1_t_max_ms=t_s1_max,
                              worst_compute_ms=rnd(1000 * c_v_rw8 / F_CLK + t_s1_max, 2)),
              accept=dict(c_v_read_wait8=[19e6, 25e6], c_v_read_wait1=[10.3e6, 11.2e6]),
              requirement_ms=200, hop_ms=250, test='MB[21] in xsim (opcode 6)')

    h2 = dict(prediction='100 % bit-exact on both fabrics; the F2 anchor equals the board cycles; compute cycles '
                         'identical with POWER_OPT 0 and 1',
              anchor=dict(image='keyword_ksdot_lut.bin', sha256=board['firmware']['images']['keyword_ksdot_lut']['sha256'],
                          cycles_min=anchor['cycles_min'], cycles_mean=anchor['cycles_mean'],
                          cycles_max=anchor['cycles_max'], vectors=40, repeats=3),
              test='xsim')

    h3 = dict(prediction='P_idle (gated clock + read enables) about 0.094 W', accept_W=[0.085, 0.105],
              derivation=dict(p_pl_static=P['pl_static'], p_infra_dyn=P['infra_dyn'], p_idle=rnd(p_idle)),
              test='Vivado state idle')

    # H4: three predictions of the average PL power.
    c_s1_awake = 60e3                         # awake cycles per request without a verifier call (H6 midpoint)
    k = verifier_kdot_cycles()
    duty_b = (k['kdot_cycles'] + CPU_READS_V) / c_v_rw8   # port-B enabled cycles: kdot streaming + one per CPU read
    cross = P['clocks'] + P['signals'] + P['slice_logic'] + P['dsp'] + P['bram'] * duty_b
    h4 = dict(prediction='(a) R2 model B1 0.119 W (-67.6 %, 428 J/h), B2 0.110 W (-70.1 %, 395 J/h); '
                         '(b) R3 central prior B1 0.109 W (-70 %), B2 0.103 W (-72 %); '
                         '(c) pessimistic B1 0.145 W, B2 0.126 W',
              rule='confirmed if B1 <= 0.150 W and B2 <= 0.130 W; refuted if either saving < 50 %',
              confirm_W=dict(B1=0.150, B2=0.130), refute_saving_below=0.50, busy_W=P_BUSY,
              central_prior=dict(dp_v_W=DP_V_CENTRAL, source='plan prior (PLAN_R3_GATING.md 1.6/2), not a fit',
                                 cross_check=dict(note='clocks + signals + slice logic + DSP of the S6 report, plus BRAM '
                                                       'x port-B duty (kdot cycles + one enabled cycle per CPU read)',
                                                  kdot=k, port_b_duty=rnd(duty_b), dp_v_W=rnd(cross)),
                                 c_s1_awake_cycles=c_s1_awake, c_v_cycles=c_v_rw8),
              designs={})
    for code, d in D.items():
        out = {}
        for case, (dp_s1, dp_v, c_v) in dict(a=(P['core_dyn'], P['core_dyn'], T_V_R2_MEAN * F_CLK),
                                             b=(DP_V_CENTRAL, DP_V_CENTRAL, c_v_rw8),
                                             c=(P['core_dyn'], P['core_dyn'], c_v_rw8)).items():
            c_s1 = float(s1['cycles_mean']) if case == 'a' else c_s1_awake
            pa, eh = p_avg(p_idle, d['calls_h'], dp_s1, c_s1, dp_v, c_v)
            out[case] = dict(p_avg_W=rnd(pa), saving=rnd(1 - pa / P_BUSY), e_h_J=rnd(eh, 1),
                             awake_share=rnd((REQUESTS_H * c_s1 + d['calls_h'] * c_v) / F_CLK / 3600))
        out['a']['r2_designs_csv_p_avg_gated_W'] = rnd(d['p_gated_r2'])
        out['calls_h'], out['calls_h_sd'] = d['calls_h'], d['calls_h_sd']
        h4['designs'][code] = out

    h5 = dict(prediction='gating the CPU clock alone saves <= 0.03 W (<= 8 %); the read enables give >= 85 % of the saving',
              derivation=dict(gated_instances_W=rnd(P['u_cpu'] + P['u_kdot'] + P['u_poisson']),
                              bound_W=0.03, bound_share_of_busy=rnd(0.03 / P_BUSY),
                              total_saving_W=rnd(P_BUSY - p_idle),
                              read_enable_share_at_bound=rnd((P_BUSY - p_idle - 0.03) / (P_BUSY - p_idle))),
              test='Vivado idle_gate_only vs idle')

    asleep = {}
    for code, d in D.items():
        for label, t_v, c_s1 in (('bound', 0.250, 80e3), ('central', c_v_rw8 / F_CLK, c_s1_awake)):
            asleep.setdefault(code, {})[label] = rnd(1 - (REQUESTS_H * c_s1 / F_CLK + d['calls_h'] * t_v) / 3600)
    h6 = dict(prediction='awake cycles per request without a verifier call 40-80 k; CPU asleep >= 79 % (B1), >= 86 % (B2)',
              accept=dict(awake_cycles=[40e3, 80e3], asleep_min=dict(B1=0.79, B2=0.86)),
              derivation=dict(note='bound: T_V 250 ms (top of H1) and 80 k cycles per request, mean calls/h; '
                                   'B1 0.785 is rounded to 79 % in the plan', asleep=asleep,
                              stage1_cycles_mean=float(s1['cycles_mean'])),
              test='xsim counters')

    h7 = dict(prediction='cost <= +150 LUT, <= +120 FF, BUFGCTRL 1 -> 2, 64 BRAM, 8 DSP, WNS >= 0',
              baseline_S6=dict(luts=fabric['luts'], registers=fabric['registers'], bram36=fabric['bram36'],
                               dsp=fabric['dsp'], bufgctrl=1, wns_ns=fabric['wns_ns']),
              limits=dict(luts=fabric['luts'] + 150, registers=fabric['registers'] + 120, bufgctrl=2,
                          bram36=64, dsp=8, wns_ns_min=0.0),
              test='Vivado reports')
    h8 = dict(prediction='POWER_OPT = 0 is equivalent to r3-base; the release firmware runs bit-exact with identical '
                         'cycles on the R3 fabric; --compare deploy identical; POWER_OPT = 0 utilization equal to S6',
              test='tb_r3_equiv.sv, release regression, Vivado POWER_OPT = 0 build')
    h9 = dict(prediction='only if the ram_decomp variant is built: verifier-state BRAM power -80 % or more; '
                         'about +1,000 LUT', test='Vivado')

    latency = {code: dict(r2_latency_ms=rnd(d['latency_ms_r2'], 1),
                          r3_latency_ms=rnd(d['latency_ms_r2'] - 1000 * T_V_R2_MAX + 1000 * c_v_rw8 / F_CLK, 1),
                          r3_latency_rep_ms=rnd(1000 * d['rep']['latency_median_s'] + t_s1_max + 1.9
                                                - 1000 * T_V_R2_MAX + 1000 * c_v_rw8 / F_CLK, 1))
               for code, d in D.items()}

    out = dict(
        experiment='R3', plan='research/PLAN_R3_GATING.md', written_before_any_r3_measurement=True,
        base_tag='r3-base',
        inputs=dict(designs_csv='results/R2/designs.csv', r1_models='results/R1/models.csv',
                    power='results/S6/power.json (energy_model.load_power()["ksdot"])',
                    fabric='results/S6/fabric.json', anchor='results/S6/board/h48_k64_seed2.json'),
        power_terms_W={k: rnd(v) for k, v in P.items()}, p_idle_W=rnd(p_idle), p_busy_W=P_BUSY,
        designs={code: {k: v for k, v in d.items()} for code, d in D.items()},
        stage1=dict(model='results/S3/models/h48_k64_seed2.npz', cycles_mean=float(s1['cycles_mean']),
                    t_max_ms=t_s1_max),
        model='E_h = 3600 P_idle + 14400 (P_S1 - P_idle) C_S1/f + N_V (P_V - P_idle) C_V/f; P_avg = E_h/3600; '
              'saving = 1 - P_avg/0.367',
        hypotheses=dict(H1=h1, H2=h2, H3=h3, H4=h4, H5=h5, H6=h6, H7=h7, H8=h8, H9=h9),
        latency=dict(prediction='word end to decision about 275-290 ms, passes 500 ms',
                     formula='R2 latency - 107.3 ms + T_V,max', designs=latency))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    summary = {code: {c: v['p_avg_W'] for c, v in h4['designs'][code].items() if c in 'abc'} for code in D}
    print(json.dumps(dict(t_v_rw8_ms=h1['derivation']['t_v_read_wait8_ms'], p_idle=rnd(p_idle), p_avg=summary,
                          asleep=asleep, cross_check_dp_v=rnd(cross), latency=latency), indent=1))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
