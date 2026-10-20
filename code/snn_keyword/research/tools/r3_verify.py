"""R3 Phase 2C: collect the xsim verification (research/PLAN_R3_GATING.md 2C) and the measured activity (2E inputs).

    r3_verify.py collect   -> results/R3/phase2/verification.json, raw xsim output in results/R3/phase2/sim/
    r3_verify.py duties    -> results/R3/phase2/activity.json and duties_rw<8|1>.tcl (for r3_power_states.tcl)

Runs (code/snn_keyword/build/r3_sim/phase2, made by r3_sim.py run):
  base_<set>   unchanged RTL (r3-base), READ_WAIT 8: the reference outputs and cycles (2A)
  eq_<set>     equivalence testbench: new RTL with POWER_OPT 0 against the r3-base RTL, every clock (2C.1)
  p1rw8_<set>  POWER_OPT 1, READ_WAIT 8 (gating alone)          p1rw1_<set>  POWER_OPT 1, READ_WAIT 1 (main design)
  p0rw1_<set>  POWER_OPT 0, READ_WAIT 1 (cycle reference for p1rw1)
  sleep_p1rw<k> sleep checks with firmware/r3_sleep_test.c (2C.3)
  aw_p1rw<k>_<set> the same sets again with the testbench's awake counter (gated clock edges per request)
Sets: v1 (opcode 1 x 40), v2_<i> (opcode 6), v3a, v3b, v4b1, v4b2 (opcode 7 cascades), v5s1 (stage-1-only start
of v3b), rel (the release firmware deploy/keyword_kdot.bin).
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import r3_sim as S  # noqa: E402

SIM = S.KWS / 'build/r3_sim/phase2'
OUT = S.RES / 'R3/phase2'
SETS = ['v1', 'v2_0', 'v2_1', 'v2_2', 'v3a', 'v3b', 'v4b1', 'v4b2', 'v5s1', 'rel']
CASCADE = ['v3a', 'v3b', 'v4b1', 'v4b2']
F = 100e6


def checked(run, s):
    log = SIM / f'{run}_{s}.log'
    return S.check(log, SIM / f'{s}.expect.json') if log.exists() else None


def cycles(c):
    """Per request: the firmware-timed cycles (MB[9] and, for command 7, MB[20..22])."""
    return [{k: v for k, v in r.items() if k.startswith('mb')} for r in c['per_request']]


def compare_cycles(a, b, n=None):
    ca, cb = cycles(a)[:n], cycles(b)[:n]
    diff = [i for i, (x, y) in enumerate(zip(ca, cb)) if x != y]
    return dict(requests=len(ca), identical=not diff and len(ca) == len(cb[:len(ca)]), differing=diff[:10])


def regressions():
    kdot = subprocess.run(['bash', str(S.FLOW / 'sim/run_tb_kdot.sh')], capture_output=True, text=True)
    rel = subprocess.run([sys.executable, str(TOOLS / 'build_firmware.py'),
                          str(S.RES / 'S2/models/h48_b8_seed1.npz'), '--out', str(S.KWS / 'build/r3_release_check'),
                          '--images', 'keyword_lut', 'keyword_kdot_lut', '--compare', str(S.KWS / 'deploy')],
                         capture_output=True, text=True)
    deploy = subprocess.run(['git', 'diff', '--stat', 'r3-base', '--', 'code/snn_keyword/deploy'], cwd=S.REPO,
                            capture_output=True, text=True).stdout.strip()
    ident = json.loads(rel.stdout).get('identical_to') if rel.returncode == 0 else None
    return dict(run_tb_kdot=dict(passed=kdot.returncode == 0, last=kdot.stdout.strip().splitlines()[-1:]),
                release_compare_deploy=dict(passed=bool(ident) and all(ident.values()), identical_to=ident),
                deploy_unchanged_since_r3_base=deploy == '')


def collect():
    rep = dict(plan='research/PLAN_R3_GATING.md 2C', runs=str(SIM.relative_to(S.REPO).as_posix()), sets={}, checks={})
    base = {s: checked('base', s) for s in SETS}
    base['v5s1'] = checked('base', 'v3b')          # v5s1 = the first requests of v3b
    p1 = {rw: {s: checked(f'p1rw{rw}', s) for s in SETS} for rw in (8, 1)}
    p0rw1 = {s: checked('p0rw1', s) for s in ('v1', 'v2_0', 'v5s1', 'rel')}
    eq = {s: checked('eq', s) for s in ('rel', 'v5s1')}
    # 2A: every set bit-exact on the unchanged RTL
    rep['checks']['base_bit_exact'] = {s: c['bit_exact'] for s, c in base.items() if c and s != 'v5s1'}
    # 2C.1 equivalence (POWER_OPT 0 against r3-base, every clock)
    rep['checks']['equivalence'] = {s: dict(bit_exact=c['bit_exact'], mismatches=c['tb'].get('equiv', {}).get('mismatches'),
                                            cycles=c['tb'].get('equiv', {}).get('cycles'), passed=c['passed'])
                                    for s, c in eq.items()}
    # 2C.2 POWER_OPT 1: bit-exact, no testbench errors, compute cycles identical to POWER_OPT 0
    for rw, runs in p1.items():
        ref = {s: base[s] for s in SETS} if rw == 8 else p0rw1
        if rw == 8:
            ref['rel'] = eq['rel']
        rep['checks'][f'power_opt1_rw{rw}'] = {
            s: dict(bit_exact=c['bit_exact'], tb_errors=c['tb_errors'], passed=c['passed'],
                    cycles_vs_power_opt0=(compare_cycles(c, ref[s], len(c['per_request'])) if ref.get(s) else None))
            for s, c in runs.items() if c}
    rep['checks']['power_opt0_rw1'] = {s: dict(bit_exact=c['bit_exact'], passed=c['passed']) for s, c in p0rw1.items() if c}
    # 2C.3 sleep checks
    rep['checks']['sleep'] = {rw: S.sleep_check(SIM / f'sleep_p1rw{rw}.log') for rw in (8, 1)
                              if (SIM / f'sleep_p1rw{rw}.log').exists()}
    # 2C.4 regressions (release on POWER_OPT 1 = cycles equal to POWER_OPT 0, already in power_opt1_rw8['rel'])
    rep['checks']['regressions'] = regressions()
    rel_board = json.loads((SIM / 'rel.expect.json').read_text())['anchor']
    if p1[8].get('rel'):
        cy = p1[8]['rel']['cycles']
        rep['checks']['regressions']['release_on_power_opt1_rw8'] = dict(
            bit_exact=p1[8]['rel']['bit_exact'], cycles=dict(min=cy['min'], mean=cy['mean'], max=cy['max']),
            board=dict(min=rel_board['cycles_min'], mean=rel_board['cycles_mean'], max=rel_board['cycles_max']),
            equal_to_board=(cy['min'], round(cy['mean'], 1), cy['max']) ==
            (rel_board['cycles_min'], round(rel_board['cycles_mean'], 1), rel_board['cycles_max']))
    # cycle summary per READ_WAIT
    for rw, runs in p1.items():
        req = [r for s in CASCADE if runs.get(s) for r in runs[s]['per_request'] if r['op'] == 7]
        s1 = [r['mb20'] for r in req if r['window'] is not None]
        tv = [r['mb21'] for r in req if r['bits'] & 4]
        op6 = [runs[f'v2_{i}']['cycles']['all'][0] for i in range(3) if runs.get(f'v2_{i}')]
        rep['sets'][f'rw{rw}'] = dict(
            stage1_cycles=dict(min=min(s1), mean=round(float(np.mean(s1)), 1), max=max(s1), n=len(s1)),
            push_cycles=dict(min=min(r['mb22'] for r in req), max=max(r['mb22'] for r in req)),
            verifier_cycles_cmd7=dict(min=min(tv), mean=round(float(np.mean(tv)), 1), max=max(tv), n=len(tv)),
            verifier_cycles_cmd6=op6, worst_request_cycles=max(r['mb9'] for r in req),
            worst_compute_ms=round(max(r['mb9'] for r in req) / 1e5, 3),
            v1_cycles=runs['v1']['cycles'] if runs.get('v1') else None)
    flat = []
    for k, v in rep['checks'].items():
        if k in ('equivalence', 'power_opt1_rw8', 'power_opt1_rw1', 'power_opt0_rw1'):
            for s, x in v.items():
                ok = x.get('passed') and (x.get('cycles_vs_power_opt0') is None or x['cycles_vs_power_opt0']['identical'])
                if k == 'equivalence':
                    ok = ok and x['mismatches'] == 0
                flat.append((f'{k}.{s}', bool(ok)))
        elif k == 'sleep':
            flat += [(f'sleep.rw{rw}', x['passed']) for rw, x in v.items()]
        elif k == 'base_bit_exact':
            flat += [(f'base.{s}', ok) for s, ok in v.items()]
    reg = rep['checks']['regressions']
    flat += [('run_tb_kdot', reg['run_tb_kdot']['passed']), ('release_compare', reg['release_compare_deploy']['passed']),
             ('deploy_unchanged', reg['deploy_unchanged_since_r3_base']),
             ('release_power_opt1_board_cycles', reg.get('release_on_power_opt1_rw8', {}).get('equal_to_board', False))]
    rep['failed'] = [k for k, ok in flat if not ok]
    rep['passed'] = not rep['failed'] and len(flat) > 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'sim').mkdir(exist_ok=True)
    for log in SIM.glob('*.log'):
        shutil.copy(log, OUT / 'sim' / (log.stem + '.xsim.txt'))
    for js in SIM.glob('*.run.json'):
        shutil.copy(js, OUT / 'sim' / js.name)
    (OUT / 'verification.json').write_text(json.dumps(rep, indent=2) + '\n', encoding='utf-8')
    return rep


# --------------------------------------------------------------------------- activity and duties (2E inputs)

def activity(rw):
    """Measured per-request activity of the gated design at READ_WAIT rw (POWER_OPT 1) from the aw_ runs
    (awake = gated clock edges from the MB[0] write to the next sleep) and the p1 runs (verifier cycles)."""
    aw = [r for s in CASCADE if (c := checked(f'aw_p1rw{rw}', s)) for r in c['per_request'] if r['op'] == 7]
    p1 = [r for s in CASCADE if (c := checked(f'p1rw{rw}', s)) for r in c['per_request'] if r['op'] == 7]
    s1 = [r for r in aw if r['window'] is not None and not r['bits'] & 4]
    vr = [r for r in aw if r['bits'] & 4]
    pb = lambda rs: sum(r['pb_re'] + r['pb_we'] for r in rs)
    tv = [r['mb21'] for r in p1 if r['bits'] & 4]
    out = dict(read_wait=rw, stage1_requests=len(s1), verifier_requests=len(vr),
               awake_s1_cycles=round(float(np.mean([r['awake'] for r in s1])), 1),
               awake_s1_max=max(r['awake'] for r in s1),
               svc_s1_cycles=round(float(np.mean([r['svc'] for r in s1])), 1),
               pb_en_s1=pb(s1) / sum(r['svc'] for r in s1), pb_we_s1=sum(r['pb_we'] for r in s1) / sum(r['svc'] for r in s1),
               pb_s1_per_request=pb(s1) / len(s1),
               verifier_cycles=dict(mean=float(np.mean(tv)), max=max(tv), min=min(tv), n=len(tv)),
               awake_verifier_request=round(float(np.mean([r['awake'] for r in vr])), 1),
               pb_en_v=pb(vr) / sum(r['svc'] for r in vr), pb_we_v=sum(r['pb_we'] for r in vr) / sum(r['svc'] for r in vr),
               payload_words_s1=float(np.mean([(r['len'] + 3) // 4 for r in s1])))
    return out


def duties():
    import r3_oracle  # noqa: F401  (thresholds only)
    designs = {}
    import csv
    for r in csv.DictReader(open(S.RES / 'R2/designs.csv', encoding='utf-8')):
        if r['design'] in ('cascade_B', 'cascade_B_calls2150'):
            designs['B1' if r['design'] == 'cascade_B' else 'B2'] = float(r['calls_per_hour'])
    rep = {}
    for rw in (8, 1):
        a = activity(rw)
        req_h = 14400
        states = [('busy', 1.0, 1.0, a['pb_we_s1'], 1.0, 0.0),        # as built: clock always on, ports always read
                  ('idle', 1e-9, 0.0, 0.0, 0.0, 0.0),
                  ('idle_gate_only', 1e-9, 1.0, 0.0, 1.0, 0.0),       # clock gated, no read enables
                  ('stage1', 1.0, a['pb_en_s1'], a['pb_we_s1'], 0.0, 0.0),
                  ('verifier', 1.0, a['pb_en_v'], a['pb_we_v'], 0.0, 0.0)]
        days = {}
        for code, calls in designs.items():
            c_v = a['verifier_cycles']['mean']
            awake = (req_h * a['awake_s1_cycles'] + calls * c_v) / (3600 * F)
            pb_en = (req_h * a['awake_s1_cycles'] * a['pb_en_s1'] + calls * c_v * a['pb_en_v']) / (3600 * F)
            pb_we = (req_h * a['awake_s1_cycles'] * a['pb_we_s1'] + calls * c_v * a['pb_we_v']) / (3600 * F)
            # PS port A on the board: payload + 3 request words written, 30 result words + about one MB[1] poll per
            # ms read per request (board_server.py polls every 1 ms)
            polls = req_h * (1 + a['awake_s1_cycles'] / 1e5) + calls * c_v / 1e5
            pa_we = req_h * (a['payload_words_s1'] + 3) / (3600 * F)
            pa_en = pa_we + (req_h * 30 + polls) / (3600 * F)
            days[code] = dict(calls_per_hour=calls, awake_share=awake, pb_en=pb_en, pb_we=pb_we, pa_en=pa_en, pa_we=pa_we)
            states.append((f'day_{code}', awake, pb_en, pb_we, pa_en, pa_we))
        rep[f'rw{rw}'] = dict(activity=a, days=days, states={s[0]: s[1:] for s in states})
        lines = ['# R3 power states (r3_verify.py duties): name cpu_clock_duty pb_en pb_we pa_en pa_we',
                 f'# READ_WAIT {rw}, POWER_OPT 1; measured in xsim (results/R3/phase2/activity.json)', 'set states {']
        lines += [f'  {n} ' + ' '.join(f'{x:.9g}' for x in v) for n, *v in states]
        lines.append('}')
        (OUT / f'duties_rw{rw}.tcl').write_text('\n'.join(lines) + '\n')
    (OUT / 'activity.json').write_text(json.dumps(rep, indent=2) + '\n')
    return rep


# --------------------------------------------------------------------------- fabric (2D)

BUILDS = {'rw1': dict(power_opt=1, read_wait=1, bit='r3_gated_rw1.bit'),     # main design
          'rw8': dict(power_opt=1, read_wait=8, bit='r3_gated_rw8.bit'),     # gating alone
          'p0': dict(power_opt=0, read_wait=8, bit=None)}                    # regression: must equal S6


def util(path):
    text = Path(path).read_text()
    def row(name):
        import re
        m = re.search(rf'^\|\s*{re.escape(name)}\s*\|\s*([\d.]+)\s*\|', text, re.M)
        return float(m.group(1)) if m else None
    return dict(luts=row('Slice LUTs'), lutram=row('LUT as Distributed RAM'), registers=row('Slice Registers'),
                bram36=row('Block RAM Tile'), dsp=row('DSPs'), bufgctrl=row('BUFGCTRL'))


def wns(path):
    lines = Path(path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if 'Design Timing Summary' in l)
    vals = next(l for l in lines[i + 3:] if l.strip() and not l.strip().startswith('-') and 'WNS' not in l).split()
    return dict(wns_ns=float(vals[0]), tns_ns=float(vals[1]), whs_ns=float(vals[4]))


def fabric():
    import hashlib
    s6 = json.loads((S.RES / 'S6/fabric.json').read_text())
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=S.REPO, capture_output=True, text=True).stdout.strip()
    rtl = {}
    for f in sorted((S.FLOW / 'rtl').glob('*.v')):
        blob = subprocess.run(['git', 'show', f'HEAD:code/pynqz2_riscv_flow/rtl/{f.name}'], cwd=S.REPO,
                              capture_output=True).stdout.replace(b'\r\n', b'\n')
        rtl[f'code/pynqz2_riscv_flow/rtl/{f.name}'] = hashlib.sha256(blob).hexdigest()
    rep = dict(source_commit=commit, rtl_sha256_lf_normalized=rtl, vivado='2026.1', builds={},
               s6=dict(luts=s6['luts'], lutram=s6['lutram'], registers=s6['registers'], bram36=s6['bram36'],
                       dsp=s6['dsp'], bufgctrl=1, wns_ns=s6['wns_ns']))
    for name, b in BUILDS.items():
        d = S.RES / f'R3/vivado/{name}'
        if not (d / 'utilization.rpt').exists():
            continue
        e = dict(generics=dict(POWER_OPT=b['power_opt'], READ_WAIT=b['read_wait']),
                 abi='0x00020007' if b['power_opt'] else '0x00020003', **util(d / 'utilization.rpt'),
                 **wns(d / 'timing_summary.rpt'))
        cb = (d / 'clock_buffers.txt').read_text().splitlines()
        e['clock_buffers'] = cb
        if b['bit']:
            bit = S.RES / 'R3' / b['bit']
            e['bitstream'] = f'research/results/R3/{b["bit"]}'
            e['bitstream_sha256'] = hashlib.sha256(bit.read_bytes()).hexdigest()
        e['delta_vs_s6'] = {k: (e[k] - rep['s6'][k]) if e.get(k) is not None else None
                            for k in ('luts', 'registers', 'bram36', 'dsp', 'bufgctrl')}
        rep['builds'][name] = e
    o = {}
    for name in ('rw8', 'rw1'):
        if name in rep['builds']:
            e = rep['builds'][name]
            o[f'H7_{name}'] = dict(confirmed=e['delta_vs_s6']['luts'] <= 150 and e['delta_vs_s6']['registers'] <= 120 and
                                   e['bufgctrl'] == 2 and e['bram36'] == 64 and e['dsp'] == 8 and e['wns_ns'] >= 0,
                                   delta=e['delta_vs_s6'], wns_ns=e['wns_ns'])
    if 'p0' in rep['builds']:
        e = rep['builds']['p0']
        o['H8_utilization'] = dict(equal_to_s6=all(e[k] == rep['s6'][k] for k in ('luts', 'registers', 'bram36', 'dsp')),
                                   delta=e['delta_vs_s6'], wns_ns=e['wns_ns'])
    if 'rw8' in rep['builds'] and 'rw1' in rep['builds']:
        o['H7_rw1_vs_rw8_luts'] = rep['builds']['rw1']['luts'] - rep['builds']['rw8']['luts']
    rep['outcomes'] = o
    (S.RES / 'R3/fabric.json').write_text(json.dumps(rep, indent=2) + '\n')
    return rep


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'fabric':
        print(json.dumps(fabric(), indent=1))
        return
    if cmd == 'collect':
        rep = collect()
        print(json.dumps({k: rep[k] for k in ('sets', 'failed', 'passed')}, indent=1))
        if not rep['passed']:
            raise SystemExit('verification failed: ' + ', '.join(rep['failed']))
    elif cmd == 'duties':
        print(json.dumps(duties(), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
