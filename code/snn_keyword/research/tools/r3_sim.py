"""R3 xsim runner for the full-SoC testbench (code/pynqz2_riscv_flow/sim/tb_r3_soc.sv).

    r3_sim.py image <firmware.bin> --out <file.hex>         65,536-word hex for $readmemh
    r3_sim.py vectors f2 --out-dir <dir>                    request file + expected outputs
    r3_sim.py vectors f3 --out-dir <dir>
    r3_sim.py run <name>=<firmware.bin|hex>,<requests.req> [...] --out-dir <dir> [--generic POWER_OPT=1]
    r3_sim.py check <name.log> <set.expect.json> [--out <json>]

Vector sets (deterministic):
  f2  opcode 1 x 40: results/sheila_hardneg/verification_vectors.npz through the stage-1 model
      (results/S3/models/h48_k64_seed2.npz); expected scores/spikes/decision = model.integer_forward;
      cycles are compared with the S6 board anchor (results/S6/board/h48_k64_seed2.json).
  f3  opcode 6 x 3: verifier windows (150 frames) cut from the R2 test cache exactly as
      verifier_cascade.windows() does: a live keyword (highest cached head of live item 0), a negative
      (negatives stream, item 0, request 1200) and a zero-padded window at the start of live item 0
      (request 1). Expected = verifier_export.oracle(); the cached head (vscores_r2_s1_test.npz)
      must equal the oracle head.
run: every job gets its own space-free temp folder (sources copied, as run_tb_kdot.sh does), then
xvlog --sv, xelab tb_r3_soc, xsim --runall; jobs run in parallel. Writes <out-dir>/<name>.log and
<name>.run.json (wall time, simulated cycles, cycles per second).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

TOOLS = Path(__file__).resolve().parent
KWS = TOOLS.parents[1]                                   # code/snn_keyword
REPO = KWS.parents[1]
FLOW = KWS.parent / 'pynqz2_riscv_flow'
RES = TOOLS.parent / 'results'
VIVADO_BIN = Path(os.environ.get('VIVADO_BIN', 'C:/AMDDesignTools/2026.1/Vivado/bin'))
RTL = ['spike_soc.v', 'ps_if.v', 'kdot_pcpi.v', 'picorv32.v', 'poisson.v']
TB = 'tb_r3_soc.sv'
MEM_WORDS = 65536
MB_FIRST = 6                                             # the REQ line's mb= list starts at MB[6]

STAGE1 = RES / 'S3/models/h48_k64_seed2.npz'
VECTORS = KWS / 'results/sheila_hardneg/verification_vectors.npz'
ANCHOR = RES / 'S6/board/h48_k64_seed2.json'
VERIFIER = RES / 'R2/verifier/sheila_head_r2_s1.pt'
CACHE = KWS / 'data_verifier/cache_test_sheila_r2_2_m.npz'
VSCORES = KWS / 'data_verifier/vscores_r2_s1_test.npz'
HOP, VFRAMES, WARMUP, BOUNDARY = 25, 150, 5, 10


def u32(x):
    return int(x) & 0xFFFFFFFF


def shown(p):
    p = Path(p).resolve()
    return p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else p.as_posix()


# --------------------------------------------------------------------------- image

def write_image(binary, out):
    data = Path(binary).read_bytes()
    if len(data) > 4 * MEM_WORDS:
        raise SystemExit(f'{binary}: {len(data)} bytes do not fit the 256 KiB BRAM')
    words = np.zeros(MEM_WORDS, np.uint32)
    words[:(len(data) + 3) // 4] = np.frombuffer(data + bytes(-len(data) % 4), '<u4')
    Path(out).write_text('\n'.join(f'{w:08x}' for w in words) + '\n')
    return out


# --------------------------------------------------------------------------- vectors

def payload_words(data):
    data = bytes(data)
    return np.frombuffer(data + bytes(-len(data) % 4), '<u4')


def write_requests(path, requests):
    """requests: dicts with op, length, gap, flags, payload (bytes)."""
    lines = []
    for r in requests:
        w = payload_words(r.get('payload', b''))
        lines.append(f"{r['op']} {r['length']} {r['gap']} {r.get('flags', 0)} {len(w)}")
        lines += [' '.join(f'{x:08x}' for x in w[i:i + 8]) for i in range(0, len(w), 8)]
    Path(path).write_text('\n'.join(lines) + '\n')


def set_f2(gap):
    sys.path.insert(0, str(KWS))
    from model import integer_forward
    q = dict(np.load(STAGE1))
    x = np.load(VECTORS)['x'].astype(np.uint8)
    if 'w1_mask' in q and (q['w1'][~q['w1_mask'].astype(bool)] != 0).any():
        raise SystemExit('stage-1 w1 has non-zero weights outside the kept synapses')
    scores, spikes = integer_forward(x, q)
    thr = int(q['decision_threshold'])
    reqs, items = [], []
    for i in range(len(x)):
        reqs.append(dict(op=1, length=768, gap=gap, payload=x[i].tobytes()))
        items.append(dict(id=f'vector{i}', op=1, cycles_mb=9,
                          expect={'6': 0, '7': u32(scores[i, 0]), '8': u32(scores[i, 1]), '10': u32(spikes[i]),
                                  '11': int(scores[i, 1] - scores[i, 0] >= thr)}))
    anchor = json.loads(ANCHOR.read_text())
    sw = anchor['variants']['ksdot_lut']['single_window']
    return reqs, dict(set='f2', model=shown(STAGE1), vectors=shown(VECTORS), items=items,
                      anchor=dict(file=shown(ANCHOR), image_sha256=anchor['firmware']['images']['keyword_ksdot_lut']['sha256'],
                                  cycles_min=sw['cycles_min'], cycles_mean=sw['cycles_mean'], cycles_max=sw['cycles_max']))


def hop_ends(n):
    return np.minimum(np.arange(HOP - 1, n + HOP - 1, HOP), n - 1)


def window(frames, end):
    pad = np.concatenate([np.zeros((VFRAMES, 24), np.uint8), frames])
    return pad[end + 1:end + 1 + VFRAMES]


def logits_hash(lg):
    flat = np.asarray(lg, np.int64).ravel()
    return u32(int((flat * np.arange(1, len(flat) + 1, dtype=np.int64)).sum()))


def set_f3(gap):
    os.environ['KWS_KEYWORD'] = 'sheila'
    sys.path.append(str(TOOLS / 'mateo_eval'))
    import verifier_export
    q = verifier_export.load_quantized(VERIFIER)
    c, v = np.load(CACHE), np.load(VSCORES)
    if str(v['score']) != 'head' or int(v['window']) != VFRAMES or Path(str(v['checkpoint']).replace('\\', '/')).name != VERIFIER.name:
        raise SystemExit(f'{VSCORES} is not the head score of {VERIFIER.name} on {VFRAMES} frames')
    picks = []
    off, vo = c['live_off'], v['live_voff']
    fr = c['live_frames'][off[0]:off[1]]
    heads = v['live_va'][vo[0]:vo[1]]
    k = int(np.argmax(heads))
    picks.append(('live item 0, request %d (highest cached head)' % k, fr, k, int(heads[k])))
    picks.append(('live item 0, request 1 (zero-padded start)', fr, 1, int(heads[1])))
    off, vo = c['neg_off'], v['neg_voff']
    fr = c['neg_frames'][off[0]:off[1]]
    k = 1200
    picks.append(('negatives item 0, request %d' % k, fr, k, int(v['neg_va'][vo[0] + k])))
    wins = np.stack([window(f, hop_ends(len(f))[k]) for _, f, k, _ in picks])
    lg, res = verifier_export.oracle(wins, q, WARMUP, BOUNDARY)
    reqs, items = [], []
    for i, (desc, f, k, cached) in enumerate(picks):
        reqs.append(dict(op=6, length=VFRAMES * 24, gap=gap, payload=wins[i].tobytes()))
        sa, ea, sb, eb, hd, eh = (int(x) for x in res[i])
        items.append(dict(id=desc, op=6, cycles_mb=21, request_end_frame=int(hop_ends(len(f))[k]),
                          zero_frames=int(max(0, VFRAMES - 1 - hop_ends(len(f))[k])), cached_head=cached,
                          oracle_head=hd, cached_equals_oracle=cached == hd,
                          expect={'6': 0, '26': u32(sa), '27': u32(ea), '28': u32(sb), '29': u32(eb),
                                  '30': logits_hash(lg[i]), '32': u32(hd), '33': u32(eh)}))
    if not all(it['cached_equals_oracle'] for it in items):
        raise SystemExit('oracle head differs from the cached head: ' + json.dumps(items, indent=1))
    return reqs, dict(set='f3', verifier=shown(VERIFIER), cache=shown(CACHE), vscores=shown(VSCORES),
                      warmup=WARMUP, boundary=BOUNDARY, frames=VFRAMES, items=items)


SETS = {'f2': set_f2, 'f3': set_f3}


# --------------------------------------------------------------------------- run

def xsim_job(name, fw, req, out_dir, generics, keep):
    out_dir = Path(out_dir)
    work = Path(tempfile.mkdtemp(prefix='5xpg0_r3_'))
    try:
        for f in RTL:
            shutil.copy(FLOW / 'rtl' / f, work / f)
        shutil.copy(FLOW / 'sim' / TB, work / TB)
        if Path(fw).suffix == '.bin':
            write_image(fw, work / 'fw.hex')
        else:
            shutil.copy(fw, work / 'fw.hex')
        shutil.copy(req, work / 'req.req')
        gen = [x for g in generics for x in ('--generic_top', g)]
        # The RTL is plain Verilog (ps_if.v uses rstate before its declaration, which SystemVerilog
        # mode rejects); only the testbench is SystemVerilog.
        steps = [[str(VIVADO_BIN / 'xvlog.bat'), '--relax', *RTL],
                 [str(VIVADO_BIN / 'xvlog.bat'), '--sv', TB],
                 [str(VIVADO_BIN / 'xelab.bat'), 'tb_r3_soc', '-s', 'snap', *gen],
                 [str(VIVADO_BIN / 'xsim.bat'), 'snap', '--runall', '--testplusarg', 'fw=fw.hex',
                  '--testplusarg', 'req=req.req']]
        times = []
        for cmd in steps:
            t0 = time.perf_counter()
            # The Vivado .bat wrappers split unquoted arguments at '=' (fw=fw.hex, POWER_OPT=1): quote them.
            line = ' '.join(f'"{x}"' if ('=' in x or ' ' in x) else x for x in cmd)
            p = subprocess.run(line, cwd=work, capture_output=True, text=True)
            times.append(time.perf_counter() - t0)
            if p.returncode != 0:
                (out_dir / f'{name}.log').write_text(p.stdout + p.stderr)
                raise RuntimeError(f'{name}: {Path(cmd[0]).name} failed (see {name}.log)')
        log = p.stdout
        (out_dir / f'{name}.log').write_text(log)
        m = re.search(r'^SUM .*cyc=(\d+)', log, re.M)
        cycles = int(m.group(1)) if m else None
        info = dict(name=name, firmware=shown(fw), requests=shown(req), generics=generics, work=str(work) if keep else None,
                    seconds=dict(xvlog=round(times[0] + times[1], 1), xelab=round(times[2], 1), xsim=round(times[3], 1)),
                    simulated_cycles=cycles, done='DONE' in log,
                    cycles_per_second=round(cycles / times[3]) if cycles else None)
        (out_dir / f'{name}.run.json').write_text(json.dumps(info, indent=2) + '\n')
        return info
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- check

def parse_log(path):
    reqs, total = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith('REQ '):
            kv = dict(t.split('=', 1) for t in line.split()[1:])
            mb = [int(x, 16) for x in kv.pop('mb').split(',')]
            r = {k: int(v) for k, v in kv.items()}
            r['mb'] = {str(MB_FIRST + i): w for i, w in enumerate(mb)}
            reqs.append(r)
        elif line.startswith('SUM '):
            total = {k: int(v) for k, v in (t.split('=', 1) for t in line.split()[1:])}
    return reqs, total


def signed(x):
    return x - (1 << 32) if x & 0x80000000 else x


def check(log, expect):
    reqs, total = parse_log(log)
    exp = json.loads(Path(expect).read_text())
    items = exp['items']
    out = dict(log=shown(log), expect=shown(expect), set=exp['set'], requests=len(reqs), expected=len(items),
               mismatches=[], total=total)
    cycles = []
    for i, it in enumerate(items):
        if i >= len(reqs):
            out['mismatches'].append(dict(item=it['id'], error='missing request'))
            continue
        r = reqs[i]
        bad = {k: dict(got=signed(r['mb'][k]), expected=signed(v)) for k, v in it['expect'].items() if r['mb'][k] != v}
        if r['op'] != it['op'] or bad:
            out['mismatches'].append(dict(item=it['id'], fields=bad))
        cycles.append(r['mb'][str(it['cycles_mb'])])
    out['bit_exact'] = not out['mismatches'] and len(reqs) == len(items)
    if cycles:
        out['cycles'] = dict(min=min(cycles), mean=round(float(np.mean(cycles)), 2), max=max(cycles), all=cycles)
    out['service_cycles'] = [r['svc'] for r in reqs]
    out['activity'] = [{k: r[k] for k in ('svc', 'pb_re', 'pb_we', 'pa_re', 'pa_we', 'clk_on', 'sleep')} for r in reqs]
    if 'anchor' in exp and cycles:
        a = exp['anchor']
        out['anchor'] = dict(board=dict(min=a['cycles_min'], mean=a['cycles_mean'], max=a['cycles_max']),
                             equal=(min(cycles), round(float(np.mean(cycles)), 1), max(cycles)) ==
                             (a['cycles_min'], round(a['cycles_mean'], 1), a['cycles_max']))
    return out


def duty(acts, key):
    svc = sum(a['svc'] for a in acts)
    return round(sum(a[key] for a in acts) / svc, 4) if svc else None


def feasibility(sim_dir, out):
    """Phase 1 F1-F3 (plan section 3, step 5) from the build reports and the checked xsim logs."""
    import hashlib
    sim_dir, out = Path(sim_dir), Path(out)
    logs = out.parent / 'sim'
    logs.mkdir(parents=True, exist_ok=True)
    builds = {d: json.loads((KWS / f'build/r3_fw/{d}/build.json').read_text()) for d in ('b1', 'b2')}
    anchor_bin = KWS / 'build/r3_s6_anchor/keyword_ksdot_lut.bin'
    f2 = check(sim_dir / 'f2.log', sim_dir / 'f2.expect.json')
    f3 = [check(sim_dir / f'f3_{i}.log', sim_dir / f'f3_{i}.expect.json') for i in range(3)]
    f3_items = [it for i in range(3) for it in json.loads((sim_dir / f'f3_{i}.expect.json').read_text())['items']]
    runs = {n: json.loads((sim_dir / f'{n}.run.json').read_text()) for n in ('f2', 'f3_0', 'f3_1', 'f3_2')}
    for n in runs:   # raw xsim output kept as .xsim.txt (*.log is git-ignored at the repo root)
        shutil.copy(sim_dir / f'{n}.log', logs / f'{n}.xsim.txt')
        shutil.copy(sim_dir / f'{n}.run.json', logs / f'{n}.run.json')
    tv = [c['cycles']['all'][0] for c in f3]
    image_sha = hashlib.sha256(anchor_bin.read_bytes()).hexdigest()
    board_sha = json.loads((sim_dir / 'f2.expect.json').read_text())['anchor']['image_sha256']
    rep = dict(
        experiment='R3 Phase 1 feasibility', rtl='unchanged (r3-base)', simulator='Vivado xsim 2026.1 (tb_r3_soc.sv)',
        F1=dict(designs={d.upper(): dict(t1=b['t1'], t2=b['t2'], sha256=b['sha256'], **b['f1_memory'])
                         for d, b in builds.items()},
                passed=all(b['f1_memory']['pass'] for b in builds.values())),
        F2=dict(image=shown(anchor_bin), image_sha256=image_sha, image_equals_board=image_sha == board_sha,
                vectors=f2['requests'], bit_exact=f2['bit_exact'], mismatches=f2['mismatches'],
                cycles={k: f2['cycles'][k] for k in ('min', 'mean', 'max')}, cycles_all=f2['cycles']['all'],
                board=f2['anchor']['board'], anchor_equal=f2['anchor']['equal'],
                port_b_read_duty=duty(f2['activity'], 'pb_re')),
        F3=dict(image=builds['b1']['out'] + '/cascade.bin', image_sha256=builds['b1']['sha256'], windows=len(f3),
                bit_exact=all(c['bit_exact'] for c in f3), mismatches=[m for c in f3 for m in c['mismatches']],
                items=[dict(id=it['id'], zero_frames=it['zero_frames'], head=it['oracle_head'],
                            cached_head=it['cached_head'], cached_equals_oracle=it['cached_equals_oracle'],
                            t_v_cycles=t, service_cycles=c['service_cycles'][0],
                            port_b_read_duty=duty(c['activity'], 'pb_re'))
                       for it, t, c in zip(f3_items, tv, f3)],
                t_v_cycles=dict(min=min(tv), mean=round(float(np.mean(tv)), 1), max=max(tv)),
                t_v_ms=dict(min=round(min(tv) / 1e5, 3), mean=round(float(np.mean(tv)) / 1e5, 3), max=round(max(tv) / 1e5, 3))),
        simulator_speed=dict(runs={n: dict(cycles=r['simulated_cycles'], xsim_s=r['seconds']['xsim'],
                                           cycles_per_second=r['cycles_per_second']) for n, r in runs.items()},
                             note='xsim wall time includes start-up (snapshot load, $readmemh); 4 jobs ran in parallel'))
    out.write_text(json.dumps(rep, indent=2) + '\n', encoding='utf-8')
    return rep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('image'); s.add_argument('bin', type=Path); s.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('vectors'); s.add_argument('set', choices=sorted(SETS))
    s.add_argument('--out-dir', type=Path, required=True); s.add_argument('--gap', type=int, default=3000)
    s.add_argument('--split', action='store_true', help='also one request file (and expect file) per request')
    s = sub.add_parser('run'); s.add_argument('jobs', nargs='+', help='name=firmware,requests')
    s.add_argument('--out-dir', type=Path, required=True); s.add_argument('--generic', action='append', default=[])
    s.add_argument('--keep', action='store_true', help='keep the temp work folders')
    s = sub.add_parser('check'); s.add_argument('log', type=Path); s.add_argument('expect', type=Path)
    s.add_argument('--out', type=Path)
    s = sub.add_parser('feasibility', help='Phase 1: collect F1-F3 into results/R3/phase1/feasibility.json')
    s.add_argument('--sim-dir', type=Path, default=KWS / 'build/r3_sim/phase1')
    s.add_argument('--out', type=Path, default=RES / 'R3/phase1/feasibility.json')
    a = p.parse_args()
    if a.cmd == 'feasibility':
        rep = feasibility(a.sim_dir, a.out)
        print(json.dumps({k: rep[k] for k in ('F1', 'F2', 'F3', 'simulator_speed')}, indent=1)[:6000])
        return
    if a.cmd == 'image':
        write_image(a.bin, a.out)
        print('wrote', a.out)
    elif a.cmd == 'vectors':
        a.out_dir.mkdir(parents=True, exist_ok=True)
        reqs, exp = SETS[a.set](a.gap)
        write_requests(a.out_dir / f'{a.set}.req', reqs)
        (a.out_dir / f'{a.set}.expect.json').write_text(json.dumps(exp, indent=2) + '\n')
        if a.split:   # one request per file, for parallel jobs
            for i, (r, it) in enumerate(zip(reqs, exp['items'])):
                write_requests(a.out_dir / f'{a.set}_{i}.req', [r])
                (a.out_dir / f'{a.set}_{i}.expect.json').write_text(json.dumps({**exp, 'items': [it]}, indent=2) + '\n')
        print(json.dumps({k: v for k, v in exp.items() if k != 'items'} | {'requests': len(reqs)}, indent=1))
    elif a.cmd == 'run':
        a.out_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        for j in a.jobs:
            name, rest = j.split('=', 1)
            fw, req = rest.split(',', 1)
            jobs.append((name, Path(fw), Path(req)))
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futs = [ex.submit(xsim_job, n, fw, req, a.out_dir, a.generic, a.keep) for n, fw, req in jobs]
            infos = [f.result() for f in futs]
        print(json.dumps(infos, indent=1))
    else:
        out = check(a.log, a.expect)
        if a.out:
            a.out.write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps({k: v for k, v in out.items() if k not in ('activity', 'service_cycles')}, indent=1))
        if not out['bit_exact']:
            raise SystemExit('not bit-exact')


if __name__ == '__main__':
    main()
