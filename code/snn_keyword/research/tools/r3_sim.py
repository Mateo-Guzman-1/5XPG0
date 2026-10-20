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


def cascade_requests(oracle, frames, xwin, code, gap, wreq=None):
    """Opcode 5 (reset) + the command-7 requests of one item, with the oracle's expected words."""
    import r3_oracle as O
    t1, t2, _ = O.thresholds(code)
    reqs = [dict(op=5, length=0, gap=gap)]
    items = [dict(id='reset', op=5, cycles_mb=9, expect={'6': 0, '18': 0})]
    for r in oracle.cascade(frames, xwin, t1, t2, wreq):
        a, b = r['frames']
        payload = (xwin[r['window']].tobytes() if r['window'] is not None else b'') + np.asarray(frames[a:b], np.uint8).tobytes()
        reqs.append(dict(op=7, length=len(payload), gap=gap, payload=payload))
        items.append(dict(id=f"request {r['k']}", op=7, cycles_mb=9, cycles_extra=['20', '21', '22'],
                          window=r['window'], margin=r['margin'], frames=b - a,
                          expect={k: u32(v) for k, v in r['expect'].items()}))
    return reqs, items


def r1_r2_item(group, i, c, r1):
    """Frames (r2 cache) and windows (r1 cache) of item i; checks the stage-1 margins against the r2 trace."""
    import r3_oracle as O
    off, roff = c[f'{group}_off'], r1['off']
    frames = c[f'{group}_frames'][off[i]:off[i + 1]]
    xwin = r1['x'][roff[i]:roff[i + 1]]
    raw = c[f'{group}_raw'][off[i]:off[i + 1]]
    return frames, xwin, raw


def check_margins(oracle, frames, xwin, raw, wframes):
    """Stage-1 margins (integer_forward) equal the r2 cache trace at the window frames (plan 2A cross-check)."""
    s, _ = oracle.stage1(xwin)
    got = s[:, 1] - s[:, 0]
    if not np.array_equal(got, raw[wframes]):
        raise SystemExit('stage-1 margins differ from the r2 cache trace')


def select_live(c, v, code, want):
    """Live keyword items (y = 1) by the cached margins and heads under the design's thresholds; returns the first
    item with the fewest verifier runs among those with want(reach, run, det, heads, ends, n) true."""
    import r3_oracle as O
    t1, t2, _ = O.thresholds(code)
    off, voff, y = c['live_off'], v['live_voff'], c['live_y']
    raw, heads = c['live_raw'], v['live_va']
    best = None
    for i in np.flatnonzero(y == 1):
        n = int(off[i + 1] - off[i])
        ends = O.hop_ends(n)
        hm = np.maximum.reduceat(raw[off[i]:off[i + 1]], np.r_[0, ends[:-1] + 1])
        hd = heads[voff[i]:voff[i + 1]]
        reach = hm >= t1
        run = reach | np.r_[False, reach[:-1]]
        det = O.decide(reach, lambda k: hd[k], ends, t2)
        if want(reach, run, det, hd, ends, n, hm) and (best is None or run.sum() < best[1]):
            best = (int(i), int(run.sum()))
    if best is None:
        raise SystemExit('no live item meets the selection rule')
    return best[0]


def vset_cascade(name, gap):
    import r3_oracle as O
    oracle = O.Oracle()
    c, v = np.load(O.CACHE), np.load(O.VSCORES)
    if name == 'v3a':
        # live keyword: reach in >= 2 consecutive requests, >= 2 runs with head >= t2 within 100 frames (the
        # hold-off suppresses the second), a run from the k-1 rule only, and a partial last request
        t1, t2, _ = O.thresholds('B1')

        def want(reach, run, det, hd, ends, n, hm):
            hits = [ends[k] for k in np.flatnonzero(run) if hd[k] >= t2]
            return (n % 25 != 0 and (reach[:-1] & reach[1:]).any() and (run & ~reach).any() and
                    any(b - a < 100 for a, b in zip(hits, hits[1:])))
        i = select_live(c, v, 'B1', want)
        r1 = np.load(KWS / 'build/r1_cache/live_test.npz')
        frames, xwin, raw = r1_r2_item('live', i, c, r1)
        check_margins(oracle, frames, xwin, raw, np.minimum(98 + 25 * np.arange(len(xwin)), len(frames) - 1))
        reqs, items = cascade_requests(oracle, frames, xwin, 'B1', gap)
        sel = dict(group='live', item=i, rule='B1: reach in 2 consecutive requests, k-1 run, 2 head hits < 100 frames, '
                   'partial last request (fewest verifier runs)', frames=len(frames))
        designs = {'v3a': ('B1', reqs, items)}
    elif name == 'v3b':
        # 1 h stream: 40 requests far (> 3 s) from every keyword, exactly one proposal (not first or last)
        t1, t2, _ = O.thresholds('B1')
        raw = c['stream_raw']
        n = len(raw)
        ends = O.hop_ends(n)
        reach = np.maximum.reduceat(raw, np.r_[0, ends[:-1] + 1]) >= t1
        marks = c['stream_marks']
        for k0 in range(3, len(ends) - 40):
            r = reach[k0:k0 + 40]
            t0, t1s = 0.25 * k0, 0.25 * (k0 + 40)
            far = not ((marks[:, 0] < t1s + 3) & (marks[:, 1] > t0 - 3)).any()
            if far and r.sum() == 1 and 1 <= int(np.flatnonzero(r)[0]) <= 38:
                break
        else:
            raise SystemExit('no stream stretch meets the rule')
        r1 = np.load(KWS / 'build/r1_cache/stream_test.npz')
        frames = c['stream_frames'][25 * k0:25 * (k0 + 40)]
        js = np.arange(k0 - 3, k0 + 37)                       # window j counts in request j + 3
        xwin = r1['x'][js]
        check_margins(oracle, c['stream_frames'][:1], xwin, raw, 98 + 25 * js)
        reqs, items = cascade_requests(oracle, frames, xwin, 'B1', gap, wreq=np.arange(40))
        sel = dict(group='stream', first_request=int(k0), requests=40, seconds=[0.25 * k0, 0.25 * (k0 + 40)],
                   rule='B1: 40 requests > 3 s from every keyword mark, exactly one reach (not first or last)')
        # v5s1: the stage-1-only start of v3b (reset + the requests before the first verifier run), for the
        # equivalence testbench and the POWER_OPT 0/1 cycle identity (plan 2C, V5)
        first_run = next(i for i, it in enumerate(items) if it['op'] == 7 and it['expect']['11'] & 4)
        designs = {'v3b': ('B1', reqs, items), 'v5s1': ('B1', reqs[:first_run], items[:first_run])}
    else:   # v4: a margin in [t1(B1), t1(B2)): B1 runs a request that B2 does not; B2 detects once
        t1b1, t2, _ = O.thresholds('B1')
        t1b2, _, _ = O.thresholds('B2')

        def want(reach, run, det, hd, ends, n, hm):
            r2 = hm >= t1b2
            run2 = r2 | np.r_[False, r2[:-1]]
            det2 = O.decide(r2, lambda k: hd[k], ends, t2)
            return ((run & ~run2).any() and ((hm >= t1b1) & (hm < t1b2)).any() and sum(b & 1 for b in det2) == 1)
        i = select_live(c, v, 'B1', want)
        r1 = np.load(KWS / 'build/r1_cache/live_test.npz')
        frames, xwin, raw = r1_r2_item('live', i, c, r1)
        check_margins(oracle, frames, xwin, raw, np.minimum(98 + 25 * np.arange(len(xwin)), len(frames) - 1))
        sel = dict(group='live', item=i, rule='a margin in [t1(B1), t1(B2)) so that B1 runs a request B2 does not; '
                   'B2 detects once (fewest B1 verifier runs)', frames=len(frames))
        designs = {f'v4{code.lower()}': (code, *cascade_requests(oracle, frames, xwin, code, gap)) for code in ('B1', 'B2')}
    out = {}
    for set_name, (code, reqs, items) in designs.items():
        runs = sum(1 for it in items if it['op'] == 7 and it['expect']['11'] & 4)
        out[set_name] = (reqs, dict(set=set_name, design=code, selection=sel, verifier_runs=runs,
                                    detections=sum(1 for it in items if it['op'] == 7 and it['expect']['11'] & 1),
                                    items=items))
    return out


def set_v1(gap):
    reqs, exp = set_f2(gap)
    return reqs, {**exp, 'set': 'v1'}


def set_rel(gap):
    """The release firmware (deploy/keyword_kdot.bin, H48 8-bit kdot + LIF table): 40 vectors through deploy/model.npz;
    board anchor results/sheila_h48_b8/board_kdot.json (regression and equivalence, plan 2C)."""
    sys.path.insert(0, str(KWS))
    from model import integer_forward
    q = dict(np.load(KWS / 'deploy/model.npz'))
    x = np.load(VECTORS)['x'].astype(np.uint8)
    scores, spikes = integer_forward(x, q)
    thr = int(q['decision_threshold'])
    reqs = [dict(op=1, length=768, gap=gap, payload=x[i].tobytes()) for i in range(len(x))]
    items = [dict(id=f'vector{i}', op=1, cycles_mb=9,
                  expect={'6': 0, '7': u32(scores[i, 0]), '8': u32(scores[i, 1]), '10': u32(spikes[i]),
                          '11': int(scores[i, 1] - scores[i, 0] >= thr)}) for i in range(len(x))]
    board = json.loads((KWS / 'results/sheila_h48_b8/board_kdot.json').read_text())['single_window']
    return reqs, dict(set='rel', model='code/snn_keyword/deploy/model.npz', vectors=shown(VECTORS), items=items,
                      anchor=dict(file='code/snn_keyword/results/sheila_h48_b8/board_kdot.json', image_sha256=None,
                                  cycles_min=board['cycles_min'], cycles_mean=board['cycles_mean'],
                                  cycles_max=board['cycles_max']))


def set_sleep(gap):
    """Sleep checks (plan 2C.3) for firmware/r3_sleep_test.c: 3 status requests, then a race sweep (MB[0] written
    0..120 clocks after the CPU's previous MB[1] write, so that at READ_WAIT 1 and 8 it lands before, at and after
    the SLEEP accept), then a CTRL reset while asleep and 2 more status requests."""
    reqs = [dict(op=2, length=0, gap=gap) for _ in range(2)] + [dict(op=2, length=0, gap=gap, flags=1)]
    reqs += [dict(op=2, length=0, gap=d, flags=3) for d in range(121)]
    reqs += [dict(op=2, length=0, gap=gap, flags=4), dict(op=2, length=0, gap=gap)]
    items = [dict(id=f'status {i}', op=2, cycles_mb=9, expect={'6': 0}) for i in range(len(reqs))]
    return reqs, dict(set='sleep', firmware='firmware/r3_sleep_test.c', items=items)


def sleep_check(log, power_opt=1):
    """Plan 2C.3 on a sleep-set log: every request answered, no testbench errors, the race sweep covers -3..+3
    clocks around the SLEEP accept, reset while asleep clears sleeping and the doorbell, the POWER counters equal
    the testbench's counts, mask 0 does not stall, and per request clock-on + asleep cycles = service cycles."""
    extra = {}
    reqs, total = parse_log(log, extra)
    races = sorted({r['race'] for r in reqs if r['race'] is not None})
    dump = extra.get('dump', {})
    counters = [(r['mb']['34'], r['tb_asleep'], r['mb']['35'], r['tb_sleeps']) for r in reqs]
    out = dict(log=shown(log), requests=len(reqs), status_ok=all(r['mb']['6'] == 0 for r in reqs),
               tb_errors=extra.get('errors'), error_lines=extra.get('error_lines', []),
               race_offsets=races, race_covers_minus3_to_plus3=all(d in races for d in range(-3, 4)),
               race_requests=sum(1 for r in reqs if r['flags'] & 2),
               reset_asleep=extra.get('reset_asleep'),
               counters_equal=all(a == b and c == d for a, b, c, d in counters),
               mask0=dict(cycles=dump.get('MB[40]'), sleeps_after=dump.get('MB[41]'), version=dump.get('MB[42]')),
               clock_accounting_max_diff=max(abs(r['clk_on'] + r['sleep'] - r['svc']) for r in reqs),
               sleeps_total=total.get('sleeps') if total else None)
    out['mask0_ok'] = (out['mask0']['cycles'] is not None and out['mask0']['cycles'] < 100 and
                       out['mask0']['sleeps_after'] == 0 and out['mask0']['version'] == (0x20007 if power_opt else 0x20003))
    out['passed'] = (out['status_ok'] and out['tb_errors'] == 0 and not out['error_lines'] and
                     out['race_covers_minus3_to_plus3'] and out['counters_equal'] and out['mask0_ok'] and
                     out['clock_accounting_max_diff'] <= 1 and
                     bool(out['reset_asleep']) and all(x == {'sleeping': 0, 'doorbell': 0} for x in out['reset_asleep']))
    return out


def set_v2(gap):
    reqs, exp = set_f3(gap)
    return reqs, {**exp, 'set': 'v2'}


SETS = {'f2': set_f2, 'f3': set_f3, 'v1': set_v1, 'v2': set_v2, 'rel': set_rel, 'sleep': set_sleep}
CASCADE_SETS = {'v3a': ('v3a',), 'v3b': ('v3b',), 'v4': ('v4b1', 'v4b2')}


# --------------------------------------------------------------------------- run

REF_TAG = 'r3-base'
REF_MODULES = {'spike_soc.v': 'spike_soc', 'ps_if.v': 'ps_if'}


def reference_rtl(work):
    """r3-base copies of spike_soc.v / ps_if.v with the modules renamed *_ref (equivalence testbench)."""
    names = []
    for f, mod in REF_MODULES.items():
        src = subprocess.run(['git', 'show', f'{REF_TAG}:code/pynqz2_riscv_flow/rtl/{f}'], cwd=REPO, check=True,
                             capture_output=True, text=True).stdout
        if src.count(f'module {mod} ') + src.count(f'module {mod}#') + src.count(f'module {mod}\n') != 1:
            raise SystemExit(f'{REF_TAG}:{f}: module {mod} not found once')
        out = re.sub(rf'\bmodule {mod}\b', f'module {mod}_ref', src)
        (work / f'ref_{f}').write_text(out)
        names.append(f'ref_{f}')
    return names


def xsim_job(name, fw, req, out_dir, generics, keep, equiv=False, plusargs=()):
    out_dir = Path(out_dir)
    work = Path(tempfile.mkdtemp(prefix='5xpg0_r3_'))
    try:
        for f in RTL:
            shutil.copy(FLOW / 'rtl' / f, work / f)
        shutil.copy(FLOW / 'sim' / TB, work / TB)
        ref = reference_rtl(work) if equiv else []
        if Path(fw).suffix == '.bin':
            write_image(fw, work / 'fw.hex')
        else:
            shutil.copy(fw, work / 'fw.hex')
        shutil.copy(req, work / 'req.req')
        gen = [x for g in generics for x in ('--generic_top', g)]
        # The RTL is plain Verilog (ps_if.v uses rstate before its declaration, which SystemVerilog
        # mode rejects); only the testbench is SystemVerilog.
        steps = [[str(VIVADO_BIN / 'xvlog.bat'), '--relax', *RTL, *ref],
                 [str(VIVADO_BIN / 'xvlog.bat'), '--sv', *(['-d', 'R3_EQUIV'] if equiv else []), TB],
                 [str(VIVADO_BIN / 'xelab.bat'), 'tb_r3_soc', '-s', 'snap', *gen],
                 [str(VIVADO_BIN / 'xsim.bat'), 'snap', '--runall', '--testplusarg', 'fw=fw.hex',
                  '--testplusarg', 'req=req.req', *[x for p in plusargs for x in ('--testplusarg', p)]]]
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
        info = dict(name=name, firmware=shown(fw), requests=shown(req), generics=generics, equiv=equiv,
                    plusargs=list(plusargs), work=str(work) if keep else None,
                    seconds=dict(xvlog=round(times[0] + times[1], 1), xelab=round(times[2], 1), xsim=round(times[3], 1)),
                    simulated_cycles=cycles, done='DONE' in log,
                    cycles_per_second=round(cycles / times[3]) if cycles else None)
        (out_dir / f'{name}.run.json').write_text(json.dumps(info, indent=2) + '\n')
        return info
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- check

def parse_log(path, extra=None):
    """REQ lines -> dicts; SUM -> totals. extra (dict, optional) receives ERRORS, EQUIV, DUMP, RESET_ASLEEP lines."""
    reqs, total = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith('REQ '):
            kv = dict(t.split('=', 1) for t in line.split()[1:])
            mb = [int(x, 16) for x in kv.pop('mb').split(',')]
            race = kv.pop('race', 'none')
            r = {k: int(v) if v.lstrip('-').isdigit() else float(v) for k, v in kv.items()}   # host_ms (board) is a float
            r['race'] = None if race == 'none' else int(race)
            r['mb'] = {str(MB_FIRST + i): w for i, w in enumerate(mb)}
            reqs.append(r)
        elif line.startswith('SUM '):
            total = {k: int(v) for k, v in (t.split('=', 1) for t in line.split()[1:])}
        elif extra is not None:
            if line.startswith('ERRORS='):
                extra['errors'] = int(line.split('=')[1])
            elif line.startswith('EQUIV '):
                extra['equiv'] = {k: int(v) for k, v in (t.split('=', 1) for t in line.split()[1:])}
            elif line.startswith('DUMP '):
                k, v = line.split()[1].split('=')
                extra.setdefault('dump', {})[k] = int(v, 16)
            elif line.startswith('RESET_ASLEEP'):
                extra.setdefault('reset_asleep', []).append({k: int(v) for k, v in (t.split('=') for t in line.split()[1:])})
            elif line.startswith(('ERROR', 'TRAP', 'EQUIV DIFF')):
                extra.setdefault('error_lines', []).append(line)
    return reqs, total


def signed(x):
    return x - (1 << 32) if x & 0x80000000 else x


def check(log, expect):
    extra = {}
    reqs, total = parse_log(log, extra)
    exp = json.loads(Path(expect).read_text())
    items = exp['items']
    out = dict(log=shown(log), expect=shown(expect), set=exp['set'], requests=len(reqs), expected=len(items),
               mismatches=[], total=total, tb=extra)
    cycles, per = [], []
    for i, it in enumerate(items):
        if i >= len(reqs):
            out['mismatches'].append(dict(item=it['id'], error='missing request'))
            continue
        r = reqs[i]
        bad = {k: dict(got=signed(r['mb'][k]), expected=signed(v)) for k, v in it['expect'].items() if r['mb'][k] != v}
        if r['op'] != it['op'] or bad:
            out['mismatches'].append(dict(item=it['id'], fields=bad))
        cycles.append(r['mb'][str(it['cycles_mb'])])
        per.append(dict(op=r['op'], bits=r['mb']['11'] if r['op'] == 7 else None,
                        **{f'mb{k}': r['mb'][k] for k in [str(it['cycles_mb'])] + it.get('cycles_extra', [])},
                        svc=r['svc'], pb_re=r['pb_re'], pb_we=r['pb_we'], pa_re=r['pa_re'], pa_we=r['pa_we'],
                        clk_on=r['clk_on'], sleep=r['sleep'], awake=r.get('awake'), window=it.get('window'),
                        len=r['len']))
    out['per_request'] = per
    out['bit_exact'] = not out['mismatches'] and len(reqs) == len(items)
    out['tb_errors'] = extra.get('errors')          # None: a testbench without the checks (Phase 1)
    out['passed'] = out['bit_exact'] and not extra.get('errors') and not extra.get('error_lines') and \
        extra.get('equiv', {}).get('mismatches', 0) == 0
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
    s = sub.add_parser('vectors'); s.add_argument('set', choices=sorted(SETS) + sorted(CASCADE_SETS))
    s.add_argument('--out-dir', type=Path, required=True); s.add_argument('--gap', type=int, default=3000)
    s.add_argument('--split', action='store_true', help='also one request file (and expect file) per request')
    s = sub.add_parser('run'); s.add_argument('jobs', nargs='+', help='name=firmware,requests')
    s.add_argument('--out-dir', type=Path, required=True); s.add_argument('--generic', action='append', default=[])
    s.add_argument('--keep', action='store_true', help='keep the temp work folders')
    s.add_argument('--equiv', action='store_true', help='equivalence testbench: also simulate the r3-base RTL')
    s.add_argument('--plusarg', action='append', default=[], help='extra testbench plusarg, e.g. dump=40:42')
    s = sub.add_parser('check'); s.add_argument('log', type=Path); s.add_argument('expect', type=Path)
    s.add_argument('--out', type=Path)
    s = sub.add_parser('sleepcheck', help='plan 2C.3 checks on a sleep-set log')
    s.add_argument('log', type=Path); s.add_argument('--power-opt', type=int, default=1); s.add_argument('--out', type=Path)
    s = sub.add_parser('feasibility', help='Phase 1: collect F1-F3 into results/R3/phase1/feasibility.json')
    s.add_argument('--sim-dir', type=Path, default=KWS / 'build/r3_sim/phase1')
    s.add_argument('--out', type=Path, default=RES / 'R3/phase1/feasibility.json')
    a = p.parse_args()
    if a.cmd == 'sleepcheck':
        out = sleep_check(a.log, a.power_opt)
        if a.out:
            a.out.write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps(out, indent=1))
        if not out['passed']:
            raise SystemExit('sleep checks failed')
        return
    if a.cmd == 'feasibility':
        rep = feasibility(a.sim_dir, a.out)
        print(json.dumps({k: rep[k] for k in ('F1', 'F2', 'F3', 'simulator_speed')}, indent=1)[:6000])
        return
    if a.cmd == 'image':
        write_image(a.bin, a.out)
        print('wrote', a.out)
    elif a.cmd == 'vectors' and a.set in CASCADE_SETS:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(TOOLS))
        record = RES / 'R3/sim/vectors.json'
        summary = json.loads(record.read_text()) if record.exists() else {}
        for name, (reqs, exp) in vset_cascade(a.set, a.gap).items():
            write_requests(a.out_dir / f'{name}.req', reqs)
            (a.out_dir / f'{name}.expect.json').write_text(json.dumps(exp, indent=2) + '\n')
            summary[name] = {k: v for k, v in exp.items() if k != 'items'} | {'requests': len(reqs)}
            print(json.dumps(summary[name], indent=1))
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(summary, indent=2) + '\n')
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
            futs = [ex.submit(xsim_job, n, fw, req, a.out_dir, a.generic, a.keep, a.equiv, a.plusarg)
                    for n, fw, req in jobs]
            infos = [f.result() for f in futs]
        print(json.dumps(infos, indent=1))
    else:
        out = check(a.log, a.expect)
        if a.out:
            a.out.write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps({k: v for k, v in out.items() if k not in ('activity', 'service_cycles', 'per_request')}, indent=1))
        if not out['passed']:
            raise SystemExit('not bit-exact, or testbench errors')


if __name__ == '__main__':
    main()
