"""Verifier track: cycles of the phoneme verifier on the PicoRV32 SoC RTL (Verilator), bit-exact.

Builds the stage-1 headers (export_model.export_stream, the keyword's stage-1
model of the main worktree, read only: "yes" W=20, "sheila" the QAT candidate), the verifier headers (verifier_export.export), the
firmware with -DUSE_VERIFIER (kdot + neuron engine, and pure RV32IM with
--rv32im), and sim/soc_verifier.cpp against rtl/spike_soc.v. Records are
validation windows: each record is streamed through stage 1 in 250 ms
requests, then command 6 verifies the last 150 frames. The oracle is
verifier_model.integer_forward + keyword_score on the same zero-padded window.
Cycles of the verifier do not depend on the weights' values (no data-dependent
branches except LUT clamping), so a randomly initialised model measures the
same cycles as a trained one.
Writes results/rtl_verifier_<tag>.csv and results/verification_verifier_rtl_<tag>.json.

--cascade T1 T2: the cascade firmware (firmware -DCASCADE, sim/soc_cascade.cpp).
Records are validation live clips (keyword and other words) and 30 s pieces
of the validation negatives stream around stage-1 proposals (--frames: the
verifier_cascade.py cache); every request is compared with
verifier_cascade.cascade_requests (detection bits, verifier scores), and a
detection must light the LED. Writes results/rtl_cascade_<tag>.csv and
results/verification_cascade_rtl_<tag>.json.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

import keyword_config as K
from export_model import export_stream
from verifier_export import export, load_quantized, oracle, records
from verify import command

ROOT = Path(__file__).resolve().parent
RTL = ROOT.parent / 'pynqz2_riscv_flow' / 'rtl'
STAGE1_W20 = Path(r'C:\Users\matut\FULL_AI\5XPG0\code\snn_keyword\runs_stream') / \
    {'yes': 's2_nokd_qat/int_last_w20.npz', 'sheila': 'sheila_qat_seed2/int_model.npz'}[K.KEYWORD]


def section_sizes(mapfile):
    txt = Path(mapfile).read_text()
    out = {}
    for sec in ('.text', '.data', '.bss', '.model'):
        m = re.search(rf'^\{sec}\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)', txt, re.M)
        out[sec] = int(m.group(2), 16) if m else 0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--stage1', type=Path, default=STAGE1_W20)
    p.add_argument('--frames', type=Path, help='cache npz (verifier_cascade.py cache): live clips as records')
    p.add_argument('--records', type=int, default=12)
    p.add_argument('--rv32im', action='store_true', help='no kdot, no engine')
    p.add_argument('--tag', default='')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--boundary', type=int, default=10)
    p.add_argument('--cascade', type=int, nargs=2, metavar=('T1', 'T2'),
                   help='cascade firmware: stage-1 threshold T1 and verifier threshold T2 (needs --frames)')
    p.add_argument('--neg-records', type=int, default=8, help='--cascade: 30 s pieces of the negatives stream')
    a = p.parse_args()
    if a.cascade:
        return cascade_main(a)
    tag = ('rv32im' if a.rv32im else 'engine') + (f'_{a.tag}' if a.tag else '')
    q = load_quantized(a.checkpoint)
    info = export(q, ROOT / 'build/verifier', warmup=a.warmup, boundary=a.boundary)
    export_stream(a.stage1, ROOT / 'build/stream')
    command(['make', '-C', 'firmware', 'verifier'])
    fw = f"build/keyword_stream_verifier{'' if a.rv32im else '_engine'}"
    command(['bash', '-lc', ' '.join([
        'verilator --cc --exe --build -j 8 -Wno-fatal --top-module spike_soc', '--Mdir build/obj_verifier',
        "-CFLAGS '-O3'", '../pynqz2_riscv_flow/rtl/spike_soc.v', '../pynqz2_riscv_flow/rtl/picorv32.v',
        '../pynqz2_riscv_flow/rtl/poisson.v', '../pynqz2_riscv_flow/rtl/kdot_pcpi.v',
        '../pynqz2_riscv_flow/rtl/neuron_engine.v', '"$PWD/sim/soc_verifier.cpp"'])])
    if a.frames:
        c = np.load(a.frames)
        off = c['live_off']
        rng = np.random.default_rng(0)
        pick = rng.choice(len(off) - 1, a.records, replace=False)
        seqs = [c['live_frames'][off[i]:off[i + 1]] for i in pick]
    else:
        rng = np.random.default_rng(0)
        seqs = [rng.integers(0, 256, (int(rng.integers(100, 260)), 24)).astype(np.uint8) for _ in range(a.records)]
    wins = np.stack([np.concatenate([np.zeros((150, 24), np.uint8), s])[-150:] for s in seqs])
    lg, res = oracle(wins, q, a.warmup, a.boundary)
    flat = lg.reshape(len(wins), -1)
    hashes = (flat.astype(np.int64) * np.arange(1, flat.shape[1] + 1)).sum(1) & 0xffffffff
    exp = np.c_[res, hashes.astype(np.uint32).view(np.int32)].astype('<i4')
    (ROOT / 'build/verifier_rtl_in.bin').write_bytes(records(seqs))
    exp.tofile(ROOT / 'build/verifier_rtl_expected.bin')
    t0 = time.perf_counter()
    r = command([f'./build/obj_verifier/Vspike_soc', fw + '.bin', 'build/verifier_rtl_in.bin',
                 'build/verifier_rtl_expected.bin', f'results/rtl_verifier_{tag}.csv', 25], capture_output=True, text=True)
    print(r.stdout.strip())
    rows = np.genfromtxt(ROOT / f'results/rtl_verifier_{tag}.csv', delimiter=',', names=True, dtype=None, encoding=None)
    cyc = rows['cycles'].astype(np.int64)
    sizes = section_sizes(ROOT / (fw + '.map'))
    summary = {'checkpoint': a.checkpoint.as_posix(), 'firmware': fw + '.bin', 'records': len(seqs),
               'bit_exact': 'PASS' in r.stdout, 'cycles': {'max': int(cyc.max()), 'mean': float(cyc.mean()),
                                                          'min': int(cyc.min())},
               'ms_at_100MHz': float(cyc.max()) / 1e5, 'verifier_export': info,
               'sections_bytes': sizes, 'model_region_used': f"{sizes['.model']} of {144 * 1024}",
               'code_region_used': f"{sizes['.text'] + sizes['.data'] + sizes['.bss']} of {64 * 1024}",
               'simulation_seconds': round(time.perf_counter() - t0, 1), 'harness': r.stdout.strip()}
    (ROOT / f'results/verification_verifier_rtl_{tag}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'harness'}, indent=2))


def cascade_main(a):
    from verifier_cascade import cascade_requests
    from verifier_model import NEG
    t1, t2 = a.cascade
    tag = 'engine' + (f'_{a.tag}' if a.tag else '')
    qv = load_quantized(a.checkpoint)
    q1 = dict(np.load(a.stage1))
    window = int(q1.get('decision_window', 1))
    info = export(qv, ROOT / 'build/verifier', warmup=a.warmup, boundary=a.boundary, thresholds=(t2, NEG), cascade_t1=t1)
    export_stream(a.stage1, ROOT / 'build/stream')
    command(['make', '-C', 'firmware', 'cascade'])
    fw = 'build/keyword_stream_cascade_engine'
    command(['bash', '-lc', ' '.join([
        'verilator --cc --exe --build -j 8 -Wno-fatal --top-module spike_soc', '--Mdir build/obj_cascade',
        "-CFLAGS '-O3'", '../pynqz2_riscv_flow/rtl/spike_soc.v', '../pynqz2_riscv_flow/rtl/picorv32.v',
        '../pynqz2_riscv_flow/rtl/poisson.v', '../pynqz2_riscv_flow/rtl/kdot_pcpi.v',
        '../pynqz2_riscv_flow/rtl/neuron_engine.v', '"$PWD/sim/soc_cascade.cpp"'])])
    if not a.frames:
        raise SystemExit('--cascade needs --frames (the verifier_cascade.py cache)')
    c = np.load(a.frames)
    rng = np.random.default_rng(0)
    off, y = c['live_off'], c['live_y']
    pick = np.r_[rng.choice(np.flatnonzero(y == 1), a.records // 2, replace=False),
                 rng.choice(np.flatnonzero(y == 0), a.records - a.records // 2, replace=False)]
    seqs = [c['live_frames'][off[i]:off[i + 1]] for i in pick]
    kinds = ['keyword' if y[i] else 'other word' for i in pick]
    # Negatives: 30 s pieces that start 10 s before a stage-1 proposal (d >= t1) of the cached stream.
    noff, nraw, nfr = c['neg_off'], c['neg_raw'], c['neg_frames']
    from model import decision_scores
    starts = []
    for i in range(len(noff) - 1):
        d = decision_scores(nraw[noff[i]:noff[i + 1]].astype(np.int64), window)
        hits = np.flatnonzero(d >= t1)
        if len(hits):
            starts.append((i, max(0, int(hits[len(hits) // 2]) - 1000)))
    for i, st in [starts[j] for j in rng.choice(len(starts), min(a.neg_records, len(starts)), replace=False)]:
        seqs.append(nfr[noff[i] + st:min(noff[i + 1], noff[i] + st + 3000)])
        kinds.append('negatives stream')
    exp, per = [], []
    for sq in seqs:
        r = cascade_requests(sq, q1, qv, t1, t2, window, a.warmup, a.boundary)
        per.append(r)
        exp += [w for req in r for w in req]
    (ROOT / 'build/cascade_rtl_in.bin').write_bytes(records(seqs))
    np.array(exp, '<i4').tofile(ROOT / 'build/cascade_rtl_expected.bin')
    t0 = time.perf_counter()
    r = command([f'./build/obj_cascade/Vspike_soc', fw + '.bin', 'build/cascade_rtl_in.bin',
                 'build/cascade_rtl_expected.bin', f'results/rtl_cascade_{tag}.csv', 25], capture_output=True, text=True)
    print(r.stdout.strip())
    rows = np.genfromtxt(ROOT / f'results/rtl_cascade_{tag}.csv', delimiter=',', names=True, dtype=None, encoding=None)
    cyc = rows['cycles'].astype(np.int64)
    ran = (rows['detected'].astype(np.int64) & 4) > 0
    sizes = section_sizes(ROOT / (fw + '.map'))
    summary = {'checkpoint': a.checkpoint.as_posix(), 'stage1': a.stage1.as_posix(), 'firmware': fw + '.bin',
               't1': t1, 't2': t2, 'window': window,
               'records': {k: kinds.count(k) for k in dict.fromkeys(kinds)}, 'requests': len(rows),
               'bit_exact': 'PASS' in r.stdout, 'verifier_runs': int(ran.sum()),
               'detections': int(((rows['detected'].astype(np.int64) & 1) > 0).sum()),
               'detections_by_kind': {k: sum(any(b & 1 for b, _, _ in p) for p, kk in zip(per, kinds) if kk == k)
                                      for k in dict.fromkeys(kinds)},
               'cycles_request_max': int(cyc.max()), 'cycles_with_verifier_max': int(cyc[ran].max()) if ran.any() else None,
               'ms_at_100MHz_max': float(cyc.max()) / 1e5, 'verifier_export': info,
               'sections_bytes': sizes, 'model_region_used': f"{sizes['.model']} of {144 * 1024}",
               'simulation_seconds': round(time.perf_counter() - t0, 1), 'harness': r.stdout.strip()}
    (ROOT / f'results/verification_cascade_rtl_{tag}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('harness', 'verifier_export')}, indent=2))


if __name__ == '__main__':
    main()
