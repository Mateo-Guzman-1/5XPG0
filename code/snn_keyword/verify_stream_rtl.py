"""Phase 5 check: streaming firmware on the PicoRV32 SoC RTL (Verilator), bit-exact with the oracle.

Builds build/stream headers (export_model.export_stream), the ABI v3 firmware
(firmware/stream_main.c; --kdot uses the kdot coprocessor for layer 1), and
sim/soc_stream.cpp against rtl/spike_soc.v. --read-wait N simulates a copy of
spike_soc.v with READ_WAIT = N (the bus fix, report E2) without changing the
source. Streams: 20 keyword and 20 other test clips placed in noise as in
robust_eval.py, sent in 250 ms hops (25 frames). Writes results/rtl_stream*.csv
and a JSON summary with cycles per hop.
"""
import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

from export_model import export_stream
from features import frame_features
from model import integer_forward_stream
from robust_eval import SPLITS, bg_noise, place_clips, sc_live_clips
from verify import command
from verify_stream import records

ROOT = Path(__file__).resolve().parent
RTL = ROOT.parent / 'pynqz2_riscv_flow' / 'rtl'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--kdot', action='store_true')
    p.add_argument('--engine', action='store_true', help='layer 2 in rtl/neuron_engine.v (implies kdot)')
    p.add_argument('--read-wait', type=int, default=1)
    p.add_argument('--streams', type=int, default=40)
    p.add_argument('--hop', type=int, default=25)
    p.add_argument('--tag', default='', help='suffix of the result files, so other models keep theirs')
    a = p.parse_args()
    tag = f"{'engine' if a.engine else 'kdot' if a.kdot else 'rv32im'}_rw{a.read_wait}"
    out_tag = tag + (f'_{a.tag}' if a.tag else '')
    q = dict(np.load(a.model))
    export_stream(a.model, ROOT / 'build/stream')
    command(['make', '-C', 'firmware', 'stream'])
    fw = f"build/keyword_stream{'_engine' if a.engine else '_kdot' if a.kdot else ''}.bin"
    # RTL copy with the requested read wait.
    src = (RTL / 'spike_soc.v').read_text()
    src, n = re.subn(r"localparam \[3:0\] READ_WAIT = 4'd\d+;", f"localparam [3:0] READ_WAIT = 4'd{a.read_wait};", src)
    assert n == 1
    rtl_dir = ROOT / f'build/rtl_{tag}'
    rtl_dir.mkdir(parents=True, exist_ok=True)
    (rtl_dir / 'spike_soc.v').write_text(src)
    rel = lambda path: path.relative_to(ROOT.parent.parent).as_posix()
    command(['bash', '-lc', ' '.join([
        'verilator --cc --exe --build -j 8 -Wno-fatal --top-module spike_soc',
        f'--Mdir build/obj_stream_{tag}', "-CFLAGS '-O3'",
        f'build/rtl_{tag}/spike_soc.v', '../pynqz2_riscv_flow/rtl/picorv32.v', '../pynqz2_riscv_flow/rtl/poisson.v',
        '../pynqz2_riscv_flow/rtl/kdot_pcpi.v', '../pynqz2_riscv_flow/rtl/neuron_engine.v',
        '"$PWD/sim/soc_stream.cpp"'])])
    # Vectors: keyword clips first, then others, all from the live test set.
    cfg = SPLITS['test']
    clips, y, rng = sc_live_clips(a.data, cfg['sc'], seed=cfg['seed'])
    audio = place_clips(clips, bg_noise(a.data), rng)
    pick = list(np.flatnonzero(y == 1)[:a.streams // 2]) + list(np.flatnonzero(y == 0)[:a.streams - a.streams // 2])
    seqs = [frame_features(audio[i], str(q.get('frontend', 'logmel'))) for i in pick]
    (ROOT / 'build/stream_rtl_in.bin').write_bytes(records(seqs))
    expected = []
    for x in seqs:
        s, _, sp = integer_forward_stream(x[None], q)
        expected.append(np.c_[s[0], sp[0]].astype('<i4'))
    np.concatenate(expected).tofile(ROOT / 'build/stream_rtl_expected.bin')
    csv = ROOT / f'results/rtl_stream_{out_tag}.csv'
    start = time.perf_counter()
    res = command([f'./build/obj_stream_{tag}/Vspike_soc', fw, 'build/stream_rtl_in.bin', 'build/stream_rtl_expected.bin',
                   f'results/rtl_stream_{out_tag}.csv', a.hop], capture_output=True, text=True)
    print(res.stdout.strip())
    rows = np.genfromtxt(csv, delimiter=',', names=True)
    full = rows[rows['frames'] == a.hop]
    summary = {'model': a.model.as_posix(), 'firmware': fw, 'read_wait': a.read_wait, 'kdot': a.kdot,
               'streams': len(seqs), 'hops': len(rows), 'bit_exact': 'PASS' in res.stdout,
               'cycles_per_hop': {'max': int(full['cycles'].max()), 'mean': float(full['cycles'].mean()),
                                  'p99': float(np.percentile(full['cycles'], 99))},
               'ms_per_hop_at_100MHz': {'max': float(full['cycles'].max()) / 1e5, 'mean': float(full['cycles'].mean()) / 1e5},
               'events_per_hop': {'max': int(full['events'].max()), 'mean': float(full['events'].mean())},
               'simulation_seconds': round(time.perf_counter() - start, 1), 'harness': res.stdout.strip()}
    out = ROOT / f'results/verification_stream_rtl_{out_tag}.json'
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'harness'}, indent=2))


if __name__ == '__main__':
    main()
