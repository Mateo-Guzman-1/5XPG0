"""Cost part of the size benchmark: measure one integer model on the PYNQ-Z2.

1. build_firmware.py builds keyword.bin (RV32IM) and keyword_kdot.bin and reports the
   .model section size (it refuses models over the 144 KiB model memory);
2. scp copies both images and start_board.sh to the board (~/snn_keyword/bench/) and
   their sha256 is compared there;
3. per variant, start_board.sh <variant> bench/<image>.bin loads the matching bitstream
   (base: keyword.bit, kdot: keyword_kdot.bit) and starts a fresh board server;
4. the check_board.py checks run: the fixed input vectors (benchmark_inputs.VECTORS) as
   single windows --repeat times, with scores, spikes and decision compared with this
   model's integer oracle, the core's cycle count and the PC round trip of each request;
   then the "2 of 3" stream patterns.
A variant that is not bit-exact reports no cycles, and the exit status is 1. The demo is
not restored: run run_board.ps1 afterwards.

    bench_board.py research/results/E2/models/h32_seed0.npz --out research/results/E2/board/h32_seed0.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

KWS = Path(__file__).resolve().parents[2]   # code/snn_keyword
sys.path.insert(0, str(KWS))
from benchmark_inputs import VECTORS, describe
from build_firmware import build, shown
from check_board import Client, check_stream, check_vectors
from model import keyword_of

IMAGES = {'base': 'keyword', 'kdot': 'keyword_kdot'}   # release images; also the deploy/ bitstream names
# variant -> (fabric = start_board.sh variant, which picks the bitstream; firmware image).
# The research images (research/PLAN_SYNAPSE.md) run on the release bitstreams.
VARIANTS = {'base': ('base', 'keyword'), 'kdot': ('kdot', 'keyword_kdot'),
            'packed': ('base', 'keyword_packed'), 'sparse': ('base', 'keyword_sparse'),
            'lut': ('base', 'keyword_lut'), 'kdot_lut': ('kdot', 'keyword_kdot_lut'),
            # S6: fabric 'ksdot' = the experimental sparse-kdot bitstream given with --bitstream
            'ksdot_lut': ('ksdot', 'keyword_ksdot_lut'), 'kdot_lut_on_ksdot': ('ksdot', 'keyword_kdot_lut')}


def call(*args):
    """Run ssh/scp from code/snn_keyword with relative paths (no drive letters for scp to misread)."""
    r = subprocess.run(args, cwd=KWS, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f'{" ".join(args)} failed:\n{r.stdout}{r.stderr}')
    return r.stdout


def milliseconds(seconds):
    ms = 1000 * np.array(seconds)
    return {'mean': round(float(ms.mean()), 2), 'median': round(float(np.median(ms)), 2),
            'p95': round(float(np.percentile(ms, 95)), 2), 'max': round(float(ms.max()), 2)}


def measure(a, variant, x, q):
    """Restart the board server with this variant's image, then run the check_board.py checks."""
    fabric, image = VARIANTS[variant]
    if fabric == 'ksdot':   # experimental bitstream (scp'd by main); the kdot name only picks defaults
        started = call('ssh', a.board, 'bash', 'snn_keyword/start_board.sh', 'kdot', f'bench/{image}.bin',
                       f'bench/{a.bitstream.name}')
    else:
        started = call('ssh', a.board, 'bash', 'snn_keyword/start_board.sh', fabric, f'bench/{image}.bin')
    if 'Listening' not in started:
        raise SystemExit(f'Board server did not start:\n{started}')
    board = Client(a.host, a.port)
    try:
        single = check_vectors(board, x, q, a.repeat)
        round_trips = list(board.rtt)   # single-window requests only
        try:
            stream = check_stream(board, x, q)
        except SystemExit as e:   # every vector on one side of this model's stream threshold
            stream = [{'skipped': str(e), 'pass': None}]
    finally:
        board.sock.close()
    if not single['bit_exact']:   # no bit-exact result, no reported cycles
        for k in ('cycles_min', 'cycles_mean', 'cycles_max', 'milliseconds_max_at_100mhz'):
            single.pop(k)
    return {'fabric': fabric, 'image': image,
            'server': started.strip().splitlines(), 'bit_exact': single['bit_exact'], 'single_window': single,
            'round_trip_ms': milliseconds(round_trips), 'stream': stream,
            'stream_pass': all(r['pass'] for r in stream) if stream[0]['pass'] is not None else None}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', type=Path, help='integer model .npz (with stream_threshold)')
    p.add_argument('--variant', nargs='+', choices=[*VARIANTS, 'both'], default=['both'],
                   help='firmware variants to measure (both = base kdot)')
    p.add_argument('--board', default='pynq', help='ssh host: ~/.ssh/config alias or user@ip')
    p.add_argument('--host', default='10.43.0.1', help='board address for the demo protocol')
    p.add_argument('--port', type=int, default=5556)
    p.add_argument('--vectors', type=Path, default=VECTORS, help='input vectors (key x)')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--out', type=Path, help='write the report as JSON')
    p.add_argument('--build-name', help='firmware folder under build/bench (default: the model file name)')
    p.add_argument('--bitstream', type=Path, help='experimental bitstream for the ksdot variants (research S6)')
    a = p.parse_args()
    q = dict(np.load(a.model))
    x = np.load(a.vectors)['x']
    variants = list(dict.fromkeys(v for x in a.variant for v in (['base', 'kdot'] if x == 'both' else [x])))
    images = [VARIANTS[v][1] for v in variants]
    folder = KWS / 'build/bench' / (a.build_name or a.model.stem)
    firmware = build(a.model, folder, images)
    rel = os.path.relpath(folder, KWS).replace('\\', '/')
    ksdot = any(VARIANTS[v][0] == 'ksdot' for v in variants)
    if ksdot and not (a.bitstream and a.bitstream.is_file()):
        raise SystemExit('the ksdot variants need --bitstream <keyword_ksdot.bit>')
    call('ssh', a.board, 'mkdir', '-p', 'snn_keyword/bench')
    call('scp', '-q', 'start_board.sh', 'board_server.py', 'protocol.py', f'{a.board}:snn_keyword/')
    call('scp', '-q', *[f'{rel}/{n}.bin' for n in images], f'{a.board}:snn_keyword/bench/')
    extra = []
    if ksdot:
        call('scp', '-q', os.path.relpath(a.bitstream.resolve(), KWS).replace('\\', '/'), f'{a.board}:snn_keyword/bench/')
        extra = [f'snn_keyword/bench/{a.bitstream.name}']
    listing = call('ssh', a.board, 'sha256sum', *[f'snn_keyword/bench/{n}.bin' for n in images], *extra,
                   'snn_keyword/deploy/keyword.bit', 'snn_keyword/deploy/keyword_kdot.bit')
    on_board = {Path(line.split()[1]).name: line.split()[0] for line in listing.splitlines()}
    for n in images:
        if on_board[f'{n}.bin'] != firmware['images'][n]['sha256']:
            raise SystemExit(f'{n}.bin on the board differs from the build')
    if ksdot and on_board[a.bitstream.name] != hashlib.sha256(a.bitstream.read_bytes()).hexdigest():
        raise SystemExit(f'{a.bitstream.name} on the board differs from {a.bitstream}')
    bench = describe(keyword_of(q))
    report = {'benchmark': bench['fingerprint'], 'date': time.strftime('%Y-%m-%d %H:%M'), 'board': a.host,
              'model': shown(a.model), 'model_sha256': hashlib.sha256(a.model.read_bytes()).hexdigest(),
              'keyword': keyword_of(q), 'hidden': int(len(q['w1'])),
              'decision_threshold': int(q['decision_threshold']),
              'stream_threshold': int(q.get('stream_threshold', q['decision_threshold'])),
              'vectors': shown(a.vectors), 'repeat': a.repeat, 'firmware': firmware,
              'experimental_bitstream': ({'file': shown(a.bitstream), 'sha256': on_board[a.bitstream.name]}
                                         if ksdot else None),
              'board_files_sha256': on_board,
              'bitstreams_match_deploy': all(on_board[f'{n}.bit'] == bench['inputs'][f'bitstream_{v}']['sha256']
                                             for v, n in IMAGES.items()),
              'variants': {}}
    for v in variants:
        r = report['variants'][v] = measure(a, v, x, q)
        s = r['single_window']
        print(f'{a.model.name} {v}: {s["requests"]} requests, {len(s["mismatches"])} mismatches; '
              + (f'cycles max {s["cycles_max"]:,} ({s["milliseconds_max_at_100mhz"]:.2f} ms); ' if r['bit_exact'] else '')
              + f'round trip mean {r["round_trip_ms"]["mean"]} ms; stream {r["stream_pass"]}', flush=True)
    report['pass'] = report['bitstreams_match_deploy'] and all(
        r['bit_exact'] and r['stream_pass'] is not False for r in report['variants'].values())
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
