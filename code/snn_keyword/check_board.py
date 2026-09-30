"""Check the PicoRV32 on the board against the integer oracle, over Ethernet.

Needs a running board server (run_board.ps1). Checks:
  vectors  the verification vectors as single windows (--repeat times): scores,
           hidden spikes and decision must equal model.integer_forward for the
           model whose firmware runs on the board; reports the core's cycles
  stream   "2 of 3" confirmation, windows 250 ms apart (JOURNAL.md entry 8):
           + -> 2; -,-,+ -> 0,0,2; +,+,-,+ -> 2,3,0,3; -,+,-,+ -> 0,2,0,3
           (detected bit0 = confirmed, bit1 = this window alone)
  --wavs   WAV clips centred in 2.5 s of silence and streamed like the live demo
           (1 s windows, 250 ms hop, real pacing): per clip, which windows passed
           and whether a confirmed detection happened; every window bit-exact
Defaults: deploy/model.npz and the verification vectors of the deployed release
(the "evidence" folder in deploy/manifest.json). Exit status 1 on any difference.
"""
import argparse
import hashlib
import json
import socket
import time
from pathlib import Path
import numpy as np
from features import SAMPLE_RATE, features, read_wav
from model import integer_forward, keyword_of
from protocol import MODE_SINGLE, MODE_STREAM, request

ROOT = Path(__file__).resolve().parent
HOP = .25   # seconds between stream windows, as in the live demo
STREAM_PATTERNS = {'+': [2], '--+': [0, 0, 2], '++-+': [2, 3, 0, 3], '-+-+': [0, 2, 0, 3]}


def release_vectors():
    """Verification vectors of the deployed model (deploy/manifest.json names their folder)."""
    manifest = json.loads((ROOT / 'deploy/manifest.json').read_text())
    return ROOT / manifest.get('evidence', 'results') / 'verification_vectors.npz'


def shown(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT.parents[1]).as_posix() if path.is_relative_to(ROOT.parents[1]) else path.as_posix()


class Client:
    """One connection to board_server.py; numbers the requests and times each round trip."""
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.seq, self.rtt = 0, []

    def __call__(self, frame, mode=MODE_SINGLE):
        self.seq += 1
        start = time.perf_counter()
        reply = request(self.sock, self.seq, frame, mode)
        self.rtt.append(time.perf_counter() - start)
        return reply


def paced(frames, send):
    """Send frames HOP seconds apart after letting older stream history expire (> 750 ms)."""
    time.sleep(1)
    start = time.monotonic()
    replies = []
    for k, frame in enumerate(frames):
        time.sleep(max(0, start + k * HOP - time.monotonic()))
        replies.append(send(frame, MODE_STREAM))
    return replies


def check_vectors(board, x, q, repeat):
    scores, spikes = integer_forward(x, q)
    threshold = int(q['decision_threshold'])
    mismatches, cycles = [], []
    for r in range(repeat):
        for i, v in enumerate(x):
            got = board(v)
            want = dict(score0=int(scores[i, 0]), score1=int(scores[i, 1]), spikes=int(spikes[i]),
                        detected=bool(scores[i, 1] - scores[i, 0] >= threshold))
            if any(got[k] != want[k] for k in want):
                mismatches.append({'repeat': r, 'vector': i, 'board': {k: got[k] for k in want}, 'oracle': want})
            cycles.append(got['cycles'])
    cycles = np.array(cycles)
    return {'requests': len(cycles), 'mismatches': mismatches, 'bit_exact': not mismatches,
            'cycles_min': int(cycles.min()), 'cycles_mean': round(float(cycles.mean()), 1),
            'cycles_max': int(cycles.max()), 'milliseconds_max_at_100mhz': float(cycles.max() / 1e5)}


def check_stream(board, x, q):
    s = integer_forward(x, q)[0]
    margin = s[:, 1] - s[:, 0]
    threshold = int(q.get('stream_threshold', q['decision_threshold']))
    if margin.max() < threshold or margin.min() >= threshold:
        raise SystemExit('Stream check needs vectors above and below the stream threshold')
    frames = {'+': x[int(np.argmax(margin))], '-': x[int(np.argmin(margin))]}
    rows = []
    for pattern, want in STREAM_PATTERNS.items():
        got = [int(r['detected']) | int(r['window']) << 1 for r in paced([frames[c] for c in pattern], board)]
        rows.append({'pattern': pattern, 'expected': want, 'board': got, 'pass': got == want})
    return rows


def stream_wavs(board, paths, q):
    rows = []
    for path in paths:
        a = read_wav(path)[:int(1.4 * SAMPLE_RATE)]
        buf = np.zeros(int(2.5 * SAMPLE_RATE), np.float32)
        off = (len(buf) - len(a)) // 2
        buf[off:off + len(a)] = a
        hop = int(HOP * SAMPLE_RATE)
        x = np.stack([features(buf[s:s + SAMPLE_RATE]) for s in range(0, len(buf) - SAMPLE_RATE + 1, hop)])
        scores, spikes = integer_forward(x, q)
        replies = paced(x, board)
        exact = [(r['score0'], r['score1'], r['spikes']) == (int(s[0]), int(s[1]), int(k))
                 for r, s, k in zip(replies, scores, spikes)]
        rows.append({'file': path.name, 'windows': ''.join('+' if r['window'] else '-' for r in replies),
                     'confirmed': any(r['detected'] for r in replies), 'mismatches': exact.count(False)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('host', nargs='?', default='10.43.0.1')
    p.add_argument('--port', type=int, default=5556)
    p.add_argument('--model', type=Path, default=ROOT / 'deploy/model.npz', help='model whose firmware runs on the board')
    p.add_argument('--vectors', type=Path, help='default: verification vectors of the deployed release')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--wavs', type=Path, nargs='*', default=[], help='WAV files or folders to stream like the live demo')
    p.add_argument('--label', default='', help='stored in the report, e.g. kdot or base')
    p.add_argument('--out', type=Path, help='write the report as JSON')
    a = p.parse_args()
    q = dict(np.load(a.model))
    vectors = a.vectors or release_vectors()
    x = np.load(vectors)['x']
    wavs = [f for w in a.wavs for f in (sorted(w.glob('*.wav')) if w.is_dir() else [w])]
    board = Client(a.host, a.port)
    report = {'board': a.host, 'label': a.label, 'date': time.strftime('%Y-%m-%d'), 'model': shown(a.model),
              'model_sha256': hashlib.sha256(a.model.read_bytes()).hexdigest(), 'keyword': keyword_of(q),
              'decision_threshold': int(q['decision_threshold']),
              'stream_threshold': int(q.get('stream_threshold', q['decision_threshold'])),
              'vectors': shown(vectors), 'single_window': check_vectors(board, x, q, a.repeat),
              'stream': check_stream(board, x, q)}
    if wavs:
        report['wavs'] = stream_wavs(board, wavs, q)
    report['round_trip_ms_mean'] = round(1000 * float(np.mean(board.rtt)), 2)
    report['pass'] = (report['single_window']['bit_exact'] and all(r['pass'] for r in report['stream'])
                      and not any(r['mismatches'] for r in report.get('wavs', [])))
    v = report['single_window']
    print(f'Model {report["model"]} (keyword "{report["keyword"]}") on {a.host} {a.label}'.rstrip())
    print(f'Vectors: {v["requests"]} requests, {len(v["mismatches"])} mismatches; cycles min/mean/max '
          f'{v["cycles_min"]:,} / {v["cycles_mean"]:,.0f} / {v["cycles_max"]:,} ({v["milliseconds_max_at_100mhz"]:.2f} ms max)')
    print('Stream:', '; '.join(f'{r["pattern"]} -> {r["board"]} {"ok" if r["pass"] else "EXPECTED " + str(r["expected"])}'
                               for r in report['stream']))
    for r in report.get('wavs', []):
        print(f'  {r["file"]}: windows {r["windows"]}, {"confirmed" if r["confirmed"] else "not confirmed"}'
              + (f', {r["mismatches"]} MISMATCHES' if r['mismatches'] else ''))
    print(f'Round trip {report["round_trip_ms_mean"]} ms mean; {"PASS" if report["pass"] else "FAIL"}')
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
