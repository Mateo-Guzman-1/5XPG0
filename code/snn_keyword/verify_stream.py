"""Phase 4 check: native C streaming kernel against the integer oracle, bit for bit.

Exports the integer model to C headers (export_model.export_stream), builds
sim/native_stream.c with firmware/stream_infer.c (gcc in WSL on Windows, as
verify.py), and runs both on the same frame sequences:
  - every clean live test clip of robust_eval.py (Speech Commands test keyword
    and 3000 other words in noise, 2.25 s each);
  - the one-hour test stream (Speech Commands words, LibriSpeech, keyword).
Scores and layer-1 / layer-2 spike counts must match on every frame.
Writes results/verification_stream.json, with synaptic events per frame
(the event-driven work of the kernel) for the activity measurements.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from export_model import export_stream
from features import frame_features
from model import integer_forward_stream
from robust_eval import bg_noise, build_stream, place_clips, sc_live_clips, SPLITS
from verify import command

ROOT = Path(__file__).resolve().parent


def records(seqs):
    out = bytearray()
    for x in seqs:
        out += np.uint32(len(x)).tobytes() + np.ascontiguousarray(x, np.uint8).tobytes()
    return bytes(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--stream-seconds', type=int, default=3600)
    p.add_argument('--out', type=Path, default=ROOT / 'results/verification_stream.json')
    a = p.parse_args()
    q = dict(np.load(a.model))
    build = ROOT / 'build/stream'
    synapses = export_stream(a.model, build)
    command(['gcc', '-O2', '-Wall', '-Wextra', '-Werror', '-Ibuild/stream', '-Ifirmware', 'sim/native_stream.c',
             'firmware/stream_infer.c', '-o', 'build/native_stream'])
    frontend = str(q.get('frontend', 'logmel'))
    cfg = SPLITS['test']
    clips, y, rng = sc_live_clips(a.data, cfg['sc'], seed=cfg['seed'])
    live = [frame_features(w, frontend) for w in place_clips(clips, bg_noise(a.data), rng)]
    audio, marks, _ = build_stream(a.data, 'test', a.stream_seconds)
    seqs = live + [frame_features(audio, frontend)]
    (build / 'input.bin').write_bytes(records(seqs))
    start = time.perf_counter()
    with open(build / 'input.bin', 'rb') as inp, open(build / 'actual.bin', 'wb') as out:
        command(['./build/native_stream'], stdin=inp, stdout=out)
    native_s = time.perf_counter() - start
    actual = np.fromfile(build / 'actual.bin', '<i4').reshape(-1, 4)
    # Oracle: live clips in batches (same length), the stream alone.
    start = time.perf_counter()
    expected = []
    for i in range(0, len(live), 256):
        part = live[i:i + 256]
        x = np.zeros((len(part), max(map(len, part)), 24), np.uint8)
        for j, f in enumerate(part):
            x[j, :len(f)] = f  # causal: padding after a clip cannot change its frames
        s, _, sp = integer_forward_stream(x, q)
        expected += [np.c_[s[j, :len(f)], sp[j, :len(f)]] for j, f in enumerate(part)]
    s, _, sp = integer_forward_stream(seqs[-1][None], q)
    expected.append(np.c_[s[0], sp[0]])
    oracle_s = time.perf_counter() - start
    expected = np.concatenate(expected)
    frames = len(expected)
    mismatch = int((actual[:, :3] != expected).any(1).sum()) if len(actual) == frames else -1
    events = actual[:, 3]
    report = {'model': a.model.as_posix(), 'sequences': len(seqs), 'frames': frames,
              'live_clips': len(live), 'stream_seconds': a.stream_seconds,
              'frame_mismatches': mismatch, 'bit_exact': mismatch == 0,
              'synapses_nonzero': synapses,
              'per_frame': {'layer1_spikes_mean': float(actual[:, 1].mean()), 'layer2_spikes_mean': float(actual[:, 2].mean()),
                            'synaptic_events_mean': float(events.mean()), 'synaptic_events_max': int(events.max()),
                            'synaptic_events_p99': float(np.percentile(events, 99))},
              'seconds': {'native_including_wsl': round(native_s, 1), 'oracle': round(oracle_s, 1)}}
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if mismatch != 0:
        raise SystemExit('native C does not match the oracle')


if __name__ == '__main__':
    main()
