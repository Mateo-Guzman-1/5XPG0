"""Write the streaming (ABI v3) board vectors for jtag/stream_board_test.tcl.

Same 40 streams as verify_stream_rtl.py (the first 20 keyword and 20 other live
test clips placed in noise as in robust_eval.py), split into hops of --hop
frames. For every hop the expected best score, last score, spikes (layer 1 +
layer 2) and detection (bit0 with the firmware's 100-frame hold-off, bit1 any
frame over the threshold, frame of the detection) come from the integer oracle.
net_board_test.py uses the same streams and expectations over TCP.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import frame_features
from model import decision_scores, integer_forward_stream
from robust_eval import SPLITS, bg_noise, place_clips, sc_live_clips

HOLDOFF = 100


def expected_hops(x, q, hop):
    """Per hop of `hop` frames: start, frames, best, last, spikes, detected (bits), at (frame or 0xffffffff)."""
    s, _, sp = integer_forward_stream(x[None], q)
    s, sp = s[0], sp[0].sum(1)
    d = decision_scores(s, int(q.get('decision_window', 1)))   # firmware: sum of the last w scores
    threshold = int(q['stream_threshold'])
    have, last_event, out = False, 0, []
    for start in range(0, len(x), hop):
        k = min(hop, len(x) - start)
        detected, at = 0, 0xffffffff
        for f in range(start, start + k):
            if d[f] >= threshold:
                detected |= 2
                if not have or f - last_event >= HOLDOFF:
                    have, last_event = True, f
                    if not detected & 1:
                        at = f - start
                    detected |= 1
        out.append(dict(start=start, frames=k, best=int(s[start:start + k].max()), last=int(s[start + k - 1]),
                        spikes=int(sp[start:start + k].sum()), detected=detected, at=at))
    return out


def test_streams(q, data, streams=40):
    """Feature frames (uint8, T x 24) of verify_stream_rtl.py's streams: first keyword clips, then others."""
    cfg = SPLITS['test']
    clips, y, rng = sc_live_clips(data, cfg['sc'], seed=cfg['seed'])
    audio = place_clips(clips, bg_noise(data), rng)
    pick = list(np.flatnonzero(y == 1)[:streams // 2]) + list(np.flatnonzero(y == 0)[:streams - streams // 2])
    return [np.ascontiguousarray(frame_features(audio[i], str(q.get('frontend', 'logmel'))), np.uint8) for i in pick]


def hops(x, q, hop):
    out = []
    for h in expected_hops(x, q, hop):
        words = ' '.join(f'0x{n:08x}' for n in np.frombuffer(x[h['start']:h['start'] + h['frames']].tobytes(), '<u4'))
        out.append(f"{{{h['frames']} {h['best']} {h['last']} {h['spikes']} {h['detected']} {h['at']} {{{words}}}}}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, default=ROOT / 'results/models/sheila_stream_int8.npz')
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--streams', type=int, default=40)
    p.add_argument('--hop', type=int, default=25)
    p.add_argument('--out', type=Path, default=ROOT / 'build/stream_board_vectors.tcl')
    a = p.parse_args()
    q = dict(np.load(a.model))
    lines = [f'set threshold {int(q["stream_threshold"])}', 'set streams {']
    n = 0
    seqs = test_streams(q, a.data, a.streams)
    for x in seqs:
        h = hops(x, q, a.hop)
        n += len(h)
        lines.append('{\n' + '\n'.join(h) + '\n}')
    lines.append('}')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(('\n'.join(lines) + '\n').encode())
    print(f'{len(seqs)} streams, {n} hops, threshold {int(q["stream_threshold"])} -> {a.out}')


if __name__ == '__main__':
    main()
