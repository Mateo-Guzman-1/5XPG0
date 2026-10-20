"""R3 firmware oracle for the cascade (firmware/cascade_main.c command 7) and gate G-O.

Stage 1 = model.integer_forward (board-exact); verifier = the vendored verifier_export.oracle();
the per-request decision = verifier_cascade.cascade_requests with our window margin instead of the streaming
SNN: window j of an item is the 1 s window ending at sample 16000 + 4000 j; its margin counts in the request that
holds frame min(98 + 25 j, n - 1), i.e. request j + 3 (research/tools/r2_cascade.py place()).

Per request k (25 frames, the last one may be partial; firmware frame counter = ends[k] + 1):
  reach = window margin >= t1 (no window: no reach); run = reach or reach of request k - 1;
  detection = run and head >= t2 and >= HOLDOFF frames since the previous detection.

    r3_oracle.py gate      gate G-O: from the R2 test caches (our margins, cached verifier heads) the oracle
                           reproduces the representative-configuration numbers of R2 (live recall and verifier
                           calls per hour on the negatives) for B1 and B2 -> results/R3/phase2/gate_GO.json
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

TOOLS = Path(__file__).resolve().parent
KWS = TOOLS.parents[1]
RES = TOOLS.parent / 'results'
sys.path.insert(0, str(KWS))
from model import integer_forward  # noqa: E402  (code/snn_keyword/model.py: the board-exact stage 1)

os.environ['KWS_KEYWORD'] = 'sheila'
sys.path.append(str(TOOLS / 'mateo_eval'))
import verifier_export  # noqa: E402
from verifier_model import NEG as VNEG  # noqa: E402

FIRST, HOP, VFRAMES, HOLDOFF, WARMUP, BOUNDARY = 98, 25, 150, 100, 5, 10
RAW_NEG = np.iinfo(np.int64).min // 2          # r2 cache: no window at this frame
STAGE1 = RES / 'S3/models/h48_k64_seed2.npz'
VERIFIER = RES / 'R2/verifier/sheila_head_r2_s1.pt'
CACHE = KWS / 'data_verifier/cache_test_sheila_r2_2_m.npz'
VSCORES = KWS / 'data_verifier/vscores_r2_s1_test.npz'
DESIGNS = {'B1': RES / 'R2/B/v1/h48_k64_seed2.json', 'B2': RES / 'R2/B/v1/budget2150/h48_k64_seed2.json'}


def thresholds(code):
    c = json.loads(DESIGNS[code].read_text(encoding='utf-8'))['cascade']
    return int(c['t1']), int(c['t2']), c


def hop_ends(n):
    return np.minimum(np.arange(HOP - 1, n + HOP - 1, HOP), n - 1)


def window_requests(n_frames, n_windows):
    """Request index of every window of an item with n_frames frames."""
    return np.minimum(FIRST + HOP * np.arange(n_windows), n_frames - 1) // HOP


def request_margins(n_frames, margins):
    """Per request: the window margin (maximum if two windows fall into one request), RAW_NEG if none."""
    out = np.full(len(hop_ends(n_frames)), RAW_NEG, np.int64)
    if len(margins):
        np.maximum.at(out, window_requests(n_frames, len(margins)), np.asarray(margins, np.int64))
    return out


def decide(reach, head, ends, t2, holdoff=HOLDOFF):
    """Firmware decision per request. reach: bool per request; head(k): the verifier head of request k (only
    called where the verifier runs). Returns (bits, ran) per request; bits as MB[11]."""
    out, prev, last = [], False, None
    for k, e in enumerate(ends):
        run = bool(reach[k]) or prev
        bits = 2 * int(bool(reach[k])) + 4 * int(run)
        if run and head(k) >= t2 and (last is None or e - last >= holdoff):
            bits |= 1
            last = e
        out.append(bits)
        prev = bool(reach[k])
    return out


def verifier_windows(frames, ends):
    """(len(ends), 150, 24): the buffer at the end of each request (zeros before the first frame)."""
    pad = np.concatenate([np.zeros((VFRAMES, 24), np.uint8), np.asarray(frames, np.uint8)])
    return np.stack([pad[e + 1:e + 1 + VFRAMES] for e in ends]) if len(ends) else np.zeros((0, VFRAMES, 24), np.uint8)


def logits_hash(lg):
    flat = np.asarray(lg, np.int64).ravel()
    return int((flat * np.arange(1, len(flat) + 1, dtype=np.int64)).sum()) & 0xFFFFFFFF


class Oracle:
    """Stage 1 + verifier oracles with the models of the representative configuration."""

    def __init__(self):
        self.q1 = dict(np.load(STAGE1))
        self.qv = verifier_export.load_quantized(VERIFIER)

    def stage1(self, x):
        scores, spikes = integer_forward(np.asarray(x, np.uint8).reshape(-1, 768), self.q1)
        return scores.astype(np.int64), spikes.astype(np.int64)

    def verifier(self, wins):
        """Per window: (score_a, end_a, score_b, end_b, head, end_head) and the logits hash."""
        if not len(wins):
            return np.zeros((0, 6), np.int64), []
        lg, res = verifier_export.oracle(np.asarray(wins, np.uint8), self.qv, WARMUP, BOUNDARY)
        return np.asarray(res, np.int64), [logits_hash(x) for x in lg]

    def cascade(self, frames, xwin, t1, t2, wreq=None):
        """Command-7 requests of one item after a reset (opcode 5): frames (n, 24), its windows (W, 768).
        wreq: request index of each window (default: the item's own placement, window_requests).
        Returns a list of dicts: payload pieces and the expected mailbox words."""
        frames = np.asarray(frames, np.uint8)
        n = len(frames)
        ends = hop_ends(n)
        scores, spikes = self.stage1(xwin) if len(xwin) else (np.zeros((0, 2), np.int64), np.zeros(0, np.int64))
        wreq = window_requests(n, len(xwin)) if wreq is None else np.asarray(wreq)
        if len(set(wreq.tolist())) != len(wreq):
            raise ValueError('two windows in one request: command 7 carries one window')
        win_of = {int(r): j for j, r in enumerate(wreq)}
        margin = {k: int(scores[j, 1] - scores[j, 0]) for k, j in win_of.items()}
        reach = [k in margin and margin[k] >= t1 for k in range(len(ends))]
        prev = [False] + reach[:-1]
        run_idx = [k for k in range(len(ends)) if reach[k] or prev[k]]
        res, _ = self.verifier(verifier_windows(frames, ends[run_idx]))
        vres = {k: res[i] for i, k in enumerate(run_idx)}
        bits = decide(reach, lambda k: int(vres[k][4]), ends, t2)
        out, start = [], 0
        for k, e in enumerate(ends):
            j = win_of.get(k)
            v = vres.get(k, np.array([VNEG, -1, VNEG, -1, VNEG, -1]))
            out.append(dict(k=k, window=j, frames=(start, int(e) + 1), margin=margin.get(k),
                            expect={'6': 0, '7': int(scores[j, 0]) if j is not None else 0,
                                    '8': int(scores[j, 1]) if j is not None else 0,
                                    '10': int(spikes[j]) if j is not None else 0, '11': bits[k], '18': int(e) + 1,
                                    '26': int(v[0]), '27': int(v[1]), '28': int(v[2]), '29': int(v[3]), '30': 0,
                                    '32': int(v[4]), '33': int(v[5])}))
            start = int(e) + 1
        return out


# --------------------------------------------------------------------------- gate G-O

def items(c, g):
    off = c[f'{g}_off']
    return [(int(off[i]), int(off[i + 1])) for i in range(len(off) - 1)]


def gate():
    c, v = np.load(CACHE), np.load(VSCORES)
    if Path(str(c['stage1'])).name != STAGE1.name or str(v['score']) != 'head':
        raise SystemExit('caches are not the representative configuration')
    out = {'cache': CACHE.name, 'vscores': VSCORES.name, 'designs': {}}
    hours = float(c['neg_seconds']) / 3600
    groups = {}
    for g in ('live', 'neg'):
        raw, heads, voff = c[f'{g}_raw'], v[f'{g}_va'], v[f'{g}_voff']
        per = []
        for i, (a, b) in enumerate(items(c, g)):
            r = raw[a:b]
            ends = hop_ends(b - a)
            per.append((np.maximum.reduceat(r, np.r_[0, ends[:-1] + 1]), heads[voff[i]:voff[i + 1]], ends))
        groups[g] = per
    y = c['live_y']
    for code in DESIGNS:
        t1, t2, ref = thresholds(code)
        live_hit = []
        for hm, hd, ends in groups['live']:
            bits = decide(hm >= t1, lambda k: hd[k], ends, t2)
            live_hit.append(any(b & 1 for b in bits))
        calls = 0
        for hm, hd, ends in groups['neg']:
            reach = hm >= t1
            calls += int((reach | np.r_[False, reach[:-1]]).sum())
        live = round(float(np.mean(np.array(live_hit)[y == 1])), 4)
        cph = round(calls / hours, 1)
        out['designs'][code] = dict(t1=t1, t2=t2, live_recall=live, live_recall_r2=ref['live_recall'],
                                    calls_per_hour=cph, calls_per_hour_r2=ref['verifier_calls_per_hour'],
                                    equal=live == ref['live_recall'] and cph == ref['verifier_calls_per_hour'])
    out['passed'] = all(d['equal'] for d in out['designs'].values())
    return out


def main():
    if sys.argv[1:] != ['gate']:
        raise SystemExit(__doc__)
    out = gate()
    path = RES / 'R3/phase2/gate_GO.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=1))
    if not out['passed']:
        raise SystemExit('G-O failed (G2a): the oracle does not reproduce R2')


if __name__ == '__main__':
    main()
