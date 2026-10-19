"""R2 step B3 (research/PLAN_R2_GRU.md): hard negatives for the verifier, mined with OUR stage 1.

Adapted from Mateo's mine_verifier_negatives.py (tools/mateo_eval/), which mines with his streaming SNN.
Here the proposer is our H48 K64 window SNN (seeds 0-2, variant chosen in A2/A5: the single-window margin):
  - every LibriSpeech train-clean-100 utterance of data_verifier/libri100_audio.npy (verifier_data.pack_libri;
    utterances whose transcript holds the keyword, a word that begins with it or another spelling are dropped,
    as Mateo's) is cut into 1 s windows every 250 ms from its start; utterances shorter than 1 s are skipped;
  - window margin = req_eval.fast_margin (integer model, exact);
  - peaks: maxima with no higher margin within 1 s (4 windows), at most 3 per utterance, the 8000 highest per seed;
  - offset = utterance start + 16000 + 4000 k: the end of window k, in libri100_audio samples (Mateo's offset is the
    end of the peak frame); train_verifier.HardWindows reads only `offset`;
  - the union of the three seeds, keeping the higher-ranked of two offsets within +-0.5 s of each other.
Output: data_verifier/hard_sheila_r2.npz (git-ignored) and research/results/R2/B/mining.json (summary).

Usage (from code/snn_keyword): python research/tools/r2_mine.py
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

os.environ['KWS_KEYWORD'] = 'sheila'
TOOLS = Path(__file__).resolve().parent
KWS = TOOLS.parents[1]
sys.path[:0] = [str(TOOLS / 'mateo_eval'), str(TOOLS)]
import keyword_config as K  # noqa: E402
import req_eval as RE  # noqa: E402

DV = KWS / 'data_verifier'
SR, HOP = 16000, 4000


def peaks(score, min_gap):
    """Indices of maxima with no higher value within min_gap windows (greedy, highest first; Mateo's peaks())."""
    order = np.argsort(score)[::-1]
    taken = np.zeros(len(score), bool)
    out = []
    for i in order:
        if taken[max(0, i - min_gap):i + min_gap + 1].any():
            continue
        taken[i] = True
        out.append(i)
    return np.array(out, np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--keep', type=int, default=8000, help='peaks kept per seed')
    p.add_argument('--per-utt', type=int, default=3)
    p.add_argument('--dedup-s', type=float, default=.5)
    a = p.parse_args()
    qs = {s: dict(np.load(KWS / f'research/results/S3/models/h48_k64_seed{s}.npz')) for s in a.seeds}
    ix = np.load(DV / 'libri100_index.npz')
    audio = np.load(DV / 'libri100_audio.npy', mmap_mode='r')
    start, length, text, uid = ix['start'], ix['length'], ix['text'], ix['uid']
    keep_utt = np.array([not any(K.prefixed(w) or w.lower() == K.KEYWORD or w.lower() in K.ALIASES
                                 for w in t.split()) for t in text]) & (length >= SR)
    utts = np.flatnonzero(keep_utt)
    rows = {s: [] for s in a.seeds}
    t0 = time.perf_counter()

    def score(u):
        x = RE.windows(audio[start[u]:start[u] + length[u]].astype(np.float32) / 32768)
        return u, {s: RE.fast_margin(x, q) for s, q in qs.items()}

    with ThreadPoolExecutor(12) as pool:
        for k, (u, margins) in enumerate(pool.map(score, utts)):
            for s, m in margins.items():
                for w in peaks(m, 4)[:a.per_utt]:
                    rows[s].append((int(u), int(w), int(m[w])))
            if k % 2000 == 0:
                print(f'{k}/{len(utts)} utterances, {time.perf_counter() - t0:.0f} s', flush=True)
    # Top `keep` per seed, then the union in rank order (rank 0 of every seed first), deduplicated within +-dedup_s.
    tops = {s: sorted(r, key=lambda x: -x[2])[:a.keep] for s, r in rows.items()}
    cand = sorted(((rank, s, u, w, m) for s, r in tops.items() for rank, (u, w, m) in enumerate(r)))
    gap = int(a.dedup_s * SR)
    kept, taken = [], {}
    for rank, s, u, w, m in cand:
        off = int(start[u] + SR + HOP * w)
        if any(abs(off - o) <= gap for o in taken.get(u, ())):
            continue
        taken.setdefault(u, []).append(off)
        kept.append((off, s, rank, u, w, m))
    off, seed, rank, u, w, m = (np.array(c) for c in zip(*kept))
    np.savez(DV / 'hard_sheila_r2.npz', offset=off, score=m, seed=seed, rank=rank, utt=u, window=w, uid=uid[u],
             text=text[u], utt_start=start[u], utt_length=length[u],
             stage1=np.array([f'research/results/S3/models/h48_k64_seed{s}.npz' for s in a.seeds]))
    info = {'utterances': int(len(utts)), 'hours': round(float(length[utts].sum()) / SR / 3600, 2),
            'peaks_per_seed': {s: len(r) for s, r in rows.items()},
            'kept_per_seed': {s: len(t) for s, t in tops.items()},
            'score_min_per_seed': {s: int(t[-1][2]) for s, t in tops.items()},
            'union_kept': len(kept), 'from_seed': {s: int((seed == s).sum()) for s in a.seeds},
            'seconds': round(time.perf_counter() - t0),
            'examples': [str(t)[:80] for t in text[u[np.argsort(rank)[:10]]]]}
    out = KWS / 'research/results/R2/B/mining.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)


if __name__ == '__main__':
    main()
