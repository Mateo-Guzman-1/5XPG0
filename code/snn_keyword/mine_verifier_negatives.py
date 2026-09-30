"""Verifier track: hard negatives for the verifier, mined with stage 1 on its training speech.

Stage 1 (the float QAT model of the keyword's candidate, snn_stream.StreamDetector)
runs over LibriSpeech train-clean-100 (data_verifier/libri100_audio.npy, the
verifier's own training speech; no transcript contains the keyword or a word
that begins with it once those utterances are dropped). Every local maximum of
its decision score with no higher score within 1 s is a candidate; the
`--keep` highest are written to data_verifier/hard_<keyword>.npz as sample
offsets of the peak into libri100_audio.npy, with the score, the utterance id
and its transcript. These are the places where the cascade asks the verifier
on speech that is not the keyword: train_verifier.py --hard trains the
verifier to score them low. Validation and test speech are not touched.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import keyword_config as K
import memguard
from augment_online import Augmenter
from snn_stream import StreamDetector

ROOT = Path(__file__).resolve().parent
DV = ROOT / 'data_verifier'
STAGE1 = Path(r'C:\Users\matut\FULL_AI\5XPG0\code\snn_keyword\runs_stream') / \
    {'yes': 's2_nokd_qat/model.pt', 'sheila': 'sheila_qat_seed2/model.pt'}[K.KEYWORD]
SR = 16000


def peaks(score, min_gap):
    """Indices of maxima with no higher value within min_gap frames (greedy, highest first)."""
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
    p.add_argument('--stage1', type=Path, default=STAGE1)
    p.add_argument('--keep', type=int, default=8000)
    p.add_argument('--per-utt', type=int, default=3, help='peaks kept per utterance before the global cut')
    p.add_argument('--batch-seconds', type=float, default=600.)
    p.add_argument('--gpu-fraction', type=float, default=.25)
    a = p.parse_args()
    dev = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(a.gpu_fraction, 0)
    det = StreamDetector(a.stage1, 'cuda')
    aug = Augmenter(dev, None, seed=0, mic_ranges='wide')
    ix = np.load(DV / 'libri100_index.npz')
    audio = np.load(DV / 'libri100_audio.npy', mmap_mode='r')
    start, length, text, uid = ix['start'], ix['length'], ix['text'], ix['uid']
    keep_utt = np.array([not any(K.prefixed(w) or w.lower() == K.KEYWORD or w.lower() in K.ALIASES
                                 for w in t.split()) for t in text])
    order = np.argsort(length)
    order = order[keep_utt[order]]
    t0, rows = time.perf_counter(), []
    i = 0
    while i < len(order):
        n = max(1, int(a.batch_seconds * SR // length[order[min(i + 1, len(order) - 1)]]))
        ids = order[i:i + n]
        i += n
        L = int(length[ids].max())
        w = np.zeros((len(ids), L), np.float32)
        for j, u in enumerate(ids):
            w[j, :length[u]] = audio[start[u]:start[u] + length[u]] / 32768
        with torch.no_grad():
            wt = torch.tensor(w, device=dev)
            x = aug.front.frames(aug.front.mel(aug.front.power(wt)), det.frontend)
            s = det.scores(x.float()).cpu().numpy()
        for j, u in enumerate(ids):
            nf = min(s.shape[1], (length[u] - 400) // 160 + 1)
            sc = s[j, :nf]
            if det.window > 1:
                sc = np.convolve(sc, np.ones(det.window), 'full')[:nf]
            for f in peaks(sc, 100)[:a.per_utt]:
                rows.append((int(u), int(f), float(sc[f])))
        memguard.check('mining')
        if len(rows) and (i // n) % 20 == 0:
            print(f'{i}/{len(order)} utterances, {len(rows)} peaks, {time.perf_counter() - t0:.0f} s', flush=True)
    rows.sort(key=lambda r: -r[2])
    rows = rows[:a.keep]
    u = np.array([r[0] for r in rows]); f = np.array([r[1] for r in rows]); sc = np.array([r[2] for r in rows])
    offset = start[u] + f * 160 + 400                       # end of the peak frame, in libri100_audio samples
    np.savez(DV / f'hard_{K.KEYWORD}.npz', offset=offset, score=sc, utt=u, frame=f, uid=uid[u], text=text[u],
             utt_start=start[u], utt_length=length[u], stage1=str(a.stage1))
    info = {'utterances': int(len(order)), 'kept': len(rows), 'score_min': float(sc.min()), 'score_max': float(sc.max()),
            'seconds': round(time.perf_counter() - t0), 'examples': [str(t)[:80] for t in text[u[:10]]]}
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
