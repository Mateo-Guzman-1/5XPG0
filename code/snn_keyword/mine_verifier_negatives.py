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

--words: the same for single words. Every kept training clip of the keyword's
multi-corpus (verifier_data.targets_file, keyword_config.MULTI) that is not
the keyword, a word that begins with it or another spelling of it is placed
at a random offset in 1.6 s and augmented as in training (microphones, rooms,
noise), `--passes` times. Its score is stage 1's highest decision score over
the passes. The `--keep-words` highest go to
data_verifier/hardwords_<keyword>_<multi>.npz (manifest rows, words,
scores) for train_verifier.py --hardwords. For "sheila" they are mostly
"zero" (JOURNAL entry 32).
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


def mine_words(a, det, aug, dev):
    from verifier_data import targets_file
    t = np.load(targets_file('train'))
    keep = t['keep']
    rows, words = t['row'][keep], np.char.lower(t['word'][keep].astype(str))
    bad = np.array([w == K.KEYWORD or K.prefixed(w) or w in K.ALIASES for w in words])
    rows, words = rows[~bad], words[~bad]
    clips = np.load(K.MULTI / 'clips_train.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    best = np.full(len(rows), -np.inf)
    t0, n = time.perf_counter(), int(1.6 * SR)
    order = np.argsort(rows)                         # memmap reads in file order
    for ps in range(a.passes):
        for i in range(0, len(order), a.batch):
            ids = order[i:i + a.batch]
            w = np.zeros((len(ids), n), np.float32)
            off = rng.integers(0, n - SR + 1, len(ids))
            c = clips[rows[ids]].astype(np.float32) / 32768
            for j in range(len(ids)):
                w[j, off[j]:off[j] + SR] = c[j]
            with torch.no_grad():
                wt, mic = aug.waveform(torch.tensor(w, device=dev))
                x = aug.spectral(wt, mic, det.frontend)
                s = det.scores(x.float()).max(1)[0].cpu().numpy()
            best[ids] = np.maximum(best[ids], s)
            memguard.check('mining words')
        print(f'pass {ps + 1}/{a.passes}: {len(rows)} clips, {time.perf_counter() - t0:.0f} s', flush=True)
    top = np.argsort(best)[::-1][:a.keep_words]
    out = DV / f'hardwords_{K.KEYWORD}_{K.MULTI.name}.npz'
    np.savez(out, row=rows[top], word=words[top], score=best[top], stage1=str(a.stage1))
    import collections
    print(json.dumps({'clips': int(len(rows)), 'kept': int(len(top)), 'score_min': float(best[top].min()),
                      'top_words': collections.Counter(words[top]).most_common(15)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage1', type=Path, default=STAGE1)
    p.add_argument('--words', action='store_true', help='mine training words instead of LibriSpeech')
    p.add_argument('--passes', type=int, default=3, help='--words: augmented passes over the clips')
    p.add_argument('--batch', type=int, default=256, help='--words: clips per batch')
    p.add_argument('--keep-words', type=int, default=3000)
    p.add_argument('--keep', type=int, default=8000)
    p.add_argument('--per-utt', type=int, default=3, help='peaks kept per utterance before the global cut')
    p.add_argument('--batch-seconds', type=float, default=600.)
    p.add_argument('--gpu-fraction', type=float, default=.25)
    a = p.parse_args()
    dev = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(a.gpu_fraction, 0)
    det = StreamDetector(a.stage1, 'cuda')
    aug = Augmenter(dev, None, seed=0, mic_ranges='wide')
    if a.words:
        noise = np.load(K.MULTI / 'noise_train.npy', mmap_mode='r') if (K.MULTI / 'noise_train.npy').exists() \
            else np.load(ROOT / 'data/multi/noise_train.npy', mmap_mode='r')
        aug.noise = torch.cat([torch.tensor(np.asarray(noise[i:i + 2 ** 24]), device=dev).half() / 32768
                               for i in range(0, len(noise) // 4, 2 ** 24)])
        return mine_words(a, det, aug, dev)
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
