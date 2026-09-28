"""Verifier track, step 2: train the causal CTC phoneme recogniser (verifier_model.Verifier).

Every step has two sub-batches, both through augment_online.Augmenter
(mic_ranges 'wide', waveform() then spectral(), logmel frames):
  libri     LibriSpeech train-clean-100 utterances (verifier_data.py), in
            length buckets, whole utterances with their phoneme sequence.
  keywords  1 s clips of data/multi train (Speech Commands, MSWC, TTS) at a
            random offset in 1.6 s, target = the word's phonemes. "yes"
            (and yes-prefixed words: yesterday, yesd, ...) are oversampled.
Waveforms are zero-padded by 12% before augmentation, because a 0.9 speed
change stretches them and would cut off the last phonemes.
The frame-stacking phase is random (the first frame is dropped half the time).

Validation (every epoch): CTC loss on LibriSpeech dev-clean utterances, and the
float keyword score (verifier_model.keyword_score_torch) on validation keyword
clips placed in noise: AUC of "yes" against the other words.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')   # OpenBLAS buffers per thread count as private memory
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import sys

import memguard
from augment_online import Augmenter
from verifier_data import SYMBOLS, encode, text_phones
from verifier_model import Verifier, keyword_score_torch, n_params

ROOT = Path(__file__).resolve().parent
DV = ROOT / 'data_verifier'
SR = 16000


class LibriBatches:
    def __init__(self, rng, max_seconds=17.):
        ix = np.load(DV / 'libri100_index.npz')
        self.audio = np.load(DV / 'libri100_audio.npy', mmap_mode='r')
        keep = ix['length'] <= max_seconds * SR
        order = np.argsort(ix['length'])
        self.order = order[keep[order]]
        self.start, self.length = ix['start'], ix['length']
        self.phones, self.off = ix['phones'], ix['phone_off']
        self.rng = rng

    def batch(self, seconds):
        """Utterances of similar length whose total is about `seconds`."""
        k = self.rng.integers(len(self.order))
        n = max(1, int(seconds * SR // self.length[self.order[k]]))
        ids = self.order[max(0, min(k, len(self.order) - n)):][:n]
        L = int(np.ceil(self.length[ids].max() / SR)) * SR     # 1 s buckets: fewer distinct shapes
        w = np.zeros((len(ids), L), np.float32)
        for j, i in enumerate(ids):
            w[j, :self.length[i]] = self.audio[self.start[i]:self.start[i] + self.length[i]] / 32768
        return w, [self.phones[self.off[i]:self.off[i + 1]] for i in ids]


class KeywordBatches:
    def __init__(self, split, rng, yes_share=.25, prefixed_share=.05):
        t = np.load(DV / f'keyword_targets_{split}.npz')
        self.clips = np.load(ROOT / f'data/multi/clips_{split}.npy', mmap_mode='r')
        keep = t['keep']
        self.row, self.word, self.corpus = t['row'][keep], t['word'][keep], t['corpus'][keep]
        off, ph = t['phone_off'], t['phones']      # read each npz member once
        self.tg = [ph[off[i]:off[i + 1]] for i in np.flatnonzero(keep)]
        w = np.char.lower(self.word.astype(str))
        self.pools = {'yes': np.flatnonzero(w == 'yes'),
                      'prefixed': np.flatnonzero(np.char.startswith(w, 'yes') & (w != 'yes')),
                      'other': np.flatnonzero(~np.char.startswith(w, 'yes'))}
        self.share = {'yes': yes_share, 'prefixed': prefixed_share, 'other': 1 - yes_share - prefixed_share}
        self.rng = rng

    def draw(self, n):
        counts = self.rng.multinomial(n, list(self.share.values()))
        return np.concatenate([self.rng.choice(self.pools[k], c) for k, c in zip(self.share, counts) if c])

    def waves(self, ids, seconds=1.6, rng=None):
        rng = rng or self.rng
        n = int(seconds * SR)
        w = np.zeros((len(ids), n), np.float32)
        rows = self.row[ids]
        order = np.argsort(rows)
        clips = np.empty((len(ids), SR), np.float32)
        clips[order] = self.clips[rows[order]].astype(np.float32) / 32768
        off = rng.integers(0, n - SR + 1, len(ids))
        for j in range(len(ids)):
            w[j, off[j]:off[j] + SR] = clips[j]
        return w, [self.tg[i] for i in ids]


def pad_targets(tg, dev):
    lens = torch.tensor([len(t) for t in tg], device=dev)
    flat = torch.tensor(np.concatenate(tg).astype(np.int64), device=dev)
    return flat, lens


def ctc(model, x, tg, dev):
    logits = model(x)
    lp = F.log_softmax(logits.float(), -1).transpose(0, 1)
    flat, lens = pad_targets(tg, dev)
    il = torch.full((x.shape[0],), lp.shape[0], dtype=torch.long, device=dev)
    return F.ctc_loss(lp, flat, il, lens, blank=0, reduction='mean', zero_infinity=True)


def features(aug, wave, dev, pad=.12, train=True):
    w = torch.tensor(wave, device=dev)
    w = F.pad(w, (0, int(pad * w.shape[1]) + 400))
    if train:
        w, mic = aug.waveform(w)
        return aug.spectral(w, mic, 'logmel')
    power = aug.front.power(w)
    return aug.front.frames(aug.front.mel(power), 'logmel')


def phase(x, rng):
    return x[:, 1:] if rng.random() < .5 else x


def dev_clean(n=200, seed=0):
    import soundfile as sf
    base = ROOT / 'data/librispeech/LibriSpeech/dev-clean'
    items = []
    for t in sorted(base.rglob('*.trans.txt')):
        for line in t.read_text().splitlines():
            uid, text = line.split(' ', 1)
            ph = text_phones(text)
            if ph is not None:
                items.append((t.parent / f'{uid}.flac', encode(ph)))
    rng = np.random.default_rng(seed)
    pick = [items[i] for i in rng.choice(len(items), n, replace=False)]
    return [(sf.read(p, dtype='float32')[0], ph) for p, ph in pick]


def auc(pos, neg):
    s = np.r_[pos, neg]
    r = s.argsort().argsort() + 1.
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def validate(model, aug, dev, libri_val, kw_val, kw_ids, noise):
    model.eval()
    losses = []
    for i in range(0, len(libri_val), 16):
        part = libri_val[i:i + 16]
        L = max(len(a) for a, _ in part)
        w = np.zeros((len(part), L), np.float32)
        for j, (a, _) in enumerate(part):
            w[j, :len(a)] = a
        x = features(aug, w, dev, train=False)
        losses.append(ctc(model, x, [p for _, p in part], dev).item())
    rng = np.random.default_rng(1)
    w, _ = kw_val.waves(kw_ids, rng=rng)
    n0 = rng.integers(0, len(noise) - w.shape[1], len(w))
    w = w + np.stack([noise[k:k + w.shape[1]] for k in n0]) * rng.uniform(.02, .1, (len(w), 1)).astype(np.float32)
    scores = []
    for i in range(0, len(w), 512):
        x = features(aug, w[i:i + 512], dev, pad=0, train=False)
        scores.append(keyword_score_torch(model(x).double(), warmup=0).cpu().numpy())
    s = np.concatenate(scores)
    word = np.char.lower(kw_val.word[kw_ids].astype(str))
    yes = word == 'yes'
    other = ~np.char.startswith(word, 'yes')
    model.train()
    return {'val_ctc': round(float(np.mean(losses)), 4), 'kw_auc': round(auc(s[yes], s[other]), 5),
            'kw_yes_median': float(np.median(s[yes])),
            'kw_other_p99': float(np.percentile(s[other], 99))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--h1', type=int, default=64)
    p.add_argument('--h2', type=int, default=64)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--steps-per-epoch', type=int, default=500)
    p.add_argument('--libri-seconds', type=float, default=180., help='audio per LibriSpeech sub-batch')
    p.add_argument('--kw-batch', type=int, default=128)
    p.add_argument('--kw-weight', type=float, default=1.)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--force-start', action='store_true')
    p.add_argument('--gpu-fraction', type=float, default=.28, help='cap on the GPU memory share (~2.3 GB)')
    p.add_argument('--name', default='v1')
    p.add_argument('--out', type=Path, default=ROOT / 'runs_verifier')
    a = p.parse_args()
    ok, free = memguard.free_ok()
    print('free at start', free, flush=True)
    if not ok and not a.force_start:
        print('not enough free RAM (10 GB) or GPU memory (2.5 GB); exiting', flush=True)
        sys.exit(3)
    dev = torch.device('cuda')
    # Shared 8 GB GPU: a hard cap turns an overrun into an OOM error here instead of WDDM
    # paging GPU memory into host RAM (which exhausted system commit memory on the first run).
    torch.cuda.set_per_process_memory_fraction(a.gpu_fraction, 0)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    libri = LibriBatches(rng)
    kw = KeywordBatches('train', rng)
    kw_val = KeywordBatches('validation', np.random.default_rng(5))
    vrng = np.random.default_rng(3)
    kw_ids = np.r_[kw_val.pools['yes'], vrng.choice(kw_val.pools['other'], 4000, replace=False)]
    noise = np.load(ROOT / 'data/multi/noise_train.npy', mmap_mode='r')
    val_noise = np.asarray(noise[:SR * 600]).astype(np.float32) / 32768
    aug = Augmenter(dev, None, seed=a.seed, mic_ranges='wide')
    # Noise to the GPU in chunks from the memmap (Augmenter would hold ~3 GB of host RAM on the way).
    # Half of the noise (about 2.3 h): on Windows GPU memory also counts against private memory.
    aug.noise = torch.cat([torch.tensor(np.asarray(noise[i:i + 2 ** 24]), device=dev).half() / 32768
                           for i in range(0, len(noise) // 2, 2 ** 24)])
    libri_val = dev_clean()
    model = Verifier(a.h1, a.h2).to(dev)
    print('parameters', n_params(model), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=.05)
    out = a.out / a.name
    out.mkdir(parents=True, exist_ok=True)
    history, best, t0 = [], -1., time.perf_counter()
    for epoch in range(a.epochs):
        sums = {'libri': 0., 'kw': 0.}
        for step in range(a.steps_per_epoch):
            if step % 50 == 0:
                memguard.check(f'epoch {epoch + 1} step {step}')
            with torch.no_grad():
                wl, tl = libri.batch(a.libri_seconds)
                xl = phase(features(aug, wl, dev), rng)
                wk, tk = kw.waves(kw.draw(a.kw_batch))
                xk = phase(features(aug, wk, dev), rng)
            ll = ctc(model, xl, tl, dev)
            lk = ctc(model, xk, tk, dev)
            loss = ll + a.kw_weight * lk
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); sched.step()
            sums['libri'] += ll.item(); sums['kw'] += lk.item()
        torch.cuda.empty_cache()
        r = {k: round(v / a.steps_per_epoch, 4) for k, v in sums.items()}
        r.update(validate(model, aug, dev, libri_val, kw_val, kw_ids, val_noise))
        r.update(epoch=epoch + 1, minutes=round((time.perf_counter() - t0) / 60, 1),
                 gpu_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20))
        history.append(r)
        print(json.dumps(r), flush=True)
        ck = {'state_dict': copy.deepcopy(model.state_dict()), 'config': model.cfg, 'args': vars(a), 'epoch': epoch + 1,
              'frontend': 'logmel', 'symbols': SYMBOLS}
        torch.save(ck, out / 'last.pt')
        if r['kw_auc'] > best:
            best = r['kw_auc']
            torch.save(ck, out / 'model.pt')
    (out / 'training.json').write_text(json.dumps(dict(vars(a), history=history), indent=1, default=str))


if __name__ == '__main__':
    main()
