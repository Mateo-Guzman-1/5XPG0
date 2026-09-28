"""Batches of labelled 1 s clips from data/multi (prepare_multicorpus.py).

Classes: the 35 Speech Commands words, _unknown_ and _silence_. Each batch
draws from the corpora with fixed shares (balance by corpus, not by size):
Speech Commands, MSWC, TTS, LibriSpeech 1 s crops (_unknown_) and silence
(noise only; the augmenter adds it). Clips are moved by up to +-0.1 s.
"""
import csv
import re
from pathlib import Path

import numpy as np
import torch

from prepare_multicorpus import CLASSES, SILENCE, UNKNOWN

ROOT = Path(__file__).resolve().parent
import keyword_config as K

MULTI = K.MULTI   # data/multi for "yes", data/multi_<keyword> otherwise
SR = 16000
MIX = {'sc': .55, 'mswc': .15, 'tts': .12, 'libri': .10, 'silence': .08}


class ClipSampler:
    def __init__(self, split, device, corpora=('sc', 'mswc', 'tts'), mix=None, seed=0, libri=True, exclude=None):
        """exclude: regular expression; words matching it in full are left out (e.g. 'yes[a-z]',
        TTS pseudo-words that contain a complete "yes" followed by one letter)."""
        rows = [r for r in csv.DictReader(open(MULTI / 'manifest.csv', encoding='utf-8'))
                if r['split'] == split and r['corpus'] in corpora
                and not (exclude and re.fullmatch(exclude, r['word']))]
        self.clips = np.load(MULTI / f'clips_{split}.npy', mmap_mode='r')
        self.row = np.array([int(r['row']) for r in rows])
        self.cls = np.array([int(r['cls']) for r in rows])
        self.word = np.array([r['word'] for r in rows])
        corpus = np.array([r['corpus'] for r in rows])
        self.pools = {c: np.flatnonzero(corpus == c) for c in corpora if (corpus == c).any()}
        self.speech = np.load(MULTI / f'speech_{split}.npy', mmap_mode='r') if libri else None
        mix = dict(MIX if mix is None else mix)
        mix = {k: v for k, v in mix.items() if k in self.pools or (k == 'libri' and libri) or k == 'silence'}
        total = sum(mix.values())
        self.mix = {k: v / total for k, v in mix.items()}
        self.rng = np.random.default_rng(seed)
        self.device = device

    def _wave(self, rows):
        order = np.argsort(rows)  # sorted reads are faster from the memmap
        w = np.empty((len(rows), SR), np.float32)
        w[order] = self.clips[rows[order]].astype(np.float32) / 32768
        return w

    def batch(self, n, max_shift=.1):
        """wave (n, SR) float32, cls (n,), silence mask (n,), all on the device."""
        counts = self.rng.multinomial(n, list(self.mix.values()))
        waves, labels, sil = [], [], []
        for (kind, _), k in zip(self.mix.items(), counts):
            if not k:
                continue
            if kind in self.pools:
                ids = self.rng.choice(self.pools[kind], k)
                waves.append(self._wave(self.row[ids])); labels.append(self.cls[ids]); sil.append(np.zeros(k, bool))
            elif kind == 'libri':
                s = self.rng.integers(0, len(self.speech) - SR, k)
                waves.append(np.stack([self.speech[i:i + SR] for i in s]).astype(np.float32) / 32768)
                labels.append(np.full(k, UNKNOWN)); sil.append(np.zeros(k, bool))
            else:
                waves.append(np.zeros((k, SR), np.float32)); labels.append(np.full(k, SILENCE)); sil.append(np.ones(k, bool))
        wave = torch.tensor(np.concatenate(waves), device=self.device)
        wave = shift(wave, int(max_shift * SR), self.rng)
        return (wave, torch.tensor(np.concatenate(labels), device=self.device),
                torch.tensor(np.concatenate(sil), device=self.device))

    def yes_batch(self, n):
        """n "yes" clips, corpora in the batch mix's proportions (unshifted waveforms, numpy)."""
        yes_cls = CLASSES.index(K.KEYWORD)
        corp = [c for c in self.pools if (self.cls[self.pools[c]] == yes_cls).any()]
        p = np.array([self.mix.get(c, 0) for c in corp]); p = p / p.sum()
        ids = np.concatenate([self.rng.choice(self.pools[c][self.cls[self.pools[c]] == yes_cls], k)
                              for c, k in zip(corp, self.rng.multinomial(n, p)) if k])
        return self._wave(self.row[ids])

    def all_clips(self, corpus=None):
        """Every clip of the split (optionally one corpus), unshifted: for validation."""
        ids = np.arange(len(self.row)) if corpus is None else self.pools[corpus]
        return self.clips, self.row[ids], self.cls[ids], self.word[ids]


def shift(wave, max_shift, rng):
    b, n = wave.shape
    s = torch.tensor(rng.integers(-max_shift, max_shift + 1, b), device=wave.device)
    pos = torch.arange(n, device=wave.device)[None] - s[:, None]
    return torch.where((pos >= 0) & (pos < n), wave.gather(1, pos.clamp(0, n - 1)), 0.)


__all__ = ['ClipSampler', 'CLASSES', 'SILENCE', 'UNKNOWN', 'shift']
