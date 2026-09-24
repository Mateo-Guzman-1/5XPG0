"""Ending- and position-aware augmentation made from real "yes" recordings.

The original model fired on words that share the "ye"/"ee" vowel but end in
/ts/ or /tʃ/ ("yeets", "pizza", "yetch"), on "yeah", and on live windows that
cut "yes" off before its /s/. Speech Commands has none of these, and its
clips are centred, while the live demo slides a 1 s window every 250 ms.
Every variant edits a real "yes" clip, so the model must learn that "yes" is
the complete word ending in a plain /s/:

  label 0  ts     stop closure + release burst + short, abrupt /s/    "yets"
  label 0  ch     closure + burst + frication moved to the /ʃ/ band    "yetch"
  label 0  sh     frication moved down, no closure                    "yesh"
  label 0  cut    the /s/ removed with a short fade                   "yeh"
  label 0  tail   window ends inside the word, at or before the /s/   "ye|"
  label 0  head   window starts after the /j/ onset                   "|es"
  label 1  shift  complete word at a random position in the window

Variants inherit their source clip's official speaker-disjoint split.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
from features import SAMPLE_RATE, features, read_wav

ROOT = Path(__file__).resolve().parent
KINDS = ('ts', 'ch', 'sh', 'cut', 'tail', 'head', 'shift')
LABEL = {k: int(k == 'shift') for k in KINDS}
HOP = 160


def s_region(a):
    """Sample span of the final /s/: last run of loud frames dominated by >=3.5 kHz."""
    n = len(a) // HOP
    spec = np.abs(np.fft.rfft(a[:n * HOP].reshape(n, HOP) * np.hanning(HOP), axis=1)) ** 2
    freqs = np.fft.rfftfreq(HOP, 1 / SAMPLE_RATE)
    total = spec.sum(1) + 1e-12
    ratio = spec[:, freqs >= 3500].sum(1) / total
    energy = 10 * np.log10(total)
    loud = energy > energy.max() - 35
    fric = (ratio > .55) & loud
    idx = np.flatnonzero(fric)
    if len(idx) == 0:
        return None
    end = start = idx[-1]
    while start > 1 and (fric[start - 1] or fric[start - 2]):
        start -= 1
    if end - start + 1 < 4 or start < 5 or not (loud[:start] & (ratio[:start] < .3)).any():
        return None
    return start * HOP, (end + 1) * HOP


def word_span(a):
    """First/last sample within 30 dB of the loudest 10 ms frame."""
    n = len(a) // HOP
    energy = 10 * np.log10((a[:n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-12)
    idx = np.flatnonzero(energy > energy.max() - 30)
    return idx[0] * HOP, (idx[-1] + 1) * HOP


def place(segment, offset):
    """Put a segment into a silent 1 s window at a sample offset."""
    out = np.zeros(SAMPLE_RATE, np.float32)
    segment = segment[:SAMPLE_RATE - offset]
    out[offset:offset + len(segment)] = segment
    return out


def lower_band(frication, factor):
    """Stretch the waveform by 1/factor, moving its spectrum down (s -> sh-like)."""
    positions = np.arange(0, len(frication) - 1, factor)
    return np.interp(positions, np.arange(len(frication)), frication).astype(np.float32)


def variant(a, span, kind, rng):
    s0, s1 = span
    w0, w1 = word_span(a)
    w1 = max(w1, s1)
    pad = int(.03 * SAMPLE_RATE)
    if kind == 'tail':
        # The live window's right edge falls inside the vowel or at the /s/ onset.
        cut = int(rng.uniform(w0 + .4 * (s0 - w0), s0 + .01 * SAMPLE_RATE))
        seg = a[max(0, w0 - pad):cut]
        return place(seg, SAMPLE_RATE - len(seg))
    if kind == 'head':
        start = int(rng.uniform(w0 + .06 * SAMPLE_RATE, max(w0 + .06 * SAMPLE_RATE + 1, min(w0 + .15 * SAMPLE_RATE, s0 - .05 * SAMPLE_RATE))))
        return place(a[start:w1 + pad], 0)
    if kind == 'shift':
        seg = a[max(0, w0 - pad):w1 + pad]
        return place(seg, int(rng.integers(0, max(1, SAMPLE_RATE - len(seg)))))
    head, fric = a[:s0], a[s0:s1]
    peak = float(np.abs(fric).max()) + 1e-6
    if kind == 'cut':
        fade = min(len(head), int(.02 * SAMPLE_RATE))
        head = head.copy()
        head[len(head) - fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        return head
    if kind in ('ch', 'sh'):
        fric = lower_band(fric, rng.uniform(.5, .65))
    closure = burst = np.zeros(0, np.float32)
    if kind in ('ts', 'ch'):
        closure = rng.normal(0, peak * .01, int(rng.uniform(.03, .13) * SAMPLE_RATE)).astype(np.float32)
        n = int(rng.uniform(.004, .012) * SAMPLE_RATE)
        burst = (rng.normal(0, 1, n) * np.exp(-np.linspace(0, 5, n)) * peak * rng.uniform(.6, 1.2)).astype(np.float32)
        # Affricate frication is shorter than a word-final /s/ and starts abruptly.
        fric = fric[:int(rng.uniform(.06, .16) * SAMPLE_RATE)]
    tail = min(len(fric), int(.015 * SAMPLE_RATE))
    fric = fric.copy()
    fric[len(fric) - tail:] *= np.linspace(1, 0, tail, dtype=np.float32)
    return np.concatenate([head, closure, burst, fric])


def fit(audio):
    """Keep the edited ending inside the 1 s window by dropping leading samples."""
    return audio[max(0, len(audio) - SAMPLE_RATE):]


def build(data, workers):
    cache = np.load(data / 'features.npz')
    names, split = cache['names'], cache['split']
    raw = data / 'speech_commands_v0.02'
    jobs = [(i, n) for i, n in enumerate(names) if n.startswith('yes/')]

    def make(job):
        i, name = job
        # Per-clip seed keeps the dataset reproducible regardless of thread order.
        rng = np.random.default_rng(i)
        audio = read_wav(raw / name)
        span = s_region(audio)
        if span is None:
            return []
        return [(features(fit(variant(audio, span, k, rng))), split[i], k, name) for k in KINDS]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for rs in pool.map(make, jobs) for r in rs]
    x = np.stack([r[0] for r in rows])
    kinds = np.array([r[2] for r in rows])
    np.savez_compressed(data / 'augment.npz', x=x, y=np.array([LABEL[k] for k in kinds], np.int64),
                        split=np.array([r[1] for r in rows], np.uint8), kind=kinds,
                        source=np.array([r[3] for r in rows]))
    summary = {'source_yes_clips': len(jobs), 'clips_with_s': len(rows) // len(KINDS), 'kinds': list(KINDS),
               'per_split': {s: int((np.array([r[1] for r in rows]) == i).sum())
                             for i, s in enumerate(['train', 'validation', 'test'])}}
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    build(a.data, a.workers)
