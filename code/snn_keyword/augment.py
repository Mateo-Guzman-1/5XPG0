"""Hard negatives and position variants made from real keyword recordings.

Speech Commands has no near-miss words for its keywords, and its clips are
centred, while the live demo slides a 1 s window every 250 ms. Every variant
edits a real recording of the keyword the data is prepared for
(prepare_data.py --keyword), so the model must learn the complete word.

"yes": the original model fired on words that share the "ye"/"ee" vowel but
end in /ts/ or /tʃ/ ("yeets", "pizza", "yetch"), on "yeah", and on live
windows that cut "yes" off before its /s/:

  label 0  ts     stop closure + release burst + short, abrupt /s/    "yets"
  label 0  ch     closure + burst + frication moved to the /ʃ/ band    "yetch"
  label 0  sh     frication moved down, no closure                    "yesh"
  label 0  cut    the /s/ removed with a short fade                   "yeh"
  label 0  tail   window ends inside the word, at or before the /s/   "ye|"
  label 0  head   window starts after the /j/ onset                   "|es"
  label 1  shift  complete word at a random position in the window

"sheila": no other dataset word contains /ʃ/, so the model learned "sh" (+ "ee")
as the keyword and fired on "sh", "she", "sheep" or "shield". The /ʃ/ and the
vowel onset are found per clip (sheila_marks; about 75% of the clips):

  label 0  sh      only the /ʃ/, sometimes repeated into a long "shhh"   "sh"
  label 0  shee    /ʃ/ + start of the vowel, "-la" removed with a fade   "shee"
  label 0  sheet   "shee" + stop closure + release burst                 "sheet"
  label 0  swap    /ʃ/ + another word without its unvoiced onset (x2)    "sh|op"
  label 0  ending  "shee" + the first 150-350 ms of another word         "shee|no"
  label 0  onset   another word's unvoiced onset + "eila"                "k|eila"
  label 0  head    window starts after the /ʃ/                           "|eila"
  label 0  noise   white to brown noise alone, sometimes fluctuating     (fan, hiss)
  label 1  shift   complete word at a random position in the window
Other words come from the same split and never start with /l/ for "ending"
("shee|l..." would be too close to the keyword). Half of the Sheila variants
(not "noise") get white-to-brown background noise at 5-30 dB SNR, so noise
alone says nothing about the label.

Variants inherit their source clips' official speaker-disjoint split.
--examples N writes N WAV files per kind to build/aug_examples_<keyword>/ to listen to.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import wave
import numpy as np
from features import SAMPLE_RATE, features, read_wav
from model import keyword_of

ROOT = Path(__file__).resolve().parent
KINDS = {'yes': ('ts', 'ch', 'sh', 'cut', 'tail', 'head', 'shift'),
         'sheila': ('sh', 'shee', 'sheet', 'swap', 'swap', 'ending', 'onset', 'head', 'noise', 'shift')}
LABEL = {k: int(k == 'shift') for kinds in KINDS.values() for k in kinds}
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


def frame_levels(a):
    """Per 10 ms frame: energy in dB below the loudest frame, share of energy >= 2 kHz and < 1 kHz."""
    n = len(a) // HOP
    spec = np.abs(np.fft.rfft(a[:n * HOP].reshape(n, HOP) * np.hanning(HOP), axis=1)) ** 2
    freqs = np.fft.rfftfreq(HOP, 1 / SAMPLE_RATE)
    total = spec.sum(1) + 1e-12
    rel = 10 * np.log10(total)
    return rel - rel.max(), spec[:, freqs >= 2000].sum(1) / total, spec[:, freqs < 1000].sum(1) / total


def runs(mask, gap):
    """[start, end) frame runs of True, bridging up to `gap` missing frames."""
    out = []
    for i in np.flatnonzero(mask):
        if out and i - out[-1][1] <= gap:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


def word_end(loud, start, limit):
    """End of the last loud frame after `start`, stopping at more than 5 quiet frames or `limit` frames."""
    end, quiet = start, 0
    for i in range(start, min(len(loud), start + limit)):
        if loud[i]:
            end, quiet = i + 1, 0
        else:
            quiet += 1
            if quiet > 5:
                break
    return end


def sheila_marks(a):
    """Samples where the /ʃ/ starts, where the vowel starts and where the word ends, or None.

    The /ʃ/ is >= 50 ms of frication (energy mostly >= 2 kHz, little < 1 kHz) directly
    followed by >= 150 ms of voicing; of several candidates, the loudest voicing wins.
    """
    rel, high, low = frame_levels(a)
    voiced = runs((low > .5) & (rel > -25), gap=4)
    best = None
    for f0, f1 in runs((high > .6) & (low < .2) & (rel > -40), gap=2):
        follow = [r for r in voiced if f1 - 1 <= r[0] <= f1 + 4]
        if f1 - f0 < 5 or not follow or follow[0][1] - follow[0][0] < 15:
            continue
        v0 = follow[0][0]
        w1 = word_end(rel > -30, v0, 70)
        if best is None or rel[v0:w1].max() > best[3]:
            best = (f0, v0, w1, rel[v0:w1].max())
    return None if best is None else tuple(k * HOP for k in best[:3])


def voice_start(rel, low):
    """First frame of the first voiced run of >= 80 ms that comes within 10 dB of the loudest frame."""
    loud_runs = [r for r in runs((low > .5) & (rel > -25), gap=4) if r[1] - r[0] >= 8 and rel[r[0]:r[1]].max() > -10]
    return loud_runs[0][0] if loud_runs else None


def rhyme(a, rng=None):
    """Samples of a word from its first loud voiced run to its end: the word without an unvoiced onset."""
    rel, _, low = frame_levels(a)
    r0 = voice_start(rel, low)
    r1 = None if r0 is None else word_end(rel > -30, r0, 80)
    return None if r0 is None or r1 - r0 < 10 else (r0 * HOP, r1 * HOP)


def onset(a, rng=None):
    """Samples of a word's unvoiced onset: 40-200 ms of mostly unvoiced sound before its voicing ("k" of "cat")."""
    rel, _, low = frame_levels(a)
    r0, w0 = voice_start(rel, low), int(np.argmax(rel > -30))
    if r0 is None or not 4 <= r0 - w0 <= 20 or low[w0:r0].mean() > .35:
        return None
    return max(0, w0 - 3) * HOP, r0 * HOP


def word_head(a, rng):
    """Samples of the first 150-350 ms of a word: its onset and the start of its vowel."""
    w0, n = int(np.argmax(frame_levels(a)[0] > -30)), int(rng.uniform(15, 35))
    return (w0 * HOP, (w0 + n) * HOP) if (w0 + n) * HOP <= len(a) else None


SOURCES = {'swap': rhyme, 'ending': word_head, 'onset': onset}   # parts of other words, per Sheila kind


def colored_noise(n, rng):
    """Unit-RMS noise with a 1/f^alpha power spectrum, alpha from 0 (white) over 1 (pink) to 2 (brown)."""
    spec = np.fft.rfft(rng.normal(size=n)) / np.maximum(np.fft.rfftfreq(n, 1 / SAMPLE_RATE), 20) ** (rng.uniform(0, 2) / 2)
    x = np.fft.irfft(spec, n)
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-12)).astype(np.float32)


def fade_out(seg, n):
    seg = seg.copy()
    n = min(n, len(seg))
    seg[len(seg) - n:] *= np.linspace(1, 0, n, dtype=np.float32)
    return seg


def crossfade(a, b, n):
    n = min(n, len(a), len(b))
    ramp = np.linspace(0, 1, n, dtype=np.float32)
    return np.concatenate([a[:len(a) - n], a[len(a) - n:] * (1 - ramp) + b[:n] * ramp, b[n:]])


def sheila_variant(a, marks, kind, rng, other=None):
    """One edited 1 s window from a "sheila" clip; `other` is the part of another word (SOURCES)."""
    f0, v0, w1 = marks
    pad, ms = int(.03 * SAMPLE_RATE), SAMPLE_RATE // 1000
    fric = a[max(0, f0 - pad):v0]
    vowel_peak = float(np.abs(a[v0:w1]).max()) + 1e-6
    if kind == 'noise':
        wobble = 1 + rng.uniform(0, .8) * np.sin(2 * np.pi * (rng.uniform(.5, 4) * np.arange(SAMPLE_RATE) / SAMPLE_RATE
                                                              + rng.uniform()))
        return (colored_noise(SAMPLE_RATE, rng) * wobble * 10 ** rng.uniform(-3, -.5)).astype(np.float32)   # -60..-10 dBFS
    window = None
    if kind == 'head':
        seg = a[v0 - 10 * ms:w1 + pad]
        window = place(seg, 0)
    elif kind == 'shift':
        seg = a[max(0, f0 - pad):w1 + pad]
    elif kind == 'ending':   # "shee" + the start of another word at this clip's vowel level
        shee = a[max(0, f0 - pad):v0 + int(rng.uniform(.25, .45) * (w1 - v0))]
        seg = fade_out(crossfade(shee, other * vowel_peak / (float(np.abs(other).max()) + 1e-6), 8 * ms), 20 * ms)
    elif kind == 'onset':    # another word's unvoiced onset at up to this clip's /ʃ/ level, then "eila"
        start = other * rng.uniform(.5, 1) * float(np.abs(fric).max()) / (float(np.abs(other).max()) + 1e-6)
        seg = crossfade(start, a[v0 - 10 * ms:w1 + pad], 8 * ms)
    elif kind == 'sh':
        seg = fric
        for _ in range(int(rng.integers(0, 3))):   # the noise-like /ʃ/ repeats into a longer "shhh"
            seg = crossfade(seg, a[f0:v0], 10 * ms)
        seg = fade_out(np.concatenate([seg, a[v0:v0 + int(rng.uniform(0, 30)) * ms]]), 15 * ms)
    elif kind in ('shee', 'sheet'):
        cut = v0 + int(rng.uniform(.2, .45) * (w1 - v0))
        seg = fade_out(a[max(0, f0 - pad):cut], int(rng.uniform(20, 40)) * ms)
        if kind == 'sheet':
            peak = float(np.abs(fric).max()) + 1e-6
            closure = rng.normal(0, peak * .01, int(rng.uniform(30, 120)) * ms).astype(np.float32)
            n = int(rng.uniform(4, 12)) * ms
            burst = (rng.normal(0, 1, n) * np.exp(-np.linspace(0, 5, n)) * peak * rng.uniform(.6, 1.2)).astype(np.float32)
            seg = np.concatenate([seg, closure, burst])
    else:   # swap: the other word's rhyme at the level of this clip's vowel
        seg = crossfade(fric, other * vowel_peak / (float(np.abs(other).max()) + 1e-6), 8 * ms)
    if window is None:
        window = place(seg, int(rng.integers(0, max(1, SAMPLE_RATE - len(seg)))))
    if rng.random() < .5:   # background noise at 5-30 dB SNR for half of the variants
        rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-9
        window = window + colored_noise(SAMPLE_RATE, rng) * rms * 10 ** (-rng.uniform(5, 30) / 20)
    return window


def write_wav(path, audio):
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SAMPLE_RATE)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())


def build(data, workers, examples=0):
    cache = np.load(data / 'features.npz')
    keyword = keyword_of(cache)
    if keyword not in KINDS:
        raise SystemExit(f'No augmentation is defined for "{keyword}"; see the docstring for "yes" and "sheila"')
    names, split = cache['names'], cache['split']
    raw = data / 'speech_commands_v0.02'
    jobs = [(i, n) for i, n in enumerate(names) if n.startswith(keyword + '/')]
    others = [np.flatnonzero((split == s) & ~np.char.startswith(names, keyword + '/')) for s in range(3)]
    listen = {i for i, _ in [j for j in jobs if split[j[0]] == 0][:4 * examples]}   # audio kept to write examples

    def make(job):
        i, name = job
        # Per-clip seed keeps the dataset reproducible regardless of thread order.
        rng = np.random.default_rng(i)
        audio = read_wav(raw / name)
        if keyword == 'yes':
            span = s_region(audio)
            made = [] if span is None else [(variant(audio, span, k, rng), k, name) for k in KINDS[keyword]]
        else:
            marks = sheila_marks(audio)
            made = []
            for k in (KINDS[keyword] if marks else ()):
                other, source = None, name
                for _ in range(20 if k in SOURCES else 0):   # part of another word from the same split
                    j = int(rng.choice(others[split[i]]))
                    if k == 'ending' and names[j].startswith(('left/', 'learn/')):
                        continue
                    b = read_wav(raw / names[j])
                    span = SOURCES[k](b, rng)
                    if span:
                        other, source = b[span[0]:span[1]], f'{name}+{names[j]}'
                        break
                if k not in SOURCES or other is not None:
                    made.append((sheila_variant(audio, marks, k, rng, other), k, source))
        return [(features(fit(v)), split[i], k, source, fit(v) if i in listen else None) for v, k, source in made]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for rs in pool.map(make, jobs) for r in rs]
    x = np.stack([r[0] for r in rows])
    kinds = np.array([r[2] for r in rows])
    np.savez_compressed(data / 'augment.npz', x=x, y=np.array([LABEL[k] for k in kinds], np.int64),
                        split=np.array([r[1] for r in rows], np.uint8), kind=kinds,
                        source=np.array([r[3] for r in rows]), keyword=np.array(keyword))
    if examples:
        out = ROOT / 'build' / f'aug_examples_{keyword}'
        out.mkdir(parents=True, exist_ok=True)
        count = {}
        for _, _, k, _, audio in rows:
            if audio is not None and count.get(k, 0) < examples:
                count[k] = count.get(k, 0) + 1
                write_wav(out / f'{k}_{count[k]:02d}.wav', audio)
    splits = np.array([r[1] for r in rows])
    summary = {'keyword': keyword, f'source_{keyword}_clips': len(jobs),
               'clips_used': len({r[3].split('+')[0] for r in rows}), 'kinds': sorted(set(KINDS[keyword])),
               'per_split': {s: int((splits == i).sum()) for i, s in enumerate(['train', 'validation', 'test'])},
               'per_kind': {k: int((kinds == k).sum()) for k in sorted(set(KINDS[keyword]))}}
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--examples', type=int, default=0, help='WAV files per kind to write for listening')
    a = p.parse_args()
    build(a.data, a.workers, a.examples)
