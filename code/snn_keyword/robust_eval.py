"""Robustness harness (IMPLEMENTATION_PLAN.md, Phase 0).

Every test feeds waveforms to a detector and scores its decision trace, so
window models (the release) and frame-streaming models (Phase 3 onwards) are
measured the same way. A detector returns, per audio, the times of its
decision steps and a score per step; it detects where score >= threshold.
For window models the score is the "2 of 3" confirmed margin
(min(this window, best of the two previous)) at the stream threshold.

  live      Speech Commands clips placed in 2.25 s of background noise, as in
            confusables.py (same seeds, so the clean numbers must agree).
  device    The same clips through held-out microphones, held-out real rooms,
            and both (channels.py). Recall drop in points against clean.
  corpus    Leave-one-source-out: MSWC English [39] clips (never trained on),
            same noise placement. Recall and false accepts at the operating
            threshold, and recall at the false-accept rate of the Speech
            Commands live test (equal-FA drop). Near-miss words per group.
  stream    One hour of continuous audio: Speech Commands non-"yes" words,
            LibriSpeech [40] read speech (utterances containing "yes"
            removed) and "yes" clips at known times, over varying background
            noise. False accepts per hour, recall, latency from the end of
            the word to the detection, and the recall/false-accept curve.
  tts       Synthesized-word probe of confusables.py (Windows voices).

--split validation builds every test from validation data (Speech Commands
validation, MSWC dev, LibriSpeech dev-clean) for threshold and checkpoint
selection; the default test split is for reporting only.
"""
import argparse
import copy
import csv
import glob
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

import channels
from features import N_TIME, SAMPLE_RATE, features, read_wav
from model import integer_forward

ROOT = Path(__file__).resolve().parent
HOP = SAMPLE_RATE // 4
LIVE_SECONDS = 2.25
HOLDOFF = 1.0            # s; one detection lights LED0 for about 1 s
TOLERANCE = 1.25         # s after the end of a "yes" in which a detection counts as a hit
STREAM_SECONDS = 3600
SPLITS = {'test': dict(sc=2, mswc='test', libri='test-clean', seed=1),
          'validation': dict(sc=1, mswc='dev', libri='dev-clean', seed=2)}
# Words that begin with a complete "yes" (yesterday, yessir, ...) are "don't care":
# a causal detector cannot tell them from "yes" without waiting for the rest of
# the word. They are not false accepts anywhere; YES_PREFIXED reports them apart.
YES_PREFIXED = 'yes-prefixed (yesterday; not a false accept)'


def yes_prefixed(word):
    w = str(word).lower()
    return w.startswith('yes') and w != 'yes'


MSWC_GROUPS = {
    YES_PREFIXED: ['yesterday'],
    'ye- (yet, yeah, yep, yell, yellow)': ['yet', 'yeah', 'yep', 'yell', 'yellow'],
    '/s/-final (guess, less, this, us, ...)': ['guess', 'less', 'mess', 'bless', 'dress', 'press', 'chess', 'address',
                                              'unless', 'success', 'says', 'this', 'us', 'plus', 'jess', 'tess'],
    '/ts/-final (its, gets, lets, eats, ...)': ['its', 'gets', 'lets', 'sets', 'bets', 'jets', 'pets', 'eats', 'meets',
                                               'seats', 'streets'],
    '/tʃ/-final (each, reach, speech, ...)': ['each', 'reach', 'beach', 'peach', 'teach', 'speech', 'fetch', 'sketch',
                                             'stretch'],
    '/st/, /ks/ (best, test, next, ...)': ['nest', 'best', 'rest', 'west', 'test', 'sex', 'next', 'ex', 'text'],
    'other near (year, you, said, cheese, ...)': ['year', 'years', 'young', 'you', 'said', 'set', 'check', 'pizza',
                                                  'jazz', 'cheese', 'ease', 'these'],
}
TTS_GROUPS = {'yes': ['yes'], 'yeets/yets/yetz': ['yeets', 'yets', 'yetz'], 'pizza(s)': ['pizza', 'pizzas'],
              'eats/its': ['eats', 'its'], 'other -ts': ['jets', 'gets', 'bets', 'lets', 'sets'],
              'ch words': ['yech', 'yetch', 'each', 'peach'], 'yeah': ['yeah'], 'yeet': ['yeet'],
              'guess/less': ['guess', 'less'], 'cheese': ['cheese'], YES_PREFIXED: ['yesterday']}


# --------------------------------------------------------------------------- detectors

class WindowDetector:
    """1 s window every 250 ms through integer_forward; score = "2 of 3" confirmed margin."""
    kind = 'window'
    _cache = {}

    def __init__(self, q):
        self.q = q
        self.threshold = int(q.get('stream_threshold', q['decision_threshold']))
        self.single_threshold = int(q['decision_threshold'])

    @classmethod
    def window_features(cls, audio):
        key = hashlib.blake2b(audio.tobytes(), digest_size=16).digest()
        if key not in cls._cache:
            starts = range(0, len(audio) - SAMPLE_RATE + 1, HOP)
            cls._cache[key] = np.stack([features(audio[s:s + SAMPLE_RATE]) for s in starts])
        return cls._cache[key]

    def traces(self, audios, workers=12):
        """Per audio: (step end times in s, confirmed score, single-window margin)."""
        with ThreadPoolExecutor(workers) as pool:
            feats = list(pool.map(self.window_features, audios))
        scores, _ = integer_forward(np.concatenate(feats), self.q)
        margin = (scores[:, 1] - scores[:, 0]).astype(np.int64)
        out, i = [], 0
        for f in feats:
            m = margin[i:i + len(f)]; i += len(f)
            c = np.full(len(m), np.iinfo(np.int64).min // 2)
            for k in range(1, len(m)):
                c[k] = min(m[k], m[max(0, k - 2):k].max())
            out.append(((np.arange(len(m)) * HOP + SAMPLE_RATE) / SAMPLE_RATE, c, m))
        return out


def load_detector(path):
    if str(path).endswith('.pt'):
        from snn_stream import StreamDetector
        return StreamDetector(path)
    q = dict(np.load(path))
    if 'kind' in q and str(q['kind']) == 'stream':
        from snn_stream import IntegerStreamDetector
        return IntegerStreamDetector(q)
    return WindowDetector(q)


# --------------------------------------------------------------------------- audio sets

def bg_noise(data):
    raw = data / 'speech_commands_v0.02' / '_background_noise_'
    return np.concatenate([read_wav(p) for p in sorted(glob.glob(str(raw / '*.wav')))])


def place_clips(clips, noise, rng, channel=None):
    """Each clip at a random offset in 2.25 s of noise, cut at a random hop phase.

    The random draws match tune_stream.live_windows, so the clean set is the
    one confusables.py scores. rng is copied, so every condition gets the
    same placements. channel: None, 'mic', 'room' or 'mic+room'.
    """
    rng = copy.deepcopy(rng)
    n = int(LIVE_SECONDS * SAMPLE_RATE)
    mics, rirs = channels.heldout_mics(), channels.heldout_rirs() if channel and 'room' in channel else ()
    out = []
    for k, a in enumerate(clips):
        a = a[:SAMPLE_RATE]
        off = rng.integers(0, n - len(a)); n0 = rng.integers(0, len(noise) - n)
        gain = rng.uniform(.02, .1); phase = int(rng.integers(0, HOP))
        if channel and 'room' in channel:
            crng = np.random.default_rng(channels.HELDOUT_SEED + k)
            a = channels.apply_rir(a, rirs[crng.integers(len(rirs))])
        buf = np.zeros(n, np.float32)
        a = a[:n - off]
        buf[off:off + len(a)] = a
        buf += noise[n0:n0 + n] * gain
        if channel and 'mic' in channel:
            buf = channels.apply_mic(buf, mics[k % len(mics)])
        out.append(buf[phase:])
    return out


def sc_live_clips(data, split, negatives=3000, seed=1):
    """Speech Commands clips in the order and selection of tune_stream.live_windows."""
    d = np.load(data / 'features.npz')
    names, y = d['names'][d['split'] == split], d['y'][d['split'] == split]
    rng = np.random.default_rng(seed)
    ids = np.r_[np.flatnonzero(y == 1), rng.choice(np.flatnonzero(y == 0), negatives, replace=False)]
    raw = data / 'speech_commands_v0.02'
    # The generator continues into place_clips, as in live_windows.
    return [read_wav(raw / names[i]) for i in ids], y[ids], rng


def edge_clipped(a, frame=160, margin_db=15):
    """True if the content ends within 15 dB of its loudest 10 ms frame: the /s/ is cut off.

    MSWC clips are zero-padded after the alignment cut, so the test looks at
    the last 20 ms of the non-silent part. A complete /s/ fades out; about
    5% of MSWC and Speech Commands "yes" clips end this loud.
    """
    nz = np.flatnonzero(np.abs(a) > 1e-4)
    if len(nz) < 4 * frame:
        return False
    a = a[nz[0]:nz[-1] + 1]
    n = len(a) // frame
    e = 10 * np.log10((a[:n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    return bool(e[-2:].max() > e.max() - margin_db)


def mswc_clips(data, split):
    base = data / 'mswc'
    table = base / f'{split}_selected.csv'
    if not table.exists():
        raise FileNotFoundError(f'{table}; run: python fetch_corpora.py mswc --split {split}')
    rows = list(csv.DictReader(open(table, encoding='utf-8')))
    return [read_wav(base / r['path']) for r in rows], [r['word'] for r in rows]


def libri_utterances(data, subset):
    base = data / 'librispeech' / 'LibriSpeech' / subset
    if not base.exists():
        raise FileNotFoundError(f'{base}; run: python fetch_corpora.py librispeech --subset {subset}')
    keep = []
    for t in sorted(base.rglob('*.trans.txt')):
        for line in t.read_text().splitlines():
            uid, text = line.split(' ', 1)
            if not any(w.startswith('YES') for w in text.split()):   # YES and YESTERDAY, YES'M, ...
                keep.append(t.parent / f'{uid}.flac')
    return keep


def word_span(a, frame=160):
    n = len(a) // frame
    e = 10 * np.log10((a[:n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    idx = np.flatnonzero(e > e.max() - 30)
    return idx[0] * frame, (idx[-1] + 1) * frame


def build_stream(data, split, seconds=STREAM_SECONDS, seed=7):
    """Continuous audio with "yes" at known times. Returns audio, yes (start, end) in s, composition."""
    import soundfile as sf
    cfg = SPLITS[split]
    d = np.load(data / 'features.npz')
    names, y = d['names'][d['split'] == cfg['sc']], d['y'][d['split'] == cfg['sc']]
    raw = data / 'speech_commands_v0.02'
    libri = libri_utterances(data, cfg['libri'])
    rng = np.random.default_rng(seed)
    yes = list(rng.permutation(np.flatnonzero(y == 1))); words = list(rng.permutation(np.flatnonzero(y == 0)))
    libri = [libri[i] for i in rng.permutation(len(libri))]
    n = int(seconds * SAMPLE_RATE)
    audio = np.zeros(n, np.float32)
    marks, segments = [], []
    count, speech = {'yes': 0, 'word': 0, 'libri': 0}, {'yes': 0., 'word': 0., 'libri': 0.}
    t = int(rng.uniform(.5, 2) * SAMPLE_RATE)
    while True:
        kind = rng.choice(['yes', 'word', 'libri'], p=[.15, .55, .30])
        if kind == 'yes':
            name = str(names[yes.pop()]); a = read_wav(raw / name)
        elif kind == 'word':
            name = str(names[words.pop()]); a = read_wav(raw / name)
        else:
            name = str(libri.pop()); a = sf.read(name, dtype='float32')[0]
        if t + len(a) > n:
            break
        audio[t:t + len(a)] += a
        if kind == 'yes':
            s, e = word_span(a)
            marks.append(((t + s) / SAMPLE_RATE, (t + e) / SAMPLE_RATE))
        segments.append((t / SAMPLE_RATE, (t + len(a)) / SAMPLE_RATE, kind, name))
        count[kind] += 1; speech[kind] += len(a) / SAMPLE_RATE
        t += len(a) + int(rng.uniform(.3, 2) * SAMPLE_RATE)
    # Background noise in 10 s pieces at random levels, joined by 50 ms crossfades.
    noise, piece, fade = bg_noise(data), 10 * SAMPLE_RATE, SAMPLE_RATE // 20
    env = np.ones(piece + fade, np.float32); env[:fade] = np.linspace(0, 1, fade); env[-fade:] = np.linspace(1, 0, fade)
    for s in range(0, n, piece):
        o = rng.integers(0, len(noise) - piece - fade)
        seg = noise[o:o + piece + fade] * env * rng.uniform(.02, .1)
        audio[s:s + len(seg)] += seg[:n - s]
    info = {'seconds': seconds, 'clips': count, 'speech_seconds': {k: round(v, 1) for k, v in speech.items()},
            'libri_subset': cfg['libri'], 'seed': seed}
    info['segments'] = segments
    return audio, np.array(marks), info


NEG_LIBRI = {'validation': ('dev-clean', 'dev-other'), 'test': ('test-clean', 'test-other')}


def negative_stream(data, split, chunk_seconds=600, seed=11):
    """Long negatives-only stream for false accepts per hour, in chunks of chunk_seconds.

    Every LibriSpeech utterance of the split's held-out subsets (NEG_LIBRI; none
    whose transcript contains YES) and every non-"yes" Speech Commands word of
    the split, shuffled, 0.3-2 s apart, over the background noise as in
    build_stream. The 1 h stream of build_stream holds only about 2 false
    accepts at the operating point, too few to set or compare a <= 2/h
    threshold; this one holds 10-15 h. Yields (audio, segments) per chunk;
    segments are (start s, end s, kind, name) relative to the chunk. The
    network state starts fresh in every chunk.
    """
    import soundfile as sf
    cfg = SPLITS[split]
    d = np.load(data / 'features.npz')
    names, y = d['names'][d['split'] == cfg['sc']], d['y'][d['split'] == cfg['sc']]
    raw = data / 'speech_commands_v0.02'
    items = [('word', str(n)) for n in names[y == 0]]
    for subset in NEG_LIBRI[split]:
        items += [('libri', str(f)) for f in libri_utterances(data, subset)]
    rng = np.random.default_rng(seed)
    items = [items[i] for i in rng.permutation(len(items))]
    noise = bg_noise(data)
    n = int(chunk_seconds * SAMPLE_RATE)
    piece, fade = 10 * SAMPLE_RATE, SAMPLE_RATE // 20
    env = np.ones(piece + fade, np.float32); env[:fade] = np.linspace(0, 1, fade); env[-fade:] = np.linspace(1, 0, fade)

    def background():
        out = np.zeros(n, np.float32)
        for s in range(0, n, piece):
            o = rng.integers(0, len(noise) - piece - fade)
            seg = noise[o:o + piece + fade] * env * rng.uniform(.02, .1)
            out[s:s + len(seg)] += seg[:n - s]
        return out

    audio, segments, t = background(), [], int(rng.uniform(.5, 2) * SAMPLE_RATE)
    for kind, name in items:
        a = read_wav(raw / name) if kind == 'word' else sf.read(name, dtype='float32')[0]
        if t + len(a) > n:
            yield audio[:t], segments
            audio, segments, t = background(), [], int(rng.uniform(.5, 2) * SAMPLE_RATE)
        a = a[:n - t]
        audio[t:t + len(a)] += a
        segments.append((t / SAMPLE_RATE, (t + len(a)) / SAMPLE_RATE, kind, name))
        t += len(a) + int(rng.uniform(.3, 2) * SAMPLE_RATE)
    yield audio[:t], segments


def negative_traces(det, data, split, batch=12, chunk_seconds=600):
    """Score traces over negative_stream: list of (times, score, segments) per chunk, and hours."""
    out, pending, seconds = [], [], 0.
    def flush():
        for (a, seg), (times, score, _) in zip(pending, det.traces([a for a, _ in pending])):
            out.append((times, score, seg))
        pending.clear()
    for audio, seg in negative_stream(data, split, chunk_seconds):
        seconds += len(audio) / SAMPLE_RATE
        pending.append((audio, seg))
        if len(pending) == batch:
            flush()
    if pending:
        flush()
    return out, seconds / 3600


def negative_fa(traces, hours, threshold, holdoff=HOLDOFF):
    """False accepts per hour on negative_traces output (every detection is false)."""
    return sum(len(detections(t, s, threshold, holdoff)) for t, s, _ in traces) / hours


def tts_clips(data, probe):
    """confusables.tts_sequences as waveforms: each file at 4 hop phases."""
    files = sorted(glob.glob(str(probe / '*.wav')))
    if not files:
        return None
    raw = data / 'speech_commands_v0.02' / '_background_noise_'
    noise = np.concatenate([read_wav(p) for p in sorted(glob.glob(str(raw / '*.wav')))[:3]])
    rng = np.random.default_rng(0)
    words, audios = [], []
    for p in files:
        a = read_wav(p)[:int(1.4 * SAMPLE_RATE)]
        buf = np.zeros(int(2.5 * SAMPLE_RATE), np.float32)
        off = rng.integers(0, len(buf) - len(a)); buf[off:off + len(a)] += a
        n0 = rng.integers(0, len(noise) - len(buf)); buf += noise[n0:n0 + len(buf)] * .05
        for phase in range(0, HOP, HOP // 4):
            words.append(os.path.basename(p).split('_')[0]); audios.append(buf[phase:])
    return np.array(words), audios


# --------------------------------------------------------------------------- scoring

def clip_scores(det, audios):
    return np.array([c.max() for _, c, _ in det.traces(audios)])


def rate(x):
    return round(float(np.mean(x)), 4) if len(x) else None


def threshold_at_fa(neg_scores, fa):
    """Lowest threshold whose false-accept rate on neg_scores is at most fa."""
    cand = np.r_[np.unique(neg_scores), neg_scores.max() + 1]
    ok = [t for t in cand if (neg_scores >= t).mean() <= fa]
    return min(ok)


def detections(times, score, threshold, holdoff=HOLDOFF):
    ev, last = [], -np.inf
    for i in np.flatnonzero(score >= threshold):
        if times[i] - last >= holdoff:
            ev.append(times[i]); last = times[i]
    return np.array(ev)


def source_of(t, segments, span=1.0):
    """Kind of the inserted clip overlapping most with the last `span` seconds before t."""
    best, kind = 0., 'noise only'
    for s, e, k, *_ in segments:
        o = min(e, t) - max(s, t - span)
        if o > best:
            best, kind = o, k
    return {'word': 'speech commands word', 'libri': 'librispeech', 'yes': 'yes (duplicate)'}.get(kind, kind)


def score_stream(times, score, threshold, marks, hours, segments=None):
    ev = detections(times, score, threshold)
    hit_of = np.full(len(ev), -1)
    for j, (s, e) in enumerate(marks):
        inside = np.flatnonzero((ev >= s) & (ev <= e + TOLERANCE))
        hit_of[inside] = j
    hits = np.unique(hit_of[hit_of >= 0])
    first = {j: ev[hit_of == j].min() for j in hits}
    latency = np.array([first[j] - marks[j][1] for j in hits])
    fa = int((hit_of < 0).sum())
    extra = {}
    if segments is not None:
        import collections
        extra['false_accepts_by_source'] = dict(collections.Counter(source_of(t, segments) for t in ev[hit_of < 0]))
    return dict(extra, recall=round(len(hits) / len(marks), 4), false_accepts=fa, fa_per_hour=round(fa / hours, 2),
                latency_median_s=round(float(np.median(latency)), 3) if len(latency) else None,
                latency_p90_s=round(float(np.percentile(latency, 90)), 3) if len(latency) else None)


def evaluate(det, sets):
    out = {'kind': det.kind, 'threshold': det.threshold}
    th = det.threshold
    # Live, clean and under held-out channels.
    y = sets['live_y']
    live = {}
    for cond, audios in sets['live'].items():
        tr = det.traces(audios)
        conf = np.array([c.max() for _, c, _ in tr])
        live[cond] = {'yes_detected': rate(conf[y == 1] >= th), 'other_accepted': rate(conf[y == 0] >= th)}
        if cond == 'clean':
            sc_scores = conf
            if det.kind == 'window':
                single = np.array([m.max() for _, _, m in tr]) >= det.single_threshold
                live['clean_single_window'] = {'yes_detected': rate(single[y == 1]), 'other_accepted': rate(single[y == 0])}
    for cond in sets['live']:
        if cond != 'clean':
            live[cond]['recall_drop_points'] = round(100 * (live['clean']['yes_detected'] - live[cond]['yes_detected']), 2)
    out['live'] = live
    # Leave-one-source-out: MSWC.
    if sets.get('mswc'):
        audios, words, clipped = sets['mswc']
        s = clip_scores(det, audios)
        pos, rnd = words == 'yes', ~np.isin(words, sum(MSWC_GROUPS.values(), []) + ['yes'])
        ref_fa = float((sc_scores[y == 0] >= th).mean())
        t_eq = threshold_at_fa(s[rnd], ref_fa)
        # Recall of Speech Commands at the same false-accept rate, measured with the same rule.
        sc_eq = float((sc_scores[y == 1] >= threshold_at_fa(sc_scores[y == 0], ref_fa)).mean())
        mswc_eq = float((s[pos] >= t_eq).mean())
        out['corpus_mswc'] = {
            'yes_detected': rate(s[pos] >= th), 'yes_detected_clean_edge': rate(s[pos & ~clipped] >= th),
            'other_accepted': rate(s[rnd] >= th),
            'near_miss_accepted': {g: rate(s[np.isin(words, ws)] >= th) for g, ws in MSWC_GROUPS.items()},
            'equal_fa': {'false_accept_rate': round(ref_fa, 5), 'speech_commands_recall': round(sc_eq, 4),
                         'mswc_recall': round(mswc_eq, 4), 'recall_drop_points': round(100 * (sc_eq - mswc_eq), 2)}}
    # Continuous stream.
    if sets.get('stream') is not None:
        audio, marks, info = sets['stream']
        (times, score, _), = det.traces([audio])
        hours = info['seconds'] / 3600
        st = score_stream(times, score, th, marks, hours, info['segments'])
        curve = []
        # Candidate thresholds from per-second maxima: works for 250 ms and 10 ms steps alike.
        valid = score > np.iinfo(np.int64).min // 2
        sec = np.floor(times[valid]).astype(int)
        starts = np.r_[0, np.flatnonzero(np.diff(sec)) + 1]
        block_max = np.maximum.reduceat(score[valid], starts)
        for t in np.unique(np.quantile(block_max, np.linspace(.5, 1, 200))):
            r = score_stream(times, score, t, marks, hours)
            curve.append({'threshold': int(t), 'recall': r['recall'], 'fa_per_hour': r['fa_per_hour']})
        st['recall_at_fa_per_hour'] = {str(b): max([c['recall'] for c in curve if c['fa_per_hour'] <= b], default=0.0)
                                       for b in (0.5, 1, 2, 5, 10)}
        st['curve'] = curve
        out['stream'] = st
    # Synthesized-word probe.
    if sets.get('tts'):
        words, audios = sets['tts']
        s = clip_scores(det, audios)
        out['tts_confirmed'] = {g: rate(s[np.isin(words, ws)] >= th) for g, ws in TTS_GROUPS.items()}
    return out


def build_sets(data, split, probe, stream_seconds, skip):
    cfg = SPLITS[split]
    clips, y, rng = sc_live_clips(data, cfg['sc'], seed=cfg['seed'])
    noise = bg_noise(data)
    sets = {'live_y': y, 'live': {}}
    for cond in ('clean', 'mic', 'room', 'mic+room'):
        if cond != 'clean' and 'device' in skip:
            continue
        sets['live'][cond] = place_clips(clips, noise, rng, None if cond == 'clean' else cond)
    info = {'split': split, 'time_bins': N_TIME, 'live': {'yes': int(y.sum()), 'other': int((1 - y).sum())},
            'device': {'mic_profiles': channels.N_PROFILES, 'mic_seed': channels.HELDOUT_SEED,
                       'real_rirs': len(channels.heldout_rirs()) if 'device' not in skip else 0}}
    if 'corpus' not in skip:
        audios, words = mswc_clips(data, cfg['mswc'])
        words = np.array(words)
        clipped = np.array([edge_clipped(a) for a in audios])
        sets['mswc'] = (place_clips(audios, noise, np.random.default_rng(cfg['seed'] + 100)), words, clipped)
        info['mswc'] = {'split': cfg['mswc'], 'yes': int((words == 'yes').sum()),
                        'yes_edge_clipped': int(clipped[words == 'yes'].sum()),
                        'near_miss': {g: int(np.isin(words, ws).sum()) for g, ws in MSWC_GROUPS.items()},
                        'random_other': int((~np.isin(words, sum(MSWC_GROUPS.values(), []) + ['yes'])).sum())}
    if 'stream' not in skip:
        audio, marks, sinfo = build_stream(data, split, stream_seconds)
        sets['stream'] = (audio, marks, sinfo)
        info['stream'] = dict({k: v for k, v in sinfo.items() if k != 'segments'}, yes_inserted=len(marks),
                              holdoff_s=HOLDOFF, hit_tolerance_s=TOLERANCE)
    if 'tts' not in skip and split == 'test':
        sets['tts'] = tts_clips(data, probe)
        info['tts_utterances'] = len(sets['tts'][0]) // 4 if sets['tts'] else 0
    return sets, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('models', nargs='+', help='name=path/to/model.npz')
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--split', choices=SPLITS, default='test')
    p.add_argument('--probe', type=Path, default=ROOT / 'build/tts_probe')
    p.add_argument('--stream-seconds', type=int, default=STREAM_SECONDS)
    p.add_argument('--skip', nargs='*', default=[], choices=['device', 'corpus', 'stream', 'tts'])
    p.add_argument('--out', type=Path, default=ROOT / 'results/robust_baseline.json')
    a = p.parse_args()
    sets, info = build_sets(a.data, a.split, a.probe, a.stream_seconds, set(a.skip))
    print(json.dumps(info), flush=True)
    report = {'config': info, 'models': {}}
    for spec in a.models:
        name, path = spec.split('=', 1)
        r = evaluate(load_detector(path), sets)
        r['file'] = Path(path).as_posix()
        report['models'][name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != 'stream'}), flush=True)
        if 'stream' in r:
            print(name, 'stream', json.dumps({k: v for k, v in r['stream'].items() if k != 'curve'}), flush=True)
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
