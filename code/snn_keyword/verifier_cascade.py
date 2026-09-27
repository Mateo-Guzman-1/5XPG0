"""Verifier track, step 4: offline cascade evaluation on validation data only.

Stage 1 (the streaming SNN, integer oracle) proposes; stage 2 (the phoneme
verifier, integer model) confirms. Board-faithful timing: frames arrive in
250 ms requests of 25 frames. After request k the firmware knows the stage-1
decision scores of its frames. If any of them reached t1 in request k or k-1,
the verifier scores the last `--window` frames (1.5 s, zero frames before the
stream start) ending with request k. At most one verification per request.
A detection is the verifier accepting (score >= t2); its time is the end of
request k plus the verifier run time (--verify-ms).

  cache    frames (logmel) and stage-1 raw scores for every validation set,
           once (data_verifier/cache_<split>.npz): live clips (clean),
           MSWC dev, the 1 h stream, the 17.85 h negatives-only stream and
           the yes-prefixed extras (policy b).
  score    verifier scores at every request end, policies a and b
           (verifier_model.keyword_score), for one quantized verifier.
  select   thresholds with stream_select.py's rule (live other-word accepts
           <= 0.2%, <= 2 FA/h on the negatives stream), stage 1 alone and
           the cascades; report recall, false accepts and latency.

Policy (b) counts yes-prefixed words as false accepts: MSWC "yesterday"
joins the other words, and the yes-prefixed extras (LibriSpeech dev
utterances with a YES... word but no "yes", validation TTS/MSWC yes+letters
clips) are reported and their utterances join the negatives stream.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import robust_eval as R
from features import frame_features
from model import decision_scores, integer_forward_stream

ROOT = Path(__file__).resolve().parent
DV = ROOT / 'data_verifier'
STAGE1 = Path(r'C:\Users\matut\FULL_AI\5XPG0\code\snn_keyword\runs_stream\s2_nokd_qat\int_last.npz')
HOP = 25
NEG = np.iinfo(np.int64).min // 4


# --------------------------------------------------------------------------- cache

def stage1_raw(q, feats, batch=64):
    out = [None] * len(feats)
    order = np.argsort([len(f) for f in feats])
    for i in range(0, len(order), batch):
        ids = order[i:i + batch]
        t = max(len(feats[k]) for k in ids)
        x = np.zeros((len(ids), t, 24), np.uint8)
        for j, k in enumerate(ids):
            x[j, :len(feats[k])] = feats[k]
        s, _, _ = integer_forward_stream(x, q)
        for j, k in enumerate(ids):
            out[k] = s[j, :len(feats[k])].astype(np.int32)
    return out


def prefixed_extras(data, split):
    """Yes-prefixed negatives for policy (b): LibriSpeech dev utterances with a YES... word
    and no YES, and validation multi-corpus clips of yes-prefixed words, placed in noise."""
    import soundfile as sf
    utts = []
    for subset in R.NEG_LIBRI[split]:
        base = data / 'librispeech' / 'LibriSpeech' / subset
        for t in sorted(base.rglob('*.trans.txt')):
            for line in t.read_text().splitlines():
                uid, text = line.split(' ', 1)
                ws = text.split()
                if any(R.yes_prefixed(w) for w in ws) and 'YES' not in ws:
                    utts.append((uid, text, sf.read(t.parent / f'{uid}.flac', dtype='float32')[0]))
    rows = [r for r in csv.DictReader(open(data / 'multi/manifest.csv', encoding='utf-8'))
            if r['split'] == 'validation' and R.yes_prefixed(r['word'])]
    clips = np.load(data / 'multi/clips_validation.npy', mmap_mode='r')
    audios = [clips[int(r['row'])].astype(np.float32) / 32768 for r in rows]
    noise = R.bg_noise(data)
    placed = R.place_clips(audios, noise, np.random.default_rng(77))
    # Utterances: 0.5 s of noise before and 1 s after, as in the negatives stream.
    rng = np.random.default_rng(78)
    lib = []
    for _, _, a in utts:
        buf = np.zeros(len(a) + 24000, np.float32)
        buf[8000:8000 + len(a)] = a
        o = rng.integers(0, len(noise) - len(buf))
        lib.append(buf + noise[o:o + len(buf)] * rng.uniform(.02, .1))
    return lib, [u[1] for u in utts], placed, [f"{r['corpus']}:{r['word']}" for r in rows]


def cache(a):
    q = dict(np.load(a.stage1))
    front = str(q.get('frontend', 'logmel'))
    t0 = time.perf_counter()
    sets, info = R.build_sets(a.data, a.split, None, R.STREAM_SECONDS, {'device', 'tts'})
    groups = {'live': sets['live']['clean'], 'mswc': sets['mswc'][0], 'stream': [sets['stream'][0]]}
    lib, lib_text, pre, pre_words = prefixed_extras(a.data, a.split)
    groups['prefixed_libri'], groups['prefixed_clips'] = lib, pre
    out = {'live_y': sets['live_y'], 'mswc_words': sets['mswc'][1], 'stream_marks': sets['stream'][1],
           'stream_seconds': np.array(sets['stream'][2]['seconds']), 'prefixed_libri_text': np.array(lib_text),
           'prefixed_clip_words': np.array(pre_words)}
    def put(name, feats):
        raw = stage1_raw(q, feats)
        out[f'{name}_frames'] = np.concatenate(feats)
        out[f'{name}_raw'] = np.concatenate(raw)
        out[f'{name}_off'] = np.r_[0, np.cumsum([len(f) for f in feats])].astype(np.int64)
        print(name, len(feats), 'items', len(out[f'{name}_frames']), 'frames', round(time.perf_counter() - t0), 's', flush=True)
    for name, audios in groups.items():
        put(name, [frame_features(x, front) for x in audios])
    # Negatives stream, in its 600 s chunks (state reset per chunk, as robust_eval.negative_traces).
    feats, seconds, kinds = [], 0., []
    for audio, seg in R.negative_stream(a.data, a.split):
        seconds += len(audio) / 16000
        feats.append(frame_features(audio, front))
        kinds.append(json.dumps(seg))
    out['neg_seconds'] = np.array(seconds)
    out['neg_segments'] = np.array(kinds)
    put('neg', feats)
    out['stage1'] = np.array(str(a.stage1))
    np.savez(DV / f'cache_{a.split}.npz', **out)
    print('cached', round(time.perf_counter() - t0), 's')


# --------------------------------------------------------------------------- helpers

def split_items(c, name):
    off = c[f'{name}_off']
    return [(c[f'{name}_frames'][off[i]:off[i + 1]], c[f'{name}_raw'][off[i]:off[i + 1]]) for i in range(len(off) - 1)]


def hop_ends(n):
    """Last frame index of every 25-frame request (the last may be partial)."""
    return np.minimum(np.arange(HOP - 1, n + HOP - 1, HOP), n - 1)


def frame_times(n):
    return (np.arange(n) * 160 + 400) / 16000


def windows(frames, ends, length):
    """(len(ends), length, 24) uint8: frames up to and including each end, zero-padded before the start."""
    pad = np.concatenate([np.zeros((length, 24), np.uint8), frames])
    idx = ends[:, None] + 1 + np.arange(length)[None]   # in padded coordinates: end + length is frame `end`
    return pad[idx]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cache')
    c.add_argument('--stage1', type=Path, default=STAGE1)
    c.add_argument('--data', type=Path, default=ROOT / 'data')
    c.add_argument('--split', choices=['validation'], default='validation')   # never the test split
    a = p.parse_args()
    if a.cmd == 'cache':
        cache(a)


if __name__ == '__main__':
    main()
