"""Verifier track, step 4: offline cascade evaluation on validation data only.

Stage 1 (the streaming SNN, integer oracle) proposes; stage 2 (the phoneme
verifier, integer model) confirms. Board-faithful timing: frames arrive in
250 ms requests of 25 frames. After request k the firmware knows the stage-1
decision scores of its frames. If any of them reached t1 in request k or k-1,
the verifier scores the last `--window` frames (1.5 s, zero frames before the
stream start) ending with request k. At most one verification per request.
A detection is the verifier accepting (score >= t2); its time is the end of
request k plus the verifier run time (--verify-ms).

  cache    frames (stage 1's front end) and stage-1 raw scores for every
           validation set, once (data_verifier/cache_<split>[_<keyword>].npz):
           live clips (clean), MSWC dev, the 1 h stream, the 17.85 h
           negatives-only stream and the keyword-prefixed extras (policy b).
  score    verifier scores at every request end, policies a and b
           (verifier_model.keyword_score), for one quantized verifier.
  select   thresholds with stream_select.py's rule (live other-word accepts
           <= 0.2%, <= 2 FA/h on the negatives stream), stage 1 alone and
           the cascades; report recall, false accepts and latency.
  final    every metric at fixed thresholds (stage 1 alone and the cascade),
           no selection: the one-time test report (cache --split test first).
           Live recall also under the held-out microphones and rooms when the
           cache has them (cache --device, or --add-device to an existing cache).
  sources  stage 1 alone at looser thresholds (given live recalls): false
           accepts per hour on the negatives stream, what they are (Speech
           Commands word, or the LibriSpeech words around the detection),
           and the live other words accepted. This is what the verifier
           has to reject.

Policy (b) counts keyword-prefixed words as false accepts: MSWC "yesterday"
joins the other words, and the prefixed extras (LibriSpeech dev utterances
with a YES... word but no "yes", validation TTS/MSWC yes+letters clips) are
reported and their utterances join the negatives stream.

The keyword is keyword_config.KEYWORD (KWS_KEYWORD); the stage-1 model and
the cache follow it ("yes": the W=20 line, "sheila": the QAT candidate).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import keyword_config as K
import memguard
import robust_eval as R
from features import frame_features
from model import decision_scores, integer_forward_stream

ROOT = Path(__file__).resolve().parent
DV = ROOT / 'data_verifier'
MAIN = Path(r'C:\Users\matut\FULL_AI\5XPG0\code\snn_keyword\runs_stream')
STAGE1 = MAIN / {'yes': 's2_nokd_qat/int_last.npz', 'sheila': 'sheila_qat_seed2/int_model.npz'}[K.KEYWORD]
HOP = 25
NEG = np.iinfo(np.int64).min // 4


# --------------------------------------------------------------------------- cache

def cache_file(split):
    return DV / (f'cache_{split}.npz' if K.KEYWORD == 'yes' else f'cache_{split}_{K.KEYWORD}.npz')


def stage1_raw(q, feats, batch=16):
    out = [None] * len(feats)
    order = np.argsort([len(f) for f in feats])
    for i in range(0, len(order), batch):
        ids = order[i:i + batch]
        t = max(len(feats[k]) for k in ids)
        x = np.zeros((len(ids), t, 24), np.uint8)
        for j, k in enumerate(ids):
            x[j, :len(feats[k])] = feats[k]
        memguard.check('stage-1 cache')
        s, _, _ = integer_forward_stream(x, q)
        for j, k in enumerate(ids):
            out[k] = s[j, :len(feats[k])].astype(np.int32)
    return out


def prefixed_extras(data, split):
    """Keyword-prefixed negatives for policy (b): LibriSpeech dev utterances with a word that
    begins with the keyword and no keyword, and validation multi-corpus clips of such words, placed in noise."""
    import soundfile as sf
    utts = []
    for subset in R.NEG_LIBRI[split]:
        base = data / 'librispeech' / 'LibriSpeech' / subset
        for t in sorted(base.rglob('*.trans.txt')):
            for line in t.read_text().splitlines():
                uid, text = line.split(' ', 1)
                ws = text.split()
                if any(K.prefixed(w) for w in ws) and K.KEYWORD.upper() not in ws:
                    utts.append((uid, text, sf.read(t.parent / f'{uid}.flac', dtype='float32')[0]))
    rows = [r for r in csv.DictReader(open(K.MULTI / 'manifest.csv', encoding='utf-8'))
            if r['split'] == split and K.prefixed(r['word'])]
    audios = []
    if rows:
        clips = np.load(K.MULTI / f'clips_{split}.npy', mmap_mode='r')
        audios = [clips[int(r['row'])].astype(np.float32) / 32768 for r in rows]
    noise = R.bg_noise(data)
    placed = R.place_clips(audios, noise, np.random.default_rng(77)) if audios else []
    # Utterances: 0.5 s of noise before and 1 s after, as in the negatives stream.
    rng = np.random.default_rng(78)
    lib = []
    for _, _, a in utts:
        buf = np.zeros(len(a) + 24000, np.float32)
        buf[8000:8000 + len(a)] = a
        o = rng.integers(0, len(noise) - len(buf))
        lib.append(buf + noise[o:o + len(buf)] * rng.uniform(.02, .1))
    return lib, [u[1] for u in utts], placed, [f"{r['corpus']}:{r['word']}" for r in rows]


DEVICE = ('mic', 'room', 'mic+room')     # robust_eval live conditions besides clean


def cache(a):
    q = dict(np.load(a.stage1))
    K.check_model_keyword(q.get('keyword'), 'stage 1')
    front = str(q.get('frontend', 'logmel'))
    t0 = time.perf_counter()
    if a.add_device:   # only the live set under the held-out channels, into the existing cache
        out = dict(np.load(cache_file(a.split)))
        sets, _ = R.build_sets(a.data, a.split, None, R.STREAM_SECONDS, {'corpus', 'stream', 'tts'})
        assert (sets['live_y'] == out['live_y']).all()
        first = frame_features(sets['live']['clean'][0], front)    # same clips and placement as the cache
        assert (first == out['live_frames'][:len(first)]).all(), 'live clips differ from the cache'
    else:
        skip = {'tts'} | (set() if a.device else {'device'})
        sets, info = R.build_sets(a.data, a.split, None, R.STREAM_SECONDS, skip)
    def put(name, feats):   # a group may be empty ("sheila" has no keyword-prefixed words)
        raw = stage1_raw(q, feats)
        out[f'{name}_frames'] = np.concatenate(feats) if feats else np.zeros((0, 24), np.uint8)
        out[f'{name}_raw'] = np.concatenate(raw) if raw else np.zeros(0, np.int32)
        out[f'{name}_off'] = np.r_[0, np.cumsum([len(f) for f in feats])].astype(np.int64)
        print(name, len(feats), 'items', len(out[f'{name}_frames']), 'frames', round(time.perf_counter() - t0), 's', flush=True)
    if a.add_device:
        for cond in DEVICE:
            put(f'live_{cond}', [frame_features(x, front) for x in sets['live'][cond]])
        np.savez(cache_file(a.split), **out)
        print('added', DEVICE, round(time.perf_counter() - t0), 's')
        return
    groups = {'live': sets['live']['clean'], 'mswc': sets['mswc'][0], 'stream': [sets['stream'][0]]}
    groups.update({f'live_{cond}': sets['live'][cond] for cond in DEVICE if cond in sets['live']})
    lib, lib_text, pre, pre_words = prefixed_extras(a.data, a.split)
    groups['prefixed_libri'], groups['prefixed_clips'] = lib, pre
    out = {'split': np.array(a.split), 'live_y': sets['live_y'], 'live_clipped': sets['live_clipped'],
           'mswc_words': sets['mswc'][1],
           'mswc_clipped': sets['mswc'][2], 'stream_marks': sets['stream'][1],
           'decision_window': np.array(int(q.get('decision_window', 1))), 'keyword': np.array(K.KEYWORD),
           'stream_seconds': np.array(sets['stream'][2]['seconds']), 'prefixed_libri_text': np.array(lib_text),
           'prefixed_clip_words': np.array(pre_words)}
    for name, audios in groups.items():
        put(name, [frame_features(x, front) for x in audios])
    # Negatives stream, in its 600 s chunks (state reset per chunk, as robust_eval.negative_traces).
    feats, seconds, kinds = [], 0., []
    for audio, seg in R.negative_stream(a.data, a.split):
        seconds += len(audio) / 16000
        memguard.check('negatives stream')
        feats.append(frame_features(audio, front))
        kinds.append(json.dumps(seg))
    out['neg_seconds'] = np.array(seconds)
    out['neg_segments'] = np.array(kinds)
    put('neg', feats)
    out['stage1'] = np.array(str(a.stage1))
    np.savez(cache_file(a.split), **out)
    print('cached', round(time.perf_counter() - t0), 's')


# --------------------------------------------------------------------------- helpers

def split_items(c, name):
    off, fr, raw = c[f'{name}_off'], c[f'{name}_frames'], c[f'{name}_raw']   # each npz member read once
    return [(fr[off[i]:off[i + 1]], raw[off[i]:off[i + 1]]) for i in range(len(off) - 1)]


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


GROUPS = ('live', 'live_mic', 'live_room', 'live_mic+room', 'mswc', 'stream', 'neg', 'prefixed_libri', 'prefixed_clips')


def present(c):
    return [g for g in GROUPS if f'{g}_off' in c]


# --------------------------------------------------------------------------- verifier scores

def score(a):
    """Verifier scores (policies a and b) at the end of every 25-frame request of every cached item."""
    import torch
    from verifier_export import load_quantized
    from verifier_model import IntegerTorch, keyword_score_torch
    torch.set_num_threads(a.threads)
    q = load_quantized(a.checkpoint)
    model = IntegerTorch(q, 'cpu')
    c = dict(np.load(cache_file(a.split)))
    out, t0 = {}, time.perf_counter()
    for g in present(c):
        items = split_items(c, g)
        va, vb, off, pend = [], [], [0], []

        def flush():
            x = np.concatenate(pend)
            lg = model(x)
            va.append(keyword_score_torch(lg, a.warmup, 0, cap=a.cap).numpy().astype(np.int64))
            vb.append(keyword_score_torch(lg, a.warmup, a.boundary, cap=a.cap).numpy().astype(np.int64))
            pend.clear()
            memguard.check(f'score {g}')

        for frames, _ in items:
            w = windows(frames, hop_ends(len(frames)), a.window)
            off.append(off[-1] + len(w))
            for i in range(0, len(w), a.batch):
                pend.append(w[i:i + a.batch])
                if sum(len(x) for x in pend) >= a.batch:
                    flush()
        if pend:
            flush()
        empty = np.zeros(0, np.int64)
        out[f'{g}_va'], out[f'{g}_vb'] = (np.concatenate(va), np.concatenate(vb)) if va else (empty, empty)
        out[f'{g}_voff'] = np.array(off, np.int64)
        print(g, len(items), 'items', off[-1], 'windows', round(time.perf_counter() - t0), 's', flush=True)
    out.update(checkpoint=np.array(str(a.checkpoint)), warmup=np.array(a.warmup), boundary=np.array(a.boundary),
               window=np.array(a.window), cap=np.array(a.cap))
    np.savez(a.out, **out)


# --------------------------------------------------------------------------- firmware oracle

def cascade_requests(frames, q1, qv, t1, t2, window=1, warmup=5, boundary=10, vframes=150, holdoff=100):
    """Per 25-frame request, what the CASCADE firmware reports: (detected bits, score_a, score_b).

    bit0 detection (verifier score_a >= t2 while stage 1 reached t1 in this request or the
    previous one, >= holdoff frames after the previous detection, counted at request ends),
    bit1 stage 1 reached t1 in this request, bit2 the verifier ran. Scores are
    verifier_model.NEG when it did not run. Same decision as Cascade.traces('cascade'),
    with the hold-off in whole frames instead of float seconds.
    """
    from verifier_model import NEG as VNEG, integer_forward, keyword_score
    from model import integer_forward_stream
    s, _, _ = integer_forward_stream(np.asarray(frames, np.uint8)[None], q1)
    d = decision_scores(s[0].astype(np.int64), window)
    ends = hop_ends(len(frames))
    reach = np.maximum.reduceat(d, np.r_[0, ends[:-1] + 1]) >= t1
    run = reach | np.r_[False, reach[:-1]]
    sa = np.full(len(ends), VNEG, np.int64)
    sb = np.full(len(ends), VNEG, np.int64)
    if run.any():
        lg = integer_forward(windows(np.asarray(frames, np.uint8), ends[run], vframes), qv)
        sa[run], sb[run] = keyword_score(lg, warmup, 0), keyword_score(lg, warmup, boundary)
    out, last = [], None
    for k, e in enumerate(ends):
        bits = 2 * int(reach[k]) + 4 * int(run[k])
        if run[k] and sa[k] >= t2 and (last is None or e - last >= holdoff):
            bits |= 1
            last = e
        out.append((bits, int(sa[k]), int(sb[k])))
    return out


# --------------------------------------------------------------------------- selection

def hop_max(d):
    ends = hop_ends(len(d))
    starts = np.r_[0, ends[:-1] + 1]
    return np.maximum.reduceat(d, starts)


class Cascade:
    """Event times and scores per item for one detector configuration."""

    def __init__(self, c, v, w, verify_ms):
        self.c, self.v, self.w, self.verify_ms = c, v, w, verify_ms
        self.items = {g: split_items(c, g) for g in present(c) if f'{g}_voff' in v}
        self.d1 = {g: [decision_scores(r.astype(np.int64), w) for _, r in self.items[g]] for g in self.items}

    def traces(self, g, mode, t2=None, policy='a'):
        """mode 'frame': stage 1 alone per frame (stream_select.py); 'hop': stage 1 alone at request
        ends (board timing); 'cascade': stage 1 reached t1 in request k or k-1 and the verifier
        scores >= t2 on the window ending with request k."""
        out = []
        voff = self.v[f'{g}_voff']
        vv = self.v[f'{g}_v{policy}']
        for i, d in enumerate(self.d1[g]):
            n = len(d)
            if mode == 'frame':
                out.append((frame_times(n), d))
                continue
            ends = hop_ends(n)
            hm = hop_max(d)
            t = frame_times(n)[ends]
            if mode == 'hop':
                out.append((t, hm))
                continue
            m = np.maximum(hm, np.r_[NEG, hm[:-1]])
            ver = vv[voff[i]:voff[i + 1]]
            out.append((t + self.verify_ms / 1000, np.where(ver >= t2, m, NEG)))
        return out


def fa_hour(tr, hours, t):
    return sum(len(R.detections(tt, s, t)) for tt, s in tr) / hours


def operating(cas, mode, t2, policy, max_live_fa=.002, max_fa_hour=2.):
    c = cas.c
    y = c['live_y']
    live = np.array([s.max() for _, s in cas.traces('live', mode, t2, policy)])
    neg = cas.traces('neg', mode, t2, policy)
    hours = float(c['neg_seconds']) / 3600
    if policy == 'b':    # yes-prefixed LibriSpeech utterances are negatives too
        neg = neg + cas.traces('prefixed_libri', mode, t2, policy)
        hours += sum(len(f) for f, _ in cas.items['prefixed_libri']) * .01 / 3600
    lo = np.unique(live[y == 1])
    lo = lo[lo > NEG]
    if not len(lo):
        return None
    neg = [(t[s >= lo[0]], s[s >= lo[0]]) for t, s in neg]     # only events that can ever count
    best = None
    for t1 in lo[::-1]:
        fa = (live[y == 0] >= t1).mean()
        if fa > max_live_fa:
            break
        fph = fa_hour(neg, hours, t1)
        if fph > max_fa_hour:
            if fph > 5 * max_fa_hour:
                break
            continue
        rec = (live[y == 1] >= t1).mean()
        if best is None or rec > best['live_recall']:
            best = dict(t1=int(t1), t2=None if t2 is None else int(t2), live_recall=round(float(rec), 4),
                        live_fa=round(float(fa), 4), fa_per_hour=round(fph, 3), neg_hours=round(hours, 2))
            if 'live_clipped' in c:   # recall on keyword recordings that are not cut off
                best['live_recall_complete'] = round(float((live[(y == 1) & ~c['live_clipped']] >= t1).mean()), 4)
    if best is None:
        return None
    return metrics(cas, mode, best['t1'], t2, policy)


def rate(hits):
    return round(float(np.mean(hits)), 4) if len(hits) else None


def metrics(cas, mode, t1, t2, policy):
    """Every reported number at fixed thresholds (no selection)."""
    c = cas.c
    y = c['live_y']
    out = {'t1': int(t1), 't2': None if t2 is None else int(t2)}
    for g in [g for g in cas.items if g.startswith('live')]:
        s = np.array([x.max() for _, x in cas.traces(g, mode, t2, policy)])
        r = {'recall': rate(s[y == 1] >= t1), 'other_accepted': rate(s[y == 0] >= t1)}
        if 'live_clipped' in c:   # recall on keyword recordings that are not cut off
            r['recall_complete'] = rate(s[(y == 1) & ~c['live_clipped']] >= t1)
        if g == 'live':
            out.update(live_recall=r['recall'], live_recall_complete=r.get('recall_complete'),
                       live_fa=r['other_accepted'])
            out['_live_hits'] = s[y == 1] >= t1
        else:
            out[g] = r
    neg = cas.traces('neg', mode, t2, policy)
    hours = float(c['neg_seconds']) / 3600
    if policy == 'b' and 'prefixed_libri' in cas.items:    # prefixed LibriSpeech utterances are negatives too
        neg = neg + cas.traces('prefixed_libri', mode, t2, policy)
        hours += sum(len(f) for f, _ in cas.items['prefixed_libri']) * .01 / 3600
    out.update(fa_per_hour=round(fa_hour(neg, hours, t1), 3), neg_hours=round(hours, 2))
    (ts, ss), = cas.traces('stream', mode, t2, policy)
    st = R.score_stream(ts, ss, t1, c['stream_marks'], float(c['stream_seconds']) / 3600)
    out.update(stream_recall=st['recall'], stream_fa=st['false_accepts'], latency_median_s=st['latency_median_s'],
               latency_p90_s=st['latency_p90_s'])
    mw = c['mswc_words']
    ms = np.array([s.max() for _, s in cas.traces('mswc', mode, t2, policy)])
    pre = np.vectorize(K.prefixed)(mw) if len(mw) else np.zeros(0, bool)
    other = (mw != K.KEYWORD) & (~pre if policy == 'a' else True)
    out.update(mswc_recall=rate(ms[mw == K.KEYWORD] >= t1), mswc_other_fa=rate(ms[other] >= t1),
               mswc_prefixed_accepted=f'{int((ms[pre] >= t1).sum())}/{int(pre.sum())}')
    groups = {g: np.isin(mw, ws) for g, ws in K.MSWC_GROUPS.items()}
    groups['random other words'] = ~np.isin(mw, sum(K.MSWC_GROUPS.values(), []) + [K.KEYWORD])
    out['mswc_groups_accepted'] = {g: f'{int((ms[m] >= t1).sum())}/{int(m.sum())}' for g, m in groups.items()}
    if 'prefixed_clips' in cas.items:
        pc = np.array([s.max() for _, s in cas.traces('prefixed_clips', mode, t2, policy)])
        out['prefixed_clips_accepted'] = f'{int((pc >= t1).sum())}/{len(pc)}'
        pl = sum(len(R.detections(t, s, t1)) > 0 for t, s in cas.traces('prefixed_libri', mode, t2, policy))
        out['prefixed_libri_with_detection'] = f'{int(pl)}/{len(cas.items["prefixed_libri"])}'
    if mode == 'cascade':   # verifier calls per hour on the negatives at t1 (CPU load)
        calls = 0
        for d in cas.d1['neg']:
            hm = hop_max(d)
            calls += int((np.maximum(hm, np.r_[NEG, hm[:-1]]) >= t1).sum())
        out['verifier_calls_per_hour'] = round(calls / hours, 1)
    return out


def mcnemar(a_hits, b_hits):
    from scipy.stats import binomtest
    b01, b10 = int((~a_hits & b_hits).sum()), int((a_hits & ~b_hits).sum())
    p = binomtest(b01, b01 + b10).pvalue if b01 + b10 else 1.
    return {'only_cascade': b01, 'only_stage1': b10, 'p_exact': round(float(p), 4)}


def select(a):
    c = dict(np.load(cache_file(a.split)))
    v = dict(np.load(a.scores))
    report = {'scores': str(a.scores), 'verify_ms': a.verify_ms,
              'keyword': K.KEYWORD,
              'rule': 'live other words <= 0.2%, <= 2 FA/h on the negatives stream (policy b: + prefixed utterances)',
              'runs': []}
    y = c['live_y']
    for w in a.windows or [int(c.get('decision_window', 20))]:
        cas = Cascade(c, v, w, a.verify_ms)
        for policy in ('a', 'b'):
            rows = {'stage1_frame': operating(cas, 'frame', None, policy),
                    'stage1_hop': operating(cas, 'hop', None, policy)}
            vv, voff = v[f'live_v{policy}'], v['live_voff']
            vmax = np.array([vv[voff[i]:voff[i + 1]].max() for i in range(len(voff) - 1)])
            grid = np.unique(np.quantile(vmax[y == 1], np.linspace(0, .6, 31)).astype(np.int64))
            cands = []
            for t2 in grid[grid > NEG // 2]:
                r = operating(cas, 'cascade', int(t2), policy)
                if r:
                    cands.append(r)
                    print(w, policy, {k: x for k, x in r.items() if not k.startswith('_')}, flush=True)
            best = max(cands, key=lambda r: (r['live_recall'], r['stream_recall'])) if cands else None
            rows['cascade'] = best
            if best and rows['stage1_frame']:
                best['paired_vs_stage1_frame'] = mcnemar(rows['stage1_frame']['_live_hits'], best['_live_hits'])
            for r in list(rows.values()) + cands:
                if r:
                    r.pop('_live_hits', None)
            report['runs'].append({'window': w, 'policy': policy, **rows, 'grid': cands})
            print('W', w, 'policy', policy, json.dumps(rows), flush=True)
    a.out.write_text(json.dumps(report, indent=1, default=int))


def final(a):
    """All metrics at the thresholds chosen on validation: stage 1 alone and the cascade."""
    c = dict(np.load(cache_file(a.split)))
    v = dict(np.load(a.scores))
    if a.checkpoint_name and Path(str(v['checkpoint'])) != Path(a.checkpoint_name):
        raise SystemExit(f"scores are of {v['checkpoint']}, not {a.checkpoint_name}")
    cas = Cascade(c, v, a.window, a.verify_ms)
    rows = {'stage1_frame': metrics(cas, 'frame', a.stage1_threshold, None, a.policy),
            'stage1_hop': metrics(cas, 'hop', a.stage1_threshold, None, a.policy),
            'cascade': metrics(cas, 'cascade', a.t1, a.t2, a.policy)}
    rows['cascade']['paired_vs_stage1_frame'] = mcnemar(rows['stage1_frame']['_live_hits'], rows['cascade']['_live_hits'])
    for r in rows.values():
        r.pop('_live_hits')
    report = {'split': a.split, 'keyword': K.KEYWORD, 'stage1': str(c['stage1']), 'scores': str(a.scores),
              'verifier': str(v['checkpoint']), 'window': a.window, 'policy': a.policy, 'verify_ms': a.verify_ms,
              'fixed_thresholds': 'chosen on validation (select); nothing is selected here', **rows}
    a.out.write_text(json.dumps(report, indent=1, default=int))
    print(json.dumps(report, indent=1, default=int))


def libri_texts(data, split):
    out = {}
    for subset in R.NEG_LIBRI[split]:
        for t in (data / 'librispeech' / 'LibriSpeech' / subset).rglob('*.trans.txt'):
            for line in t.read_text().splitlines():
                uid, text = line.split(' ', 1)
                out[uid] = text
    return out


def describe(t, segments, texts, span=1.0):
    """The inserted item overlapping most with the last `span` s before t: SC word, or LibriSpeech
    words around the proportional position of t in the utterance (no alignment; a rough pointer)."""
    best, item = 0., None
    for s, e, kind, name in segments:
        o = min(e, t) - max(s, t - span)
        if o > best:
            best, item = o, (s, e, kind, name)
    if item is None:
        return 'noise', 'noise only'
    s, e, kind, name = item
    if kind == 'word':
        return 'word', name.replace('\\', '/').split('/')[0]
    words = texts.get(Path(name).stem, '?').split()
    i = int(np.clip((t - s) / max(e - s, 1e-3), 0, 1) * len(words))
    return 'libri', ' '.join(words[max(0, i - 4):i + 1]).lower()


def sources(a):
    import collections
    c = dict(np.load(cache_file(a.split)))
    w = a.window or int(c.get('decision_window', 1))
    y, clipped = c['live_y'], c.get('live_clipped')
    live = np.array([decision_scores(r.astype(np.int64), w).max() for _, r in split_items(c, 'live')])
    pos = np.sort(live[y == 1])[::-1]
    hours = float(c['neg_seconds']) / 3600
    segs = [json.loads(s) for s in c['neg_segments']]
    negd = [decision_scores(r.astype(np.int64), w) for _, r in split_items(c, 'neg')]
    texts = libri_texts(a.data, a.split)
    report = {'keyword': K.KEYWORD, 'stage1': str(c['stage1']), 'window': w, 'neg_hours': round(hours, 2), 'levels': []}
    for rec in a.recalls:
        t1 = int(pos[min(len(pos) - 1, int(np.ceil(rec * len(pos))) - 1)])
        events = []
        for d, seg in zip(negd, segs):
            for t in R.detections(frame_times(len(d)), d, t1):
                events.append(describe(t, seg, texts))
        kinds = collections.Counter(k for k, _ in events)
        row = {'live_recall': round(float((live[y == 1] >= t1).mean()), 4), 't1': t1,
               'live_other_accepted': round(float((live[y == 0] >= t1).mean()), 4),
               'fa_per_hour': round(len(events) / hours, 2), 'by_kind': dict(kinds),
               'sc_words': collections.Counter(n for k, n in events if k == 'word').most_common(12),
               'libri_examples': [n for k, n in events if k == 'libri'][:a.examples]}
        if clipped is not None:
            row['live_recall_complete'] = round(float((live[(y == 1) & ~clipped] >= t1).mean()), 4)
        report['levels'].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'libri_examples'}), flush=True)
    a.out.write_text(json.dumps(report, indent=1))


def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')     # the MSWC group names are in IPA
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cache')
    c.add_argument('--stage1', type=Path, default=STAGE1)
    c.add_argument('--data', type=Path, default=ROOT / 'data')
    c.add_argument('--split', choices=['validation', 'test'], default='validation',
                   help='test: only for the one-time final report')
    c.add_argument('--device', action='store_true', help='also the live set under held-out microphones and rooms')
    c.add_argument('--add-device', action='store_true', help='add only those to an existing cache')
    s = sub.add_parser('score')
    s.add_argument('checkpoint', type=Path)
    s.add_argument('--split', choices=['validation', 'test'], default='validation')
    s.add_argument('--window', type=int, default=150, help='frames (10 ms) given to the verifier')
    s.add_argument('--warmup', type=int, default=5, help='20 ms steps before a keyword may start')
    s.add_argument('--boundary', type=int, default=10, help='policy b: 20 ms steps without a new phoneme after S')
    s.add_argument('--cap', type=int, default=0, help='capped-margin score (verifier_model.step_costs), Q10 units')
    s.add_argument('--batch', type=int, default=2048)
    s.add_argument('--threads', type=int, default=4)
    s.add_argument('--out', type=Path, required=True)
    e = sub.add_parser('select')
    e.add_argument('scores', type=Path)
    e.add_argument('--split', choices=['validation'], default='validation')
    e.add_argument('--windows', type=int, nargs='+', help='stage-1 decision windows (default: the model\'s own)')
    e.add_argument('--verify-ms', type=float, default=0., help='verifier run time added to cascade detections')
    e.add_argument('--out', type=Path, required=True)
    f = sub.add_parser('final')
    f.add_argument('scores', type=Path)
    f.add_argument('--split', choices=['validation', 'test'], default='test')
    f.add_argument('--t1', type=int, required=True)
    f.add_argument('--t2', type=int, required=True)
    f.add_argument('--stage1-threshold', type=int, required=True, help='stage 1 alone (its own selection)')
    f.add_argument('--window', type=int, default=1)
    f.add_argument('--policy', choices=['a', 'b'], default='a')
    f.add_argument('--verify-ms', type=float, default=0.)
    f.add_argument('--checkpoint-name', help='refuse scores of another verifier')
    f.add_argument('--out', type=Path, required=True)
    o = sub.add_parser('sources')
    o.add_argument('--split', choices=['validation'], default='validation')
    o.add_argument('--data', type=Path, default=ROOT / 'data')
    o.add_argument('--window', type=int, help='stage-1 decision window (default: the model\'s own)')
    o.add_argument('--recalls', type=float, nargs='+', default=[.65, .7, .75, .8, .85, .9])
    o.add_argument('--examples', type=int, default=40, help='LibriSpeech examples per level')
    o.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    {'cache': cache, 'score': score, 'select': select, 'final': final, 'sources': sources}[a.cmd](a)


if __name__ == '__main__':
    main()
