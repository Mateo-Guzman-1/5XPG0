"""Reevaluate all encoding variables with Pedro R1's waveform/event protocol.

The pinned Pedro worktree supplies the unmodified corpus construction and
event scoring. The compact SNN keeps its own frontend and bit-exact arithmetic.
Three matched seeds and the original 12-epoch training recipe are used for
every configuration. No synthetic-word augmentation is used.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import torch
from numba import njit
import frontend_sweep as B
import extended_encoding_sweep as E
import export_extended_candidate as Q

HERE = Path(__file__).resolve().parent
KWS = HERE.parent


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf8')
    temp.replace(path)


def catalog():
    cfgs, groups = E.all_configs()
    cfgs = {c.key: c for c in cfgs}
    primary, pg = B.all_configs()
    lookup = {}
    for c in primary:
        ec = E.EncodingConfig(**asdict(c))
        cfgs[ec.key] = ec
        lookup[c.key] = ec.key
    groups.update({k: [lookup[v] for v in values] for k, values in pg.items()})
    for c in (replace(E.BASELINE, bands=16, high_hz=6500),
              replace(E.BASELINE, low_hz=40, high_hz=5120)):
        cfgs[c.key] = c
    return list(cfgs.values()), groups


def encodings(audio, configs, precomputed=None):
    """Share FFTs within a batch; otherwise identical to E.feature_batch."""
    fitted, spectra, masters, bands, pooled = {}, {}, {}, {}, {}
    if precomputed is not None:
        masters.update(precomputed)
    for c in configs:
        if c.duration_ms not in fitted:
            a = E.fit_duration(audio, c.duration_ms)
            fitted[c.duration_ms] = a, np.sqrt(np.mean(a * a, axis=1))
        a, rms = fitted[c.duration_ms]
        sk = c.duration_ms, c.fft_window_ms, c.fft_hop_ms
        if sk not in spectra:
            n, hop = c.fft_window_ms * 16, c.fft_hop_ms * 16
            if n not in masters:
                related = [q for q in configs if q.fft_window_ms == c.fft_window_ms]
                duration = max(q.duration_ms for q in related)
                # All swept durations/crop offsets are multiples of this 5 ms grid.
                master = E.fit_duration(audio, duration)
                frames = np.lib.stride_tricks.sliding_window_view(master, n, axis=1)[:, ::80]
                masters[n] = duration, np.abs(np.fft.rfft(frames * np.hanning(n).astype(np.float32), axis=2)).astype(np.float32)
            duration, spectrum = masters[n]
            offset = (duration - c.duration_ms) * 8 // 80
            count = (a.shape[1] - n) // hop + 1
            spectra[sk] = spectrum[:, offset:offset + count * (hop // 80):hop // 80]
        bk = sk, c.bands, c.low_hz, c.high_hz, c.band_mapping
        if bk not in bands:
            f = np.fft.rfftfreq(c.fft_window_ms * 16, 1 / 16000)
            bands[bk] = np.log1p(np.einsum('bf,ntf->nbt', E.band_weights(f, c), spectra[sk], optimize=True))
        pk = bk, c.time_bins
        if pk not in pooled:
            v = bands[bk]
            v = np.stack([v[:, :, g].mean(axis=2) for g in np.array_split(np.arange(v.shape[2]), c.time_bins)], axis=2)
            peak = v.max(axis=(1, 2), keepdims=True)
            pooled[pk] = np.divide(v, peak, out=np.zeros_like(v), where=peak > 0)
        activity = np.zeros_like(rms)
        active = rms > c.rms_gate
        activity[active] = np.clip(np.log(rms[active] / c.rms_gate) / np.log(c.rms_full_scale / c.rms_gate), 0, 1)
        v = pooled[pk] * activity[:, None, None]
        if c.feature_mode == 'absolute_delta':
            v = np.concatenate((v, np.abs(np.diff(v, axis=2, prepend=v[:, :, :1]))), axis=1)
        yield E.quantize(v, c).reshape(len(audio), -1).astype(np.float32)


@njit(cache=True, nogil=True)
def lif_counts(current, w2, b2):
    counts = np.zeros((len(current), 2), np.int32)
    for row in range(len(current)):
        m1 = np.zeros(48, np.int32)
        m2a, m2b = 0, 0
        for step in range(16):
            a, b = int(b2[0]), int(b2[1])
            for j in range(48):
                v = np.int32((np.int32(m1[j] * 230) >> 8) + current[row, j])
                if v >= 16320:
                    v -= 16320
                    a += int(w2[0, j])
                    b += int(w2[1, j])
                m1[j] = v
            m2a = np.int32((np.int32(m2a * 230) >> 8) + a)
            m2b = np.int32((np.int32(m2b * 230) >> 8) + b)
            if m2a >= 64:
                m2a -= 64
                counts[row, 0] += 1
            if m2b >= 64:
                m2b -= 64
                counts[row, 1] += 1
    return counts


def fast_counts(q, x):
    # Integer inputs and int8 weights: float64 product is exact below 2**53.
    xx = np.rint(x * 255).astype(np.float64)
    current = (xx @ q['w1'].astype(np.float64).T).astype(np.int32) + q['b1']
    return lif_counts(current, q['w2'], q['b2'])


def models(configs, out):
    paths, y, speakers = B.discover_dataset(KWS / 'data/speech_commands_v2_sheila_subset', 'sheila')
    split = B.dataset_split(KWS / 'data/speech_commands_v2_sheila_subset', paths, speakers)
    result, metadata = [], []
    for i, c in enumerate(configs):
        print('Models', i + 1, '/', len(configs), c.key, flush=True)
        x = E.get_features(paths, [c], KWS / 'data/extended_encoding_cache_sheila_v2', 64)[c.key]
        qs, rows = [], []
        for seed in range(3):
            path = out / 'models' / f'{c.key}_seed{seed}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                ck = torch.load(path, map_location='cpu', weights_only=True)
            else:
                run, state = E.train_one(x, y, split, c, seed, 12, 4096, 128, True)
                ck = {'run': run, 'state_dict': state}
                torch.save(ck, path)
            q = Q.quantize_weights(ck['state_dict'])
            va = fast_counts(q, x[split == 'val'])
            threshold = B.choose_threshold(y[split == 'val'], va[:, 1] - va[:, 0])
            te = fast_counts(q, x[split == 'test'])
            rows.append(dict(seed=seed, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             old_clip_threshold=int(threshold),
                             old_clip_test=B.metrics(y[split == 'test'], te[:, 1] - te[:, 0] >= threshold)))
            assert np.array_equal(fast_counts(q, x[:40]), Q.integer_counts(q, x[:40]))
            stress = np.r_[np.zeros((1, c.inputs)), np.ones((1, c.inputs)),
                           np.random.default_rng(seed).random((40, c.inputs))].astype(np.float32)
            assert np.array_equal(fast_counts(q, stress), Q.integer_counts(q, stress))
            qs.append(q)
        result.append(qs)
        metadata.append(rows)
    return result, metadata


def confirmed(m):
    c = np.full(len(m), -100, np.int16)
    if len(m) > 1:
        prev = m[:-1].copy()
        prev[1:] = np.maximum(m[1:-1], m[:-2])
        c[1:] = np.minimum(m[1:], prev)
    return c


def score_audio(audio, configs, qs):
    n = max(0, (len(audio) - 16000) // 4000 + 1)
    result = np.empty((len(configs), 3, n), np.int8)
    # Reuse interior STFT frames between overlapping one-second requests.
    # Frames crossing a request boundary are computed from its own zero padding.
    global_spectra = {}
    durations = {c.fft_window_ms * 16: max(q.duration_ms for q in configs if q.fft_window_ms == c.fft_window_ms) for c in configs}
    for width in durations:
        frames = np.lib.stride_tricks.sliding_window_view(audio, width)[::80]
        spec = np.empty((len(frames), width // 2 + 1), np.float32)
        window = np.hanning(width).astype(np.float32)
        for k in range(0, len(frames), 8192):
            spec[k:k + 8192] = np.abs(np.fft.rfft(frames[k:k + 8192] * window, axis=1)).astype(np.float32)
        global_spectra[width] = spec
    def score_part(start):
        stop = min(start + 128, n)
        batch = np.stack([audio[k * 4000:k * 4000 + 16000] for k in range(start, stop)])
        masters = {}
        for width, duration in durations.items():
            fitted = E.fit_duration(batch, duration)
            offset = (16000 - fitted.shape[1]) // 2
            positions = np.arange(0, fitted.shape[1] - width + 1, 80) + offset
            master = np.zeros((len(batch), len(positions), width // 2 + 1), np.float32)
            inside = (positions >= 0) & (positions + width <= 16000)
            indices = np.arange(start, stop)[:, None] * 50 + positions[None, inside] // 80
            master[:, inside] = global_spectra[width][indices]
            boundary = ~inside & (positions + width > 0) & (positions < 16000)
            if boundary.any():
                frames = np.lib.stride_tricks.sliding_window_view(fitted, width, axis=1)[:, ::80][:, boundary]
                master[:, boundary] = np.abs(np.fft.rfft(frames * np.hanning(width).astype(np.float32), axis=2)).astype(np.float32)
            masters[width] = duration, master
        part = np.empty((len(configs), 3, stop - start), np.int8)
        for i, x in enumerate(encodings(batch, configs, masters)):
            for s, q in enumerate(qs[i]):
                counts = fast_counts(q, x)
                part[i, s] = counts[:, 1] - counts[:, 0]
        return start, part
    with ThreadPoolExecutor(4) as pool:
        for start, part in pool.map(score_part, range(0, n, 128)):
            result[:, :, start:start + part.shape[2]] = part
    return result


def build_scores(R, data, configs, qs, out, splits=('validation', 'test')):
    f = out / 'clips_test.npz'
    if 'test' in splits and not f.exists():
        index = np.load(data / 'features.npz')
        names = index['names'][index['split'] == 2]
        margins = np.empty((len(configs), 3, len(names)), np.int8)
        for start in range(0, len(names), 128):
            batch = np.stack([E.read_pcm16(data / 'speech_commands_v0.02' / str(n)) for n in names[start:start + 128]])
            for i, x in enumerate(encodings(batch, configs)):
                for s, q in enumerate(qs[i]):
                    cc = fast_counts(q, x)
                    margins[i, s, start:start + len(batch)] = cc[:, 1] - cc[:, 0]
            if start % 1280 == 0:
                print('Full test clips', start, '/', len(names), flush=True)
        np.savez(f, margin=margins, names=names, y=np.array([str(n).startswith('sheila/') for n in names]))
    for split in splits:
        f = out / f'live_{split}.npz'
        if not f.exists():
            spec = R.SPLITS[split]
            clips, labels, rng = R.sc_live_clips(data, spec['sc'], seed=spec['seed'])
            audios = R.place_clips(clips, R.bg_noise(data), rng)
            # Batch clips together for FFT efficiency, then split before confirmation.
            arrays, offsets = [], [0]
            for start in range(0, len(audios), 32):
                chunk = audios[start:start + 32]
                ws = [np.stack([a[k:k + 16000] for k in range(0, len(a) - 16000 + 1, 4000)]) for a in chunk]
                features = list(encodings(np.concatenate(ws), configs))
                margins = np.empty((len(configs), 3, sum(map(len, ws))), np.int8)
                for i, x in enumerate(features):
                    for s, q in enumerate(qs[i]):
                        cc = fast_counts(q, x)
                        margins[i, s] = cc[:, 1] - cc[:, 0]
                arrays.append(margins)
                for w in ws:
                    offsets.append(offsets[-1] + len(w))
                if start % 320 == 0:
                    print('Live', split, start, '/', len(audios), flush=True)
            np.savez(f, margin=np.concatenate(arrays, axis=2), off=offsets, y=labels,
                     clipped=np.array([R.edge_clipped(c) for c in clips]))
        manifest = out / f'negatives_{split}.json'
        if not manifest.exists():
            rows = []
            for j, (audio, segments) in enumerate(R.negative_stream(data, split)):
                f = out / f'negative_{split}_{j:03d}.npy'
                if not f.exists():
                    np.save(f, score_audio(audio, configs, qs))
                rows.append(dict(file=f.name, seconds=len(audio) / 16000))
                print('Negatives', split, j, 'hours', round(sum(r['seconds'] for r in rows) / 3600, 2), flush=True)
            dump(manifest, rows)
    f = out / 'stream_test.npz'
    if 'test' in splits and not f.exists():
        audio, marks, info = R.build_stream(data, 'test')
        np.savez(f, margin=score_audio(audio, configs, qs), marks=marks)
        dump(out / 'stream_test_info.json', info)


def summarize(R, configs, groups, meta, out):
    lives = {s: dict(np.load(out / f'live_{s}.npz')) for s in ('validation', 'test')}
    negatives, hours = {}, {}
    for split in lives:
        rows = json.loads((out / f'negatives_{split}.json').read_text())
        negatives[split] = [np.load(out / r['file'], mmap_mode='r') for r in rows]
        hours[split] = sum(r['seconds'] for r in rows) / 3600
    stream = np.load(out / 'stream_test.npz')
    clips = np.load(out / 'clips_test.npz')
    results = []
    for i, c in enumerate(configs):
        runs = []
        for seed in range(3):
            peaks = {s: np.array([confirmed(d['margin'][i, seed, a:b]).max() for a, b in zip(d['off'][:-1], d['off'][1:])]) for s, d in lives.items()}
            traces = {s: [confirmed(a[i, seed]) for a in negatives[s]] for s in lives}
            def fph(split, t):
                return sum(len(R.detections(1 + np.arange(len(v)) * .25, v, t)) for v in traces[split]) / hours[split]
            candidates = []
            for t in range(-16, 18):
                other = float((peaks['validation'][lives['validation']['y'] == 0] >= t).mean())
                if other > .002:
                    continue
                fa = fph('validation', t)
                if fa <= 2:
                    recall = float((peaks['validation'][lives['validation']['y'] == 1] >= t).mean())
                    candidates.append((recall, t, other, fa))
            # Pedro resolves equal recall towards the highest threshold.
            recall, t, other, fa = max(candidates)
            trace = confirmed(stream['margin'][i, seed])
            old = meta[i][seed]['old_clip_threshold']
            runs.append(dict(meta[i][seed], threshold=t,
                             validation=dict(recall=recall, other_acceptance=other, fa_per_hour=fa),
                             live_test=B.metrics(lives['test']['y'], peaks['test'] >= t),
                             complete_live_recall=float((peaks['test'][(lives['test']['y'] == 1) & ~lives['test']['clipped']] >= t).mean()),
                             full_clip_test=B.metrics(clips['y'], clips['margin'][i, seed] >= old),
                             test_fa_per_hour=fph('test', t),
                             old_threshold_live_test=B.metrics(lives['test']['y'], peaks['test'] >= old),
                             old_threshold_test_fa_per_hour=fph('test', old),
                             stream=R.score_stream(1 + np.arange(len(trace)) * .25, trace, t, stream['marks'], 1)))
        results.append(dict(key=c.key, config=asdict(c), input_bits=c.inputs * c.bits,
                            parameter_bytes=48 * c.inputs + 296, runs=runs))
    dump(out / 'summary.json', dict(groups=groups, negative_hours=hours, configurations=results))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pedro', type=Path, required=True)
    p.add_argument('--out', type=Path, default=HERE / 'results/pedro_encoding')
    p.add_argument('--models-only', action='store_true')
    p.add_argument('--split', choices=['validation', 'test'], help='Run one independent waveform partition; summarize after both complete.')
    p.add_argument('--summarize-only', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    configs, groups = catalog()
    a.out.mkdir(parents=True, exist_ok=True)
    kws = a.pedro / 'code/snn_keyword'
    if a.summarize_only:
        sys.path[:0] = [str(kws / 'research/tools/mateo_eval'), str(kws)]
        import robust_eval as R
        summarize(R, configs, groups, json.loads((a.out / 'models.json').read_text()), a.out)
        return
    commit = subprocess.check_output(['git', '-C', str(a.pedro), 'rev-parse', 'HEAD'], text=True).strip()
    protocol = dict(pedro_commit=commit, config_keys=[c.key for c in configs], seeds=[0, 1, 2],
                    hop_ms=250, outer_window_ms=1000, confirmation='current and one of previous two',
                    holdoff_s=1, validation_other_limit=.002, validation_fa_per_hour_limit=2,
                    epochs=12, samples_per_epoch=4096, training='original real Speech Commands subset',
                    duration='original centre crop / zero pad within the one-second request')
    protocol['source_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in (Path(__file__), Path(B.__file__), Path(E.__file__), Path(Q.__file__))}
    protocol['versions'] = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__)
    dump(a.out / 'protocol.json', protocol)
    qs, meta = models(configs, a.out)
    dump(a.out / 'models.json', meta)
    rng = np.random.default_rng(12)
    probe = rng.normal(0, .03, (5, 16000)).astype(np.float32)
    for c, x in zip(configs, encodings(probe, configs)):
        assert np.array_equal(np.rint(x * 255), np.rint(E.feature_batch(probe, c) * 255)), c.key
    print('PASS: shared FFT frontend and integer forward gates', flush=True)
    if a.models_only:
        return
    sys.path[:0] = [str(kws / 'research/tools/mateo_eval'), str(kws)]
    import robust_eval as R
    build_scores(R, kws / 'data', configs, qs, a.out, (a.split,) if a.split else ('validation', 'test'))
    if a.split:
        return
    summarize(R, configs, groups, meta, a.out)
    print('Complete', a.out / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
