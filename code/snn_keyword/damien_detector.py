"""Port of the teammate's deployed "sheila" window model (branch damien-dicking-around).

Evaluates his 8x16 grouped-FFT front end + two-layer integer LIF SNN under OUR
validation rule (stream_select.select), so it can be compared with the
streaming SNN on the same sets.

Source files on his branch (read with git show, never checked out):
  code/snn_keyword/audio_features.py                 extract_features / quantize_features (live demo path)
  code/snn_keyword/research/frontend_sweep.py        feature_batch (the training / board-vector path)
  code/pynqz2_riscv_flow/firmware/keyword_main.c     classify() and the decision rule
  code/snn_keyword/research/results/sheila_v2_deployment/keyword_model.h, board_vectors.npz
  code/snn_keyword/pc_keyword_demo.py                1 s window every HOP_MS = 250 ms, no confirmation

  python damien_detector.py verify     bit-exactness against his 40 board vectors
  python damien_detector.py evaluate   our rule on validation data -> results/sheila_damien_ourrule.json
                                       (run with KWS_KEYWORD=sheila)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BRANCH = 'upstream/damien-dicking-around'
DEPLOY = 'code/snn_keyword/research/results/sheila_v2_deployment'

# --------------------------------------------------------------------------- his front end (verbatim)
# Copied from audio_features.py on his branch; this is what pc_keyword_demo.py sends to the board.

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = SAMPLE_RATE
FFT_WINDOW = 400          # 25 ms
FFT_HOP = 160             # 10 ms
N_MELS = 8                # selected compact frequency-band count
N_TIME = 16               # fixed board input length
N_INPUTS = N_MELS * N_TIME
LOW_HZ = 40
HIGH_HZ = 5120
RMS_GATE = 0.003
RMS_FULL_SCALE = 0.040
HOP_MS = 250              # pc_keyword_demo.py
HOP = SAMPLE_RATE * HOP_MS // 1000


def _fit_one_second(wav: np.ndarray) -> np.ndarray:
    """Return exactly one second of mono float32 audio."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size >= WINDOW_SAMPLES:
        start = (wav.size - WINDOW_SAMPLES) // 2
        return wav[start:start + WINDOW_SAMPLES]
    padding = WINDOW_SAMPLES - wav.size
    left = padding // 2
    return np.pad(wav, (left, padding - left))


def read_wav(path) -> np.ndarray:
    """His audio_features.read_wav for 16 kHz PCM16 mono (the only format used here)."""
    with wave.open(str(path), 'rb') as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f'expected mono 16 kHz PCM16: {path}')
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0


def extract_features(wav: np.ndarray) -> np.ndarray:
    """His audio_features.extract_features: 8x16 map in [0, 1] ([band, time])."""
    wav = _fit_one_second(wav)
    rms = float(np.sqrt(np.mean(wav * wav)))
    if rms <= RMS_GATE:
        return np.zeros((N_MELS, N_TIME), dtype=np.float32)

    activity = np.log(rms / RMS_GATE) / np.log(RMS_FULL_SCALE / RMS_GATE)
    activity = float(np.clip(activity, 0.0, 1.0))
    starts = range(0, WINDOW_SAMPLES - FFT_WINDOW + 1, FFT_HOP)
    frames = np.stack([wav[i:i + FFT_WINDOW] for i in starts])
    spectrum = np.abs(
        np.fft.rfft(frames * np.hanning(FFT_WINDOW), axis=1)
    )

    frequencies = np.fft.rfftfreq(FFT_WINDOW, 1.0 / SAMPLE_RATE)
    selected = np.flatnonzero(
        (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)
    )
    groups = np.array_split(selected, N_MELS)
    bands = np.stack(
        [spectrum[:, group].mean(axis=1) for group in groups], axis=0
    )
    bands = np.log1p(bands)  # [frequency, short-time frame]

    frame_groups = np.array_split(np.arange(bands.shape[1]), N_TIME)
    pooled = np.stack(
        [bands[:, group].mean(axis=1) for group in frame_groups],
        axis=1,
    )
    peak = float(pooled.max())
    if peak > 0.0:
        pooled /= peak
    pooled *= activity
    return pooled.astype(np.float32)


def quantize_features(features: np.ndarray) -> np.ndarray:
    """His audio_features.quantize_features: uint8 board input."""
    a = np.asarray(features, dtype=np.float32)
    if a.shape != (N_MELS, N_TIME):
        raise ValueError(f'expected {(N_MELS, N_TIME)}, got {a.shape}')
    return np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


def damien_features(wav: np.ndarray) -> np.ndarray:
    """1 s of 16 kHz float32 audio -> 128 uint8, his layout (band-major, C order), as sent to the board."""
    return quantize_features(extract_features(wav)).reshape(-1)


def training_features(wav: np.ndarray) -> np.ndarray:
    """His research/frontend_sweep.feature_batch (b8_f8_hz40-5120_t1000), the path that made
    the training cache and board_vectors.npz; only used to explain any LSB differences."""
    audio = np.asarray(wav, np.float32)[None]
    rms = np.sqrt(np.mean(audio * audio, axis=1))
    activity = np.zeros_like(rms)
    active = rms > RMS_GATE
    activity[active] = np.clip(np.log(rms[active] / RMS_GATE) / np.log(RMS_FULL_SCALE / RMS_GATE), 0.0, 1.0)
    frames = np.lib.stride_tricks.sliding_window_view(audio, FFT_WINDOW, axis=1)[:, ::FFT_HOP, :]
    window = np.hanning(FFT_WINDOW).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=2)).astype(np.float32)
    frequencies = np.fft.rfftfreq(FFT_WINDOW, 1.0 / SAMPLE_RATE)
    selected = np.flatnonzero((frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ))
    groups = np.array_split(selected, N_MELS)
    band_values = np.stack([spectrum[:, :, g].mean(axis=2) for g in groups], axis=1)
    band_values = np.log1p(band_values)
    frame_groups = np.array_split(np.arange(band_values.shape[2]), N_TIME)
    pooled = np.stack([band_values[:, :, g].mean(axis=2) for g in frame_groups], axis=2)
    peak = pooled.max(axis=(1, 2), keepdims=True)
    pooled = np.divide(pooled, peak, out=np.zeros_like(pooled), where=peak > 0)
    pooled *= activity[:, None, None]
    pooled = np.rint(np.clip(pooled, 0.0, 1.0) * 255) / 255
    return np.rint(pooled.reshape(-1) * 255).astype(np.uint8)   # as export_frontend_candidate stores x


def read_pcm16_training(path) -> np.ndarray:
    """frontend_sweep.read_pcm16: first second, or centred zero padding."""
    source = read_wav(path)
    result = np.zeros(SAMPLE_RATE, np.float32)
    if len(source) >= SAMPLE_RATE:
        result[:] = source[:SAMPLE_RATE]
    else:
        start = (SAMPLE_RATE - len(source)) // 2
        result[start:start + len(source)] = source
    return result


# --------------------------------------------------------------------------- his firmware (integer oracle)

def branch_file(name, path=DEPLOY) -> bytes:
    """A file of his branch, read with git show (nothing is checked out or written)."""
    return subprocess.run(['git', 'show', f'{BRANCH}:{path}/{name}'], cwd=ROOT, check=True,
                          capture_output=True).stdout


def parse_header(text=None) -> dict:
    """KW_* constants and the integer weights of his keyword_model.h."""
    if text is None:
        text = branch_file('keyword_model.h').decode('ascii')
    m = {k: v for k, v in re.findall(r'#define\s+(KW_\w+)\s+(-?\d+)\s*$', text, re.M)}
    q = {k: int(v) for k, v in m.items()}
    q['KW_KEYWORD'] = re.search(r'#define\s+KW_KEYWORD\s+"(\w+)"', text).group(1)
    for name, dtype in (('kw_w1', np.int8), ('kw_b1', np.int32), ('kw_w2', np.int8), ('kw_b2', np.int32)):
        body = re.search(rf'{name}\[[^\]]*\]\s*=\s*\{{(.*?)\}};', text, re.S).group(1)
        vals = np.array([int(v) for v in body.replace('\n', ' ').split(',') if v.strip()], np.int64)
        info = np.iinfo(dtype)
        assert vals.min() >= info.min and vals.max() <= info.max, name
        q[name] = vals.astype(dtype)
    H, I, O = q['KW_HIDDEN'], q['KW_INPUTS'], q['KW_OUTPUTS']
    q['kw_w1'] = q['kw_w1'].reshape(H, I)
    q['kw_w2'] = q['kw_w2'].reshape(O, H)
    assert q['kw_b1'].shape == (H,) and q['kw_b2'].shape == (O,)
    return q


def _wrap32(x):
    """int64 -> the value an RV32 s32 register holds (two's-complement wrap)."""
    return ((x + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


def classify(x: np.ndarray, q: dict) -> tuple[np.ndarray, int]:
    """keyword_main.c classify() on uint8 inputs [N, 128]. Returns spike counts [N, 2] and the
    number of s32 overflows seen (C wraps; they are emulated, but should never happen)."""
    x = np.asarray(x, np.int64).reshape(-1, q['KW_INPUTS'])
    beta, th1, th2 = q['KW_BETA_Q8'], q['KW_THRESHOLD1'], q['KW_THRESHOLD2']
    w1, w2 = q['kw_w1'].astype(np.int64), q['kw_w2'].astype(np.int64)
    current1 = x @ w1.T + q['kw_b1'].astype(np.int64)
    overflow = int(np.count_nonzero(current1 != _wrap32(current1)))
    current1 = _wrap32(current1)
    mem1 = np.zeros_like(current1)
    mem2 = np.zeros((len(x), q['KW_OUTPUTS']), np.int64)
    score = np.zeros_like(mem2)
    for _ in range(q['KW_TIMESTEPS']):
        p = mem1 * beta
        overflow += int(np.count_nonzero(p != _wrap32(p)))
        mem1 = _wrap32((_wrap32(p) >> 8) + current1)
        spike1 = mem1 >= th1
        mem1 = mem1 - spike1 * th1                     # soft reset
        current2 = spike1.astype(np.int64) @ w2.T + q['kw_b2'].astype(np.int64)
        p = mem2 * beta
        overflow += int(np.count_nonzero(p != _wrap32(p)))
        mem2 = _wrap32((_wrap32(p) >> 8) + current2)
        spike2 = mem2 >= th2
        score += spike2
        mem2 = mem2 - spike2 * th2
    return score.astype(np.int32), overflow


def decision(score: np.ndarray, q: dict) -> np.ndarray:
    """keyword_main.c: score[1] >= KW_MIN_KEYWORD_SPIKES && score[1] >= score[0] + KW_DECISION_MARGIN."""
    return (score[:, 1] >= q['KW_MIN_KEYWORD_SPIKES']) & (score[:, 1] >= score[:, 0] + q['KW_DECISION_MARGIN'])


# --------------------------------------------------------------------------- detector (robust_eval interface)

class DamienDetector:
    """His live demo: a 1 s window every 250 ms, each window decided on its own (no confirmation).
    Score = keyword spikes - other spikes (his margin); decision score = raw score."""
    kind = 'window'
    _cache = {}

    def __init__(self, q=None, workers=4):
        if q is None:
            q = parse_header()
        self.q = q
        self.workers = workers
        self.overflows = 0
        # His rule is margin >= KW_DECISION_MARGIN with KW_MIN_KEYWORD_SPIKES = 0 (always true:
        # counts are >= 0), so thresholding the margin reproduces the firmware decision exactly.
        assert q['KW_MIN_KEYWORD_SPIKES'] <= 0
        self.threshold = int(q['KW_DECISION_MARGIN'])
        self.single_threshold = int(q['KW_DECISION_MARGIN'])

    def window_features(self, audio):
        audio = np.asarray(audio, np.float32)
        key = hashlib.blake2b(audio.tobytes(), digest_size=16).digest()
        if key not in self._cache:
            starts = range(0, len(audio) - SAMPLE_RATE + 1, HOP)
            self._cache[key] = (np.stack([damien_features(audio[s:s + SAMPLE_RATE]) for s in starts])
                                if len(audio) >= SAMPLE_RATE else np.zeros((0, N_INPUTS), np.uint8))
        return self._cache[key]

    def traces(self, audios, workers=None):
        """Per audio: (window end times in s, decision score, raw score); both scores are the margin."""
        with ThreadPoolExecutor(workers or self.workers) as pool:
            feats = list(pool.map(self.window_features, audios))
        allf = np.concatenate(feats) if feats else np.zeros((0, N_INPUTS), np.uint8)
        counts, ov = classify(allf, self.q)
        self.overflows += ov
        margin = (counts[:, 1] - counts[:, 0]).astype(np.int64)
        out, i = [], 0
        for f in feats:
            m = margin[i:i + len(f)]; i += len(f)
            out.append(((np.arange(len(m)) * HOP + SAMPLE_RATE) / SAMPLE_RATE, m, m))
        return out


# --------------------------------------------------------------------------- verification

def verify(data=ROOT / 'data'):
    q = parse_header()
    d = np.load(io.BytesIO(branch_file('board_vectors.npz')))
    x, y, counts, paths = d['x'], d['y'], d['counts'], d['paths']
    got, ov = classify(x, q)
    exact = bool(np.array_equal(got, counts))
    rep = {'vectors': int(len(x)), 'npz_keys': list(d.keys()), 'counts_bit_exact': exact,
           'count_mismatches': int((got != counts).any(1).sum()), 'int32_overflows': ov,
           'decisions_margin2': {'keyword_vectors_accepted': int(decision(got, q)[y == 1].sum()),
                                 'other_vectors_accepted': int(decision(got, q)[y == 0].sum())}}
    # Front end: recompute the 40 inputs from the original wavs (our copy of Speech Commands v0.02).
    raw = data / 'speech_commands_v0.02'
    live_diff, train_diff, n_found = [], [], 0
    for p, xv in zip(paths, x):
        parts = str(p).replace('\\', '/').split('/')
        f = raw / parts[-2] / parts[-1]
        if not f.exists():
            continue
        n_found += 1
        a = read_pcm16_training(f)
        live_diff.append(np.abs(damien_features(a).astype(int) - xv).max())
        train_diff.append(np.abs(training_features(a).astype(int) - xv).max())
    live_diff, train_diff = np.array(live_diff), np.array(train_diff)
    rep['features_from_wav'] = {
        'wavs_found': n_found,
        'demo_path_exact_vectors': int((live_diff == 0).sum()), 'demo_path_max_lsb_diff': int(live_diff.max()),
        'training_path_exact_vectors': int((train_diff == 0).sum()), 'training_path_max_lsb_diff': int(train_diff.max())}
    # Counts from features recomputed with the demo path.
    feats = np.stack([damien_features(read_pcm16_training(raw / str(p).replace('\\', '/').split('/')[-2]
                                                          / str(p).replace('\\', '/').split('/')[-1])) for p in paths])
    rep['features_from_wav']['demo_path_counts_equal_board'] = bool(np.array_equal(classify(feats, q)[0], counts))
    return rep, q


# --------------------------------------------------------------------------- our rule

def sweep_row(t, live, y, clipped, neg_tr, neg_hours, stream, mswc, words, other):
    from robust_eval import score_stream
    from stream_select import fa_per_hour
    times, score, marks, hours = stream
    complete = (y == 1) & ~clipped
    return dict(threshold=int(t), live_recall=float((live[y == 1] >= t).mean()),
                live_fa=float((live[y == 0] >= t).mean()),
                live_recall_complete=float((live[complete] >= t).mean()),
                fa_per_hour=round(fa_per_hour(neg_tr, neg_hours, t), 3), fa_hours=round(neg_hours, 2),
                stream=score_stream(times, score, t, marks, hours),
                mswc_recall=float((mswc[words == K.KEYWORD] >= t).mean()),
                mswc_other_fa=float((mswc[other] >= t).mean()))


def evaluate(out, max_live_fa=.002, max_fa_hour=2.):
    global K
    import keyword_config as K
    from robust_eval import build_sets, negative_traces, yes_prefixed
    from stream_select import select
    assert K.KEYWORD == 'sheila', 'run with KWS_KEYWORD=sheila'
    t0 = time.time()
    ver, q = verify()
    assert ver['counts_bit_exact'], ver
    det = DamienDetector(q)
    sets, info = build_sets(ROOT / 'data', 'validation', None, 3600, {'device', 'tts'})
    print('sets built', round(time.time() - t0), 's', flush=True)
    neg = negative_traces(det, ROOT / 'data', 'validation')
    print('negatives', round(neg[1], 2), 'h', round(time.time() - t0), 's', flush=True)
    y = sets['live_y']
    live = np.array([r.max() for _, _, r in det.traces(sets['live']['clean'])])
    audio, marks, sinfo = sets['stream']
    (times, _, score), = det.traces([audio])
    hours = sinfo['seconds'] / 3600
    mswc_audio, words, _ = sets['mswc']
    mswc = np.array([r.max() for _, _, r in det.traces(mswc_audio)])
    other = (words != K.KEYWORD) & ~np.vectorize(yes_prefixed)(words)
    neg_tr = [(t, s) for t, s, _ in neg[0]]
    cands = np.unique(np.r_[live, score, mswc, np.concatenate([s for _, s in neg_tr])])
    table = [sweep_row(t, live, y, sets['live_clipped'], neg_tr, neg[1], (times, score, marks, hours), mswc, words, other)
             for t in cands]
    ok = [r for r in table if r['live_fa'] <= max_live_fa and r['fa_per_hour'] <= max_fa_hour]
    best = max(ok, key=lambda r: (r['live_recall'], -r['threshold'])) if ok else None
    own = next(r for r in table if r['threshold'] == q['KW_DECISION_MARGIN'])
    # Cross-check with stream_select.select itself (its candidates are the positive peaks).
    sel = select(det, sets, neg, max_live_fa, max_fa_hour)
    bench = json.loads(branch_file('board_benchmark.json'))
    ms = bench['milliseconds_mean_at_100mhz']
    report = {
        'model': f'{BRANCH}:{DEPLOY}/keyword_model.h (sheila_v2, 8 bands x 16 bins, 48 hidden, 16 steps)',
        'keyword': K.KEYWORD, 'config': info, 'verification': ver,
        'detector': {'window_s': 1.0, 'hop_ms': HOP_MS, 'confirmation': 'none (pc_keyword_demo.py decides every window)',
                     'score': 'spike_count[keyword] - spike_count[other]', 'his_decision_margin': q['KW_DECISION_MARGIN'],
                     'int32_overflows': det.overflows},
        'rule': {'split': 'validation', 'max_live_fa': max_live_fa, 'max_fa_hour': max_fa_hour,
                 'fa_source': 'robust_eval.negative_traces', 'negatives_hours': round(neg[1], 3),
                 'holdoff_s': 1.0, 'candidates': 'every distinct score value'},
        'best_under_our_rule': best,
        'stream_select_select_crosscheck': sel,
        'his_operating_point': own,
        'sweep': table,
        'cost': {'board_benchmark': bench, 'windows_per_second': 1000 / HOP_MS,
                 'cpu_duty_at_250ms_hop': round(ms / HOP_MS, 4),
                 'cycles_per_second_at_250ms_hop': round(bench['cycles_mean'] * 1000 / HOP_MS),
                 'note': 'PicoRV32 at 100 MHz; front end (FFT) runs on the PC, not counted'},
        'seconds': round(time.time() - t0, 1),
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    short = lambda r: None if r is None else {k: v for k, v in r.items() if k != 'stream'} | {'stream_recall': r['stream']['recall']}
    print(json.dumps({'best': short(best), 'own': short(own), 'select': short(sel)}, indent=1))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['verify', 'evaluate'])
    p.add_argument('--out', type=Path, default=ROOT / 'results/sheila_damien_ourrule.json')
    a = p.parse_args()
    if a.cmd == 'verify':
        print(json.dumps(verify()[0], indent=1))
    else:
        evaluate(a.out)


if __name__ == '__main__':
    sys.exit(main())
