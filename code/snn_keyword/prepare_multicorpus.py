"""One manifest and packed arrays for all training corpora (IMPLEMENTATION_PLAN.md, Phase 1).

Sources (fetch_corpora.py, make_tts_negatives.py):
  sc      Speech Commands v0.02 [1], official speaker-disjoint splits     CC BY 4.0
  mswc    MSWC English [39] train / dev selections                        CC BY 4.0
  tts     Piper multi-speaker hard negatives and "yes", speaker splits    CC BY 4.0 (voice data)
  libri   LibriSpeech [40] train-clean-100 / dev-clean running speech,
          utterances whose transcript contains "yes" removed              CC BY 4.0
  noise   MUSAN noise and music [18], OpenSLR 28 point-source noises      per corpus
Test data is not packed: robust_eval.py and confusables.py read it directly.

Labels: the 35 Speech Commands words, _unknown_ (35) and _silence_ (36).
MSWC and TTS words in that vocabulary keep their class ("yes", "no", digits,
...); all other words are _unknown_. MSWC "yes" clips whose last 20 ms are
within 20 dB of the peak (the /s/ may be cut by the forced alignment) are
dropped from training. MSWC publishes no alignment score beyond VALID, which
fetch_corpora.py already requires. MSWC's official splits share speakers, so
rows_mswc() removes test speakers from dev and train, and dev speakers from train.

Output (data/multi/):
  manifest.csv          split, row, corpus, word, cls, speaker, licence, source
  clips_<split>.npy     int16 (N, 16000); short clips centred, long ones cut to 1 s
  speech_<split>.npy    int16 LibriSpeech stream; speech_<split>_starts.npy utterance starts
  noise_train.npy       int16 noise and music
"""
import argparse
import csv
import glob
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from features import read_wav
from robust_eval import edge_clipped

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = DATA / 'multi'
SR = 16000
WORDS = sorted(p.name for p in (DATA / 'speech_commands_v0.02').iterdir()
               if p.is_dir() and not p.name.startswith('_'))
CLASSES = WORDS + ['_unknown_', '_silence_']
UNKNOWN, SILENCE = len(WORDS), len(WORDS) + 1
LICENCE = {'sc': 'CC BY 4.0', 'mswc': 'CC BY 4.0', 'tts': 'CC BY 4.0 (Piper voice training data)', 'libri': 'CC BY 4.0'}


def cls(word):
    return WORDS.index(word) if word in WORDS else UNKNOWN


def to_second(a):
    """Centre a short clip in 1 s; cut a long one around its loudest second."""
    if len(a) <= SR:
        out = np.zeros(SR, np.float32)
        o = (SR - len(a)) // 2
        out[o:o + len(a)] = a
        return out
    e = np.convolve(a ** 2, np.ones(SR // 10), 'valid')
    c = int(np.argmax(e)) + SR // 20
    s = int(np.clip(c - SR // 2, 0, len(a) - SR))
    return a[s:s + SR]


def rows_sc(split):
    d = np.load(DATA / 'features.npz')
    code = {'train': 0, 'validation': 1}[split]
    names = d['names'][d['split'] == code]
    return [dict(corpus='sc', word=n.split('/')[0], speaker='sc:' + Path(n).stem.split('_nohash_')[0],
                 source=f'speech_commands_v0.02/{n}') for n in names]


def mswc_speakers(split):
    table = DATA / 'mswc' / f'{split}_selected.csv'
    return {r['speaker'] for r in csv.DictReader(open(table, encoding='utf-8'))} if table.exists() else set()


def rows_mswc(split):
    """MSWC's own splits share speakers (about 10k of 33k), so speakers are made
    disjoint here: test speakers are removed from dev and train, dev speakers from train."""
    name = {'train': 'train', 'validation': 'dev'}[split]
    table = DATA / 'mswc' / f'{name}_selected.csv'
    if not table.exists():
        print(f'missing {table}: MSWC skipped for {split}', flush=True)
        return [], 0
    banned = mswc_speakers('test') | (mswc_speakers('dev') if split == 'train' else set())
    rows = list(csv.DictReader(open(table, encoding='utf-8')))
    keep = [dict(corpus='mswc', word=r['word'], speaker='mswc:' + r['speaker'], source=f"mswc/{r['path']}")
            for r in rows if r['speaker'] not in banned]
    return keep, len(rows) - len(keep)


def rows_tts(split):
    table = DATA / 'tts' / 'manifest.csv'
    if not table.exists():
        print('missing TTS manifest: TTS skipped', flush=True)
        return []
    return [dict(corpus='tts', word=r['word'], speaker=f"tts:{r['model']}:{r['speaker']}", source=f"tts/{r['path']}")
            for r in csv.DictReader(open(table, encoding='utf-8')) if r['split'] == split]


def pack_clips(split, workers):
    mswc, speaker_overlap = rows_mswc(split)
    rows = rows_sc(split) + mswc + rows_tts(split)
    audio = [None] * len(rows)

    def load(i):
        audio[i] = read_wav(DATA / rows[i]['source'])

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(load, range(len(rows))))
    keep = [i for i, r in enumerate(rows)
            if not (r['corpus'] == 'mswc' and r['word'] == 'yes' and edge_clipped(audio[i]))]
    dropped = len(rows) - len(keep)
    arr = np.lib.format.open_memmap(OUT / f'clips_{split}.npy', 'w+', np.int16, (len(keep), SR))
    for j, i in enumerate(keep):
        arr[j] = np.clip(np.rint(to_second(audio[i]) * 32767), -32768, 32767)
    arr.flush()
    out = []
    for j, i in enumerate(keep):
        r = rows[i]
        out.append(dict(split=split, row=j, corpus=r['corpus'], word=r['word'], cls=cls(r['word']),
                        speaker=r['speaker'], licence=LICENCE[r['corpus']], source=r['source']))
    return out, {'mswc_yes_dropped_edge_clipped': dropped, 'mswc_dropped_shared_speakers': speaker_overlap}


def pack_speech(split, hours, seed):
    import soundfile as sf
    from robust_eval import libri_utterances
    subset = {'train': 'train-clean-100', 'validation': 'dev-clean'}[split]
    try:
        files = libri_utterances(DATA, subset)
    except FileNotFoundError as e:
        print(e, flush=True)
        return None
    files = [files[i] for i in np.random.default_rng(seed).permutation(len(files))]
    chunks, starts, total = [], [], 0
    for f in files:
        a = sf.read(f, dtype='float32')[0]
        starts.append(total); chunks.append(a); total += len(a)
        if total >= hours * 3600 * SR:
            break
    speech = np.clip(np.rint(np.concatenate(chunks) * 32767), -32768, 32767).astype(np.int16)
    np.save(OUT / f'speech_{split}.npy', speech)
    np.save(OUT / f'speech_{split}_starts.npy', np.array(starts, np.int64))
    return {'subset': subset, 'utterances': len(starts), 'hours': round(total / SR / 3600, 2)}


def pack_noise(hours_per_kind, seed):
    import soundfile as sf
    rng = np.random.default_rng(seed)
    summary, parts = {}, []
    kinds = {'musan_noise': sorted(glob.glob(str(DATA / 'musan/musan/noise/*/*.wav'))),
             'musan_music': sorted(glob.glob(str(DATA / 'musan/musan/music/*/*.wav'))),
             'rirs_pointsource': sorted(glob.glob(str(DATA / 'rirs/RIRS_NOISES/pointsource_noises/*.wav')))}
    for kind, files in kinds.items():
        total = 0
        for k in rng.permutation(len(files)):
            a, rate = sf.read(files[k], dtype='float32', always_2d=True)
            if rate != SR:
                continue
            parts.append(a[:, 0]); total += len(a)
            if total >= hours_per_kind * 3600 * SR:
                break
        summary[kind] = round(total / SR / 3600, 2)
    if not parts:
        return None
    noise = np.concatenate(parts)
    noise = noise / (np.abs(noise).max() + 1e-9)
    np.save(OUT / 'noise_train.npy', np.rint(noise * 32767).astype(np.int16))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--libri-hours', type=float, default=20)
    p.add_argument('--noise-hours', type=float, default=1.5, help='hours per noise kind')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--only', nargs='*', default=['clips', 'speech', 'noise'])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((OUT / 'summary.json').read_text()) if (OUT / 'summary.json').exists() else {}
    summary['classes'] = CLASSES
    if 'clips' in a.only:
        rows = []
        for split in ('train', 'validation'):
            r, drops = pack_clips(split, a.workers)
            rows += r
            by = {}
            for x in r:
                by.setdefault(x['corpus'], {'clips': 0, 'yes': 0, 'speakers': set()})
                by[x['corpus']]['clips'] += 1; by[x['corpus']]['yes'] += x['word'] == 'yes'
                by[x['corpus']]['speakers'].add(x['speaker'])
            summary[split] = {c: {'clips': v['clips'], 'yes': v['yes'], 'speakers': len(v['speakers'])} for c, v in by.items()}
            summary[split].update(drops)
        with open(OUT / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    if 'speech' in a.only:
        summary['speech'] = {s: pack_speech(s, a.libri_hours if s == 'train' else 3, 0) for s in ('train', 'validation')}
    if 'noise' in a.only:
        summary['noise_train'] = pack_noise(a.noise_hours, 0)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
