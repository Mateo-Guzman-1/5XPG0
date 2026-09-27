"""Verifier track, step 1: phoneme targets for the second-stage "yes" verifier.

Pronunciations come from the public CMU Pronouncing Dictionary
(github.com/cmusphinx/cmudict, BSD licence; data_verifier/cmudict.dict):
39 phonemes, stress removed, first pronunciation of each word.

  libri     LibriSpeech train-clean-100 transcripts -> phoneme sequences;
            utterances with any out-of-dictionary word are skipped. The
            audio is packed once into data_verifier/libri100_audio.npy
            (int16, concatenated) with an index (libri100_index.npz).
  keywords  Every word of data/multi/manifest.csv. Out-of-dictionary words:
            the TTS pseudo-words (yesd, yeets, pes, ...) get a small
            rule-based letter-to-sound fallback (fallback_g2p); other OOV
            words (MSWC names, mis-encoded apostrophes) are dropped.

data/ is read only; everything is written to data_verifier/.
"""
import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'data_verifier'
PHONES = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
          'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y',
          'Z', 'ZH']
BLANK = 0
SYMBOLS = ['<b>'] + PHONES          # CTC classes: blank + 39 phonemes
PID = {p: i + 1 for i, p in enumerate(PHONES)}
KEYWORD = [PID['Y'], PID['EH'], PID['S']]


@lru_cache(maxsize=1)
def cmudict():
    d = {}
    for line in open(OUT / 'cmudict.dict', encoding='utf-8'):
        line = line.split('#')[0].strip()
        if not line:
            continue
        word, *ph = line.split()
        if '(' in word:          # alternative pronunciations: keep the first
            continue
        d[word] = [re.sub(r'\d', '', p) for p in ph]
    return d


# Letter-to-sound for the TTS pseudo-words only (all are 2-6 letters, mostly
# "yes" with one letter changed, added or removed). Crude on purpose; Piper
# (espeak-ng) may say some of them differently.
_DIGRAPHS = [('ee', ['IY']), ('ea', ['IY']), ('ey', ['EY']), ('ay', ['EY']), ('oo', ['UW']), ('ou', ['AW']),
             ('ch', ['CH']), ('sh', ['SH']), ('th', ['TH']), ('ck', ['K']), ('ph', ['F']), ('ie', ['IY']),
             ('oe', ['OW'])]
_LETTER = dict(b='B', c='K', d='D', f='F', g='G', h='HH', j='JH', k='K', l='L', m='M', n='N', p='P', q='K',
               r='R', s='S', t='T', v='V', w='W', z='Z', a='AE', e='EH', i='IH', o='AA', u='AH')


def fallback_g2p(word):
    w, out, i = word.lower(), [], 0
    if not re.fullmatch(r'[a-z]+', w):
        return None
    while i < len(w):
        for dg, ph in _DIGRAPHS:
            if w.startswith(dg, i):
                out += ph; i += 2
                break
        else:
            c = w[i]
            if c == 'y':
                out.append('Y' if i + 1 < len(w) and w[i + 1] in 'aeiou' else 'IY')
            elif c == 'x':
                out += ['K', 'S']
            else:
                out.append(_LETTER[c])
            i += 1
    return out


def word_phones(word, fallback=False):
    w = word.lower().replace('’', "'")
    d = cmudict()
    if w in d:
        return d[w]
    return fallback_g2p(w) if fallback else None


def text_phones(text):
    out = []
    for w in text.split():
        p = word_phones(w)
        if p is None:
            return None
        out += p
    return out


def encode(ph):
    return np.array([PID[p] for p in ph], np.int16)


def keyword_targets(split):
    """Per manifest row of `split` (sc/mswc/tts): phoneme ids or None (dropped)."""
    rows = [r for r in csv.DictReader(open(DATA / 'multi/manifest.csv', encoding='utf-8')) if r['split'] == split]
    out = []
    for r in rows:
        ph = word_phones(r['word'], fallback=r['corpus'] == 'tts')
        out.append((int(r['row']), r['corpus'], r['word'], None if ph is None else encode(ph)))
    return out


def pack_libri(subset='train-clean-100', workers=12):
    import soundfile as sf
    base = DATA / 'librispeech' / 'LibriSpeech' / subset
    items, skipped = [], 0
    for t in sorted(base.rglob('*.trans.txt')):
        for line in t.read_text().splitlines():
            uid, text = line.split(' ', 1)
            ph = text_phones(text)
            if ph is None:
                skipped += 1
                continue
            items.append((uid, t.parent / f'{uid}.flac', text, encode(ph)))
    print(f'{subset}: {len(items)} utterances kept, {skipped} skipped (out-of-dictionary words)', flush=True)
    def load(it):
        return sf.read(it[1], dtype='int16')[0]
    lens = [sf.info(str(it[1])).frames for it in items]   # from the headers, no decoding
    starts = np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64)
    audio = np.lib.format.open_memmap(OUT / 'libri100_audio.npy', 'w+', np.int16, (int(sum(lens)),))
    with ThreadPoolExecutor(workers) as pool:
        for k, a in enumerate(pool.map(load, items)):
            assert len(a) == lens[k]
            audio[starts[k]:starts[k] + lens[k]] = a
            if k % 2000 == 0:
                print(k, flush=True)
    audio.flush()
    tgt = [it[3] for it in items]
    np.savez(OUT / 'libri100_index.npz', uid=np.array([it[0] for it in items]), start=starts,
             length=np.array(lens, np.int64), text=np.array([it[2] for it in items]),
             phones=np.concatenate(tgt), phone_off=np.r_[0, np.cumsum([len(x) for x in tgt])].astype(np.int64))
    return {'kept': len(items), 'skipped_oov': skipped, 'hours': round(sum(lens) / 16000 / 3600, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--skip-libri', action='store_true')
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    summary = {}
    for split in ('train', 'validation'):
        kt = keyword_targets(split)
        rows = np.array([k[0] for k in kt])
        keep = np.array([k[3] is not None for k in kt])
        tg = [k[3] if k[3] is not None else np.zeros(0, np.int16) for k in kt]
        np.savez(OUT / f'keyword_targets_{split}.npz', row=rows, keep=keep, corpus=np.array([k[1] for k in kt]),
                 word=np.array([k[2] for k in kt]), phones=np.concatenate(tg),
                 phone_off=np.r_[0, np.cumsum([len(x) for x in tg])].astype(np.int64))
        fb = sorted({k[2] for k in kt if k[1] == 'tts' and k[2].lower() not in cmudict()})
        summary[split] = {'rows': len(kt), 'kept': int(keep.sum()), 'dropped_oov': int((~keep).sum()),
                          'tts_fallback_words': len(fb),
                          'fallback_examples': {w: ' '.join(word_phones(w, True)) for w in fb[:12]}}
        print(split, json.dumps(summary[split]), flush=True)
    if not a.skip_libri:
        summary['libri'] = pack_libri()
        print(json.dumps(summary['libri']))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
