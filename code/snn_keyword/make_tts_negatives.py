"""Multi-speaker TTS hard negatives, plus TTS "yes" positives [14][15].

Words (IMPLEMENTATION_PLAN.md, Phase 1, item 2):
  - every single-letter insertion, deletion and substitution of "yes"
    (GraphemeAug [15]), minus edits espeak pronounces as "yes" or that only
    voice the final fricative (yez, yess): those would be label noise;
  - a fixed list of real /s/-, /ts/- and /tʃ/-final and "ye-" words.
"yes" itself is synthesized with the same voices, so a synthetic voice is not
a cue for "no".

Voices (Piper [rhasspy/piper-voices], all CC BY 4.0 training data):
  en_US-libritts_r-medium, 904 speakers, split by speaker 80/10/10 into
  train / validation / test (seeded); en_GB-vctk-medium, en_US-l2arctic-medium
  and en_US-arctic-medium are whole models held out for test only. The two
  Windows voices of make_tts_probe.ps1 stay a separate probe.

Every utterance draws a word, a speaker of its split, and length, noise and
duration-noise scales. Output: data/tts/<split>/<model>/<word>_<speaker>_<n>.wav
at 16 kHz and data/tts/manifest.csv.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import string
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/tts'
HF = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en'
MODELS = {'libritts_r': 'en_US/libritts_r/medium/en_US-libritts_r-medium',
          'vctk': 'en_GB/vctk/medium/en_GB-vctk-medium',
          'l2arctic': 'en_US/l2arctic/medium/en_US-l2arctic-medium',
          'arctic': 'en_US/arctic/medium/en_US-arctic-medium'}
HELDOUT_MODELS = ('vctk', 'l2arctic', 'arctic')
REAL_WORDS = ('yeah', 'yea', 'yet', 'yep', 'yeet', 'yeets', 'yets', 'yetz', 'yech', 'yetch', 'yesterday', 'yellow',
              'guess', 'less', 'mess', 'bless', 'dress', 'press', 'chess', 'jess', 'tess', 'this', 'us', 'plus',
              'says', 'kiss', 'miss', 'gas', 'pass', 'nice', 'ice', 'piece', 'peace', 'cheese', 'ease', 'these',
              'its', 'eats', 'gets', 'lets', 'sets', 'bets', 'jets', 'pets', 'meets', 'seats', 'beats', 'pizza',
              'pizzas', 'each', 'peach', 'reach', 'teach', 'speech', 'fetch', 'sketch', 'best', 'rest', 'test',
              'next', 'text', 'sex', 'ex', 'ras', 'tos', 'mes', 'tes', 'yas', 'yus', 'yos')
AMBIGUOUS = {'yez', 'yess', 'yesz', 'yezs'}
SPLIT_SHARE = {'train': .8, 'validation': .1, 'test': .1}
COUNTS = {'train': (8000, 32000), 'validation': (1000, 4000), 'test': (1000, 4000), 'test_heldout_models': (1000, 4000)}


def grapheme_edits(base='yes'):
    L = string.ascii_lowercase
    eds = {base[:i] + c + base[i:] for i in range(len(base) + 1) for c in L}
    eds |= {base[:i] + base[i + 1:] for i in range(len(base))}
    eds |= {base[:i] + c + base[i + 1:] for i in range(len(base)) for c in L}
    eds.discard(base)
    return eds


def negative_words():
    from piper.phonemize_espeak import EspeakPhonemizer
    ph = EspeakPhonemizer()
    say = lambda w: ''.join(sum(ph.phonemize('en-us', w), []))
    ref = say('yes')
    words = sorted((grapheme_edits() | set(REAL_WORDS)) - AMBIGUOUS)
    return [w for w in words if say(w) != ref]


def voice_files(name):
    base = OUT / 'voices'
    base.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ('.onnx', '.onnx.json'):
        p = base / (Path(MODELS[name]).name + ext)
        if not p.exists():
            urllib.request.urlretrieve(f'{HF}/{MODELS[name]}{ext}', p)
        paths.append(p)
    return paths


@lru_cache(maxsize=None)
def num_speakers(name):
    return json.loads(voice_files(name)[1].read_text(encoding='utf-8'))['num_speakers']


def speaker_splits(seed=0):
    """libritts_r speakers by split; held-out models are test-only."""
    n = num_speakers('libritts_r')
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(SPLIT_SHARE['train'] * n), int((SPLIT_SHARE['train'] + SPLIT_SHARE['validation']) * n)
    return {'train': perm[:a], 'validation': perm[a:b], 'test': perm[b:]}


def plan(seed=0):
    words = negative_words()
    rng = np.random.default_rng(seed + 1)
    spk = speaker_splits(seed)
    jobs = []
    for split, (n_pos, n_neg) in COUNTS.items():
        for k in range(n_pos + n_neg):
            word = 'yes' if k < n_pos else words[rng.integers(len(words))]
            if split == 'test_heldout_models':
                model = HELDOUT_MODELS[rng.integers(len(HELDOUT_MODELS))]
                speaker = int(rng.integers(num_speakers(model)))
            else:
                model, speaker = 'libritts_r', int(rng.choice(spk[split]))
            jobs.append(dict(split=split, model=model, speaker=speaker, word=word, n=k,
                             length_scale=float(rng.uniform(.8, 1.35)), noise_scale=float(rng.uniform(.4, .9)),
                             noise_w_scale=float(rng.uniform(.5, 1.))))
    return jobs, words


_voices = {}


def synth(job):
    from piper import PiperVoice, SynthesisConfig
    from scipy.signal import resample_poly
    from fetch_corpora import write_wav
    path = OUT / job['split'] / job['model'] / f"{job['word']}_{job['speaker']}_{job['n']}.wav"
    if path.exists():
        return path
    if job['model'] not in _voices:
        _voices[job['model']] = PiperVoice.load(*voice_files(job['model']))
    v = _voices[job['model']]
    cfg = SynthesisConfig(speaker_id=job['speaker'], length_scale=job['length_scale'],
                          noise_scale=job['noise_scale'], noise_w_scale=job['noise_w_scale'])
    audio = np.concatenate([c.audio_float_array for c in v.synthesize(job['word'], cfg)])
    rate = v.config.sample_rate
    g = np.gcd(rate, 16000)
    audio = resample_poly(audio, 16000 // g, rate // g).astype(np.float32)
    write_wav(path, audio / max(1e-6, np.abs(audio).max()) * .5)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--limit', type=int, default=0, help='synthesize only the first N jobs per split (smoke test)')
    a = p.parse_args()
    for m in MODELS:
        voice_files(m)
    jobs, words = plan()
    if a.limit:
        jobs = [j for s in COUNTS for j in [j for j in jobs if j['split'] == s][:a.limit]]
    with ProcessPoolExecutor(a.workers) as pool:
        paths = list(pool.map(synth, jobs, chunksize=64))
    with open(OUT / 'manifest.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'word', 'label', 'split', 'model', 'speaker'])
        w.writeheader()
        for j, path in zip(jobs, paths):
            w.writerow(dict(path=path.relative_to(OUT).as_posix(), word=j['word'], label=int(j['word'] == 'yes'),
                            split=j['split'], model=j['model'], speaker=j['speaker']))
    (OUT / 'words.json').write_text(json.dumps({'negative_words': words, 'counts': COUNTS}, indent=1))
    print(json.dumps({'utterances': len(jobs), 'negative_words': len(words)}), flush=True)


if __name__ == '__main__':
    main()
