"""Download the corpora used beyond Speech Commands (IMPLEMENTATION_PLAN.md, Phases 0-1).

  mswc         Multilingual Spoken Words Corpus, English [39] (CC BY 4.0). The
               archives are sorted by word, so each is streamed once and only
               the selected clips are kept, decoded from 48 kHz opus to 16 kHz
               PCM16 WAV under data/mswc/<split>/<word>/.
               Selection: every "yes", up to --per-near-miss clips of each word
               in NEAR_MISS, and --random other clips (seeded). VALID=True only.
  librispeech  LibriSpeech [40] subset (CC BY 4.0), FLAC, under data/librispeech/.
  rirs         OpenSLR 28 room impulse responses and noises [17] (Apache 2.0),
               under data/rirs/. Real RIRs are held out for evaluation
               (channels.py); simulated RIRs are for training.
  musan        MUSAN music, speech and noise [18] (per-file licences in the
               corpus; mostly CC BY and public domain), under data/musan/.

Common Voice Single Word Target Segment [38] is distributed through the Mozilla
Data Collective behind a login and is not downloaded here.
"""
import argparse
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
import wave
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
HF = 'https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/main/data'
LIBRISPEECH = 'https://www.openslr.org/resources/12/{subset}.tar.gz'
RIRS = 'https://www.openslr.org/resources/28/rirs_noises.zip'
MUSAN = 'https://www.openslr.org/resources/17/musan.tar.gz'
# Real English words that share the vowel, the /s/ ending, or a /ts/, /tʃ/ ending with "yes".
NEAR_MISS = ('yet', 'yeah', 'yesterday', 'yellow', 'yell', 'yep', 'year', 'years', 'young', 'you',
             'guess', 'less', 'mess', 'bless', 'dress', 'press', 'chess', 'address', 'unless', 'success',
             'says', 'this', 'us', 'plus', 'yes', 'its', 'gets', 'lets', 'sets', 'bets', 'jets', 'pets',
             'eats', 'meets', 'seats', 'streets', 'each', 'reach', 'beach', 'peach', 'teach', 'speech',
             'fetch', 'sketch', 'stretch', 'check', 'jess', 'tess', 'nest', 'best', 'rest', 'west', 'test',
             'said', 'set', 'sex', 'next', 'ex', 'text', 'pizza', 'jazz', 'cheese', 'ease', 'these')


def write_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.rint(audio * 32767), -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
        f.writeframes(pcm.tobytes())


def decode_opus(data):
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, rate = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        g = np.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // g, rate // g).astype(np.float32)
    return audio


def archive_first_words(split):
    """First word of each archive, read from the first 64 KB of each (members are sorted by word)."""
    import zlib
    cache = ROOT / 'data/mswc' / f'{split}_archive_first_word.json'
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    n = int(urllib.request.urlopen(f'{HF}/opus/en/{split}/n_files.txt').read())
    first = {}
    for i in range(n):
        req = urllib.request.Request(f'{HF}/opus/en/{split}/audio/{i}.tar.gz', headers={'Range': 'bytes=0-65535'})
        d = zlib.decompressobj(31).decompress(urllib.request.urlopen(req, timeout=60).read())
        off = 0
        while off + 512 <= len(d):
            name = d[off:off + 100].split(b'\x00')[0].decode().lstrip('/')
            size = int(d[off + 124:off + 136].split(b'\x00')[0].strip() or b'0', 8)
            if name.endswith('.opus'):  # skip pax headers
                first[i] = name.split('_common_voice')[0]
                break
            off += 512 + (size + 511) // 512 * 512
    cache.write_text(json.dumps(first))
    return first


def word_archives(word, first):
    """Archives that may hold a word: from the last one starting before it to the last starting at or before it."""
    at_or_before = [k for k in first if first[k] <= word]
    before = [k for k in at_or_before if first[k] < word]
    return set(range(max(before) if before else 0, max(at_or_before) + 1))


def mswc_select(split, per_near_miss, n_random, seed, archives=None):
    """archives: optional set of archive indices; random negatives then come only from those."""
    rows = [r for r in csv.DictReader(open(ROOT / 'data/mswc' / f'{split}.csv', encoding='utf-8')) if r['VALID'] == 'True']
    rng = np.random.default_rng(seed)
    by_word = {}
    for r in rows:
        by_word.setdefault(r['WORD'], []).append(r)
    chosen = list(by_word.get('yes', []))
    for w in NEAR_MISS:
        if w != 'yes' and w in by_word:
            group = by_word[w]
            chosen += [group[i] for i in sorted(rng.choice(len(group), min(per_near_miss, len(group)), replace=False))]
    taken = {r['LINK'] for r in chosen}
    rest = [r for r in rows if r['LINK'] not in taken and r['WORD'] not in NEAR_MISS]
    if archives is not None:
        first = archive_first_words(split)
        ok = {w for w in {r['WORD'] for r in rest} if word_archives(w, first) <= archives}
        rest = [r for r in rest if r['WORD'] in ok]
    chosen += [rest[i] for i in sorted(rng.choice(len(rest), min(n_random, len(rest)), replace=False))]
    return chosen


def remote_size(url):
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
        return int(r.headers['Content-Length'])


def download_resume(url, target, attempts=100):
    """Download with HTTP range resume until the file has the server's size.

    Multi-GB transfers from the hub drop mid-way, sometimes without an error,
    so completeness is checked by size, never by a clean end of stream.
    """
    total = remote_size(url)
    tmp = target.with_suffix(target.suffix + '.partial')
    if target.exists():
        if target.stat().st_size == total:
            return target
        target.replace(tmp)  # an earlier, truncated download: resume it
    for attempt in range(attempts):
        have = tmp.stat().st_size if tmp.exists() else 0
        if have > total:
            tmp.unlink(); have = 0
        if have == total:
            tmp.replace(target)
            return target
        req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if have and resp.status != 206:
                    have = 0  # server ignored the range: start over
                with open(tmp, 'ab' if have else 'wb') as f:
                    shutil.copyfileobj(resp, f, 1 << 20)
        except Exception as e:
            print(f'  retry {attempt + 1} {url}: {e}', flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))  # outlasts DNS and network drop-outs
    raise RuntimeError(f'failed: {url}')


def extract_archive(url, wanted, out):
    """Fetch one tar.gz, write every wanted member (opus name -> WAV), delete the archive."""
    (out / 'archives').mkdir(parents=True, exist_ok=True)
    archive = download_resume(url, out / 'archives' / (url.split('/audio/')[0].rsplit('/', 1)[1] + '_' + url.rsplit('/', 1)[1]))
    done = 0
    with tarfile.open(archive, 'r:gz') as tf:
        for m in tf:
            name = m.name.lstrip('/')
            if name in wanted:
                target = out / wanted[name]
                if not target.exists():
                    write_wav(target, decode_opus(tf.extractfile(m).read()))
                done += 1
    archive.unlink()
    return done


def fetch_mswc(split, per_near_miss, n_random, seed, workers, needed_only=False, extra_archives=()):
    """needed_only: fetch only the archives holding "yes" and NEAR_MISS words (plus
    extra_archives); random negatives are then drawn from those archives only."""
    base = ROOT / 'data/mswc'
    if not (base / f'{split}.csv').exists():
        with urllib.request.urlopen(f'{HF}/splits/en/splits.tar.gz') as resp, tarfile.open(fileobj=resp, mode='r|gz') as tf:
            tf.extractall(base, filter=lambda m, p: m.replace(name=m.name.lstrip('/')) if m.name.endswith('.csv') else None)
    n_files = int(urllib.request.urlopen(f'{HF}/opus/en/{split}/n_files.txt').read())
    archives = None
    if needed_only:
        first = archive_first_words(split)
        archives = set(extra_archives)
        for w in set(NEAR_MISS) | {'yes'}:
            archives |= word_archives(w, first)
    chosen = mswc_select(split, per_near_miss, n_random, seed, archives)
    # Archive member names flatten "word/file.opus" to "word_file.opus".
    wanted = {r['LINK'].replace('/', '_'): Path(split) / r['WORD'] / (Path(r['LINK']).stem + '.wav') for r in chosen}
    todo = sorted(archives) if archives is not None else range(n_files)
    if archives is not None:
        # Skip archives whose selected clips are all on disk already.
        first = archive_first_words(split)
        todo = [i for i in todo if not all((base / t).exists() for n, t in wanted.items()
                                           if i in word_archives(n.split('_common_voice')[0], first))]
    urls = [f'{HF}/opus/en/{split}/audio/{i}.tar.gz' for i in todo]
    print(f'MSWC {split}: {len(chosen)} clips, {len(urls)} of {n_files} archives to fetch', flush=True)
    with ThreadPoolExecutor(workers) as pool:
        counts = list(pool.map(lambda u: extract_archive(u, wanted, base), urls))
    with open(base / f'{split}_selected.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'word', 'speaker', 'gender', 'link'])
        w.writeheader()
        for r in chosen:
            path = Path(split) / r['WORD'] / (Path(r['LINK']).stem + '.wav')
            if (base / path).exists():
                w.writerow(dict(path=path.as_posix(), word=r['WORD'], speaker=r['SPEAKER'], gender=r['GENDER'], link=r['LINK']))
    print(json.dumps({'split': split, 'selected': len(chosen), 'written': sum(counts)}), flush=True)


def download(url, target):
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url}', flush=True)
    return download_resume(url, target)


def fetch_librispeech(subset):
    base = ROOT / 'data/librispeech'
    if (base / 'LibriSpeech' / subset).exists():
        return
    archive = download(LIBRISPEECH.format(subset=subset), base / f'{subset}.tar.gz')
    with tarfile.open(archive) as tf:
        tf.extractall(base, filter='data')
    archive.unlink()


def fetch_rirs():
    base = ROOT / 'data/rirs'
    if (base / 'RIRS_NOISES').exists():
        return
    archive = download(RIRS, base / 'rirs_noises.zip')
    with zipfile.ZipFile(archive) as z:
        z.extractall(base)
    archive.unlink()


def fetch_musan():
    base = ROOT / 'data/musan'
    if (base / 'musan').exists():
        return
    archive = download(MUSAN, base / 'musan.tar.gz')
    with tarfile.open(archive) as tf:
        tf.extractall(base, filter='data')
    archive.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('what', choices=['mswc', 'librispeech', 'rirs', 'musan'])
    p.add_argument('--split', default='test', help='MSWC split: train, dev, test')
    p.add_argument('--subset', default='test-clean', help='LibriSpeech subset')
    p.add_argument('--per-near-miss', type=int, default=60)
    p.add_argument('--random', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--needed-only', action='store_true', help='MSWC: only archives with yes and near-miss words')
    p.add_argument('--extra-archives', type=int, nargs='*', default=[], help='MSWC: archives to add to --needed-only')
    a = p.parse_args()
    if a.what == 'mswc':
        fetch_mswc(a.split, a.per_near_miss, a.random, a.seed, a.workers, a.needed_only, a.extra_archives)
    elif a.what == 'librispeech':
        fetch_librispeech(a.subset)
    elif a.what == 'rirs':
        fetch_rirs()
    else:
        fetch_musan()
