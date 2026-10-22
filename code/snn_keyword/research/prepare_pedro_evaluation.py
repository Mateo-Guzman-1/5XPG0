"""Fetch the original evaluation corpora and reproduce Pedro's ordered clip index.

Only official validation/test Speech Commands WAVs are extracted. Training is
unchanged; the resulting features.npz is an audio index, not a feature cache.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tarfile
import urllib.request
import numpy as np


def fetch(url, archive, target):
    marker = archive.with_suffix('.extracted')
    if marker.exists():
        return
    if not archive.exists():
        partial = archive.with_suffix('.partial')
        print('Downloading', url, flush=True)
        urllib.request.urlretrieve(url, partial)
        partial.replace(archive)
    with tarfile.open(archive) as tar:
        if 'speech_commands' in archive.name:
            names = {m.name.removeprefix('./'): m for m in tar.getmembers()}
            val = set(tar.extractfile(names['validation_list.txt']).read().decode().split())
            test = set(tar.extractfile(names['testing_list.txt']).read().decode().split())
            members = [m for n, m in names.items() if n in val | test
                       or n.startswith('_background_noise_/')
                       or n in ('validation_list.txt', 'testing_list.txt', 'LICENSE')]
        else:
            members = tar.getmembers()
        tar.extractall(target, members=members, filter='data')
    print('Extracted', archive.name, flush=True)
    marker.touch()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    args = p.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    jobs = [('https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
             args.data / 'speech_commands_v0.02.tar.gz', args.data / 'speech_commands_v0.02')]
    for split in ('dev-clean', 'dev-other', 'test-clean', 'test-other'):
        jobs.append((f'https://www.openslr.org/resources/12/{split}.tar.gz',
                     args.data / f'{split}.tar.gz', args.data / 'librispeech'))
    with ThreadPoolExecutor(5) as pool:
        list(pool.map(lambda job: fetch(*job), jobs))
    raw = args.data / 'speech_commands_v0.02'
    val = set((raw / 'validation_list.txt').read_text().split())
    test = set((raw / 'testing_list.txt').read_text().split())
    names = sorted(val | test)
    assert all((raw / n).exists() for n in names)
    np.savez(args.data / 'features.npz', names=names,
             split=np.array([1 if n in val else 2 for n in names]),
             y=np.array([int(n.startswith('sheila/')) for n in names]))
    print('Ready:', len(val), 'validation;', len(test), 'test', flush=True)


if __name__ == '__main__':
    main()
