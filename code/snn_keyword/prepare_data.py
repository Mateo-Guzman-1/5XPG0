"""Download Speech Commands v0.02 and cache the shared PC mel frontend."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
from features import features, read_wav

ROOT = Path(__file__).resolve().parent
URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
ARCHIVE_SHA256 = 'af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58'


def prepare(root=ROOT / 'data', workers=12):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'speech_commands_v0.02.tar.gz'
    raw = root / 'speech_commands_v0.02'
    if not archive.exists():
        print('Downloading official Speech Commands v0.02...', flush=True)
        tmp = archive.with_suffix('.partial')
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(archive)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError('Dataset SHA256 mismatch; discard the corrupt archive and retry')
    if not (raw / '.extracted').exists():
        raw.mkdir(exist_ok=True)
        print('Extracting dataset...', flush=True)
        with tarfile.open(archive) as tf:
            tf.extractall(raw, filter='data')
        (raw / '.extracted').touch()
    valid = set((raw / 'validation_list.txt').read_text().splitlines())
    test = set((raw / 'testing_list.txt').read_text().splitlines())
    paths = sorted(p for p in raw.glob('*/*.wav') if not p.parent.name.startswith('_'))
    names = [p.relative_to(raw).as_posix() for p in paths]
    splits = np.array([2 if n in test else 1 if n in valid else 0 for n in names], dtype=np.uint8)
    speakers = [p.stem.split('_nohash_')[0] for p in paths]
    speaker_sets = [set(s for s, k in zip(speakers, splits) if k == i) for i in range(3)]
    assert not any(speaker_sets[i] & speaker_sets[j] for i in range(3) for j in range(i))
    def transform(p):
        return features(read_wav(p))
    print(f'Computing features for {len(paths)} recordings...', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        x = np.stack(list(pool.map(transform, paths)))
    y = np.array([p.parent.name == 'yes' for p in paths], dtype=np.int64)
    np.savez_compressed(root / 'features.npz', x=x, y=y, split=splits, names=np.array(names))
    manifest = {'url': URL, 'archive_sha256': digest,
                'keyword': 'yes', 'features': list(x.shape[1:]), 'speaker_disjoint': True,
                'splits': {n: {'total': int((splits == i).sum()), 'positive': int(y[splits == i].sum()),
                               'speakers': len(speaker_sets[i])} for i, n in enumerate(['train', 'validation', 'test'])}}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    prepare(a.data, a.workers)
