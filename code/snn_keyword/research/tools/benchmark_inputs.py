"""The fixed inputs of the size benchmark (E2) and their fingerprint.

Every model is scored on exactly these files. eval_model.py and bench_board.py put the
fingerprint into each result, size_sweep.py stores the full list in
research/results/E2/benchmark_inputs.json, and plot_size_sweep.py refuses to combine
results whose fingerprints differ.

    benchmark_inputs.py          prints the inputs and the fingerprint
"""
import hashlib
import json
from pathlib import Path
import sys

KWS = Path(__file__).resolve().parents[2]   # code/snn_keyword
sys.path.insert(0, str(KWS))
from features import N_TIME

# Input vectors for the board check (key x). The expected scores come from each model's own
# integer oracle, so one fixed set serves every size.
VECTORS = KWS / 'results/sheila_hardneg/verification_vectors.npz'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files(keyword):
    """Name -> path or glob (relative to code/snn_keyword) of every fixed input."""
    return {
        'features': 'data/features.npz',
        'augment': 'data/augment.npz',
        # tune_stream.live_windows caches: validation (seed 2) tunes, test (seed 1) scores.
        'live_validation_windows': f'build/live_{keyword}_split1_3000_2_{N_TIME}bins.npz',
        'live_test_windows': f'build/live_{keyword}_split2_3000_1_{N_TIME}bins.npz',
        'tts_probe': f'build/tts_probe_{keyword}/*.wav',
        'background_noise': 'data/speech_commands_v0.02/_background_noise_/*.wav',
        'verification_vectors': VECTORS.relative_to(KWS).as_posix(),
        'bitstream_base': 'deploy/keyword.bit',
        'bitstream_kdot': 'deploy/keyword_kdot.bit',
    }


def describe(keyword):
    """sha256 of every input (of the sorted name/sha256 list for globs) and one fingerprint over all."""
    inputs = {}
    for name, pattern in files(keyword).items():
        paths = sorted(KWS.glob(pattern)) if '*' in pattern else [KWS / pattern]
        if not paths or not all(p.exists() for p in paths):
            inputs[name] = {'path': pattern, 'sha256': None}
        elif '*' in pattern:
            listing = ''.join(f'{p.name} {sha256(p)}\n' for p in paths)
            inputs[name] = {'path': pattern, 'files': len(paths), 'sha256': hashlib.sha256(listing.encode()).hexdigest()}
        else:
            inputs[name] = {'path': pattern, 'bytes': paths[0].stat().st_size, 'sha256': sha256(paths[0])}
    digest = json.dumps({k: v['sha256'] for k, v in inputs.items()}, sort_keys=True).encode()
    return {'keyword': keyword, 'fingerprint': hashlib.sha256(digest).hexdigest()[:16], 'inputs': inputs}


if __name__ == '__main__':
    import numpy as np
    from model import keyword_of
    print(json.dumps(describe(keyword_of(np.load(KWS / 'data/features.npz'))), indent=2))
