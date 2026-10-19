"""R2 gate G-A1 (research/PLAN_R2_GRU.md): Mateo's window front end and integer model equal Pedro's, bit-exact.

The R1 live caches (build/r1_cache/live_<split>.npz) hold Pedro's window features of Mateo's live sets.
1. The live audio is rebuilt with the vendored robust_eval; Mateo's features() (tools/mateo_eval/features.py)
   and Pedro's features() (code/snn_keyword/features.py) must both reproduce the cached windows.
2. Mateo's model.integer_forward and Pedro's must give identical scores and spike counts on those windows
   for the three H48 K64 seeds (research/results/S3/models/).

Usage: python research/tools/r2_gate_a1.py   ->   research/results/R2/gate_A1.json
"""
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault('KWS_KEYWORD', 'sheila')
KWS = Path(__file__).resolve().parents[2]
MATEO = Path(__file__).resolve().parent / 'mateo_eval'
sys.path[:0] = [str(KWS), str(MATEO)]
import robust_eval as R  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PF, PM = load('pedro_features', KWS / 'features.py'), load('pedro_model', KWS / 'model.py')
MF, MM = load('mateo_features', MATEO / 'features.py'), load('mateo_model', MATEO / 'model.py')
HOP = 4000


def windows(audio, features):
    starts = range(0, len(audio) - 16000 + 1, HOP)
    return np.stack([features(audio[s:s + 16000]) for s in starts])


def main():
    out = {'features': {}, 'integer_forward': {}}
    data = KWS / 'data'
    for split in ('test', 'validation'):
        cache = dict(np.load(KWS / f'build/r1_cache/live_{split}.npz'))
        cfg = R.SPLITS[split]
        clips, y, rng = R.sc_live_clips(data, cfg['sc'], seed=cfg['seed'])
        audios = R.place_clips(clips, R.bg_noise(data), rng, None)
        assert (y == cache['y']).all()
        with ThreadPoolExecutor(12) as pool:
            xm = np.concatenate(list(pool.map(lambda a: windows(a, MF.features), audios)))
            xp = np.concatenate(list(pool.map(lambda a: windows(a, PF.features), audios)))
        out['features'][split] = {'windows': len(cache['x']), 'clips': len(audios),
                                  'mateo_equals_cache': bool(np.array_equal(xm, cache['x'])),
                                  'pedro_equals_cache': bool(np.array_equal(xp, cache['x']))}
        print(split, out['features'][split], flush=True)
        if split != 'test':
            continue
        for seed in (0, 1, 2):
            q = dict(np.load(KWS / f'research/results/S3/models/h48_k64_seed{seed}.npz'))
            sp, kp = PM.integer_forward(cache['x'], q)
            sm, km = MM.integer_forward(cache['x'], q)
            out['integer_forward'][f'h48_k64_seed{seed}'] = {
                'windows': len(cache['x']), 'scores_equal': bool(np.array_equal(sp, sm)),
                'spikes_equal': bool(np.array_equal(kp, km)),
                'margin_range': [int((sp[:, 1] - sp[:, 0]).min()), int((sp[:, 1] - sp[:, 0]).max())]}
            print(seed, out['integer_forward'][f'h48_k64_seed{seed}'], flush=True)
    ok = all(v['mateo_equals_cache'] and v['pedro_equals_cache'] for v in out['features'].values()) and \
        all(v['scores_equal'] and v['spikes_equal'] for v in out['integer_forward'].values())
    out['gate_G_A1'] = 'pass' if ok else 'FAIL'
    dst = KWS / 'research/results/R2/gate_A1.json'
    dst.write_text(json.dumps(out, indent=1))
    print(out['gate_G_A1'], '->', dst)


if __name__ == '__main__':
    main()
