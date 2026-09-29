"""One-time test-split report for final candidates (run once per candidate).

Each detector is scored at a FIXED threshold chosen on validation data:
  live       Speech Commands test clips in noise (robust_eval.place_clips):
             keyword recall, recall on complete recordings, other words accepted
  device     the same clips through held-out microphones, real rooms, both
  stream     the 1 h test stream (recall, latency)
  negatives  false accepts per hour on robust_eval.negative_stream('test')
             (LibriSpeech test-clean + test-other and every non-keyword
             Speech Commands test word; about 15-18 h)
Usage: python final_eval.py name=path[@threshold] ... --out results/final_<kw>.json
  path: a model accepted by robust_eval.load_detector, or 'damien' for the
  teammate's clip model (damien_detector.py); @threshold overrides the stored one.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import keyword_config as K
import robust_eval as R

ROOT = Path(__file__).resolve().parent


def load(spec):
    name, path = spec.split('=', 1)
    path, _, th = path.partition('@')
    if path == 'damien':
        from damien_detector import DamienDetector
        det = DamienDetector()
    else:
        det = R.load_detector(path)
    if th:
        det.threshold = float(th)
    return name, path, det


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('models', nargs='+')
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sets, info = R.build_sets(a.data, 'test', None, R.STREAM_SECONDS, {'corpus', 'tts'})
    y, clipped = sets['live_y'], sets['live_clipped']
    report = {'keyword': K.KEYWORD, 'split': 'test', 'config': info, 'models': {}}
    for spec in a.models:
        name, path, det = load(spec)
        th = det.threshold
        r = {'file': path, 'threshold': th}
        for cond, audios in sets['live'].items():
            conf = np.array([c.max() for _, c, _ in det.traces(audios)])
            r[f'live_{cond}'] = {'recall': round(float((conf[y == 1] >= th).mean()), 4),
                                 'recall_complete': round(float((conf[(y == 1) & ~clipped] >= th).mean()), 4),
                                 'other_accepted': round(float((conf[y == 0] >= th).mean()), 5)}
        audio, marks, sinfo = sets['stream']
        (times, score, _), = det.traces([audio])
        r['stream_1h'] = R.score_stream(times, score, th, marks, sinfo['seconds'] / 3600, sinfo['segments'])
        neg, hours = R.negative_traces(det, a.data, 'test')
        r['negatives'] = {'hours': round(hours, 2), 'fa_per_hour': round(R.negative_fa(neg, hours, th), 3)}
        report['models'][name] = r
        print(name, json.dumps(r), flush=True)
        a.out.write_text(json.dumps(report, indent=1, default=str))


if __name__ == '__main__':
    main()
