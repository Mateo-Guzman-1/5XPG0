"""Quality part of the size benchmark: score integer models exactly as the board computes them.

For every model (name=path.npz), on the fixed inputs of benchmark_inputs.py:
  test        the 11,005 official test clips (confusion matrix, precision, recall, F1,
              false-positive rate, mean hidden spikes) and the confusables.py metrics:
              live stream (212 keyword + 3,000 other test clips in noise, 250 ms hop,
              single window and "2 of 3"), hard-negative test variants (augment.py) and
              the synthesized-word probe; plus the background-noise probe (robustness.py)
  validation  the same clip, live and variant metrics on the validation split, the only
              data used for choices (the probe stays held out)
Thresholds are the ones stored in each model (decision: training; stream: tune_stream.py).
The integer arithmetic is bit-exact with the board; bench_board.py checks that per model.

    eval_model.py h32_seed0=research/results/E2/models/h32_seed0.npz ... --out quality.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

KWS = Path(__file__).resolve().parents[2]   # code/snn_keyword
sys.path.insert(0, str(KWS))
from benchmark_inputs import describe
from confusables import evaluate, tts_sequences
from model import integer_forward, keyword_of, metrics
from robustness import background_probe, background_windows
from tune_stream import live_windows


def shown(path):
    """Repository-relative path when possible, for the report."""
    path = Path(path).resolve()
    return path.relative_to(KWS.parents[1]).as_posix() if path.is_relative_to(KWS.parents[1]) else path.as_posix()


def clip_metrics(x, y, q):
    """Full confusion-matrix metrics at the decision threshold, and mean hidden spikes per clip."""
    scores, spikes = integer_forward(x, q)
    m = metrics(y, scores[:, 1] - scores[:, 0] >= int(q['decision_threshold']))
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, round(float(spikes.mean()), 2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('models', nargs='+', help='name=path/to/model.npz')
    p.add_argument('--data', type=Path, default=KWS / 'data')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    d, z = np.load(a.data / 'features.npz'), np.load(a.data / 'augment.npz')
    keyword = keyword_of(d)
    if keyword_of(z) != keyword:
        raise SystemExit(f'augment.npz was made for "{keyword_of(z)}"; run augment.py for "{keyword}" first')
    sets = {name: {'clips': (d['x'][d['split'] == s], d['y'][d['split'] == s]),
                   'aug': (z['x'][z['split'] == s], z['kind'][z['split'] == s]),
                   # The windows of confusables.py (test) and tune_stream.py (validation).
                   'live': live_windows(a.data, 3000, seed=1, split=2) if s == 2 else live_windows(a.data, 3000)}
            for name, s in (('test', 2), ('validation', 1))}
    tts = tts_sequences(a.data, KWS / f'build/tts_probe_{keyword}')
    if tts is None:
        raise SystemExit(f'No synthesized-word probe: run make_tts_probe.ps1 -Keyword {keyword}')
    noise = background_windows()
    report = {'benchmark': describe(keyword)['fingerprint'], 'keyword': keyword,
              'test_clips': len(sets['test']['clips'][1]), 'tts_utterances': len(tts[0]) // 4, 'models': {}}
    for spec in a.models:
        name, path = spec.split('=', 1)
        q = dict(np.load(path))
        if keyword_of(q) != keyword:
            raise SystemExit(f'{path} detects "{keyword_of(q)}" but the data is prepared for "{keyword}"')
        if 'stream_threshold' not in q:
            raise SystemExit(f'{path} has no stream threshold: run tune_stream.py first')
        out = {'file': shown(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               'hidden': int(len(q['w1'])), 'parameters': int(sum(q[k].size for k in ('w1', 'b1', 'w2', 'b2'))),
               'parameter_bytes': int(sum(q[k].nbytes for k in ('w1', 'b1', 'w2', 'b2'))),
               'decision_threshold': int(q['decision_threshold']), 'stream_threshold': int(q['stream_threshold'])}
        for split, probe in (('test', tts), ('validation', None)):
            s = sets[split]
            m = evaluate(q, s['clips'], s['live'], s['aug'], probe, keyword)
            for k in ('decision_threshold', 'stream_threshold', 'hidden'):
                del m[k]
            m['clips'], m['hidden_spikes_mean'] = clip_metrics(*s['clips'], q)
            out[split] = m
        out['test']['background_noise'] = background_probe(q, noise)
        report['models'][name] = out
        t = out['test']
        print(f'{name}: test F1 {t["clips"]["f1"]:.4f}, live 2 of 3 {t["live"]["two_of_three"]}, '
              f'noise {t["background_noise"]["false_detections"]}/{t["background_noise"]["windows"]}', flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
