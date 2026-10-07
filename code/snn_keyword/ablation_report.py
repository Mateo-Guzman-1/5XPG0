"""Collect Phase 3 validation selections into results/stream_ablation.json.

Each variant's checkpoints (model.pt, last.pt) were scored by stream_select.py
on validation data under one rule (live false accepts <= 0.2%, <= 2 stream
false accepts per hour); the better one by live + stream recall represents
the variant. Differences are against the baseline (no distillation).
"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {'s2_nokd': 'baseline (no distillation)', 's2_kd': '+ distillation', 's2_pcen': 'PCEN front end',
          's2_nodelay': 'no delays', 's2_lif': 'LIF (no adaptation)', 's2_norec': 'no recurrence',
          's2_notts': 'no TTS negatives', 's2_n64': '64 + 64 neurons'}


def main():
    rows = {}
    for f in sorted(glob.glob(str(ROOT / 'results/stream_selection_s2_*.json'))):
        for ck, r in json.load(open(f))['models'].items():
            if r is None:
                continue
            run = Path(ck).parent.name.replace('_seed0', '')
            score = r['live_recall'] + r['stream']['recall']
            if run not in rows or score > rows[run]['score']:
                rows[run] = dict(score=score, checkpoint=Path(ck).as_posix(), live_recall=round(r['live_recall'], 4),
                                 live_fa=round(r['live_fa'], 4), stream_recall=r['stream']['recall'],
                                 stream_fa_per_hour=r['stream']['fa_per_hour'],
                                 latency_median_s=r['stream']['latency_median_s'],
                                 mswc_recall=round(r['mswc_recall'], 4))
    base = rows.get('s2_nokd')
    out = {'rule': 'validation only; live FA <= 0.2%, stream <= 2 FA/h; best of model.pt/last.pt', 'variants': {}}
    for run, r in rows.items():
        r = dict(r); r.pop('score')
        if base:
            for k in ('live_recall', 'stream_recall', 'mswc_recall'):
                r[f'{k}_vs_baseline_points'] = round(100 * (r[k] - base[k]), 1)
        out['variants'][LABELS.get(run, run)] = r
    (ROOT / 'results/stream_ablation.json').write_text(json.dumps(out, indent=2))
    for name, r in out['variants'].items():
        print(f"{name:30s} live {r['live_recall']:.3f} stream {r['stream_recall']:.3f} mswc {r['mswc_recall']:.3f}"
              f"  Δ {r.get('live_recall_vs_baseline_points', 0):+.1f} / {r.get('stream_recall_vs_baseline_points', 0):+.1f}"
              f" / {r.get('mswc_recall_vs_baseline_points', 0):+.1f}")


if __name__ == '__main__':
    main()
