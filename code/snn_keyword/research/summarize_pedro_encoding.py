"""Produce a concise paired-method comparison from completed evaluation data."""
import json
from pathlib import Path
import numpy as np
from pedro_encoding_eval import E, dump

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/pedro_encoding'


def stats(values):
    return {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1))}


def main():
    source = json.loads((OUT / 'summary.json').read_text())
    board = json.loads((OUT / 'board.json').read_text())['candidates']
    rows = {}
    for c in source['configurations']:
        runs = c['runs']
        row = {k: c[k] for k in ('key', 'config', 'input_bits', 'parameter_bytes')}
        for name, get in {
            'validation_recall': lambda r: r['validation']['recall'],
            'clip_accuracy': lambda r: r['old_clip_test']['accuracy'],
            'clip_f1': lambda r: r['old_clip_test']['f1'],
            'full_clip_f1': lambda r: r['full_clip_test']['f1'],
            'live_recall': lambda r: r['live_test']['recall'],
            'live_accuracy': lambda r: r['live_test']['accuracy'],
            'live_f1': lambda r: r['live_test']['f1'],
            'other_acceptance': lambda r: r['live_test']['fp'] / 3000,
            'fa_per_hour': lambda r: r['test_fa_per_hour'],
            'old_rule_live_recall': lambda r: r['old_threshold_live_test']['recall'],
            'old_rule_fa_per_hour': lambda r: r['old_threshold_test_fa_per_hour'],
        }.items():
            row[name] = stats([get(r) for r in runs])
        row['stream_latency_median_ms'] = [None if r['stream']['latency_median_s'] is None else 1000 * r['stream']['latency_median_s'] for r in runs]
        row['stream_recall'] = stats([r['stream']['recall'] for r in runs])
        row['thresholds'] = [r['threshold'] for r in runs]
        row['fpga_ms'] = board[c['key']]['milliseconds_mean_at_100mhz']
        rows[c['key']] = row
    def rank(key):
        r = rows[key]
        return r['validation_recall']['mean'], -r['parameter_bytes'], -r['input_bits']
    selected = max(rows, key=rank)
    winners = {g: max(keys, key=rank) for g, keys in source['groups'].items()}
    dump(OUT / 'aggregate.json', dict(selected_on_validation=selected, group_winners=winners, rows=rows))
    def pct(r, metric):
        v = r[metric]
        return f"{100*v['mean']:.2f} ± {100*v['sd']:.2f}"
    baseline, chosen = rows[E.BASELINE.key], rows[selected]
    originals = {r['key']: r for r in source['configurations']}
    paired = {}
    for key in rows:
        differences = np.array([r['live_test']['recall'] for r in originals[key]['runs']]) - np.array([r['live_test']['recall'] for r in originals[E.BASELINE.key]['runs']])
        mean, sd = float(differences.mean()) * 100, float(differences.std(ddof=1)) * 100
        half = 4.302652729911275 * sd / np.sqrt(3)
        paired[key] = dict(mean_pp=mean, sd_pp=sd, ci95_low_pp=mean-half, ci95_high_pp=mean+half)
    dump(OUT / 'paired_differences.json', dict(metric='live recall difference relative to 8-band reference',
         uncertainty='paired t interval over three training seeds, not speakers; no multiplicity correction', differences=paired))
    lines = ['# Encoding: comparison of clip and continuous-listening evaluation', '',
             'All 61 encodings were evaluated with three matched seeds using the original compact SNN training recipe. '
             'The same integer checkpoints were then scored at the clip-F1 threshold and at Pedro R1’s validation-selected streaming threshold. '
             'Numbers below are mean ± sample SD across seeds. The detailed per-model results and margins are saved with the run.', '',
             f"Negative audio: {source['negative_hours']['validation']:.3f} validation hours and {source['negative_hours']['test']:.3f} test hours. "
             'Live tests use 212 Sheila and 3,000 other-word clips, randomly placed in background noise. '
             'The standalone validation limits are 0.2% other-word acceptance and 2 false accepts/hour.', '',
             '| Baseline metric | Result |', '|---|---:|',
             f"| Isolated-clip accuracy (%) | {pct(baseline, 'clip_accuracy')} |",
             f"| Isolated-clip F1 (%) | {pct(baseline, 'clip_f1')} |",
             f"| Full 11,005-clip test F1 (%) | {pct(baseline, 'full_clip_f1')} |",
             f"| Live recall, clip-F1 threshold (%) | {pct(baseline, 'old_rule_live_recall')} |",
             f"| FA/h, clip-F1 threshold | {baseline['old_rule_fa_per_hour']['mean']:.2f} |",
             f"| Live recall, streaming threshold (%) | {pct(baseline, 'live_recall')} |",
             f"| FA/h, streaming threshold | {baseline['fa_per_hour']['mean']:.2f} |",
             f"| Input / parameters / FPGA inference | {baseline['input_bits']} bits / {baseline['parameter_bytes']} bytes / {baseline['fpga_ms']:.2f} ms |", '',
             '## Sweep comparison', '',
             'Each row shows the setting selected on mean validation recall within that sweep. '
             'Test values describe that fixed choice; they were not used to select it.', '',
             '| Variable | Validation-selected setting | Clip accuracy (%) | Clip F1 (%) | Live recall (%) | FA/h | Input bits | Parameters (B) | FPGA (ms) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    fields = {'bits': ['bits'], 'bands': ['bands'], 'range': ['low_hz', 'high_hz'], 'duration': ['duration_ms'],
              'time_bins': ['time_bins'], 'activity': ['rms_gate_u6', 'rms_full_scale_u6'],
              'band_mapping': ['band_mapping'], 'fft': ['fft_window_ms', 'fft_hop_ms'],
              'quantizer': ['bits', 'quantizer'], 'feature_mode': ['feature_mode']}
    for group, key in winners.items():
        r = rows[key]
        setting = ', '.join(f'{k}={r["config"][k]}' for k in fields[group])
        lines.append(f"| {group} | {setting} | {pct(r, 'clip_accuracy')} | {pct(r, 'clip_f1')} | {pct(r, 'live_recall')} | {r['fa_per_hour']['mean']:.2f} | {r['input_bits']} | {r['parameter_bytes']} | {r['fpga_ms']:.2f} |")
    lines += ['', '## Interpretation', '',
              'Clip F1 and live recall have different denominators and operating points, so their difference is not an accuracy loss in percentage points. '
              'The paired old/new threshold comparison above isolates the threshold trade-off on the same streaming audio. '
              'Changing the evaluation protocol does not change model memory or input bits. FPGA latency is measured separately; acoustic latency is conditional on successfully detected words.', '',
              'The duration sweep retains centre cropping/zero padding inside a one-second request. Longer durations add no new acoustic context. '
              'Seed SD does not quantify uncertainty across speakers or rooms. The offline streaming measurements do not constitute a physical microphone-to-LED test.', '',
              f'Validation-selected configuration: `{selected}`.',
              f"Board checks: {sum(r['vectors'] for r in board.values())} vectors; {sum(r['score_mismatches'] for r in board.values())} mismatches."]
    (ROOT / 'PEDRO_ENCODING_COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'baseline': baseline, 'selected': chosen}, indent=2))


if __name__ == '__main__':
    main()
