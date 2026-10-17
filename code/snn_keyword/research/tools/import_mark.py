"""S2: copy Mark's bits-per-weight results (branch Quantization) into research/results/S2/mark_comparison.csv.

Mark quantized only the input-layer weights of Damien's 128-input network (8 x 16 features,
128 -> 48 -> 2 LIF, 16 steps), with one power-of-two scale per layer, and compared quantization-aware
fine-tuning (4 epochs from float checkpoints) with a float fine-tuning control followed by rounding.
Thresholds were chosen on validation F1. His test set has 3,212 clips (212 "sheila", 3,000 others),
ours 11,005, so only changes relative to each network's own 8-bit result are compared.

    python import_mark.py            (reads the file with `git show`; nothing is checked out)
"""
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANCH = 'origin/Quantization'
FILE = 'code/snn_keyword/research/results/quant/qat.json'
OUT = ROOT / 'results/S2/mark_comparison.csv'


def main():
    commit = subprocess.run(['git', 'rev-parse', '--short', BRANCH], capture_output=True, text=True,
                            check=True, cwd=ROOT).stdout.strip()
    data = json.loads(subprocess.run(['git', 'show', f'{BRANCH}:{FILE}'], capture_output=True, text=True,
                                     check=True, cwd=ROOT).stdout)
    rows = [dict(source=f'{BRANCH} {commit} {FILE}', method=r['method'], bits=r['bits_w1'], seed=r['seed'],
                 val_f1=r['validation']['f1'], test_f1=r['test']['f1'], test_accuracy=r['test']['accuracy'],
                 w1_zero_fraction=r['w1_zero_fraction'], clipped_weights=r['clipped_weights'],
                 weight_bytes=r['bytes']['total'])
            for r in data['runs']]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for method in ('qat', 'control_ft_then_ptq'):
        for bits in sorted({r['bits'] for r in rows}, reverse=True):
            f1 = [r['test_f1'] for r in rows if r['method'] == method and r['bits'] == bits]
            print(f'{method:20s} {bits} bits: test F1 {sum(f1) / len(f1):.4f} ({len(f1)} seeds)')
    print('written', OUT)


if __name__ == '__main__':
    main()
