"""Write the 40 verification vectors and a model's expected outputs for jtag/board_test.tcl."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import integer_forward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'deploy/model.npz')
    p.add_argument('--out', type=Path, default=ROOT / 'build/board_vectors.tcl')
    a = p.parse_args()
    q = dict(np.load(a.model))
    x = np.load(ROOT / 'results/verification_vectors.npz')['x']
    scores, spikes = integer_forward(x, q)   # independent integer oracle
    lines = [f'set threshold {int(q["decision_threshold"])}', 'set frames {']
    for v, (s0, s1), k in zip(x, scores, spikes):
        words = ' '.join(f'0x{n:08x}' for n in np.frombuffer(v.tobytes(), '<u4'))
        lines.append(f'{{{int(s0)} {int(s1)} {int(k)} {{{words}}}}}')
    lines.append('}')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text('\n'.join(lines) + '\n')
    print(f'{len(x)} vectors, threshold {int(q["decision_threshold"])} -> {a.out}')


if __name__ == '__main__':
    main()
