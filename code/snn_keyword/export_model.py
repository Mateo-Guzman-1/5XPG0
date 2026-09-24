"""Export a selected integer model into C headers, retaining overflow checks."""
import argparse
from pathlib import Path
import numpy as np
from features import N_INPUT


def export(model, out):
    q = dict(np.load(model))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    config = '\n'.join(['#ifndef MODEL_CONFIG_H', '#define MODEL_CONFIG_H',
        f'#define MODEL_INPUT {N_INPUT}', f'#define MODEL_HIDDEN {len(q["w1"])}',
        f'#define MODEL_STEPS {int(q["steps"])}', f'#define MODEL_ENCODING {int(q["encoding"])}',
        f'#define MODEL_DECISION_THRESHOLD ({int(q["decision_threshold"])})', '#endif', ''])
    (out / 'model_config.h').write_text(config)
    lines = ['#include <stdint.h>', '#include "model_config.h"']
    for name in ['w1', 'b1', 'w2', 'b2']:
        kind = 'int16_t' if name.startswith('w') else 'int32_t'
        values = q[name].flatten()
        lines.append(f'static const {kind} model_{name}[{len(values)}] __attribute__((section(".model"), aligned(4))) = {{')
        lines.extend(','.join(map(str, values[i:i+24])) + ',' for i in range(0, len(values), 24))
        lines.append('};')
    (out / 'model_data.h').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model', type=Path)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'build')
    a = p.parse_args(); export(a.model, a.out)
