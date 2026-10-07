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
        f'#define MODEL_DECISION_THRESHOLD ({int(q["decision_threshold"])})',
        # Per-window threshold for "2 of 3" stream confirmation (tune_stream.py).
        f'#define MODEL_STREAM_THRESHOLD ({int(q.get("stream_threshold", q["decision_threshold"]))})', '#endif', ''])
    (out / 'model_config.h').write_text(config)
    lines = ['#include <stdint.h>', '#include "model_config.h"']
    for name in ['w1', 'b1', 'w2', 'b2']:
        kind = 'int16_t' if name.startswith('w') else 'int32_t'
        values = q[name].flatten()
        lines.append(f'static const {kind} model_{name}[{len(values)}] __attribute__((section(".model"), aligned(4))) = {{')
        lines.extend(','.join(map(str, values[i:i+24])) + ',' for i in range(0, len(values), 24))
        lines.append('};')
    (out / 'model_data.h').write_text('\n'.join(lines) + '\n')


def _array(ctype, name, values, per_line=24):
    values = np.asarray(values).flatten()
    lines = [f'static const {ctype} stream_{name}[{len(values)}] __attribute__((section(".model"), aligned(4))) = {{']
    lines += [','.join(map(str, values[i:i + per_line])) + ',' for i in range(0, len(values), per_line)]
    return lines + ['};']


def export_stream(model, out):
    """C headers for firmware/stream_infer.c from an integer stream model (model.quantize_stream).

    Layer-1 -> layer-2 synapses are stored as a CSR table grouped by (source
    neuron, delay), zero weights dropped, so the kernel touches only synapses
    whose delayed spike arrives this frame. Also checks that every int32
    intermediate of the kernel stays in range (the oracle computes in int64).
    """
    q = dict(np.load(model))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    n1, n2, c = int(q['n1']), int(q['n2']), int(q['classes'])
    # Overflow bounds of the C kernel (int32 intermediates).
    acc1 = np.abs(q['w1'].astype(np.int64)).sum(1) * 255
    acc2 = (np.abs(q['w12'].astype(np.int64)).sum(1) + np.abs(q['rec'].astype(np.int64)).sum(1)) << q['p2']
    a_max = {k: 256 << q['ka' + k].astype(np.int64) for k in ('1', '2')}
    checks = {'layer-1 accumulator': acc1.max(), 'layer-2 current': (acc2 + np.abs(q['b2'])).max(),
              'layer-1 threshold product': (np.abs(q['bq1']) * a_max['1']).max(),
              'layer-2 threshold product': (np.abs(q['bq2']) * a_max['2']).max()}
    for what, v in checks.items():
        if v >= 2 ** 31:
            raise ValueError(f'{what} can reach {v}, beyond int32')
    syn_post, syn_w, off = [], [], [0]
    delay, w12 = q['delay'].astype(int), q['w12'].astype(int)
    for i in range(n1):
        for tau in range(32):
            js = np.flatnonzero((delay[:, i] == tau) & (w12[:, i] != 0))
            syn_post += js.tolist(); syn_w += w12[js, i].tolist(); off.append(len(syn_post))
    others = [k for k in range(c) if k != int(q['yes_class'])]
    config = ['#ifndef MODEL_STREAM_CONFIG_H', '#define MODEL_STREAM_CONFIG_H',
              '#define STREAM_BANDS 24', f'#define STREAM_N1 {n1}', f'#define STREAM_N2 {n2}',
              f'#define STREAM_CLASSES {c}', f'#define STREAM_YES {int(q["yes_class"])}',
              f'#define STREAM_PO {int(q["po"])}', f'#define STREAM_SYNAPSES {len(syn_w)}',
              f'#define STREAM_THRESHOLD ({int(q["stream_threshold"])})',
              f'#define STREAM_WINDOW {int(q.get("decision_window", 1))}', '#endif', '']
    assert 1 <= int(q.get('decision_window', 1)) <= 64
    assert others and len(syn_w) < 65536
    (out / 'model_stream_config.h').write_text('\n'.join(config))
    lines = ['#include <stdint.h>', '#include "model_stream_config.h"']
    lines += _array('int16_t', 'w1', q['w1']) + _array('int32_t', 'b1', q['b1']) + _array('int32_t', 'theta1', q['theta1'])
    lines += _array('uint8_t', 'r1', q['r1']) + _array('uint8_t', 'km1', q['km1']) + _array('uint8_t', 'ka1', q['ka1'])
    lines += _array('int32_t', 'bq1', q['bq1'])
    lines += _array('uint16_t', 'syn_off', off) + _array('uint8_t', 'syn_post', syn_post) + _array('int8_t', 'syn_w', syn_w)
    lines += _array('int8_t', 'rect', q['rec'].T) + _array('int32_t', 'b2', q['b2']) + _array('int32_t', 'theta2', q['theta2'])
    lines += _array('uint8_t', 'p2', q['p2']) + _array('uint8_t', 'km2', q['km2']) + _array('uint8_t', 'ka2', q['ka2'])
    lines += _array('int32_t', 'bq2', q['bq2'])
    lines += _array('int8_t', 'wot', q['wo'].T) + _array('int32_t', 'bo', q['bo']) + _array('uint8_t', 'ko', q['ko'])
    (out / 'model_stream_data.h').write_text('\n'.join(lines) + '\n')
    return len(syn_w)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model', type=Path)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'build')
    a = p.parse_args()
    if str(np.load(a.model).get('kind', 'window')) == 'stream':
        print('synapses', export_stream(a.model, a.out))
    else:
        export(a.model, a.out)
