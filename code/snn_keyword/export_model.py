"""Export a selected integer model into C headers, retaining overflow checks."""
import argparse
from pathlib import Path
import numpy as np
from features import N_INPUT


def pack_w1(q):
    """Input weights as packed n-bit codes for the USE_PACKED loop (research S2).

    The model file stores full-scale Q10 weights w1 = c * 2**w1_shift; this recovers the n-bit
    codes c and puts 32/n of them in each 32-bit word, lane 0 in the low bits, neuron-major."""
    bits, shift = int(q.get('w1_bits', 16)), int(q.get('w1_shift', 0))
    if bits not in (16, 8, 4, 2):
        raise ValueError(f'packed loop supports 16/8/4/2-bit weights, not {bits}')
    w = q['w1'].astype(np.int64)
    c = w >> shift
    if not np.array_equal(c << shift, w) or c.min() < -2 ** (bits - 1) or c.max() > 2 ** (bits - 1) - 1:
        raise ValueError(f'w1 is not a {bits}-bit code times 2**{shift}')
    lanes = 32 // bits
    codes = (c & (2 ** bits - 1)).astype(np.uint64).reshape(len(w), N_INPUT // lanes, lanes)
    words = (codes << (np.arange(lanes, dtype=np.uint64) * bits)).sum(axis=2)
    return bits, shift, lanes, words.astype(np.uint32).flatten()


def sparse_w1(q):
    """Fixed fan-in synapse list for the USE_SPARSE loop (research S3): one 32-bit word per kept
    synapse, (int16 Q10 weight << 16) | input index, in input order, fan_in words per neuron."""
    mask, fan_in = q['w1_mask'].astype(bool), int(q['fan_in'])
    if not (mask.sum(1) == fan_in).all() or N_INPUT > 1024:
        raise ValueError('every neuron must keep exactly fan_in inputs, indices must fit 10 bits')
    rows, cols = np.nonzero(mask)                     # row-major: each neuron's inputs in order
    weights = q['w1'][rows, cols].astype(np.int64) & 0xFFFF
    return fan_in, ((weights << 16) | cols).astype(np.uint32)


def export(model, out, packed=False, sparse=False, lif_table=False):
    q = dict(np.load(model))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    config = ['#ifndef MODEL_CONFIG_H', '#define MODEL_CONFIG_H',
        f'#define MODEL_INPUT {N_INPUT}', f'#define MODEL_HIDDEN {len(q["w1"])}',
        f'#define MODEL_STEPS {int(q["steps"])}', f'#define MODEL_ENCODING {int(q["encoding"])}',
        f'#define MODEL_DECISION_THRESHOLD ({int(q["decision_threshold"])})',
        # Per-window threshold for "2 of 3" stream confirmation (tune_stream.py).
        f'#define MODEL_STREAM_THRESHOLD ({int(q.get("stream_threshold", q["decision_threshold"]))})']
    arrays = []   # (C type, name, values) of the research formats, appended after the release arrays
    if packed:
        bits, shift, lanes, words = pack_w1(q)
        config += [f'#define MODEL_W1_BITS {bits}', f'#define MODEL_W1_SHIFT {shift}', f'#define MODEL_W1_LANES {lanes}']
        arrays.append(('uint32_t', 'w1p', words))
    if sparse:
        fan_in, words = sparse_w1(q)
        config += [f'#define MODEL_W1_FANIN {fan_in}']
        arrays.append(('uint32_t', 'w1s', words))
    if lif_table:
        from model import lif_thresholds
        if int(q['encoding']):
            raise ValueError('the LIF threshold table needs current encoding (constant drive)')
        arrays.append(('int32_t', 'lif_threshold', np.array(lif_thresholds(int(q['steps'])))))
    (out / 'model_config.h').write_text('\n'.join(config + ['#endif', '']))
    lines = ['#include <stdint.h>', '#include "model_config.h"']
    for name in ['w1', 'b1', 'w2', 'b2']:
        kind = 'int16_t' if name.startswith('w') else 'int32_t'
        values = q[name].flatten()
        lines.append(f'static const {kind} model_{name}[{len(values)}] __attribute__((section(".model"), aligned(4))) = {{')
        lines.extend(','.join(map(str, values[i:i+24])) + ',' for i in range(0, len(values), 24))
        lines.append('};')
    for kind, name, values in arrays:
        lines.append(f'static const {kind} model_{name}[{len(values)}] __attribute__((section(".model"), aligned(4))) = {{')
        fmt = (lambda v: f'0x{int(v):08x}u') if kind == 'uint32_t' else str
        lines.extend(','.join(map(fmt, values[i:i+12])) + ',' for i in range(0, len(values), 12))
        lines.append('};')
    (out / 'model_data.h').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model', type=Path)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'build')
    a = p.parse_args(); export(a.model, a.out)
