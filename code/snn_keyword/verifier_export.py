"""Verifier track, step 5: C headers for firmware/verifier.c and the bit-exact checks.

  export   q (verifier_model.quantize) -> build/verifier/verifier_config.h, verifier_data.h,
           with the int32 range checks of every intermediate of the C kernel.
  native   gcc build of sim/native_verifier.c (WSL) against the Python integer
           oracle on validation windows: logits and both keyword scores.

The weights are int8 stored as uint8 (w + 128), rows word aligned, in the .model
section; the tables and per-row exponents and biases as well.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import keyword_config as K
from verifier_data import BLANK, KEYWORD
from verifier_model import NEG, ONE, Verifier, integer_forward, keyword_score, quantize

ROOT = Path(__file__).resolve().parent


def load_quantized(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    K.check_model_keyword(ck.get('keyword'), 'verifier')     # checkpoints before the setting are "yes"
    m = Verifier(**ck['config'])
    m.load_state_dict(ck['state_dict'])
    return quantize(m)


def _arr(ctype, name, values, per_line=32):
    v = np.asarray(values).flatten()
    out = [f'static const {ctype} verifier_{name}[{len(v)}] __attribute__((section(".model"), aligned(4))) = {{']
    out += [','.join(map(str, v[i:i + per_line])) + ',' for i in range(0, len(v), per_line)]
    return out + ['};']


def range_checks(q, steps):
    checks = {}
    for name, nin, H in (('l1', 24 * int(q['stack']), int(q['h1'])), ('l2', int(q['h1']), int(q['h2']))):
        w = np.abs(q[f'{name}_w'].astype(np.int64))
        e, b = q[f'{name}_e'], np.abs(q[f'{name}_b'].astype(np.int64))
        acc = w.sum(1) * ONE + 255 * ONE * w.shape[1]          # kdot sum with the +128 offset
        gh = ((w[2 * H:, nin:].sum(1) * ONE) >> e[2 * H:]) + b[3 * H:]
        checks[f'{name} accumulator'] = int(acc.max())
        checks[f'{name} r * gh'] = int(gh.max() * ONE)
        if (e < 0).any():
            raise ValueError(f'{name}: negative row exponent')
    wo = np.abs(q['out_w'].astype(np.int64))
    lg = ((wo.sum(1) * ONE) >> q['out_e']) + np.abs(q['out_b'].astype(np.int64))
    checks['keyword path sum'] = int(2 * lg.max() * steps)
    if 'head_w' in q:
        if (q['head_e'] < 0).any():
            raise ValueError('head: negative row exponent')
        checks['head accumulator'] = int(np.abs(q['head_w'].astype(np.int64)).sum() * ONE + 255 * ONE * q['head_w'].shape[1])
    for k, v in checks.items():
        if v >= 2 ** 30:
            raise ValueError(f'{k} can reach {v}')
    return checks


def export(q, out, frames=150, warmup=5, boundary=10, thresholds=(NEG, NEG), cascade_t1=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    stack = int(q['stack'])
    checks = range_checks(q, frames // stack)
    cfg = ['#ifndef VERIFIER_CONFIG_H', '#define VERIFIER_CONFIG_H',
           f'#define VERIFIER_STACK {stack}', f'#define VERIFIER_IN {24 * stack}',
           f'#define VERIFIER_H1 {int(q["h1"])}', f'#define VERIFIER_H2 {int(q["h2"])}',
           f'#define VERIFIER_CLASSES {len(q["out_b"])}', f'#define VERIFIER_BLANK {BLANK}',
           f'#define VERIFIER_NPH {len(KEYWORD)}',
           f'#define VERIFIER_PHONES {{{", ".join(str(int(k)) for k in KEYWORD)}}}',
           f'#define VERIFIER_FRAMES {frames}', f'#define VERIFIER_WARMUP {warmup}',
           f'#define VERIFIER_BOUNDARY {boundary}', f'#define VERIFIER_NEG ({NEG})',
           f'#define VERIFIER_THRESHOLD_A ({int(thresholds[0])})', f'#define VERIFIER_THRESHOLD_B ({int(thresholds[1])})']
    if cascade_t1 is not None:   # stage-1 decision score that asks the verifier (firmware CASCADE builds)
        cfg.append(f'#define CASCADE_T1 ({int(cascade_t1)})')
    if 'head_w' in q:            # keyword head: the cascade decides on it (VERIFIER_THRESHOLD_A is its threshold)
        cfg += ['#define VERIFIER_HEAD 1', f'#define VERIFIER_HEAD_E {int(q["head_e"][0])}',
                f'#define VERIFIER_HEAD_B ({int(q["head_b"][0])})']
    cfg += ['#endif', '']
    (out / 'verifier_config.h').write_bytes('\n'.join(cfg).encode())
    lines = ['#include <stdint.h>', '#include "verifier_config.h"']
    for name in ('l1', 'l2'):
        k = name[1]
        lines += _arr('uint8_t', f'w{k}', q[f'{name}_w'].astype(np.int16) + 128)
        lines += _arr('uint8_t', f'e{k}', q[f'{name}_e']) + _arr('int32_t', f'b{k}', q[f'{name}_b'])
    lines += _arr('uint8_t', 'wo', q['out_w'].astype(np.int16) + 128)
    lines += _arr('uint8_t', 'eo', q['out_e']) + _arr('int32_t', 'bo', q['out_b'])
    if 'head_w' in q:
        lines += _arr('uint8_t', 'wh', q['head_w'].astype(np.int16) + 128)
    lines += _arr('int16_t', 'sig', q['sig']) + _arr('int16_t', 'tanh', q['tanh'])
    (out / 'verifier_data.h').write_bytes(('\n'.join(lines) + '\n').encode())
    size = sum(q[k].size for k in ('l1_w', 'l2_w', 'out_w', 'l1_e', 'l2_e', 'out_e')) + \
        4 * sum(q[k].size for k in ('l1_b', 'l2_b', 'out_b')) + 2 * (q['sig'].size + q['tanh'].size)
    return {'model_bytes': int(size), 'weights': int(sum(q[k].size for k in ('l1_w', 'l2_w', 'out_w'))),
            'range_checks': checks}


def oracle(frames, q, warmup, boundary):
    """Python reference of verifier_run: logits, (score_a, end_a, score_b, end_b, head, end_head) per window."""
    if 'head_w' in q:
        lg, hd = integer_forward(frames, q, with_head=True)
        h = hd[:, warmup:]
        head, end_head = h.max(1), h.argmax(1) + warmup
    else:
        lg = integer_forward(frames, q)
        head, end_head = np.full(len(frames), NEG, np.int64), np.full(len(frames), -1, np.int64)
    sa, ea = keyword_score(lg, warmup, 0, return_end=True)
    sb, eb = keyword_score(lg, warmup, boundary, return_end=True)
    return lg, np.c_[sa, ea, sb, eb, head, end_head]


def records(frames):
    out = bytearray()
    for f in frames:
        out += np.uint32(len(f)).tobytes() + np.ascontiguousarray(f, np.uint8).tobytes()
    return bytes(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--windows', type=Path, help='npz with "frames" (N, 150, 24) uint8 validation windows')
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--boundary', type=int, default=10)
    a = p.parse_args()
    from verify import command
    q = load_quantized(a.checkpoint)
    info = export(q, ROOT / 'build/verifier', warmup=a.warmup, boundary=a.boundary)
    print(json.dumps(info))
    if a.windows:
        frames = np.load(a.windows)['frames'][:a.n]
    else:   # random frames still exercise every path of the arithmetic
        frames = np.random.default_rng(0).integers(0, 256, (a.n, 150, 24)).astype(np.uint8)
    lg, res = oracle(frames, q, a.warmup, a.boundary)
    (ROOT / 'build/verifier_in.bin').write_bytes(records(frames))
    command(['bash', '-lc', 'gcc -O2 -Wall -Wextra -Werror -Ibuild/verifier -Ifirmware sim/native_verifier.c '
             'firmware/verifier.c -o build/native_verifier && ./build/native_verifier < build/verifier_in.bin '
             '> build/verifier_out.bin'])
    steps, C = lg.shape[1], lg.shape[2]
    got = np.fromfile(ROOT / 'build/verifier_out.bin', '<i4').reshape(len(frames), steps * C + 6)
    ok_l = (got[:, :-6].reshape(lg.shape) == lg).all()
    ok_r = (got[:, -6:] == res).all()
    print(json.dumps({'windows': len(frames), 'logits_bit_exact': bool(ok_l), 'scores_bit_exact': bool(ok_r),
                      'score_a_median': float(np.median(res[:, 0])), 'head': 'head_w' in q,
                      'head_median': float(np.median(res[:, 4]))}))
    if not (ok_l and ok_r):
        raise SystemExit('native C verifier differs from the oracle')


if __name__ == '__main__':
    main()
