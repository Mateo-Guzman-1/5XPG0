"""Synapse benchmark (research/PLAN_SYNAPSE.md S2/S3): the E2 benchmark for dense networks with fewer
bits per input weight (S2) or a fixed number of input weights per neuron (S3).

A configuration is the hidden size H, the bits per input weight and the fan-in, named
h<H>[_b<bits>][_k<fan-in>] (h48_b4, h95_k64, h48_b4_k128); 16 bits and fan-in 0 are the release
(E2) values and are left out of the name. Per configuration and seed, exactly as size_sweep.py:
  train  train_keyword_snn.py, E2 recipe + --w1-bits/--fan-in -> runs_s/<config>/ (ignored by git)
  tune   tune_stream.py; model and histories copied to research/results/<exp>/models|training/
  eval   eval_model.py (all seeds of a configuration) -> research/results/<exp>/quality/<config>.json
  bench  bench_board.py --variant <variants> -> research/results/<exp>/board/<config>_seed<s>.json
The bench variants default to the research firmware the configuration needs: `packed` for
16/8/4/2-bit dense models, `sparse` for fan-in models. Configurations without such firmware
(6 or 3 bits) are not benchmarked. Steps whose output exists are skipped (--force redoes them).
The fixed inputs must have the same fingerprint as E2's, so all results are comparable.
Afterwards restore the demo with run_board.ps1.

    synapse_sweep.py --exp S2 --hidden 16 48 --bits 8 6 4 3 2 --steps train tune eval
    synapse_sweep.py --exp S2 --hidden 48 --bits 8 4 2 --steps bench
    synapse_sweep.py --exp S3 --hidden 48 95 --fan-in 16 32 64 128 256 512

--ptq replaces training by post-training rounding of the E2 checkpoints (runs_e2/h<H>/, same seed):
the input weights are rounded to n bits with the same power-of-two rule, and the decision threshold
is chosen on the same validation data as in training (name suffix _ptq).
"""
import argparse
import itertools
import json
import shutil
from pathlib import Path
import sys
import numpy as np

TOOLS = Path(__file__).resolve().parent
KWS = TOOLS.parents[1]   # code/snn_keyword
sys.path.insert(0, str(KWS))
sys.path.insert(0, str(TOOLS))
from benchmark_inputs import describe
from model import keyword_of
from size_sweep import RECIPE, run
from tune_stream import live_windows

STEPS = ['train', 'tune', 'eval', 'bench']
E2_INPUTS = TOOLS.parent / 'results/E2/benchmark_inputs.json'


def config_name(hidden, bits, fan_in):
    return f'h{hidden}' + (f'_b{bits}' if bits < 16 else '') + (f'_k{fan_in}' if fan_in else '')


def default_variants(bits, fan_in):
    if fan_in:
        return ['sparse']
    return ['packed'] if bits in (16, 8, 4, 2) else []


def ptq(checkpoint, bits, out):
    """Post-training rounding control: an E2 checkpoint with its input weights rounded to n bits."""
    import torch
    from model import SpikeMLP, quantize, integer_forward, choose_threshold, metrics
    ck = torch.load(checkpoint, map_location='cpu')
    m = SpikeMLP(ck['hidden'], ck['steps'], ck['encoding'], w1_bits=bits)
    m.load_state_dict(ck['state_dict'])
    m.fix_w1_shift()
    q = quantize(m)
    # The validation data train_keyword_snn.py chooses thresholds on (E2 recipe: clips + all variants).
    data = KWS / 'data'
    cache, aug = np.load(data / 'features.npz'), np.load(data / 'augment.npz')
    va = aug['split'] == 1
    xv = np.concatenate([cache['x'][cache['split'] == 1], aug['x'][va]])
    yv = np.concatenate([cache['y'][cache['split'] == 1], aug['y'][va]])
    logits, _ = integer_forward(xv, q)
    margin = logits[:, 1] - logits[:, 0]
    q['decision_threshold'] = int(choose_threshold(yv, margin))
    q['keyword'] = np.array(keyword_of(cache))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'model.npz', **q)
    (out / 'training.json').write_text(json.dumps(dict(
        method='post-training rounding of an E2 checkpoint', checkpoint=str(checkpoint), w1_bits=bits,
        w1_shift=int(q['w1_shift']), w1_clipped=int(q['w1_clipped']), hidden=ck['hidden'],
        decision_threshold=q['decision_threshold'], validation=metrics(yv, margin >= q['decision_threshold'])), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exp', required=True, help='experiment folder under research/results, e.g. S2')
    p.add_argument('--hidden', nargs='+', type=int, required=True)
    p.add_argument('--bits', nargs='+', type=int, default=[16], help='bits per input weight')
    p.add_argument('--fan-in', nargs='+', type=int, default=[0], help='input weights kept per neuron (0 = all)')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', nargs='+', choices=STEPS, default=STEPS)
    p.add_argument('--variants', nargs='+', help='bench_board.py variants (default: the research firmware)')
    p.add_argument('--runs', type=Path, default=KWS / 'runs_s', help='training runs (ignored by git)')
    p.add_argument('--board', default='pynq')
    p.add_argument('--host', default='10.43.0.1')
    p.add_argument('--force', action='store_true')
    p.add_argument('--ptq', action='store_true', help='round the E2 checkpoints instead of training (S2 control)')
    a = p.parse_args()
    out = TOOLS.parent / 'results' / a.exp
    out.mkdir(parents=True, exist_ok=True)
    data = KWS / 'data'
    keyword = keyword_of(np.load(data / 'features.npz'))
    live_windows(data, 3000)
    live_windows(data, 3000, seed=1, split=2)
    bench = describe(keyword)
    if json.loads(E2_INPUTS.read_text())['fingerprint'] != bench['fingerprint']:
        raise SystemExit('The benchmark inputs differ from E2; results would not be comparable')
    (out / 'benchmark_inputs.json').write_text(json.dumps(bench, indent=2))
    for h, bits, fan_in in itertools.product(a.hidden, a.bits, a.fan_in):
        name = config_name(h, bits, fan_in) + ('_ptq' if a.ptq else '')
        runs = a.runs / name
        models = {s: out / 'models' / f'{name}_seed{s}.npz' for s in a.seeds}
        if 'train' in a.steps:
            todo = [s for s in a.seeds if a.force or not (runs / f'current_seed{s}/model.npz').exists()]
            if a.ptq:
                if fan_in:
                    raise SystemExit('--ptq is for precision only')
                for s in todo:
                    print(f'> post-training rounding {name} seed {s}', flush=True)
                    ptq(KWS / 'runs_e2' / f'h{h}' / f'current_seed{s}' / 'checkpoint.pt', bits, runs / f'current_seed{s}')
            elif todo:
                run(KWS / 'train_keyword_snn.py', *RECIPE, '--hidden', h, '--w1-bits', bits, '--fan-in', fan_in,
                    '--seeds', *todo, '--out', runs, log=runs / 'train.log')
        if 'tune' in a.steps:
            for s, model in models.items():
                if model.exists() and not a.force:
                    continue
                src = runs / f'current_seed{s}'
                run(KWS / 'tune_stream.py', src / 'model.npz', log=out / 'training' / f'{name}_seed{s}_stream.json')
                shutil.copyfile(src / 'training.json', out / 'training' / f'{name}_seed{s}_training.json')
                model.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src / 'model.npz', model)
        if 'eval' in a.steps:
            quality = out / 'quality' / f'{name}.json'
            if a.force or not quality.exists():
                run(TOOLS / 'eval_model.py', *[f'{name}_seed{s}={m}' for s, m in models.items()], '--out', quality)
        if 'bench' in a.steps:
            variants = a.variants or default_variants(bits, fan_in)
            if not variants:
                print(f'{name}: no research firmware for {bits}-bit weights; not benchmarked', flush=True)
                continue
            for s, model in models.items():
                board = out / 'board' / f'{name}_seed{s}.json'
                if a.force or not board.exists():
                    run(TOOLS / 'bench_board.py', model, '--variant', *variants, '--board', a.board,
                        '--host', a.host, '--out', board)
    print('Done. Restore the demo with run_board.ps1.', flush=True)


if __name__ == '__main__':
    main()
