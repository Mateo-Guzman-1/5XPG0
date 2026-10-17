"""Collect every model of E2, S2, S3 and S4 into two tables (research/PLAN_SYNAPSE.md S5).

    research/results/S5/models.csv    one row per model: configuration, synapses, memory and quality
    research/results/S5/designs.csv   one row per model and measured firmware variant: cycles, memory, energy

Quality columns use exactly E2's definitions (plot_size_sweep.py). The selection metric is val_f1:
the validation F1 of training's checkpoint and threshold choice, clips plus hard-negative variants.
Cycles come only from bit-exact board runs (bench_board.py), joined to the model by its file path.
Energy comes from energy_model.py with the fabric each variant runs on.

    python collect_designs.py
"""
import csv
import json
from pathlib import Path
import numpy as np
from energy_model import load_power, energy

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
REPO = ROOT.parents[2]      # 5XPG0: model paths in the result files are relative to it
EXPERIMENTS = ['E2', 'S2', 'S3']                           # experiments that own models
BOARD = ['E2', 'S2', 'S3', 'S4']                           # experiments with board runs
FABRIC = {'base': 'base', 'kdot': 'kdot', 'packed': 'base', 'sparse': 'base', 'lut': 'base', 'kdot_lut': 'kdot'}
N_INPUT = 768


def memory(q, variant):
    """Bytes of the model's arrays as stored by this firmware variant (computed; the board JSON has
    the linker's measured .model size, which may hold small arrays in the code instead)."""
    h = len(q['w1'])
    rest = 4 * h + 2 * 2 * h + 4 * 2          # b1 int32, w2 int16 [2,H], b2 int32 [2]
    if variant == 'sparse':
        return 4 * h * int(q['fan_in']) + rest
    if variant == 'packed':
        return h * N_INPUT * int(q.get('w1_bits', 16)) // 8 + rest
    return 2 * h * N_INPUT + rest


def model_rows():
    rows = {}
    for exp in EXPERIMENTS:
        for qfile in sorted((RESULTS / exp / 'quality').glob('*.json')):
            data = json.loads(qfile.read_text())
            keyword = data.get('keyword', 'sheila')
            for name, m in data['models'].items():
                q = dict(np.load(REPO / m['file']))
                training = json.loads((RESULTS / exp / 'training' / f'{name}_training.json').read_text())
                t, v = m['test'], m['validation']
                h, bits = len(q['w1']), int(q.get('w1_bits', 16))
                fan_in = int(q.get('fan_in', 0)) or N_INPUT
                nonzero = int((q['w1'] != 0).sum())
                row = dict(name=name, exp=exp, config=name.rsplit('_seed', 1)[0], seed=int(name.rsplit('_seed', 1)[1]),
                           model=m['file'], hidden=h, bits=bits, fan_in=fan_in,
                           method='ptq' if name.split('_seed')[0].endswith('_ptq') else 'qat',
                           synapses=h * fan_in + 2 * h, w1_nonzero=nonzero,
                           w1_zero_fraction=1 - nonzero / (h * N_INPUT), w1_shift=int(q.get('w1_shift', 0)),
                           w1_clipped=int(q.get('w1_clipped', 0)),
                           memory_int16_dense=memory(q, 'base'),
                           memory_packed=memory(q, 'packed') if fan_in == N_INPUT else '',
                           memory_sparse=memory(q, 'sparse') if fan_in < N_INPUT else '',
                           memory_compact=(h * fan_in * (10 + bits) + 7) // 8 + 4 * h + 4 * h + 8 if fan_in < N_INPUT else '',
                           val_f1=training['validation']['f1'],
                           val_live_2of3_detected=v['live']['two_of_three'][f'{keyword}_detected'],
                           test_precision=t['clips']['precision'], test_recall=t['clips']['recall'],
                           test_f1=t['clips']['f1'], test_fpr=t['clips']['fpr'],
                           live_single_detected=t['live']['single_window'][f'{keyword}_detected'],
                           live_single_other=t['live']['single_window']['other_accepted'],
                           live_2of3_detected=t['live']['two_of_three'][f'{keyword}_detected'],
                           live_2of3_other=t['live']['two_of_three']['other_accepted'],
                           noise_false_detections=t['background_noise']['false_detections'],
                           hidden_spikes_mean=t['hidden_spikes_mean'])
                row.update({f'hardneg_{k}': x for k, x in t['augmented_detected']['stream_threshold'].items()})
                row.update({f'tts_2of3_{k}': x for k, x in t['tts']['two_of_three'].items()})
                rows[str(Path(m['file']).as_posix())] = row
    return rows


def design_rows(models):
    power = load_power()
    out = []
    for exp in BOARD:
        for bfile in sorted((RESULTS / exp / 'board').glob('*.json')):
            b = json.loads(bfile.read_text())
            m = models.get(Path(b['model']).as_posix())
            if m is None:
                raise SystemExit(f'{bfile}: model {b["model"]} has no quality results')
            for variant, r in b['variants'].items():
                if not r['bit_exact']:
                    raise SystemExit(f'{bfile} {variant}: not bit-exact; its cycles cannot be used')
                s = r['single_window']
                fabric = FABRIC[variant]
                q = dict(np.load(REPO / b['model']))
                macs = (len(q['w1']) * (m['fan_in'] if variant == 'sparse' else N_INPUT)) + 2 * len(q['w1'])
                e = energy(s['cycles_mean'], s['cycles_max'], power[fabric], macs)
                image = r.get('image', {'base': 'keyword', 'kdot': 'keyword_kdot'}.get(variant))
                out.append(dict(name=m['name'], exp=m['exp'], config=m['config'], seed=m['seed'], hidden=m['hidden'],
                                bits=m['bits'], fan_in=m['fan_in'], method=m['method'], variant=variant, fabric=fabric,
                                board_file=bfile.relative_to(ROOT).as_posix(),
                                model_bytes_linked=b['firmware']['images'][image]['model_bytes'],
                                memory=memory(q, variant), macs=macs,
                                cycles_mean=s['cycles_mean'], cycles_max=s['cycles_max'],
                                t_max_ms=e['t_max'] * 1e3, e_inf_uJ=e['e_inf'] * 1e6, e_inf_all_uJ=e['e_inf_all'] * 1e6,
                                p_avg_busy_mW=e['p_avg_busy'] * 1e3, p_avg_gated_mW=e['p_avg_gated'] * 1e3,
                                p_avg_scaled_mW=e['p_avg_scaled'] * 1e3, e_per_mac_nJ=e['e_per_mac'] * 1e9,
                                val_f1=m['val_f1'], test_f1=m['test_f1'], live_2of3_detected=m['live_2of3_detected'],
                                live_2of3_other=m['live_2of3_other']))
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    models = model_rows()
    designs = design_rows(models)
    write_csv(RESULTS / 'S5/models.csv', list(models.values()))
    write_csv(RESULTS / 'S5/designs.csv', designs)
    print(f'{len(models)} models, {len(designs)} measured model/variant pairs -> {RESULTS / "S5"}')


if __name__ == '__main__':
    main()
