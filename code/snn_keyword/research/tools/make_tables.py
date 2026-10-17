"""Report tables, generated from the result files (Markdown for research/ENERGY.md, LaTeX for the report).

    python make_tables.py power      raw Vivado power of both fabrics (T1)
    python make_tables.py worked     worked energy calculation for the deployed model (T2)
    python make_tables.py size       energy and quality per network size (S1)

Add --latex to get a booktabs table instead of Markdown.
"""
import argparse
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from energy_model import load_power, energy, F_CLK, T_HOP, MARGIN

ROOT = Path(__file__).resolve().parent.parent
POWER = ROOT / 'results/S0/power.json'
FABRIC = {'base': 'RV32IM', 'kdot': 'RV32IM + kdot'}


def fmt(x, digits):
    """Round half up (as in the earlier reports), after removing floating-point noise."""
    if isinstance(x, str):
        return x
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(f'{x:.9f}').quantize(q, rounding=ROUND_HALF_UP))


def render(header, rows, latex, align=None, head=None):
    """Markdown or booktabs table; `head` replaces the LaTeX header row (e.g. a two-level header)."""
    if latex:
        align = align or 'l' + 'r' * (len(header) - 1)
        lines = [r'\begin{tabular}{' + align + '}', r'\toprule', *(head or [' & '.join(header) + r' \\']), r'\midrule']
        for r in rows:
            if r is None:
                lines.append(r'\midrule')
            else:
                lines.append(' & '.join(str(c) for c in r) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        return '\n'.join(lines)
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows if r is not None]
    return '\n'.join(lines)


def table_power(latex):
    raw = json.loads(POWER.read_text())
    split = load_power()
    w = lambda x: fmt(x, 3)
    def comp(fabric, name):
        for r in raw[fabric]['power']['on_chip']:
            if r['On-Chip'] == name:
                return w(r['Power (W)'])
        return '--'
    def inst(fabric, name):
        for r in raw[fabric]['power']['hierarchy']:
            if r['Name'] == name:
                return w(r['Power (W)'])
        return '--'
    rows = []
    for label, get in [('Total on-chip power', lambda f: raw[f]['power']['summary']['Total On-Chip Power (W)']),
                       ('Dynamic power', lambda f: raw[f]['power']['summary']['Dynamic (W)']),
                       ('Device static power', lambda f: raw[f]['power']['summary']['Device Static (W)'])]:
        rows.append([label] + [w(get(f)) for f in ('base', 'kdot')])
    rows.append(None)
    for name in ('PS7', 'Clocks', 'Slice Logic', 'Signals', 'Block RAM', 'DSPs', 'I/O'):
        rows.append([f'Dynamic: {name}'] + [comp(f, name) for f in ('base', 'kdot')])
    rows.append(None)
    for name, label in (('u_soc', 'Dynamic: core u_soc (CPU, kdot, BRAM)'), ('u_cpu', '  of which PicoRV32 u_cpu'),
                        ('u_kdot', '  of which kdot u_kdot'), ('u_ps_if', 'Dynamic: AXI interface u_ps_if'),
                        ('pc0', 'Dynamic: AXI protocol converter pc0')):
        rows.append([label] + [inst(f, name) for f in ('base', 'kdot')])
    rows.append(None)
    rows.append(['Static, PL rails (Vccint, Vccaux, Vccbram, Vccadc, Vcco)'] + [w(split[f]['pl_static']) for f in ('base', 'kdot')])
    rows.append(['Static, PS rails (Vccp*, DDR, MIO)'] + [w(split[f]['ps_static']) for f in ('base', 'kdot')])
    rows.append(None)
    rows.append(['Derived: PL dynamic P_PL,dyn = dynamic - PS7'] + [w(split[f]['pl_dyn']) for f in ('base', 'kdot')])
    rows.append(['Derived: core P_core,dyn (u_soc)'] + [w(split[f]['core_dyn']) for f in ('base', 'kdot')])
    rows.append(['Derived: interface P_infra,dyn = P_PL,dyn - P_core,dyn'] + [w(split[f]['infra_dyn']) for f in ('base', 'kdot')])
    header = ['Quantity (W)', FABRIC['base'], FABRIC['kdot']]
    if latex:
        nice = {'Dynamic: core u_soc (CPU, kdot, BRAM)': r'Dynamic: core \texttt{u\_soc} (CPU, \texttt{kdot}, BRAM)',
                '  of which PicoRV32 u_cpu': r'\quad of which PicoRV32 \texttt{u\_cpu}',
                '  of which kdot u_kdot': r'\quad of which \texttt{kdot} \texttt{u\_kdot}',
                'Dynamic: AXI interface u_ps_if': r'Dynamic: AXI interface \texttt{u\_ps\_if}',
                'Dynamic: AXI protocol converter pc0': r'Dynamic: AXI protocol converter \texttt{pc0}',
                'Static, PS rails (Vccp*, DDR, MIO)': r'Static, PS rails (Vccp$^\ast$, DDR, MIO)',
                'Derived: PL dynamic P_PL,dyn = dynamic - PS7': r'Derived: $P_\mathrm{PL,dyn}$ = dynamic $-$ PS7',
                'Derived: core P_core,dyn (u_soc)': r'Derived: $P_\mathrm{core}$ (\texttt{u\_soc})',
                'Derived: interface P_infra,dyn = P_PL,dyn - P_core,dyn':
                    r'Derived: $P_\mathrm{infra} = P_\mathrm{PL,dyn} - P_\mathrm{core}$'}
        rows = [[nice.get(r[0], r[0]), *r[1:]] if r else None for r in rows]
        header = ['Quantity (W)', 'RV32IM', r'RV32IM + \texttt{kdot}']
    return render(header, rows, latex)


def table_worked(latex):
    with open(ROOT / 'results/E2/models.csv', newline='') as f:
        m = next(r for r in csv.DictReader(f) if r['name'] == 'h95_seed2')   # = deploy/model.npz
    split = load_power()
    rows = []
    for fabric in ('base', 'kdot'):
        p = split[fabric]
        c_mean, c_max = float(m[f'{fabric}_cycles_mean']), float(m[f'{fabric}_cycles_max'])
        e = energy(c_mean, c_max, p, macs=768 * 95 + 2 * 95)
        rows += [[FABRIC[fabric], 'Mean / max cycles (board)', f'{c_mean:,.0f} / {c_max:,.0f}'],
                 ['', f'E_inf = P_PL,dyn x C_mean / f = {p["pl_dyn"]:.3f} W x {c_mean / F_CLK * 1e3:.3f} ms',
                  f'{e["e_inf"] * 1e6:,.1f} uJ'],
                 ['', f'E_inf,all = (P_PL,dyn + P_PL,stat) x C_mean / f', f'{e["e_inf_all"] * 1e6:,.1f} uJ'],
                 ['', f'(a) busy-poll: (P_PL,dyn + P_PL,stat) x T = {p["pl_dyn"] + p["pl_static"]:.3f} W x {T_HOP} s',
                  f'{e["e_win_busy"] * 1e3:.2f} mJ = {e["p_avg_busy"] * 1e3:.1f} mW'],
                 ['', f'(b) gated: (P_PL,stat + P_infra,dyn) x T + P_core,dyn x C_mean / f',
                  f'{e["e_win_gated"] * 1e3:.2f} mJ = {e["p_avg_gated"] * 1e3:.1f} mW'],
                 ['', f'(c) scaled: f_min = {MARGIN} C_max / T = {e["f_min"] / 1e6:.2f} MHz; P_PL,stat x T + P_PL,dyn x {MARGIN} C_max / f',
                  f'{e["e_win_scaled"] * 1e3:.2f} mJ = {e["p_avg_scaled"] * 1e3:.1f} mW'],
                 ['', 'Energy per synaptic multiply-add (73,150 per inference)', f'{e["e_per_mac"] * 1e9:.1f} nJ'],
                 None]
    return render(['Fabric', 'Step', 'Value'], rows[:-1], latex, 'llr')


def table_size(latex):
    with open(ROOT / 'results/S1/energy_summary.csv', newline='') as f:
        s = list(csv.DictReader(f))
    rows = []
    for h in sorted({int(float(r['hidden'])) for r in s}):
        b = next(r for r in s if int(float(r['hidden'])) == h and r['fabric'] == 'base')
        k = next(r for r in s if int(float(r['hidden'])) == h and r['fabric'] == 'kdot')
        pm = r' $\pm$ ' if latex else ' ± '
        rows.append([h, fmt(float(b['model_bytes']) / 1024, 1),
                     fmt(float(b['t_max_ms']), 2), fmt(float(b['e_inf_uJ']) / 1e3, 2),
                     fmt(float(k['t_max_ms']), 2), fmt(float(k['e_inf_uJ']), 0),
                     fmt(float(b['test_f1']), 3) + pm + fmt(float(b['test_f1_sd']), 3),
                     fmt(float(b['live_2of3_detected']) * 100, 1)])
    header = ['H', 'Model (KiB)', 'RV32IM t_max (ms)', 'RV32IM E_inf (mJ)', 'kdot t_max (ms)',
              'kdot E_inf (uJ)', 'Test F1', 'Live 2 of 3 (%)']
    head = [r' & & \multicolumn{2}{c}{RV32IM} & \multicolumn{2}{c}{RV32IM + \texttt{kdot}} & & \\',
            r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
            r'$H$ & Model (KiB) & $t_\mathrm{max}$ (ms) & $E_\mathrm{inf}$ (mJ) & $t_\mathrm{max}$ (ms) & '
            r'$E_\mathrm{inf}$ ($\mu$J) & Test F1 & Live 2 of 3 (\%) \\']
    return render(header, rows, latex, head=head)


def _rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _ms(rows, key):
    import numpy as np
    v = np.array([float(r[key]) for r in rows])
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def table_precision(latex):
    """S2: quality, zeros, memory and RV32IM energy per bit width."""
    models = _rows(ROOT / 'results/S5/models.csv')
    designs = _rows(ROOT / 'results/S5/designs.csv')
    pm = r' $\pm$ ' if latex else ' ± '
    rows = []
    for n in (16, 8, 6, 4, 3, 2):
        def sel(h, method):
            return [r for r in models if int(r['hidden']) == h and int(r['bits']) == n and int(r['fan_in']) == 768
                    and (r['exp'] == 'E2' if n == 16 else r['method'] == method)]
        cells = [n]
        for h, method in ((16, 'qat'), (48, 'qat'), (48, 'ptq')):
            s = sel(h, method)
            if s:
                m, sd = _ms(s, 'val_f1')
                cells.append(fmt(m, 3) + pm + fmt(sd, 3))
            else:
                cells.append('n/a' if n != 16 else '--')
        s = sel(48, 'qat')
        cells.append(fmt(100 * _ms(s, 'w1_zero_fraction')[0], 0))
        cells.append(fmt((48 * 768 * n / 8 + 4 * 48 + 4 * 48 + 8) / 1024, 1))
        d = [r for r in designs if int(r['hidden']) == 48 and int(r['bits']) == n and r['variant'] == 'packed'
             and int(r['fan_in']) == 768 and r['method'] == 'qat']
        cells.append(fmt(_ms(d, 'e_inf_uJ')[0] / 1e3, 1) if d else '--')
        rows.append(cells)
    header = ['Bits', 'Val F1, H=16', 'Val F1, H=48', 'Val F1, H=48, rounding only', 'Zero weights H=48 (%)',
              'Memory H=48 (KiB)', 'E_inf packed RV32IM H=48 (mJ)']
    if latex:
        header = ['$n$', r'$H=16$', r'$H=48$', r'$H=48$, rounding only', r'Zeros (\%)', 'Memory (KiB)',
                  r'$E_\mathrm{inf}$ (mJ)']
    return render(header, rows, latex)


FIRMWARE = {'base': 'RV32IM', 'lut': 'RV32IM + table', 'packed': 'RV32IM packed', 'sparse': 'RV32IM sparse',
            'kdot': 'kdot', 'kdot_lut': 'kdot + table', 'ksdot_lut': 'sparse kdot + table',
            'kdot_lut_on_ksdot': 'kdot + table (S6 fabric)'}


def describe(d, latex=False):
    h, bits, k = int(float(d['hidden'])), int(float(d['bits'])), int(float(d['fan_in']))
    parts = [f'H={h}' if not latex else f'$H={h}$', f'{bits} bit']
    if k < 768:
        parts.append(f'K={k}' if not latex else f'$K={k}$')
    fw = FIRMWARE[d['variant']]
    return ', '.join(parts) + '; ' + (fw.replace('kdot', r'\texttt{kdot}') if latex else fw)


def table_designs(latex):
    """S5: the Pareto-front designs and reference points."""
    s = {d['design']: d for d in _rows(ROOT / 'results/S5/design_summary.csv')}
    choice = json.loads((ROOT / 'results/S5/choice.json').read_text())
    keep = list(dict.fromkeys(choice['pareto_val_f1_vs_e_inf'] + [x for x in (
        'h95/kdot', 'h95/base', 'h48/kdot', 'h48/base', 'h16/kdot_lut', 'h48_b4/packed', 'h48_k64/sparse',
        'h48_b4_k256/sparse', choice['primary']['choice'], choice['sensitivity_pooled_sd']['choice']) if x in s]))
    keep.sort(key=lambda k: float(s[k]['e_inf_uJ']))
    pm = r' $\pm$ ' if latex else ' ± '
    rows = []
    for k in keep:
        d = s[k]
        mark = ('*' if k == choice['primary']['choice'] else '') + ('P' if k in choice['pareto_val_f1_vs_e_inf'] else '')
        rows.append([describe(d, latex) + (f' ({mark})' if mark else ''),
                     fmt(float(d['val_f1']), 3) + pm + fmt(float(d['val_f1_sd']), 3), fmt(float(d['test_f1']), 3),
                     fmt(100 * float(d['live_2of3_detected']), 1), fmt(float(d['memory']) / 1024, 1),
                     fmt(float(d['t_max_ms']), 2), fmt(float(d['e_inf_uJ']), 0), fmt(float(d['p_avg_gated_mW']), 1)])
    header = ['Design (* chosen, P Pareto)', 'Val F1', 'Test F1', 'Live 2 of 3 (%)', 'Memory (KiB)', 't_max (ms)',
              'E_inf (uJ)', 'P gated (mW)']
    if latex:
        header = [r'Design', 'Val F1', 'Test F1', r'Live (\%)', r'Mem.\ (KiB)', r'$t_\mathrm{max}$ (ms)',
                  r'$E_\mathrm{inf}$ ($\mu$J)', r'$P_\mathrm{gated}$ (mW)']
    return render(header, rows, latex, 'l' + 'r' * 7)


def table_sparse(latex):
    """S6: sparse kdot per pruned configuration: measured vs predicted cycles (H6), energy, memory, quality."""
    designs = _rows(ROOT / 'results/S5/designs.csv')
    models = _rows(ROOT / 'results/S5/models.csv')
    ref = {h: _ms([d for d in designs if int(d['hidden']) == h and d['variant'] == 'kdot_lut' and d['exp'] == 'E2'],
                  'cycles_mean')[0] for h in (48, 95)}
    pm = r' $\pm$ ' if latex else ' ± '
    rows = []
    configs = []
    for d in designs:
        if d['variant'] == 'ksdot_lut' and d['config'] not in configs:
            configs.append(d['config'])
    configs.sort(key=lambda c: (int(c.split('_')[0][1:]), 'b' in c, int(c.split('_k')[1])))
    rows.append(['deployed: H=48, 8 bit, dense; kdot + table' if not latex else r'deployed: $H=48$, 8 bit, dense; \texttt{kdot} + table',
                 fmt(ref[48], 0), '--', fmt(165.4, 0), fmt(74120 / 1024, 1), '0.745' + pm + '0.005', '0.789', '65.6'])
    for c in configs:
        d = [x for x in designs if x['config'] == c and x['variant'] == 'ksdot_lut']
        m = [x for x in models if x['config'] == c]
        h, k, bits = int(d[0]['hidden']), int(d[0]['fan_in']), int(d[0]['bits'])
        meas = _ms(d, 'cycles_mean')[0]
        pred = ref[h] - h * (576 - k) + 200
        v, sd = _ms(m, 'val_f1')
        label = f'H={h}, {bits} bit, K={k}' if not latex else f'$H={h}$, {bits} bit, $K={k}$'
        rows.append([label, fmt(meas, 0), fmt(meas - pred, 0), fmt(_ms(d, 'e_inf_uJ')[0], 0),
                     fmt(float(d[0]['memory']) / 1024, 1), fmt(v, 3) + pm + fmt(sd, 3),
                     fmt(_ms(m, 'test_f1')[0], 3), fmt(100 * _ms(m, 'live_2of3_detected')[0], 1)])
    header = ['Model (sparse kdot + table)', 'Cycles', 'Measured - predicted', 'E_inf (uJ)', 'Memory (KiB)', 'Val F1',
              'Test F1', 'Live 2 of 3 (%)']
    if latex:
        header = ['Model', 'Cycles', r'$\Delta$ H6', r'$E_\mathrm{inf}$ ($\mu$J)', r'Mem.\ (KiB)', 'Val F1',
                  'Test F1', r'Live (\%)']
    return render(header, rows, latex)


def table_choice(latex):
    """T4: the chosen design (mean over its seeds) next to the deployed model (h95_seed2 on kdot)."""
    choice = json.loads((ROOT / 'results/S5/choice.json').read_text())['primary']['choice']
    config, variant = choice.split('/')
    models = _rows(ROOT / 'results/S5/models.csv')
    designs = _rows(ROOT / 'results/S5/designs.csv')
    chosen_m = [r for r in models if r['config'] == config]
    chosen_d = [r for r in designs if r['config'] == config and r['variant'] == variant]
    dep_m = [r for r in models if r['name'] == 'h95_seed2']
    dep_d = [r for r in designs if r['name'] == 'h95_seed2' and r['variant'] == 'kdot']
    pm = r' $\pm$ ' if latex else ' ± '
    pct = r'\%' if latex else '%'
    lines = [('Test precision', 'test_precision', 100, 1, pct), ('Test recall', 'test_recall', 100, 1, pct),
             ('Test F1', 'test_f1', 1, 3, ''), ('Live "Sheila" detected, 2 of 3', 'live_2of3_detected', 100, 1, pct),
             ('Live other words accepted, 2 of 3', 'live_2of3_other', 100, 2, pct),
             ('Near miss "shee" + other word', 'hardneg_ending', 100, 1, pct),
             ('Near miss other onset + "eila"', 'hardneg_onset', 100, 1, pct),
             ('Synthesized she- words, 2 of 3', 'tts_2of3_she- words', 100, 1, pct)]
    rows = []
    for label, key, scale, digits, unit in lines:
        m, sd = _ms(chosen_m, key)
        rows.append([label, fmt(scale * m, digits) + pm + fmt(scale * sd, digits) + unit,
                     fmt(scale * float(dep_m[0][key]), digits) + unit])
    rows.append(None)
    for label, key, scale, digits, unit in [('Model memory', 'memory', 1 / 1024, 1, ' KiB'),
                                            ('Worst-case latency', 't_max_ms', 1, 2, ' ms'),
                                            ('Energy per inference', 'e_inf_uJ', 1, 0, r' $\mu$J' if latex else ' uJ'),
                                            ('Average PL power, busy-polling', 'p_avg_busy_mW', 1, 1, ' mW'),
                                            ('Average PL power, clock-gated', 'p_avg_gated_mW', 1, 1, ' mW'),
                                            ('Average PL power, clock scaled', 'p_avg_scaled_mW', 1, 1, ' mW')]:
        rows.append([label, fmt(scale * _ms(chosen_d, key)[0], digits) + unit,
                     fmt(scale * float(dep_d[0][key]), digits) + unit])
    c0 = chosen_d[0]
    head = [f'Chosen: {describe(c0, latex)} (mean{pm.strip()}sd, 3 seeds)', 'Previous release: H=95, 16 bit; kdot (seed 2)']
    if latex:
        head = [r'Chosen: ' + describe(c0, True), r'Previous: $H=95$; \texttt{kdot}']
    if latex:   # TeX quotes
        tex = {'Live "Sheila" detected, 2 of 3': "Live ``Sheila'' detected, 2 of 3",
               'Near miss "shee" + other word': "Near miss ``shee'' + other word",
               'Near miss other onset + "eila"': "Near miss other onset + ``eila''"}
        rows = [[tex.get(r[0], r[0]), *r[1:]] if r else None for r in rows]
    return render(['Quantity', *head], rows, latex, 'lrr')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('table', choices=['power', 'worked', 'size', 'precision', 'designs', 'choice', 'sparse'])
    p.add_argument('--latex', action='store_true')
    p.add_argument('--out', type=Path, help='write the table to this file (UTF-8) instead of printing it')
    a = p.parse_args()
    text = {'power': table_power, 'worked': table_worked, 'size': table_size, 'precision': table_precision, 'designs': table_designs, 'choice': table_choice, 'sparse': table_sparse}[a.table](a.latex)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + '\n', encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
