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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('table', choices=['power', 'worked', 'size', 'precision'])
    p.add_argument('--latex', action='store_true')
    p.add_argument('--out', type=Path, help='write the table to this file (UTF-8) instead of printing it')
    a = p.parse_args()
    text = {'power': table_power, 'worked': table_worked, 'size': table_size, 'precision': table_precision}[a.table](a.latex)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + '\n', encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
