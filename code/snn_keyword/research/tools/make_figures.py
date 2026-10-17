"""Report figures in the journal style of plot_style.py (PDF for LaTeX + PNG).

    python make_figures.py s1        figures of S1 (size with energy) -> research/results/S1/

Every figure reads only the CSV/JSON files written by the analysis scripts.
"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plot_style as ps

ROOT = Path(__file__).resolve().parent.parent
SIZES = [8, 16, 32, 48, 64, 80, 95]
FABRIC_LABEL = {'base': 'RV32IM', 'kdot': 'RV32IM + kdot'}


def read_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def column(rows, key, **match):
    sel = [r for r in rows if all(r[k] == v for k, v in match.items())]
    sel.sort(key=lambda r: r['hidden'])
    return (np.array([r['hidden'] for r in sel]), np.array([r[key] for r in sel]),
            np.array([r.get(key + '_sd', 0.0) for r in sel]))


def size_axis(ax):
    ax.set_xlabel('Hidden neurons $H$')
    ax.set_xticks(SIZES)
    ax.set_xlim(0, 100)


def fig_size_energy(summary, out):
    """(a) energy per inference vs H for both fabrics; (b) quality vs H."""
    fig, (a, b) = plt.subplots(1, 2, figsize=ps.FULL)
    for i, fabric in enumerate(('base', 'kdot')):
        h, e, sd = column(summary, 'e_inf_uJ', fabric=fabric)
        a.errorbar(h, e, yerr=sd, label=FABRIC_LABEL[fabric], **ps.series_style(i))
    a.set_yscale('log')
    a.set_ylabel(r'Energy per inference ($\mu$J)')
    size_axis(a)
    a.legend(loc='lower right')
    ps.panel_label(a, '(a)')
    for i, (key, label) in enumerate((('test_f1', 'Test F1 (clips)'),
                                      ('live_2of3_detected', 'Live detection, 2 of 3'))):
        h, v, sd = column(summary, key, fabric='kdot')
        b.errorbar(h, v, yerr=sd, label=label, **ps.series_style(i, filled=i == 0))
    b.set_ylabel('F1 score / detection rate')
    b.set_ylim(0.4, 0.9)
    size_axis(b)
    b.legend(loc='lower right')
    ps.panel_label(b, '(b)')
    ps.save(fig, out / 'fig_size_energy')
    plt.close(fig)


def fig_size_power(summary, out):
    """Average PL power at one window per 250 ms under the three operating scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=ps.FULL, sharey=True)
    scenarios = (('p_avg_busy_mW', 'Busy-polling (deployed)'), ('p_avg_gated_mW', 'Clock-gated idle'),
                 ('p_avg_scaled_mW', 'Clock scaled to deadline'))
    for ax, fabric, label in zip(axes, ('base', 'kdot'), ('(a)', '(b)')):
        for i, (key, name) in enumerate(scenarios):
            h, v, sd = column(summary, key, fabric=fabric)
            ax.errorbar(h, v, yerr=sd, label=name, **ps.series_style(i))
        size_axis(ax)
        ps.panel_label(ax, f'{label} {FABRIC_LABEL[fabric]}')
    axes[0].set_ylabel('Average PL power (mW)')
    axes[0].set_ylim(0, 440)   # room above the busy-polling line for the panel labels
    axes[0].legend(loc='center right', bbox_to_anchor=(1.0, 0.62))
    ps.save(fig, out / 'fig_size_power')
    plt.close(fig)


def s1():
    out = ROOT / 'results/S1'
    summary = read_csv(out / 'energy_summary.csv')
    fig_size_energy(summary, out)
    fig_size_power(summary, out)
    print('S1 figures written to', out)


def stats(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


BITS = [2, 3, 4, 6, 8, 16]


def bits_axis(ax):
    ax.set_xscale('log', base=2)
    ax.set_xticks(BITS)
    ax.set_xticklabels([str(b) for b in BITS])
    ax.minorticks_off()
    ax.set_xlim(1.8, 18)
    ax.set_xlabel('Bits per input weight $n$')


def fig_precision_quality(models, mark, out):
    """(a) validation F1 vs bits (QAT and post-training rounding); (b) test F1 change from 8 bits,
    ours vs Mark's network."""
    fig, (a, b) = plt.subplots(1, 2, figsize=ps.FULL)
    series = [(16, 'qat', '$H$ = 16, QAT'), (48, 'qat', '$H$ = 48, QAT'),
              (16, 'ptq', '$H$ = 16, rounding only'), (48, 'ptq', '$H$ = 48, rounding only')]
    for i, (h, method, label) in enumerate(series):
        pts = []
        for n in BITS:
            sel = [r for r in models if r['hidden'] == h and r['bits'] == n and
                   (r['exp'] == 'E2' if n == 16 else r['method'] == method)]
            if sel:
                pts.append((n, *stats(sel, 'val_f1')))
        if pts:
            n, m, sd = map(np.array, zip(*pts))
            style = ps.series_style(i % 2, filled=method == 'qat')
            style['ls'] = '-' if method == 'qat' else ':'
            a.errorbar(n, m, yerr=sd, label=label, **style)
    bits_axis(a)
    a.set_ylabel('Validation F1 (selection metric)')
    a.legend(loc='lower right', ncol=1)
    ps.panel_label(a, '(a)')
    # (b) change in test F1 relative to each network's own 8-bit model (test sets differ). The 2-bit
    # collapse is visible in (a); (b) zooms in on 3-8 bits and gives the 2-bit values as a note.
    note = []
    for i, (label, short, rows) in enumerate([
            ('This work, $H$ = 16, QAT', '$H$ = 16', [r for r in models if r['hidden'] == 16 and r['method'] == 'qat' and r['fan_in'] == 768]),
            ('This work, $H$ = 48, QAT', '$H$ = 48', [r for r in models if r['hidden'] == 48 and r['method'] == 'qat' and r['fan_in'] == 768]),
            ("Mark's 128-48-2 network, QAT", 'Mark', [r for r in mark if r['method'] == 'qat'])]):
        ref = [r for r in rows if r['bits'] == 8]
        if not ref:
            continue
        ref_mean = stats(ref, 'test_f1')[0]
        pts = [(n, *stats([r for r in rows if r['bits'] == n], 'test_f1')) for n in [3, 4, 5, 6, 8]
               if any(r['bits'] == n for r in rows)]
        n, m, sd = map(np.array, zip(*pts))
        b.errorbar(n, m - ref_mean, yerr=sd, label=label, **ps.series_style(i, filled=i < 2))
        two = [r for r in rows if r['bits'] == 2]
        if two:
            note.append(f'{short} {stats(two, "test_f1")[0] - ref_mean:+.2f}'.replace('-', '$-$'))
    b.axhline(0, color=ps.GREY, lw=0.6, ls='--', zorder=0)
    bits_axis(b)
    b.set_xlim(2.7, 9)
    b.set_xticks([3, 4, 5, 6, 8])
    b.set_xticklabels(['3', '4', '5', '6', '8'])
    b.set_ylabel('Test F1 minus 8-bit test F1')
    b.set_ylim(-0.16, 0.06)   # room for the legend below the data
    b.legend(loc='lower right')
    if note:   # the 2-bit collapse goes into the caption
        print('2-bit test F1 change (for the caption):', ', '.join(note))
    ps.panel_label(b, '(b)')
    ps.save(fig, out / 'fig_precision_quality')
    plt.close(fig)


def fig_precision_cost(models, designs, out):
    """(a) model memory vs bits (packed storage) with the 144 KiB limit; (b) RV32IM energy per
    inference at H = 48 for the plain int16 loop and the packed loop at 16/8/4/2 bits."""
    fig, (a, b) = plt.subplots(1, 2, figsize=ps.FULL)
    for i, h in enumerate((16, 48, 95)):
        n = np.array(BITS)
        rest = 4 * h + 4 * h + 8
        a.plot(n, (h * 768 * n / 8 + rest) / 1024, label=f'$H$ = {h}, packed', **ps.series_style(i))
    a.plot([16], [(95 * 768 * 2 + 95 * 8 + 8) / 1024], ls='none', marker='x', color='black', label='Deployed ($H$ = 95, int16)')
    ps.limit_line(a, 144, '144 KiB model memory', where=0.97)
    a.set_yscale('log')
    a.set_ylabel('Model memory (KiB)')
    bits_axis(a)
    a.legend(loc='lower right')
    ps.panel_label(a, '(a)')
    labels, means, sds = [], [], []
    for variant, bits, label in [('base', 16, 'int16\nplain'), ('packed', 16, '16\npacked'), ('packed', 8, '8\npacked'),
                                 ('packed', 4, '4\npacked'), ('packed', 2, '2\npacked')]:
        sel = [d for d in designs if d['hidden'] == 48 and d['variant'] == variant and d['bits'] == bits
               and d['method'] == 'qat' and d['fan_in'] == 768]
        if sel:
            m, sd = stats(sel, 'e_inf_uJ')
            labels.append(label); means.append(m / 1e3); sds.append(sd / 1e3)
    x = np.arange(len(labels))
    b.bar(x, means, yerr=sds, width=0.6, color='white', edgecolor='black', linewidth=0.6,
          error_kw=dict(lw=0.6, capsize=2))
    b.set_xticks(x)
    b.set_xticklabels(labels)
    b.set_xlabel('Input-layer loop and bits per weight ($H$ = 48, RV32IM)')
    b.set_ylabel('Energy per inference (mJ)')
    b.set_ylim(0, max(means) * 1.15 if means else 1)
    ps.panel_label(b, '(b)')
    ps.save(fig, out / 'fig_precision_cost')
    plt.close(fig)


VARIANT_LABEL = {'base': 'RV32IM', 'lut': 'RV32IM + LIF table', 'kdot': 'RV32IM + kdot',
                 'kdot_lut': 'RV32IM + kdot + LIF table', 'sparse': 'RV32IM, sparse loop', 'packed': 'RV32IM, packed loop'}
FIXED_CYCLES = {'base': 7900, 'kdot': 7950}   # E2 intercepts of cycles vs H (request handling, readout)


def mean_by(designs, key, **match):
    """{hidden: (mean, sd)} of `key` over the seeds of the matching designs."""
    groups = {}
    for d in designs:
        if all(d[k] == v for k, v in match.items()):
            groups.setdefault(int(d['hidden']), []).append(d)
    return {h: stats(rows, key) for h, rows in sorted(groups.items())}


def fig_neuron_update(designs, out):
    """(a) cycles per inference vs H for the four dense firmware variants; (b) where the cycles go."""
    fig, (a, b) = plt.subplots(1, 2, figsize=ps.FULL)
    dense = [d for d in designs if d['fan_in'] == 768 and d['bits'] == 16 and d['exp'] == 'E2']
    for i, variant in enumerate(('base', 'lut', 'kdot', 'kdot_lut')):
        pts = mean_by(dense, 'cycles_mean', variant=variant)
        if pts:
            h = np.array(list(pts)); m, sd = map(np.array, zip(*pts.values()))
            a.errorbar(h, m, yerr=sd, label=VARIANT_LABEL[variant], **ps.series_style(i, filled=variant in ('base', 'kdot')))
    a.set_yscale('log')
    a.set_ylim(1.5e3, 2e7)   # room below the data for the legend
    a.set_ylabel('Cycles per inference (mean)')
    size_axis(a)
    a.legend(loc='lower right', fontsize=7)
    ps.panel_label(a, '(a)')
    # (b) Split of the cycles: neuron update (removed by the exact table) vs the rest.
    labels, lif, rest, fixed = [], [], [], []
    for fabric, plain, table in (('base', 'base', 'lut'), ('kdot', 'kdot', 'kdot_lut')):
        c_plain, c_table = mean_by(dense, 'cycles_mean', variant=plain), mean_by(dense, 'cycles_mean', variant=table)
        for h in sorted(set(c_plain) & set(c_table)):
            total = c_plain[h][0]
            labels.append(f'{h}\n{"RV32IM" if fabric == "base" else "kdot"}')
            lif.append(100 * (total - c_table[h][0]) / total)
            fixed.append(100 * FIXED_CYCLES[fabric] / total)
            rest.append(100 - lif[-1] - fixed[-1])
    x = np.arange(len(labels))
    b.bar(x, rest, width=0.6, color='white', edgecolor='black', lw=0.6, label='Input layer (synapses) + output layer')
    b.bar(x, lif, width=0.6, bottom=rest, color='#bdbdbd', edgecolor='black', lw=0.6, label='12-step neuron update')
    b.bar(x, fixed, width=0.6, bottom=np.array(rest) + np.array(lif), color='black', edgecolor='black', lw=0.6,
          label='Fixed (request, readout)')
    b.set_xticks(x)
    b.set_xticklabels(labels, fontsize=7)
    b.set_xlabel('Hidden neurons $H$ and fabric')
    b.set_ylabel('Share of the cycles (%)')
    b.set_ylim(0, 135)
    b.set_yticks([0, 20, 40, 60, 80, 100])
    b.legend(loc='upper right', fontsize=7, ncol=1)
    ps.panel_label(b, '(b)')
    ps.save(fig, out / 'fig_neuron_update')
    plt.close(fig)


def fig_pruning(models, designs, out):
    """(a) validation F1 vs non-zero input synapses: small dense networks (E2) vs pruned large ones
    (S3); (b) RV32IM energy per inference vs non-zero synapses, dense vs sparse loop, with kdot."""
    fig, (a, b) = plt.subplots(1, 2, figsize=ps.FULL)
    families = [('Dense, $H$ = 8-95 (E2)', lambda r: r['exp'] == 'E2' and r['bits'] == 16),
                ('Pruned, $H$ = 48', lambda r: r['exp'] == 'S3' and r['hidden'] == 48 and r['bits'] == 16),
                ('Pruned, $H$ = 95', lambda r: r['exp'] == 'S3' and r['hidden'] == 95 and r['bits'] == 16)]
    for i, (label, sel) in enumerate(families):
        groups = {}
        for r in models:
            if sel(r):
                groups.setdefault(int(r['hidden'] * (r['fan_in'])), []).append(r)
        if groups:
            n = np.array(sorted(groups)); m, sd = map(np.array, zip(*[stats(groups[k], 'val_f1') for k in n]))
            a.errorbar(n, m, yerr=sd, label=label, **ps.series_style(i, filled=i == 0))
    a.set_xscale('log')
    a.set_xlabel('Non-zero input synapses $H\\cdot K$')
    a.set_ylabel('Validation F1 (selection metric)')
    a.legend(loc='lower right')
    ps.panel_label(a, '(a)')
    for i, (label, sel) in enumerate([
            ('Dense loop, RV32IM (E2)', lambda d: d['exp'] == 'E2' and d['variant'] == 'base'),
            ('Sparse loop, RV32IM (pruned)', lambda d: d['exp'] == 'S3' and d['variant'] == 'sparse'),
            ('kdot, dense (E2)', lambda d: d['exp'] == 'E2' and d['variant'] == 'kdot')]):
        groups = {}
        for d in designs:
            if sel(d) and d['bits'] == 16:
                groups.setdefault(int(d['hidden'] * d['fan_in']), []).append(d)
        if groups:
            n = np.array(sorted(groups)); m, sd = map(np.array, zip(*[stats(groups[k], 'e_inf_uJ') for k in n]))
            b.errorbar(n, m, yerr=sd, label=label, ls='none' if i == 1 else ps.LINES[i],
                       **{k: v for k, v in ps.series_style(i).items() if k != 'ls'})
    b.set_xscale('log'); b.set_yscale('log')
    ps.log_grid(b)
    b.set_xlabel('Non-zero input synapses $H\\cdot K$')
    b.set_ylabel(r'Energy per inference ($\mu$J)')
    b.legend(loc='lower right', fontsize=7)
    ps.panel_label(b, '(b)')
    ps.save(fig, out / 'fig_pruning')
    plt.close(fig)


def s3():
    out = ROOT / 'results/S3'
    fig_pruning(read_csv(ROOT / 'results/S5/models.csv'), read_csv(ROOT / 'results/S5/designs.csv'), out)
    print('S3 figures written to', out)


def s4():
    out = ROOT / 'results/S4'
    fig_neuron_update(read_csv(ROOT / 'results/S5/designs.csv'), out)
    print('S4 figures written to', out)


def s2():
    out = ROOT / 'results/S2'
    models = read_csv(ROOT / 'results/S5/models.csv')
    designs = read_csv(ROOT / 'results/S5/designs.csv')
    mark = read_csv(out / 'mark_comparison.csv')
    fig_precision_quality(models, mark, out)
    fig_precision_cost(models, designs, out)
    print('S2 figures written to', out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('part', choices=['s1', 's2', 's3', 's4'])
    a = p.parse_args()
    ps.apply()
    {'s1': s1, 's2': s2, 's3': s3, 's4': s4}[a.part]()


if __name__ == '__main__':
    main()
