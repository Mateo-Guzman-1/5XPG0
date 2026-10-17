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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('part', choices=['s1'])
    a = p.parse_args()
    ps.apply()
    {'s1': s1}[a.part]()


if __name__ == '__main__':
    main()
