"""Tables and figures of the size benchmark (E2) from research/results/E2/.

Reads quality/h<H>.json, board/h<H>_seed<s>.json, training/h<H>_seed<s>_training.json and
pc_timing.json, refuses results whose benchmark fingerprint differs from
benchmark_inputs.json, and writes into the same folder:
  summary.json, summary.csv   one row per size: mean and sample sd over seeds
  models.csv                  one row per trained model
  fig_overview.png            cost as % of each platform limit, and quality, vs size
  fig_cost.png                model memory, RV32IM and kdot inference time vs size
  fig_quality.png             clip metrics, live stream, near misses vs size
  fig_tradeoff.png            test F1 vs inference cycles (both firmware variants)
  fig_delay.png               where the time of one window goes, per size
  fig_network.png, fig_memory_map.png   explanatory figures (E2_network_size_explained.md)
Cycles count only from bit-exact board runs.

    plot_size_sweep.py [--results research/results/E2]
"""
import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

TOOLS = Path(__file__).resolve().parent
KWS = TOOLS.parents[1]   # code/snn_keyword
sys.path.insert(0, str(KWS))
from benchmark_inputs import VECTORS

MODEL_LIMIT = 144 * 1024       # bytes, model section 0x18000-0x3bfff
BUDGET = 25_000_000            # cycles per 250 ms hop at 100 MHz
CLOCK = 100e6
BYTES_PER_NEURON, BYTES_FIXED = 1544, 8
# GCC unrolls the neuron loop of smaller networks and keeps their state in registers
# (code_shape.py -> results/E2/compiled_code.json); from this size on it is a loop.
LOOP_FROM = 32

# Reference palette of the dataviz skill (light surface): ink, chrome, categorical slots 1-4.
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, LIMIT = '#e1e0d9', '#c3c2b7', '#d03b3b'
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MARKERS = ['o', 's', '^', 'D']
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def style():
    # Text is Segoe UI; matplotlib still probes its fallback font for the semibold titles.
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    plt.rcParams.update({
        'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'axes.labelsize': 9,
        'axes.titlesize': 10.5, 'axes.titleweight': 'semibold', 'axes.titlecolor': INK, 'axes.titlelocation': 'left',
        'axes.titlepad': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
        'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False, 'legend.fontsize': 8.5,
        'lines.linewidth': 1.8, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round'})


def mean_sd(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def load(results):
    """Per-model records and per-size summaries; refuses results of other benchmark inputs."""
    fingerprint = json.loads((results / 'benchmark_inputs.json').read_text())['fingerprint']
    records = []
    for qfile in sorted((results / 'quality').glob('h*.json')):
        quality = json.loads(qfile.read_text())
        if quality['benchmark'] != fingerprint:
            raise SystemExit(f'{qfile} was measured on other benchmark inputs')
        keyword = quality['keyword']
        for name, m in quality['models'].items():
            seed = int(name.rsplit('seed', 1)[1])
            h = m['hidden']
            board = json.loads((results / 'board' / f'h{h}_seed{seed}.json').read_text())
            if board['benchmark'] != fingerprint or board['model_sha256'] != m['sha256']:
                raise SystemExit(f'board result of {name} belongs to other inputs or another model file')
            training = json.loads((results / 'training' / f'h{h}_seed{seed}_training.json').read_text())
            t, v = m['test'], m['validation']
            r = {'name': name, 'hidden': h, 'seed': seed, 'parameters': m['parameters'],
                 'model_bytes': board['firmware']['images']['keyword']['model_bytes'],
                 'decision_threshold': m['decision_threshold'], 'stream_threshold': m['stream_threshold'],
                 'training_seconds': training['training_seconds'], 'val_f1': training['validation']['f1'],
                 'val_live_2of3_detected': v['live']['two_of_three'][f'{keyword}_detected'],
                 'test_precision': t['clips']['precision'], 'test_recall': t['clips']['recall'],
                 'test_f1': t['clips']['f1'], 'test_fpr': t['clips']['fpr'],
                 'live_single_detected': t['live']['single_window'][f'{keyword}_detected'],
                 'live_single_other': t['live']['single_window']['other_accepted'],
                 'live_2of3_detected': t['live']['two_of_three'][f'{keyword}_detected'],
                 'live_2of3_other': t['live']['two_of_three']['other_accepted'],
                 'noise_false_detections': t['background_noise']['false_detections'],
                 'hidden_spikes_mean': t['hidden_spikes_mean'], 'bit_exact': True}
            r.update({f'hardneg_{k}': x for k, x in t['augmented_detected']['stream_threshold'].items()})
            r.update({f'tts_2of3_{k}': x for k, x in t['tts']['two_of_three'].items()})
            for variant, run in board['variants'].items():
                r['bit_exact'] &= run['bit_exact'] and run['stream_pass'] is not False
                if run['bit_exact']:
                    s = run['single_window']
                    r[f'{variant}_cycles_mean'], r[f'{variant}_cycles_max'] = s['cycles_mean'], s['cycles_max']
                r[f'{variant}_rtt_mean_ms'] = run['round_trip_ms']['mean']
            records.append(r)
    if not records:
        raise SystemExit(f'No results in {results}')
    sizes = sorted({r['hidden'] for r in records})
    summary = []
    for h in sizes:
        rows = [r for r in records if r['hidden'] == h]
        row = {'hidden': h, 'seeds': len(rows), 'bit_exact': all(r['bit_exact'] for r in rows)}
        for key in rows[0]:
            if key in ('name', 'hidden', 'seed', 'bit_exact'):
                continue
            values = [r[key] for r in rows if key in r]
            if len(values) == len(rows):
                row[key], row[key + '_sd'] = mean_sd(values)
        summary.append(row)
    return keyword, records, summary


def line(points):
    """Least-squares line cycles = slope * H + intercept, and the largest H within the budget."""
    h, c = np.array(points, dtype=float).T
    slope, intercept = np.polyfit(h, c, 1)
    pred = slope * h + intercept
    return {'cycles_per_neuron': round(float(slope), 1), 'fixed_cycles': round(float(intercept), 1),
            'r2': round(float(1 - ((c - pred) ** 2).sum() / ((c - c.mean()) ** 2).sum()), 6),
            'neurons_within_budget': int((BUDGET - intercept) // slope)}


def fits(records):
    """Cycle lines through all bit-exact models and through the sizes compiled as a loop."""
    out = {}
    for variant in ('base', 'kdot'):
        pts = [(r['hidden'], r[f'{variant}_cycles_max']) for r in records if f'{variant}_cycles_max' in r]
        out[variant] = {'all_sizes': line(pts), f'loop_sizes_from_{LOOP_FROM}': line([p for p in pts if p[0] >= LOOP_FROM])}
    h, b = np.array([(r['hidden'], r['model_bytes']) for r in records], dtype=float).T
    out['model_bytes'] = {'formula': f'{BYTES_PER_NEURON} * H + {BYTES_FIXED}',
                          'max_abs_deviation_bytes': int(np.abs(b - (BYTES_PER_NEURON * h + BYTES_FIXED)).max()),
                          'largest_fitting_H': int((MODEL_LIMIT - BYTES_FIXED) // BYTES_PER_NEURON)}
    return out


def efficient_size(summary):
    """Smallest size whose mean validation F1 is within one sd of the best size (validation only)."""
    best = max(summary, key=lambda r: r['val_f1'])
    bar = best['val_f1'] - best['val_f1_sd']
    return {'rule': 'smallest H with mean validation F1 >= best mean - best sd (validation includes hard variants)',
            'best_hidden': best['hidden'], 'best_val_f1': round(best['val_f1'], 4),
            'best_val_f1_sd': round(best['val_f1_sd'], 4), 'threshold': round(bar, 4),
            'efficient_hidden': min(r['hidden'] for r in summary if r['val_f1'] >= bar)}


def band(ax, x, rows, key, color, marker, label, scale=1.0):
    """Mean line with markers and a +-1 sd wash over the seeds."""
    m = np.array([r[key] for r in rows]) * scale
    s = np.array([r[key + '_sd'] for r in rows]) * scale
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.12, linewidth=0)
    ax.plot(x, m, color=color, marker=marker, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
            label=label, zorder=3)
    return m


def limit_line(ax, y, text, horizontal=True, va='bottom'):
    if horizontal:
        ax.axhline(y, color=LIMIT, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(text, (0.01, y), xycoords=('axes fraction', 'data'), xytext=(0, 3 if va == 'bottom' else -3),
                    textcoords='offset points', va=va, color=INK2, fontsize=8)
    else:
        ax.axvline(y, color=LIMIT, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(text, (y, 0.98), xycoords=('data', 'axes fraction'), xytext=(-4, 0), textcoords='offset points',
                    ha='right', va='top', color=INK2, fontsize=8)


def size_axis(ax, sizes):
    ax.set_xticks(sizes)
    ax.set_xlim(0, 100)
    ax.set_xlabel('hidden neurons H')


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords='offset points', va='center', color=INK2, fontsize=8)


def fig_overview(summary, out):
    sizes = [r['hidden'] for r in summary]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
    for (key, scale, label), color, marker in zip(
            [('model_bytes', 100 / MODEL_LIMIT, 'model memory'), ('base_cycles_max', 100 / BUDGET, 'RV32IM inference'),
             ('kdot_cycles_max', 100 / BUDGET, 'kdot inference')], SLOT, MARKERS):
        m = band(a, sizes, summary, key, color, marker, label, scale)
        end_label(a, sizes[-1], m[-1], f'{m[-1]:.1f}%', dy=-9 if key == 'model_bytes' else 0)
    limit_line(a, 100, 'platform limit: 144 KiB model memory; 250 ms = 25 M cycles per window')
    a.set_ylim(0, 112)
    a.set_ylabel('share of the platform limit (%)')
    a.set_title('Cost grows in a straight line; memory hits its limit first')
    a.legend(loc='upper left', bbox_to_anchor=(0, 0.86))
    size_axis(a, sizes)
    for (key, label), color, marker in zip([('test_f1', 'test F1 (clips)'),
                                            ('live_2of3_detected', 'live "Sheila" detected (2 of 3)')], SLOT, MARKERS):
        m = band(b, sizes, summary, key, color, marker, label)
        end_label(b, sizes[-1], m[-1], f'{m[-1]:.3f}' if key == 'test_f1' else f'{100 * m[-1]:.1f}%')
    b.set_ylim(0, 1)
    b.set_ylabel('score (fraction)')
    b.set_title('Quality: flat from 16 neurons on (mean ± 1 sd, 3 seeds)')
    b.legend(loc='lower right')
    size_axis(b, sizes)
    fig.tight_layout()
    fig.savefig(out / 'fig_overview.png', dpi=200)
    plt.close(fig)


def fig_cost(summary, out):
    sizes = [r['hidden'] for r in summary]
    hx = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    a, b, c = axes
    a.plot(hx, (BYTES_PER_NEURON * hx + BYTES_FIXED) / 1024, color=MUTED, linewidth=1, zorder=1)
    m = band(a, sizes, summary, 'model_bytes', SLOT[0], 'o', 'measured', 1 / 1024)
    limit_line(a, MODEL_LIMIT / 1024, 'model memory 144 KiB', va='top')
    end_label(a, sizes[-1], m[-1], f'{m[-1]:.1f} KiB', dy=-10)
    a.annotate('1,544 bytes per neuron + 8', (56, (BYTES_PER_NEURON * 56 + BYTES_FIXED) / 1024), xytext=(8, -12),
               textcoords='offset points', ha='left', va='top', color=INK2, fontsize=8)
    a.set_ylim(0, 160)
    a.set_ylabel('model memory (KiB)')
    a.set_title('Model memory')
    for ax, variant, title, top in ((b, 'base', 'Inference time, plain RV32IM', 260), (c, 'kdot', 'Inference time, kdot', None)):
        ms = band(ax, sizes, summary, f'{variant}_cycles_max', SLOT[0], 'o', 'measured', 1e3 / CLOCK)
        end_label(ax, sizes[-1], ms[-1], f'{ms[-1]:.2f} ms', dy=-10)
        loop = [(h, v) for h, v in zip(sizes, ms) if h >= LOOP_FROM]
        slope = np.polyfit(*np.array(loop).T, 1)[0]
        ax.set_ylabel('worst-case inference time (ms)')
        ax.set_title(title)
        if top:
            limit_line(ax, 250, 'real-time budget: 250 ms per window', va='top')
            ax.set_ylim(0, top)
            ax.annotate(f'+{slope:.2f} ms per neuron', (0.97, 0.06), xycoords='axes fraction', ha='right',
                        color=INK2, fontsize=8)
        else:
            ax.set_ylim(0, ms.max() * 1.25)
            ax.annotate(f'real-time budget 250 ms: {250 / ms.max():.0f}× the slowest size here', (0.01, 0.97),
                        xycoords='axes fraction', va='top', color=INK2, fontsize=8)
            ax.annotate(f'+{1000 * slope:.0f} µs per neuron from {LOOP_FROM} neurons on.\n'
                        f'Below {LOOP_FROM}, the compiler unrolls the\nneuron loop and keeps the state in registers.',
                        (sizes[1], ms[1]), xytext=(0.36, 0.06), textcoords='axes fraction', va='bottom',
                        color=INK2, fontsize=8, arrowprops={'arrowstyle': '-', 'color': MUTED, 'linewidth': 0.8})
    for ax in axes:
        size_axis(ax, sizes)
    fig.tight_layout()
    fig.savefig(out / 'fig_cost.png', dpi=200)
    plt.close(fig)


def fig_quality(summary, keyword, out):
    sizes = [r['hidden'] for r in summary]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    a, b, c, d = axes.flat
    for (key, label), color, marker in zip([('test_f1', 'F1'), ('test_precision', 'precision'),
                                            ('test_recall', 'recall')], SLOT, MARKERS):
        band(a, sizes, summary, key, color, marker, label)
    a.set_title('Test clips (11,005 clips, 212 "Sheila")')
    a.set_ylabel('score')
    for (key, label), color, marker in zip([('live_2of3_detected', '2 of 3 (live demo rule)'),
                                            ('live_single_detected', 'single window')], SLOT, MARKERS):
        band(b, sizes, summary, key, color, marker, label, 100)
    b.set_title('Live stream: "Sheila" detected')
    b.set_ylabel('"Sheila" clips detected (%)')
    for (key, label), color, marker in zip([('live_2of3_other', '2 of 3 (live demo rule)'),
                                            ('live_single_other', 'single window')], SLOT, MARKERS):
        band(c, sizes, summary, key, color, marker, label, 100)
    c.set_title('Live stream: other words wrongly accepted')
    c.set_ylabel('other-word clips accepted (%)')
    near = [('hardneg_sh', '"sh" alone (per window)'), ('hardneg_ending', '"shee" + other word (per window)'),
            ('hardneg_onset', 'other onset + "eila" (per window)'), ('tts_2of3_she- words', 'computer voices: she-, sheet, sheep, shield, Sheena (2 of 3)')]
    for (key, label), color, marker in zip(near, SLOT, MARKERS):
        band(d, sizes, summary, key, color, marker, label, 100)
    d.set_title('Near misses wrongly accepted (at the live threshold)')
    d.set_ylabel('accepted (%)')
    for ax in axes.flat:
        size_axis(ax, sizes)
        ax.set_ylim(bottom=0)
        ax.legend(loc='best')
    a.set_ylim(0, 1)
    b.set_ylim(0, 100)
    d.set_ylim(0, 75)
    d.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'fig_quality.png', dpi=200)
    plt.close(fig)


def fig_tradeoff(summary, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    f1 = np.array([r['test_f1'] for r in summary])
    sd = np.array([r['test_f1_sd'] for r in summary])
    for variant, label, color, marker in (('base', 'plain RV32IM', SLOT[0], 'o'), ('kdot', 'kdot', SLOT[1], 's')):
        cyc = np.array([r[f'{variant}_cycles_max'] for r in summary])
        ax.errorbar(cyc, f1, yerr=sd, color=color, marker=marker, markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, linewidth=1.8, elinewidth=1, capsize=0, label=label, zorder=3)
        for r, x, y in zip(summary, cyc, f1):
            ax.annotate(f'{r["hidden"]}', (x, y), xytext=(4, -11), textcoords='offset points', ha='left',
                        color=MUTED, fontsize=7.5)
    ax.set_xscale('log')
    limit_line(ax, BUDGET, '25 M cycles = 250 ms budget', horizontal=False)
    ax.set_xlim(1e4, 4e7)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f'{v / 1e6:g} M' if v >= 1e6 else f'{v / 1e3:g} k'))
    ax.set_ylim(0, 1)
    ax.set_xlabel('worst-case inference cycles (log scale; labels = hidden neurons)')
    ax.set_ylabel('test F1 (mean ± 1 sd)')
    ax.set_title('Same quality, 22–34× fewer cycles with kdot; all sizes far inside the budget')
    ax.grid(True, axis='x', which='major')
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(out / 'fig_tradeoff.png', dpi=200)
    plt.close(fig)


def fig_delay(summary, pc, out):
    sizes = [r['hidden'] for r in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    y = np.arange(len(sizes))
    for ax, variant, title, xmax in ((axes[0], 'base', 'Plain RV32IM', 270), (axes[1], 'kdot', 'kdot', None)):
        infer = np.array([r[f'{variant}_cycles_mean'] for r in summary]) * 1e3 / CLOCK
        rtt = np.array([r[f'{variant}_rtt_mean_ms'] for r in summary])
        parts = [('PC: sound picture (features.py)', np.full(len(sizes), pc['median_ms']), SLOT[2]),
                 ('Ethernet + board server', np.maximum(rtt - infer, 0), SLOT[1]),
                 ('inference on the RISC-V core', infer, SLOT[0])]
        left = np.zeros(len(sizes))
        for label, width, color in parts:
            ax.barh(y, width, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=1.5, label=label)
            left += width
        for yy, total in zip(y, left):
            ax.annotate(f'{total:.1f} ms', (total, yy), xytext=(5, 0), textcoords='offset points', va='center',
                        color=INK2, fontsize=8)
        if xmax:
            limit_line(ax, 250, 'next window after 250 ms', horizontal=False)
            ax.set_xlim(0, xmax)
        else:
            ax.set_xlim(0, left.max() * 1.35)
        ax.set_title(title)
        ax.set_xlabel('time per window (ms, mean)')
        ax.grid(True, axis='x')
        ax.grid(False, axis='y')
    axes[0].set_yticks(y, [f'H = {h}' for h in sizes])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.01, 'Not shown, the same for every size: after the word ends, the LED waits for the next 250 ms '
             'window and the "2 of 3" confirmation (about 0.25-0.5 s).', ha='center', color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out / 'fig_delay.png', dpi=200)
    plt.close(fig)


def fig_network(out):
    x = np.load(VECTORS)['x'][0].reshape(24, 32)   # vector 0: a "Sheila" test clip, mel-major
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 4.6)
    ax.set_aspect('equal')   # round neurons
    ax.imshow(x, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('blues', [SURFACE] + BLUES),
              origin='lower', extent=(0.2, 3.4, 0.6, 3.0), aspect='auto', zorder=2)
    ax.add_patch(Rectangle((0.2, 0.6), 3.2, 2.4, fill=False, edgecolor=AXIS, linewidth=0.8, zorder=3))
    ax.text(1.8, 3.25, 'Input: 24 mel bands × 32 time slices\n= 768 numbers (one 1 s window of "Sheila")',
            ha='center', va='bottom', color=INK, fontsize=9)
    ax.text(0.1, 1.8, 'pitch →', rotation=90, ha='right', va='center', color=MUTED, fontsize=8)
    ax.text(1.8, 0.45, 'time →', ha='center', va='top', color=MUTED, fontsize=8)
    hidden_y = np.linspace(0.3, 3.3, 7)
    outputs = {'Sheila': 2.4, 'not Sheila': 1.2}
    for hy in hidden_y:
        for py in np.linspace(0.7, 2.9, 6):
            ax.plot([3.45, 5.55], [py, hy], color=GRID, linewidth=0.6, zorder=1)
        for oy in outputs.values():
            ax.plot([5.75, 7.65], [hy, oy], color=GRID, linewidth=0.6, zorder=1)
        ax.add_patch(Circle((5.65, hy), 0.17, color=SLOT[0], zorder=3))
    ax.text(5.65, 3.75, '⋮', ha='center', va='center', color=INK2, fontsize=12)
    ax.text(5.65, 4.05, 'H hidden spiking neurons\n= the network size', ha='center', va='bottom', color=INK, fontsize=9)
    ax.text(5.65, -0.2, 'each: 768 weights + 1 bias + 2 output weights\n= 1,544 bytes on the board',
            ha='center', va='top', color=INK2, fontsize=8)
    for name, oy in outputs.items():
        ax.add_patch(Circle((7.8, oy), 0.2, color=SLOT[1], zorder=3))
        ax.text(8.1, oy, f'score "{name}"', va='center', color=INK, fontsize=9)
    ax.text(8.45, 0.55, 'decision: "Sheila" score minus\n"not Sheila" score ≥ threshold', ha='center', va='top',
            color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'fig_network.png', dpi=200)
    plt.close(fig)


def fig_memory_map(sizes, out):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5.2), gridspec_kw={'height_ratios': [1, 2.6]})
    kib = 1024
    regions = [(0x00000, 0x10000, 'code + variables\n(64 KiB area, ~1.2 KB used)', AXIS),
               (0x10000, 0x18000, 'mailbox, input\n(rest unused)', GRID),
               (0x18000, 0x3c000, 'model weights: 144 KiB', BLUES[1]),
               (0x3c000, 0x40000, 'stack', AXIS)]
    for start, end, label, color in regions:
        top.barh(0, (end - start) / kib, left=start / kib, height=0.6, color=color, edgecolor=SURFACE, linewidth=2)
        top.text((start + end) / 2 / kib, 0, label, ha='center', va='center', color=INK, fontsize=8)
    top.set_xlim(0, 256)
    top.set_yticks([])
    top.set_xticks([0, 64, 96, 240, 256])
    top.set_xlabel('address in the 256 KiB on-chip memory (KiB)')
    top.set_title('The RISC-V core\'s memory (BRAM on the FPGA)')
    top.grid(False)
    rows = list(sizes) + [96]
    for i, h in enumerate(rows):
        used = (BYTES_PER_NEURON * h + BYTES_FIXED) / kib
        fits = h <= 95
        bottom.barh(i, used, height=0.55, color=SLOT[0] if fits else SURFACE, edgecolor=SLOT[0] if fits else LIMIT,
                    linewidth=0 if fits else 1.2, hatch=None if fits else '////')
        text = f'{used:.1f} KiB ({100 * used * kib / MODEL_LIMIT:.0f}%)' if fits else f'{used:.1f} KiB: does not fit'
        bottom.annotate(text, (used, i), xytext=(5, 0), textcoords='offset points', va='center', color=INK2, fontsize=8)
    limit_line(bottom, MODEL_LIMIT / kib, '144 KiB', horizontal=False)
    bottom.set_yticks(range(len(rows)), [f'H = {h}' for h in rows])
    bottom.invert_yaxis()
    bottom.set_xlim(0, 175)
    bottom.set_xlabel('model weights (KiB) = (1,544 · H + 8) / 1,024')
    bottom.set_title('How full the 144 KiB model area is at each size')
    bottom.grid(True, axis='x')
    bottom.grid(False, axis='y')
    fig.tight_layout()
    fig.savefig(out / 'fig_memory_map.png', dpi=200)
    plt.close(fig)


def write_tables(records, summary, out):
    keys = list(dict.fromkeys(k for r in records for k in r))
    with open(out / 'models.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(records)
    keys = list(dict.fromkeys(k for r in summary for k in r))
    with open(out / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in summary)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, default=TOOLS.parent / 'results/E2')
    a = p.parse_args()
    out = a.results
    keyword, records, summary = load(out)
    pc = json.loads((out / 'pc_timing.json').read_text())
    report = {'keyword': keyword, 'sizes': [r['hidden'] for r in summary],
              'all_bit_exact': all(r['bit_exact'] for r in records), 'fits': fits(records),
              'efficient_size': efficient_size(summary), 'pc_features_ms': pc['median_ms'],
              'summary': [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in summary]}
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    write_tables(records, summary, out)
    style()
    fig_overview(summary, out)
    fig_cost(summary, out)
    fig_quality(summary, keyword, out)
    fig_tradeoff(summary, out)
    fig_delay(summary, pc, out)
    fig_network(out)
    fig_memory_map([r['hidden'] for r in summary], out)
    print(json.dumps({k: report[k] for k in ('sizes', 'all_bit_exact', 'fits', 'efficient_size', 'pc_features_ms')},
                     indent=2))


if __name__ == '__main__':
    main()
