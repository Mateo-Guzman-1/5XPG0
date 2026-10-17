"""Shared figure style for the report: plain journal look that matches the LaTeX template.

The report template (pedronavarro17/report-template) typesets in mlmodern (Computer Modern) on A4
with 2.5 cm margins, so the text width is 16 cm. Figures are made at their final size and included
without scaling (\\includegraphics[width=\\textwidth] or [width=0.49\\textwidth]), so the 9 pt labels
print at 9 pt.

Usage:
    from plot_style import apply, FULL, HALF, COLORS, MARKERS, LINES, save, limit_line, panel_label
    apply()
    fig, ax = plt.subplots(figsize=HALF)
    ...
    save(fig, out_dir / 'fig_name')      # writes fig_name.pdf (vector) and fig_name.png (300 dpi)
"""
import logging
from pathlib import Path
import matplotlib as mpl
from cycler import cycler

CM = 1 / 2.54
FULL = (16.0 * CM, 6.0 * CM)    # full text width
HALF = (7.8 * CM, 5.5 * CM)     # one of two side-by-side subfigures
TALL = (16.0 * CM, 9.0 * CM)    # full width, two rows

# Okabe-Ito colour-blind-safe palette, black first. Every series also gets its own marker and line
# style, so the figures stay readable when printed in grayscale.
COLORS = ['#000000', '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
LINES = ['-', '--', ':', '-.', (0, (5, 1.5)), (0, (3, 1, 1, 1, 1, 1))]
GREY = '#7f7f7f'


def apply():
    # fontTools complains about the creation date stored in matplotlib's cmr10 font when it embeds
    # it in a PDF; the warning is harmless.
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    mpl.rcParams.update({
        # Computer Modern for text and numbers (tick labels go through mathtext, which also gives
        # a proper minus sign; the cmr10 text font has none).
        'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True,
        'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8,
        # Thin black frame on all four sides, ticks pointing in.
        'axes.linewidth': 0.6, 'axes.edgecolor': 'black', 'axes.facecolor': 'white',
        'axes.grid': False, 'axes.prop_cycle': cycler(color=COLORS),
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5, 'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
        'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'lines.linewidth': 0.9, 'lines.markersize': 3.5, 'errorbar.capsize': 2.0,
        'legend.frameon': False, 'legend.handlelength': 2.2, 'legend.borderaxespad': 0.4,
        # Fixed figure size (no tight cropping), so the PDF is exactly FULL or HALF.
        'figure.facecolor': 'white', 'figure.dpi': 150, 'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.02, 'figure.constrained_layout.w_pad': 0.02,
        'savefig.dpi': 300, 'savefig.facecolor': 'white', 'savefig.bbox': None,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def save(fig, stem):
    """Write <stem>.pdf (vector, fonts embedded) and <stem>.png (300 dpi)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'))


def limit_line(ax, value, label, axis='y', where=0.98, va='bottom', ha='right'):
    """Thin dashed grey line for a platform limit, labelled in small text."""
    if axis == 'y':
        ax.axhline(value, color=GREY, lw=0.6, ls='--', zorder=0)
        ax.annotate(label, (where, value), xycoords=('axes fraction', 'data'), fontsize=7,
                    color=GREY, ha=ha, va=va, xytext=(0, 1.5 if va == 'bottom' else -1.5),
                    textcoords='offset points')
    else:
        ax.axvline(value, color=GREY, lw=0.6, ls='--', zorder=0)
        ax.annotate(label, (value, where), xycoords=('data', 'axes fraction'), fontsize=7,
                    color=GREY, ha=ha, va='top', rotation=90, xytext=(-1.5, 0),
                    textcoords='offset points')


def panel_label(ax, text):
    """(a), (b), ... in the top-left corner inside the axes."""
    ax.text(0.025, 0.965, text, transform=ax.transAxes, ha='left', va='top', fontsize=9)


def log_grid(ax):
    """Light dotted major grid, used only on log-log axes."""
    ax.grid(True, which='major', ls=':', lw=0.4, color='#b0b0b0')
    ax.set_axisbelow(True)


def series_style(i, filled=True):
    """Colour, marker and line style of series number i."""
    color = COLORS[i % len(COLORS)]
    return dict(color=color, marker=MARKERS[i % len(MARKERS)], ls=LINES[i % len(LINES)],
                mfc=color if filled else 'white', mec=color, mew=0.7)
