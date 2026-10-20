"""R2 figures and tables as self-contained LaTeX (pgfplots with inline coordinates, booktabs).

Reads results/R2/{designs.csv, figdata_test.json}, results/R1/models.csv (stage 1 alone, false accepts by source),
results/R2/verifier/*_training.json and results/R2/mateo/*.pt (fine-tune settings), and the Vivado power terms
(energy_model.load_power). Writes reports/latex/cascade_gru/{figures,tables}/*.tex.
Every figure uses figures/r1_style.tex (the group report needs that one \\input in its preamble) and defines its own
R2 marker styles inside the figure, so each file compiles alone. No fillbetween: the group preamble lacks it.
Pedro's choice at G2: both Phase B results are shown side by side, neither is recommended over the other.
"""
import csv
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from energy_model import load_power  # noqa: E402

ROOT = TOOLS.parent
R2 = ROOT / 'results/R2'
OUT = ROOT / 'reports/latex/cascade_gru'
TITLE = 'title style={font=\\footnotesize, at={(0.02,0.97)}, anchor=north west}'
T_V_MAX, T_V_MEAN = 107.3, 105.83     # ms, verifier call on plain kdot (FINAL REPORT, Parallel section)
VERIFIER_KIB = 53320 / 1024

# design key, legend label, short axis label, marker style
# Design codes used in the report (no personal names): letter = networks, number = threshold rule
# (0: no verifier, 1: highest live recall, 2: at most 2,150 verifier calls per hour).
DESIGNS = [('stage1_R1', 'W0: window SNN alone', 'W0', 'r2s1'),
           ('cascade_A', 'A1: original verifier, highest recall', 'A1', 'r2A'),
           ('cascade_A_calls2150', 'A2: original verifier, $\\leq$2,150 calls/h', 'A2', 'r2A5'),
           ('cascade_B', 'B1: fine-tuned verifier, highest recall', 'B1', 'r2Br'),
           ('cascade_B_calls2150', 'B2: fine-tuned verifier, $\\leq$2,150 calls/h', 'B2', 'r2Bc'),
           ('mateo_stage1', 'S0: streaming SNN alone', 'S0', 'r2ms1'),
           ('mateo_cascade', 'S1: streaming SNN + original verifier', 'S1', 'r2mateo')]
LABEL = {k: lab for k, lab, _, _ in DESIGNS}
SHORT = {k: s for k, _, s, _ in DESIGNS}
STYLE = {k: st for k, _, _, st in DESIGNS}

# Marker styles: colour = verifier (blue: original, green: fine-tuned), filled = highest-recall rule,
# hollow = call limit. Defined inside every figure so that the file needs only r1_style.tex.
R2STYLE = r"""  \pgfplotsset{
    r2s1/.style={r1pruned},
    r2A/.style={only marks, color=r1dense, mark=*, mark size=2pt, mark options={solid, fill=r1dense}},
    r2A5/.style={only marks, color=r1dense, mark=o, mark size=2.2pt, mark options={solid, line width=0.9pt}},
    r2Br/.style={only marks, color=r1prunedbits, mark=diamond*, mark size=2.8pt, mark options={solid, fill=r1prunedbits}},
    r2Bc/.style={only marks, color=r1prunedbits, mark=diamond, mark size=3pt, mark options={solid, line width=0.9pt}},
    r2ms1/.style={r1ref},
    r2mateo/.style={only marks, color=r1ref, mark=+, mark size=3.2pt, very thick},
    r2big/.style={mark size=3.8pt},
  }"""


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load():
    d = {r['design']: {k: (num(v) if num(v) is not None else v) for k, v in r.items()}
         for r in csv.DictReader(open(R2 / 'designs.csv', encoding='utf-8'))}
    fig = json.loads((R2 / 'figdata_test.json').read_text())
    r1 = [r for r in csv.DictReader(open(ROOT / 'results/R1/models.csv', encoding='utf-8'))
          if r['config'] == 'h48_k64' and r['variant'] == 'ksdot_lut']
    return d, fig, r1


def coords(points):
    return ' '.join(f'({x:.6g},{y:.6g})' if e is None else f'({x:.6g},{y:.6g}) +- (0,{e:.6g})' for x, y, e in points)


def write(name, text, kind='figures'):
    (OUT / kind).mkdir(parents=True, exist_ok=True)
    (OUT / kind / f'{name}.tex').write_text(text, encoding='utf-8')
    print('wrote', kind, name)


def sd(d, key, scale=1.):
    v = d.get(key + '_sd')
    return None if v in (None, '') else scale * v


# ----------------------------------------------------------------------------- figures

CASCADES = ('cascade_A', 'cascade_A_calls2150', 'cascade_B', 'cascade_B_calls2150', 'mateo_cascade')


def fig_quality(D):
    """(a) live recall vs other words accepted, (b) live recall vs false accepts per hour; mean +- sd over seeds.
    Zoomed to 80-90 % live so that no two markers overlap; the stand-alone stage-1 designs are in the caption."""
    pa, pb = [], []
    for k in CASCADES:
        st = STYLE[k]
        d = D[k]
        e = sd(d, 'live_recall', 100)
        pa.append(f'    \\addplot[{st}, r1err] coordinates {{{coords([(100 * d["live_other"], 100 * d["live_recall"], e)])}}};\n'
                  f'    \\addlegendentry{{{LABEL[k]}}}')
        pb.append(f'    \\addplot[{st}, r1err] coordinates {{{coords([(d["fa_per_hour"], 100 * d["live_recall"], e)])}}};')
    s1, ms1 = D['stage1_R1'], D['mateo_stage1']
    off_scale = (f"Off this scale, the first stages alone: W0 (window SNN) {100 * s1['live_recall']:.1f}\\,\\% live "
                 f"({100 * s1['live_other']:.2f}\\,\\% other words, {s1['fa_per_hour']:.2f} per hour) and S0 "
                 f"(streaming SNN) {100 * ms1['live_recall']:.1f}\\,\\% ({100 * ms1['live_other']:.2f}\\,\\%, "
                 f"{ms1['fa_per_hour']:.2f} per hour).")
    text = f"""% R2: the cascades against the detection requirements (generated by make_r2_latex.py).
\\begin{{figure*}}[t]
{R2STYLE}
  \\centering
  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, width=0.49\\textwidth, height=6.0cm, xmin=0, xmax=0.25, ymin=80, ymax=90,
      xtick={{0,0.05,0.1,0.15,0.2,0.25}}, xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=2}},
      xlabel={{Other words accepted (\\%)}}, ylabel={{Live ``Sheila'' detected (\\%)}},
      legend to name=r2qualitylegend, legend columns=3,
      legend style={{/tikz/every even column/.append style={{column sep=10pt}}}}, title={{(a)}}, {TITLE}]
    \\fill[r1pass] (axis cs:0,85) rectangle (axis cs:0.2,90);
    \\draw[r1req] (axis cs:0.2,80) -- (axis cs:0.2,90);
    \\draw[r1req] (axis cs:0,85) -- (axis cs:0.25,85);
    \\node[r1reqlabel, anchor=south west] at (axis cs:0.203,80.15) {{$\\leq$0.2\\,\\%}};
    \\node[r1reqlabel, anchor=north east] at (axis cs:0.248,84.85) {{$\\geq$85\\,\\%}};
{chr(10).join(pa)}
  \\end{{axis}}
  \\end{{tikzpicture}}\\hfill
  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, width=0.49\\textwidth, height=6.0cm, xmin=0.8, xmax=2.4, ymin=80, ymax=90,
      xlabel={{False accepts per hour}}, ylabel={{Live ``Sheila'' detected (\\%)}}, title={{(b)}}, {TITLE}]
    \\fill[r1pass] (axis cs:0.8,85) rectangle (axis cs:2,90);
    \\draw[r1req] (axis cs:2,80) -- (axis cs:2,90);
    \\draw[r1req] (axis cs:0.8,85) -- (axis cs:2.4,85);
    \\node[r1reqlabel, anchor=south west] at (axis cs:2.02,80.15) {{$\\leq$2 per hour}};
    \\node[r1reqlabel, anchor=north east] at (axis cs:2.39,84.85) {{$\\geq$85\\,\\%}};
{chr(10).join(pb)}
  \\end{{axis}}
  \\end{{tikzpicture}}\\\\[2pt]
  \\pgfplotslegendfromname{{r2qualitylegend}}
  \\caption{{Detection quality of the cascades on the test sets against the requirements; the shaded corner
  meets both. Each threshold was fixed on validation and the test was run once. Mean $\\pm$ sd: 3 window-SNN
  seeds (A1, A2), 9 cascades for B1 and B2 (3 fine-tuned verifiers $\\times$ 3 window-SNN seeds); S1 is a single
  model. Colour: the verifier (blue: original; green: fine-tuned on the window SNN's false proposals).
  Filled: the highest-recall rule; hollow: at most 2,150 verifier calls per hour. {off_scale}}}
  \\label{{fig:r2-quality}}
\\end{{figure*}}
"""
    write('fig_r2_quality', text)


def fig_curves(fig):
    """Test operating curves in three panels, so that no markers overlap: (a) overview live vs FA/h, all curves;
    (b) the requirement corner of (a), with the operating points; (c) live vs verifier calls per hour
    (requirement-feasible points only), with the region above the call limit shaded."""
    cv = fig['curves']
    line = {'stage1_alone': 'color=r1pruned', 'cascade_A': 'color=r1dense', 'cascade_B': 'color=r1prunedbits',
            'mateo_cascade': 'color=r1ref, densely dotted'}
    name = {'stage1_alone': f'W0: window SNN alone (seed {cv["stage1_alone"]["stage1_seed"]})',
            'cascade_A': f"A: original verifier (window-SNN seed {cv['cascade_A']['stage1_seed']})",
            'cascade_B': f"B: fine-tuned verifier {cv['cascade_B']['verifier'].split()[-1]} (window-SNN seed "
                         f"{cv['cascade_B']['stage1_seed']})",
            'mateo_cascade': 'S1: streaming SNN + original verifier'}
    chosen = {('stage1_alone', 'R1'): 'r2s1', ('cascade_A', 'A'): 'r2A', ('cascade_A', 'A5'): 'r2A5',
              ('cascade_B', 'B_recall'): 'r2Br', ('cascade_B', 'B_calls'): 'r2Bc', ('mateo_cascade', 'mateo'): 'r2mateo'}
    small = 'mark size=2.3pt'
    pa, pb, pc, ma, mb, mc = [], [], [], [], [], []
    for k, c in cv.items():
        pts = sorted(c['points'], key=lambda p: p['t1'])
        a = ' '.join(f'({max(p["fa_per_hour"], .3):.4g},{100 * p["live_recall"]:.4g})' for p in pts)
        pa.append(f'    \\addplot[{line[k]}, thick, mark=none] coordinates {{{a}}};\n    \\addlegendentry{{{name[k]}}}')
        if k != 'stage1_alone':
            pb.append(f'    \\addplot[{line[k]}, thick, mark=none] coordinates {{{a}}};')
        ok = [p for p in pts if p['calls_per_hour'] > 0 and p['fa_per_hour'] <= 2 and p['live_other'] <= .002]
        if ok:
            b = ' '.join(f'({p["calls_per_hour"]:.5g},{100 * p["live_recall"]:.4g})' for p in ok)
            pc.append(f'    \\addplot[{line[k]}, thick, mark=none] coordinates {{{b}}};')
        ch = c.get('chosen') or {'R1': c['chosen_t1']}
        for n, t in ch.items():
            p = next(x for x in pts if x['t1'] == t)
            st = chosen[(k, n)]
            xy = f'({max(p["fa_per_hour"], .3):.4g},{100 * p["live_recall"]:.4g})'
            if k == 'stage1_alone':
                ma.append(f'    \\addplot[{st}, r2big] coordinates {{{xy}}};')
                continue
            mb.append(f'    \\addplot[{st}, {small}] coordinates {{{xy}}};')
            if p['calls_per_hour'] > 0:
                mc.append(f'    \\addplot[{st}, {small}] coordinates {{({p["calls_per_hour"]:.5g},{100 * p["live_recall"]:.4g})}};')

    def feasible(k):
        return [p for p in cv[k]['points'] if p['fa_per_hour'] <= 2 and p['live_other'] <= .002]

    def calls_for(k, rec=.85):
        return min(p['calls_per_hour'] for p in feasible(k) if p['live_recall'] >= rec)
    m120 = max(p['live_recall'] for p in feasible('mateo_cascade') if p['calls_per_hour'] <= 120)
    n = lambda x: f'{x:,.0f}'.replace(',', '{,}')
    claim = (f"On test, the streaming SNN gives {100 * m120:.0f}\\,\\% with at most 120 verifier calls per hour and "
             f"85\\,\\% from {n(calls_for('mateo_cascade'))}; the window SNN needs {n(calls_for('cascade_B'))} calls per "
             f"hour for 85\\,\\% with the fine-tuned verifier and {n(calls_for('cascade_A'))} with the original one.")
    ax = 'r1axis, scale only axis, width=0.235\\textwidth, height=4.3cm'
    text = f"""% R2: test operating curves of one representative cascade per verifier (generated by make_r2_latex.py).
\\begin{{figure*}}[t]
{R2STYLE}
  \\centering
  \\begin{{tikzpicture}}
  \\begin{{axis}}[{ax}, xmode=log, xmin=0.3, xmax=60, ymin=20, ymax=100,
      xlabel={{False accepts per hour}}, ylabel={{Live ``Sheila'' detected (\\%)}},
      legend to name=r2curveslegend, legend columns=2,
      legend style={{/tikz/every even column/.append style={{column sep=10pt}}}},
      title={{(a) all thresholds}}, title style={{font=\\footnotesize}}]
    \\fill[r1pass] (axis cs:0.3,85) rectangle (axis cs:2,100);
    \\draw[r1req] (axis cs:2,20) -- (axis cs:2,100);
    \\draw[r1req] (axis cs:0.3,85) -- (axis cs:60,85);
    \\node[r1reqlabel, anchor=south west] at (axis cs:2.08,21) {{$\\leq$2 per hour}};
{chr(10).join(pa)}
{chr(10).join(ma)}
    \\draw[black!60, thin] (axis cs:0.5,80) rectangle (axis cs:3,90);
    \\node[font=\\scriptsize, text=black!70, anchor=north west, inner sep=1pt] at (axis cs:3,90) {{(b)}};
  \\end{{axis}}
  \\end{{tikzpicture}}\\hfill
  \\begin{{tikzpicture}}
  \\begin{{axis}}[{ax}, xmode=log, xmin=0.5, xmax=3, ymin=80, ymax=90, xtick={{0.5,1,2,3}}, xticklabels={{0.5,1,2,3}},
      xlabel={{False accepts per hour}}, ylabel={{Live ``Sheila'' detected (\\%)}},
      title={{(b) the requirement corner of (a)}}, title style={{font=\\footnotesize}}]
    \\fill[r1pass] (axis cs:0.5,85) rectangle (axis cs:2,90);
    \\draw[r1req] (axis cs:2,80) -- (axis cs:2,90);
    \\draw[r1req] (axis cs:0.5,85) -- (axis cs:3,85);
    \\node[r1reqlabel, anchor=south west] at (axis cs:2.04,80.15) {{$\\leq$2/h}};
    \\node[r1reqlabel, anchor=north east] at (axis cs:2.97,84.85) {{$\\geq$85\\,\\%}};
{chr(10).join(pb)}
{chr(10).join(mb)}
  \\end{{axis}}
  \\end{{tikzpicture}}\\hfill
  \\begin{{tikzpicture}}
  \\begin{{axis}}[{ax}, xmode=log, xmin=100, xmax=15000, ymin=80, ymax=90,
      xlabel={{Verifier calls per hour}}, ylabel={{Live ``Sheila'' detected (\\%)}},
      title={{(c) verifier calls}}, title style={{font=\\footnotesize}}]
    \\fill[black!6] (axis cs:2150,80) rectangle (axis cs:15000,90);
    \\node[font=\\scriptsize, text=black!60, anchor=north east, align=right, inner sep=1.5pt] at (axis cs:15000,90)
      {{above the call limit\\\\(2{{,}}150 per hour)}};
    \\draw[r1req] (axis cs:100,85) -- (axis cs:15000,85);
    \\node[r1reqlabel, anchor=north east] at (axis cs:14500,84.85) {{$\\geq$85\\,\\%}};
{chr(10).join(pc)}
{chr(10).join(mc)}
  \\end{{axis}}
  \\end{{tikzpicture}}\\\\[2pt]
  \\pgfplotslegendfromname{{r2curveslegend}}
  \\caption{{What each verifier can trade, on test. Lines: every stage-1 threshold $t_1$ at the verifier
  threshold $t_2$ fixed on validation; markers: the operating points fixed on validation (styles as in
  Fig.~\\ref{{fig:r2-quality}}). The configuration shown sits at the design means. (a) All thresholds;
  (b) the requirement corner of (a); (c) live detection against verifier calls, only for thresholds that also
  meet $\\leq$0.2\\,\\% other words and $\\leq$2 false accepts per hour. {claim} Illustration only: no
  operating point was chosen on these curves.}}
  \\label{{fig:r2-curves}}
\\end{{figure*}}
"""
    write('fig_r2_curves', text)


COST = ['stage1_R1', 'cascade_A', 'cascade_A_calls2150', 'cascade_B', 'cascade_B_calls2150', 'mateo_cascade']


def bar(x0, x1, y0, y1, fill):
    return f'    \\fill[{fill}] (axis cs:{x0:.3f},{y0:.5g}) rectangle (axis cs:{x1:.3f},{y1:.5g});'


def fig_cost(D):
    """(a) worst compute per request, (b) energy per hour (busy-polling vs clock-gated breakdown), (c) memory."""
    p = load_power()['ksdot']
    n = len(COST)
    xt = ','.join(str(i) for i in range(n))
    xl = ','.join('{' + SHORT[k] + '}' for k in COST)
    common = (f'scale only axis, width=0.25\\textwidth, height=4.0cm, xmin=-0.6, xmax={n - 0.4}, xtick={{{xt}}}, '
              f'xticklabels={{{xl}}}, x tick label style={{rotate=50, anchor=east, font=\\scriptsize}}')
    # (a) compute: stage 1 + verifier
    a = []
    for i, k in enumerate(COST):
        d = D[k]
        tv = (T_V_MAX if k != 'mateo_cascade' else T_V_MEAN) if k != 'stage1_R1' else 0.
        t1 = d['t_max_ms'] - tv
        a += [bar(i - .3, i + .3, 0, t1, 'r1pruned!80'), bar(i - .3, i + .3, t1, d['t_max_ms'], 'r1dense!45')]
        a.append(f'    \\node[font=\\tiny, anchor=south, inner sep=1pt] at (axis cs:{i},{d["t_max_ms"]:.4g}) {{{d["t_max_ms"]:.3g}}};')
    # (b) energy per hour: left bar busy-polling, right bar clock-gated = floor + stage 1 + verifier (core power)
    b = []
    for i, k in enumerate(COST):
        d = D[k]
        busy = d['p_avg_busy_W'] * 3600
        b.append(bar(i - .36, i - .02, 0, busy, 'black!55'))
        if k == 'mateo_cascade':
            b.append(f'    \\node[font=\\tiny, rotate=90, anchor=west, text=black!70] at (axis cs:{i + .19},40) {{not computed}};')
            continue
        floor = (p['pl_static'] + p['infra_dyn']) * 3600
        ver = d['calls_per_hour'] * p['core_dyn'] * T_V_MEAN / 1000
        s1 = d['p_avg_gated_W'] * 3600 - floor - ver
        b += [bar(i + .02, i + .36, 0, floor, 'black!18'), bar(i + .02, i + .36, floor, floor + s1, 'r1pruned!80'),
              bar(i + .02, i + .36, floor + s1, floor + s1 + ver, 'r1dense!45')]
    # (c) memory: stage-1 model + verifier export
    c = []
    for i, k in enumerate(COST):
        d = D[k]
        if d.get('memory_kib') in (None, ''):
            c.append(f'    \\node[font=\\tiny, rotate=90, anchor=west, text=black!70] at (axis cs:{i},4) {{not reported}};')
            continue
        m1 = d['memory_kib'] - (VERIFIER_KIB if k != 'stage1_R1' else 0.)
        c += [bar(i - .3, i + .3, 0, m1, 'r1pruned!80'), bar(i - .3, i + .3, m1, d['memory_kib'], 'r1dense!45')]
        c.append(f'    \\node[font=\\tiny, anchor=south, inner sep=1pt] at (axis cs:{i},{d["memory_kib"]:.4g}) {{{d["memory_kib"]:.1f}}};')

    def panel(label, ylabel, extra, req, req_text, body):
        return f"""  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, {common}, {extra}, ylabel={{{ylabel}}}, title={{({label})}}, {TITLE}]
    \\draw[r1req] (axis cs:-0.6,{req}) -- (axis cs:{n - 0.4},{req});
    \\node[r1reqlabel, anchor=south east] at (axis cs:{n - 0.5},{req}) {{{req_text}}};
{chr(10).join(body)}
  \\end{{axis}}
  \\end{{tikzpicture}}"""
    pa = panel('a', 'Worst compute per request (ms)', 'ymin=0, ymax=240', 200, '$\\leq$200\\,ms', a)
    pb = panel('b', 'PL energy per hour (J)', 'ymin=0, ymax=4200', 3600, '$\\leq$1\\,W = 3,600\\,J/h', b)
    pc = panel('c', 'Model memory (KiB)', 'ymin=0, ymax=165', 144, '$\\leq$144\\,KiB', c)
    key = ('\\tikz\\fill[r1pruned!80] (0,0) rectangle (2.2mm,2.2mm); stage 1 \\quad '
           '\\tikz\\fill[r1dense!45] (0,0) rectangle (2.2mm,2.2mm); GRU verifier \\quad '
           '\\tikz\\fill[black!55] (0,0) rectangle (2.2mm,2.2mm); busy-polling, as built \\quad '
           '\\tikz\\fill[black!18] (0,0) rectangle (2.2mm,2.2mm); clock-gated floor (static + interface)')
    text = f"""% R2: cost requirements of the cascades (generated by make_r2_latex.py).
\\begin{{figure*}}[t]
  \\centering
{pa}\\hfill
{pb}\\hfill
{pc}\\\\[3pt]
  {{\\scriptsize {key}}}
  \\caption{{Cost of the cascades. (a) Worst compute per 250\\,ms request: the window SNN on \\texttt{{ksdot}}+table
  (0.35\\,ms, from board runs) plus one verifier call on plain \\texttt{{kdot}} (107.3\\,ms, from the RTL simulation
  of the parallel-processing design); S1 as reported for the streaming design. (b) PL energy per hour: left bar,
  busy-polling (the firmware as built: the core never idles, so the energy does not depend on the work); right
  bar, clock gating (a calculation): the floor plus the core energy of stage 1 and of the verifier calls. W0, A
  and B on the \\texttt{{ksdot}} fabric; S1 on the dual-core fabric of the parallel-processing design (0.727\\,W).
  (c) Model memory: stage-1 model plus the 52.1\\,KiB int8 verifier.}}
  \\label{{fig:r2-cost}}
\\end{{figure*}}
"""
    write('fig_r2_cost', text)


REQ_ROWS = [('live_recall', "Live ``Sheila'' detected", '>=', .85, lambda v: f'{100 * v:.1f}\\,\\%', '$\\geq$85\\,\\%'),
            ('fa_per_hour', 'False accepts per hour', '<=', 2., lambda v: f'{v:.2f}', '$\\leq$2'),
            ('live_other', 'Other words accepted', '<=', .002, lambda v: f'{100 * v:.2f}\\,\\%', '$\\leq$0.2\\,\\%'),
            ('latency_ms', 'Word end to decision', '<=', 500., lambda v: f'{v:.0f}\\,ms', '$\\leq$500\\,ms'),
            ('t_max_ms', 'Compute per request', '<=', 200., lambda v: f'{v:.1f}\\,ms' if v >= 1 else f'{v:.2f}\\,ms',
             '$\\leq$200\\,ms'),
            ('memory_kib', 'Model memory', '<=', 144., lambda v: f'{v:.1f}\\,KiB', '$\\leq$144\\,KiB'),
            ('p_avg_busy_W', 'Average PL power', '<=', 1., lambda v: f'{1000 * v:.0f}\\,mW', '$\\leq$1\\,W')]


def fig_margin(D):
    """Every requirement as a factor (>1 met by that factor, <1 missed) for both Phase B results, side by side."""
    rows = list(reversed(REQ_ROWS))
    yt = ','.join(str(i) for i in range(len(rows)))
    yl = ','.join('{' + r[1] + '}' for r in rows)
    bars, labels = [], []
    for j, (k, fill) in enumerate((('cascade_B_calls2150', 'r1prunedbits!35'), ('cascade_B', 'r1prunedbits!90'))):
        d = D[k]
        off = -0.19 if j == 0 else 0.19
        pts = []
        for i, (key, _, op, target, fmt, _) in enumerate(rows):
            v = d[key]
            r = v / target if op == '>=' else target / v
            pts.append(f'({r:.4g},{i})')
            labels.append(f'    \\node[font=\\tiny, anchor=west, inner sep=0.5pt] at (axis cs:{r * 1.08:.4g},{i + off:.3g}) {{{fmt(v)}}};')
        bars.append(f'    \\addplot[draw=black!60, fill={fill}, area legend] coordinates {{{" ".join(pts)}}};\n'
                    f'    \\addlegendentry{{{LABEL[k]}}}')
    top = len(rows) - 0.5
    bc = D['cascade_B_calls2150']
    f_live, f_other = bc['live_recall'] / .85, .002 / bc['live_other']
    text = f"""% R2: by what factor both Phase B cascades meet every requirement (generated by make_r2_latex.py).
\\begin{{figure}}[t]
  \\centering
  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, scale only axis, width=0.58\\columnwidth, height=5.8cm, xbar, bar width=4.2pt, xmode=log,
      log origin=0, xmin=0.5, xmax=300, ymin=-0.5, ymax={top}, ytick={{{yt}}}, yticklabels={{{yl}}},
      y tick label style={{font=\\scriptsize}}, enlarge y limits=false, xlabel={{Factor by which the requirement is met}},
      legend style={{at={{(0.5,1.02)}}, anchor=south}}, reverse legend]
    \\fill[black!7] (axis cs:0.5,-0.5) rectangle (axis cs:1,{top});
    \\draw[r1req] (axis cs:1,-0.5) -- (axis cs:1,{top});
{chr(10).join(bars)}
{chr(10).join(labels)}
  \\end{{axis}}
  \\end{{tikzpicture}}
  \\caption{{B1 and B2 against every requirement, as the factor by which each is met (value/target for live
  detection, target/value otherwise; mean of 9 cascades); numbers are the measured values. Both meet all seven.
  Live detection and other words are the tight ones: {f_live:.2f} and {f_other:.1f} times for B2.}}
  \\label{{fig:r2-margin}}
\\end{{figure}}
"""
    write('fig_r2_margin', text)


CALLS = ['cascade_A', 'cascade_A_calls2150', 'cascade_B', 'cascade_B_calls2150', 'mateo_cascade']


def fig_calls(D, fig, r1):
    """(a) verifier calls per hour, (b) false accepts per hour by source, per design."""
    import numpy as np
    names = list(reversed(['stage1_R1'] + CALLS))
    yt = ','.join(str(i) for i in range(len(names)))
    yl = ','.join('{' + SHORT[k] + '}' for k in names)
    top = len(names) - 0.5
    a = []
    for i, k in enumerate(names):
        # labels start right of the bar and of the dotted 2,150 line, so no text crosses it
        if k == 'stage1_R1':
            a.append(f'    \\node[font=\\tiny, anchor=west, inner sep=1pt] at (axis cs:2400,{i}) {{0 (no verifier)}};')
            continue
        d = D[k]
        a.append(bar(0, d['calls_per_hour'], i - .3, i + .3, 'r1dense!45'))   # (x0, x1, y0, y1): horizontal bar
        e = sd(d, 'calls_per_hour')
        lab = f'{d["calls_per_hour"]:,.0f}' + (f'\\,$\\pm$\\,{e:,.0f}' if e else '')
        a.append(f'    \\node[font=\\tiny, anchor=west, inner sep=1pt] at (axis cs:{max(d["calls_per_hour"], 2150) + 250:.5g},{i}) {{{lab}}};')
    # sources: stage 1 alone from R1 (fa_libri = LibriSpeech false accepts in 18.97 h), cascades from figdata
    hours = 18.97
    src = {'stage1_R1': {'librispeech': float(np.mean([int(r['fa_libri']) / hours for r in r1])),
                         'speech commands word': float(np.mean([float(r['fa_per_hour']) - int(r['fa_libri']) / hours
                                                                for r in r1]))}}
    for k in CALLS:
        rows = fig['fa_sources'][k]
        kinds = {x for r in rows for x in r if x != 'stage1_seed'}
        src[k] = {x: float(np.mean([r.get(x, 0.) for r in rows])) for x in kinds}
    b = []
    for i, k in enumerate(names):
        x0 = 0.
        for kind, fill in (('speech commands word', 'r1bits!75'), ('librispeech', 'r1dense!60'), ('noise only', 'black!30')):
            v = src[k].get(kind, 0.)
            if v > 0:
                b.append(bar(x0, x0 + v, i - .3, i + .3, fill))
                x0 += v
        share = src[k].get('librispeech', 0.) / x0
        b.append(f'    \\node[font=\\tiny, anchor=west, inner sep=1pt] '
                 f'at (axis cs:{max(x0, 2) + 0.08:.4g},{i}) {{{x0:.2f} ({100 * share:.0f}\\,\\% LibriSpeech)}};')
    key = ('\\tikz\\fill[r1bits!75] (0,0) rectangle (2.2mm,2.2mm); Speech Commands word \\quad '
           '\\tikz\\fill[r1dense!60] (0,0) rectangle (2.2mm,2.2mm); LibriSpeech running speech \\quad '
           '\\tikz\\fill[black!30] (0,0) rectangle (2.2mm,2.2mm); noise only')
    text = f"""% R2: verifier calls and the sources of false accepts (generated by make_r2_latex.py).
\\begin{{figure}}[t]
  \\centering
  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, scale only axis, width=0.62\\columnwidth, height=3.0cm, xmin=0, xmax=15500, ymin=-0.5,
      ymax={top}, ytick={{{yt}}}, yticklabels={{{yl}}}, y tick label style={{font=\\scriptsize}},
      enlarge y limits=false, xlabel={{Verifier calls per hour}}, scaled x ticks=false, xtick={{0,5000,10000,15000}},
      xticklabels={{0,5{{,}}000,10{{,}}000,15{{,}}000}}, title={{(a)}}, title style={{font=\\footnotesize}}]
    \\draw[black!55, densely dotted, thick] (axis cs:2150,-0.5) -- (axis cs:2150,{top});
{chr(10).join(a)}
  \\end{{axis}}
  \\end{{tikzpicture}}\\\\[4pt]
  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, scale only axis, width=0.62\\columnwidth, height=3.0cm, xmin=0, xmax=5.2, ymin=-0.5,
      ymax={top}, ytick={{{yt}}}, yticklabels={{{yl}}}, y tick label style={{font=\\scriptsize}},
      enlarge y limits=false, xlabel={{False accepts per hour}}, title={{(b)}}, title style={{font=\\footnotesize}}]
    \\draw[r1req] (axis cs:2,-0.5) -- (axis cs:2,{top});
{chr(10).join(b)}
  \\end{{axis}}
  \\end{{tikzpicture}}\\\\[2pt]
  {{\\scriptsize {key}}}
  \\caption{{(a) Verifier calls per hour on the 19\\,h negatives stream (mean $\\pm$ sd; dotted: the 2,150 call
  limit). (b) False accepts per hour on the same stream by what was playing in the second before the
  detection (dashed: the requirement). Running speech is most of the false accepts in every design.}}
  \\label{{fig:r2-calls}}
\\end{{figure}}
"""
    write('fig_r2_calls', text)


# ----------------------------------------------------------------------------- tables

def mark(ok):
    return '$\\surd$' if ok else '$\\times$'


RESULT_COLS = ['stage1_R1', 'cascade_A', 'cascade_A_calls2150', 'cascade_B', 'cascade_B_calls2150', 'mateo_cascade']
HEAD = {'stage1_R1': 'W0\\\\window SNN\\\\alone', 'cascade_A': 'A1\\\\original verif.\\\\highest recall',
        'cascade_A_calls2150': 'A2\\\\original verif.\\\\$\\leq$2,150 calls/h',
        'cascade_B': 'B1\\\\fine-tuned\\\\highest recall', 'cascade_B_calls2150': 'B2\\\\fine-tuned\\\\$\\leq$2,150 calls/h',
        'mateo_cascade': 'S1\\\\streaming SNN\\\\+ orig.\\ verif.'}


def head(cols):
    return ' & '.join('\\begin{tabular}[b]{@{}c@{}}' + HEAD[k] + '\\end{tabular}' for k in cols)


def tab_results(D):
    lines = []
    for key, name, op, target, fmt, tgt in REQ_ROWS:
        vals = []
        for k in RESULT_COLS:
            v = D[k].get(key)
            if v in (None, ''):
                vals.append('---')
                continue
            e = D[k].get(key + '_sd') if key == 'live_recall' else None
            s = fmt(v) if not e else f'{100 * v:.1f}\\,$\\pm$\\,{100 * e:.1f}\\,\\%'
            vals.append(f'{s} {mark(v >= target if op == ">=" else v <= target)}')
        lines.append(f'    {name} & {tgt} & ' + ' & '.join(vals) + ' \\\\')
    extra = [('Verifier calls per hour', lambda d: f'{d["calls_per_hour"]:,.0f}'.replace(',', '{,}') if d['calls_per_hour'] else '0'),
             ('Live, complete recordings', lambda d: f'{100 * d["live_recall_complete"]:.1f}\\,\\%'),
             ('Stream recall (1\\,h)', lambda d: f'{100 * d["stream_recall"]:.1f}\\,\\%')]
    for name, f in extra:
        lines.append(f'    {name} & & ' + ' & '.join(f(D[k]) for k in RESULT_COLS) + ' \\\\')
    text = f"""% R2: the cascades against the requirements (generated by make_r2_latex.py). Needs: booktabs.
\\begin{{table*}}[t]
  \\centering
  \\caption{{Results on the test sets (thresholds fixed on validation, test run once). W0, A1, A2: mean over 3
  window-SNN seeds; B1, B2: mean over 9 cascades (3 fine-tuned verifiers $\\times$ 3 window-SNN seeds). S1: the
  streaming design's final test, reproduced exactly here; its memory is reported with that design. Power:
  busy-polling, as built (W0, A and B on the \\texttt{{ksdot}} fabric, S1 on the dual-core fabric).
  $\\surd$ met, $\\times$ missed.}}
  \\label{{tab:r2-results}}
  \\footnotesize
  \\setlength{{\\tabcolsep}}{{3.5pt}}
  \\begin{{tabular}}{{@{{}}ll{'c' * len(RESULT_COLS)}@{{}}}}
    \\toprule
    Requirement & Target & {head(RESULT_COLS)} \\\\
    \\midrule
{chr(10).join(lines[:len(REQ_ROWS)])}
    \\midrule
{chr(10).join(lines[len(REQ_ROWS):])}
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
"""
    write('tab_r2_results', text, 'tables')


def tab_cost(D):
    p = load_power()['ksdot']
    rows = []
    for k in RESULT_COLS:
        d = D[k]
        mateo = k == 'mateo_cascade'
        # W x ms = mJ; Mateo's from the FINAL REPORT (0.727 W x 105.83 ms)
        ev = 76.9 if mateo else ((p['pl_dyn'] + p['pl_static']) * T_V_MEAN if k != 'stage1_R1' else None)
        cells = [SHORT[k],
                 f'{d["t_max_ms"]:.1f}' if d['t_max_ms'] >= 1 else f'{d["t_max_ms"]:.2f}',
                 '---' if d.get('memory_kib') in (None, '') else f'{d["memory_kib"]:.1f}',
                 f'{d["calls_per_hour"]:,.0f}'.replace(',', '{,}'),
                 '---' if ev is None else f'{ev:.1f}',
                 f'{d["e_h_work_J"]:.0f}', f'{1000 * d["p_avg_busy_W"]:.0f}',
                 '---' if d.get('p_avg_gated_W') in (None, '') else f'{1000 * d["p_avg_gated_W"]:.0f}']
        rows.append('    ' + ' & '.join(cells) + ' \\\\')
    text = f"""% R2: cost breakdown (generated by make_r2_latex.py). Needs: booktabs.
\\begin{{table}}[t]
  \\centering
  \\caption{{Cost breakdown. $E_V$: PL energy per verifier call (total PL power $\\times$ 105.83\\,ms); work:
  14,400 stage-1 requests plus the verifier calls per hour, both at total PL power (the method of the
  parallel-processing design). Busy: average PL power as built. Gated: with clock gating, a calculation. W0, A
  and B: \\texttt{{ksdot}} fabric (0.367\\,W); S1: dual-core fabric (0.727\\,W, 7.40\\,mJ per stage-1 request).}}
  \\label{{tab:r2-cost}}
  \\footnotesize
  \\setlength{{\\tabcolsep}}{{3pt}}
  \\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
    \\toprule
    & $t_{{\\max}}$ & Memory & Calls & $E_V$ & Work & Busy & Gated \\\\
    Design & (ms) & (KiB) & per h & (mJ) & (J/h) & (mW) & (mW) \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    write('tab_r2_cost', text, 'tables')


def tab_finetune():
    import torch
    m = torch.load(R2 / 'mateo/sheila_verifier_head_near.pt', map_location='cpu', weights_only=False)['args']
    ours = json.loads((R2 / 'verifier/sheila_head_r2_s0_training.json').read_text())
    mining = json.loads((R2 / 'B/mining.json').read_text())
    same = [('Epochs $\\times$ steps', lambda a: f'{a["epochs"]} $\\times$ {a["steps_per_epoch"]}'),
            ('Learning rate (one-cycle)', lambda a: f'{a["lr"]:g}'),
            ('LibriSpeech per step', lambda a: f'{a["libri_seconds"]:.0f}\\,s'),
            ('Keyword batch (keyword / near-miss share)', lambda a: f'{a["kw_batch"]} ({a["kw_share"]:g} / {a["near_share"]:g})'),
            ('Hard windows per step', lambda a: f'{a["hard_batch"]}'),
            ('Loss weights (head / hard hinge / pos.\\ hinge)', lambda a: f'{a["head_weight"]:g} / {a["hard_weight"]:g} / {a["pos_weight"]:g}'),
            ('Real keyword clips only', lambda a: 'yes' if a['kw_real_only'] else 'no')]
    lines = [f'    {n} & {f(m)} & {f(ours)} \\\\' for n, f in same]
    lines += ['    \\midrule',
              "    Starting point & CTC base (\\texttt{sheila\\_logmel}) & the original verifier \\\\",
              f"    Hard negatives & 8{{,}}000 from the streaming SNN & {mining['union_kept']:,} from the 3 window-SNN seeds \\\\".replace(f"{mining['union_kept']:,}", f"{mining['union_kept']:,}".replace(',', '{,}')),
              '    Seeds & 1 & 3 \\\\',
              '    Hardware & RTX 4060 Laptop (CUDA) & RX 7800 XT (ROCm, no MIOpen) \\\\']
    text = f"""% R2: fine-tune settings next to the original verifier's (generated by make_r2_latex.py). Needs: booktabs.
\\begin{{table}}[t]
  \\centering
  \\caption{{Verifier fine-tune: the original verifier's training recipe, unchanged except for the starting point
  and the hard negatives (windows where the window SNN proposes ``Sheila'' in LibriSpeech train-clean-100).
  Settings read from the checkpoints.}}
  \\label{{tab:r2-finetune}}
  \\footnotesize
  \\setlength{{\\tabcolsep}}{{3pt}}
  \\begin{{tabular}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{3.1cm}}>{{\\raggedright\\arraybackslash}}p{{2.35cm}}>{{\\raggedright\\arraybackslash}}p{{2.35cm}}@{{}}}}
    \\toprule
    Setting & Original verifier (A, S1) & Fine-tuned verifier (B) \\\\
    \\midrule
{chr(10).join(lines)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    write('tab_r2_finetune', text, 'tables')


def main():
    D, fig, r1 = load()
    fig_quality(D)
    fig_curves(fig)
    fig_cost(D)
    fig_margin(D)
    fig_calls(D, fig, r1)
    tab_results(D)
    tab_cost(D)
    tab_finetune()


if __name__ == '__main__':
    main()
