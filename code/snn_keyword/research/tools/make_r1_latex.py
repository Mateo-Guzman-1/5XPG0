"""R1 figures and tables as self-contained LaTeX (pgfplots with inline coordinates, booktabs).

Reads results/R1/{designs.csv, choice.json, references.json, requirements/*.json}.
Writes reports/latex/requirements_eval/{figures,tables}/*.tex. Every figure uses the styles of
figures/r1_style.tex; the group report needs that one \\input in its preamble.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / 'results/R1'
OUT = ROOT / 'reports/latex/requirements_eval'
FAMS = ('dense', 'bits', 'pruned', 'pruned_bits')
FAM_STYLE = {'dense': 'r1dense', 'bits': 'r1bits', 'pruned': 'r1pruned', 'pruned_bits': 'r1prunedbits'}
FAM_LABEL = {'dense': 'dense, 16-bit', 'bits': 'dense, 8/4-bit', 'pruned': 'pruned, 16-bit', 'pruned_bits': 'pruned, 8-bit'}
HW_LABEL = {'base': 'RV32IM', 'kdot': '\\texttt{kdot}', 'kdot_lut': '\\texttt{kdot}+table', 'packed': 'RV32IM packed',
            'sparse': 'RV32IM sparse', 'ksdot_lut': '\\texttt{ksdot}+table'}
TITLE = 'title style={font=\\footnotesize, at={(0.02,0.97)}, anchor=north west}'


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load():
    designs = [{k: (num(v) if num(v) is not None else v) for k, v in r.items()}
               for r in csv.DictReader(open(R1 / 'designs.csv', encoding='utf-8'))]
    choice = json.loads((R1 / 'choice.json').read_text())
    refs = json.loads((R1 / 'references.json').read_text())['designs']
    curves = {}
    for p in sorted((R1 / 'requirements').glob('*.json')):
        rep = json.loads(p.read_text())
        curves[rep['config']] = rep['models']
    return designs, choice, refs, curves


def short(config):
    """h48_b8_k64 -> H48 8b K64."""
    out = []
    for part in config.split('_'):
        out.append({'h': 'H', 'b': '', 'k': 'K'}[part[0]] + part[1:] + ('b' if part[0] == 'b' else ''))
    return '\\,'.join(out)


def per_config(designs):
    """One row per config: quality (identical across hardware variants) and its most efficient measured hardware."""
    rows = {}
    for d in designs:
        if d['config'] not in rows or d['e_inf_uJ'] < rows[d['config']]['e_inf_uJ']:
            rows[d['config']] = d
    order = ['h16', 'h32', 'h48', 'h95', 'h48_b8', 'h48_b4', 'h48_k512', 'h48_k256', 'h48_k128', 'h48_k64',
             'h48_b8_k256', 'h48_b8_k128', 'h48_b8_k64']
    return [rows[c] for c in order if c in rows]


def coords(points):
    return ' '.join(f'({x:.6g},{y:.6g})' if e is None else f'({x:.6g},{y:.6g}) +- (0,{e:.6g})' for x, y, e in points)


def write(name, text, kind='figures'):
    (OUT / kind).mkdir(parents=True, exist_ok=True)
    (OUT / kind / f'{name}.tex').write_text(text, encoding='utf-8')
    print('wrote', kind, name)


def champions(choice):
    """Dense and pruned champion, in that order (both are shown; the verdict says which is recommended)."""
    m = choice['main']
    if 'winner' in m:
        return [m['winner']]
    return [m['dense']['champion'], m['pruned']['champion']]


def design(designs, name):
    return next(d for d in designs if d['design'] == name)


def callout(x, y, tx, ty, text, anchor='west', color='black'):
    return (f'    \\draw[{color}, thin] (axis cs:{x:.4g},{y:.4g}) -- (axis cs:{tx:.4g},{ty:.4g}) '
            f'node[font=\\scriptsize, anchor={anchor}, inner sep=1pt, text={color}] {{{text}}};')


# ----------------------------------------------------------------------------- figures

def fig_requirements(designs, refs, choice):
    """(a) live recall vs other words accepted, (b) live recall vs false accepts per hour."""
    cfg = per_config(designs)
    D, P = [design(designs, c) for c in champions(choice)]
    pa, pb = [], []
    for fam in FAMS:
        rs = [d for d in cfg if d['family'] == fam]
        a = coords([(100 * d['live_other'], 100 * d['live_recall'], 100 * d['live_recall_sd']) for d in rs])
        b = coords([(d['fa_per_hour'], 100 * d['live_recall'], 100 * d['live_recall_sd']) for d in rs])
        pa.append(f'    \\addplot[{FAM_STYLE[fam]}, r1err] coordinates {{{a}}};\n    \\addlegendentry{{{FAM_LABEL[fam]}}}')
        pb.append(f'    \\addplot[{FAM_STYLE[fam]}, r1err] coordinates {{{b}}};')
    sat_a = coords([(100 * d['s_rule_live_other'], 100 * d['s_rule_live_recall'], None) for d in cfg])
    sat_b = coords([(d['s_rule_fa_per_hour'], 100 * d['s_rule_live_recall'], None) for d in cfg])
    s1, casc, dam, dam_own = refs
    xa = lambda r: min(100 * r['live_other'], 0.98)
    ref_a = coords([(xa(r), 100 * r['live_recall'], None) for r in refs])
    ref_b = coords([(max(r['fa_per_hour'], 0.06), 100 * r['live_recall'], None) for r in refs])
    lab_a = '\n'.join([
        callout(xa(casc), 100 * casc['live_recall'], 0.32, 94, 'Mateo: SNN + GRU cascade', color='r1ref'),
        callout(xa(s1), 100 * s1['live_recall'], 0.12, 72, 'Mateo: streaming SNN', color='r1ref'),
        callout(xa(dam), 100 * dam['live_recall'], 0.12, 12, "Damien, Mateo's rule", color='r1ref'),
        f'    \\node[font=\\scriptsize, text=r1ref, anchor=east] at (axis cs:0.97,{100 * dam_own["live_recall"]:.4g}) '
        f'{{Damien, own margin ({100 * dam_own["live_other"]:.1f}\\,\\%)\\,$\\rightarrow$}};',
        callout(100 * D['live_other'], 100 * D['live_recall'], 0.24, 53, f'{short(D["config"])}, dense'),
        callout(100 * P['live_other'], 100 * P['live_recall'], 0.2, 22, f'{short(P["config"])}, pruned'),
        f'    \\node[font=\\scriptsize, text=black!55, anchor=west] at (axis cs:0.43,63) {{S-study threshold}};'])
    lab_b = '\n'.join([
        callout(casc['fa_per_hour'], 100 * casc['live_recall'], 0.12, 94, 'Mateo: cascade', anchor='west', color='r1ref'),
        callout(s1['fa_per_hour'], 100 * s1['live_recall'], 0.12, 72, 'Mateo: SNN', anchor='west', color='r1ref'),
        callout(dam['fa_per_hour'], 100 * dam['live_recall'], 0.12, 12, 'Damien', anchor='west', color='r1ref'),
        callout(dam_own['fa_per_hour'], 100 * dam_own['live_recall'], 60, 92, 'Damien, own', anchor='west', color='r1ref'),
        callout(D['fa_per_hour'], 100 * D['live_recall'], 0.12, 50, short(D['config'])),
        callout(P['fa_per_hour'], 100 * P['live_recall'], 0.12, 30, short(P['config'])),
        f'    \\node[font=\\scriptsize, text=black!55, anchor=north] at (axis cs:20,54) {{S-study threshold}};'])
    text = f"""% R1: our designs against the detection requirements (generated by make_r1_latex.py).
\\begin{{figure*}}[t]
  \\centering
  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, width=0.49\\textwidth, height=6.4cm, xmin=0, xmax=1, ymin=0, ymax=100,
      xlabel={{Other words accepted (\\%)}}, ylabel={{Live ``Sheila'' detected (\\%)}},
      legend pos=south east, title={{(a)}}, {TITLE}]
    \\fill[r1pass] (axis cs:0,85) rectangle (axis cs:0.2,100);
    \\draw[r1req] (axis cs:0.2,0) -- (axis cs:0.2,100);
    \\draw[r1req] (axis cs:0,85) -- (axis cs:1,85);
    \\node[r1reqlabel, anchor=south west] at (axis cs:0.205,1) {{$\\leq$0.2\\,\\%}};
    \\node[r1reqlabel, anchor=south east] at (axis cs:0.99,85.5) {{$\\geq$85\\,\\%}};
{chr(10).join(pa)}
    \\addplot[r1old] coordinates {{{sat_a}}};
    \\addlegendentry{{same models, S-study threshold}}
    \\addplot[r1ref] coordinates {{{ref_a}}};
    \\addlegendentry{{teammates' designs}}
{lab_a}
  \\end{{axis}}
  \\end{{tikzpicture}}\\hfill
  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, width=0.49\\textwidth, height=6.4cm, xmode=log, xmin=0.05, xmax=1000, ymin=0, ymax=100,
      xlabel={{False accepts per hour}}, ylabel={{Live ``Sheila'' detected (\\%)}}, title={{(b)}}, {TITLE}]
    \\fill[r1pass] (axis cs:0.05,85) rectangle (axis cs:2,100);
    \\draw[r1req] (axis cs:2,0) -- (axis cs:2,100);
    \\draw[r1req] (axis cs:0.05,85) -- (axis cs:1000,85);
    \\node[r1reqlabel, anchor=north west] at (axis cs:2.1,99) {{$\\leq$2 per hour}};
{chr(10).join(pb)}
    \\addplot[r1old] coordinates {{{sat_b}}};
    \\addplot[r1ref] coordinates {{{ref_b}}};
{lab_b}
  \\end{{axis}}
  \\end{{tikzpicture}}
  \\caption{{Detection quality against the requirements; the shaded corner meets both. Filled markers: our
  designs at the threshold chosen on validation with Mateo's rule (mean $\\pm$ sd over 3 seeds). Hollow
  markers: the same models at the earlier S-study threshold. Crosses: teammates' designs on the same test
  sets. Damien's model at its own margin lies at 3.5\\,\\% other words, off the scale in (a).}}
  \\label{{fig:r1-req-quality}}
\\end{{figure*}}
"""
    write('fig_req_quality', text)


GROUPS = [('RV32IM (plain, packed or sparse loop)', ('base', 'packed', 'sparse'), 'r1hwrv'),
          ('\\texttt{kdot}', ('kdot',), 'r1hwkdot'),
          ('\\texttt{kdot} + LIF table', ('kdot_lut',), 'r1hwkdotlut'),
          ('\\texttt{ksdot} + LIF table', ('ksdot_lut',), 'r1hwksdot')]


def fig_cost(designs):
    """(a) worst compute per request, (b) energy per inference, (c) memory: every config x hardware variant."""
    cfg = [d['config'] for d in per_config(designs)]
    xt = ','.join(str(i) for i in range(len(cfg)))
    xl = ','.join('{' + short(c) + '}' for c in cfg)
    n = len(cfg) - 0.5

    def series(key, scale=1., legend=False):
        out = []
        for label, vs, style in GROUPS:
            pts = [(cfg.index(d['config']), scale * d[key], None) for d in designs if d['variant'] in vs and d['config'] in cfg]
            out.append(f'    \\addplot[{style}] coordinates {{{coords(pts)}}};' + (f'\n    \\addlegendentry{{{label}}}' if legend else ''))
        return '\n'.join(out)
    common = (f'scale only axis, width=0.245\\textwidth, height=4.2cm, xmin=-0.6, xmax={n + 0.1}, xtick={{{xt}}}, '
              f'xticklabels={{{xl}}}, x tick label style={{rotate=65, anchor=east, font=\\tiny}}')

    def panel(label, ylabel, axis_extra, req, req_text, body):
        return f"""  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, {common}, {axis_extra}, ylabel={{{ylabel}}}, title={{({label})}}, {TITLE}]
    \\draw[r1req] (axis cs:-0.6,{req}) -- (axis cs:{n + 0.1},{req});
    \\node[r1reqlabel, anchor=north east] at (axis cs:{n},{req * 0.93 if 'log' in axis_extra else req - 3}) {{{req_text}}};
{body}
  \\end{{axis}}
  \\end{{tikzpicture}}"""
    a = panel('a', 'Worst compute per request (ms)', 'ymode=log, ymin=0.08, ymax=700, legend to name=r1costlegend, legend columns=4',
              200, '$\\leq$200\\,ms', series('t_max_ms', legend=True))
    b = panel('b', 'Energy per inference (mJ)', 'ymode=log, ymin=0.02, ymax=900', 250, '$\\leq$250\\,mJ (1\\,W $\\times$ 250\\,ms)',
              series('e_inf_uJ', 1e-3))
    c = panel('c', 'Model memory (KiB)', 'ymin=0, ymax=165', 144, '$\\leq$144\\,KiB', series('memory_kib'))
    text = f"""% R1: cost requirements per design (generated by make_r1_latex.py).
\\begin{{figure*}}[t]
  \\centering
{a}\\hfill
{b}\\hfill
{c}\\\\[3pt]
  \\pgfplotslegendfromname{{r1costlegend}}
  \\caption{{Cost requirements for every design and hardware (mean of 3 seeds; cycles from bit-exact board
  runs, energy from Vivado power estimates). Dashed: the requirement. Every design meets all three; the
  energy budget is met by a factor of 10 or more.}}
  \\label{{fig:r1-req-cost}}
\\end{{figure*}}
"""
    write('fig_req_cost', text)


def fig_power(designs):
    """Average PL power while listening, three operating scenarios, against 1 W."""
    keep = [('h95', 'base'), ('h48', 'base'), ('h48_k64', 'sparse'), ('h95', 'kdot_lut'), ('h48_b8', 'kdot_lut'),
            ('h16', 'kdot_lut'), ('h48_b8_k128', 'ksdot_lut'), ('h48_k64', 'ksdot_lut')]
    show = [design(designs, f'{c}/{v}') for c, v in keep]
    names = [f'{short(d["config"])}, {HW_LABEL[d["variant"]]}' for d in show]
    yt = ','.join(str(i) for i in range(len(show)))
    yl = ','.join('{' + n + '}' for n in names)
    bars = []
    for key, style, lab in (('p_avg_busy_W', 'fill=black!60', 'busy-polling (as built)'),
                            ('p_avg_gated_W', 'fill=r1dense!75', 'clock-gated idle'),
                            ('p_avg_scaled_W', 'fill=r1prunedbits!70', 'clock scaled to the deadline')):
        c = ' '.join(f'({d[key]:.4g},{i})' for i, d in enumerate(show))
        bars.append(f'    \\addplot[draw=none, {style}] coordinates {{{c}}};\n    \\addlegendentry{{{lab}}}')
    top = len(show) - 0.5
    text = f"""% R1: average PL power while listening (generated by make_r1_latex.py).
\\begin{{figure}}[t]
  \\centering
  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, scale only axis, width=0.6\\columnwidth, height=5.4cm, xbar, bar width=2.6pt, xmin=0, xmax=1.1, ymin=-0.5, ymax={top},
      ytick={{{yt}}}, yticklabels={{{yl}}}, y tick label style={{font=\\tiny}}, enlarge y limits=false,
      xlabel={{Average PL power while listening (W)}}, reverse legend,
      legend style={{at={{(0.5,1.03)}}, anchor=south}}, legend columns=3]
    \\draw[r1req] (axis cs:1,-0.5) -- (axis cs:1,{top});
    \\node[r1reqlabel, anchor=north east] at (axis cs:0.99,{top - 0.05}) {{$\\leq$1\\,W}};
{chr(10).join(bars)}
  \\end{{axis}}
  \\end{{tikzpicture}}
  \\caption{{Average power of the programmable logic at one window per 250\\,ms. Busy-polling is the firmware
  as built; clock gating and clock scaling are calculated. Every design is far below the 1\\,W budget: the
  static power and the block RAM, not the network, set the power.}}
  \\label{{fig:r1-req-power}}
\\end{{figure}}
"""
    write('fig_req_power', text)


def fig_sweeps(designs):
    """Live detection at the requirement threshold vs H, bits and fan-in K (S-study style)."""
    cfg = {d['config']: d for d in per_config(designs)}
    def pts(cs, x):
        return coords([(x(c), 100 * cfg[c]['live_recall'], 100 * cfg[c]['live_recall_sd']) for c in cs if c in cfg])
    size = pts(['h16', 'h32', 'h48', 'h95'], lambda c: int(c[1:]))
    bits = pts(['h48_b4', 'h48_b8', 'h48'], lambda c: 16 if c == 'h48' else int(c.split('_b')[1]))
    k16 = pts(['h48_k64', 'h48_k128', 'h48_k256', 'h48_k512', 'h48'], lambda c: 768 if c == 'h48' else int(c.split('_k')[1]))
    k8 = pts(['h48_b8_k64', 'h48_b8_k128', 'h48_b8_k256', 'h48_b8'], lambda c: 768 if c == 'h48_b8' else int(c.split('_k')[1]))

    def panel(label, xlabel, extra, body):
        return f"""  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, scale only axis, width=0.245\\textwidth, height=3.6cm, ymin=0, ymax=100, {extra},
      xlabel={{{xlabel}}}, ylabel={{Live ``Sheila'' detected (\\%)}}, title={{({label})}}, {TITLE}]
    \\draw[r1req] (rel axis cs:0,0.85) -- (rel axis cs:1,0.85);
    \\node[r1reqlabel, anchor=south east] at (rel axis cs:0.98,0.855) {{requirement $\\geq$85\\,\\%}};
{body}
  \\end{{axis}}
  \\end{{tikzpicture}}"""
    a = panel('a', 'Hidden neurons $H$', 'xmin=8, xmax=103, xtick={16,32,48,95}',
              f'    \\addplot[r1denseL, r1err] coordinates {{{size}}};')
    b = panel('b', 'Bits per input synapse ($H=48$)', 'xmin=2, xmax=18, xtick={4,8,16}',
              f'    \\addplot[r1bitsL, r1err] coordinates {{{bits}}};')
    c = panel('c', 'Synapses kept per neuron $K$ ($H=48$)',
              'xmode=log, log basis x=2, xmin=50, xmax=1000, xtick={64,128,256,512,768}, '
              'xticklabels={64,128,256,512,768}, legend pos=south east',
              f'    \\addplot[r1prunedL, r1err] coordinates {{{k16}}};\n    \\addlegendentry{{16-bit}}\n'
              f'    \\addplot[r1prunedbitsL, r1err] coordinates {{{k8}}};\n    \\addlegendentry{{8-bit}}')
    text = f"""% R1: live detection at the requirement threshold vs the three size levers (generated by make_r1_latex.py).
\\begin{{figure*}}[t]
  \\centering
{a}\\hfill
{b}\\hfill
{c}
  \\caption{{Live detection at the requirement threshold against (a) the number of neurons, (b) the bits per
  input synapse and (c) the synapses kept per neuron ($K=768$: dense). Mean $\\pm$ sd over 3 seeds. Above 16
  neurons no lever changes the detection by more than the seed spread, and none comes near 85\\,\\%.}}
  \\label{{fig:r1-sweeps}}
\\end{{figure*}}
"""
    write('fig_sweeps', text)


def fig_best(designs, choice, refs, curves):
    """Operating curves of the two champions (test threshold sweep) with the requirement lines."""
    body_a, body_b = [], []
    casc = next(r for r in refs if 'cascade' in r['name'])
    for name, style in zip(champions(choice), ('r1dense', 'r1pruned')):
        d = design(designs, name)
        seeds = curves[d['config']]
        seed = min(seeds, key=lambda n: abs(seeds[n]['test']['live_recall'] - d['live_recall']))   # a real model near the mean
        cur, t = seeds[seed]['test_curve'], seeds[seed]['test']
        ca = ' '.join(f'({100 * c["live_other"]:.4g},{100 * c["live_recall"]:.4g})' for c in cur)
        cb = ' '.join(f'({max(c["fa_per_hour"], .05):.4g},{100 * c["live_recall"]:.4g})' for c in cur)
        lab = f'{short(d["config"])}, {HW_LABEL[d["variant"]]} (seed {seed[-1]})'
        body_a.append(f'    \\addplot[{style}L, mark=none, thick] coordinates {{{ca}}};\n    \\addlegendentry{{{lab}}}\n'
                      f'    \\addplot[{style}, r1chosen, forget plot] coordinates {{({100 * t["live_other"]:.4g},{100 * t["live_recall"]:.4g})}};')
        body_b.append(f'    \\addplot[{style}L, mark=none, thick] coordinates {{{cb}}};\n'
                      f'    \\addplot[{style}, r1chosen] coordinates {{({max(t["fa_per_hour"], .05):.4g},{100 * t["live_recall"]:.4g})}};')
    text = f"""% R1: operating curves of the two chosen designs (generated by make_r1_latex.py).
\\begin{{figure*}}[t]
  \\centering
  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, width=0.49\\textwidth, height=6.0cm, xmin=0, xmax=2, ymin=20, ymax=100,
      xlabel={{Other words accepted (\\%)}}, ylabel={{Live ``Sheila'' detected (\\%)}}, legend pos=south east,
      title={{(a)}}, {TITLE}]
    \\fill[r1pass] (axis cs:0,85) rectangle (axis cs:0.2,100);
    \\draw[r1req] (axis cs:0.2,20) -- (axis cs:0.2,100);
    \\draw[r1req] (axis cs:0,85) -- (axis cs:2,85);
    \\node[r1reqlabel, anchor=north west] at (axis cs:0.21,99) {{$\\leq$0.2\\,\\%}};
    \\node[r1reqlabel, anchor=south east] at (axis cs:1.98,85.5) {{$\\geq$85\\,\\%}};
{chr(10).join(body_a)}
    \\addplot[r1ref] coordinates {{({100 * casc['live_other']:.4g},{100 * casc['live_recall']:.4g})}};
    \\addlegendentry{{Mateo's cascade (SNN + GRU)}}
  \\end{{axis}}
  \\end{{tikzpicture}}\\hfill
  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, width=0.49\\textwidth, height=6.0cm, xmode=log, xmin=0.3, xmax=300, ymin=20, ymax=100,
      xlabel={{False accepts per hour}}, ylabel={{Live ``Sheila'' detected (\\%)}}, title={{(b)}}, {TITLE}]
    \\fill[r1pass] (axis cs:0.3,85) rectangle (axis cs:2,100);
    \\draw[r1req] (axis cs:2,20) -- (axis cs:2,100);
    \\draw[r1req] (axis cs:0.3,85) -- (axis cs:300,85);
    \\node[r1reqlabel, anchor=north west] at (axis cs:2.1,99) {{$\\leq$2 per hour}};
{chr(10).join(body_b)}
    \\addplot[r1ref] coordinates {{({casc['fa_per_hour']:.4g},{100 * casc['live_recall']:.4g})}};
  \\end{{axis}}
  \\end{{tikzpicture}}
  \\caption{{What the two chosen designs can trade (one seed each, the one closest to the 3-seed mean). Lines:
  the test result at every threshold; large markers: the threshold fixed on validation. Even at 2\\,\\% other
  words (ten times the limit) and about 70 false accepts per hour, live detection stays below 80\\,\\%: these
  one-window networks cannot reach 85\\,\\% at any threshold. Mateo's cascade meets both requirements.}}
  \\label{{fig:r1-best-curves}}
\\end{{figure*}}
"""
    write('fig_best_curves', text)


# key, row label, direction, target, value format, target text
REQ_ROWS = [('live_recall', "Live ``Sheila'' detected", '>=', .85, lambda v: f'{100 * v:.1f}\\,\\%', '$\\geq$85\\,\\%'),
            ('fa_per_hour', 'False accepts per hour', '<=', 2., lambda v: f'{v:.2f}', '$\\leq$2'),
            ('live_other', 'Other words accepted', '<=', .002, lambda v: f'{100 * v:.2f}\\,\\%', '$\\leq$0.2\\,\\%'),
            ('latency_ms', 'Word end to decision', '<=', 500., lambda v: f'{v:.0f}\\,ms', '$\\leq$500\\,ms'),
            ('t_max_ms', 'Compute per request', '<=', 200., lambda v: f'{v:.2f}\\,ms', '$\\leq$200\\,ms'),
            ('memory_kib', 'Model memory', '<=', 144., lambda v: f'{v:.1f}\\,KiB', '$\\leq$144\\,KiB'),
            ('p_avg_busy_W', 'Average PL power', '<=', 1., lambda v: f'{1000 * v:.0f}\\,mW', '$\\leq$1\\,W')]


def fig_margin(designs, choice):
    """Every requirement as a factor (>1: met by that factor; <1: missed), for the two champions."""
    rows = list(reversed(REQ_ROWS))
    yt = ','.join(str(i) for i in range(len(rows)))
    yl = ','.join('{' + r[1] + '}' for r in rows)
    bars, labels = [], []
    for j, (name, style) in enumerate(zip(champions(choice), ('r1dense', 'r1pruned'))):
        d = design(designs, name)
        off = -0.19 if j == 0 else 0.19   # pgfplots shifts the first xbar plot down
        pts = []
        for i, (key, _, op, target, fmt, _) in enumerate(rows):
            v = d[key]
            r = v / target if op == '>=' else (target / v if v > 0 else 1e4)
            pts.append(f'({r:.4g},{i})')
            anchor, x = ('west', r * 1.08) if r >= 1 else ('east', r / 1.08)
            labels.append(f'    \\node[font=\\tiny, anchor={anchor}, inner sep=0.5pt] at (axis cs:{x:.4g},{i + off:.3g}) {{{fmt(v)}}};')
        bars.append(f'    \\addplot[draw=none, fill={style}!80] coordinates {{{" ".join(pts)}}};\n'
                    f'    \\addlegendentry{{{short(d["config"])}, {HW_LABEL[d["variant"]]}}}')
    top = len(rows) - 0.5
    text = f"""% R1: by what factor each chosen design meets (>1) or misses (<1) every requirement (generated by make_r1_latex.py).
\\begin{{figure}}[t]
  \\centering
  \\begin{{tikzpicture}}
  \\begin{{axis}}[r1axis, scale only axis, width=0.62\\columnwidth, height=5.8cm, xbar, bar width=4.2pt, xmode=log, log origin=0, xmin=0.15, xmax=3e4,
      ymin=-0.5, ymax={top}, ytick={{{yt}}}, yticklabels={{{yl}}}, y tick label style={{font=\\scriptsize}},
      enlarge y limits=false, xlabel={{Factor by which the requirement is met}}, legend pos=north east, reverse legend]
    \\fill[black!7] (axis cs:0.15,-0.5) rectangle (axis cs:1,{top});
    \\draw[r1req] (axis cs:1,-0.5) -- (axis cs:1,{top});
{chr(10).join(bars)}
{chr(10).join(labels)}
  \\end{{axis}}
  \\end{{tikzpicture}}
  \\caption{{The two chosen designs against every requirement, as the factor by which each is met
  (value/target for live detection, target/value otherwise); numbers are the measured values. Both meet the
  five cost and latency requirements by a wide margin and miss live detection and false accepts per hour.}}
  \\label{{fig:r1-best-margin}}
\\end{{figure}}
"""
    write('fig_best_margin', text)


# ----------------------------------------------------------------------------- tables

def mark(ok):
    return '$\\surd$' if ok else '$\\times$'


def tab_designs(designs):
    lines, prev = [], None
    for d in per_config(designs):
        if prev in ('dense', 'bits') and d['family'] in ('pruned', 'pruned_bits'):
            lines.append('    \\midrule')
        prev = d['family']
        ok = [d['live_recall'] >= .85, d['fa_per_hour'] <= 2, d['live_other'] <= .002, d['latency_ms'] <= 500,
              d['t_max_ms'] <= 200, d['memory_kib'] <= 144, d['p_avg_busy_W'] <= 1]
        cells = [short(d['config']), f'{d["test_f1"]:.3f}', f'{d["test_f1_subset"]:.3f}',
                 f'{100 * d["live_recall"]:.1f}\\,$\\pm$\\,{100 * d["live_recall_sd"]:.1f}',
                 f'{100 * d["live_other"]:.2f}', f'{d["fa_per_hour"]:.2f}', f'{d["latency_ms"]:.0f}',
                 HW_LABEL[d['variant']], f'{d["t_max_ms"]:.2f}', f'{d["memory_kib"]:.1f}', f'{d["e_inf_uJ"]:.0f}',
                 f'{1000 * d["p_avg_busy_W"]:.0f}', f'{sum(ok)}/7']
        lines.append('    ' + ' & '.join(cells) + ' \\\\')
    text = f"""% R1: every design against the requirements (generated by make_r1_latex.py). Needs: booktabs.
\\begin{{table*}}[t]
  \\centering
  \\caption{{All designs at the threshold chosen on validation (mean of 3 seeds). Test F1 is the clip-level F1
  on the full Speech Commands test split and on Damien's 3,212-clip subset. The cost columns give the most
  efficient hardware measured for each design. Met: number of the seven requirements met.}}
  \\label{{tab:r1-designs}}
  \\footnotesize
  \\setlength{{\\tabcolsep}}{{3.4pt}}
  \\begin{{tabular}}{{@{{}}lrrrrrrlrrrrr@{{}}}}
    \\toprule
    & \\multicolumn{{2}}{{c}}{{Test F1}} & Live & Other & FA & Latency & & $t_{{\\max}}$ & Memory & $E_{{\\mathrm{{inf}}}}$ & $P_{{\\mathrm{{PL}}}}$ & \\\\
    \\cmidrule(lr){{2-3}}
    Design & full & subset & (\\%) & (\\%) & per h & (ms) & Hardware & (ms) & (KiB) & ($\\mu$J) & (mW) & Met \\\\
    \\midrule
    Requirement & & & $\\geq$85 & $\\leq$0.2 & $\\leq$2 & $\\leq$500 & & $\\leq$200 & $\\leq$144 & & $\\leq$1000 & \\\\
    \\midrule
{chr(10).join(lines)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
"""
    write('tab_designs', text, 'tables')


def tab_best(designs, choice, refs):
    ds = [design(designs, c) for c in champions(choice)]
    s1 = next(r for r in refs if 'stage 1' in r['name'])
    casc = next(r for r in refs if 'cascade' in r['name'])
    verdict = choice['main']['verdict']
    heads = [f'{short(d["config"])}\\\\{HW_LABEL[d["variant"]]}' for d in ds]
    heads[-1] += '\\\\(recommended)' if verdict == 'pruned champion alone' else ''
    head = ' & '.join('\\begin{tabular}[b]{@{}c@{}}' + h + '\\end{tabular}' for h in heads)
    lines = []
    for key, name, op, target, fmt, tgt in REQ_ROWS:
        vals = []
        for v in [d[key] for d in ds] + [s1.get(key), casc.get(key)]:
            if v is None:
                vals.append('---')
            else:
                vals.append(f'{fmt(v)} {mark(v >= target if op == ">=" else v <= target)}')
        lines.append(f'    {name} & {tgt} & ' + ' & '.join(vals) + ' \\\\')
    text = f"""% R1: chosen designs against the requirements and Mateo's designs (generated by make_r1_latex.py).
\\begin{{table*}}[t]
  \\centering
  \\caption{{The two chosen designs against the requirements, next to Mateo's designs on the same test sets
  (Mateo's values from his final test report; compute from his README; his memory and power are reported in his
  own sections). $\\surd$ met, $\\times$ missed.}}
  \\label{{tab:r1-best}}
  \\footnotesize
  \\begin{{tabular}}{{@{{}}ll{'c' * len(ds)}cc@{{}}}}
    \\toprule
    Requirement & Target & {head} & \\begin{{tabular}}[b]{{@{{}}c@{{}}}}Mateo\\\\streaming SNN\\end{{tabular}} & \\begin{{tabular}}[b]{{@{{}}c@{{}}}}Mateo\\\\SNN + GRU\\end{{tabular}} \\\\
    \\midrule
{chr(10).join(lines)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
"""
    write('tab_best', text, 'tables')


def main():
    designs, choice, refs, curves = load()
    fig_requirements(designs, refs, choice)
    fig_cost(designs)
    fig_power(designs)
    fig_sweeps(designs)
    fig_best(designs, choice, refs, curves)
    fig_margin(designs, choice)
    tab_designs(designs)
    tab_best(designs, choice, refs)


if __name__ == '__main__':
    main()
