"""R3 mini reports as LaTeX (research/PLAN_R3_GATING.md, Phase 1 step 6 and Phase 3).

    make_r3_latex.py plan      -> reports/latex/clock_gating_plan/ (main.tex, figures/, .gitignore)

The plan report is one paragraph and one figure: (a) the idea as a TikZ sketch, (b) PL power over 1 s
(4 requests) for busy-polling and for the gated design. Every number comes from
results/R3/phase1/hypothesis.json (pre-registered) and results/R3/phase1/feasibility.json (F2, F3).
Same class and preamble as reports/latex/cascade_gru/main.tex (the group report's packages: pgfplots,
TikZ arrows.meta and fit; no fillbetween); figures/r1_style.tex is copied from that report.
Design codes only, no personal names (B1, B2; "the streaming design").
Build: cd reports/latex/clock_gating_plan; latexmk -pdf main.tex
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # research/
P1 = ROOT / 'results/R3/phase1'
R2TEX = ROOT / 'reports/latex/cascade_gru'
F_CLK = 100e6

PREAMBLE = r"""\documentclass[lettersize,journal]{IEEEtran}
\usepackage{amsmath,amsfonts}
\usepackage{array}
\usepackage{textcomp}
\usepackage{stfloats}
\usepackage{url}
\usepackage{graphicx}
\usepackage{cite}
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,fit}
\input{figures/r1_style}
"""
GITIGNORE = R2TEX / '.gitignore'


def fmt(x, nd=0):
    s = f'{x:,.{nd}f}'
    return s.replace('-', '$-$')


def load():
    h = json.loads((P1 / 'hypothesis.json').read_text(encoding='utf-8'))
    f = json.loads((P1 / 'feasibility.json').read_text(encoding='utf-8'))
    return h, f


def fig_idea(h, f):
    p_idle, p_busy = h['p_idle_W'], h['p_busy_W']
    dp = h['hypotheses']['H4']['central_prior']['dp_v_W']
    p_awake = p_idle + dp
    t_v = f['F3']['t_v_ms']['mean']
    s1_ms = 6.0                                     # stage-1 bursts (0.35 ms) drawn 6 ms wide to be visible
    # Gated trace: requests at 0, 250, 500, 750 ms; the verifier runs after stage 1 of the request at 250 ms.
    pts, t = [], 0.0
    for k in range(4):
        t0 = 250.0 * k
        pts += [(t0, p_awake)]
        end = t0 + s1_ms + (t_v if k == 1 else 0.0)
        pts += [(end, p_idle)]
    pts.append((1000.0, p_idle))
    gated = ' '.join(f'({x:.2f},{y:.4f})' for x, y in pts)
    return rf"""% R3 plan: idea and power trace (TikZ + pgfplots). Needs: pgfplots, arrows.meta, figures/r1_style.tex.
\begin{{figure}}[!t]
  \centering
  \begin{{tikzpicture}}[font=\sffamily\scriptsize,
      blk/.style={{draw,rounded corners=1pt,align=center,inner sep=2pt,minimum height=0.75cm,fill=white}},
      arr/.style={{-{{Stealth[length=1.4mm]}},thick}},
      al/.style={{font=\sffamily\tiny,inner sep=1.5pt,align=center}}]
    \node[blk,minimum width=1.55cm] (ps) at (0,0) {{ARM (PS)\\AXI master}};
    \node[blk,minimum width=1.45cm] (db) at (2.75,0) {{doorbell\\(sticky)}};
    \node[blk,minimum width=1.45cm] (fsm) at (5.6,0) {{sleep\\FSM}};
    \node[blk,minimum width=1.55cm] (fclk) at (0,-1.55) {{FCLK0\\100\,MHz}};
    \node[blk,minimum width=1.45cm] (bg) at (2.75,-1.55) {{BUFGCE}};
    \node[blk,minimum width=1.45cm] (cpu) at (5.6,-1.55) {{PicoRV32\\+ \texttt{{kdot}}}};
    \node[blk,minimum width=3.4cm] (bram) at (2.75,-3.05) {{block RAM, 256\,KiB (64 RAMB36)\\0.250\,W of 0.282\,W, as built}};
    \draw[arr] (ps) -- node[al,above]{{write MB[0]}} (db);
    \draw[arr] (db) -- node[al,above]{{wake}} (fsm);
    \draw[arr] ([xshift=0.3cm]cpu.north) -- node[al,right]{{store to\\POWER}} ([xshift=0.3cm]fsm.south);
    \draw[arr] ([xshift=-0.3cm]fsm.south) |- ++(0,-0.4) -| node[al,above,pos=0.25]{{CE}} (bg.north);
    \draw[arr] (fclk) -- (bg);
    \draw[arr] (bg) -- node[al,above]{{gated clock}} (cpu);
    \draw[arr,{{Stealth[length=1.4mm]}}-{{Stealth[length=1.4mm]}}] (ps.west) -- ++(-0.3,0) |- node[al,above,pos=0.78]{{port A:\\PS reads only}} (bram.west);
    \draw[arr,{{Stealth[length=1.4mm]}}-{{Stealth[length=1.4mm]}}] (cpu.south) |- (bram.east);
    \node[al,anchor=west,align=left] at (5.7,-2.45) {{port B: \texttt{{kdot}}\\or CPU reads only}};
    \node[font=\sffamily\scriptsize\bfseries,anchor=north west] at (-1.25,0.75) {{(a)}};
  \end{{tikzpicture}}

  \vspace{{4mm}}
  \begin{{tikzpicture}}
    \begin{{axis}}[r1axis, height=4.6cm, xmin=0, xmax=1000, ymin=0, ymax=0.54,
        xlabel={{time (ms)}}, ylabel={{PL power (W)}}, xtick={{0,250,500,750,1000}},
        ytick={{0,0.1,0.2,0.3,0.4}}, yticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=1}},
        legend pos=north east, legend columns=1]
      \addplot[draw=r1ref, fill=r1ref!18, thick] coordinates {{(0,{p_busy}) (1000,{p_busy})}} \closedcycle;
      \addlegendentry{{busy-polling, as built ({p_busy:.3f}\,W)}}
      \addplot[const plot, draw=r1prunedbits, fill=r1prunedbits!30, thick] coordinates {{{gated}}} \closedcycle;
      \addlegendentry{{gated + read enables (pre-registered)}}
      \node[font=\scriptsize, anchor=south, align=center] at (axis cs:{250 + (s1_ms + t_v) / 2:.1f},{p_awake + 0.004:.4f})
        {{verifier, {t_v:.1f}\,ms}};
      \node[font=\scriptsize, anchor=south west, align=left, inner sep=1pt] at (axis cs:{750 + 10:.1f},{p_awake + 0.004:.4f})
        {{stage 1 only}};
      \node[font=\scriptsize, anchor=north, inner sep=1pt] at (axis cs:{625:.1f},{p_idle - 0.006:.4f})
        {{idle {p_idle:.3f}\,W}};
      \node[font=\sffamily\scriptsize\bfseries, anchor=north west] at (rel axis cs:0.01,0.99) {{(b)}};
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{(a) The idle mode R3 builds. After a request the core stores to a power register and stops; a
  BUFGCE then stops the clock of the core and the \texttt{{kdot}} unit, and the ARM's write of the mailbox word
  MB[0] wakes it. Read enables let each BRAM port read only when needed. (b) PL power over 1\,s (four
  requests) for one verifier call: as built the PL draws {p_busy:.3f}\,W all the time; with the idle mode it
  draws about {p_idle:.3f}\,W between requests and about {p_awake:.3f}\,W while awake (pre-registered estimate;
  verifier time from the RTL simulation). Stage-1 bursts last 0.35\,ms and are drawn {s1_ms:.0f}\,ms wide.}}
  \label{{fig:r3-idea}}
\end{{figure}}
"""


def body_plan(h, f):
    H4 = h['hypotheses']['H4']['designs']
    b1, b2 = H4['B1'], H4['B2']
    f2, f3 = f['F2'], f['F3']
    tv, tv_max = f3['t_v_ms']['mean'], f3['t_v_ms']['max']
    worst = tv_max + h['stage1']['t_max_ms']
    verdict = 'above' if worst > 200 else 'within'
    anchor = ('equal to the board' if f2['anchor_equal'] else 'different from the board')
    c = f2['cycles']
    return rf"""Clock gating stops a circuit's clock while it has no work, so its registers stop switching and draw no
dynamic power. Our keyword spotter never idles: the PicoRV32 busy-polls its mailbox and both ports of the
256\,KiB block RAM read on every cycle, so the programmable logic draws {h['p_busy_W']:.3f}\,W whatever it
computes. R3 adds an idle mode ported from the streaming design (Fig.~\ref{{fig:r3-idea}}a). The block RAM is
{h['power_terms_W']['bram']:.3f}\,W of the {h['power_terms_W']['pl_dyn']:.3f}\,W dynamic power, so BRAM read
enables carry most of the saving; gating the core's clock alone would save about
{1000 * (h['power_terms_W']['u_cpu'] + h['power_terms_W']['u_kdot'] + h['power_terms_W']['u_poisson']):.0f}\,mW.
Pre-registered average power for the two cascades of R2 (B1: highest recall; B2: at most 2,150 verifier calls
per hour): B1 {b1['b']['p_avg_W']:.3f}\,W ($-${100 * b1['b']['saving']:.0f}\,\%), B2
{b2['b']['p_avg_W']:.3f}\,W ($-${100 * b2['b']['saving']:.0f}\,\%); at most {b1['c']['p_avg_W']:.3f} and
{b2['c']['p_avg_W']:.3f}\,W if the whole core stays on while awake. Feasibility, simulated on the unchanged
SoC: stage~1 takes {fmt(c['min'])}/{fmt(c['mean'], 1)}/{fmt(c['max'])} cycles (min/mean/max), {anchor},
bit-exact on {f2['vectors']} windows. A verifier call takes {tv:.1f}\,ms, bit-exact on {f3['windows']}
windows, so the worst compute per request ({worst:.1f}\,ms) is {verdict} the 200\,ms requirement.

\input{{figures/fig_r3_idea}}
"""


def plan():
    h, f = load()
    out = ROOT / 'reports/latex/clock_gating_plan'
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    shutil.copy(R2TEX / 'figures/r1_style.tex', out / 'figures/r1_style.tex')
    shutil.copy(GITIGNORE, out / '.gitignore')
    (out / 'figures/fig_r3_idea.tex').write_text(fig_idea(h, f), encoding='utf-8')
    main = ('% R3 plan mini report: clock gating and BRAM read enables for the B1/B2 cascades.\n'
            '% Same class and packages as the group report, so figures/ can be \\input there unchanged\n'
            '% (plus \\input{figures/r1_style} in its preamble). Build: latexmk -pdf main.tex\n'
            + PREAMBLE +
            '\n\\begin{document}\n'
            '\\title{Clock gating for the cascade keyword spotter:\\\\plan and feasibility}\n'
            '\\author{Pedro Navarro (2132907)}\n\\maketitle\n\n'
            + body_plan(h, f) + '\n\\end{document}\n')
    (out / 'main.tex').write_text(main, encoding='utf-8')
    print('wrote', out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('report', choices=['plan'])
    a = p.parse_args()
    plan()


if __name__ == '__main__':
    main()
