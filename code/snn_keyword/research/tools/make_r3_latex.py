"""R3 mini reports as LaTeX (research/PLAN_R3_GATING.md, Phase 1 step 6 and Phase 3).

    make_r3_latex.py plan      -> reports/latex/clock_gating_plan/ (main.tex, figures/, .gitignore)
    make_r3_latex.py results   -> reports/latex/clock_gating_results/ (main.tex, sections/, figures/, tables/):
                                  three diagrams (clock gating, BRAM layouts, final design), power per state,
                                  requirements across our iterations (R1, R2, R3), tables (requirements, cost,
                                  predictions); data from results/R2/designs.csv and results/R3/*.json

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


# =========================================================================== results report (Phase 3)
# Pedro's structure: busy-polling -> step 1, clock gating alone (G8, READ_WAIT 8) -> final design (G1-D: READ_WAIT 1 +
# address-decoded BRAM). Previous iterations are our own only (W0 from R1, B1/B2 as estimated in R2).

RES = ROOT / 'results'
OUT_RES = ROOT / 'reports/latex/clock_gating_results'


def tpl(text, **kw):
    """Fill <<NAME>> placeholders (TikZ code is full of braces, so no str.format)."""
    for k, v in kw.items():
        text = text.replace(f'<<{k}>>', str(v))
    if '<<' in text:
        raise ValueError('unfilled placeholder: ' + text[text.index('<<'):text.index('<<') + 30])
    return text


def load_results():
    import csv
    r2 = {r['design']: r for r in csv.DictReader(open(RES / 'R2/designs.csv', encoding='utf-8'))}
    e = json.loads((RES / 'R3/energy.json').read_text())
    fab = json.loads((RES / 'R3/fabric.json').read_text())
    ver = json.loads((RES / 'R3/phase2/verification.json').read_text())
    act = json.loads((RES / 'R3/phase2/activity.json').read_text())
    s6 = json.loads((RES / 'S6/power.json').read_text())['ksdot']['power']
    f = lambda d, k: float(r2[d][k])
    prev = {code: dict(live=f(d, 'live_recall'), live_sd=f(d, 'live_recall_sd'), fa=f(d, 'fa_per_hour'),
                       fa_sd=f(d, 'fa_per_hour_sd'), other=f(d, 'live_other'), latency=f(d, 'latency_ms'),
                       t_max=f(d, 't_max_ms'), memory=f(d, 'memory_kib'), p_busy=f(d, 'p_avg_busy_W'))
            for code, d in (('W0', 'stage1_R1'), ('B1', 'cascade_B'), ('B2', 'cascade_B_calls2150'))}
    V = e['variants']
    pm = lambda var, c: V[var]['designs'][c]['12.5']['mean']
    r3 = {c: dict(p_g8=pm('asbuilt_rw8', c)['p_avg_W'], p_g1=pm('asbuilt_rw1', c)['p_avg_W'],
                  p_d=pm('decomp_rw1', c)['p_avg_W'], save_g8=pm('asbuilt_rw8', c)['saving'],
                  save_g1=pm('asbuilt_rw1', c)['saving'], save_d=pm('decomp_rw1', c)['saving'],
                  e_h_d=pm('decomp_rw1', c)['e_h_J'], asleep=1 - pm('decomp_rw1', c)['awake_share'],
                  latency=V['decomp_rw1']['designs'][c]['latency_ms'],
                  lo=V['decomp_rw1']['designs'][c]['12.5']['mean-sd']['p_avg_W'],
                  hi=V['decomp_rw1']['designs'][c]['12.5']['mean+sd']['p_avg_W'])
          for c in ('B1', 'B2')}
    chip = {r['On-Chip']: (0.0 if isinstance(r['Power (W)'], str) else float(r['Power (W)'])) for r in s6['on_chip']}
    s6c = dict(pl_static=0.085, bram=chip['Block RAM'], clocks=chip['Clocks'],
               logic=chip['Slice Logic'] + chip['Signals'] + chip['DSPs'])
    # central prediction for the gated READ_WAIT 1 design, written at G1 before any READ_WAIT 1 run
    h1 = json.loads((RES / 'R3/phase2/hypothesis_rw1.json').read_text())['hypotheses']['H4_rw1']['designs']
    pre = {c: h1[c]['rw1']['p_avg_central_W'] for c in ('B1', 'B2')}
    return dict(prev=prev, r3=r3, e=e, V=V, fab=fab, ver=ver, act=act, s6c=s6c, pre=pre)


def fig_gating(R):
    a1 = R['act']['asbuilt_rw1']['activity']
    t_s1 = a1['awake_s1_cycles'] / 1e5                      # ms
    t_v = a1['verifier_cycles']['mean'] / 1e5
    w = 6.0                                                  # stage-1 bursts drawn 6 ms wide
    pts = []
    for k in range(4):
        t0 = 250.0 * k
        end = t0 + w + (t_v if k == 1 else 0.0)
        pts += [(t0, 0), (t0, 1), (end, 1), (end, 0)]
    trace = ' '.join(f'({x:.2f},{y})' for x, y in [(0, 0)] + pts + [(1000, 0)])
    return tpl(r"""% R3 results: clock gating (TikZ + pgfplots). Needs: pgfplots, arrows.meta, figures/r1_style.tex.
\begin{figure}[!t]
  \centering
  \begin{tikzpicture}[font=\sffamily\scriptsize,
      blk/.style={draw,rounded corners=1pt,align=center,inner sep=2pt,minimum height=0.8cm,fill=white},
      gat/.style={blk,fill=r1prunedbits!15},
      arr/.style={-{Stealth[length=1.4mm]},thick},
      al/.style={font=\sffamily\tiny,inner sep=1.5pt,align=center}]
    \node[blk,minimum width=1.6cm] (ps) at (0,0) {ARM (PS)\\new request,\\MB[0] last};
    \node[blk,minimum width=1.45cm] (db) at (2.75,0) {doorbell\\(sticky)};
    \node[blk,minimum width=1.45cm] (fsm) at (5.6,0) {sleep\\FSM};
    \node[blk,minimum width=1.6cm] (fclk) at (0,-1.7) {FCLK0\\100\,MHz};
    \node[blk,minimum width=1.45cm] (bg) at (2.75,-1.7) {BUFGCE};
    \node[gat,minimum width=1.6cm] (cpu) at (5.6,-1.7) {PicoRV32\\+ \texttt{kdot}};
    \draw[arr] (ps) -- node[al,above]{AXI write} (db);
    \draw[arr] (db) -- node[al,above]{wake} (fsm);
    \draw[arr] ([xshift=0.3cm]cpu.north) -- node[al,right]{SLEEP store\\after reply} ([xshift=0.3cm]fsm.south);
    \draw[arr] ([xshift=-0.3cm]fsm.south) |- ++(0,-0.42) -| node[al,above,pos=0.25]{clock enable} (bg.north);
    \draw[arr] (fclk) -- (bg);
    \draw[arr] (bg) -- node[al,above]{gated clock} (cpu);
    \node[al,anchor=north,text=black!65] at (2.75,-2.35) {free-running clock: timer, LED, bus logic, AXI interface, BRAM};
    \node[font=\sffamily\scriptsize\bfseries,anchor=north west] at (-1.05,0.85) {(a)};
  \end{tikzpicture}

  \vspace{2mm}
  \begin{tikzpicture}
    \begin{axis}[r1axis, width=0.97\columnwidth, height=2.9cm, xmin=0, xmax=1000, ymin=-0.15, ymax=1.75, xlabel={time (ms)},
        ytick={0,1}, yticklabels={asleep, awake}, xtick={0,250,500,750,1000}, ylabel={CPU clock}]
      \addplot[const plot, draw=r1prunedbits, fill=r1prunedbits!30, thick] coordinates {<<TRACE>>} \closedcycle;
      \node[font=\scriptsize, anchor=south] at (axis cs:<<VX>>,1.02) {verifier, <<TV>>\,ms};
      \node[font=\scriptsize, anchor=south west, inner sep=1pt] at (axis cs:508,1.02) {stage 1, <<TS1>>\,ms};
      \node[font=\sffamily\scriptsize\bfseries, anchor=north west] at (rel axis cs:0.005,0.98) {(b)};
    \end{axis}
  \end{tikzpicture}
  \caption{Clock gating. (a) After each reply the core stores to a power register and stalls; the sleep FSM then
  stops its clock (one BUFGCE). The ARM writes the mailbox word MB[0] last; this rings a sticky doorbell that
  wakes the core, also if it rang just before the core went to sleep. (b) The CPU clock of step~2 over
  1\,s with one verifier call: awake <<TS1>>\,ms per request (drawn <<W>>\,ms wide) and <<TV>>\,ms per verifier
  call; asleep <<SLB1>>\,\% (B1) and <<SLB2>>\,\% (B2) of the time.}
  \label{fig:r3-gating}
\end{figure}
""", TRACE=trace, VX=f'{250 + (w + t_v) / 2:.1f}', TV=f'{t_v:.0f}', TS1=f'{t_s1:.2f}', W=f'{w:.0f}',
               SLB1=f'{100 * R["r3"]["B1"]["asleep"]:.1f}', SLB2=f'{100 * R["r3"]["B2"]["asleep"]:.1f}')


def fig_memory(R):
    V = R['V']
    b_built = V['asbuilt_rw1']['components']['busy']['bram']
    b_dec = V['decomp_rw1']['components']['busy']['bram']
    cw, ch = 0.17, 0.4                                       # (a) 32 bit columns x 2 stacked blocks
    cells_a = [rf'\draw[fill=r1pruned!75, draw=white, line width=0.4pt] ({c * cw:.3f},{r * ch:.3f}) rectangle ++({cw},{ch});'
               for c in range(32) for r in range(2)]
    bw, bh, x0, y0 = 0.4, 0.27, 1.5, -3.75                   # (b) 8 x 8 address-decoded blocks; block 21 addressed
    on_r, on_c = 2, 5
    cells_b = [rf'\draw[fill={"r1prunedbits" if (r, c) == (on_r, on_c) else "black!7"}, draw=black!40, line width=0.3pt] '
               rf'({x0 + c * bw:.3f},{y0 + (7 - r) * bh:.3f}) rectangle ++({bw},{bh});' for r in range(8) for c in range(8)]
    yon = y0 + (7 - on_r) * bh + bh / 2
    return tpl(r"""% R3 results: BRAM layouts (TikZ). Needs: arrows.meta, figures/r1_style.tex.
\begin{figure}[!t]
  \centering
  \begin{tikzpicture}[font=\sffamily\scriptsize, arr/.style={-{Stealth[length=1.4mm]},thick},
      al/.style={font=\sffamily\tiny,inner sep=1.5pt,align=center},
      big/.style={font=\sffamily\footnotesize\bfseries,align=center}]
    \node[font=\sffamily\scriptsize\bfseries,anchor=south west] at (-1.25,1.05) {(a) as built: bit-sliced};
    <<CELLSA>>
    \node[al,anchor=east] at (-0.05,0.2) {words 0--32K};
    \node[al,anchor=east] at (-0.05,0.6) {32K--64K};
    \node[al,anchor=south] at (0.09,0.8) {bit 0};
    \node[al,anchor=south] at (5.35,0.8) {bit 31};
    \node[big,anchor=west,text=r1pruned] at (5.6,0.4) {64 on\\<<BA>>\,W};
    \node[font=\sffamily\scriptsize\bfseries,anchor=south west] at (-1.25,-1.4) {(b) address-decoded};
    <<CELLSB>>
    \node[draw,rounded corners=1pt,align=center,fill=white,inner sep=2pt] (dec) at (0.25,<<YON>>) {decoder\\(address)};
    \draw[arr] (dec.east) -- (<<X0>>,<<YON>>);
    \node[draw,rounded corners=1pt,align=center,fill=white,inner sep=2pt] (mux) at (5.4,<<YON>>) {64-way\\selector};
    \draw[arr] (<<X1>>,<<YON>>) -- (mux.west);
    \draw[arr] (mux.east) -- ++(0.3,0) node[right,al]{data};
    \node[al,anchor=north] at (<<XM>>,<<YB>>) {64 blocks of 1,024 words};
    \node[big,anchor=west,text=r1prunedbits] at (5.0,<<YLB>>) {1 on\\<<BD>>\,W};
  \end{tikzpicture}
  \caption{The 256\,KiB memory: 64 blocks of 36\,Kbit, power while the CPU sleeps. (a) As built, each block holds one
  bit of every word, and with byte writes Vivado ties every enable to 1: all blocks read in every cycle. (b) Address-decoded,
  each block holds 1,024 whole words; only the block that holds the address is enabled, one per port.}
  \label{fig:r3-memory}
\end{figure}
""", CELLSA='\n    '.join(cells_a), CELLSB='\n    '.join(cells_b), BA=f'{b_built:.3f}', BD=f'{b_dec:.3f}',
               YON=f'{yon:.3f}', X0=f'{x0:.3f}', X1=f'{x0 + 8 * bw:.3f}', XM=f'{x0 + 4 * bw:.3f}',
               YB=f'{y0 - 0.05:.3f}', YLB=f'{y0 + 0.45:.3f}')


def fig_system(R):
    a1 = R['act']['asbuilt_rw1']['activity']
    s1 = R['ver']['sets']['rw1']['stage1_cycles']['mean'] / 1e5
    t_v = a1['verifier_cycles']['mean'] / 1e5
    model_kb = 66056 / 1000
    return tpl(r"""% R3 results: architecture of the final design (TikZ). Needs: arrows.meta, fit, figures/r1_style.tex.
\begin{figure*}[!t]
  \centering
  \resizebox{\textwidth}{!}{\begin{tikzpicture}[font=\sffamily\scriptsize,
      blk/.style={draw,rounded corners=1pt,align=center,minimum height=0.95cm,inner sep=2.5pt,fill=white},
      gat/.style={blk,fill=r1prunedbits!15},
      mem/.style={blk,fill=r1dense!10},
      dec/.style={draw,rounded corners=6pt,align=center,inner sep=2.5pt,minimum height=0.8cm,fill=black!5,font=\sffamily\tiny},
      grp/.style={draw,dashed,rounded corners=2pt,inner sep=5pt},
      lab/.style={font=\sffamily\scriptsize\bfseries,anchor=south west,inner sep=1.5pt},
      arr/.style={-{Stealth[length=1.6mm]},thick},
      al/.style={above,font=\sffamily\tiny,inner sep=2pt,align=center},
      note/.style={font=\sffamily\tiny,align=center,text=black!70}]
    % PC
    \node[blk,minimum width=1.5cm] (mic) at (-0.6,0) {Microphone\\16\,kHz};
    \node[blk,minimum width=2.3cm] (fr) at (1.95,0.7) {25 new 10\,ms frames\\(600\,B)};
    \node[blk,minimum width=2.3cm] (win) at (1.95,-0.7) {1\,s window 24\,$\times$\,32\\(768\,B)};
    \node[grp,fit=(mic)(fr)(win)] (pc) {};
    \node[lab] at (pc.north west) {PC, every 250\,ms};
    \draw[arr] (mic) -- (fr.west); \draw[arr] (mic) -- (win.west);
    % PS
    \node[blk,minimum width=2.0cm,minimum height=2.3cm] (arm) at (5.9,0) {ARM Cortex-A9\\writes the\\request, then\\MB[0]};
    \node[grp,fit=(arm)] (ps) {};
    \node[lab] at (ps.north west) {PS};
    \draw[arr] (fr.east) -- (fr.east -| ps.west); \draw[arr] (win.east) -- (win.east -| ps.west);
    \node[note,anchor=north] at (4.35,-1.25) {TCP};
    % PL hardware
    \node[blk,minimum width=2.0cm] (psif) at (9.4,1.15) {AXI slave\\mailbox, doorbell};
    \node[gat,minimum width=2.0cm] (gate) at (9.4,-0.35) {sleep FSM\\+ BUFGCE};
    \node[gat,minimum width=2.9cm,minimum height=1.25cm] (cpu) at (12.9,-0.35) {PicoRV32 RV32IM, 100\,MHz\\\texttt{kdot}/\texttt{ksdot}, READ\_WAIT 1\\(gated clock)};
    \node[mem,minimum width=2.9cm,minimum height=1.25cm] (ram) at (12.9,1.45) {BRAM 256\,KiB, 64 blocks,\\address-decoded: firmware,\\models <<MODEL>>\,KB, 1.5\,s frames};
    \draw[arr] (ps.east |- psif) -- (psif.west);
    \draw[arr] (psif) -- (gate);
    \draw[arr] (gate) -- node[al]{clock} (cpu);
    \draw[arr,{Stealth[length=1.6mm]}-{Stealth[length=1.6mm]}] (psif.east) -- (ram.west |- psif.east);
    \draw[arr,{Stealth[length=1.6mm]}-{Stealth[length=1.6mm]}] (cpu) -- (ram);
    % firmware flow per request
    \node[blk,minimum width=1.7cm] (push) at (9.0,-2.45) {push frames\\into buffer};
    \node[blk,minimum width=2.2cm] (s1) at (11.35,-2.45) {stage 1: pruned SNN\\48 LIF, <<S1>>\,ms};
    \node[dec] (t1) at (13.75,-2.45) {margin $\geq t_1$\\in $k$ or $k{-}1$?};
    \node[blk,minimum width=2.2cm] (gru) at (16.3,-2.45) {stage 2: GRU verifier\\<<TV>>\,ms};
    \node[dec] (t2) at (18.6,-2.45) {head $\geq t_2$?};
    \node[blk,minimum width=1.2cm] (led) at (20.4,-2.45) {LED0\\1\,s};
    \draw[arr] (push) -- (s1); \draw[arr] (s1) -- (t1); \draw[arr] (t1) -- node[al]{yes} (gru);
    \draw[arr] (gru) -- (t2); \draw[arr] (t2) -- node[al]{yes} (led);
    \draw[arr] (t1.south) -- ++(0,-0.45) node[below,note,inner sep=1pt] (nosl) {no: reply, sleep};
    \node[grp,fit=(push)(led)(nosl)] (fw) {};
    \node[lab] at (fw.north west) {firmware, per request};
    \draw[arr,dashed,black!60] (cpu.south) -- (cpu.south |- fw.north);
    \node[grp,fit=(psif)(ram)(cpu)(gate)(fw)] (pl) {};
    \node[lab] at (pl.north west) {PL (FPGA fabric)};
  \end{tikzpicture}}
  \caption{The final design (B1 and B2 differ only in $t_1$). Every 250\,ms the PC sends the newest 1\,s window and
  the 25 new 10\,ms frames. The core wakes, pushes the frames into a 1.5\,s buffer and runs stage~1; if its margin
  reached $t_1$ in this request or the previous one, the GRU verifier scores the buffer, and a head score of at least
  $t_2$ (1\,s hold-off) lights LED0. Then the core replies and sleeps (Fig.~\ref{fig:r3-gating}).}
  \label{fig:r3-system}
\end{figure*}
""", MODEL=f'{model_kb:.0f}', S1=f'{s1:.2f}', TV=f'{t_v:.0f}')


def stacked(x, parts, width=0.6):
    out, y = [], 0.0
    for val, style in parts:
        out.append(rf'\fill[{style}] (axis cs:{x - width / 2:.3f},{y:.5f}) rectangle (axis cs:{x + width / 2:.3f},{y + val:.5f});')
        y += val
    return out, y


def fig_states(R):
    V, s6 = R['V'], R['s6c']
    comps = lambda c: [(c['pl_static'], 'black!30'), (c['bram'], 'r1pruned!80'), (c['clocks'], 'r1bits!80'),
                       (c['logic'] + c['signals'] + (c.get('dsp') or 0.0), 'r1dense!55')]
    bars = [('Busy\\\\(as built)', [(s6['pl_static'], 'black!30'), (s6['bram'], 'r1pruned!80'), (s6['clocks'], 'r1bits!80'),
                                   (s6['logic'], 'r1dense!55')], 0.367),
            ('Step 1\\\\awake', comps(V['asbuilt_rw8']['components']['busy']), V['asbuilt_rw8']['states']['busy']['12.5']['p_pl']),
            ('Step 1\\\\asleep', comps(V['asbuilt_rw8']['components']['idle']), V['asbuilt_rw8']['states']['idle']['12.5']['p_pl']),
            ('Step 2\\\\awake', comps(V['decomp_rw1']['components']['busy']), V['decomp_rw1']['states']['busy']['12.5']['p_pl']),
            ('Step 2\\\\asleep', comps(V['decomp_rw1']['components']['idle']), V['decomp_rw1']['states']['idle']['12.5']['p_pl'])]
    body = []
    for i, (_, parts, total) in enumerate(bars):
        code, top = stacked(i, parts)
        body += code
        body.append(rf'\node[font=\scriptsize, anchor=south, inner sep=1pt] at (axis cs:{i},{max(top, total):.4f}) {{{total:.3f}}};')
    labels = ','.join('{' + b[0] + '}' for b in bars)
    # (b) average PL power against verifier calls per hour: E_h model of r3_energy.py, linear in the calls
    line = lambda var, calls: (V[var]['states']['idle']['12.5']['p_pl'] +
                               (V[var]['states']['busy']['12.5']['p_pl'] - V[var]['states']['idle']['12.5']['p_pl']) *
                               (14400 * V[var]['a_s1_cycles'] + calls * V[var]['c_v_cycles']) / 1e8 / 3600)
    xmax = 5000
    pts = lambda var: f'(0,{line(var, 0):.5f}) ({xmax},{line(var, xmax):.5f})'
    marks = []
    for code, mk in (('B1', 'diamond*'), ('B2', 'square*')):
        d = V['decomp_rw1']['designs'][code]['12.5']['mean']
        sd = V['decomp_rw1']['designs'][code]['12.5']['mean+sd']['calls'] - d['calls']
        for var, col in (('asbuilt_rw8', C_S1.replace('!35', '!70')), ('decomp_rw1', C_FIN)):
            y = V[var]['designs'][code]['12.5']['mean']['p_avg_W']
            marks.append(rf'\addplot[only marks, mark={mk}, mark size=2.3pt, color={col}, mark options={{fill={col}}}, '
                         rf'error bars/.cd, x dir=both, x explicit] coordinates {{({d["calls"]:.1f},{y:.5f}) +- ({sd:.1f},0)}};')
        pre = R['pre'][code]
        marks.append(rf'\addplot[only marks, mark={mk.rstrip("*")}, mark size=2.6pt, color=black, thick] '
                     rf'coordinates {{({d["calls"]:.1f},{pre:.5f})}};')
        marks.append(rf'\node[font=\tiny, anchor=north, inner sep=3pt] at (axis cs:{d["calls"]:.1f},{line("decomp_rw1", d["calls"]) - 0.008:.4f}) {{{code}}};')
    return tpl(r"""% R3 results: PL power per state and against verifier calls (pgfplots). Needs: figures/r1_style.tex.
\begin{figure*}[!t]
  \centering
  \begin{tikzpicture}
    \begin{axis}[r1axis, scale only axis, width=0.36\textwidth, height=4.2cm, xmin=-0.6, xmax=4.6, ymin=0, ymax=0.45,
        xtick={0,1,2,3,4}, xticklabels={<<LABELS>>}, x tick label style={font=\scriptsize, align=center},
        ylabel={PL power (W)}, yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1},
        title={(a)}, title style={font=\footnotesize, at={(0.02,0.97)}, anchor=north west}]
      <<BODY>>
    \end{axis}
  \end{tikzpicture}\hfill
  \begin{tikzpicture}
    \begin{axis}[r1axis, scale only axis, width=0.42\textwidth, height=4.2cm, xmin=0, xmax=<<XMAX>>, ymin=0, ymax=0.45,
        xlabel={Verifier calls per hour}, ylabel={Average PL power (W)},
        yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1},
        xticklabel style={/pgf/number format/.cd, 1000 sep={}}, scaled x ticks=false,
        legend style={at={(0.98,0.6)}, anchor=east},
        title={(b)}, title style={font=\footnotesize, at={(0.02,0.97)}, anchor=north west}]
      \addplot[color=r1ref, thick, dashed, no marks] coordinates {(0,0.367) (<<XMAX>>,0.367)};
      \addlegendentry{busy-polling}
      \addplot[color=<<CS1L>>, thick, no marks] coordinates {<<L8>>};
      \addlegendentry{step 1: clock gating}
      \addplot[color=<<CFIN>>, thick, no marks] coordinates {<<LD>>};
      \addlegendentry{step 2: + address-decoded BRAM}
      <<MARKS>>
    \end{axis}
  \end{tikzpicture}\\[2pt]
  {\scriptsize \tikz\fill[black!30] (0,0) rectangle (2.2mm,2.2mm); PL static \quad
   \tikz\fill[r1pruned!80] (0,0) rectangle (2.2mm,2.2mm); BRAM \quad
   \tikz\fill[r1bits!80] (0,0) rectangle (2.2mm,2.2mm); clocks \quad
   \tikz\fill[r1dense!55] (0,0) rectangle (2.2mm,2.2mm); logic and signals \qquad
   filled markers: B1 ($\diamond$), B2 ($\square$), calls $\pm$ sd; hollow: predicted before the measurements}
  \caption{PL power (Vivado). (a) Per state: step~1 gates the CPU clock (READ\_WAIT 8); asleep it saves only
  <<SAVE>>\,W, because the BRAM keeps reading. Step~2 (READ\_WAIT 1, address-decoded BRAM) cuts the BRAM from <<BA>>
  to <<BD>>\,W; asleep, static power is most of what remains. (b) Average PL power against the verifier calls per hour
  (simulated awake time per request and per call): step~2 stays near its idle floor for any realistic call rate.}
  \label{fig:r3-states}
\end{figure*}
""", LABELS=labels, BODY='\n      '.join(body), XMAX=xmax, L8=pts('asbuilt_rw8'), LD=pts('decomp_rw1'),
               CS1L=C_S1.replace('!35', '!70'), CFIN=C_FIN, MARKS='\n      '.join(marks),
               SAVE=f"{0.367 - V['asbuilt_rw8']['states']['idle']['12.5']['p_pl']:.3f}",
               BA=f"{V['asbuilt_rw8']['components']['busy']['bram']:.3f}", BD=f"{V['decomp_rw1']['components']['idle']['bram']:.3f}")


# requirement panels: x = 0 W0 (R1), 1 B1 (R2), 2 B2 (R2), 3 B1 (R3), 4 B2 (R3)
XL = '{W0\\\\(R1)},{B1\\\\(R2)},{B2\\\\(R2)},{B1\\\\(R3)},{B2\\\\(R3)}'
C_W0, C_R2, C_S1, C_FIN = 'r1pruned!80', 'r1dense!55', 'r1prunedbits!35', 'r1prunedbits'


def panel(title, ylabel, ymax, req, req_label, bars, errs=(), fmt='{:.1f}', extra='', req_left=False):
    """bars: (x_center, half_width, value, style, label or None); value labels sit above the error bar if there is one."""
    top = {x: v + sd for x, v, sd in errs}
    req_node = (rf'\node[r1reqlabel, anchor=south west] at (axis cs:-0.55,{req}) {{{req_label}}};' if req_left else
                rf'\node[r1reqlabel, anchor=south east] at (axis cs:4.55,{req}) {{{req_label}}};')
    body = [rf'\draw[r1req] (axis cs:-0.6,{req}) -- (axis cs:4.6,{req});' if req is not None else '',
            req_node if req is not None else '']
    for x, hw, v, style, lab in bars:
        body.append(rf'\fill[{style}] (axis cs:{x - hw:.3f},0) rectangle (axis cs:{x + hw:.3f},{v:.5f});')
        if lab is not None:
            body.append(rf'\node[font=\tiny, anchor=south, inner sep=1pt] at (axis cs:{x:.3f},{max(v, top.get(x, v)):.5f}) {{{lab}}};')
    for x, v, sd in errs:
        body.append(rf'\draw[black, line width=0.5pt] (axis cs:{x:.3f},{v - sd:.5f}) -- (axis cs:{x:.3f},{v + sd:.5f});')
    return tpl(r"""\begin{tikzpicture}
  \begin{axis}[r1axis, scale only axis, width=0.185\textwidth, height=3.6cm, xmin=-0.6, xmax=4.6, xtick={0,1,2,3,4},
      xticklabels={<<XL>>}, x tick label style={font=\tiny, align=center}, ymin=0, ymax=<<YMAX>>,
      ylabel={<<YLABEL>>}, title={<<TITLE>>}, title style={font=\footnotesize, at={(0.02,0.97)}, anchor=north west}<<EXTRA>>]
    <<BODY>>
  \end{axis}
\end{tikzpicture}""", XL=XL, YMAX=ymax, YLABEL=ylabel, TITLE=title, EXTRA=extra, BODY='\n    '.join(b for b in body if b))


def fig_requirements(R):
    P, r3, V = R['prev'], R['r3'], R['V']
    det = lambda key, scale, fmt: [(i, 0.3, scale * P[c][key], C_W0 if c == 'W0' else C_R2 if i < 3 else C_FIN,
                                    fmt.format(scale * P[c][key]))
                                   for i, c in ((0, 'W0'), (1, 'B1'), (2, 'B2'), (3, 'B1'), (4, 'B2'))]
    err = lambda key, sdkey, scale: [(i, scale * P[c][key], scale * P[c][sdkey])
                                     for i, c in ((0, 'W0'), (1, 'B1'), (2, 'B2'), (3, 'B1'), (4, 'B2'))]
    pa = panel('(a)', 'Live ``Sheila\'\' detected (\\%)', 105, 85, '$\\geq$85\\,\\%', det('live', 100, '{:.1f}'),
               err('live', 'live_sd', 100), req_left=True)
    pb = panel('(b)', 'False accepts per hour', 4.2, 2, '$\\leq$2', det('fa', 1, '{:.2f}'), err('fa', 'fa_sd', 1))
    tv8 = R['ver']['sets']['rw8']['worst_compute_ms']
    tv1 = R['ver']['sets']['rw1']['worst_compute_ms']
    cbars = [(0, 0.3, P['W0']['t_max'], C_W0, f"{P['W0']['t_max']:.2f}"),
             (1, 0.3, P['B1']['t_max'], C_R2, f"{P['B1']['t_max']:.0f}"), (2, 0.3, P['B2']['t_max'], C_R2, f"{P['B2']['t_max']:.0f}")]
    for x in (3, 4):
        cbars += [(x - 0.16, 0.15, tv8, C_S1, None), (x + 0.16, 0.15, tv1, C_FIN, None)]
    cbars_lab = [rf'\node[font=\tiny, anchor=south, inner sep=1pt] at (axis cs:{x - 0.16:.2f},{tv8:.2f}) {{{tv8:.0f}}};' for x in (3, 4)] + \
                [rf'\node[font=\tiny, anchor=south, inner sep=1pt] at (axis cs:{x + 0.16:.2f},{tv1:.2f}) {{{tv1:.0f}}};' for x in (3, 4)]
    pc = panel('(c)', 'Worst compute per request (ms)', 250, 200, '$\\leq$200\\,ms', cbars, req_left=True)
    pc = pc.replace(r'\end{axis}', '  ' + '\n    '.join(cbars_lab) + '\n  \\end{axis}')
    dbars = [(0, 0.3, P['W0']['p_busy'], C_W0, f"{P['W0']['p_busy']:.3f}"),
             (1, 0.3, P['B1']['p_busy'], C_R2, f"{P['B1']['p_busy']:.3f}"), (2, 0.3, P['B2']['p_busy'], C_R2, f"{P['B2']['p_busy']:.3f}")]
    dlab = []
    for x, c in ((3, 'B1'), (4, 'B2')):
        dbars += [(x - 0.16, 0.15, r3[c]['p_g8'], C_S1, None), (x + 0.16, 0.15, r3[c]['p_d'], C_FIN, None)]
        dlab += [rf'\node[font=\tiny, anchor=south, inner sep=1pt, rotate=90, anchor=west] at (axis cs:{x - 0.16:.2f},{r3[c]["p_g8"]:.4f}) {{{r3[c]["p_g8"]:.3f}}};',
                 rf'\node[font=\tiny, anchor=south, inner sep=1pt, rotate=90, anchor=west] at (axis cs:{x + 0.16:.2f},{r3[c]["p_d"]:.4f}) {{{r3[c]["p_d"]:.3f}}};']
    pd = panel('(d)', 'Average PL power (W)', 0.52, None, '', dbars,
               extra=', yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1}')
    pd = pd.replace(r'\end{axis}', '  ' + '\n    '.join(dlab) +
                    '\n    \\node[r1reqlabel, anchor=north east] at (axis cs:4.55,0.515) {$\\leq$1\\,W: all pass};\n  \\end{axis}')
    return tpl(r"""% R3 results: the requirement metrics across our iterations (generated by make_r3_latex.py results).
\begin{figure*}[!t]
  \centering
  <<PA>>\hfill
  <<PB>>\hfill
  <<PC>>\hfill
  <<PD>>\\[3pt]
  {\scriptsize \tikz\fill[<<CW0>>] (0,0) rectangle (2.2mm,2.2mm); window SNN alone (R1) \quad
   \tikz\fill[<<CR2>>] (0,0) rectangle (2.2mm,2.2mm); cascade, R2 (cost estimated) \quad
   \tikz\fill[<<CS1>>] (0,0) rectangle (2.2mm,2.2mm); R3 step 1: clock gating alone, READ\_WAIT 8 \quad
   \tikz\fill[<<CFIN>>] (0,0) rectangle (2.2mm,2.2mm); R3 step 2: + READ\_WAIT 1 + address-decoded BRAM}
  \caption{Requirement metrics of our iterations (test set; mean $\pm$ sd over seeds). (a, b) Detection: R3 runs the
  R2 models bit-exactly, so these values carry over. (c) Worst compute per request: R2 assumed a verifier time
  measured on another SoC; on our fabric it is <<TV8>>\,ms with READ\_WAIT 8 and <<TV1>>\,ms with READ\_WAIT 1.
  (d) Average PL power with the verifier calls of R2 (B1 <<CB1>>, B2 <<CB2>> per hour). The other words accepted,
  latency and memory requirements are in Table~\ref{tab:r3-requirements}.}
  \label{fig:r3-requirements}
\end{figure*}
""", PA=pa, PB=pb, PC=pc, PD=pd, CW0=C_W0, CR2=C_R2, CS1=C_S1, CFIN=C_FIN, TV8=f'{tv8:.0f}', TV1=f'{tv1:.0f}',
               CB1=f"{V['decomp_rw1']['designs']['B1']['12.5']['mean']['calls']:,.0f}",
               CB2=f"{V['decomp_rw1']['designs']['B2']['12.5']['mean']['calls']:,.0f}")


def tab_requirements(R):
    P, r3, ver = R['prev'], R['r3'], R['ver']
    ok, no = r'\,\checkmark', r'\,$\times$'
    mark = lambda good: ok if good else no
    t1 = ver['sets']['rw1']['worst_compute_ms']
    rows = [
        ('Live ``Sheila\'\'', '$\\geq$85\\,\\%',
         [f"{100 * P[c]['live']:.1f}{mark(P[c]['live'] >= .85)}" for c in ('W0', 'B1', 'B2')]),
        ('False accepts per hour', '$\\leq$2', [f"{P[c]['fa']:.2f}{mark(P[c]['fa'] <= 2)}" for c in ('W0', 'B1', 'B2')]),
        ('Other words accepted', '$\\leq$0.2\\,\\%', [f"{100 * P[c]['other']:.2f}{mark(P[c]['other'] <= .002)}" for c in ('W0', 'B1', 'B2')]),
        ('Decision latency (ms)', '$\\leq$500', [f"{P['W0']['latency']:.0f}{ok}"] +
         [f"{P[c]['latency']:.0f}$\\to${r3[c]['latency']:.0f}{ok}" for c in ('B1', 'B2')]),
        ('Worst compute (ms)', '$\\leq$200', [f"{P['W0']['t_max']:.2f}{ok}"] +
         [f"{P[c]['t_max']:.0f}$^*\\!\\to${t1:.0f}{ok}" for c in ('B1', 'B2')]),
        ('Memory (KiB)', '$\\leq$144', [f"{P[c]['memory']:.1f}{ok}" for c in ('W0', 'B1', 'B2')]),
        ('PL power (W)', '$\\leq$1', [f"{P['W0']['p_busy']:.3f}{ok}"] +
         [f"{P[c]['p_busy']:.3f}$\\to${r3[c]['p_d']:.3f}{ok}" for c in ('B1', 'B2')]),
    ]
    body = '\n'.join(f'    {a} & {b} & ' + ' & '.join(v) + r' \\' for a, b, v in rows)
    return tpl(r"""% R3 results: requirements across iterations (generated by make_r3_latex.py results).
\begin{table}[!t]
  \centering
  \caption{Group requirements: W0 from R1; B1 and B2 from R2 $\to$ R3 step~2 where R3 changes the value.}
  \label{tab:r3-requirements}
  \scriptsize
  \setlength{\tabcolsep}{2.5pt}
  \begin{tabular}{@{}llccc@{}}
    \toprule
    Requirement & Target & W0 & B1 & B2 \\
    \midrule
<<BODY>>
    \bottomrule
  \end{tabular}\\[2pt]
  {\scriptsize $^*$verifier time measured on another SoC (R2); R3 measures it on ours.}
\end{table}
""", BODY=body)


def tab_cost(R):
    fab, V = R['fab'], R['V']
    s6 = fab['s6']
    rows = [('Busy (as built)', s6['luts'], s6['registers'], 1, s6['wns_ns'], 0.367, None),
            ('1: clock gating', *[fab['builds']['rw8'][k] for k in ('luts', 'registers', 'bufgctrl', 'wns_ns')],
             V['asbuilt_rw8']['states']['idle']['12.5']['p_pl'], 'asbuilt_rw8'),
            ('\\quad + READ\\_WAIT 1', *[fab['builds']['rw1'][k] for k in ('luts', 'registers', 'bufgctrl', 'wns_ns')],
             V['asbuilt_rw1']['states']['idle']['12.5']['p_pl'], 'asbuilt_rw1'),
            ('2: + addr.-dec.\\ BRAM', *[fab['builds']['dec_B'][k] for k in ('luts', 'registers', 'bufgctrl', 'wns_ns')],
             V['decomp_rw1']['states']['idle']['12.5']['p_pl'], 'decomp_rw1')]
    body = []
    for name, lut, ff, bufg, wns, idle, var in rows:
        b1 = '0.367' if var is None else f"{V[var]['designs']['B1']['12.5']['mean']['p_avg_W']:.3f}"
        b2 = '0.367' if var is None else f"{V[var]['designs']['B2']['12.5']['mean']['p_avg_W']:.3f}"
        body.append(f'    {name} & {lut:,.0f} & {ff:,.0f} & {bufg:.0f} & +{wns:.3f} & {idle:.3f} & {b1} & {b2} \\\\')
    return tpl(r"""% R3 results: cost and power of the steps (generated by make_r3_latex.py results).
\begin{table}[!t]
  \centering
  \caption{Cost and PL power of the steps (64 BRAM36 and 8 DSP in all; WNS = timing margin at 100\,MHz).}
  \label{tab:r3-cost}
  \scriptsize
  \setlength{\tabcolsep}{2.5pt}
  \begin{tabular}{@{}lrrrrrrr@{}}
    \toprule
    Design & LUT & FF & BUFG & WNS & Idle & B1 & B2 \\
     & & & & (ns) & (W) & (W) & (W) \\
    \midrule
<<BODY>>
    \bottomrule
  \end{tabular}
\end{table}
""", BODY='\n'.join(body))


def tab_hypotheses(R):
    e, V, ver = R['e'], R['V'], R['ver']
    o = e['outcomes']
    yes, no = r'\checkmark', r'$\times$'
    rows = [
        ('H1', 'verifier 19--25\\,M cycles at RW\\,8, worst $>$200\\,ms', f"{yes} {o['H1']['c_v'] / 1e6:.1f}\\,M, {o['H1']['worst_compute_ms']:.0f}\\,ms"),
        ('H1$'"'"'$', '10.3--11.2\\,M at RW\\,1, worst $\\leq$200\\,ms', f"{yes} {o['H1_rw1']['c_v'] / 1e6:.1f}\\,M, {o['H1_rw1']['worst_compute_ms']:.0f}\\,ms"),
        ('H2', 'bit-exact; same cycles with gating on/off', f'{yes} all sets'),
        ('H3', 'idle 0.085--0.105\\,W', f"{no} step 1 ({o['H3']['asbuilt_rw8']['p_idle_W']:.3f}); {yes} step 2 ({o['H3']['decomp_rw1']['p_idle_W']:.3f})"),
        ('H4', 'B1 $\\leq$0.150, B2 $\\leq$0.130\\,W', f"{no} step 1; {yes} step 2"),
        ('H5', 'gate alone $\\leq$0.03\\,W; enables most of the rest', f"{yes} {o['H5_asbuilt_rw8']['gate_only_saving_W']:.3f}\\,W; {no} enables"),
        ('H6', 'asleep $\\geq$90 / 93\\,\\% (B1 / B2)', f"{yes} {100 * o['H6_rw1']['asleep_share']['B1']:.1f} / {100 * o['H6_rw1']['asleep_share']['B2']:.1f}\\,\\%"),
        ('H7', 'gating $\\leq$+150 LUT, $\\leq$+120 FF', f'{yes} +60, +84'),
        ('H8', 'gating off = previous design', f'{yes} (+11 LUT)'),
        ('H9', 'address-dec.: BRAM $-$80\\,\\%, $\\leq$+150 FF', f"{yes} $-${100 * o['H9']['bram_awake_reduction']:.0f}\\,\\%; {no} +{o['H9_cost']['delta_vs_asbuilt_rw1']['registers']:.0f} FF"),
        ('H10', 'READ\\_WAIT 1 saves only 1--5\\,mW', f"{yes} {-1000 * o['H10']['rw1_minus_rw8_W']['B1']:.1f}\\,mW"),
    ]
    body = '\n'.join(f'    {a} & {b} & {c} \\\\' for a, b, c in rows)
    return tpl(r"""% R3 results: pre-registered predictions and outcomes (generated by make_r3_latex.py results).
\begin{table}[!t]
  \centering
  \caption{Predictions written before the measurements, and outcomes.}
  \label{tab:r3-hypotheses}
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{@{}l>{\raggedright\arraybackslash}p{3.9cm}>{\raggedright\arraybackslash}p{3.5cm}@{}}
    \toprule
     & Prediction & Outcome \\
    \midrule
<<BODY>>
    \bottomrule
  \end{tabular}
\end{table}
""", BODY=body)


def body_results(R):
    r3, V, fab, ver = R['r3'], R['V'], R['fab'], R['ver']
    g8 = V['asbuilt_rw8']['states']['idle']['12.5']['p_pl']
    d_idle = V['decomp_rw1']['states']['idle']['12.5']['p_pl']
    return tpl(r"""% R3 results report body (generated by make_r3_latex.py results). Design codes only, no names.
\section{What}
The cascades B1 and B2 of R2 (Fig.~\ref{fig:r3-system}) meet all seven group requirements, but their cost was
estimated: the verifier time came from another SoC, and the PL draws 0.367\,W whatever it computes, because the core
busy-polls its mailbox and the 256\,KiB BRAM reads on every cycle. R3 runs B1 and B2 on our fabric as one firmware and
adds an idle mode in two steps. Step~1, clock gating, saves only <<SG8B1>>\,\% (B1) and <<SG8B2>>\,\% (B2). Step~2,
an address-decoded memory, brings the PL power to <<PB1>>\,W and <<PB2>>\,W, $-$<<SB1>>\,\% and $-$<<SB2>>\,\%; every
requirement still holds.

\input{figures/fig_r3_system}

\section{How}
\textbf{Step 1: clock gating} (Fig.~\ref{fig:r3-gating}). Between requests the core stops on a store to a power
register, and a BUFGCE stops its clock until the ARM's next request rings a doorbell. The logic is adapted from the
streaming design's sleep branch; it needs no interrupts, and on an older bitstream the store simply returns.

\textbf{Step 2: address-decoded memory} (Fig.~\ref{fig:r3-memory}). Step~1 saves only <<GSAVE>>\,W, because the BRAM,
0.25 of the 0.28\,W dynamic power, keeps reading. Read enables in the RTL cannot stop it: with byte writes, Vivado ties
every enable of the bit-sliced layout to 1. In the address-decoded layout only the block that holds the address is on.
Step~2 also reads the memory in 1 cycle instead of 8 (READ\_WAIT 1): a verifier call takes <<TV1>>\,ms instead of
<<TV8>>\,ms, which meets the 200\,ms requirement.

\textbf{Checks.} Firmware and RTL were simulated (xsim) against the integer models: 117 requests with 13 verifier
calls, bit-exact with and without gating and with identical cycle counts; with gating off, the RTL equals the previous
design in every clock. Power is Vivado's estimate per state (CPU clock at its simulated duty, BRAM enables as built),
combined with the simulated awake time; the board measurement is next.

\input{figures/fig_r3_gating}
\input{figures/fig_r3_memory}
\input{figures/fig_r3_requirements}

\section{Results}
Fig.~\ref{fig:r3-states} shows where the power goes; Table~\ref{tab:r3-cost} lists the cost of each step. Asleep,
step~2 draws <<DIDLE>>\,W, mostly static power; the CPU sleeps <<SLB1>>\,\% (B1) and <<SLB2>>\,\% (B2) of the time. Fig.~\ref{fig:r3-requirements} and Table~\ref{tab:r3-requirements} compare the requirements with our earlier
iterations: detection is unchanged, the measured compute time matches R2's estimate, and the power drops by about
70\,\%. Table~\ref{tab:r3-hypotheses} lists the predictions.

\input{figures/fig_r3_states}
\input{tables/tab_r3_requirements}
\input{tables/tab_r3_cost}

\section{Pros and cons}
\begin{itemize}
  \item[+] PL power $-$71\,\% for B1 and B2 (<<PB1>> and <<PB2>>\,W), all seven requirements met.
  \item[+] Bit-exact: detection is unchanged; all options are off by default, so earlier builds are reproduced.
  \item[+] Worst compute <<TV1>>\,ms on our fabric, well within 200\,ms.
  \item[$-$] +<<DLUT>> LUT and +<<DFF>> FF; the timing margin (+<<WNS>>\,ns) needs a timing-driven Vivado strategy.
  \item[$-$] One BRAM block per port stays on while asleep; explicit banks could switch it off.
  \item[$-$] Static power (<<STAT>>\,W) now dominates; READ\_WAIT 1 changes the cycle counts of earlier experiments.
  \item[$-$] Vivado estimates only: the board measurement is still to do.
\end{itemize}

\input{tables/tab_r3_hypotheses}
""", PB1=f"{r3['B1']['p_d']:.3f}", PB2=f"{r3['B2']['p_d']:.3f}", SB1=f"{100 * r3['B1']['save_d']:.0f}",
               SB2=f"{100 * r3['B2']['save_d']:.0f}", GSAVE=f'{0.367 - g8:.3f}',
               SG8B1=f"{100 * r3['B1']['save_g8']:.1f}", SG8B2=f"{100 * r3['B2']['save_g8']:.1f}",
               TV1=f"{ver['sets']['rw1']['worst_compute_ms']:.0f}", TV8=f"{ver['sets']['rw8']['worst_compute_ms']:.0f}",
               DIDLE=f'{d_idle:.3f}', SLB1=f"{100 * r3['B1']['asleep']:.1f}", SLB2=f"{100 * r3['B2']['asleep']:.1f}",
               DLUT=f"{fab['builds']['dec_B']['luts'] - fab['s6']['luts']:,.0f}",
               DFF=f"{fab['builds']['dec_B']['registers'] - fab['s6']['registers']:,.0f}",
               WNS=f"{fab['builds']['dec_B']['wns_ns']:.2f}",
               STAT=f"{V['decomp_rw1']['components']['idle']['pl_static']:.3f}")


def results():
    R = load_results()
    out = OUT_RES
    for sub in ('figures', 'tables', 'sections'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(R2TEX / 'figures/r1_style.tex', out / 'figures/r1_style.tex')
    shutil.copy(GITIGNORE, out / '.gitignore')
    files = {'figures/fig_r3_gating.tex': fig_gating(R), 'figures/fig_r3_memory.tex': fig_memory(R),
             'figures/fig_r3_system.tex': fig_system(R), 'figures/fig_r3_states.tex': fig_states(R),
             'figures/fig_r3_requirements.tex': fig_requirements(R),
             'tables/tab_r3_requirements.tex': tab_requirements(R), 'tables/tab_r3_cost.tex': tab_cost(R),
             'tables/tab_r3_hypotheses.tex': tab_hypotheses(R), 'sections/body.tex': body_results(R)}
    for name, text in files.items():
        (out / name).write_text(text, encoding='utf-8')
    main = ('% R3 results mini report: clock gating and an address-decoded BRAM for the B1/B2 cascades.\n'
            '% Same class and packages as the group report, so figures/ and tables/ can be \\input there unchanged\n'
            '% (plus \\input{figures/r1_style} in its preamble). Build: latexmk -pdf main.tex\n'
            + PREAMBLE +
            '% full-width floats on a float page: packed at the top, not spread over the page\n'
            '\\makeatletter\\setlength{\\@dblfptop}{0pt}\\setlength{\\@dblfpsep}{14pt}'
            '\\setlength{\\@dblfpbot}{0pt plus 1fil}\\makeatother\n'
            '\n\\begin{document}\n'
            '\\title{Clock gating and an address-decoded memory\\\\for the cascade keyword spotter}\n'
            '\\author{Pedro Navarro (2132907)}\n\\maketitle\n\n'
            '\\input{sections/body}\n\n\\end{document}\n')
    (out / 'main.tex').write_text(main, encoding='utf-8')
    print('wrote', out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('report', choices=['plan', 'results'])
    a = p.parse_args()
    plan() if a.report == 'plan' else results()


if __name__ == '__main__':
    main()
