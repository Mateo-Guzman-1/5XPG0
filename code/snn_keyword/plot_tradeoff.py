#!/usr/bin/env python3
"""plot_tradeoff.py — recall vs false-alarm trade-off of ONE run, per epoch.

Reads runs/<name>/history.csv (validation numbers after every epoch, written
by train_sheila.py) and draws two panels:
  left:  validation recall and false-alarm rate against epoch
  right: the trade-off itself, recall against false-alarm rate, one point per
         epoch, joined in epoch order
The false-alarm limit (default 10 %) is drawn on both. The chosen epoch is the
one with the highest recall whose false-alarm rate stays within the limit
(ties: the earliest such epoch, i.e. the least training), unless --epoch
picks one by hand.

Run:  python plot_tradeoff.py epochs20
      python plot_tradeoff.py epochs20 --limit 0.05
      python plot_tradeoff.py epochs20 --epoch 10   (mark epoch 10 instead)
Out:  runs/<name>/tradeoff.png and tradeoff.pdf (the pdf is for LaTeX)
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")                  # write a file, no window needed
import matplotlib.pyplot as plt

# same colors as compare_runs.py
RECALL, FALSE_ALARM = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e6e5e1", "#fcfcfb"


def load(name):
    with open(os.path.join("runs", name, "history.csv"), newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def choose_epoch(hist, limit):
    ok = [h for h in hist if h["val_false_alarm"] <= limit]
    if not ok:
        return None
    best = max(h["val_recall"] for h in ok)
    return next(h for h in ok if h["val_recall"] == best)


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", help="run name (folder in runs/)")
    p.add_argument("--limit", type=float, default=0.10,
                   help="highest acceptable false-alarm rate (default 0.10)")
    p.add_argument("--epoch", type=int,
                   help="mark this epoch as chosen instead of using the rule")
    a = p.parse_args()

    hist = load(a.name)
    ep = [int(h["epoch"]) for h in hist]
    rec = [100 * h["val_recall"] for h in hist]
    fa = [100 * h["val_false_alarm"] for h in hist]
    lim = 100 * a.limit
    pick = (next(h for h in hist if int(h["epoch"]) == a.epoch) if a.epoch
            else choose_epoch(hist, a.limit))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    fig.patch.set_facecolor(SURFACE)

    # --- left: both rates against epoch (same unit, so one axis) ---
    style(ax1, "Validation recall and false-alarm rate per epoch")
    ax1.axhline(lim, color=MUTED, linestyle="--", linewidth=1)
    ax1.text(ep[-1], lim + 1.5, f"false-alarm limit ({lim:g} %)",
             ha="right", va="bottom", color=MUTED, fontsize=8.5)
    ax1.plot(ep, rec, color=RECALL, linewidth=2, marker="o", markersize=4,
             label="Recall ('sheila' caught)")
    ax1.plot(ep, fa, color=FALSE_ALARM, linewidth=2, marker="s", markersize=4,
             label="False-alarm rate (other words fired on)")
    if pick:
        ax1.axvline(pick["epoch"], color=INK, linewidth=0.8, alpha=0.5)
        ax1.text(pick["epoch"] - 0.3, 97, f"chosen: epoch {int(pick['epoch'])}",
                 color=INK, fontsize=8.5, ha="right", va="top")
    # direct labels at the line ends
    ax1.text(ep[-1] + 0.3, rec[-1], f"{rec[-1]:.1f} %", color=INK,
             fontsize=8.5, va="center")
    ax1.text(ep[-1] + 0.3, fa[-1], f"{fa[-1]:.1f} %", color=INK,
             fontsize=8.5, va="center")
    ax1.set_xlabel("Epoch", color=INK)
    ax1.set_ylabel("% of validation clips", color=INK)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0.5, ep[-1] + 1.8)
    ax1.set_xticks([e for e in ep if e == 1 or e % 2 == 0])
    ax1.legend(loc="center left", fontsize=8.5, frameon=False,
               labelcolor=INK)

    # --- right: the trade-off curve ---
    style(ax2, "Trade-off: recall vs false-alarm rate")
    xmax = max(lim * 1.25, max(fa) * 1.15)
    ax2.axvspan(lim, xmax, color=GRID, alpha=0.7, linewidth=0)
    ax2.text(lim + 0.15 * (xmax - lim), 3, "above limit",
             color=MUTED, fontsize=8.5)
    ax2.axvline(lim, color=MUTED, linestyle="--", linewidth=1)
    # one hue, light -> dark = early -> late epoch (sequential)
    shade = matplotlib.colors.LinearSegmentedColormap.from_list(
        "epochs", ["#b7d3f2", RECALL, "#0d3a73"])
    ax2.scatter(fa, rec, s=40, c=ep, cmap=shade, edgecolor=SURFACE,
                linewidth=1.5, zorder=2)
    ax2.text(0.02, 0.97, "lighter = early epoch, darker = late epoch",
             transform=ax2.transAxes, color=MUTED, fontsize=8, va="top")
    # label the first, every 5th and the last epoch, plus the chosen one
    show = {1, ep[-1]} | {e for e in ep if e % 5 == 0}
    if pick:
        show.add(int(pick["epoch"]))
    for e, x, y in zip(ep, fa, rec):
        if e in show:
            ax2.annotate(f"{e}", (x, y), xytext=(5, -11),
                         textcoords="offset points", color=INK, fontsize=8.5)
    if pick:
        ax2.scatter([100 * pick["val_false_alarm"]],
                    [100 * pick["val_recall"]], s=150, facecolor="none",
                    edgecolor=INK, linewidth=1.5, zorder=3)
        ax2.annotate(f"chosen: epoch {int(pick['epoch'])}\n"
                     f"recall {100 * pick['val_recall']:.1f} %, "
                     f"false alarm {100 * pick['val_false_alarm']:.1f} %",
                     (100 * pick["val_false_alarm"], 100 * pick["val_recall"]),
                     xytext=(0.45, 0.25), textcoords="axes fraction",
                     color=INK, fontsize=8.5,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax2.set_xlabel("False-alarm rate (%)  —  lower is better", color=INK)
    ax2.set_ylabel("Recall (%)  —  higher is better", color=INK)
    ax2.set_xlim(0, xmax)
    ax2.set_ylim(5 * (min(rec) // 5) - 5, 100)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join("runs", a.name, f"tradeoff.{ext}")
        fig.savefig(out, dpi=200, facecolor=SURFACE)
        print(f"saved {out}")
    if pick:
        print(f"chosen epoch {int(pick['epoch'])}: "
              f"recall {pick['val_recall']:.3f}, "
              f"false alarm {pick['val_false_alarm']:.3f}, "
              f"accuracy {pick['val_accuracy']:.3f}")


if __name__ == "__main__":
    main()
