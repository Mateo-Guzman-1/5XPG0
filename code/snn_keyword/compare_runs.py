#!/usr/bin/env python3
"""compare_runs.py — plot the learning curves of several runs side by side.

Reads runs/<name>/history.csv (written by train_sheila.py) and draws three
panels sharing the epoch axis: training loss, validation recall ("sheila"
caught) and validation false-alarm rate. One line per run.

Run:  python compare_runs.py baseline epochs10 lr1e-2
      python compare_runs.py                 (all runs in runs/experiments.csv)
Out:  runs/compare.png  (or --out other.png)
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")                  # write a file, no window needed
import matplotlib.pyplot as plt

# fixed categorical order: run 1 is always blue, run 2 orange, ... (max 8)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e6e5e1", "#fcfcfb"
PANELS = [("train_loss", "Training loss", "lower is better"),
          ("val_recall", "Validation recall", "share of 'sheila' caught, higher is better"),
          ("val_false_alarm", "Validation false-alarm rate",
           "share of other words fired on, lower is better")]


def load(name):
    with open(os.path.join("runs", name, "history.csv"), newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", help="run names (default: all)")
    p.add_argument("--out", default=os.path.join("runs", "compare.png"))
    a = p.parse_args()

    names = a.names
    if not names:
        with open(os.path.join("runs", "experiments.csv"), newline="") as f:
            names = list(dict.fromkeys(r["name"] for r in csv.DictReader(f)))
    if len(names) > len(COLORS):
        raise SystemExit(f"max {len(COLORS)} runs per figure - pass the "
                         "names you want to compare")

    runs = {n: load(n) for n in names}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)
    for ax, (key, title, sub) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for k, (color, (name, hist)) in enumerate(zip(COLORS, runs.items())):
            ep = [h["epoch"] for h in hist]
            val = [h[key] for h in hist]
            # earlier runs drawn on top, and single-epoch runs get a bigger
            # dot, so a short run is not hidden under a longer one that
            # starts identically (same seed -> same first epoch)
            ax.plot(ep, val, color=color, linewidth=2, marker="o",
                    markersize=9 if len(ep) == 1 else 5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                    zorder=10 + len(runs) - k, label=name)
            if len(runs) <= 4:          # direct label at the line's end
                ax.annotate(name, (ep[-1], val[-1]), xytext=(6, 0),
                            textcoords="offset points", va="center",
                            fontsize=8, color=INK)
        ax.set_title(f"{title}\n", fontsize=11, color=INK, loc="left")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=MUTED)
        ax.set_xlabel("epoch", color=MUTED, fontsize=9)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        if key != "train_loss":
            ax.set_ylim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(8, len(runs)),
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(a.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {a.out}")

    # the same numbers as a table (final epoch of each run)
    print(f"\n{'run':16s} {'epochs':>6s} {'loss':>7s} {'recall':>7s} {'false_al':>8s}")
    for name, hist in runs.items():
        h = hist[-1]
        print(f"{name:16s} {int(h['epoch']):6d} {h['train_loss']:7.4f} "
              f"{h['val_recall']:7.3f} {h['val_false_alarm']:8.3f}")


if __name__ == "__main__":
    main()
