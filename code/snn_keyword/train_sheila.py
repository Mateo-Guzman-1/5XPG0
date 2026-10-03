#!/usr/bin/env python3
"""train_sheila.py — the ORIGINAL course skeleton, with only two changes:

  1. CHANGED: the fake SyntheticSpectrograms dataset is replaced by real
     audio from Google Speech Commands: "sheila" (label 1) vs other words
     (label 0).
  2. NEW: a test bench. After training, the network classifies every clip
     and the result for EACH clip is saved to a CSV file, plus a summary
     (accuracy, hits, misses, false alarms).

Everything else (network, encoding, training loop) is the skeleton, and the
default settings are the skeleton's, so a run with no options is the
BASELINE that training experiments (epochs, learning rate, surrogate
gradient, ...) are compared against.
Sections that differ from the skeleton are marked "# CHANGED" or "# NEW".

Run one experiment (anything not given keeps the skeleton value):
  python train_sheila.py --name epochs10 --epochs 10 --note "more epochs"
  python train_sheila.py --name lr1e-2 --lr 1e-2
  python train_sheila.py --name atan --surrogate atan --slope 2
  python train_sheila.py --help                     (all options)

Output per run, in runs/<name>/:
  config.json          every setting used (your design choices)
  model.pt             trained weights
  history.csv          train loss + validation numbers per epoch
  val_predictions.csv  what it decided, per validation clip
  summary.txt          the final numbers in words
plus ONE ROW per run appended to runs/experiments.csv: settings + results of
every experiment side by side (open it in Excel).

Validation vs test: compare experiments on the VALIDATION set. The test set
is only evaluated with --final, for the configuration you finally choose;
tuning on the test set would make its numbers too optimistic.
"""

import argparse # reads the experiment settings from the command line
import csv # writes result tables
import json # saves each run's settings (config.json)
import os
import time # timestamps + training time
import random #picks which "other word" clips to use
import tarfile #unpacks downloaded dataset from its compressed .tar.gz file
import urllib.request # downloads dataset from google's server
import wave # reads .wav audio files

import numpy as np #fast arrays of numbers, used for audio samples & spectogram math
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from features import clip_features   # NEW: 1 s of audio -> 16x16 spectrogram, loads spectogram function from features.py

# ----------------------------------------------------------------------
# Configuration (skeleton values; change ONE at a time per experiment)
# ----------------------------------------------------------------------
N_FREQ = 16          # spectrogram "frequency bins"
N_TIME = 16          # spectrogram time frames per sample
N_INPUT = N_FREQ * N_TIME
N_HIDDEN = 64
N_CLASSES = 2        # 0 = not "sheila", 1 = "sheila"
N_TIMESTEPS = 25     # simulation steps per sample (rate encoding)
BATCH = 16
EPOCHS = 1           # <-- skeleton value: the baseline is under-trained
LR = 1e-3
SEED = 0             # training seed: initial weights + shuffling order
OUT_DIR = "runs"
# (these defaults can be overridden per run from the command line, see main)

# NEW: dataset settings. These are DATA choices: keep them fixed while you
# experiment with training, so results stay comparable.
KEYWORD = "sheila"
NEG_RATIO = 3        # "other word" clips per "sheila" clip
DATA_SEED = 0        # which "other word" clips are picked. Separate from
                     # SEED on purpose: changing the training seed must not
                     # change the dataset.
DATA_DIR = "data"
DATASET_URL = ("http://download.tensorflow.org/data/"
               "speech_commands_v0.02.tar.gz")


# ----------------------------------------------------------------------
# CHANGED: real dataset instead of SyntheticSpectrograms
# ----------------------------------------------------------------------
def download_speech_commands():
    """Download + unpack Speech Commands (~2.3 GB) once; return its folder.

    The dataset is one folder per word (sheila/, yes/, no/, ...), each full
    of 1 s, 16 kHz .wav files from many different speakers.
    """
    root = os.path.join(DATA_DIR, "speech_commands")
    if os.path.isdir(os.path.join(root, KEYWORD)):
        return root                                    # already there
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(DATA_DIR, "speech_commands_v0.02.tar.gz")
    if not os.path.exists(archive):
        print(f"downloading {DATASET_URL} (~2.3 GB) ...")
        urllib.request.urlretrieve(DATASET_URL, archive)
    print("extracting ...")
    with tarfile.open(archive) as tar:
        tar.extractall(root)
    return root


def read_wav(path):
    """Read a 16-bit wav file -> numpy array of samples in [-1, 1].""" # 0 stays slienet, and ±1 is the loudest possible
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return data.astype(np.float32) / 32768.0


def list_split(root, split):
    """List the clips of one split ("train", "val" or "test").

    The dataset ships validation_list.txt and testing_list.txt, which put
    every SPEAKER in exactly one split. So the test set contains only voices
    the network never heard in training: a fair test of generalization.
    Returns a list of (relative path like "sheila/abc_nohash_0.wav", label).
    """
    def read_list(name):
        with open(os.path.join(root, name)) as f:
            return {line.strip() for line in f if line.strip()}

    val, test = read_list("validation_list.txt"), read_list("testing_list.txt")
    pos, neg = [], []
    for word in sorted(os.listdir(root)):
        folder = os.path.join(root, word)
        if not os.path.isdir(folder) or word.startswith("_"):
            continue                       # skip _background_noise_ folder
        for fn in sorted(os.listdir(folder)):
            rel = f"{word}/{fn}"
            s = "val" if rel in val else "test" if rel in test else "train"
            if s != split:
                continue
            # the LABEL comes from the folder name: that is the "answer"
            (pos if word == KEYWORD else neg).append((rel, int(word == KEYWORD)))

    # keep every "sheila" clip, but only NEG_RATIO x as many other words
    # (there are ~50x more other words; using all would swamp "sheila").
    # Fixed seed -> the same clips are picked on every run.
    neg = random.Random(DATA_SEED).sample(neg,
                                          min(len(neg), NEG_RATIO * len(pos)))
    return pos + neg


class SpeechCommandsSheila(torch.utils.data.Dataset):
    """Real 16x16 spectrograms: "sheila" (1) vs other words (0).

    Same interface as the skeleton's SyntheticSpectrograms (x in [0, 1],
    shape [16, 16]), so the network and training loop need no changes.
    Turning audio into spectrograms is slow, so the result is cached in
    data/ and later runs load it in seconds.
    """

    def __init__(self, split):
        root = download_speech_commands()
        self.items = list_split(root, split)        # [(path, label), ...]
        cache = os.path.join(DATA_DIR, f"sheila_{split}_skel.pt")
        if os.path.exists(cache):
            self.x, self.y = torch.load(cache)
        else:
            print(f"computing {len(self.items)} spectrograms for '{split}' ...")
            x = np.stack([clip_features(read_wav(os.path.join(root, p)))
                          for p, _ in self.items])   # [n, 16, 16]
            self.x = torch.from_numpy(x)
            self.y = torch.tensor([lab for _, lab in self.items])
            torch.save((self.x, self.y), cache)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # also return i, so the test bench can find the file name later
        return self.x[i], self.y[i], i


# ----------------------------------------------------------------------
# Encoding + network: UNCHANGED from the skeleton, except that the
# surrogate gradient can now be chosen (a TRAINING choice, it only affects
# the backward pass, never the spikes themselves)
# ----------------------------------------------------------------------
def make_surrogate(name, slope):
    """Surrogate-gradient function used in the backward pass.

    fast_sigmoid is the skeleton default (slope 25). 'slope' sets how steep
    the smooth curve is: steeper = closer to the real step, but fewer neurons
    get a learning signal.
    """
    if name == "fast_sigmoid":
        return surrogate.fast_sigmoid(slope=slope)
    if name == "sigmoid":
        return surrogate.sigmoid(slope=slope)
    if name == "atan":
        return surrogate.atan(alpha=slope)
    raise ValueError(f"unknown surrogate {name}")


def rate_encode(x, n_steps):
    """Rate encoding: repeat the (analog) input for n_steps steps.

    Each input value is treated as a constant drive for every time step.
    A Bernoulli/rate encoder or latency encoder would be a design change.
    """
    return x.unsqueeze(0).repeat(n_steps, 1, 1)   # [T, B, N_INPUT]


class SpikeMLP(nn.Module):
    def __init__(self, spike_grad=None):
        super().__init__()
        if spike_grad is None:                      # skeleton default
            spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(N_INPUT, N_HIDDEN)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc2 = nn.Linear(N_HIDDEN, N_CLASSES)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        # x: [T, B, N_INPUT]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for step in range(x.shape[0]):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec, dim=0)      # [T, B, N_CLASSES]


# ----------------------------------------------------------------------
# NEW: test bench — run the network on a dataset and record every decision
# ----------------------------------------------------------------------
@torch.no_grad()                  # testing only: no gradients needed
def run_testbench(net, ds):
    """Classify every clip in ds. Returns (metrics dict, per-clip rows).

    HOW A DECISION IS MADE: the clip's spectrogram is fed in for 25 time
    steps; we count the spikes of output neuron 0 ("not sheila") and output
    neuron 1 ("sheila"). Whichever spiked more is the answer (argmax). On a
    tie, argmax picks neuron 0, i.e. "not sheila".
    """
    net.eval()
    rows = []
    loader = torch.utils.data.DataLoader(ds, batch_size=256)
    for x, y, idx in loader:
        x = x.reshape(x.shape[0], -1)                 # [B, N_INPUT]
        counts = net(rate_encode(x, N_TIMESTEPS)).sum(dim=0)   # [B, 2]
        pred = counts.argmax(dim=1)                   # 0 or 1 per clip
        for c, p, t, i in zip(counts.tolist(), pred.tolist(),
                              y.tolist(), idx.tolist()):
            path = ds.items[i][0]
            rows.append({"file": path, "word": path.split("/")[0],
                         "true": t, "predicted": p,
                         "spikes_not_sheila": int(c[0]),
                         "spikes_sheila": int(c[1]),
                         "correct": int(p == t)})
    net.train()

    # the four possible outcomes for a keyword detector
    tp = sum(r["true"] == 1 and r["predicted"] == 1 for r in rows)  # hit
    fn = sum(r["true"] == 1 and r["predicted"] == 0 for r in rows)  # miss
    fp = sum(r["true"] == 0 and r["predicted"] == 1 for r in rows)  # false alarm
    tn = sum(r["true"] == 0 and r["predicted"] == 0 for r in rows)  # correct reject
    metrics = {
        "accuracy": (tp + tn) / len(rows),
        "recall": tp / max(1, tp + fn),        # share of "sheila"s caught
        "false_alarm": fp / max(1, fp + tn),   # share of other words fired on
        "precision": tp / max(1, tp + fp),     # when it says sheila, how often right
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }
    return metrics, rows


def fmt(m):
    return (f"acc={m['accuracy']:.3f}  recall={m['recall']:.3f}  "
            f"false_alarm={m['false_alarm']:.3f}")


def parse_args():
    """NEW: every TRAINING choice can be set per run; defaults = skeleton."""
    p = argparse.ArgumentParser(description="train the sheila SNN once")
    p.add_argument("--name", default="baseline",
                   help="run name = output folder runs/<name>/")
    p.add_argument("--note", default="",
                   help="why you ran this (hypothesis), saved with the results")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR, help="learning rate")
    p.add_argument("--batch", type=int, default=BATCH, help="batch size")
    p.add_argument("--surrogate", default="fast_sigmoid",
                   choices=["fast_sigmoid", "sigmoid", "atan"])
    p.add_argument("--slope", type=float, default=25.0,
                   help="surrogate steepness (atan: alpha, default 2 there)")
    p.add_argument("--scheduler", default="none", choices=["none", "cosine"],
                   help="cosine: learning rate shrinks smoothly to 0 by the "
                        "last epoch")
    p.add_argument("--seed", type=int, default=SEED,
                   help="training seed (initial weights, shuffle order)")
    p.add_argument("--final", action="store_true",
                   help="also evaluate the TEST set (only for your final, "
                        "chosen configuration)")
    return p.parse_args()


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = os.path.join(OUT_DIR, args.name)                # NEW
    os.makedirs(run_dir, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k != "final"}
    config.update(hidden=N_HIDDEN, timesteps=N_TIMESTEPS, beta=0.9,
                  encoding="rate", neg_ratio=NEG_RATIO, data_seed=DATA_SEED)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # CHANGED: three real splits instead of one fake dataset
    ds = SpeechCommandsSheila("train")      # learn from this
    val_ds = SpeechCommandsSheila("val")    # judge settings on this
    print(f"train {len(ds)}  val {len(val_ds)} clips   run '{args.name}'")
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch,
                                         shuffle=True)

    net = SpikeMLP(make_surrogate(args.surrogate, args.slope))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
             if args.scheduler == "cosine" else None)
    loss_fn = nn.CrossEntropyLoss()

    history = []                                              # NEW
    t_start = time.time()
    net.train()
    for epoch in range(args.epochs):
        total = 0.0
        for x, y, _ in loader:          # CHANGED: dataset also returns index
            x = x.reshape(x.shape[0], -1)             # [B, N_INPUT]
            spikes = net(rate_encode(x, N_TIMESTEPS))  # [T, B, N_CLASSES]
            # decode: use the spike count (firing rate) over time
            out = spikes.sum(dim=0)                    # [B, N_CLASSES]
            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * y.shape[0]

        # NEW: after every epoch, check on the validation set
        val_m, _ = run_testbench(net, val_ds)
        history.append({"epoch": epoch + 1, "train_loss": total / len(ds),
                        "lr": opt.param_groups[0]["lr"],
                        **{f"val_{k}": v for k, v in val_m.items()}})
        print(f"epoch {epoch+1}/{args.epochs}  loss={total/len(ds):.4f}  "
              f"val: {fmt(val_m)}")
        if sched:
            sched.step()                    # shrink the learning rate

    train_minutes = (time.time() - t_start) / 60
    torch.save(net.state_dict(), os.path.join(run_dir, "model.pt"))
    write_csv(os.path.join(run_dir, "history.csv"), history)   # NEW

    # NEW: test bench on the validation set, every clip recorded
    sections = [("validation", *run_testbench(net, val_ds))]
    write_csv(os.path.join(run_dir, "val_predictions.csv"), sections[0][2])
    if args.final:                          # the test set: only at the end
        test_ds = SpeechCommandsSheila("test")
        sections.append(("TEST", *run_testbench(net, test_ds)))
        write_csv(os.path.join(run_dir, "test_predictions.csv"),
                  sections[1][2])

    summary = f"run: {args.name}   note: {args.note}\nsettings: {config}\n"
    for label, m, rows in sections:
        # which words are most often mistaken for "sheila"?
        fa_words = {}
        for r in rows:
            if r["true"] == 0 and r["predicted"] == 1:
                fa_words[r["word"]] = fa_words.get(r["word"], 0) + 1
        worst = sorted(fa_words.items(), key=lambda kv: -kv[1])[:5]
        silent = sum(r["spikes_not_sheila"] == 0 and r["spikes_sheila"] == 0
                     for r in rows)
        summary += (
            f"\n[{label}] {len(rows)} clips ({m['tp'] + m['fn']} sheila, "
            f"{m['fp'] + m['tn']} other)\n"
            f"accuracy     {m['accuracy']:.3f}\n"
            f"recall       {m['recall']:.3f}   (sheila caught: {m['tp']}, "
            f"missed: {m['fn']})\n"
            f"false alarm  {m['false_alarm']:.3f}   (other words fired on: "
            f"{m['fp']} of {m['fp'] + m['tn']})\n"
            f"precision    {m['precision']:.3f}\n"
            f"clips with no output spikes at all: {silent}\n"
            f"most confused words: {worst}\n")
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(summary)
    print("\n--- TEST BENCH ---\n" + summary)

    # NEW: one row per run in the shared experiment log
    val_m = sections[0][1]
    best = max(history, key=lambda h: h["val_accuracy"])
    row = {"date": time.strftime("%Y-%m-%d %H:%M"), **config,
           "val_accuracy": round(val_m["accuracy"], 4),
           "val_recall": round(val_m["recall"], 4),
           "val_false_alarm": round(val_m["false_alarm"], 4),
           "best_val_acc_epoch": best["epoch"],
           "final_train_loss": round(history[-1]["train_loss"], 4),
           "train_minutes": round(train_minutes, 1)}
    if args.final:
        tm = sections[1][1]
        row.update(test_accuracy=round(tm["accuracy"], 4),
                   test_recall=round(tm["recall"], 4),
                   test_false_alarm=round(tm["false_alarm"], 4))
    log = os.path.join(OUT_DIR, "experiments.csv")
    fields = list(row.keys())
    if os.path.exists(log):                 # keep earlier columns + order
        with open(log, newline="") as f:
            old = list(csv.DictReader(f))
        fields = list(dict.fromkeys((list(old[0].keys()) if old else [])
                                    + fields))
    else:
        old = []
    write_csv(log, [{k: r.get(k, "") for k in fields} for r in old + [row]])
    print(f"saved everything in {run_dir}/ and added a row to {log}")


if __name__ == "__main__":
    main()
