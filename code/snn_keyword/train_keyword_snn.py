#!/usr/bin/env python3
"""train_keyword_snn.py — GROUP 2 single-keyword detection: "sheila".

Pipeline:

    1 s audio  ->  16x16 spectrogram (features.py)  ->  spike encoding
    ->  2-layer SNN (snnTorch)  ->  training loop  ->  saved weights

Data: Google Speech Commands v0.02, which contains the word "sheila".
Class 1 = "sheila", class 0 = any other word, plus background noise and
silence at random volumes. The first run downloads the
dataset (~2.3 GB) into data/ and caches the computed features, so later runs
start in seconds.

What you should still change (design choices!):
  1. INPUT ENCODING   — try rate, latency (time-to-first-spike), or delta.
  2. TOPOLOGY         — number of hidden neurons, recurrent layers, ...
  3. TRAINING         — epochs, LR schedule, surrogate gradients.
  4. NEGATIVES        — NEG_RATIO, adding silence / background noise.

Run:  python train_keyword_snn.py
Output: runs/keyword_snn.pt  (state_dict), training loss and test metrics.
"""

import os
import random
import tarfile
import urllib.request
import wave

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from features import N_MELS, N_TIME, SAMPLE_RATE, clip_features

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
KEYWORD = "sheila"
N_INPUT = N_MELS * N_TIME
N_HIDDEN = 64
N_CLASSES = 2        # {not-keyword, keyword}
N_TIMESTEPS = 25     # simulation steps per sample (rate encoding)
BATCH = 64
NEG_RATIO = 3        # non-keyword clips per keyword clip in each split
EPOCHS = 15
LR = 1e-3
SEED = 0
OUT_DIR = "runs"
DATA_DIR = "data"
DATASET_URL = ("http://download.tensorflow.org/data/"
               "speech_commands_v0.02.tar.gz")


# ----------------------------------------------------------------------
# Speech Commands dataset: KEYWORD vs everything else
# ----------------------------------------------------------------------
def download_speech_commands():
    """Download + extract Speech Commands v0.02 into DATA_DIR (once)."""
    root = os.path.join(DATA_DIR, "speech_commands")
    if os.path.isdir(os.path.join(root, KEYWORD)):
        return root
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
    """16-bit mono PCM wav -> float32 in [-1, 1]."""
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return data.astype(np.float32) / 32768.0


def split_files(root):
    """Return {train, val, test} -> list of (relative path, label)."""
    def read_list(name):
        with open(os.path.join(root, name)) as f:
            return {line.strip() for line in f if line.strip()}

    val, test = read_list("validation_list.txt"), read_list("testing_list.txt")
    splits = {"train": [], "val": [], "test": []}
    for word in sorted(os.listdir(root)):
        d = os.path.join(root, word)
        if not os.path.isdir(d) or word.startswith("_"):
            continue                              # skip _background_noise_
        for fn in os.listdir(d):
            rel = f"{word}/{fn}"
            split = "val" if rel in val else "test" if rel in test else "train"
            splits[split].append((rel, int(word == KEYWORD)))

    # keep all keyword clips, subsample the (huge) rest to NEG_RATIO x
    rng = random.Random(SEED)
    for split, items in splits.items():
        pos = [it for it in items if it[1] == 1]
        neg = [it for it in items if it[1] == 0]
        neg = rng.sample(neg, min(len(neg), NEG_RATIO * len(pos)))
        splits[split] = pos + neg
    return splits


def noise_clips(root, split, n):
    """n non-speech 1 s clips (label 0): background noise at random volume.

    Without these the net never sees silence or room noise and happily calls
    them "sheila". Each noise file is cut 80/10/10 in time so the splits
    never share audio. Volumes go down to (near) digital silence.
    """
    lo, hi = {"train": (0.0, 0.8), "val": (0.8, 0.9), "test": (0.9, 1.0)}[split]
    bg = os.path.join(root, "_background_noise_")
    wavs = [read_wav(os.path.join(bg, f)) for f in sorted(os.listdir(bg))
            if f.endswith(".wav")]
    rng = np.random.default_rng(SEED + len(split))
    clips = []
    for i in range(n):
        w = wavs[i % len(wavs)]
        a, b = int(lo * len(w)), int(hi * len(w)) - SAMPLE_RATE
        start = rng.integers(a, b)
        gain = 0.0 if i % 10 == 0 else 10 ** rng.uniform(-4, 0)
        clips.append(w[start:start + SAMPLE_RATE] * gain)
    return clips


def load_split(root, split, items, cache):
    """Compute (or load cached) features for one split -> TensorDataset."""
    if os.path.exists(cache):
        x, y = torch.load(cache)
    else:
        n_noise = sum(lab for _, lab in items)      # as many as keyword clips
        print(f"computing features for {len(items)} clips + {n_noise} noise "
              f"-> {cache}")
        wavs = [read_wav(os.path.join(root, rel)) for rel, _ in items]
        wavs += noise_clips(root, split, n_noise)
        x = torch.from_numpy(np.stack([clip_features(w) for w in wavs]))
        y = torch.tensor([lab for _, lab in items] + [0] * n_noise)
        torch.save((x, y), cache)
    return torch.utils.data.TensorDataset(x, y)


def rate_encode(x, n_steps):
    """Rate encoding: repeat the (analog) input for n_steps steps.

    Each input value is treated as a constant drive for every time step.
    A Bernoulli/rate encoder or latency encoder would be a design change.
    """
    return x.unsqueeze(0).repeat(n_steps, 1, 1)   # [T, B, N_INPUT]


# ----------------------------------------------------------------------
# 2-layer spiking MLP
# ----------------------------------------------------------------------
class SpikeMLP(nn.Module):
    def __init__(self):
        super().__init__()
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


@torch.no_grad()
def evaluate(net, loader):
    """Accuracy, keyword recall (hit rate) and false-alarm rate."""
    net.eval()
    tp = fp = tn = fn = 0
    for x, y in loader:
        x = x.reshape(x.shape[0], -1)
        pred = net(rate_encode(x, N_TIMESTEPS)).sum(dim=0).argmax(dim=1)
        tp += int(((pred == 1) & (y == 1)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
    net.train()
    return {"acc": (tp + tn) / max(1, tp + tn + fp + fn),
            "recall": tp / max(1, tp + fn),
            "false_alarm": fp / max(1, fp + tn)}


def fmt(m):
    return (f"acc={m['acc']:.3f}  {KEYWORD}-recall={m['recall']:.3f}  "
            f"false-alarm={m['false_alarm']:.3f}")


def main():
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    root = download_speech_commands()
    splits = split_files(root)
    ds = {s: load_split(root, s, items,
                        os.path.join(DATA_DIR, f"{KEYWORD}_{s}_features.pt"))
          for s, items in splits.items()}
    loader = torch.utils.data.DataLoader(ds["train"], batch_size=BATCH,
                                         shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds["val"], batch_size=256)
    test_loader = torch.utils.data.DataLoader(ds["test"], batch_size=256)
    print({s: len(d) for s, d in ds.items()})

    net = SpikeMLP()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    # weight the rarer keyword class so the net cannot just say "no" always
    y_train = ds["train"].tensors[1]
    pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))

    net.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in loader:
            x = x.reshape(x.shape[0], -1)             # [B, N_INPUT]
            spikes = net(rate_encode(x, N_TIMESTEPS))  # [T, B, N_CLASSES]
            # decode: use the spike count (firing rate) over time
            out = spikes.sum(dim=0)                    # [B, N_CLASSES]
            loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * y.shape[0]
        print(f"epoch {epoch+1}/{EPOCHS}  loss={total/len(ds['train']):.4f}  "
              f"val: {fmt(evaluate(net, val_loader))}")

    print(f"test: {fmt(evaluate(net, test_loader))}")
    path = os.path.join(OUT_DIR, "keyword_snn.pt")
    torch.save(net.state_dict(), path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
