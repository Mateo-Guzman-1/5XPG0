"""Phase 1 gate: the release's dense SNN, retrained on the new data path only.

Same model (SpikeMLP, 64 hidden, current encoding, 12 steps, QAT), same 1 s
window with 32 pooled bins and the same integer export, so any change against
the release comes from data and augmentation alone (IMPLEMENTATION_PLAN.md,
Phase 1 gate). What changes:
  - clips from Speech Commands, MSWC and Piper TTS (prepare_multicorpus.py),
    sampled with equal weight per corpus within each class half-batch;
  - negatives also from 1 s crops of LibriSpeech running speech and from
    noise-only "silence";
  - the words at a random position (positives keep the whole word in the window);
  - augment_online.py on the GPU (speed, simulated rooms, noise, level,
    training microphones, VTLP).
Checkpoint and decision threshold: best F1 on the packed validation clips
(Speech Commands validation, MSWC dev, TTS validation), clean. The stream
threshold is then set by tune_stream.py, as for the release.
"""
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from augment_online import Augmenter, FrontEnd
from model import SpikeMLP, choose_threshold, integer_forward, metrics, quantize

ROOT = Path(__file__).resolve().parent
MULTI = ROOT / 'data/multi'
SR = 16000


def load_split(split, corpora):
    rows = [r for r in csv.DictReader(open(MULTI / 'manifest.csv', encoding='utf-8')) if r['split'] == split]
    clips = np.load(MULTI / f'clips_{split}.npy', mmap_mode='r')
    rows = [r for r in rows if r['corpus'] in corpora]
    idx = np.array([int(r['row']) for r in rows])
    return clips, idx, np.array([r['word'] == 'yes' for r in rows]), np.array([r['corpus'] for r in rows])


def shift(wave, max_shift, gen):
    """Move each 1 s clip by up to +-max_shift samples, zero-filled."""
    b, n = wave.shape
    s = ((torch.rand(b, generator=gen, device=wave.device) * 2 - 1) * max_shift).long()
    pos = torch.arange(n, device=wave.device)[None] - s[:, None]
    ok = (pos >= 0) & (pos < n)
    return torch.where(ok, wave.gather(1, pos.clamp(0, n - 1)), 0.)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpora', nargs='+', default=['sc', 'mswc', 'tts'])
    p.add_argument('--no-libri', action='store_true')
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--qat-epochs', type=int, default=10)
    p.add_argument('--samples-per-epoch', type=int, default=32768)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--out', type=Path, default=ROOT / 'runs_phase1')
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device('cuda')
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    clips, idx, yes, corpus = load_split('train', a.corpora)
    pools = {(c, y): idx[(corpus == c) & (yes == y)] for c in a.corpora for y in (True, False)}
    pools = {k: v for k, v in pools.items() if len(v)}
    speech = None if a.no_libri else np.load(MULTI / 'speech_train.npy', mmap_mode='r')
    noise = np.load(MULTI / 'noise_train.npy')
    aug = Augmenter(dev, noise, seed=a.seed)
    front = FrontEnd(dev)
    # Validation: clean pooled features of the packed validation clips.
    vclips, vidx, vyes, vcorp = load_split('validation', a.corpora)
    with torch.no_grad():
        vx = torch.cat([front.pooled(front.mel(front.power(torch.tensor(vclips[vidx[i:i + 2048]].astype(np.float32) / 32768, device=dev))))
                        for i in range(0, len(vidx), 2048)]) / 255
    vx_np = (vx * 255).round().cpu().numpy().astype(np.uint8)
    model = SpikeMLP(a.hidden, 12, 'current').to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.lr / 10)
    half = a.batch // 2
    pos_keys = [k for k in pools if k[1]]
    neg_keys = [k for k in pools if not k[1]]
    best, best_state, history = -1, None, []
    start = time.perf_counter()
    for epoch in range(a.epochs):
        model.train()
        model.qat = epoch >= a.epochs - a.qat_epochs
        total = 0.
        for _ in range(a.samples_per_epoch // a.batch):
            # Positives: equal share per corpus. Negatives: per corpus, plus LibriSpeech and silence.
            ids = [rng.choice(pools[k], half // len(pos_keys)) for k in pos_keys]
            n_extra = 0 if speech is None else half // 5
            n_sil = half // 16
            n_neg = half - n_extra - n_sil
            ids += [rng.choice(pools[k], n_neg // len(neg_keys)) for k in neg_keys]
            ids = np.concatenate(ids)
            n_pos = sum(half // len(pos_keys) for _ in pos_keys)
            wave = torch.tensor(clips[np.sort(ids)].astype(np.float32) / 32768, device=dev)
            order = np.argsort(np.argsort(ids))  # undo the sort used for fast memmap reads
            wave = wave[torch.tensor(order, device=dev)]
            y = torch.zeros(len(wave), dtype=torch.long, device=dev); y[:n_pos] = 1
            # Positives stay whole in the window; negatives may sit anywhere.
            wave = torch.cat([shift(wave[:n_pos], int(.12 * SR), gen), shift(wave[n_pos:], int(.45 * SR), gen)])
            extra = []
            if speech is not None:
                s = rng.integers(0, len(speech) - SR, n_extra)
                extra.append(torch.tensor(np.stack([speech[i:i + SR] for i in s]).astype(np.float32) / 32768, device=dev))
            extra.append(torch.zeros(n_sil, SR, device=dev))
            wave = torch.cat([wave] + extra)
            y = torch.cat([y, torch.zeros(len(wave) - len(y), dtype=torch.long, device=dev)])
            silence = torch.zeros(len(wave), dtype=torch.bool, device=dev); silence[-n_sil:] = True
            with torch.no_grad():
                if a.no_augment:
                    x = front.pooled(front.mel(front.power(wave)))
                else:
                    x = aug(wave, labels_silence=silence, pooled=True)
            logits, spikes = model(x / 255)
            loss = nn.functional.cross_entropy(logits, y) + 1e-4 * spikes.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total += loss.item()
        sched.step()
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(b)[0] for b in vx.split(1024)]).cpu().numpy()
        margin = logits[:, 1] - logits[:, 0]
        th = choose_threshold(vyes.astype(int), margin)
        r = metrics(vyes, margin >= th)
        r.update(epoch=epoch + 1, loss=total / (a.samples_per_epoch // a.batch), qat=model.qat)
        r['per_corpus_recall'] = {c: round(float((margin[(vcorp == c) & vyes] >= th).mean()), 4) for c in a.corpora}
        history.append(r)
        if model.qat and r['f1'] > best:
            best, best_state = r['f1'], copy.deepcopy(model.state_dict())
        print(f"epoch {epoch + 1}/{a.epochs} loss={r['loss']:.4f} val_f1={r['f1']:.4f} recall={r['recall']:.4f} "
              f"fpr={r['fpr']:.4f} {r['per_corpus_recall']} qat={model.qat}", flush=True)
    model.load_state_dict(best_state)
    q = quantize(model)
    s, _ = integer_forward(vx_np, q)
    margin = s[:, 1] - s[:, 0]
    q['decision_threshold'] = int(choose_threshold(vyes.astype(int), margin))
    name = f"dense_{'-'.join(a.corpora)}{'' if a.no_libri else '+libri'}{'_noaug' if a.no_augment else ''}_seed{a.seed}"
    out = a.out / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'model.npz', **q)
    summary = dict(vars(a), out=str(out), training_seconds=time.perf_counter() - start,
                   validation=metrics(vyes, margin >= q['decision_threshold']),
                   decision_threshold=q['decision_threshold'], history=history)
    (out / 'training.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({'out': str(out), 'validation_f1': summary['validation']['f1']}), flush=True)


if __name__ == '__main__':
    main()
