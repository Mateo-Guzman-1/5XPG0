"""Surrogate-gradient training, QAT, and validation-only checkpoint selection."""
import argparse
import os
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from model import SpikeMLP, integer_forward, quantize, metrics, choose_threshold

ROOT = Path(__file__).resolve().parent


def evaluate(model, x, batch=512):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(b)[0] for b in x.split(batch)]).cpu().numpy()


def run(args, seed, encoding):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    cache = np.load(args.data / 'features.npz')
    x = cache['x']; y = cache['y']; split = cache['split']
    xt = torch.tensor(x[split == 0], device=device, dtype=torch.float32) / 255
    yt = torch.tensor(y[split == 0], device=device, dtype=torch.long)
    xv_np, yv = x[split == 1], y[split == 1]
    n_orig = len(yt)
    if args.aug_fraction > 0:
        # Edited "yes" clips from augment.py (confusable endings, window-edge cuts,
        # shifted positives); they inherit the official speaker-disjoint splits.
        aug = np.load(args.data / 'augment.npz')
        use = np.isin(aug['kind'], args.aug_kinds)
        tr, va = use & (aug['split'] == 0), use & (aug['split'] == 1)
        xt = torch.cat([xt, torch.tensor(aug['x'][tr], device=device, dtype=torch.float32) / 255])
        yt = torch.cat([yt, torch.tensor(aug['y'][tr], device=device, dtype=torch.long)])
        xv_np = np.concatenate([xv_np, aug['x'][va]]); yv = np.concatenate([yv, aug['y'][va]])
    xv = torch.tensor(xv_np, device=device, dtype=torch.float32) / 255
    model = SpikeMLP(args.hidden, args.steps, encoding).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr / 10)
    original = torch.arange(len(yt), device=device) < n_orig
    pools = [torch.where(original & (yt == c))[0] for c in (1, 0)]
    aug_pools = [torch.where(~original & (yt == c))[0] for c in (1, 0)]
    half = args.batch // 2
    n_aug = [int(half * args.aug_fraction) if len(a) else 0 for a in aug_pools]
    best, best_state, history = -1, None, []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        model.qat = epoch >= args.epochs - args.qat_epochs
        total = 0.
        for _ in range(args.samples_per_epoch // args.batch):
            # Each class fills half the batch; aug_fraction of that half is augmented.
            ids = torch.cat([pool[torch.randint(len(pool), (n,), device=device)]
                             for c in range(2) for pool, n in ((pools[c], half - n_aug[c]), (aug_pools[c], n_aug[c])) if n])
            xb, yb = xt[ids].clone(), yt[ids]
            xb = (xb + torch.randn(len(xb), 1, device=device) * .025 + torch.randn_like(xb) * .01).clamp(0, 1)
            xb = torch.round(xb * 255) / 255
            silent = torch.rand(len(xb), device=device) < .04
            xb[silent] = 0; yb[silent] = 0
            logits, spikes = model(xb)
            loss = nn.functional.cross_entropy(logits, yb) + 1e-4 * spikes.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total += loss.item()
        scheduler.step()
        logits = evaluate(model, xv)
        margin = logits[:, 1] - logits[:, 0]
        threshold = choose_threshold(yv, margin)
        result = metrics(yv, margin >= threshold)
        result.update(epoch=epoch + 1, loss=total / (args.samples_per_epoch // args.batch), qat=model.qat)
        history.append(result)
        if model.qat and result['f1'] > best:
            best, best_state = result['f1'], copy.deepcopy(model.state_dict())
        print(f'{encoding} seed={seed} epoch={epoch+1}/{args.epochs} loss={result["loss"]:.4f} val_f1={result["f1"]:.4f} recall={result["recall"]:.4f} fpr={result["fpr"]:.4f} qat={model.qat}', flush=True)
    model.load_state_dict(best_state)
    q = quantize(model)
    qlogits, _ = integer_forward(xv_np, q)
    margin = qlogits[:, 1] - qlogits[:, 0]
    q['decision_threshold'] = int(choose_threshold(yv, margin))
    out = args.out / f'{encoding}_seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'model.npz', **q)
    torch.save({'state_dict': best_state, 'hidden': args.hidden, 'steps': args.steps, 'encoding': encoding}, out / 'checkpoint.pt')
    summary = {'seed': seed, 'encoding': encoding, 'hidden': args.hidden, 'steps': args.steps,
               'aug_fraction': args.aug_fraction, 'aug_kinds': args.aug_kinds, 'augmented_train': len(yt) - n_orig,
               'epochs': args.epochs, 'samples_per_epoch': args.samples_per_epoch,
               'training_seconds': time.perf_counter() - start, 'device': str(device),
               'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
               'torch': torch.__version__, 'validation': metrics(yv, margin >= q['decision_threshold']),
               'decision_threshold': q['decision_threshold'], 'history': history}
    (out / 'training.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--out', type=Path, default=ROOT / 'runs')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--encodings', nargs='+', choices=['current', 'rate'], default=['current', 'rate'])
    p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--qat-epochs', type=int, default=10)
    p.add_argument('--samples-per-epoch', type=int, default=16384)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--aug-fraction', type=float, default=0,
                   help='Share of each class half-batch drawn from data/augment.npz (augment.py); '
                        '0 = release training, 0.3 = augmented trial model')
    p.add_argument('--aug-kinds', nargs='+', default=['ts', 'ch', 'sh', 'cut', 'tail', 'head', 'shift'],
                   choices=['ts', 'ch', 'sh', 'cut', 'tail', 'head', 'shift'], help='augment.py variants to train on')
    a = p.parse_args()
    if not 0 < a.qat_epochs <= a.epochs or a.batch % 2 or a.samples_per_epoch < a.batch or not 0 <= a.aug_fraction < 1:
        p.error('Require 0 < qat_epochs <= epochs, even batch, samples >= batch, 0 <= aug_fraction < 1')
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    for encoding in a.encodings:
        for seed in a.seeds:
            run(a, seed, encoding)


if __name__ == '__main__':
    main()
