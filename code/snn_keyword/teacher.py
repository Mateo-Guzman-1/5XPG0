"""Phase 2: BC-ResNet-8 teacher [8] for distillation (never deployed).

Architecture as in the authors' reference implementation
(github.com/Qualcomm-AI-research/bcresnet): a 5x5 head with frequency stride
2, four stages of broadcasted residual blocks (2, 2, 4, 4 blocks; frequency
stride in stages 1 and 2; time dilation 1, 2, 4, 8), sub-spectral norm with
5 sub-bands, and a depthwise 5x5 classifier. Width tau = 8 (base 64 channels,
about 320k parameters).

Input: 40-band log-mel, 30 ms Hann window zero-padded to 512, 10 ms hop,
centred frames (101 per second). The window and FFT size match features.py,
so the augmenter's training microphones apply to both. The teacher sees the
same augmented audio as the student (augment_online.py); soft labels are
computed online during student training, so they cover every augmented
example (IMPLEMENTATION_PLAN.md asks for exported soft labels; online is
equivalent and has no storage cost).

Classes: 35 Speech Commands words, _unknown_, _silence_. Done-when: >= 95%
on the Speech Commands 35-word validation split.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from augment_online import Augmenter
from features import SAMPLE_RATE
from multicorpus import CLASSES, ClipSampler

ROOT = Path(__file__).resolve().parent


class SubSpectralNorm(nn.Module):
    def __init__(self, channels, sub_bands=5):
        super().__init__()
        self.s = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands)

    def forward(self, x):
        n, c, f, t = x.shape
        return self.bn(x.reshape(n, c * self.s, f // self.s, t)).reshape(n, c, f, t)


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, idx, kernel=3, stride=1, groups=1, dilation=False, act='relu', ssn=False):
        super().__init__()
        kernel = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        rate = [2 ** idx if dilation and k > 1 else 1 for k in kernel]
        pad = [r * (k - 1) // 2 for k, r in zip(kernel, rate)]
        self.conv = nn.Conv2d(cin, cout, kernel, stride, pad, rate, groups, bias=False)
        self.norm = SubSpectralNorm(cout) if ssn else nn.BatchNorm2d(cout)
        self.act = {'relu': nn.ReLU(), 'swish': nn.SiLU(), None: nn.Identity()}[act]

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BCResBlock(nn.Module):
    def __init__(self, cin, cout, idx, stride=False):
        super().__init__()
        self.transition = cin != cout
        layers = []
        if self.transition:
            layers.append(ConvBNReLU(cin, cout, idx, 1, 1))
            cin = cout
        layers.append(ConvBNReLU(cin, cout, idx, (3, 1), (2, 1) if stride else (1, 1), groups=cin, act=None, ssn=True))
        self.f2 = nn.Sequential(*layers)
        self.f1 = nn.Sequential(ConvBNReLU(cout, cout, idx, (1, 3), 1, groups=cout, dilation=True, act='swish'),
                                nn.Conv2d(cout, cout, 1, bias=False), nn.Dropout2d(.1))

    def forward(self, x):
        shortcut = x
        x = self.f2(x)
        y = self.f1(x.mean(2, keepdim=True)) + x
        return F.relu(y if self.transition else y + shortcut)


class BCResNet(nn.Module):
    def __init__(self, base=64, classes=len(CLASSES)):
        super().__init__()
        n, strided = [2, 2, 4, 4], (1, 2)
        c = [base * 2, base, int(base * 1.5), base * 2, int(base * 2.5), base * 4]
        self.head = nn.Sequential(nn.Conv2d(1, c[0], 5, (2, 1), 2, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU())
        blocks = []
        for idx, k in enumerate(n):
            blocks.append(BCResBlock(c[idx], c[idx + 1], idx, idx in strided))
            blocks += [BCResBlock(c[idx + 1], c[idx + 1], idx) for _ in range(k - 1)]
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Conv2d(c[-2], c[-2], 5, bias=False, groups=c[-2], padding=(0, 2)),
            nn.Conv2d(c[-2], c[-1], 1, bias=False), nn.BatchNorm2d(c[-1]), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c[-1], classes, 1))

    def forward(self, x):
        return self.classifier(self.blocks(self.head(x))).flatten(1)


class TeacherFrontEnd:
    """40-band log-mel, centred frames; mic: training microphone per example (-1 = none)."""
    def __init__(self, device, n_mels=40):
        self.window = torch.hann_window(480, periodic=True, device=device)
        freqs = np.fft.rfftfreq(512, 1 / SAMPLE_RATE)
        mel = lambda hz: 2595 * np.log10(1 + hz / 700)
        pts = 700 * (10 ** (np.linspace(mel(20), mel(8000), n_mels + 2) / 2595) - 1)
        bank = np.maximum(0, np.minimum((freqs[None] - pts[:-2, None]) / (pts[1:-1, None] - pts[:-2, None]),
                                        (pts[2:, None] - freqs[None]) / (pts[2:, None] - pts[1:-1, None])))
        self.bank = torch.tensor(bank, dtype=torch.float32, device=device)

    def __call__(self, wave, augmenter=None, mic=None, specaug=False, gen=None):
        spec = torch.stft(wave, 512, 160, 480, self.window, center=True, return_complex=True)
        power = (spec.real ** 2 + spec.imag ** 2).transpose(1, 2)  # (B, T, 257)
        if augmenter is not None and mic is not None:
            power = augmenter.mic_power(power, mic)
        x = torch.log(power @ self.bank.T + 1e-6).transpose(1, 2)[:, None]  # (B, 1, 40, T)
        if specaug:
            b, _, f, t = x.shape
            dev = x.device
            for _ in range(2):
                w = torch.randint(0, 8, (b,), device=dev, generator=gen); s = (torch.rand(b, device=dev, generator=gen) * (f - w)).long()
                m = (torch.arange(f, device=dev)[None] >= s[:, None]) & (torch.arange(f, device=dev)[None] < (s + w)[:, None])
                x = x.masked_fill(m[:, None, :, None], 0.)
                w = torch.randint(0, 6, (b,), device=dev, generator=gen); s = (torch.rand(b, device=dev, generator=gen) * (t - w)).long()
                m = (torch.arange(t, device=dev)[None] >= s[:, None]) & (torch.arange(t, device=dev)[None] < (s + w)[:, None])
                x = x.masked_fill(m[:, None, None, :], 0.)
        return x


def load_teacher(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = BCResNet(ck['base']).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, TeacherFrontEnd(device)


def evaluate(model, front, sampler, device, corpus='sc', batch=512):
    clips, rows, cls, _ = sampler.all_clips(corpus)
    order = np.argsort(rows)  # sorted memmap reads; labels follow the same order
    rows, cls = rows[order], cls[order]
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            w = torch.tensor(clips[rows[i:i + batch]].astype(np.float32) / 32768, device=device)
            correct += int((model(front(w)).argmax(1).cpu().numpy() == cls[i:i + batch]).sum())
    return correct / len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', type=int, default=64, help='64 = BC-ResNet-8')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--steps-per-epoch', type=int, default=500)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--corpora', nargs='+', default=['sc', 'mswc', 'tts'])
    p.add_argument('--out', type=Path, default=ROOT / 'runs_teacher')
    a = p.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device('cuda')
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    train = ClipSampler('train', dev, a.corpora, seed=a.seed)
    val = ClipSampler('validation', dev, a.corpora, seed=a.seed, libri=False)
    aug = Augmenter(dev, np.load(ROOT / 'data/multi/noise_train.npy'), seed=a.seed)
    front = TeacherFrontEnd(dev)
    model = BCResNet(a.base).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=.9, weight_decay=1e-3, nesterov=True)
    total = a.epochs * a.steps_per_epoch
    warm = a.steps_per_epoch * 2
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * .5 * (1 + np.cos(np.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler()
    history, best, start = [], -1, time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(a.epochs):
        model.train()
        loss_sum = 0.
        for _ in range(a.steps_per_epoch):
            wave, y, sil = train.batch(a.batch)
            with torch.no_grad():
                wave, mic = aug.waveform(wave, sil)
                x = front(wave, aug, mic, specaug=True, gen=gen)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y, label_smoothing=.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            loss_sum += loss.item()
        acc = evaluate(model, front, val, dev, 'sc')
        r = {'epoch': epoch + 1, 'loss': loss_sum / a.steps_per_epoch, 'val_sc_35': acc}
        history.append(r)
        print(json.dumps(r), flush=True)
        if acc > best:
            best = acc
            torch.save({'state_dict': model.state_dict(), 'base': a.base, 'classes': CLASSES}, a.out / 'bcresnet8.pt')
    summary = dict(vars(a), out=str(a.out), params=n_params, best_val_sc_35=best,
                   training_seconds=time.perf_counter() - start, history=history)
    (a.out / 'teacher.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({'best_val_sc_35': best, 'params': n_params}), flush=True)


if __name__ == '__main__':
    main()
