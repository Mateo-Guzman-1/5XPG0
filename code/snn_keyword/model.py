"""LIF SNN and independent integer reference. Divisions truncate to zero."""
import numpy as np
import torch
from torch import nn
from features import N_INPUT
Q = 1024


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad / (1 + 10 * x.abs()).square()


def fake_round(x):
    return x + (torch.round(x * Q) / Q - x).detach()


def fake_trunc(x):
    return x + (torch.trunc(x * Q) / Q - x).detach()


class SpikeMLP(nn.Module):
    def __init__(self, hidden=64, steps=12, encoding='current'):
        super().__init__()
        self.fc1 = nn.Linear(N_INPUT, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.hidden, self.steps, self.encoding = hidden, steps, encoding
        self.qat = False

    def forward(self, x):
        w1, b1, w2, b2 = self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias
        if self.qat:
            w1, b1, w2, b2 = map(fake_round, (w1, b1, w2, b2))
        mem = x.new_zeros((len(x), self.hidden))
        counts = torch.zeros_like(mem)
        phase = torch.zeros_like(x)
        drive = nn.functional.linear(x, w1)
        if self.qat:
            drive = fake_trunc(drive)
        drive = drive + b1
        for _ in range(self.steps):
            if self.encoding == 'rate':
                phase = phase + x * 255
                inp = (phase >= 255).float()
                phase = phase - inp * 255
                current = nn.functional.linear(inp, w1, b1)
            else:
                current = drive
            leak = mem * .875
            mem = (fake_trunc(leak) if self.qat else leak) + current
            spike = Spike.apply(mem - 1)
            mem = mem - spike
            counts = counts + spike
        return nn.functional.linear(counts, w2, b2 * self.steps), counts.sum(1)


def quantize(model):
    q = {name: np.rint(value.detach().cpu().numpy() * Q).astype(np.int32)
         for name, value in [('w1', model.fc1.weight), ('b1', model.fc1.bias),
                             ('w2', model.fc2.weight), ('b2', model.fc2.bias)]}
    for name in ['w1', 'w2']:
        if np.abs(q[name]).max() > 32767:
            raise ValueError('Weights exceed int16 range')
        q[name] = q[name].astype(np.int16)
    drive_bound = np.abs(q['w1'].astype(np.int64)).sum(1) + np.abs(q['b1'])
    if max(drive_bound.max() * 255, drive_bound.max() * 8 * 7,
           (np.abs(q['w2'].astype(np.int64)).sum(1) + np.abs(q['b2'])).max() * model.steps) >= 2**31:
        raise ValueError('Model could overflow a signed 32-bit accumulator')
    q.update(steps=model.steps, encoding=0 if model.encoding == 'current' else 1, q=Q)
    return q


def trunc_div(x, d):
    return np.where(x < 0, -((-x) // d), x // d)


def integer_forward(x, q, batch=512):
    """Numpy int64 oracle, independent from PyTorch and firmware."""
    outputs, spikes = [], []
    x = np.asarray(x, dtype=np.uint8).reshape(-1, N_INPUT)
    w1, w2 = q['w1'].astype(np.int64), q['w2'].astype(np.int64)
    for start in range(0, len(x), batch):
        xb = x[start:start+batch].astype(np.int64)
        drive = trunc_div(xb @ w1.T, 255) + q['b1']
        mem = np.zeros((len(xb), len(w1)), dtype=np.int64)
        count = np.zeros_like(mem)
        phase = np.zeros_like(xb)
        for _ in range(int(q['steps'])):
            if int(q['encoding']):
                phase += xb
                inp = phase >= 255
                phase -= inp * 255
                current = inp.astype(np.int64) @ w1.T + q['b1']
            else:
                current = drive
            mem = trunc_div(mem * 7, 8) + current
            spk = mem >= Q
            mem -= spk * Q
            count += spk
        outputs.append(count @ w2.T + q['b2'] * int(q['steps']))
        spikes.append(count.sum(1))
    return np.concatenate(outputs), np.concatenate(spikes)


def metrics(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp = int((y & pred).sum()), int((~y & pred).sum())
    fn, tn = int((y & ~pred).sum()), int((~y & ~pred).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=(tp+tn)/len(y),
                precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn), fpr=fp/max(1,fp+tn),
                f1=2*tp/max(1,2*tp+fp+fn), balanced_accuracy=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)))


def choose_threshold(y, scores):
    order = np.argsort(-scores, kind='stable')
    yy, ss = y[order], scores[order]
    tp = np.cumsum(yy); fp = np.cumsum(1-yy)
    f1 = 2*tp / np.maximum(1, tp + fp + yy.sum())
    ends = np.r_[ss[:-1] != ss[1:], True]
    f1[~ends] = -1
    return ss[int(np.argmax(f1))].item()
