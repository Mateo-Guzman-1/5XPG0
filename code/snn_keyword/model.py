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


def pow2_shift(max_abs_q, bits):
    """Smallest s >= 0 with max|w*Q| / 2**s <= 2**(bits-1) - 1: one power-of-two scale per layer, chosen
    from the largest weight so that nothing is clipped when it is fixed (Mark's rule, in Q10 units)."""
    s = 0
    while max_abs_q / 2 ** s > 2 ** (bits - 1) - 1:
        s += 1
    return s


def fake_quant(x, shift, bits):
    """Straight-through rounding to the n-bit grid c * 2**shift / Q, c in [-2**(bits-1), 2**(bits-1) - 1]."""
    step = 2 ** shift / Q
    c = torch.clamp(torch.round(x / step), -2 ** (bits - 1), 2 ** (bits - 1) - 1)
    return x + (c * step - x).detach()


class SpikeMLP(nn.Module):
    def __init__(self, hidden=64, steps=12, encoding='current', w1_bits=16, fan_in=0):
        super().__init__()
        self.fc1 = nn.Linear(N_INPUT, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.hidden, self.steps, self.encoding = hidden, steps, encoding
        self.qat = False
        # Research options (synapse precision and count). Off by default; then no buffer is registered,
        # so the release model, its state_dict and its training are unchanged.
        self.w1_bits, self.fan_in, self.w1_shift = w1_bits, fan_in, None
        if fan_in:
            self.register_buffer('w1_mask', torch.ones_like(self.fc1.weight))

    def fix_w1_shift(self):
        """Fix the n-bit scale of w1 from the current weights (called once, at the first QAT epoch)."""
        w = self.fc1.weight.detach() * (self.w1_mask if self.fan_in else 1)
        self.w1_shift = pow2_shift(float(w.abs().max()) * Q, self.w1_bits)

    @torch.no_grad()
    def prune_to(self, keep):
        """Keep the `keep` largest-magnitude input weights of every hidden neuron (fixed fan-in)."""
        w = self.fc1.weight * self.w1_mask
        idx = w.abs().topk(keep, dim=1).indices
        self.w1_mask.zero_().scatter_(1, idx, 1.0)
        self.fc1.weight.mul_(self.w1_mask)

    def forward(self, x):
        w1, b1, w2, b2 = self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias
        if self.fan_in:
            w1 = w1 * self.w1_mask
        if self.qat:
            b1, w2, b2 = map(fake_round, (b1, w2, b2))
            w1 = fake_quant(w1, self.w1_shift, self.w1_bits) if self.w1_bits < 16 else fake_round(w1)
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
    extra = {}
    if model.fan_in:
        mask = model.w1_mask.detach().cpu().numpy().astype(np.uint8)
        if not (mask.sum(1) == model.fan_in).all():
            raise ValueError('every hidden neuron must keep exactly fan_in input weights')
        q['w1'] = q['w1'] * mask
        extra.update(fan_in=model.fan_in, w1_mask=mask)
    if model.w1_bits < 16:
        # n-bit synapses: c * 2**shift with c in n bits. The model file keeps the full-scale values
        # (int16, Q10), so integer_forward, kdot and the plain loop need no change; export_model.py
        # packs c for the packed firmware loop.
        w = model.fc1.weight.detach().cpu().numpy() * Q
        if model.fan_in:
            w = w * extra['w1_mask']
        lo, hi = -2 ** (model.w1_bits - 1), 2 ** (model.w1_bits - 1) - 1
        c = np.rint(w / 2 ** model.w1_shift)
        clipped = int(((c < lo) | (c > hi)).sum())
        q['w1'] = (np.clip(c, lo, hi) * 2 ** model.w1_shift).astype(np.int32)
        extra.update(w1_bits=model.w1_bits, w1_shift=model.w1_shift, w1_clipped=clipped)
    for name in ['w1', 'w2']:
        if np.abs(q[name]).max() > 32767:
            raise ValueError('Weights exceed int16 range')
        q[name] = q[name].astype(np.int16)
    drive_bound = np.abs(q['w1'].astype(np.int64)).sum(1) + np.abs(q['b1'])
    if max(drive_bound.max() * 255, drive_bound.max() * 8 * 7,
           (np.abs(q['w2'].astype(np.int64)).sum(1) + np.abs(q['b2'])).max() * model.steps) >= 2**31:
        raise ValueError('Model could overflow a signed 32-bit accumulator')
    q.update(steps=model.steps, encoding=0 if model.encoding == 'current' else 1, q=Q)
    q.update(extra)
    return q


def lif_count(drive, steps):
    """Spikes of one hidden neuron in `steps` steps of the integer LIF update for a constant drive."""
    v = count = 0
    for _ in range(steps):
        v = int(trunc_div(np.int64(v * 7), 8)) + drive
        if v >= Q:
            v -= Q
            count += 1
    return count


def lif_thresholds(steps):
    """Smallest drive that gives at least k spikes, k = 1..steps.

    With current encoding the drive is constant, so the spike count is a staircase function of the
    drive: 0 spikes for drive <= 0 (the membrane never becomes positive) and `steps` spikes for
    drive >= Q (every step crosses the threshold). In between the count never decreases as the drive
    grows (checked exhaustively in tests/test_pipeline.py). Counting the thresholds a drive reaches is
    therefore bit-exact with the loop (firmware USE_LIF_TABLE)."""
    counts = [lif_count(d, steps) for d in range(0, Q + 1)]
    if any(a > b for a, b in zip(counts, counts[1:])) or counts[0] or counts[Q] != steps:
        raise ValueError('spike count is not a monotone staircase')
    return [next(d for d, c in enumerate(counts) if c >= k) for k in range(1, steps + 1)]


def keyword_of(npz):
    """Keyword stored in a features cache or model file; files made before it was stored are "yes"."""
    return str(npz['keyword']) if 'keyword' in npz else 'yes'


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
