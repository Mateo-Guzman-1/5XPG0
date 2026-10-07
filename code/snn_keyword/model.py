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


# ----------------------------------------------------------------------------
# Streaming SNN (snn_stream.py): integer export and frame-by-frame oracle
# (IMPLEMENTATION_PLAN.md, Phase 4). Shifts are arithmetic (floor), as in C
# on the RV32 target; membranes saturate to int16, the readout to int32.
#
# Units: every neuron has an integer threshold theta in [1024, 2048) that
# stands for the float threshold 1. Its int8 weight scale is folded into
# theta (layer 1: current = (x . w1q) >> r; layer 2: current =
# (sum of int8 weights of arriving spikes) << p), so no multiplications by
# scales are needed at run time. Adaptation traces are Q8 (256 = one spike);
# the adaptive threshold is theta + (bq * a) >> 8 with bq = beta * theta.
# Readout: one scale for all classes (theta_o), so the decision score
# o[keyword] - max(o[other]) and its threshold are plain integers.

I16 = (-32768, 32767)
I32 = (-2 ** 31, 2 ** 31 - 1)


def _theta_shift(unit, up):
    """Integer theta in [1024, 2048) for a float unit, and the power of two applied.

    up=False: theta = unit / 2^k (k >= 0, layer 1: current is shifted right).
    up=True:  theta = unit * 2^k (layer 2 and readout: current is shifted left).
    """
    k = np.floor(np.log2(unit / 1024)) if not up else np.ceil(np.log2(1024 / unit))
    k = np.maximum(k, 0).astype(np.int64)
    theta = np.rint(unit / 2.0 ** k if not up else unit * 2.0 ** k).astype(np.int64)
    return theta, k


def quantize_stream(model, threshold):
    """Export a float StreamSNN (hard delays, QAT-rounded shifts) to integers."""
    t = lambda v: v.detach().cpu().double().numpy()
    q = {'kind': np.array('stream'), 'n1': np.array(model.n1), 'n2': np.array(model.n2),
         'classes': np.array(model.classes)}
    # Layer 1: int8 per neuron; unit = 255 / scale (input bytes x weight LSBs per float 1).
    w1 = t(model.fc1.weight)
    s1 = np.maximum(np.abs(w1).max(1), 1e-8) / 127
    q['w1'] = np.clip(np.rint(w1 / s1[:, None]), -127, 127).astype(np.int8)
    q['theta1'], q['r1'] = _theta_shift(255 / s1, up=False)
    q['b1'] = np.rint(t(model.fc1.bias) * q['theta1']).astype(np.int32)
    # Layer 2: delayed layer-1 synapses and recurrent synapses share a scale per neuron.
    w12 = t(model.w12)
    rec = t(model.rec) if model.rec is not None else np.zeros((model.n2, model.n2))
    s2 = np.maximum(np.maximum(np.abs(w12).max(1), np.abs(rec).max(1)), 1e-8) / 127
    q['w12'] = np.clip(np.rint(w12 / s2[:, None]), -127, 127).astype(np.int8)
    q['rec'] = np.clip(np.rint(rec / s2[:, None]), -127, 127).astype(np.int8)
    q['theta2'], q['p2'] = _theta_shift(1 / s2, up=True)
    q['b2'] = np.rint(t(model.b2) * q['theta2']).astype(np.int32)
    q['delay'] = np.rint(np.clip(t(model.pos), 0, 31)).astype(np.uint8)
    for name, layer in (('1', model.l1), ('2', model.l2)):
        km = .5 + 7.5 / (1 + np.exp(-t(layer.km)))
        ka = 1 + 8 / (1 + np.exp(-t(layer.ka)))
        q['km' + name] = np.rint(km).astype(np.int64)
        q['ka' + name] = np.rint(ka).astype(np.int64)
        beta = np.log1p(np.exp(t(layer.beta)))
        q['bq' + name] = np.rint(beta * q['theta' + name]).astype(np.int32)
    # Readout: one scale for every class.
    wo = t(model.out.weight)
    so = max(np.abs(wo).max(), 1e-8) / 127
    q['wo'] = np.clip(np.rint(wo / so), -127, 127).astype(np.int8)
    theta_o, po = _theta_shift(np.array(1 / so), up=True)
    q['theta_o'], q['po'] = theta_o.reshape(()), po.reshape(())
    q['bo'] = np.rint(t(model.out.bias) * q['theta_o']).astype(np.int32)
    q['ko'] = np.rint(1 + 5 / (1 + np.exp(-t(model.ko)))).astype(np.int64)
    q['stream_threshold'] = np.array(int(np.rint(threshold * q['theta_o'])))
    return q


def stream_state(q, b):
    n1, n2, c = int(q['n1']), int(q['n2']), int(q['classes'])
    z = lambda *s: np.zeros(s, np.int64)
    return dict(u1=z(b, n1), a1=z(b, n1), s1=z(b, n1), hist=z(b, 32, n1), pos=0,
                u2=z(b, n2), a2=z(b, n2), s2=z(b, n2), o=z(b, c))


def _alif(u, a, s_prev, current, theta, bq, km, ka):
    a = a - (a >> ka) + (s_prev << 8)
    thr = theta + ((bq * a) >> 8)
    u = np.clip(u - (u >> km) + current, *I16)
    s = (u >= thr).astype(np.int64)
    u = np.clip(u - s * thr, *I16)
    return u, a, s


def integer_forward_stream(x, q, state=None, yes=None):
    """Frame-by-frame integer oracle.

    x: (B, T, 24) uint8 frames. Returns the decision score o[yes] - max(o[other])
    as int64 (B, T), the state after the last frame, and layer-1 / layer-2 spike
    counts per frame (B, T, 2) for the activity measurements.
    """
    x = np.asarray(x, np.int64)
    b, t, _ = x.shape
    st = stream_state(q, b) if state is None else {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}
    w1, w12, rec, wo = (q[k].astype(np.int64) for k in ('w1', 'w12', 'rec', 'wo'))
    delay = q['delay'].astype(np.int64)
    n2, n1 = w12.shape
    cols = np.arange(n1)[None, :].repeat(n2, 0)
    yes = int(q.get('yes_class', 2)) if yes is None else yes
    others = [c for c in range(int(q['classes'])) if c != yes]
    scores = np.zeros((b, t), np.int64)
    spikes = np.zeros((b, t, 2), np.int64)
    for i in range(t):
        cur1 = ((x[:, i] @ w1.T) >> q['r1']) + q['b1']
        st['u1'], st['a1'], st['s1'] = _alif(st['u1'], st['a1'], st['s1'], cur1, q['theta1'], q['bq1'], q['km1'], q['ka1'])
        pos = st['pos']
        st['hist'][:, pos] = st['s1']
        # Synapse (j, i) sees layer-1 neuron i as it was delay[j, i] frames ago.
        past = st['hist'][:, (pos - delay) % 32, cols]  # (B, n2, n1)
        acc2 = (past * w12[None]).sum(2) + st['s2'] @ rec.T
        cur2 = (acc2 << q['p2']) + q['b2']
        st['u2'], st['a2'], st['s2'] = _alif(st['u2'], st['a2'], st['s2'], cur2, q['theta2'], q['bq2'], q['km2'], q['ka2'])
        st['pos'] = (pos + 1) % 32
        st['o'] = np.clip(st['o'] - (st['o'] >> q['ko']) + ((st['s2'] @ wo.T) << q['po']) + q['bo'], *I32)
        scores[:, i] = np.clip(st['o'][:, yes] - st['o'][:, others].max(1), *I32)
        spikes[:, i, 0] = st['s1'].sum(1); spikes[:, i, 1] = st['s2'].sum(1)
    return scores, st, spikes


def decision_scores(scores, window=1, history=None):
    """Detection score of a stream: the integer sum of the last `window` frame scores
    (last axis; a partial sum over the first frames after a reset). history: the
    previous raw scores when a stream continues across calls (the last window-1
    are used). firmware/stream_main.c keeps the same running sum in int64 and
    compares it with stream_threshold (in sum units). window = 1: the raw score.
    """
    s = np.asarray(scores, np.int64)
    if window <= 1:
        return s
    h = np.zeros(s.shape[:-1] + (0,), np.int64) if history is None else \
        np.asarray(history, np.int64)[..., -(window - 1):]
    full = np.concatenate([h, s], -1)
    c = np.cumsum(np.concatenate([np.zeros(full.shape[:-1] + (1,), np.int64), full], -1), -1)
    idx = np.arange(h.shape[-1], full.shape[-1])
    return c[..., idx + 1] - c[..., np.maximum(idx + 1 - window, 0)]
