"""Phase 3: streaming two-layer adaptive-LIF SNN with learnable delays (float).

One 24-band frame every 10 ms (features.frame_features) drives:
  layer 1   N1 ALIF neurons [3], dense from the 24 inputs (the kdot layer)
  delays    layer 1 -> layer 2 synapses carry a learnable delay of 0-31
            frames, trained with DCLS [4]: each synapse is a Gaussian over the
            32 taps whose centre is learned and whose width shrinks during
            training; at the end every synapse is rounded to one tap. In
            hardware this is the spike-history ring buffer (report §3.5, E4).
  layer 2   N2 ALIF neurons, recurrent (one-frame delay)
  readout   leaky integrator per class, no spikes

ALIF, per neuron, in the form the integer oracle (Phase 4) uses with shifts:
  a[t] = a[t-1] - a[t-1] * 2^-ka + s[t-1]      adaptation trace
  B[t] = 1 + beta * a[t]                       adaptive threshold
  u[t] = u[t-1] - u[t-1] * 2^-km + I[t]         leak and input
  s[t] = H(u[t] - B[t]);  u[t] -= s[t] B[t]     spike, reset by subtraction
km, ka (time constants 10 ms * 2^k) and beta are learned per neuron [3][36];
Phase 4 rounds km, ka to integers so the leaks become exact shifts.

Surrogate gradient [2]: the fast-sigmoid derivative of model.Spike.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import Spike

ROOT = Path(__file__).resolve().parent
N_MELS = 24
MAX_DELAY = 32


def shift_param(lo, hi, init, shape):
    """Unconstrained parameter mapped to [lo, hi] with a sigmoid; init inside (lo, hi)."""
    p = torch.logit(torch.as_tensor((np.asarray(init) - lo) / (hi - lo), dtype=torch.float32))
    return nn.Parameter(p.expand(shape).clone())


class ALIF(nn.Module):
    def __init__(self, n, km=(2, 6), ka=(3, 7), beta=.1, seed=0):
        super().__init__()
        g = np.random.default_rng(seed)
        self.km = shift_param(.5, 8, g.uniform(*km, n), (n,))
        self.ka = shift_param(1, 9, g.uniform(*ka, n), (n,))
        self.beta = nn.Parameter(torch.full((n,), float(np.log(np.expm1(beta)))))
        self.hard = False  # Phase 4: round km, ka

    def constants(self):
        km = .5 + 7.5 * torch.sigmoid(self.km)
        ka = 1 + 8 * torch.sigmoid(self.ka)
        if self.hard:
            km = km + (km.round() - km).detach(); ka = ka + (ka.round() - ka).detach()
        return 2. ** -km, 2. ** -ka, F.softplus(self.beta)

    def init_state(self, b, device):
        z = torch.zeros(b, len(self.km), device=device)
        return z, z.clone(), z.clone()  # u, a, s

    def step(self, current, state, consts):
        u, a, s = state
        lm, la, beta = consts
        a = a - a * la + s
        thr = 1 + beta * a
        u = u - u * lm + current
        s = Spike.apply((u - thr) / thr)
        u = u - s * thr
        return u, a, s


class StreamSNN(nn.Module):
    def __init__(self, n1=128, n2=128, classes=3, delays=True, recurrent=True, adaptive=True, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n1, self.n2, self.classes = n1, n2, classes
        self.delays, self.recurrent, self.adaptive = delays, recurrent, adaptive
        self.fc1 = nn.Linear(N_MELS, n1)
        self.l1 = ALIF(n1, seed=seed)
        self.w12 = nn.Parameter(torch.randn(n2, n1) * (1 / np.sqrt(n1)))
        self.b2 = nn.Parameter(torch.zeros(n2))
        # Delay centres start spread over the 32 taps (DCLS initialisation).
        self.pos = nn.Parameter(torch.rand(n2, n1) * (MAX_DELAY - 1) if delays else torch.zeros(n2, n1),
                                requires_grad=delays)
        self.sigma = MAX_DELAY / 2  # annealed by the trainer
        self.hard_delays = False
        self.rec = nn.Parameter(torch.randn(n2, n2) * (.5 / np.sqrt(n2))) if recurrent else None
        self.l2 = ALIF(n2, seed=seed + 1)
        self.out = nn.Linear(n2, classes)
        self.ko = shift_param(1, 6, 3., (classes,))
        if not adaptive:
            for l in (self.l1, self.l2):
                l.beta.data.fill_(-20.)
                l.beta.requires_grad_(False)

    # Phase 4 QAT: weights fake-quantized to int8 as model.quantize_stream() exports them.
    qat = False

    @staticmethod
    def fake_int8(w, scale):
        return w + (torch.round(w / scale).clamp(-127, 127) * scale - w).detach()

    def w1(self):
        w = self.fc1.weight
        return self.fake_int8(w, w.abs().amax(1, keepdim=True).detach().clamp_min(1e-8) / 127) if self.qat else w

    def w2(self):
        """Layer-2 inputs (delayed layer 1, recurrent) share one scale per layer-2 neuron."""
        if not self.qat:
            return self.w12, self.rec
        m = self.w12.abs().amax(1, keepdim=True)
        if self.rec is not None:
            m = torch.maximum(m, self.rec.abs().amax(1, keepdim=True))
        scale = m.detach().clamp_min(1e-8) / 127
        return self.fake_int8(self.w12, scale), None if self.rec is None else self.fake_int8(self.rec, scale)

    def wo(self):
        w = self.out.weight
        return self.fake_int8(w, w.abs().max().detach().clamp_min(1e-8) / 127) if self.qat else w

    def delay_kernel(self):
        """(n2, n1, MAX_DELAY) weights per tap; tap d = delay of d frames."""
        taps = torch.arange(MAX_DELAY, device=self.pos.device, dtype=torch.float32)
        pos = self.pos.clamp(0, MAX_DELAY - 1)
        if self.hard_delays or not self.delays:
            g = F.one_hot(pos.round().long(), MAX_DELAY).float()
        else:
            g = torch.exp(-.5 * ((taps - pos[..., None]) / self.sigma) ** 2)
            g = g / g.sum(-1, keepdim=True)
        return self.w2()[0][..., None] * g

    def readout_leak(self):
        k = 1 + 5 * torch.sigmoid(self.ko)
        if self.qat:
            k = k + (k.round() - k).detach()
        return 2. ** -k

    def init_state(self, b, device):
        return dict(l1=self.l1.init_state(b, device), l2=self.l2.init_state(b, device),
                    hist=torch.zeros(b, self.n1, MAX_DELAY - 1, device=device),
                    o=torch.zeros(b, self.classes, device=device))

    def forward(self, x, state=None):
        """x: (B, T, 24) uint8-valued frames. Returns readout (B, T, C), state, mean rates (l1, l2)."""
        b, t, _ = x.shape
        state = self.init_state(b, x.device) if state is None else state
        c1 = F.linear(x / 255, self.w1(), self.fc1.bias)
        k1 = self.l1.constants()
        st, s1 = state['l1'], []
        for i in range(t):
            st = self.l1.step(c1[:, i], st, k1)
            s1.append(st[2])
        s1 = torch.stack(s1, 2)  # (B, n1, T)
        l1_state = st
        # Causal delays: output at t sums tap d of s1[t - d]; conv1d correlates, so flip taps.
        full = torch.cat([state['hist'], s1], 2)
        d_in = F.conv1d(full, self.delay_kernel().flip(-1)) + self.b2[None, :, None]  # (B, n2, T)
        hist = full[:, :, -(MAX_DELAY - 1):]
        k2 = self.l2.constants()
        st, o = state['l2'], state['o']
        lo = self.readout_leak()
        rec, wo = self.w2()[1], self.wo()
        outs, s2sum = [], 0.
        for i in range(t):
            cur = d_in[:, :, i]
            if rec is not None:
                cur = cur + st[2] @ rec.T
            st = self.l2.step(cur, st, k2)
            o = o - o * lo + F.linear(st[2], wo, self.out.bias)
            outs.append(o)
            s2sum = s2sum + st[2].mean()
        new_state = dict(l1=l1_state, l2=st, hist=hist, o=o)
        rates = (s1.mean(), s2sum / t)
        return torch.stack(outs, 1), new_state, rates


def detach_state(state):
    return {k: tuple(x.detach() for x in v) if isinstance(v, tuple) else v.detach() for k, v in state.items()}


def _moving_sum_float(s, w):
    c = np.cumsum(np.concatenate([np.zeros(s.shape[:-1] + (1,)), s.astype(np.float64)], -1), -1)
    idx = np.arange(s.shape[-1])
    return c[..., idx + 1] - c[..., np.maximum(idx + 1 - w, 0)]


class StreamDetector:
    """robust_eval interface for a float StreamSNN checkpoint.

    Score per 10 ms frame: readout of "yes" minus the largest other class.
    The decision rule (threshold, hold-off) is the harness's.
    """
    kind = 'stream'

    def __init__(self, path, device='cuda', chunk=4000):
        from features import frame_features
        ck = torch.load(path, map_location=device, weights_only=False)
        self.model = StreamSNN(**ck['config']).to(device)
        self.model.load_state_dict(ck['state_dict'])
        self.model.hard_delays = True
        for l in (self.model.l1, self.model.l2):
            l.hard = ck.get('hard_shifts', False)
        self.model.qat = ck.get('qat', False)
        self.model.eval()
        from keyword_config import check_model_keyword
        check_model_keyword(ck.get('keyword'), str(path))
        self.threshold = float(ck['threshold'])
        self.window = int(ck.get('window', 1))   # moving-sum decision (model.decision_scores)
        self.yes = ck['yes_class']
        self.frontend = ck.get('frontend', 'logmel')
        self.device, self.chunk = device, chunk
        self.frame_features = frame_features

    def scores(self, x):
        """(B, T, 24) frames -> (B, T) decision score."""
        state, out = None, []
        with torch.no_grad():
            for i in range(0, x.shape[1], self.chunk):
                o, state, _ = self.model(x[:, i:i + self.chunk], state)
                others = torch.cat([o[..., :self.yes], o[..., self.yes + 1:]], -1).max(-1)[0]
                out.append(o[..., self.yes] - others)
        return torch.cat(out, 1)

    def traces(self, audios, batch=512):
        feats = [self.frame_features(a, self.frontend) for a in audios]
        out = [None] * len(feats)
        order = np.argsort([len(f) for f in feats])  # batch clips of similar length
        for i in range(0, len(order), batch):
            ids = order[i:i + batch]
            t = max(len(feats[k]) for k in ids)
            x = np.zeros((len(ids), t, N_MELS), np.float32)
            for j, k in enumerate(ids):
                x[j, :len(feats[k])] = feats[k]
            s = self.scores(torch.tensor(x, device=self.device)).cpu().numpy()
            d = s if self.window <= 1 else _moving_sum_float(s, self.window)
            for j, k in enumerate(ids):
                n = len(feats[k])
                times = (np.arange(n) * 160 + 400) / 16000  # end of each frame
                out[k] = (times, d[j, :n], s[j, :n])   # decision score, raw score
        return out


class IntegerStreamDetector:
    """robust_eval interface for an exported integer model (model.quantize_stream):
    runs model.integer_forward_stream, the arithmetic the firmware must match."""
    kind = 'stream'

    def __init__(self, q, batch=256):
        from features import frame_features
        self.q = q
        from keyword_config import check_model_keyword
        check_model_keyword(q['keyword'] if 'keyword' in q else None, 'integer stream model')
        self.threshold = int(q['stream_threshold'])
        self.window = int(q.get('decision_window', 1))   # moving-sum decision (model.decision_scores)
        self.frontend = str(q.get('frontend', 'logmel'))
        self.batch = batch
        self.frame_features = frame_features

    def traces(self, audios):
        from model import decision_scores, integer_forward_stream
        feats = [self.frame_features(a, self.frontend) for a in audios]
        out = [None] * len(feats)
        order = np.argsort([len(f) for f in feats])
        for i in range(0, len(order), self.batch):
            ids = order[i:i + self.batch]
            t = max(len(feats[k]) for k in ids)
            x = np.zeros((len(ids), t, N_MELS), np.uint8)
            for j, k in enumerate(ids):
                x[j, :len(feats[k])] = feats[k]
            s, _, _ = integer_forward_stream(x, self.q)
            d = decision_scores(s, self.window)
            for j, k in enumerate(ids):
                n = len(feats[k])
                times = (np.arange(n) * 160 + 400) / 16000
                out[k] = (times, d[j, :n], s[j, :n])   # decision score, raw score
        return out
