"""Verifier track: small causal CTC phoneme recogniser and its integer twin.

Stage 2 of the cascade. It sees the last ~1.5 s of the stage-1 input frames
(features.frame_features, 24 uint8 log-mel bands per 10 ms) and checks for
the phoneme sequence Y EH S (CMUdict "yes").

Float model (training):
  two 10 ms frames stacked -> 48 inputs per 20 ms step, scaled x / 256
  GRU(48 -> H1) -> GRU(H1 -> H2) -> linear(H2 -> 40)      (PyTorch GRU equations)
  40 classes: CTC blank + 39 phonemes (verifier_data.SYMBOLS)

Integer model (quantize / integer_forward; firmware/verifier.c is bit-exact):
  activations Q10 (1.0 = 1024) in int16; inputs x_q = frame * 4;
  weights int8 with one power-of-two scale per output row (w_q = round(w 2^e));
  the r/z rows of a layer use one scale over [W_i | W_h];
  pre-activation = (sum w_q a_q) >> e + b_q  (Q10, arithmetic shift);
  sigmoid, tanh: 512-entry int16 tables (sigmoid over [-8, 8), tanh over [-4, 4));
  n   = tanh(gx_n + (r * gh_n) >> 10);  h' = n + (z * (h - n)) >> 10;
  logits Q10.

Keyword score (keyword_score): with c_t(k) = logit_t(k) - max_j logit_t(j) <= 0,
the best-scoring path Y+ b* EH+ b* S+ (b = blank) over any segment of the
window, where the unconstrained best path costs 0. The softmax normaliser
cancels, so no exp or log is needed. The segment must start at step >= warmup.
Policy (b) (reject-prefix) appends `boundary` steps after the S in which no
new phoneme may start: each costs max(c_t(blank), c_t(S)).
"""
import numpy as np
import torch
from torch import nn

from verifier_data import BLANK, KEYWORD, SYMBOLS

N_MELS = 24
Q = 10                       # activation fraction bits
ONE = 1 << Q
SIG_RANGE, TANH_RANGE = 8, 4  # table input ranges [-R, R)
LUT = 512
NEG = -(1 << 30)


class Verifier(nn.Module):
    def __init__(self, h1=64, h2=64, stack=2, classes=len(SYMBOLS)):
        super().__init__()
        self.cfg = dict(h1=h1, h2=h2, stack=stack, classes=classes)
        self.stack = stack
        self.gru1 = nn.GRU(N_MELS * stack, h1, batch_first=True)
        self.gru2 = nn.GRU(h1, h2, batch_first=True)
        self.out = nn.Linear(h2, classes)
        # Per-input normalisation (x - mu) / sd, folded into layer 1 by quantize(). Without it
        # CTC training stays on the "same label sequence for every input" plateau (JOURNAL).
        self.register_buffer('mu', torch.zeros(N_MELS * stack))
        self.register_buffer('sd', torch.full((N_MELS * stack,), 256.))

    def stack_frames(self, x):
        b, t, m = x.shape
        t = t // self.stack * self.stack
        return x[:, :t].reshape(b, t // self.stack, m * self.stack)

    def forward(self, x):
        """x: (B, T, 24) uint8-valued frames -> logits (B, T // stack, classes)."""
        h, _ = self.gru1((self.stack_frames(x) - self.mu) / self.sd)
        h, _ = self.gru2(h)
        return self.out(h)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- integer model

def tables():
    i = np.arange(LUT)
    xs = (i * (2 * SIG_RANGE * ONE // LUT) + SIG_RANGE * ONE // LUT - SIG_RANGE * ONE) / ONE
    xt = (i * (2 * TANH_RANGE * ONE // LUT) + TANH_RANGE * ONE // LUT - TANH_RANGE * ONE) / ONE
    return (np.rint(ONE / (1 + np.exp(-xs))).astype(np.int16), np.rint(ONE * np.tanh(xt)).astype(np.int16))


SIG_SHIFT = int(np.log2(2 * SIG_RANGE * ONE // LUT))    # 5
TANH_SHIFT = int(np.log2(2 * TANH_RANGE * ONE // LUT))  # 4


def row_exponent(w):
    """Largest e with max|w_row| * 2^e <= 127 (per row)."""
    m = np.abs(w).max(1)
    return np.floor(np.log2(127 / np.maximum(m, 1e-12))).clip(-8, 24).astype(np.int64)


def quantize_rows(w, e):
    return np.clip(np.rint(w * 2.0 ** e[:, None]), -127, 127).astype(np.int8)


def quantize(model):
    """Float Verifier -> dict of numpy arrays for integer_forward (and the C export)."""
    q = {k: np.array(v) for k, v in model.cfg.items()}
    mu, sd = model.mu.detach().cpu().double().numpy(), model.sd.detach().cpu().double().numpy()
    for name, gru in (('l1', model.gru1), ('l2', model.gru2)):
        wi = gru.weight_ih_l0.detach().cpu().double().numpy()
        wh = gru.weight_hh_l0.detach().cpu().double().numpy()
        bi = gru.bias_ih_l0.detach().cpu().double().numpy()
        if name == 'l1':   # W (x - mu) / sd = (W 256 / sd) (x / 256) - W mu / sd; the integer input is x / 256
            bi = bi - wi @ (mu / sd)
            wi = wi * (256 / sd)[None]
        bh = gru.bias_hh_l0.detach().cpu().double().numpy()
        H = wh.shape[1]
        w = np.concatenate([wi, wh], 1)           # (3H, in + H)
        e = row_exponent(w)
        q[f'{name}_w'] = quantize_rows(w, e)
        q[f'{name}_e'] = e
        # r, z: one bias (b_i + b_h); n: b_in and b_hn apart (b_hn is gated by r).
        b = np.concatenate([bi[:2 * H] + bh[:2 * H], bi[2 * H:], bh[2 * H:]])
        q[f'{name}_b'] = np.rint(b * ONE).astype(np.int32)
    wo = model.out.weight.detach().cpu().double().numpy()
    e = row_exponent(wo)
    q['out_w'] = quantize_rows(wo, e)
    q['out_e'] = e
    q['out_b'] = np.rint(model.out.bias.detach().cpu().double().numpy() * ONE).astype(np.int32)
    q['sig'], q['tanh'] = tables()
    return q


def _lut(table, pre, rng, shift):
    idx = (np.clip(pre, -rng * ONE, rng * ONE - 1) + rng * ONE) >> shift
    return table[idx].astype(np.int64)


def gru_step_int(a, h, w, e, b, sig, tanh):
    """One integer GRU step for a batch: a (B, in), h (B, H) int64 Q10 -> new h."""
    H = h.shape[1]
    n_in = a.shape[1]
    ah = np.concatenate([a, h], 1)
    acc = ah @ w[:2 * H].astype(np.int64).T                            # (B, 2H)
    pre = (acc >> e[:2 * H]) + b[:2 * H]
    r = _lut(sig, pre[:, :H], SIG_RANGE, SIG_SHIFT)
    z = _lut(sig, pre[:, H:], SIG_RANGE, SIG_SHIFT)
    gx = ((a @ w[2 * H:, :n_in].astype(np.int64).T) >> e[2 * H:]) + b[2 * H:3 * H]
    gh = ((h @ w[2 * H:, n_in:].astype(np.int64).T) >> e[2 * H:]) + b[3 * H:]
    n = _lut(tanh, gx + ((r * gh) >> Q), TANH_RANGE, TANH_SHIFT)
    return n + ((z * (h - n)) >> Q)


def integer_forward(frames, q):
    """frames (B, T, 24) uint8 -> logits (B, T // stack, classes) int64 (Q10). Zero initial state."""
    frames = np.asarray(frames)
    b, t, m = frames.shape
    st = int(q['stack'])
    t = t // st * st
    x = frames[:, :t].reshape(b, t // st, m * st).astype(np.int64) << (Q - 8)
    h1 = np.zeros((b, int(q['h1'])), np.int64)
    h2 = np.zeros((b, int(q['h2'])), np.int64)
    wo = q['out_w'].astype(np.int64)
    out = np.empty((b, t // st, len(q['out_b'])), np.int64)
    for k in range(t // st):
        h1 = gru_step_int(x[:, k], h1, q['l1_w'], q['l1_e'], q['l1_b'], q['sig'], q['tanh'])
        h2 = gru_step_int(h1, h2, q['l2_w'], q['l2_e'], q['l2_b'], q['sig'], q['tanh'])
        out[:, k] = ((h2 @ wo.T) >> q['out_e']) + q['out_b']
    return out


class IntegerTorch:
    """integer_forward in float64 on a torch device (exact: every value < 2^53). For bulk evaluation."""

    def __init__(self, q, device='cuda'):
        d = lambda v: torch.tensor(np.asarray(v), dtype=torch.float64, device=device)
        self.q, self.device = q, device
        self.st, self.h1, self.h2 = int(q['stack']), int(q['h1']), int(q['h2'])
        self.L = {}
        for name in ('l1', 'l2'):
            w, e = q[f'{name}_w'], q[f'{name}_e']
            self.L[name] = (d(w), d(2.0 ** -e), d(q[f'{name}_b']), w.shape[1] - (self.h1 if name == 'l1' else self.h2))
        self.wo, self.so, self.bo = d(q['out_w']), d(2.0 ** -q['out_e']), d(q['out_b'])
        self.sig, self.tanh = d(q['sig']), d(q['tanh'])

    @staticmethod
    def _shift(acc, scale):   # arithmetic >> e == floor(acc * 2^-e)
        return torch.floor(acc * scale)

    def _lut(self, table, pre, rng, shift):
        idx = torch.floor((pre.clamp(-rng * ONE, rng * ONE - 1) + rng * ONE) / (1 << shift)).long()
        return table[idx]

    def _step(self, a, h, name):
        w, s, b, n_in = self.L[name]
        H = h.shape[1]
        ah = torch.cat([a, h], 1)
        pre = self._shift(ah @ w[:2 * H].T, s[:2 * H]) + b[:2 * H]
        r = self._lut(self.sig, pre[:, :H], SIG_RANGE, SIG_SHIFT)
        z = self._lut(self.sig, pre[:, H:], SIG_RANGE, SIG_SHIFT)
        gx = self._shift(a @ w[2 * H:, :n_in].T, s[2 * H:]) + b[2 * H:3 * H]
        gh = self._shift(h @ w[2 * H:, n_in:].T, s[2 * H:]) + b[3 * H:]
        n = self._lut(self.tanh, gx + torch.floor(r * gh / ONE), TANH_RANGE, TANH_SHIFT)
        return n + torch.floor(z * (h - n) / ONE)

    @torch.no_grad()
    def __call__(self, frames):
        x = torch.as_tensor(np.asarray(frames), device=self.device).double()
        b, t, m = x.shape
        t = t // self.st * self.st
        x = x[:, :t].reshape(b, t // self.st, m * self.st) * (1 << (Q - 8))
        h1 = torch.zeros(b, self.h1, dtype=torch.float64, device=self.device)
        h2 = torch.zeros(b, self.h2, dtype=torch.float64, device=self.device)
        out = []
        for k in range(t // self.st):
            h1 = self._step(x[:, k], h1, 'l1')
            h2 = self._step(h1, h2, 'l2')
            out.append(self._shift(h2 @ self.wo.T, self.so) + self.bo)
        return torch.stack(out, 1)


# --------------------------------------------------------------------------- keyword score

def keyword_score(logits, warmup=0, boundary=0, keyword=KEYWORD, return_end=False):
    """Best Y+ b* EH+ b* S+ [boundary] path score per window (int64, <= 0; NEG if none fits).

    logits: (B, T, C) integer (or float) array. Works on numpy int64 exactly as
    firmware/verifier.c does in int32. boundary > 0: that many steps after the
    last S step, each costing max(c(blank), c(S)) (policy b).
    """
    lg = np.asarray(logits)
    c = lg - lg.max(-1, keepdims=True)
    b_, t_, _ = c.shape
    k1, k2, k3 = keyword
    cy, ce, cs, cb = c[..., k1], c[..., k2], c[..., k3], c[..., BLANK]
    cp = np.maximum(cb, cs)
    ns = 5 + boundary
    D = np.full((b_, ns), NEG, np.int64)
    best = np.full(b_, NEG, np.int64)
    end = np.full(b_, -1, np.int64)
    for t in range(t_):
        P = D.copy()
        start = 0 if t >= warmup else NEG
        D[:, 0] = np.maximum(P[:, 0], start) + cy[:, t]
        D[:, 1] = np.maximum(P[:, 0], P[:, 1]) + cb[:, t]
        D[:, 2] = np.maximum(np.maximum(P[:, 0], P[:, 1]), P[:, 2]) + ce[:, t]
        D[:, 3] = np.maximum(P[:, 2], P[:, 3]) + cb[:, t]
        D[:, 4] = np.maximum(np.maximum(P[:, 2], P[:, 3]), P[:, 4]) + cs[:, t]
        for j in range(boundary):
            D[:, 5 + j] = P[:, 4 + j] + cp[:, t]
        D = np.maximum(D, NEG)             # keep "impossible" from drifting (int32 in C)
        fin = D[:, ns - 1]
        upd = fin > best
        best = np.where(upd, fin, best)
        end = np.where(upd, t, end)
    return (best, end) if return_end else best


def keyword_score_torch(logits, warmup=0, boundary=0, keyword=KEYWORD):
    """keyword_score on a torch tensor (float64 of integers, or float for training diagnostics)."""
    c = logits - logits.max(-1, keepdim=True)[0]
    b_, t_, _ = c.shape
    k1, k2, k3 = keyword
    cy, ce, cs, cb = c[..., k1], c[..., k2], c[..., k3], c[..., BLANK]
    cp = torch.maximum(cb, cs)
    neg = torch.full((b_,), float(NEG), dtype=c.dtype, device=c.device)
    D = [neg.clone() for _ in range(5 + boundary)]
    best = neg.clone()
    for t in range(t_):
        P = D
        start = torch.zeros_like(neg) if t >= warmup else neg
        D = [torch.maximum(P[0], start) + cy[:, t],
             torch.maximum(P[0], P[1]) + cb[:, t],
             torch.maximum(torch.maximum(P[0], P[1]), P[2]) + ce[:, t],
             torch.maximum(P[2], P[3]) + cb[:, t],
             torch.maximum(torch.maximum(P[2], P[3]), P[4]) + cs[:, t]]
        D += [P[4 + j] + cp[:, t] for j in range(boundary)]
        D = [torch.maximum(d, neg) for d in D]
        best = torch.maximum(best, D[-1])
    return best
