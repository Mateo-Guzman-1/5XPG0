#!/usr/bin/env python3
"""golden_lif.py -- integer golden model for rtl/snn_layer.v, test-vector
generator for sim/tb_snn_layer.v, and self-tests.

WHAT IT MODELS (bit-exact target for the RTL; numpy int64 arithmetic)

    during a time step, for each input event (row, x) in the order they are pushed:
        v <- sat(v + W[row] * x)                    # saturate to VBITS, signed
    at TICK (end of the step):
        fire <- (v > THR)                           # strict '>', on the un-updated v
        v    <- v - (v >> K) - (THR if fire else 0) # arithmetic shift, subtract-reset
                                                    # applied to the LEAKED value

WHY THIS IS snnTorch's ORDER (snn.Leaky, reset_mechanism="subtract",
reset_delay=True, the default in snntorch 1.0.0):
    reset_t = (mem_{t-1} > thr)
    mem_t   = beta*mem_{t-1} + I_t - reset_t*thr
    spk_t   = (mem_t > thr)
The RTL does the "beta*mem - reset*thr" part at the END of step t-1 and adds I_t
during step t, so v just before the tick of step t equals mem_t (up to the
integer rounding of the shift-leak, beta = 1 - 2^-K).  `selftest` checks this
against a numpy transcription of snnTorch's forward(), and against the real
snnTorch if it is installed.

USAGE
    python3 golden_lif.py regress              # write sim/vec_*/ (used by run_sim.sh)
    python3 golden_lif.py gen --outdir D [...] # one custom vector set
    python3 golden_lif.py hwvec [--out F]      # snn_vectors.h for the on-board smoke test
    python3 golden_lif.py selftest [--snntorch]

Files written per vector set (all hex, one 32-bit word per line):
    tb_params.vh  `define TB_* constants for the testbench
    weights.mem   weight window words: word w = row*GROUPS + g,
                  byte j = weight of neuron g*4+j for input `row`
    cmds.mem      [31:30]=op (0 EVT, 1 TICK, 2 END); EVT payload = (x<<16)|row
    expect.mem    per tick: SPKW fire words, then P post-tick v (32-bit two's
                  complement); after the last tick: P spike counters, then #ticks
"""
import argparse
import os
import re
import sys
import tempfile
from dataclasses import dataclass, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Cfg:
    P: int = 8            # parallel neurons (multiple of 4)
    N_IN: int = 16        # inputs = weight rows
    XBITS: int = 1        # 1 = binary spikes, up to 8 = direct value
    VBITS: int = 24       # membrane width (signed, saturating)
    K: int = 3            # beta = 1 - 2^-K
    THR: int = 64
    T: int = 30           # time steps
    GAP: int = 0          # 0: TB polls busy after each event; >=2: fixed spacing, no polling
    WBITS: int = 8        # fixed by the RTL sketch
    CBITS: int = 8        # per-neuron spike counter width (RTL default)

    @property
    def groups(self):
        return self.P // 4

    @property
    def spkw(self):
        return (self.P + 31) // 32

    @property
    def words(self):
        return self.N_IN * self.groups

    def check(self):
        assert self.WBITS == 8, "the RTL sketch assumes 8-bit weights"
        assert self.P >= 4 and self.P % 4 == 0, "P must be a multiple of 4"
        assert 1 <= self.XBITS <= 8
        assert self.VBITS >= self.WBITS + self.XBITS + 1, \
            "RTL needs VBITS >= WBITS+XBITS+1 (product must fit the guard-bit adder)"
        assert self.VBITS <= 31, "register reads sign-extend into 32 bits"
        assert 0 < self.THR < (1 << (self.VBITS - 1))
        assert 1 <= self.K <= self.VBITS - 2
        assert self.words <= 4096, "weight window is 16 KB = 4096 words"
        assert self.GAP == 0 or self.GAP >= 2, "GAP=1 overflows the single event register"
        assert 1 <= self.T


# ----------------------------------------------------------------------------
# tick rules: the RTL one, and deliberately-wrong alternatives (see selftest)
# ----------------------------------------------------------------------------
def tick_rtl(v, c):
    fire = v > c.THR
    return fire, v - (v >> c.K) - np.where(fire, c.THR, 0)


def tick_compare_after_leak(v, c):           # wrong order A
    leak = v - (v >> c.K)
    fire = leak > c.THR
    return fire, leak - np.where(fire, c.THR, 0)


def tick_reset_before_leak(v, c):            # wrong order B (no reset delay)
    fire = v > c.THR
    v1 = v - np.where(fire, c.THR, 0)
    return fire, v1 - (v1 >> c.K)


def tick_non_strict(v, c):                   # wrong compare C
    fire = v >= c.THR
    return fire, v - (v >> c.K) - np.where(fire, c.THR, 0)


def tick_clamp_zero(v, c):                   # wrong D: clamp at 0 (main.c / Sankaran style)
    fire, vp = tick_rtl(v, c)
    return fire, np.maximum(vp, 0)


ALTERNATIVES = {
    "A compare after leak": tick_compare_after_leak,
    "B reset before leak": tick_reset_before_leak,
    "C fire on >= thr": tick_non_strict,
    "D clamp v at 0": tick_clamp_zero,
}


# ----------------------------------------------------------------------------
# the layer
# ----------------------------------------------------------------------------
class Layer:
    def __init__(self, cfg, W, tick_fn=tick_rtl):
        cfg.check()
        self.c = cfg
        self.W = np.asarray(W, dtype=np.int64)
        assert self.W.shape == (cfg.N_IN, cfg.P)
        assert self.W.min() >= -128 and self.W.max() <= 127
        self.vmax = (1 << (cfg.VBITS - 1)) - 1
        self.vmin = -(1 << (cfg.VBITS - 1))
        self.tick_fn = tick_fn
        self.clear()

    def clear(self):
        self.v = np.zeros(self.c.P, dtype=np.int64)
        self.cnt = np.zeros(self.c.P, dtype=np.int64)
        self.ticks = 0
        self.sat_pos = 0
        self.sat_neg = 0

    def event(self, row, x):
        raw = self.v + self.W[row] * x
        self.sat_pos += int((raw > self.vmax).sum())
        self.sat_neg += int((raw < self.vmin).sum())
        self.v = np.clip(raw, self.vmin, self.vmax)

    def tick(self):
        v_pre = self.v.copy()
        fire, v_post = self.tick_fn(v_pre, self.c)
        v_post = np.asarray(v_post, dtype=np.int64)
        assert v_post.min() >= self.vmin and v_post.max() <= self.vmax, \
            "tick left the VBITS range (RTL would wrap): pick smaller weights or bigger VBITS"
        self.v = v_post
        self.cnt = (self.cnt + fire) & ((1 << self.c.CBITS) - 1)
        self.ticks += 1
        return fire, v_pre, v_post


def run(cfg, W, steps, tick_fn=tick_rtl):
    L = Layer(cfg, W, tick_fn)
    rec = {"fire": [], "v_pre": [], "v_post": [], "I": []}
    for ev in steps:
        cur = np.zeros(cfg.P, dtype=np.int64)
        for row, x in ev:
            L.event(row, x)
            cur += L.W[row] * x
        f, vpre, vpost = L.tick()
        rec["fire"].append(f)
        rec["v_pre"].append(vpre)
        rec["v_post"].append(vpost)
        rec["I"].append(cur)
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(cnt=L.cnt.copy(), ticks=L.ticks, sat_pos=L.sat_pos, sat_neg=L.sat_neg)
    return out


# ----------------------------------------------------------------------------
# vector generation
# ----------------------------------------------------------------------------
def make_set(cfg, seed, density, wlo, whi, wstd, shuffle=True):
    """Random weights (per-neuron mean in [wlo, whi], so some neurons are
    excitatory and some inhibitory) and random event streams."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(wlo, whi, size=cfg.P)
    W = np.clip(np.rint(rng.normal(mu, wstd, size=(cfg.N_IN, cfg.P))), -128, 127).astype(np.int64)
    xmax = (1 << cfg.XBITS) - 1
    steps = []
    for _ in range(cfg.T):
        rows = np.flatnonzero(rng.random(cfg.N_IN) < density)
        if shuffle:
            rng.shuffle(rows)
        steps.append([(int(r), 1 if cfg.XBITS == 1 else int(rng.integers(1, xmax + 1)))
                      for r in rows])
    return W, steps


def pack_weights(cfg, W):
    words = []
    for row in range(cfg.N_IN):
        for g in range(cfg.groups):
            w = 0
            for j in range(4):
                w |= (int(W[row, g * 4 + j]) & 0xFF) << (8 * j)
            words.append(w)
    return words


def _hexfile(path, words):
    with open(path, "w") as f:
        for w in words:
            f.write("%08x\n" % (int(w) & 0xFFFFFFFF))


def write_set(outdir, cfg, W, steps, res):
    os.makedirs(outdir, exist_ok=True)
    cmds = []
    for ev in steps:
        cmds += [(int(x) << 16) | int(row) for row, x in ev]      # op 0 = EVT
        cmds.append(1 << 30)                                       # op 1 = TICK
    cmds.append(2 << 30)                                           # op 2 = END
    exp = []
    for t in range(cfg.T):
        f = res["fire"][t]
        for n in range(cfg.spkw):
            word = 0
            for j in range(32):
                idx = 32 * n + j
                if idx < cfg.P and f[idx]:
                    word |= 1 << j
            exp.append(word)
        exp += [int(v) & 0xFFFFFFFF for v in res["v_post"][t]]
    exp += [int(c) for c in res["cnt"]]
    exp.append(res["ticks"])
    _hexfile(os.path.join(outdir, "weights.mem"), pack_weights(cfg, W))
    _hexfile(os.path.join(outdir, "cmds.mem"), cmds)
    _hexfile(os.path.join(outdir, "expect.mem"), exp)
    with open(os.path.join(outdir, "tb_params.vh"), "w") as f:
        f.write("// generated by golden_lif.py -- do not edit\n")
        for k, v in [("TB_P", cfg.P), ("TB_NIN", cfg.N_IN), ("TB_XBITS", cfg.XBITS),
                     ("TB_VBITS", cfg.VBITS), ("TB_K", cfg.K), ("TB_THR", cfg.THR),
                     ("TB_WORDS", cfg.words), ("TB_NCMD", len(cmds)),
                     ("TB_NTICK", cfg.T), ("TB_GAP", cfg.GAP)]:
            f.write("`define %s %d\n" % (k, v))


def read_hex(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


def load_set(outdir):
    """Parse a vector set exactly the way the testbench will."""
    defs = dict(re.findall(r"`define\s+(TB_\w+)\s+(-?\d+)", open(os.path.join(outdir, "tb_params.vh")).read()))
    d = {k: int(v) for k, v in defs.items()}
    cfg = Cfg(P=d["TB_P"], N_IN=d["TB_NIN"], XBITS=d["TB_XBITS"], VBITS=d["TB_VBITS"],
              K=d["TB_K"], THR=d["TB_THR"], T=d["TB_NTICK"], GAP=d["TB_GAP"])
    words = read_hex(os.path.join(outdir, "weights.mem"))
    assert len(words) == cfg.words == d["TB_WORDS"]
    W = np.zeros((cfg.N_IN, cfg.P), dtype=np.int64)
    for w, word in enumerate(words):                    # the documented mapping
        row, g = divmod(w, cfg.groups)
        for j in range(4):
            b = (word >> (8 * j)) & 0xFF
            W[row, g * 4 + j] = b - 256 if b >= 128 else b
    cmds = read_hex(os.path.join(outdir, "cmds.mem"))
    assert len(cmds) == d["TB_NCMD"]
    steps, cur = [], []
    for c in cmds:
        op = c >> 30
        if op == 0:
            cur.append((c & 0xFFFF, (c >> 16) & 0xFF))
        elif op == 1:
            steps.append(cur)
            cur = []
        else:
            break
    exp = read_hex(os.path.join(outdir, "expect.mem"))
    return cfg, W, steps, exp


# ----------------------------------------------------------------------------
# the standard regression sets
# ----------------------------------------------------------------------------
REGRESS = {
    #        cfg overrides                                      seed  dens  wlo  whi  wstd
    "small":  (dict(P=8,  N_IN=16,  XBITS=1, VBITS=24, K=3, THR=64,  T=30),          1, 0.40,  -8, 16, 12),
    "fast":   (dict(P=8,  N_IN=16,  XBITS=1, VBITS=24, K=4, THR=64,  T=30, GAP=2),   2, 0.40,  -8, 16, 12),
    "sat":    (dict(P=16, N_IN=16,  XBITS=1, VBITS=12, K=3, THR=64,  T=20),          3, 0.60, -60, 60, 40),
    "direct": (dict(P=8,  N_IN=16,  XBITS=8, VBITS=24, K=3, THR=2048, T=20),         4, 0.50,  -3,  4,  3),
    "full":   (dict(P=64, N_IN=256, XBITS=1, VBITS=24, K=3, THR=256, T=25),          5, 0.05, -10, 40, 30),
    "tie":    None,   # directed set, see build_tie_set()
}


def build_tie_set():
    """Directed set: weights that land v exactly on THR (and 1 above / 1 below), plus
    ticks with no events (pure leak) and a row hit several times in one step.
    Random data almost never produces v == THR, which is the only case where
    'v > thr' and 'v >= thr' differ."""
    cfg = Cfg(P=4, N_IN=4, XBITS=1, VBITS=24, K=3, THR=64, T=12)
    W = np.array([[64,   65,  63,  32],      # row 0: neuron 0 lands ON thr, 1 above, 2 below
                  [0,     0,   1,  32],
                  [-64,  -1,   0,   0],
                  [127, -128,  10,   5]], dtype=np.int64)
    steps = [[(0, 1)], [(0, 1)], [(0, 1), (1, 1)], [(1, 1), (1, 1)], [(3, 1)], [(3, 1), (2, 1)],
             [], [(0, 1), (3, 1)], [], [(2, 1), (2, 1), (2, 1)], [(0, 1)], []]
    assert len(steps) == cfg.T
    return cfg, W, steps


def build_regress_set(name):
    if name == "tie":
        cfg, W, steps = build_tie_set()
        return cfg, W, steps, run(cfg, W, steps)
    over, seed, dens, wlo, whi, wstd = REGRESS[name]
    cfg = Cfg(**over)
    cfg.check()
    W, steps = make_set(cfg, seed, dens, wlo, whi, wstd)
    return cfg, W, steps, run(cfg, W, steps)


def coverage(cfg, res):
    f = res["fire"]
    return ("spikes=%d  neurons_that_fire=%d/%d  ticks_with_a_spike=%d/%d  "
            "negative_v_samples=%d  sat+=%d sat-=%d"
            % (f.sum(), int((f.sum(0) > 0).sum()), cfg.P, int((f.sum(1) > 0).sum()), cfg.T,
               int((res["v_pre"] < 0).sum()), res["sat_pos"], res["sat_neg"]))


def cmd_regress(base):
    for name in REGRESS:
        cfg, W, steps, res = build_regress_set(name)
        out = os.path.join(base, "vec_" + name)
        write_set(out, cfg, W, steps, res)
        print("%-7s -> %s\n        %s" % (name, out, coverage(cfg, res)))


# ----------------------------------------------------------------------------
# vectors for the ON-BOARD smoke test (firmware/snn_smoke.c)
# ----------------------------------------------------------------------------
def make_steps(cfg, rng, density, shuffle=True):
    """Event streams only (weights supplied by the caller)."""
    steps = []
    for _ in range(cfg.T):
        rows = np.flatnonzero(rng.random(cfg.N_IN) < density)
        if shuffle:
            rng.shuffle(rows)
        steps.append([(int(r), 1) for r in rows])
    return steps


def cmd_hwvec(outpath):
    """C header with the weights and three event scripts for the bitstream as built:
    P=64, N_IN=256, XBITS=1, VBITS=24, K=3 (see the snn_layer instance in spike_soc.v).
    Set A is the regression 'full' set; B is sparse, C dense.  B and C are meant to be
    pushed WITHOUT polling busy, to prove the single event register never overflows."""
    cfg = Cfg(P=64, N_IN=256, XBITS=1, VBITS=24, K=3, THR=256, T=25)
    cfg.check()
    _, seed, dens, wlo, whi, wstd = REGRESS["full"]
    W, steps_a = make_set(cfg, seed, dens, wlo, whi, wstd)
    rng = np.random.default_rng(seed + 1000)
    scripts = [("A: regression 'full', density %.2f" % dens, steps_a, 1),
               ("B: sparse, density 0.02", make_steps(cfg, rng, 0.02), 0),
               ("C: dense, density 0.20", make_steps(cfg, rng, 0.20), 0)]
    L = ["// snn_vectors.h -- generated by `golden_lif.py hwvec` -- DO NOT EDIT.",
         "// Weights + event scripts + expected results for firmware/snn_smoke.c.",
         "// Bitstream configuration these vectors are valid for:",
         "//   P=%d N_IN=%d XBITS=%d VBITS=%d LEAK_SHIFT=%d THR=%d, %d time steps per set"
         % (cfg.P, cfg.N_IN, cfg.XBITS, cfg.VBITS, cfg.K, cfg.THR, cfg.T),
         "#ifndef SNN_VECTORS_H", "#define SNN_VECTORS_H", "",
         "#define SNN_P %d" % cfg.P, "#define SNN_NIN %d" % cfg.N_IN,
         "#define SNN_XBITS %d" % cfg.XBITS, "#define SNN_VBITS %d" % cfg.VBITS,
         "#define SNN_K %d" % cfg.K, "#define SNN_THR_VAL %d" % cfg.THR,
         "#define SNN_T %d" % cfg.T, "#define SNN_WORDS %d" % cfg.words,
         "#define SNN_SPKW %d" % cfg.spkw, "#define SNN_NSETS %d" % len(scripts), ""]

    def arr(name, words):
        L.append("static const u32 %s[%d] = {" % (name, len(words)))
        for i in range(0, len(words), 8):
            L.append("    " + ", ".join("0x%08xu" % (int(w) & 0xFFFFFFFF) for w in words[i:i + 8]) + ",")
        L.append("};")
        L.append("")

    arr("snn_weights", pack_weights(cfg, W))
    total_words = cfg.words
    table = []
    print("hwvec: bitstream config P=%d N_IN=%d XBITS=%d VBITS=%d K=%d THR=%d T=%d"
          % (cfg.P, cfg.N_IN, cfg.XBITS, cfg.VBITS, cfg.K, cfg.THR, cfg.T))
    for i, (label, steps, polled) in enumerate(scripts):
        res = run(cfg, W, steps)
        assert res["sat_pos"] == 0 and res["sat_neg"] == 0
        cmds = []
        for ev in steps:
            cmds += [(1 << 16) | int(row) for row, _ in ev]
            cmds.append(1 << 30)
        cmds.append(2 << 30)
        exp = []
        for t in range(cfg.T):
            f = res["fire"][t]
            for n in range(cfg.spkw):
                exp.append(sum(1 << j for j in range(32) if 32 * n + j < cfg.P and f[32 * n + j]))
            exp += [int(v) & 0xFFFFFFFF for v in res["v_post"][t]]
        exp += [int(c) for c in res["cnt"]]
        exp.append(res["ticks"])
        arr("set%d_cmds" % i, cmds)
        arr("set%d_expect" % i, exp)
        table.append((i, len(cmds), polled))
        total_words += len(cmds) + len(exp)
        print("  set %s  %s\n     events=%d  cmd_words=%d  expect_words=%d  %s"
              % ("ABC"[i], "polled" if polled else "FAST (no busy polling)",
                 sum(len(e) for e in steps), len(cmds), len(exp), coverage(cfg, res)))
    L.append("typedef struct { const u32 *cmds; u32 ncmd; const u32 *expect; u32 polled; } snn_set_t;")
    L.append("static const snn_set_t snn_sets[SNN_NSETS] = {")
    for i, n, polled in table:
        L.append("    { set%d_cmds, %du, set%d_expect, %du }," % (i, n, i, polled))
    L.append("};")
    L += ["", "#endif // SNN_VECTORS_H", ""]
    with open(outpath, "w") as f:
        f.write("\n".join(L))
    print("wrote %s  (%d data words = %.1f KB of firmware rodata; image limit is 63.75 KB incl. code)"
          % (outpath, total_words, total_words * 4 / 1024.0))


# ----------------------------------------------------------------------------
# self-tests
# ----------------------------------------------------------------------------
def snntorch_reference(I, K, thr):
    """numpy transcription of snn.Leaky.forward() (snntorch 1.0.0, subtract,
    reset_delay=True) for integer-valued input currents I[T, P]."""
    beta = 1.0 - 2.0 ** -K
    mem = np.zeros(I.shape[1])
    spk_all, mem_all = [], []
    for t in range(I.shape[0]):
        reset = (mem - thr) > 0                # self.reset = mem_reset(self.mem)   (previous mem)
        mem = beta * mem + I[t] - reset * thr  # _base_sub(input_)
        spk = (mem - thr) > 0                  # fire(self.mem)
        spk_all.append(spk)
        mem_all.append(mem.copy())
    return np.array(spk_all), np.array(mem_all)


def bound_check(cfg, res, spk_ref, mem_ref, tol, label):
    """While a neuron's spike train equals the reference, its pre-tick v must satisfy
    0 <= v_int - mem_ref < 2^K  (floor-leak error e' = beta*e + d, d in [0,1)).
    Returns (ok, first_divergence_per_neuron, n_diverged)."""
    ok = True
    diverged = 0
    for n in range(cfg.P):
        diff = np.flatnonzero(res["fire"][:, n] != spk_ref[:, n])
        td = diff[0] if len(diff) else cfg.T - 1
        diverged += 1 if len(diff) else 0
        e = res["v_pre"][: td + 1, n] - mem_ref[: td + 1, n]
        if e.min() < -tol or e.max() >= 2 ** cfg.K + tol:
            ok = False
            print("   %s: neuron %d error out of bound: min %.4f max %.4f (limit %d)"
                  % (label, n, e.min(), e.max(), 2 ** cfg.K))
    return ok, diverged


def selftest(use_snntorch):
    fails = 0

    print("[1] shift-leak facts (v - (v >> K))")
    for K in (3, 4):
        v, vn = 1000, -1000
        for _ in range(400):
            v, vn = v - (v >> K), vn - (vn >> K)
        good = (v == 2 ** K - 1) and (vn == 0)
        fails += not good
        print("    K=%d beta=%.4f: +1000 -> %d (floor 2^K-1=%d), -1000 -> %d   %s"
              % (K, 1 - 2 ** -K, v, 2 ** K - 1, vn, "ok" if good else "FAIL"))

    print("[2] file round trip (generator -> parse like the testbench -> replay -> compare)")
    for name in REGRESS:
        cfg, W, steps, res = build_regress_set(name)
        with tempfile.TemporaryDirectory() as d:
            write_set(d, cfg, W, steps, res)
            cfg2, W2, steps2, exp = load_set(d)
        res2 = run(cfg2, W2, steps2)
        # rebuild the expect word stream from the replay and compare with the file
        with tempfile.TemporaryDirectory() as d2:
            write_set(d2, cfg2, W2, steps2, res2)
            exp2 = read_hex(os.path.join(d2, "expect.mem"))
        good = (cfg2 == cfg and np.array_equal(W2, W) and steps2 == steps and exp == exp2)
        fails += not good
        print("    %-7s %s" % (name, "ok" if good else "FAIL"))

    print("[3] agreement with snnTorch's update order (numpy transcription of Leaky.forward)")
    print("    bound: while a neuron's spikes match, 0 <= v_int - mem_float < 2^K")
    for name in REGRESS:
        cfg, W, steps, res = build_regress_set(name)
        if res["sat_pos"] or res["sat_neg"]:
            print("    %-7s skipped (saturates on purpose; float reference has no saturation)" % name)
            continue
        spk_ref, mem_ref = snntorch_reference(res["I"].astype(float), cfg.K, cfg.THR)
        ok, div = bound_check(cfg, res, spk_ref, mem_ref, 1e-6, name)
        agree = 100.0 * (res["fire"] == spk_ref).mean()
        fails += not ok
        print("    %-7s %s  spike agreement %.2f%%  neurons whose spike train ever diverged: %d/%d"
              % (name, "ok" if ok else "FAIL", agree, div, cfg.P))

    print("[4] do the vectors discriminate update orders? (changed v_post samples vs the RTL rule)")
    sets = ("small", "full", "tie")
    detected = {label: 0 for label in ALTERNATIVES}
    for name in sets:
        cfg, W, steps, base = build_regress_set(name)
        for label, fn in ALTERNATIVES.items():
            try:
                alt = run(cfg, W, steps, fn)
            except AssertionError:
                print("    %-6s %-22s left the VBITS range (counts as detected)" % (name, label))
                detected[label] += 1
                continue
            dfire = int((alt["fire"] != base["fire"]).sum())
            dv = int((alt["v_post"] != base["v_post"]).sum())
            detected[label] += dv
            print("    %-6s %-22s fire differs in %4d of %5d, v differs in %5d of %5d"
                  % (name, label, dfire, base["fire"].size, dv, base["v_post"].size))
    for label, n in detected.items():
        good = n > 0
        fails += not good
        print("    => %-22s %s" % (label, "caught by at least one set" if good else "NOT CAUGHT BY ANY SET"))

    print("[5] real snnTorch cross-check")
    fails += snntorch_real(use_snntorch)

    print("\nselftest: %s" % ("ALL OK" if fails == 0 else "%d FAILURE(S)" % fails))
    return 1 if fails else 0


def snntorch_real(requested):
    try:
        import torch
        import snntorch as snn
    except ImportError:
        print("    SKIPPED: snntorch/torch not installed here (pip install torch snntorch, then rerun with --snntorch)")
        return 0
    if not requested:
        print("    snnTorch is installed; rerun with --snntorch to run this check")
        return 0
    bad = 0
    for name in ("small", "fast", "direct", "full"):
        cfg, W, steps, res = build_regress_set(name)
        if res["sat_pos"] or res["sat_neg"]:
            continue
        try:
            lif = snn.Leaky(beta=1.0 - 2.0 ** -cfg.K, threshold=float(cfg.THR),
                            reset_mechanism="subtract", reset_delay=True)
        except TypeError:
            print("    your snnTorch has no reset_delay argument: older release, older update order -> not comparable")
            return 0
        mem, spk_l, mem_l = None, [], []
        for t in range(cfg.T):
            cur = torch.tensor(res["I"][t], dtype=torch.float32).unsqueeze(0)
            spk, mem = lif(cur) if mem is None else lif(cur, mem)
            spk_l.append(spk.squeeze(0).numpy() > 0.5)
            mem_l.append(mem.squeeze(0).detach().numpy().astype(float))
        ok, div = bound_check(cfg, res, np.array(spk_l), np.array(mem_l), 1e-2, "snntorch " + name)
        bad += not ok
        print("    %-7s %s (neurons whose spike train diverged: %d/%d)" % (name, "ok" if ok else "FAIL", div, cfg.P))
    return bad


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("regress")
    hv = sub.add_parser("hwvec")
    hv.add_argument("--out", default=os.path.join(HERE, "snn_vectors.h"))
    st = sub.add_parser("selftest")
    st.add_argument("--snntorch", action="store_true")
    g = sub.add_parser("gen")
    g.add_argument("--outdir", required=True)
    for name, default in [("P", 8), ("N_IN", 16), ("XBITS", 1), ("VBITS", 24), ("K", 3),
                          ("THR", 64), ("T", 30), ("GAP", 0), ("seed", 1)]:
        g.add_argument("--" + name, type=int, default=default)
    g.add_argument("--density", type=float, default=0.4)
    g.add_argument("--wlo", type=float, default=-8)
    g.add_argument("--whi", type=float, default=16)
    g.add_argument("--wstd", type=float, default=12)
    a = ap.parse_args()

    if a.cmd == "regress":
        cmd_regress(HERE)
    elif a.cmd == "hwvec":
        cmd_hwvec(a.out)
    elif a.cmd == "selftest":
        sys.exit(selftest(a.snntorch))
    else:
        cfg = Cfg(P=a.P, N_IN=a.N_IN, XBITS=a.XBITS, VBITS=a.VBITS, K=a.K, THR=a.THR, T=a.T, GAP=a.GAP)
        cfg.check()
        W, steps = make_set(cfg, a.seed, a.density, a.wlo, a.whi, a.wstd)
        res = run(cfg, W, steps)
        write_set(a.outdir, cfg, W, steps, res)
        print("wrote %s\n%s" % (a.outdir, coverage(cfg, res)))


if __name__ == "__main__":
    main()
