#!/usr/bin/env python3
"""golden_alif.py -- integer golden model for the ALIF rtl/snn_layer.v, test-vector
generator for sim/tb_snn_alif.v and firmware/snn_smoke.c, and cross-check against
the ALIF of branch ALIF-Layer1 (snn_keyword/model.py).

WHAT IT MODELS (bit-exact target for the RTL; numpy int64 arithmetic)

    during a time step, for each input event (row, x):   acc += W[row] * x
    at TICK, per neuron (this is _alif() of model.py with layer 1's current):
        cur = (acc >> r) + b
        a   = a - (a >> ka) + (s << 8)          s = spike of the previous tick
        thr = theta + ((bq * a) >> 8)
        u   = clip(u - (u >> km) + cur, int16)
        s   = u >= thr
        u   = clip(u - s * thr, int16)
        acc = 0
Shifts are arithmetic (floor). With the inputs of ALIF-Layer1's layer 1 -- the 24
bytes of a log-mel frame as events (row i, x = byte i), one TICK per frame -- this
is exactly layer 1 of model.integer_forward_stream(); `crosscheck` runs that
function itself (taken from git) against this model.

Hardware limits checked here (see rtl/snn_layer.v): bq is 18-bit signed, acc stays
inside int32, theta 16-bit unsigned, r/km/ka 4-bit.

USAGE
    python3 golden_alif.py regress [--base D]   # write D/vec_*/ (default sim/), used by run_snn_alif.sh
    python3 golden_alif.py hwvec [--out F]      # firmware/snn_vectors.h for the on-board smoke test
    python3 golden_alif.py crosscheck [--model-py F --npz F]   # against ALIF-Layer1 (default: from git)

Files written per vector set (hex, one 32-bit word per line):
    tb_params.vh  `define TB_* constants for the testbench
    weights.mem   weight window words: word w = row*GROUPS + g,
                  byte j = weight of neuron g*4+j for input `row`
    params.mem    per neuron: NP (theta | r<<16 | km<<20 | ka<<24), BQ, BIAS
    cmds.mem      [31:30]=op (0 EVT, 1 TICK, 2 END); EVT payload = (x<<16)|row
    expect.mem    per tick: SPKW fire words, P post-tick u (sign-extended), P post-tick a;
                  after the last tick: P spike counters, then #ticks
"""
import argparse
import ast
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
I16 = (-32768, 32767)
I32 = (-2 ** 31, 2 ** 31 - 1)
BQ_RANGE = (-2 ** 17, 2 ** 17 - 1)
TICK_CYCLES = 4


# ----------------------------------------------------------------------------
# configuration and parameters
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Cfg:
    P: int = 16           # parallel neurons (multiple of 4, <= 64)
    N_IN: int = 256       # inputs = weight rows
    XBITS: int = 8        # input value width (unsigned)
    T: int = 25           # time steps
    GAP: int = 0          # 0: TB polls busy after each event; >=2: fixed spacing, no polling
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

    def check(self, rtl=True):
        assert self.P >= 4 and self.P % 4 == 0, "P must be a multiple of 4"
        if rtl:
            assert self.P <= 64, "the register map holds 64 neurons per region"
            assert self.words <= 4096, "weight window is 16 KB = 4096 words"
        assert 1 <= self.XBITS <= 8
        assert self.GAP == 0 or self.GAP >= 2, "GAP=1 overflows the single event register"
        assert 1 <= self.T


@dataclass
class Params:
    theta: np.ndarray     # 16-bit unsigned
    r: np.ndarray         # 4-bit: current = (acc >> r) + b
    km: np.ndarray        # 4-bit membrane leak shift
    ka: np.ndarray        # 4-bit adaptation decay shift
    bq: np.ndarray        # 18-bit signed adaptation gain
    b: np.ndarray         # int32 bias

    def check(self):
        assert (0 <= self.theta).all() and (self.theta < 2 ** 16).all()
        for k in (self.r, self.km, self.ka):
            assert (0 <= k).all() and (k < 16).all()
        assert (self.bq >= BQ_RANGE[0]).all() and (self.bq <= BQ_RANGE[1]).all(), \
            "bq must fit 18 bits signed (rtl/snn_layer.v)"
        assert (self.b >= I32[0]).all() and (self.b <= I32[1]).all()

    def words(self):
        """params.mem / snn_vectors.h order: per neuron NP, BQ, BIAS."""
        out = []
        for n in range(len(self.theta)):
            np_word = int(self.theta[n]) | int(self.r[n]) << 16 | int(self.km[n]) << 20 | int(self.ka[n]) << 24
            out += [np_word, int(self.bq[n]) & 0xFFFFFFFF, int(self.b[n]) & 0xFFFFFFFF]
        return out


def np_word(theta, r, km, ka):
    return int(theta) | int(r) << 16 | int(km) << 20 | int(ka) << 24


# ----------------------------------------------------------------------------
# the layer
# ----------------------------------------------------------------------------
class Layer:
    def __init__(self, cfg, W, prm, rtl=True):
        cfg.check(rtl)
        prm.check()
        self.c, self.prm = cfg, prm
        self.W = np.asarray(W, dtype=np.int64)
        assert self.W.shape == (cfg.N_IN, cfg.P)
        assert self.W.min() >= -128 and self.W.max() <= 127
        self.clear()

    def clear(self):
        z = lambda: np.zeros(self.c.P, dtype=np.int64)
        self.acc, self.u, self.a, self.s, self.cnt = z(), z(), z(), z(), z()
        self.ticks = 0
        self.stats = dict(u_sat=0, a_max=0, thr_max=0, thr_min=0, adapted_fire_blocks=0)

    def event(self, row, x):
        assert 0 <= x < (1 << self.c.XBITS)
        self.acc = self.acc + self.W[row] * x
        assert self.acc.min() >= I32[0] and self.acc.max() <= I32[1], "acc left int32 (RTL would wrap)"

    def tick(self):
        p = self.prm
        cur = (self.acc >> p.r) + p.b
        a = self.a - (self.a >> p.ka) + (self.s << 8)
        thr = p.theta + ((p.bq * a) >> 8)
        raw = self.u - (self.u >> p.km) + cur
        u = np.clip(raw, *I16)
        s = (u >= thr).astype(np.int64)
        u_post = np.clip(u - s * thr, *I16)
        st = self.stats
        st["u_sat"] += int(((raw < I16[0]) | (raw > I16[1])).sum())
        st["a_max"] = max(st["a_max"], int(a.max()))
        st["thr_max"] = max(st["thr_max"], int(thr.max()))
        st["thr_min"] = min(st["thr_min"], int(thr.min()))
        st["adapted_fire_blocks"] += int(((u >= p.theta) & (u < thr)).sum())   # adaptation mattered
        assert a.max() < 2 ** 24, "a left 24 bits"
        self.u, self.a, self.s = u_post, a, s
        self.acc = np.zeros_like(self.acc)
        self.cnt = (self.cnt + s) & ((1 << self.c.CBITS) - 1)
        self.ticks += 1
        return s.astype(bool), u_post.copy(), a.copy()


def run(cfg, W, prm, steps, rtl=True):
    L = Layer(cfg, W, prm, rtl)
    rec = {"fire": [], "u": [], "a": []}
    for ev in steps:
        for row, x in ev:
            L.event(row, x)
        f, u, a = L.tick()
        rec["fire"].append(f)
        rec["u"].append(u)
        rec["a"].append(a)
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(cnt=L.cnt.copy(), ticks=L.ticks, stats=L.stats)
    return out


# ----------------------------------------------------------------------------
# vector sets
# ----------------------------------------------------------------------------
def random_params(rng, P, bq_hi=500, b_lo=-600, b_hi=800, r_lo=2, r_hi=7, wide=False):
    """Parameters in the ranges of the trained ALIF-Layer1 model (sheila_stream_int8:
    theta 1024..2022, r 2..7, km 1..8, ka 2..8, bq 36..448, b -581..780); wide=True
    also uses the edges of the hardware fields (km/ka 0 and 15, negative / 18-bit bq)."""
    prm = Params(theta=rng.integers(1024, 2048, P), r=rng.integers(r_lo, r_hi + 1, P),
                 km=rng.integers(1, 9, P), ka=rng.integers(2, 9, P),
                 bq=rng.integers(0, bq_hi + 1, P), b=rng.integers(b_lo, b_hi + 1, P))
    if wide:
        prm.km[:4] = [0, 15, 1, 8]
        prm.ka[:4] = [0, 15, 1, 9]
        prm.bq[4:8] = [BQ_RANGE[1], BQ_RANGE[0], -300, 70000]
        prm.theta[8:10] = [0, 65535]
        prm.r[10:12] = [0, 15]
    return prm


def random_weights(rng, cfg, wlo=-20, whi=40, wstd=40):
    mu = rng.uniform(wlo, whi, size=cfg.P)
    return np.clip(np.rint(rng.normal(mu, wstd, size=(cfg.N_IN, cfg.P))), -127, 127).astype(np.int64)


def random_steps(rng, cfg, density, rows=None, xlo=1):
    rows = np.arange(cfg.N_IN) if rows is None else np.asarray(rows)
    xmax = (1 << cfg.XBITS) - 1
    steps = []
    for _ in range(cfg.T):
        hit = rows[rng.random(len(rows)) < density]
        rng.shuffle(hit)
        steps.append([(int(r), int(rng.integers(xlo, xmax + 1))) for r in hit])
    return steps


def git_file(ref_path):
    """Contents of a file on branch ALIF-Layer1 (local or origin)."""
    for ref in ("ALIF-Layer1", "origin/ALIF-Layer1"):
        try:
            return subprocess.run(["git", "show", "%s:%s" % (ref, ref_path)], cwd=HERE,
                                  check=True, capture_output=True).stdout
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError("branch ALIF-Layer1 not found (git fetch origin ALIF-Layer1)")


def sheila_layer1(n_out, npz=None):
    """Layer 1 of the trained 'sheila' ALIF model (24 log-mel bytes -> 128 ALIF):
    weights as W[input, neuron] and Params, first n_out neurons."""
    import io
    q = np.load(npz if npz else io.BytesIO(git_file("code/snn_keyword/results/models/sheila_stream_int8.npz")),
                allow_pickle=True)
    sel = slice(0, n_out)
    W = q["w1"].astype(np.int64)[sel].T                       # (24, n_out)
    prm = Params(theta=q["theta1"][sel].astype(np.int64), r=q["r1"][sel].astype(np.int64),
                 km=q["km1"][sel].astype(np.int64), ka=q["ka1"][sel].astype(np.int64),
                 bq=q["bq1"][sel].astype(np.int64), b=q["b1"][sel].astype(np.int64))
    return W, prm, q


def frames_to_steps(frames):
    """(T, 24) uint8 frames -> one event per input byte, one tick per frame (zero
    bytes add nothing, so they are skipped like a sparse event stream would)."""
    return [[(i, int(v)) for i, v in enumerate(fr) if v] for fr in frames]


def keyword_frames(rng, T):
    """log-mel-like bytes: a slowly moving band profile plus noise, some loud frames."""
    base = rng.uniform(40, 140, 24)
    fr = []
    for t in range(T):
        level = 1.6 if (t // 5) % 2 else 0.8
        fr.append(np.clip(base * level + rng.normal(0, 25, 24), 0, 255))
    return np.rint(np.array(fr)).astype(np.int64)


def build_keyword_set(cfg, seed):
    """The trained sheila layer 1 (first P neurons) on 24 inputs of an N_IN-row layer."""
    rng = np.random.default_rng(seed)
    W24, prm, _ = sheila_layer1(cfg.P)
    W = np.zeros((cfg.N_IN, cfg.P), dtype=np.int64)
    W[:24] = W24
    return W, prm, frames_to_steps(keyword_frames(rng, cfg.T))


def build_tie_set():
    """Directed: u lands exactly on thr (fires: '>='), 1 below (does not), steps with
    no events (pure leak + adaptation decay), a row hit several times in one step,
    and saturation of u at both ends. bq = 0 on neurons 0..3 keeps thr = theta."""
    cfg = Cfg(P=8, N_IN=4, XBITS=8, T=12)
    W = np.array([[100,  100,  100, 127, -127, 50,  10,   1],
                  [  0,    1,   -1, 127, -127,  0,   0,   0],
                  [ -5,    0,    0, 127, -127, 50,  10,   1],
                  [  0,    0,    0,   0,    0,  0,   0,   0]], dtype=np.int64)
    prm = Params(theta=np.array([1000, 1000, 1000, 1024, 1024, 300, 1500, 1]),
                 r=np.array([0, 0, 0, 0, 0, 1, 2, 0]),
                 km=np.array([15, 15, 15, 1, 1, 2, 3, 0]),
                 ka=np.array([3, 3, 3, 1, 1, 2, 4, 0]),
                 bq=np.array([0, 0, 0, 0, 0, 400, 2000, 50]),
                 b=np.array([0, 0, 0, 0, 0, -10, 100, 0]))
    steps = [[(0, 10)],                  # n0: u = 1000 = thr -> fires; n1 1000 fires; n2 1000 fires
             [(0, 10), (1, 1)],          # n1: 1001, n2: 999 -> n2 below thr
             [], [(3, 0)],               # leak only (x = 0 event adds nothing)
             [(1, 255), (1, 255), (1, 255)] + [(0, 255)] * 2,     # saturation +
             [(2, 255)] * 4, [],
             [(0, 200), (2, 200)], [(0, 1)], [], [(2, 7), (0, 9), (1, 3)], []]
    assert len(steps) == cfg.T
    return cfg, W, prm, steps


REGRESS = {
    #          cfg overrides                                     seed density wide
    "small":  (dict(P=8,  N_IN=16,  T=30),                        1, 0.40, False),
    "fast":   (dict(P=8,  N_IN=16,  T=30, GAP=2),                 2, 0.40, False),
    "edges":  (dict(P=16, N_IN=32,  T=40),                        3, 0.50, True),
    "spikes": (dict(P=8,  N_IN=16,  T=30, XBITS=1),               4, 0.60, False),
    "full":   (dict(P=16, N_IN=256, T=25),                        5, 0.05, False),
    "keyword": None,   # trained sheila layer 1, see build_keyword_set()
    "tie":     None,   # directed, see build_tie_set()
}


def build_regress_set(name):
    if name == "tie":
        cfg, W, prm, steps = build_tie_set()
    elif name == "keyword":
        cfg = Cfg(P=16, N_IN=256, T=30)
        W, prm, steps = build_keyword_set(cfg, 6)
    else:
        over, seed, dens, wide = REGRESS[name]
        cfg = Cfg(**over)
        rng = np.random.default_rng(seed)
        prm = random_params(rng, cfg.P, wide=wide)
        if cfg.XBITS == 1:
            prm.r[:] = 0
            prm.b = rng.integers(-50, 50, cfg.P)
            W = random_weights(rng, cfg, wlo=200, whi=400, wstd=300)  # clipped to int8: strong weights
        else:
            W = random_weights(rng, cfg)
        steps = random_steps(rng, cfg, dens)
    return cfg, W, prm, steps, run(cfg, W, prm, steps)


def coverage(cfg, res):
    f, st = res["fire"], res["stats"]
    return ("spikes=%d  neurons_that_fire=%d/%d  ticks_with_a_spike=%d/%d  u_sat=%d  "
            "a_max=%d  thr=[%d, %d]  fires_blocked_by_adaptation=%d"
            % (f.sum(), int((f.sum(0) > 0).sum()), cfg.P, int((f.sum(1) > 0).sum()), cfg.T,
               st["u_sat"], st["a_max"], st["thr_min"], st["thr_max"], st["adapted_fire_blocks"]))


# ----------------------------------------------------------------------------
# files
# ----------------------------------------------------------------------------
def pack_weights(cfg, W):
    words = []
    for row in range(cfg.N_IN):
        for g in range(cfg.groups):
            w = 0
            for j in range(4):
                w |= (int(W[row, g * 4 + j]) & 0xFF) << (8 * j)
            words.append(w)
    return words


def cmd_words(steps):
    cmds = []
    for ev in steps:
        cmds += [(int(x) << 16) | int(row) for row, x in ev]      # op 0 = EVT
        cmds.append(1 << 30)                                       # op 1 = TICK
    cmds.append(2 << 30)                                           # op 2 = END
    return cmds


def expect_words(cfg, res):
    exp = []
    for t in range(cfg.T):
        f = res["fire"][t]
        for n in range(cfg.spkw):
            exp.append(sum(1 << j for j in range(32) if 32 * n + j < cfg.P and f[32 * n + j]))
        exp += [int(v) & 0xFFFFFFFF for v in res["u"][t]]
        exp += [int(v) for v in res["a"][t]]
    exp += [int(c) for c in res["cnt"]]
    exp.append(res["ticks"])
    return exp


def _hexfile(path, words):
    with open(path, "w") as f:
        for w in words:
            f.write("%08x\n" % (int(w) & 0xFFFFFFFF))


def write_set(outdir, cfg, W, prm, steps, res):
    os.makedirs(outdir, exist_ok=True)
    cmds = cmd_words(steps)
    _hexfile(os.path.join(outdir, "weights.mem"), pack_weights(cfg, W))
    _hexfile(os.path.join(outdir, "params.mem"), prm.words())
    _hexfile(os.path.join(outdir, "cmds.mem"), cmds)
    _hexfile(os.path.join(outdir, "expect.mem"), expect_words(cfg, res))
    with open(os.path.join(outdir, "tb_params.vh"), "w") as f:
        f.write("// generated by golden_alif.py -- do not edit\n")
        for k, v in [("TB_P", cfg.P), ("TB_NIN", cfg.N_IN), ("TB_XBITS", cfg.XBITS),
                     ("TB_WORDS", cfg.words), ("TB_NCMD", len(cmds)),
                     ("TB_NTICK", cfg.T), ("TB_GAP", cfg.GAP), ("TB_TICK_CYCLES", TICK_CYCLES)]:
            f.write("`define %s %d\n" % (k, v))


def cmd_regress(base):
    for name in REGRESS:
        cfg, W, prm, steps, res = build_regress_set(name)
        out = os.path.join(base, "vec_" + name)
        write_set(out, cfg, W, prm, steps, res)
        print("%-8s -> %s\n         %s" % (name, out, coverage(cfg, res)))


# ----------------------------------------------------------------------------
# vectors for the ON-BOARD smoke test (firmware/snn_smoke.c)
# ----------------------------------------------------------------------------
HW = Cfg(P=16, N_IN=256, XBITS=8, T=25)     # the snn_layer instance in rtl/spike_soc.v


def cmd_hwvec(outpath):
    """C header: three sets, each with its own weights and parameters. A is the
    trained sheila layer 1 (first 16 neurons) on log-mel-like frames, pushed with
    busy polling; B (sparse, all 256 rows) and C (dense, hardware-field edges) are
    pushed WITHOUT polling, to prove the single event register never overflows."""
    cfg = HW
    cfg.check()
    rng = np.random.default_rng(2026)
    sets = []
    W, prm, steps = build_keyword_set(cfg, 11)
    sets.append(("A: trained sheila layer 1, 16 neurons, log-mel-like frames", W, prm, steps, 1))
    prm = random_params(rng, cfg.P)
    sets.append(("B: random, sparse (density 0.02)", random_weights(rng, cfg), prm,
                 random_steps(rng, cfg, 0.02), 0))
    prm = random_params(rng, cfg.P, wide=True)
    sets.append(("C: random, dense (density 0.15), hardware-field edges", random_weights(rng, cfg), prm,
                 random_steps(rng, cfg, 0.15), 0))

    L = ["// snn_vectors.h -- generated by `golden_alif.py hwvec` -- DO NOT EDIT.",
         "// Weights, ALIF parameters, event scripts and expected results for firmware/snn_smoke.c.",
         "// Bitstream configuration these vectors are valid for (rtl/spike_soc.v):",
         "//   ALIF snn_layer P=%d N_IN=%d XBITS=%d, %d time steps per set" % (cfg.P, cfg.N_IN, cfg.XBITS, cfg.T),
         "#ifndef SNN_VECTORS_H", "#define SNN_VECTORS_H", "",
         "#define SNN_P %d" % cfg.P, "#define SNN_NIN %d" % cfg.N_IN,
         "#define SNN_XBITS %d" % cfg.XBITS, "#define SNN_T %d" % cfg.T,
         "#define SNN_WORDS %d" % cfg.words, "#define SNN_SPKW %d" % cfg.spkw,
         "#define SNN_NSETS %d" % len(sets), ""]

    def arr(name, words):
        L.append("static const u32 %s[%d] = {" % (name, len(words)))
        for i in range(0, len(words), 8):
            L.append("    " + ", ".join("0x%08xu" % (int(w) & 0xFFFFFFFF) for w in words[i:i + 8]) + ",")
        L.append("};")
        L.append("")

    total = 0
    table = []
    print("hwvec: ALIF P=%d N_IN=%d XBITS=%d T=%d" % (cfg.P, cfg.N_IN, cfg.XBITS, cfg.T))
    for i, (label, W, prm, steps, polled) in enumerate(sets):
        res = run(cfg, W, prm, steps)
        cmds, exp, wts, prw = cmd_words(steps), expect_words(cfg, res), pack_weights(cfg, W), prm.words()
        L.append("// set %s" % label)
        arr("set%d_weights" % i, wts)
        arr("set%d_params" % i, prw)
        arr("set%d_cmds" % i, cmds)
        arr("set%d_expect" % i, exp)
        table.append((i, len(cmds), polled))
        total += len(wts) + len(prw) + len(cmds) + len(exp)
        print("  set %s  %s\n     events=%d  %s"
              % ("ABC"[i], "polled" if polled else "FAST (no busy polling)",
                 sum(len(e) for e in steps), coverage(cfg, res)))
    L.append("typedef struct { const u32 *weights; const u32 *params; const u32 *cmds; u32 ncmd;")
    L.append("                 const u32 *expect; u32 polled; } snn_set_t;")
    L.append("static const snn_set_t snn_sets[SNN_NSETS] = {")
    for i, n, polled in table:
        L.append("    { set%d_weights, set%d_params, set%d_cmds, %du, set%d_expect, %du }," % (i, i, i, n, i, polled))
    L.append("};")
    L += ["", "#endif // SNN_VECTORS_H", ""]
    with open(outpath, "w", newline="\n") as f:
        f.write("\n".join(L))
    print("wrote %s  (%d data words = %.1f KB of firmware rodata; image limit is 63.75 KB incl. code)"
          % (outpath, total, total * 4 / 1024.0))


# ----------------------------------------------------------------------------
# cross-check against the ALIF of branch ALIF-Layer1
# ----------------------------------------------------------------------------
def load_model_py(path=None):
    """The integer-only functions of snn_keyword/model.py (its torch imports are skipped)."""
    src = open(path).read() if path else git_file("code/snn_keyword/model.py").decode()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in ("_alif", "stream_state", "integer_forward_stream"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("I16", "I32") for t in n.targets))]
    assert len(keep) == 5, "model.py changed: expected I16, I32, stream_state, _alif, integer_forward_stream"
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "model.py", "exec"), ns)
    return ns


def cmd_crosscheck(model_py, npz, frames_n=200, seeds=3):
    """Layer 1 of model.integer_forward_stream() (all 128 neurons of the trained
    sheila model, frame by frame) against this event-driven model: u, a and s of
    every neuron after every frame must be equal."""
    m = load_model_py(model_py)
    W, prm, q = sheila_layer1(int(q_n1(npz)), npz)
    qd = {k: q[k] for k in q.files}
    cfg = Cfg(P=W.shape[1], N_IN=24, XBITS=8, T=frames_n)
    total = bad = 0
    for seed in range(seeds):
        rng = np.random.default_rng(100 + seed)
        frames = keyword_frames(rng, frames_n) if seed < 2 else rng.integers(0, 256, (frames_n, 24))
        L = Layer(cfg, W, prm, rtl=False)
        st = None
        for t in range(frames_n):
            _, st, _ = m["integer_forward_stream"](frames[None, t:t + 1], qd, state=st)
            for i, v in enumerate(frames[t]):
                L.event(i, int(v))
            L.tick()
            for name, ours in (("u1", L.u), ("a1", L.a), ("s1", L.s)):
                total += 1
                if not np.array_equal(st[name][0], ours):
                    bad += 1
                    if bad <= 5:
                        print("   MISMATCH seed %d frame %d %s" % (seed, t, name))
        print("   frames set %d: %d frames, %d layer-1 spikes, max a %d, %d fires blocked by adaptation"
              % (seed, frames_n, int(L.cnt.sum()), L.stats["a_max"], L.stats["adapted_fire_blocks"]))
    print("crosscheck: %d state comparisons (u1, a1, s1 of %d neurons per frame), %d mismatches -> %s"
          % (total, cfg.P, bad, "PASS" if bad == 0 else "FAIL"))
    return 0 if bad == 0 else 1


def q_n1(npz):
    import io
    q = np.load(npz if npz else io.BytesIO(git_file("code/snn_keyword/results/models/sheila_stream_int8.npz")),
                allow_pickle=True)
    return q["n1"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("regress"); sp.add_argument("--base", default=HERE)
    sp = sub.add_parser("hwvec")
    sp.add_argument("--out", default=os.path.join(HERE, "..", "firmware", "snn_vectors.h"))
    sp = sub.add_parser("crosscheck"); sp.add_argument("--model-py"); sp.add_argument("--npz")
    a = ap.parse_args()
    if a.cmd == "regress":
        cmd_regress(a.base)
    elif a.cmd == "hwvec":
        cmd_hwvec(a.out)
    else:
        sys.exit(cmd_crosscheck(a.model_py, a.npz))


if __name__ == "__main__":
    main()
