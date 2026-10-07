"""Instruction-level vectors for sim/tb_kx.sv: rtl/kdot_pcpi.v against the golden semantics.

Writes <out>/mem.hex (64K words of BRAM), <out>/prog.hex (per instruction: insn, rs1, rs2,
flags, expected rd; flags bit0 = writes rd, bit1 = must be claimed (pcpi_wait), bit2 = legacy,
bits 8.. = idle gap before it) and <out>/tb.vh. Golden semantics (firmware/verifier.c):
  kdot   rd = sum int16 a[i] * uint8 u[i]                       (32-bit wrap)
  kdotc  rd = S(a, u, n) = sum int16 a[i] * (uint8 u[i] - 128)  (32-bit wrap)
  kload  buf[0..n) = a[0..n);  kset 0 (off);  kdotb rd = S(buf[4*off ..], u, n)
  kset 1..4 (e, b, sigmoid, tanh bases); P = (S >> e[ei]) + b[bi] (arithmetic, e[4:0], wrap)
  kpre   rd = P;  ksig rd = sig(P);  ktanh rd = tnh(rs1)
  kset 5..7 (row pointer, {bi, ei}, stride); kpren / ksign: kpre / ksig on them, then advance
  sig(x) = ts[(clamp(x, -8192, 8191) + 8192) >> 5], tnh(x) = tt[(clamp(x, -4096, 4095) + 4096) >> 4]
usage: python kx_vectors.py <outdir> [--seed N] [--stage A] [--kx 0|1]
"""
import argparse
from pathlib import Path

import numpy as np

WORDS = 1 << 16
OP = 0x0B
M32 = 0xFFFFFFFF


def insn(funct3, funct7=0, rd=5, rs1=6, rs2=7):
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OP


class Mem:
    def __init__(self, rng):
        self.w = rng.integers(0, 1 << 32, WORDS, dtype=np.uint64).astype(np.uint32)
        self.next = 0x100                       # word address; low words stay random

    def alloc(self, nwords):
        assert self.next + nwords < WORDS - 16, 'directed vectors do not fit the 256 KB BRAM'
        a = self.next
        self.next += nwords + 1
        return a

    def i16(self, wa, n):
        v = np.frombuffer(self.w[wa:wa + (n + 1) // 2].tobytes(), '<i2')[:n]
        return v.astype(np.int64)

    def u8(self, wa, n):
        return np.frombuffer(self.w[wa:wa + (n + 3) // 4].tobytes(), np.uint8)[:n].astype(np.int64)

    def put_i16(self, wa, vals):
        v = np.asarray(vals, np.int64) & 0xFFFF
        for k in range(0, len(v), 2):
            self.w[wa + k // 2] = int(v[k]) | (int(v[k + 1]) << 16)

    def put_u8(self, wa, vals):
        v = np.asarray(vals, np.int64) & 0xFF
        for k in range(0, len(v), 4):
            self.w[wa + k // 4] = int(v[k]) | int(v[k + 1]) << 8 | int(v[k + 2]) << 16 | int(v[k + 3]) << 24


def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def gen_a(rng, n, kind):
    if kind == 'max':
        return np.full(n, 32767)
    if kind == 'min':
        return np.full(n, -32768)
    if kind == 'zero':
        return np.zeros(n, np.int64)
    if kind == 'mix':
        return rng.choice([-32768, 32767, 0, -1, 1], n)
    if kind == 'q10':                           # verifier activations: Q10, |a| <= 1024 mostly
        return rng.integers(-1024, 1025, n)
    return rng.integers(-32768, 32768, n)


def gen_u(rng, n, kind):
    if kind in (0, 128, 255):
        return np.full(n, kind)
    if kind == 'edge':
        return rng.choice([0, 1, 127, 128, 129, 254, 255], n)
    return rng.integers(0, 256, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--kx', type=int, default=1)
    ap.add_argument('--n', type=int, default=400, help='random dot products')
    ap.add_argument('--buf-groups', type=int, default=32)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    mem = Mem(rng)
    prog = []          # (insn, rs1, rs2, flags, expected or a function of the final memory)
    cur_len = 0

    def emit(word, rs1, rs2, writes, claimed, legacy, expected, gap):
        prog.append([word, rs1 & M32, rs2 & M32, (writes) | (claimed << 1) | (legacy << 2) | (gap << 8), expected])

    def klen(n, gap=0):
        nonlocal cur_len
        cur_len = n
        emit(insn(1), n, 0, 0, 1, 1, 0, gap)

    def golden(kind, aw, uw, n):     # evaluated on the final memory image
        def f():
            av, uv = mem.i16(aw, n), mem.u8(uw, n)
            return int(np.sum(av * uv)) if kind == 'kdot' else int(np.sum(av * (uv - 128)))
        return f

    def dot_at(kind, aw, uw, n, gap):
        if kind == 'kdot':
            emit(insn(0), aw * 4, uw * 4, 1, 1, 1, golden(kind, aw, uw, n), gap)
        else:
            emit(insn(2), aw * 4, uw * 4, 1, a.kx, 0, golden(kind, aw, uw, n), gap)

    def dot(kinds_, n, akind, ukind, gap):   # directed operands, written once, used by every kind
        aw, uw = mem.alloc(n // 2), mem.alloc(n // 4)
        mem.put_i16(aw, gen_a(rng, n, akind))
        mem.put_u8(uw, gen_u(rng, n, ukind))
        for kind in kinds_:
            dot_at(kind, aw, uw, n, gap)

    kinds = ['kdot'] + (['kdotc'] if a.kx else [])
    buf = {'aw': None, 'n': 0}             # what the last kload put in the buffer
    off = [0]

    def kload(aw, n, gap=0):
        klen(n, gap)
        emit(insn(3), aw * 4, 0, 0, 1, 0, 0, 0)
        buf['aw'], buf['n'] = aw, n

    def kset(sel, v, gap=0):
        if sel == 0:
            off[0] = v
        emit(insn(5, sel), v, 0, 0, 1, 0, 0, gap)

    base = {1: 0, 2: 0, 3: 0, 4: 0}       # kset 1..4 byte addresses

    def kbase(sel, byte_addr, gap=0):
        base[sel] = byte_addr
        emit(insn(5, sel), byte_addr, 0, 0, 1, 0, 0, gap)

    def table(ba, x, kind):
        lo, hi, sh = (-8192, 8191, 5) if kind == 'sig' else (-4096, 4095, 4)
        idx = (min(max(x, lo), hi) - lo) >> sh
        bap = ba + 2 * idx
        w = int(mem.w[bap >> 2])
        v = (w >> 16) & 0xFFFF if bap & 2 else w & 0xFFFF
        return v - 0x10000 if v & 0x8000 else v

    rowreg = {'ptr': 0, 'ei': 0, 'bi': 0, 'step': 0}

    def krows(ptr, ei, bi, step, gap=0):
        rowreg.update(ptr=ptr, ei=ei, bi=bi, step=step)
        emit(insn(5, 5), ptr, 0, 0, 1, 0, 0, gap)
        emit(insn(5, 6), (bi << 16) | ei, 0, 0, 1, 0, 0, 0)
        emit(insn(5, 7), step, 0, 0, 1, 0, 0, 0)

    def kpost_next(n, sigm, gap):
        r = rowreg
        kpost(r['ptr'] // 4, n, r['ei'], r['bi'], sigm, gap, nxt=True)
        r['ptr'] = (r['ptr'] + r['step']) & 0x3FFFF
        r['ei'] = (r['ei'] + 1) & 0xFFFF
        r['bi'] = (r['bi'] + 1) & 0xFFFF

    def kpost(uw, n, ei, bi, sigm, gap, nxt=False):
        aw0, o, bn = buf['aw'], off[0], buf['n']
        eb, bb, sb = base[1], base[2], base[3]
        assert 4 * o + n <= bn
        def f():
            av, uv = mem.i16(aw0, bn)[4 * o:4 * o + n], mem.u8(uw, n)
            S = s32(int(np.sum(av * (uv - 128))))
            ea = eb + ei
            e = (int(mem.w[ea >> 2]) >> (8 * (ea & 3))) & 31
            b = s32(int(mem.w[(bb + 4 * bi) >> 2]))
            P = s32((S >> e) + b)
            return table(sb, P, 'sig') if sigm else P
        if nxt:
            emit(insn(6, 3 if sigm else 2, rs1=0, rs2=0), 0, 0, 1, 1, 0, f, gap)
        else:
            emit(insn(6, 1 if sigm else 0), uw * 4, (bi << 16) | ei, 1, 1, 0, f, gap)

    def ktanh(x, gap=0):
        tb = base[4]
        emit(insn(7), x & M32, 0, 1, 1, 0, lambda: table(tb, s32(x), 'tanh'), gap)

    def kdotb(uw, n, gap):
        aw0, o = buf['aw'], off[0]
        assert 4 * o + n <= buf['n']
        def f():
            av, uv = mem.i16(aw0, buf_n)[4 * o:4 * o + n], mem.u8(uw, n)
            return int(np.sum(av * (uv - 128)))
        buf_n = buf['n']
        emit(insn(4), uw * 4, 0, 1, 1, 0, f, gap)
    # directed: every length 4..256 step 4 with edge operands, both instructions
    for n in range(4, 260, 4):
        klen(n)
        for akind, ukind in (('max', 255), ('min', 255), ('min', 0), ('max', 0), ('mix', 'edge'), ('zero', 128), ('rand', 128)):
            dot(kinds, n, akind, ukind, 0)
    # random: lengths, addresses anywhere in the (random) BRAM, idle gaps, back-to-back
    for _ in range(a.n):
        n = int(rng.integers(1, 65)) * 4
        klen(n, int(rng.integers(0, 3)))
        for _ in range(int(rng.integers(1, 4))):
            kind = kinds[int(rng.integers(0, len(kinds)))]
            aw = int(rng.integers(0, WORDS - n // 2))
            uw = int(rng.integers(0, WORDS - n // 4))
            dot_at(kind, aw, uw, n, int(rng.integers(0, 3)))
    if a.kx:
        G = a.buf_groups
        # directed: full buffer with extreme activations, every offset, short and long rows
        for akind in ('max', 'min', 'mix', 'q10', 'zero'):
            aw = mem.alloc(2 * G)
            mem.put_i16(aw, gen_a(rng, 4 * G, akind))
            kload(aw, 4 * G)
            for o in range(G):
                for m in sorted({1, G - o}):
                    for ukind in (0, 128, 255, 'edge'):
                        uw = mem.alloc(m)
                        mem.put_u8(uw, gen_u(rng, 4 * m, ukind))
                        kset(0, o)
                        klen(4 * m)
                        kdotb(uw, 4 * m, 0)
        # random: kload of 1..G groups from anywhere, then kdotb / kdot / kdotc mixed
        for _ in range(a.n):
            g = int(rng.integers(1, G + 1))
            kload(int(rng.integers(0, WORDS - 2 * g)), 4 * g, int(rng.integers(0, 3)))
            for _ in range(int(rng.integers(1, 6))):
                o = int(rng.integers(0, g)); m = int(rng.integers(1, g - o + 1))
                kset(0, o, int(rng.integers(0, 2)))
                klen(4 * m, int(rng.integers(0, 2)))
                kdotb(int(rng.integers(0, WORDS - m)), 4 * m, int(rng.integers(0, 3)))
                if rng.random() < 0.3:          # legacy / kdotc in between must not disturb the buffer
                    kind = kinds[int(rng.integers(0, len(kinds)))]
                    dot_at(kind, int(rng.integers(0, WORDS - 2 * m)), int(rng.integers(0, WORDS - m)), 4 * m, 0)
        # ---- stage C: e / b arrays, tables (4- and 2-byte aligned), kpre / ksig / ktanh ----
        ew, bw_ = mem.alloc(64), mem.alloc(256)           # 256 exponents, 256 biases
        evals = rng.integers(0, 256, 256)
        evals[:32] = np.arange(32)                          # every shift amount
        mem.put_u8(ew, evals)
        bvals = rng.integers(-(1 << 31), 1 << 31, 256, dtype=np.int64)
        edges = [-(1 << 31), (1 << 31) - 1, 0, -1, 1, -8193, -8192, -8191, -8190, 8190, 8191, 8192, 8193,
                 -4097, -4096, -4095, 4094, 4095, 4096, 4097, -100000, 100000]
        bvals[32:32 + len(edges)] = edges
        mem.w[bw_:bw_ + 256] = (bvals & M32).astype(np.uint32)
        tabs = []
        for align in (0, 2):
            tw = mem.alloc(257)
            tv = rng.integers(-32768, 32768, 512)
            tv[0], tv[511], tv[1], tv[510] = -32768, 32767, 12345, -12345
            raw = np.zeros(514, np.int64)
            raw[align // 2:align // 2 + 512] = tv
            mem.put_i16(tw, raw)
            tabs.append(tw * 4 + align)
        for ts, tt in ((tabs[0], tabs[1]), (tabs[1], tabs[0])):
            kbase(1, ew * 4); kbase(2, bw_ * 4); kbase(3, ts); kbase(4, tt)
            # S = 0 (u = 128 everywhere): P = b exactly, so the clamp edges are hit exactly
            aw = mem.alloc(2 * G); mem.put_i16(aw, gen_a(rng, 4 * G, 'rand'))
            kload(aw, 4 * G)
            uz = mem.alloc(G); mem.put_u8(uz, np.full(4 * G, 128))
            for bi in range(32, 32 + len(edges)):
                for o, m in ((0, G), (G - 1, 1), (3, 5)):
                    kset(0, o); klen(4 * m)
                    kpost(uz, 4 * m, int(rng.integers(0, 256)), bi, False, 0)
                    kpost(uz, 4 * m, int(rng.integers(0, 256)), bi, True, 0)
            # every exponent 0..31 with negative and positive S, extreme activations
            for akind in ('max', 'min', 'q10'):
                aw = mem.alloc(2 * G); mem.put_i16(aw, gen_a(rng, 4 * G, akind)); kload(aw, 4 * G)
                for ei in range(32):
                    m = int(rng.integers(1, G + 1)); o = int(rng.integers(0, G - m + 1))
                    uw = mem.alloc(m); mem.put_u8(uw, gen_u(rng, 4 * m, ['edge', 0, 255][ei % 3]))
                    kset(0, o); klen(4 * m)
                    kpost(uw, 4 * m, ei, int(rng.integers(0, 256)), bool(ei % 2), int(rng.integers(0, 2)))
            # ktanh: clamp edges, index 0 / 511, extremes, random
            for x in [-(1 << 31), (1 << 31) - 1, 0, -1, 1, -4097, -4096, -4095, -4081, 4079, 4080, 4094, 4095, 4096]:
                ktanh(x)
            for _ in range(200):
                ktanh(int(rng.integers(-6000, 6000)), int(rng.integers(0, 3)))
        # random mix: kload, kpre / ksig / kdotb / ktanh / legacy, back to back
        for _ in range(a.n):
            g = int(rng.integers(1, G + 1))
            kload(int(rng.integers(0, WORDS - 2 * g)), 4 * g, int(rng.integers(0, 2)))
            for _ in range(int(rng.integers(1, 5))):
                o = int(rng.integers(0, g)); m = int(rng.integers(1, g - o + 1))
                kset(0, o); klen(4 * m)
                uw = int(rng.integers(0, WORDS - m))
                r = rng.random()
                if r < 0.4:
                    kpost(uw, 4 * m, int(rng.integers(0, 256)), int(rng.integers(0, 256)), r < 0.2, int(rng.integers(0, 3)))
                elif r < 0.6:
                    kdotb(uw, 4 * m, 0)
                elif r < 0.8:
                    ktanh(int(rng.integers(-8000, 8000)), int(rng.integers(0, 2)))
                else:
                    kind = kinds[int(rng.integers(0, len(kinds)))]
                    dot_at(kind, int(rng.integers(0, WORDS - 2 * m)), int(rng.integers(0, WORDS - m)), 4 * m, 0)
        # ---- stage C2: auto-incrementing rows (as the verifier's layer loops) ----
        kbase(1, ew * 4); kbase(2, bw_ * 4); kbase(3, tabs[0]); kbase(4, tabs[1])
        for _ in range(60):
            g = int(rng.integers(1, G + 1))
            kload(int(rng.integers(0, WORDS - 2 * g)), 4 * g)
            o = int(rng.integers(0, g)); m = int(rng.integers(1, g - o + 1))
            kset(0, o); klen(4 * m)
            nrows = int(rng.integers(1, 40))
            step = 4 * int(rng.integers(m, 2 * m + 3))                  # rows contiguous (or padded)
            first = int(rng.integers(0, WORDS - nrows * step // 4 - m - 4)) * 4
            ei0 = int(rng.integers(0, 256 - nrows)); bi0 = int(rng.integers(0, 256 - nrows))
            krows(first, ei0, bi0, step)
            sigm = bool(rng.integers(0, 2))
            for k in range(nrows):
                kpost_next(4 * m, sigm, int(rng.integers(0, 2)))
                if rng.random() < 0.15:                                  # a normal kpre in between
                    kpost(int(rng.integers(0, WORDS - m)), 4 * m, int(rng.integers(0, 256)), int(rng.integers(0, 256)), False, 0)
        # klen 0: kpre returns b[bi] >> 0-shift of 0 = (0 >> e) + b
        klen(0)
        kpost(0x400, 0, 0, 40, False, 0)
        # klen 0: kdotb returns 0, kload loads nothing
        klen(0)
        emit(insn(4), 0x400, 0, 1, 1, 0, 0, 0)
    # length 0 (klen 0): legacy kdot returns 0 at once
    klen(0)
    emit(insn(0), 0x400, 0x800, 1, 1, 1, 0, 0)
    # unclaimed encodings must not assert pcpi_wait: funct7 != 0, other opcodes, and
    # (KX = 0) the KX funct3 range
    for f3, f7 in ([(0, 1), (1, 0x40), (2, 1), (3, 1), (4, 1), (5, 8), (6, 4), (7, 1)] if a.kx else
                   [(0, 1), (1, 0x40), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]):
        emit(insn(f3, f7), 0, 0, 0, 0, 0, 0, 0)
    emit(0x00000013, 0, 0, 0, 0, 0, 0, 0)       # addi x0, x0, 0 (not custom-0)

    for p_ in prog:                              # expected values from the final memory
        if callable(p_[4]):
            p_[4] = p_[4]()
        p_[4] &= M32
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'mem.hex').write_text('\n'.join('%08x' % int(w) for w in mem.w) + '\n')
    (a.out / 'prog.hex').write_text('\n'.join('%08x' % v for p in prog for v in p) + '\n')
    (a.out / 'tb.vh').write_text(f'`define NPROG {len(prog)}\n`define KX {a.kx}\n`define SEED {a.seed}\n')
    n_dots = sum(1 for p in prog if p[3] & 1)
    print(f'kx_vectors: {len(prog)} instructions ({n_dots} with a result), KX={a.kx}, seed={a.seed} -> {a.out}')


if __name__ == '__main__':
    main()
