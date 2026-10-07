/* Second-stage keyword verifier, bit-exact with verifier_model.integer_forward and keyword_score.
 *
 * Two GRU layers and a linear readout per 20 ms step (two stacked frames), Q10
 * activations in int16, int8 weights stored offset by +128 as uint8 with one
 * power-of-two scale per row. A dot product is sum(a[i] * (u[i] - 128)) =
 * kdot(a, u) - 128 * sum(a): with USE_KDOT the kdot coprocessor (rtl/kdot_pcpi.v,
 * int16 x uint8, 3 cycles per 4 MACs) takes the int16 activations as its "weights"
 * and the uint8 weights as its "inputs", so the model stays one byte per weight.
 * Shifts are arithmetic (GCC); every int32 intermediate is range-checked at export.
 *
 * USE_KX (rtl/kdot_pcpi.v with KX = 1, ABI bit3): the KX instructions of kdot_pcpi
 * take over the work around the dot products, in levels (KX_LEVEL, default the highest):
 *   1  kdotc: S(a, u, n) = sum a[i] * (u[i] - 128) in hardware, no vsum() and no 128 * s.
 *   2  activation buffer: kload copies a layer step's activations into the unit once,
 *      kdotb streams only the weight rows (1 word per 4 MACs instead of 3).
 *   3  the work after each dot product in the unit too: kpre = (S >> e[i]) + b[k],
 *      ksig = sigmoid_q(kpre), ktanh = tanh_q(x); the loops only issue and store.
 *   4  as 3, with the row pointer and the e / b indices kept and advanced by the unit
 *      (kpren / ksign take no operands; the rows of a loop are contiguous).
 * Without USE_KDOT the same functions are plain C with identical results (native
 * reference, pure-RV32IM builds).
 */
#include "verifier.h"
#include "verifier_data.h"

#define ONE (1 << 10)
#if defined(USE_KX) && !defined(KX_LEVEL)
#define KX_LEVEL 4
#endif

/* [x (IN) | h1 (H1) | h2 (H2)]: layer 1 reads [x | h1], layer 2 reads [h1 | h2]. */
static int16_t act[VERIFIER_IN + VERIFIER_H1 + VERIFIER_H2] __attribute__((aligned(4)));
static int16_t hnew[VERIFIER_H1 > VERIFIER_H2 ? VERIFIER_H1 : VERIFIER_H2];
static int32_t gate_r[VERIFIER_H1 > VERIFIER_H2 ? VERIFIER_H1 : VERIFIER_H2];
static int32_t gate_z[VERIFIER_H1 > VERIFIER_H2 ? VERIFIER_H1 : VERIFIER_H2];
static int32_t gate_x[VERIFIER_H1 > VERIFIER_H2 ? VERIFIER_H1 : VERIFIER_H2];

_Static_assert(VERIFIER_IN % 4 == 0 && VERIFIER_H1 % 4 == 0 && VERIFIER_H2 % 4 == 0, "word-aligned rows");

#ifdef USE_KDOT
static inline void klen(unsigned n) { __asm__ volatile (".insn r 0x0b, 1, 0, x0, %0, x0" :: "r"(n)); }
static inline int32_t dot(const int16_t *a, const uint8_t *u, unsigned n)
{
    int32_t s;
    (void)n;
    __asm__ volatile (".insn r 0x0b, 0, 0, %0, %1, %2" : "=r"(s) : "r"(a), "r"(u) : "memory");
    return s;
}
#else
#ifdef USE_KX
static unsigned kx_len;                       /* the length register, for the C versions below */
static inline void klen(unsigned n) { kx_len = n; }
#else
static inline void klen(unsigned n) { (void)n; }
#endif
static inline int32_t dot(const int16_t *a, const uint8_t *u, unsigned n)
{
    int32_t s = 0;
    for (unsigned i = 0; i < n; ++i) s += (int32_t)a[i] * u[i];
    return s;
}
#endif

#ifndef USE_KX
static int32_t vsum(const int16_t *a, unsigned n)
{
    int32_t s = 0;
    for (unsigned i = 0; i < n; ++i) s += a[i];
    return s;
}
/* The corrected dot product S(a, u, n) = dot(a, u, n) - 128 * sum(a); s = sum(a) over the same n. */
#define DOTC(a, u, n, s) (dot(a, u, n) - 128 * (s))
#else
/* S(a, u, n) = sum a[i] * (u[i] - 128): kdotc, the same operands and length register as kdot. */
#ifdef USE_KDOT
static inline int32_t dotc(const int16_t *a, const uint8_t *u, unsigned n)
{
    int32_t s;
    (void)n;
    __asm__ volatile (".insn r 0x0b, 2, 0, %0, %1, %2" : "=r"(s) : "r"(a), "r"(u) : "memory");
    return s;
}
#else
static inline int32_t dotc(const int16_t *a, const uint8_t *u, unsigned n)
{
    int32_t s = 0;
    for (unsigned i = 0; i < n; ++i) s += (int32_t)a[i] * ((int32_t)u[i] - 128);
    return s;
}
#endif
#define DOTC(a, u, n, s) dotc(a, u, n)

#if KX_LEVEL >= 2
/* Activation buffer (KX_BUF elements in kdot_pcpi): kload copies klen elements of a, kseto
 * selects the first group (of 4) for kdotb, kdotb streams one weight row u of klen elements:
 * dotb(u) = S(buffer + 4 * off, u, klen). */
#define KX_BUF 128
_Static_assert(VERIFIER_IN + VERIFIER_H1 <= KX_BUF && VERIFIER_H1 + VERIFIER_H2 <= KX_BUF, "rows fit the KX buffer");
#ifdef USE_KDOT
static inline void kload(const int16_t *a) { __asm__ volatile (".insn r 0x0b, 3, 0, x0, %0, x0" :: "r"(a) : "memory"); }
static inline void kseto(unsigned groups) { __asm__ volatile (".insn r 0x0b, 5, 0, x0, %0, x0" :: "r"(groups)); }
static inline int32_t dotb(const uint8_t *u)
{
    int32_t s;
    __asm__ volatile (".insn r 0x0b, 4, 0, %0, %1, x0" : "=r"(s) : "r"(u) : "memory");
    return s;
}
#else
static int16_t kx_buf[KX_BUF];
static unsigned kx_off;
static inline void kload(const int16_t *a) { for (unsigned i = 0; i < kx_len; ++i) kx_buf[i] = a[i]; }
static inline void kseto(unsigned groups) { kx_off = 4 * groups; }
static inline int32_t dotb(const uint8_t *u) { return dotc(kx_buf + kx_off, u, kx_len); }
#endif
/* A row's dot product against the buffer (the activation pointer is the buffer's). */
#define DOTR(a, u, n, s) dotb(u)
#else
#define DOTR(a, u, n, s) DOTC(a, u, n, s)
#endif
#endif
#ifndef USE_KX
#define DOTR(a, u, n, s) DOTC(a, u, n, s)
#endif

static inline int32_t sigmoid_q(int32_t pre)
{
    pre = pre < -8 * ONE ? -8 * ONE : pre > 8 * ONE - 1 ? 8 * ONE - 1 : pre;
    return verifier_sig[(pre + 8 * ONE) >> 5];
}

static inline int32_t tanh_q(int32_t pre)
{
    pre = pre < -4 * ONE ? -4 * ONE : pre > 4 * ONE - 1 ? 4 * ONE - 1 : pre;
    return verifier_tanh[(pre + 4 * ONE) >> 4];
}

#if KX_LEVEL >= 3
/* Setup registers (byte addresses): e and b of the current layer, the two tables. kpre / ksig
 * stream one weight row against the buffer like kdotb, then apply (S >> e[ei]) + b[bi] and,
 * for ksig, sigmoid_q; the unit reads e[ei], b[bi] and the table entry itself. */
#define KX_PACK(ei, bi) (((uint32_t)(bi) << 16) | (uint32_t)(ei))
#ifdef USE_KDOT
#define KSET(sel, p) __asm__ volatile (".insn r 0x0b, 5, " #sel ", x0, %0, x0" :: "r"((uint32_t)(uintptr_t)(p)))
static inline int32_t kpre(const uint8_t *u, unsigned ei, unsigned bi)
{
    int32_t s;
    __asm__ volatile (".insn r 0x0b, 6, 0, %0, %1, %2" : "=r"(s) : "r"(u), "r"(KX_PACK(ei, bi)) : "memory");
    return s;
}
static inline int32_t ksig(const uint8_t *u, unsigned ei, unsigned bi)
{
    int32_t s;
    __asm__ volatile (".insn r 0x0b, 6, 1, %0, %1, %2" : "=r"(s) : "r"(u), "r"(KX_PACK(ei, bi)) : "memory");
    return s;
}
static inline int32_t ktanh(int32_t x)
{
    int32_t s;
    __asm__ volatile (".insn r 0x0b, 7, 0, %0, %1, x0" : "=r"(s) : "r"(x) : "memory");
    return s;
}
static inline void kx_layer(const uint8_t *e, const int32_t *b) { KSET(1, e); KSET(2, b); }
static inline void kx_tables(void) { KSET(3, verifier_sig); KSET(4, verifier_tanh); }
/* Rows w, w + stride, ... with indices (ei, bi), (ei + 1, bi + 1), ...: kpren / ksign. */
static inline void kx_rows(const uint8_t *w, unsigned ei, unsigned bi, unsigned stride)
{
    KSET(5, w);
    KSET(6, KX_PACK(ei, bi));
    KSET(7, stride);
}
static inline int32_t kpren(void)
{
    int32_t s;
    __asm__ volatile (".insn r 0x0b, 6, 2, %0, x0, x0" : "=r"(s) :: "memory");
    return s;
}
static inline int32_t ksign(void)
{
    int32_t s;
    __asm__ volatile (".insn r 0x0b, 6, 3, %0, x0, x0" : "=r"(s) :: "memory");
    return s;
}
#else
static const uint8_t *kx_e;
static const int32_t *kx_b;
static inline int32_t kpre(const uint8_t *u, unsigned ei, unsigned bi) { return (dotb(u) >> kx_e[ei]) + kx_b[bi]; }
static inline int32_t ksig(const uint8_t *u, unsigned ei, unsigned bi) { return sigmoid_q(kpre(u, ei, bi)); }
static inline int32_t ktanh(int32_t x) { return tanh_q(x); }
static inline void kx_layer(const uint8_t *e, const int32_t *b) { kx_e = e; kx_b = b; }
static inline void kx_tables(void) {}
static const uint8_t *kx_row;
static unsigned kx_ei, kx_bi, kx_stride;
static inline void kx_rows(const uint8_t *w, unsigned ei, unsigned bi, unsigned stride)
{
    kx_row = w; kx_ei = ei; kx_bi = bi; kx_stride = stride;
}
static inline int32_t kpren(void)
{
    const int32_t v = kpre(kx_row, kx_ei++, kx_bi++);
    kx_row += kx_stride;
    return v;
}
static inline int32_t ksign(void) { return sigmoid_q(kpren()); }
#endif
#endif

/* One GRU layer: a = [input (nin) | h (H)] in act; writes the new h into a[nin..]. */
#if KX_LEVEL >= 3
static void gru(int16_t *a, unsigned nin, unsigned H, const uint8_t *w, const uint8_t *e, const int32_t *b)
{
    const unsigned row = nin + H;
    kx_layer(e, b);
    klen(row);
    kload(a);                                 /* a is constant until the hnew copy at the end */
    kseto(0);
#if KX_LEVEL >= 4
    kx_rows(w, 0, 0, row);
    for (unsigned j = 0; j < H; ++j) gate_r[j] = ksign();
    for (unsigned j = 0; j < H; ++j) gate_z[j] = ksign();
    klen(nin);
    kx_rows(w + 2 * H * row, 2 * H, 2 * H, row);
    for (unsigned j = 0; j < H; ++j) gate_x[j] = kpren();
    klen(H);
    kseto(nin / 4);
    kx_rows(w + 2 * H * row + nin, 2 * H, 3 * H, row);
#else
    for (unsigned j = 0; j < H; ++j) gate_r[j] = ksig(w + j * row, j, j);
    for (unsigned j = 0; j < H; ++j) gate_z[j] = ksig(w + (H + j) * row, H + j, H + j);
    klen(nin);
    for (unsigned j = 0; j < H; ++j) gate_x[j] = kpre(w + (2 * H + j) * row, 2 * H + j, 2 * H + j);
    klen(H);
    kseto(nin / 4);
#endif
    for (unsigned j = 0; j < H; ++j) {
#if KX_LEVEL >= 4
        const int32_t gh = kpren();
#else
        const int32_t gh = kpre(w + (2 * H + j) * row + nin, 2 * H + j, 3 * H + j);
#endif
        const int32_t n = ktanh(gate_x[j] + ((gate_r[j] * gh) >> 10));
        const int32_t h = a[nin + j];
        hnew[j] = (int16_t)(n + ((gate_z[j] * (h - n)) >> 10));
    }
    for (unsigned j = 0; j < H; ++j) a[nin + j] = hnew[j];
}
#else
static void gru(int16_t *a, unsigned nin, unsigned H, const uint8_t *w, const uint8_t *e, const int32_t *b)
{
    const unsigned row = nin + H;
#ifndef USE_KX
    const int32_t s_all = vsum(a, row), s_x = vsum(a, nin), s_h = s_all - s_x;
#endif
    klen(row);
#if KX_LEVEL >= 2
    kload(a);                                 /* a is constant until the hnew copy at the end */
    kseto(0);
#endif
    for (unsigned j = 0; j < 2 * H; ++j) {
        const int32_t acc = DOTR(a, w + j * row, row, s_all);
        const int32_t g = sigmoid_q((acc >> e[j]) + b[j]);
        if (j < H) gate_r[j] = g; else gate_z[j - H] = g;
    }
    klen(nin);
    for (unsigned j = 0; j < H; ++j)
        gate_x[j] = (DOTR(a, w + (2 * H + j) * row, nin, s_x) >> e[2 * H + j]) + b[2 * H + j];
    klen(H);
#if KX_LEVEL >= 2
    kseto(nin / 4);
#endif
    for (unsigned j = 0; j < H; ++j) {
        const int32_t gh = (DOTR(a + nin, w + (2 * H + j) * row + nin, H, s_h) >> e[2 * H + j]) + b[3 * H + j];
        const int32_t n = tanh_q(gate_x[j] + ((gate_r[j] * gh) >> 10));
        const int32_t h = a[nin + j];
        hnew[j] = (int16_t)(n + ((gate_z[j] * (h - n)) >> 10));
    }
    for (unsigned j = 0; j < H; ++j) a[nin + j] = hnew[j];
}
#endif

#define LAST (2 * VERIFIER_NPH - 2)          /* state of the last phoneme; 2i+1: blank after phoneme i */
#define NS (LAST + 1 + VERIFIER_BOUNDARY)
static const uint8_t phones[VERIFIER_NPH] = VERIFIER_PHONES;
static inline int32_t max2(int32_t x, int32_t y) { return x > y ? x : y; }

void verifier_run(const uint8_t *frames, unsigned n, verifier_result_t *res, int32_t *logits)
{
    int32_t D[NS], P[NS], lg[VERIFIER_CLASSES];
    const unsigned steps = n / VERIFIER_STACK;
    for (unsigned i = 0; i < VERIFIER_IN + VERIFIER_H1 + VERIFIER_H2; ++i) act[i] = 0;
#if KX_LEVEL >= 3
    kx_tables();
#endif
    for (unsigned s = 0; s < NS; ++s) D[s] = VERIFIER_NEG;
    res->score_a = res->score_b = res->head = VERIFIER_NEG;
    res->end_a = res->end_b = res->end_head = -1;
    for (unsigned t = 0; t < steps; ++t) {
        const uint8_t *f = frames + t * VERIFIER_IN;
        for (unsigned i = 0; i < VERIFIER_IN; ++i) act[i] = (int16_t)(f[i] << 2);
        gru(act, VERIFIER_IN, VERIFIER_H1, verifier_w1, verifier_e1, verifier_b1);
        gru(act + VERIFIER_IN, VERIFIER_H1, VERIFIER_H2, verifier_w2, verifier_e2, verifier_b2);
        const int16_t *h2 = act + VERIFIER_IN + VERIFIER_H1;
#ifndef USE_KX
        const int32_t s2 = vsum(h2, VERIFIER_H2);
#endif
        int32_t top = INT32_MIN;
        klen(VERIFIER_H2);
#if KX_LEVEL >= 2
        kload(h2);
        kseto(0);
#endif
#if KX_LEVEL >= 3
        kx_layer(verifier_eo, verifier_bo);
#endif
#if KX_LEVEL >= 4
        kx_rows(verifier_wo, 0, 0, VERIFIER_H2);
#endif
        for (unsigned c = 0; c < VERIFIER_CLASSES; ++c) {
#if KX_LEVEL >= 4
            lg[c] = kpren();
#elif KX_LEVEL >= 3
            lg[c] = kpre(verifier_wo + c * VERIFIER_H2, c, c);
#else
            lg[c] = (DOTR(h2, verifier_wo + c * VERIFIER_H2, VERIFIER_H2, s2) >> verifier_eo[c]) + verifier_bo[c];
#endif
            if (lg[c] > top) top = lg[c];
            if (logits) logits[t * VERIFIER_CLASSES + c] = lg[c];
        }
#ifdef VERIFIER_HEAD
        {   /* keyword head (verifier_model.head_score): w . h2 per step, maximum over steps >= warmup */
            const int32_t hv = (DOTR(h2, verifier_wh, VERIFIER_H2, s2) >> VERIFIER_HEAD_E) + VERIFIER_HEAD_B;
            if (t >= VERIFIER_WARMUP && hv > res->head) { res->head = hv; res->end_head = (int32_t)t; }
        }
#endif
        /* Keyword path (verifier_model.keyword_score); costs c(k) = logit(k) - max <= 0. */
        const int32_t cb = lg[VERIFIER_BLANK] - top, cp = max2(cb, lg[phones[VERIFIER_NPH - 1]] - top);
        for (unsigned s = 0; s < NS; ++s) P[s] = D[s];
        const int32_t start = t >= VERIFIER_WARMUP ? 0 : VERIFIER_NEG;
        D[0] = max2(P[0], start) + (lg[phones[0]] - top);
        for (unsigned i = 1; i < VERIFIER_NPH; ++i) {
            const unsigned s = 2 * i;
            D[s - 1] = max2(P[s - 2], P[s - 1]) + cb;
            int32_t prev = max2(P[s - 1], P[s]);
            if (phones[i] != phones[i - 1]) prev = max2(prev, P[s - 2]);   /* CTC: equal phonemes need a blank */
            D[s] = prev + (lg[phones[i]] - top);
        }
        for (unsigned j = 0; j < VERIFIER_BOUNDARY; ++j) D[LAST + 1 + j] = P[LAST + j] + cp;
        for (unsigned s = 0; s < NS; ++s) D[s] = max2(D[s], VERIFIER_NEG);
        if (D[LAST] > res->score_a) { res->score_a = D[LAST]; res->end_a = (int32_t)t; }
        if (D[NS - 1] > res->score_b) { res->score_b = D[NS - 1]; res->end_b = (int32_t)t; }
    }
}
