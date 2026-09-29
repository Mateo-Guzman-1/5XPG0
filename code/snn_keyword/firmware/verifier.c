/* Second-stage "yes" verifier, bit-exact with verifier_model.integer_forward and keyword_score.
 *
 * Two GRU layers and a linear readout per 20 ms step (two stacked frames), Q10
 * activations in int16, int8 weights stored offset by +128 as uint8 with one
 * power-of-two scale per row. A dot product is sum(a[i] * (u[i] - 128)) =
 * kdot(a, u) - 128 * sum(a): with USE_KDOT the kdot coprocessor (rtl/kdot_pcpi.v,
 * int16 x uint8, 3 cycles per 4 MACs) takes the int16 activations as its "weights"
 * and the uint8 weights as its "inputs", so the model stays one byte per weight.
 * Shifts are arithmetic (GCC); every int32 intermediate is range-checked at export.
 */
#include "verifier.h"
#include "verifier_data.h"

#define ONE (1 << 10)

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
static inline void klen(unsigned n) { (void)n; }
static inline int32_t dot(const int16_t *a, const uint8_t *u, unsigned n)
{
    int32_t s = 0;
    for (unsigned i = 0; i < n; ++i) s += (int32_t)a[i] * u[i];
    return s;
}
#endif

static int32_t vsum(const int16_t *a, unsigned n)
{
    int32_t s = 0;
    for (unsigned i = 0; i < n; ++i) s += a[i];
    return s;
}

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

/* One GRU layer: a = [input (nin) | h (H)] in act; writes the new h into a[nin..]. */
static void gru(int16_t *a, unsigned nin, unsigned H, const uint8_t *w, const uint8_t *e, const int32_t *b)
{
    const unsigned row = nin + H;
    const int32_t s_all = vsum(a, row), s_x = vsum(a, nin), s_h = s_all - s_x;
    klen(row);
    for (unsigned j = 0; j < 2 * H; ++j) {
        const int32_t acc = dot(a, w + j * row, row) - 128 * s_all;
        const int32_t g = sigmoid_q((acc >> e[j]) + b[j]);
        if (j < H) gate_r[j] = g; else gate_z[j - H] = g;
    }
    klen(nin);
    for (unsigned j = 0; j < H; ++j)
        gate_x[j] = ((dot(a, w + (2 * H + j) * row, nin) - 128 * s_x) >> e[2 * H + j]) + b[2 * H + j];
    klen(H);
    for (unsigned j = 0; j < H; ++j) {
        const int32_t gh = ((dot(a + nin, w + (2 * H + j) * row + nin, H) - 128 * s_h) >> e[2 * H + j]) + b[3 * H + j];
        const int32_t n = tanh_q(gate_x[j] + ((gate_r[j] * gh) >> 10));
        const int32_t h = a[nin + j];
        hnew[j] = (int16_t)(n + ((gate_z[j] * (h - n)) >> 10));
    }
    for (unsigned j = 0; j < H; ++j) a[nin + j] = hnew[j];
}

#define NS (5 + VERIFIER_BOUNDARY)
static inline int32_t max2(int32_t x, int32_t y) { return x > y ? x : y; }

void verifier_run(const uint8_t *frames, unsigned n, verifier_result_t *res, int32_t *logits)
{
    int32_t D[NS], P[NS], lg[VERIFIER_CLASSES];
    const unsigned steps = n / VERIFIER_STACK;
    for (unsigned i = 0; i < VERIFIER_IN + VERIFIER_H1 + VERIFIER_H2; ++i) act[i] = 0;
    for (unsigned s = 0; s < NS; ++s) D[s] = VERIFIER_NEG;
    res->score_a = res->score_b = VERIFIER_NEG;
    res->end_a = res->end_b = -1;
    for (unsigned t = 0; t < steps; ++t) {
        const uint8_t *f = frames + t * VERIFIER_IN;
        for (unsigned i = 0; i < VERIFIER_IN; ++i) act[i] = (int16_t)(f[i] << 2);
        gru(act, VERIFIER_IN, VERIFIER_H1, verifier_w1, verifier_e1, verifier_b1);
        gru(act + VERIFIER_IN, VERIFIER_H1, VERIFIER_H2, verifier_w2, verifier_e2, verifier_b2);
        const int16_t *h2 = act + VERIFIER_IN + VERIFIER_H1;
        const int32_t s2 = vsum(h2, VERIFIER_H2);
        int32_t top = INT32_MIN;
        klen(VERIFIER_H2);
        for (unsigned c = 0; c < VERIFIER_CLASSES; ++c) {
            lg[c] = ((dot(h2, verifier_wo + c * VERIFIER_H2, VERIFIER_H2) - 128 * s2) >> verifier_eo[c]) + verifier_bo[c];
            if (lg[c] > top) top = lg[c];
            if (logits) logits[t * VERIFIER_CLASSES + c] = lg[c];
        }
        /* Keyword path (verifier_model.keyword_score); costs c(k) = logit(k) - max <= 0. */
        const int32_t cy = lg[VERIFIER_Y] - top, ce = lg[VERIFIER_EH] - top, cs = lg[VERIFIER_S] - top;
        const int32_t cb = lg[VERIFIER_BLANK] - top, cp = max2(cb, cs);
        for (unsigned s = 0; s < NS; ++s) P[s] = D[s];
        const int32_t start = t >= VERIFIER_WARMUP ? 0 : VERIFIER_NEG;
        D[0] = max2(P[0], start) + cy;
        D[1] = max2(P[0], P[1]) + cb;
        D[2] = max2(max2(P[0], P[1]), P[2]) + ce;
        D[3] = max2(P[2], P[3]) + cb;
        D[4] = max2(max2(P[2], P[3]), P[4]) + cs;
        for (unsigned j = 0; j < VERIFIER_BOUNDARY; ++j) D[5 + j] = P[4 + j] + cp;
        for (unsigned s = 0; s < NS; ++s) D[s] = max2(D[s], VERIFIER_NEG);
        if (D[4] > res->score_a) { res->score_a = D[4]; res->end_a = (int32_t)t; }
        if (D[NS - 1] > res->score_b) { res->score_b = D[NS - 1]; res->end_b = (int32_t)t; }
    }
}
