/* Streaming SNN kernel, bit-exact with model.integer_forward_stream().
 *
 * Arithmetic: shifts are arithmetic (GCC), membranes saturate to int16, the
 * readout and the score to int32. Layer 1 is a dense 24-input dot product
 * per neuron (int16 weights, the kdot format). Layer 2 and the readout are
 * event-driven: a layer-1 spike emitted tau frames ago reaches exactly the
 * synapses whose delay is tau, listed per (neuron, delay) in a CSR table;
 * the delay ring keeps, per frame, the list of layer-1 neurons that spiked;
 * recurrent and readout weights are added per spiking neuron only.
 */
#include "stream_infer.h"
#include "model_stream_data.h"

#ifdef PROFILE
/* Cycles per section, summed; stream_main.c publishes them per hop (MB[20..24]). */
uint32_t stream_prof[5];
#define TIMER_NOW (*(volatile uint32_t *)0x10001000u)
#define MARK(i) do { uint32_t now_ = TIMER_NOW; stream_prof[i] += now_ - t_; t_ = now_; } while (0)
#else
#define MARK(i) do { } while (0)
#endif

static int32_t sat16(int32_t v) { return v < -32768 ? -32768 : v > 32767 ? 32767 : v; }
static int32_t sat32(int64_t v) { return v < INT32_MIN ? INT32_MIN : v > INT32_MAX ? INT32_MAX : (int32_t)v; }

static uint8_t alif(int32_t *u, int32_t *a, uint8_t s_prev, int32_t current, int32_t theta, int32_t bq,
                    uint8_t km, uint8_t ka)
{
    *a = *a - (*a >> ka) + ((int32_t)s_prev << 8);
    int32_t thr = theta + ((bq * *a) >> 8);
    *u = sat16(*u - (*u >> km) + current);
    uint8_t s = *u >= thr;
    if (s) *u = sat16(*u - thr);
    return s;
}

#ifdef USE_ENGINE
/* rtl/neuron_engine.v at 0x1000_4000: layer 2 and the readout in the fabric. */
#define ENG(offset) (*(volatile uint32_t *)(0x10004000u + (offset)))
_Static_assert(STREAM_N1 <= 128 && STREAM_N2 == 128, "engine is built for N1 <= 128, N2 = 128");
_Static_assert(STREAM_CLASSES <= 4 && STREAM_PO < 16, "engine readout: <= 4 classes, PO < 16");

static void engine_begin(unsigned table) { ENG(0x20) = (uint32_t)table << 28; }

void stream_init(void)
{
    engine_begin(0);
    for (unsigned k = 0; k <= STREAM_N1 * 32; ++k) ENG(0x24) = stream_syn_off[k];
    engine_begin(1);
    for (unsigned e = 0; e < STREAM_SYNAPSES; ++e)
        ENG(0x24) = ((uint32_t)(uint8_t)stream_syn_w[e] << 8) | stream_syn_post[e];
    engine_begin(2);   /* rec[j][k] at byte k*N2 + j: stream_rect is that layout, word aligned */
    for (unsigned w = 0; w < STREAM_N2 * STREAM_N2 / 4; ++w) ENG(0x24) = ((const uint32_t *)stream_rect)[w];
    engine_begin(3);
    for (unsigned j = 0; j < STREAM_N2; ++j) {
        uint32_t word = 0;
        for (unsigned c = 0; c < STREAM_CLASSES; ++c)
            word |= (uint32_t)(uint8_t)stream_wot[j * STREAM_CLASSES + c] << (8 * c);
        ENG(0x24) = word;
    }
    engine_begin(4);
    for (unsigned j = 0; j < STREAM_N2; ++j)
        ENG(0x24) = (uint32_t)stream_theta2[j] | (uint32_t)stream_p2[j] << 16 |
                    (uint32_t)stream_km2[j] << 20 | (uint32_t)stream_ka2[j] << 24;
    engine_begin(5);
    for (unsigned j = 0; j < STREAM_N2; ++j) ENG(0x24) = (uint32_t)stream_bq2[j];
    engine_begin(6);
    for (unsigned j = 0; j < STREAM_N2; ++j) ENG(0x24) = (uint32_t)stream_b2[j];
    engine_begin(7);
    for (unsigned c = 0; c < STREAM_CLASSES; ++c) ENG(0x24) = stream_ko[c];
    engine_begin(8);
    for (unsigned c = 0; c < STREAM_CLASSES; ++c) ENG(0x24) = (uint32_t)stream_bo[c];
    ENG(0x28) = STREAM_YES | STREAM_CLASSES << 4 | STREAM_PO << 8;
}
#else
void stream_init(void) {}
#endif

void stream_reset(stream_state_t *st)
{
    uint8_t *p = (uint8_t *)st;
    for (unsigned i = 0; i < sizeof *st; ++i) p[i] = 0;
#ifdef USE_ENGINE
    ENG(0x00) = 1;
    while (ENG(0x04) & 1) ;
#endif
}

int32_t stream_step(stream_state_t *st, const uint8_t *frame, uint32_t spikes[2], uint32_t *events)
{
    uint32_t n1 = 0, n2 = 0, ev = 0;
#ifdef PROFILE
    uint32_t t_ = TIMER_NOW;
#endif
#ifdef USE_KDOT
    /* kdot PCPI (rtl/kdot_pcpi.v): int16 weights x uint8 inputs, 4 per group; frame must be word aligned. */
    _Static_assert(STREAM_BANDS % 4 == 0, "kdot processes 4 elements per group");
    __asm__ volatile (".insn r 0x0b, 1, 0, x0, %0, x0" :: "r"(STREAM_BANDS));
#endif
    for (unsigned h = 0; h < STREAM_N1; ++h) {
        int32_t acc = 0;
#ifdef USE_KDOT
        __asm__ volatile (".insn r 0x0b, 0, 0, %0, %1, %2"
                          : "=r"(acc) : "r"(&stream_w1[h * STREAM_BANDS]), "r"(frame) : "memory");
#else
        for (unsigned i = 0; i < STREAM_BANDS; ++i)
            acc += (int32_t)stream_w1[h * STREAM_BANDS + i] * frame[i];
#endif
        int32_t cur = (acc >> stream_r1[h]) + stream_b1[h];
        st->s1[h] = alif(&st->u1[h], &st->a1[h], st->s1[h], cur, stream_theta1[h], stream_bq1[h],
                         stream_km1[h], stream_ka1[h]);
        n1 += st->s1[h];
#ifdef USE_ENGINE
        if (st->s1[h]) ENG(0x08) = h;
#endif
    }
    MARK(0);  /* layer 1 */
#ifdef USE_ENGINE
    ENG(0x00) = 2;
    while (ENG(0x04) & 1) ;
    MARK(1);  /* engine: delayed, recurrent, layer 2, readout */
    spikes[0] = n1; spikes[1] = ENG(0x10); *events = ENG(0x14);
    return (int32_t)ENG(0x0c);
#endif
    const uint32_t pos = st->pos;
    uint32_t count = 0;
    for (unsigned h = 0; h < STREAM_N1; ++h)
        if (st->s1[h]) st->spk[pos][count++] = (uint8_t)h;
    st->spk_n[pos] = (uint8_t)count;

    /* Only the ~5 neurons per frame that spiked in each of the last 32 frames are visited. */
    int32_t acc2[STREAM_N2] = {0};
    for (unsigned tau = 0; tau < 32; ++tau) {
        const unsigned slot = (pos - tau) & 31;
        const uint8_t *list = st->spk[slot];
        for (unsigned n = st->spk_n[slot]; n; --n) {
            const unsigned k = (unsigned)*list++ * 32 + tau;
            const unsigned e0 = stream_syn_off[k], e1 = stream_syn_off[k + 1];
            for (unsigned e = e0; e < e1; ++e)
                acc2[stream_syn_post[e]] += stream_syn_w[e];
            ev += e1 - e0;
        }
    }
    MARK(1);  /* delayed synapses */
    for (unsigned k = 0; k < STREAM_N2; ++k) {   /* recurrence: previous frame's layer-2 spikes */
        if (!st->s2[k]) continue;
        for (unsigned j = 0; j < STREAM_N2; ++j) acc2[j] += stream_rect[k * STREAM_N2 + j];
        ev += STREAM_N2;
    }
    MARK(2);  /* recurrence */
    for (unsigned j = 0; j < STREAM_N2; ++j) {
        int32_t cur = acc2[j] * ((int32_t)1 << stream_p2[j]) + stream_b2[j];  /* << without UB for negatives */
        st->s2[j] = alif(&st->u2[j], &st->a2[j], st->s2[j], cur, stream_theta2[j], stream_bq2[j],
                         stream_km2[j], stream_ka2[j]);
        n2 += st->s2[j];
    }
    st->pos = (pos + 1) & 31;
    MARK(3);  /* layer 2 neurons */

    int32_t acc_o[STREAM_CLASSES] = {0};
    for (unsigned j = 0; j < STREAM_N2; ++j) {
        if (!st->s2[j]) continue;
        for (unsigned c = 0; c < STREAM_CLASSES; ++c) acc_o[c] += stream_wot[j * STREAM_CLASSES + c];
        ev += STREAM_CLASSES;
    }
    int64_t best = INT64_MIN;
    for (unsigned c = 0; c < STREAM_CLASSES; ++c) {
        int32_t o = st->o[c];
        st->o[c] = sat32((int64_t)o - (o >> stream_ko[c]) + (int64_t)acc_o[c] * ((int64_t)1 << STREAM_PO) + stream_bo[c]);
        if (c != STREAM_YES && st->o[c] > best) best = st->o[c];
    }
    MARK(4);  /* readout */
    spikes[0] = n1; spikes[1] = n2; *events = ev;
    return sat32((int64_t)st->o[STREAM_YES] - best);
}
