// Group-2 keyword detector firmware for PicoRV32.
//
// A board-side Python bridge writes one uint8 spectrogram to BRAM and
// submits MB_CMD_CLASSIFY.  This firmware runs the exported two-layer LIF SNN
// with integer arithmetic, returns both output spike counts and the measured
// cycle count, and lights all board LEDs for one second on a keyword decision.

#include "board.h"
#include "hal.h"
// KW_MODEL_HEADER selects another generated model (quantization experiments);
// without it the deployed keyword_model.h next to this file is used.
#ifdef KW_MODEL_HEADER
#include KW_MODEL_HEADER
#else
#include "keyword_model.h"
#endif

// KW_VARIANT selects the layer-1 inner loop (research/build_quant_firmware.sh):
//   0  one int8 weight per byte, one lb per weight (Damien's loop)
//   1  same storage, loop unrolled 4x (control for the packed loops)
//   2  int8 weights packed 4 per 32-bit word, one lw per 4 weights
//   3  int4 weights packed 8 per 32-bit word, one lw per 8 weights
// Without -DKW_VARIANT the model header decides: a 4-bit packed model (the
// default QAT int4 model) uses loop 3, any other model uses loop 0.
#ifndef KW_VARIANT
#if defined(KW_WEIGHT_BITS) && KW_WEIGHT_BITS == 4
#define KW_VARIANT 3
#else
#define KW_VARIANT 0
#endif
#endif
#if defined(KW_WEIGHT_BITS)
#define KW_REPORT_VARIANT 1
#endif

// Sign-extended lane j of a packed word: shift the lane to the top, then
// arithmetic-shift it back down (matches research/export_quant.py).
#define KW_LANE(word, j, bits) \
    (((s32)((word) << (32 - ((j) + 1) * (bits)))) >> (32 - (bits)))

// Clock used for the layer-1 timer; the native host check
// (research/firmware_host_check.c) has no TIMER peripheral and stubs it out.
#ifndef KW_LAYER1_NOW
#define KW_LAYER1_NOW() timer_now()
#endif

static console_ring_t *const con = (console_ring_t *)CONSOLE_BASE;
static mailbox_t *const mb = (mailbox_t *)MAILBOX_BASE;
static volatile u32 *const sph = (volatile u32 *)SPIKE_LOG_BASE;
static volatile u32 *const timing = (volatile u32 *)KW_TIMING_BASE;

static u32 led_off_at;
static int led_on;
static u32 last_sequence;
static u32 last_decision;
static u32 last_cycles;
static u32 classifications;


static void con_putc(char c)
{
    if (c == '\n')
        con_putc('\r');
    u32 h = con->head, t = con->tail;
    if (h - t >= 512)
        return;
    con->buf[h & CONSOLE_BUF_MASK] = c;
    con->head = h + 1;
}


static void con_puts(const char *s)
{
    while (*s)
        con_putc(*s++);
}


static void con_putdec(u32 v)
{
    char b[12];
    int i = 0;
    do {
        b[i++] = '0' + (v % 10);
        v /= 10;
    } while (v);
    while (i)
        con_putc(b[--i]);
}


static void spike_log_init(void)
{
    sph[0] = 0;
    sph[1] = 0;
    sph[2] = 0;
    sph[3] = SPIKE_LOG_VERSION;
}


static void spike_log_push(u32 out_id)
{
    u32 tail = sph[1], head = sph[0];
    if (head - tail >= SPIKE_LOG_NWORDS) {
        sph[2] = sph[2] + 1;
        return;
    }
    u32 ts = timer_now() & 0xFFFFFFu;
    ((volatile u32 *)SPIKE_LOG_DATA)[head & SPIKE_LOG_MASK] =
        (ts << 8) | (out_id & 0xFFu);
    sph[0] = head + 1;
}


static void classify(const volatile u8 *input, s32 score[KW_OUTPUTS],
                     u32 *layer1_cycles)
{
    s32 current1[KW_HIDDEN];
    s32 mem1[KW_HIDDEN];
    s32 mem2[KW_OUTPUTS];
    u8 spike1[KW_HIDDEN];

    // The first-layer current is constant across the SNN time axis.  Hoisting
    // this dot product is mathematically identical to recomputing it at each
    // step and saves most of the PicoRV32 work.
    u32 layer1_started = KW_LAYER1_NOW();
    for (int h = 0; h < KW_HIDDEN; h++) {
        s32 sum = kw_b1[h];
#if KW_VARIANT == 0
        for (int i = 0; i < KW_INPUTS; i++)
            sum += (s32)kw_w1[h * KW_INPUTS + i] * (s32)input[i];
#elif KW_VARIANT == 1
        _Static_assert(KW_INPUTS % 4 == 0, "unroll needs a multiple of 4 inputs");
        for (int i = 0; i < KW_INPUTS; i += 4) {
            const int8_t *w = &kw_w1[h * KW_INPUTS + i];
            sum += (s32)w[0] * (s32)input[i];
            sum += (s32)w[1] * (s32)input[i + 1];
            sum += (s32)w[2] * (s32)input[i + 2];
            sum += (s32)w[3] * (s32)input[i + 3];
        }
#elif KW_VARIANT == 2
#if !defined(KW_WEIGHT_BITS) || KW_WEIGHT_BITS != 8
#error "variant 2 needs a model exported with 8-bit packed weights"
#endif
        const u32 *wp = &kw_w1_packed[h * (KW_INPUTS / 4)];
        for (int k = 0; k < KW_INPUTS / 4; k++) {
            u32 word = wp[k];
            const volatile u8 *x = &input[4 * k];
            sum += KW_LANE(word, 0, 8) * (s32)x[0];
            sum += KW_LANE(word, 1, 8) * (s32)x[1];
            sum += KW_LANE(word, 2, 8) * (s32)x[2];
            sum += KW_LANE(word, 3, 8) * (s32)x[3];
        }
#elif KW_VARIANT == 3
#if !defined(KW_WEIGHT_BITS) || KW_WEIGHT_BITS != 4
#error "variant 3 needs a model exported with 4-bit packed weights"
#endif
        const u32 *wp = &kw_w1_packed[h * (KW_INPUTS / 8)];
        for (int k = 0; k < KW_INPUTS / 8; k++) {
            u32 word = wp[k];
            const volatile u8 *x = &input[8 * k];
            sum += KW_LANE(word, 0, 4) * (s32)x[0];
            sum += KW_LANE(word, 1, 4) * (s32)x[1];
            sum += KW_LANE(word, 2, 4) * (s32)x[2];
            sum += KW_LANE(word, 3, 4) * (s32)x[3];
            sum += KW_LANE(word, 4, 4) * (s32)x[4];
            sum += KW_LANE(word, 5, 4) * (s32)x[5];
            sum += KW_LANE(word, 6, 4) * (s32)x[6];
            sum += KW_LANE(word, 7, 4) * (s32)x[7];
        }
#else
#error "unknown KW_VARIANT"
#endif
        current1[h] = sum;
        mem1[h] = 0;
    }
    *layer1_cycles = KW_LAYER1_NOW() - layer1_started;
    for (int o = 0; o < KW_OUTPUTS; o++) {
        mem2[o] = 0;
        score[o] = 0;
    }

    for (int step = 0; step < KW_TIMESTEPS; step++) {
        for (int h = 0; h < KW_HIDDEN; h++) {
            mem1[h] = ((mem1[h] * KW_BETA_Q8) >> 8) + current1[h];
            if (mem1[h] >= KW_THRESHOLD1) {
                spike1[h] = 1;
                mem1[h] -= KW_THRESHOLD1;       // soft reset
            } else {
                spike1[h] = 0;
            }
        }

        for (int o = 0; o < KW_OUTPUTS; o++) {
            s32 current2 = kw_b2[o];
            for (int h = 0; h < KW_HIDDEN; h++)
                if (spike1[h])
                    current2 += (s32)kw_w2[o * KW_HIDDEN + h];
            mem2[o] = ((mem2[o] * KW_BETA_Q8) >> 8) + current2;
            if (mem2[o] >= KW_THRESHOLD2) {
                score[o]++;
                mem2[o] -= KW_THRESHOLD2;
            }
        }
    }
}


static void mb_reply(u32 r0, u32 r1, u32 r2, u32 r3)
{
    mb->resp[0] = r0;
    mb->resp[1] = r1;
    mb->resp[2] = r2;
    mb->resp[3] = r3;
    mb->seq_out = mb->seq_in;
}


static void mb_service(void)
{
    if (mb->seq_in == mb->seq_out)
        return;

    u32 cmd = mb->cmd[0];
    u32 a0 = mb->cmd[1], a1 = mb->cmd[2], a2 = mb->cmd[3];
    if (cmd == MB_CMD_NOP) {
        mb_reply(0, 0, 0, 0);
    } else if (cmd == MB_CMD_ECHO) {
        mb_reply(a0, a1, a2, 0);
    } else if (cmd == MB_CMD_STATUS) {
        mb_reply(last_sequence, last_decision, last_cycles, classifications);
    } else if (cmd == MB_CMD_CLASSIFY) {
        // The generated model is the authority for the frontend shape.  This
        // keeps sweep candidates with a different band count deployable
        // without editing the board-wide memory-map header.
        if (a0 != INPUT_FRAME_BASE || a1 != KW_INPUTS) {
            mb_reply(0xFFFFFFFFu, a0, a1, 0);
            return;
        }
        s32 score[KW_OUTPUTS];
        u32 layer1_cycles;
        u32 started = timer_now();
        classify((const volatile u8 *)a0, score, &layer1_cycles);
        u32 cycles = timer_now() - started;
        timing[0] = KW_TIMING_MAGIC;
        timing[1] = a2;
        timing[2] = layer1_cycles;
        timing[3] = cycles;
        // A strict confidence policy rejects ties, weak output activity, and
        // marginal wins that are common for silence or stationary noise.
        u32 decision =
            score[1] >= KW_MIN_KEYWORD_SPIKES &&
            score[1] >= score[0] + KW_DECISION_MARGIN;
        last_sequence = a2;
        last_decision = decision;
        last_cycles = cycles;
        classifications++;
        if (decision) {
            led_write(0x3FFu);
            led_off_at = timer_now() + CLK_HZ;  // one second
            led_on = 1;
            spike_log_push(1);
        }
        mb_reply(decision, (u32)score[0], (u32)score[1], cycles);
    } else {
        mb_reply(0xDEADBEEFu, cmd, 0, 0);
    }
}


void main(void)
{
    con->head = 0;
    con->tail = 0;
    spike_log_init();
    mb->seq_out = 0;
    led_write(0x001u);

    con_puts("Group 2 keyword SNN ready: keyword='");
    con_puts(KW_KEYWORD);
    con_puts("' inputs=");
    con_putdec(KW_INPUTS);
    con_puts(" hidden=");
    con_putdec(KW_HIDDEN);
    con_puts(" steps=");
    con_putdec(KW_TIMESTEPS);
#ifdef KW_REPORT_VARIANT
    con_puts(" variant=");
    con_putdec(KW_VARIANT);
#ifdef KW_WEIGHT_BITS
    con_puts(" weight_bits=");
    con_putdec(KW_WEIGHT_BITS);
#endif
#endif
    con_puts("\n");

    for (;;) {
        mb_service();
        if (led_on && (s32)(timer_now() - led_off_at) >= 0) {
            led_write(0x001u);
            led_on = 0;
        }
    }
}
