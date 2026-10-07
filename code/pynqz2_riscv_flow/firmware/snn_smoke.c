// snn_smoke.c -- ON-BOARD functional check of the ALIF rtl/snn_layer.v.
//
// Replays the golden-model vectors (snn_vectors.h, made by `golden_alif.py hwvec`)
// through the same MMIO paths the real firmware will use. Per set it loads the
// weights and the per-neuron ALIF parameters (NP, BQ, BIAS; read back), then
// compares EVERY tick against the expected result:
//   * fire vector (SPK regs), membrane u (U regs) and adaptation trace a (A regs)
//     of every neuron,
//   * final spike counters and tick counter,
//   * evt_ovf must stay 0 (sets B and C push events WITHOUT polling busy).
// Set A is layer 1 of the trained 'sheila' ALIF model (first SNN_P neurons).
// Also checks the reset state (ID "ALIF", CONFIG, NP reset value, ticks 0, not busy).
//
// Result reporting (the PS reads it with the existing host tool):
//   spike_pynq.py console      -> short text report (ring is only 512 bytes)
//   spike_pynq.py cmd status   -> 4 words:  time  checks  verdict  mismatches
//                                 verdict = 0x50415353 ("PASS") or 0x4641494c ("FAIL")
//
// Build with firmware/build_smoke.sh (does not touch spike.bin / the demo).
//
// Written without a RISC-V toolchain at hand: the test LOGIC was compiled natively
// against a C model of the register map (and shown to catch injected bugs), but this
// file has not been compiled for rv32im.
#include "board.h"
#ifndef SNN_HOST_MOCK
#include "hal.h"
#endif
#include "snn_vectors.h"

// ---- snn_layer register map (see the header of rtl/snn_layer.v) ----
#define SNN_WGT_BASE   0x1000C000u                        // weight window, write-only (0x1000_4000 is the neuron engine)
#define SNN_REG_BASE   0x10008000u
#define SNN_CTRL       (SNN_REG_BASE + 0x000u)            // W: bit0 CLR, bit1 TICK
#define SNN_STATUS     (SNN_REG_BASE + 0x004u)            // R: bit0 busy, bit1 evt_ovf
#define SNN_EVT        (SNN_REG_BASE + 0x008u)            // W: (x << 16) | input row
#define SNN_CONFIG     (SNN_REG_BASE + 0x00Cu)            // R: P | XBITS << 8 | N_IN << 16
#define SNN_SPK(n)     (SNN_REG_BASE + 0x010u + 4u * (u32)(n))
#define SNN_TICKS      (SNN_REG_BASE + 0x030u)
#define SNN_ID         (SNN_REG_BASE + 0x034u)            // R: 0x414C4946 "ALIF"
#define SNN_CNT(c)     (SNN_REG_BASE + 0x400u + 4u * (u32)(c))
#define SNN_NP(c)      (SNN_REG_BASE + 0x500u + 4u * (u32)(c))  // theta | r<<16 | km<<20 | ka<<24
#define SNN_BQ(c)      (SNN_REG_BASE + 0x600u + 4u * (u32)(c))  // signed, 18 bits
#define SNN_BIAS(c)    (SNN_REG_BASE + 0x700u + 4u * (u32)(c))
#define SNN_U(c)       (SNN_REG_BASE + 0x800u + 4u * (u32)(c))  // membrane (sign-extended)
#define SNN_A(c)       (SNN_REG_BASE + 0x900u + 4u * (u32)(c))  // adaptation trace

#define VERDICT_PASS   0x50415353u                        // "PASS"
#define VERDICT_FAIL   0x4641494Cu                        // "FAIL"

_Static_assert(SNN_WORDS == SNN_NIN * SNN_P / 4, "vector header does not match the weight packing");

// ------------------------------------------------------------------
// platform glue: real hardware vs. host mock (mock only used for testing the test)
// ------------------------------------------------------------------
#ifdef SNN_HOST_MOCK
extern void mock_wr(u32 addr, u32 v);
extern u32  mock_rd(u32 addr);
extern void mock_putc(char c);
extern u32  mock_now(void);
#define REG_WR(a, v)   mock_wr((a), (v))
#define REG_RD(a)      mock_rd(a)
static void out_putc(char c) { mock_putc(c); }
static u32  now(void)        { return mock_now(); }
static void led(u32 v)       { (void)v; }
static mailbox_t mb_store;
static mailbox_t *const mb = &mb_store;
#else
#define REG_WR(a, v)   wr32((a), (v))
#define REG_RD(a)      rd32(a)
static console_ring_t *const con = (console_ring_t *)CONSOLE_BASE;
static void out_putc(char c)
{
    if (c == '\n')
        out_putc('\r');
    u32 h = con->head, t = con->tail;
    if (h - t >= 512u)
        return;                                   // ring full: drop
    con->buf[h & CONSOLE_BUF_MASK] = c;
    con->head = h + 1u;
}
static u32  now(void)        { return timer_now(); }
static void led(u32 v)       { led_write(v); }
static mailbox_t *const mb = (mailbox_t *)MAILBOX_BASE;
#endif

static void out_puts(const char *s)
{
    while (*s)
        out_putc(*s++);
}

static void out_dec(u32 v)
{
    char b[12];
    int i = 0;
    do {
        b[i++] = (char)('0' + (v % 10u));
        v /= 10u;
    } while (v);
    while (i)
        out_putc(b[--i]);
}

static void out_hex(u32 v)
{
    static const char hex[] = "0123456789abcdef";
    for (int i = 28; i >= 0; i -= 4)
        out_putc(hex[(v >> i) & 0xFu]);
}

// ------------------------------------------------------------------
// checking
// ------------------------------------------------------------------
static u32 checks, mismatches, shown;

static void chk(u32 got, u32 want, const char *what, u32 set, u32 a, u32 b)
{
    checks++;
    if (got == want)
        return;
    mismatches++;
    if (shown < 3u) {                             // the console ring is only 512 bytes
        shown++;
        out_puts("BAD ");
        out_puts(what);
        out_puts(" s");
        out_dec(set);
        out_putc(' ');
        out_dec(a);
        out_putc('/');
        out_dec(b);
        out_puts(" got ");
        out_hex(got);
        out_puts(" want ");
        out_hex(want);
        out_putc('\n');
    }
}

// 1 = idle, 0 = still busy after a generous timeout
static int wait_idle(void)
{
    for (u32 i = 0; i < 200000u; i++)
        if (!(REG_RD(SNN_STATUS) & 1u))
            return 1;
    return 0;
}

#define EXP_PER_TICK (SNN_SPKW + 2u * SNN_P)              // fire words, u[P], a[P]

static void check_tick(const u32 *exp, u32 t, u32 set)
{
    const u32 *e = exp + t * EXP_PER_TICK;
    for (u32 n = 0; n < SNN_SPKW; n++)
        chk(REG_RD(SNN_SPK(n)), e[n], "fire", set, t, n);
    for (u32 c = 0; c < SNN_P; c++) {
        chk(REG_RD(SNN_U(c)), e[SNN_SPKW + c], "u", set, t, c);
        chk(REG_RD(SNN_A(c)), e[SNN_SPKW + SNN_P + c], "a", set, t, c);
    }
}

static void run_set(const snn_set_t *s, u32 idx)
{
    u32 before = mismatches;

    // this set's weights (4 packed 8-bit weights per word, word w = row*(P/4) + group)
    // and ALIF parameters, read back
    for (u32 w = 0; w < SNN_WORDS; w++)
        REG_WR(SNN_WGT_BASE + 4u * w, s->weights[w]);
    for (u32 c = 0; c < SNN_P; c++) {
        REG_WR(SNN_NP(c), s->params[3u * c]);
        REG_WR(SNN_BQ(c), s->params[3u * c + 1u]);
        REG_WR(SNN_BIAS(c), s->params[3u * c + 2u]);
    }
    for (u32 c = 0; c < SNN_P; c++) {
        chk(REG_RD(SNN_NP(c)), s->params[3u * c], "np", idx, c, 0);
        chk(REG_RD(SNN_BQ(c)), s->params[3u * c + 1u], "bq", idx, c, 0);
        chk(REG_RD(SNN_BIAS(c)), s->params[3u * c + 2u], "bias", idx, c, 0);
    }

    REG_WR(SNN_CTRL, 1u);                         // CLR: acc, u, a, s, counters, ticks, evt_ovf
    for (int i = 0; i < 4; i++)                   // clearing the state is registered
        (void)REG_RD(SNN_STATUS);

    u32 t = 0;
    for (u32 i = 0; i < s->ncmd; i++) {
        u32 c  = s->cmds[i];
        u32 op = c >> 30;
        if (op == 0u) {                           // EVT
            REG_WR(SNN_EVT, c & 0x00FFFFFFu);
            if (s->polled && !wait_idle())
                chk(1u, 0u, "busy-stuck", idx, i, 0);
        } else if (op == 1u) {                    // TICK
            REG_WR(SNN_CTRL, 2u);
            if (!wait_idle())
                chk(1u, 0u, "busy-stuck", idx, i, 1);
            check_tick(s->expect, t, idx);
            t++;
        } else {                                  // END
            break;
        }
    }
    chk(t, SNN_T, "ticks-run", idx, 0, 0);

    const u32 *fin = s->expect + SNN_T * EXP_PER_TICK;
    for (u32 c = 0; c < SNN_P; c++)
        chk(REG_RD(SNN_CNT(c)), fin[c], "cnt", idx, c, 0);
    chk(REG_RD(SNN_TICKS), fin[SNN_P], "ticks", idx, 0, 0);
    chk((REG_RD(SNN_STATUS) >> 1) & 1u, 0u, "evt_ovf", idx, 0, 0);

    out_puts("set ");
    out_putc((char)('A' + idx));
    out_puts(s->polled ? " polled: " : " fast:   ");
    if (mismatches == before) {
        out_puts("ok\n");
    } else {
        out_puts("FAIL (");
        out_dec(mismatches - before);
        out_puts(")\n");
    }
}

// ------------------------------------------------------------------
// mailbox: the PS can ask for the verdict with `spike_pynq.py cmd status`
// ------------------------------------------------------------------
static u32 verdict;

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
    switch (mb->cmd[0]) {
    case MB_CMD_NOP:
        mb_reply(0, 0, 0, 0);
        break;
    case MB_CMD_ECHO:
        mb_reply(mb->cmd[1], mb->cmd[2], mb->cmd[3], 0);
        break;
    case MB_CMD_STATUS:
        mb_reply(now(), checks, verdict, mismatches);
        break;
    default:
        mb_reply(0xDEADBEEFu, 0, 0, 0);
        break;
    }
}

// ------------------------------------------------------------------
#ifdef SNN_HOST_MOCK
void snn_main(void)
#else
void main(void)
#endif
{
#ifndef SNN_HOST_MOCK
    con->head = 0;                                // the PS zeroed the rings before starting us
    con->tail = 0;
#endif
    mb->seq_out = 0;
    verdict = 0;
    led(0x001u);

    out_puts("SNN smoke P=");
    out_dec(SNN_P);
    out_puts(" N=");
    out_dec(SNN_NIN);
    out_puts(" T=");
    out_dec(SNN_T);
    out_puts(" ALIF\n");

    // state right after reset (the CPU reset also resets snn_layer)
    chk(REG_RD(SNN_ID), 0x414C4946u, "id", 0, 0, 0);
    chk(REG_RD(SNN_CONFIG), (u32)SNN_P | ((u32)SNN_XBITS << 8) | ((u32)SNN_NIN << 16), "config", 0, 0, 0);
    chk(REG_RD(SNN_NP(0)), 0x00330400u, "np@rst", 0, 0, 0);
    chk(REG_RD(SNN_TICKS), 0u, "ticks@rst", 0, 0, 0);
    chk(REG_RD(SNN_STATUS) & 3u, 0u, "stat@rst", 0, 0, 0);

    for (u32 i = 0; i < SNN_NSETS; i++)
        run_set(&snn_sets[i], i);

    verdict = (mismatches == 0u) ? VERDICT_PASS : VERDICT_FAIL;
    out_puts(mismatches == 0u ? "RESULT: PASS (" : "RESULT: FAIL (");
    out_dec(checks);
    out_puts(" checks, ");
    out_dec(mismatches);
    out_puts(" bad)\n");
    led(mismatches == 0u ? 0x00Fu : 0x3F0u);

#ifdef SNN_HOST_MOCK
    return;
#else
    for (;;)
        mb_service();
#endif
}

#ifdef SNN_HOST_MOCK
// what the PS would do with `cmd status`: one mailbox round trip
void snn_mock_status(u32 *resp)
{
    mb->cmd[0] = MB_CMD_STATUS;
    mb->seq_in = mb->seq_out + 1u;
    mb_service();
    for (int i = 0; i < 4; i++)
        resp[i] = mb->resp[i];
}
#endif
