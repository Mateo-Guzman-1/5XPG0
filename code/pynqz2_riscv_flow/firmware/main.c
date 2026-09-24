// main.c — minimal SNN bring-up demo: ONE leaky integrate-and-fire (LIF)
// neuron running in software on PicoRV32.
//
// The input spikes are produced by the HARDWARE Poisson generator
// (rtl/poisson.v). The firmware:
//   1. enables NCH Poisson channels at different rates,
//   2. polls POIS_PENDING for input spikes,
//   3. integrates each spike into a membrane potential v with a per-channel
//      weight, leaks v once every millisecond, and
//   4. fires an OUTPUT spike when v crosses the threshold.
//
// Output spikes are written to the spike-log ring in BRAM (the PS reads them
// over AXI), and the board LEDs flash on every spike.
//
// THIS IS A SKELETON, NOT A SOLUTION. Natural next steps for the project:
//   * replace the single neuron with a small layer / MLP,
//   * change the input encoding (rates, more channels, real data),
//   * make the leak use the measured time step dt instead of a fixed 1 ms,
//   * use fixed-point weights (Q8/Q16) and study the accuracy/area trade-off.

#include "board.h"
#include "hal.h"

// ------------------------------------------------------------------
// console ring (in BRAM, drained by the PS)
// ------------------------------------------------------------------
static console_ring_t *const con = (console_ring_t *)CONSOLE_BASE;

static void con_putc(char c)
{
    if (c == '\n')
        con_putc('\r');
    u32 h = con->head, t = con->tail;
    if (h - t >= 512)
        return;                        // full: drop
    con->buf[h & CONSOLE_BUF_MASK] = c;
    con->head = h + 1;
}

static void con_puts(const char *s)
{
    while (*s)
        con_putc(*s++);
}

static void con_puthex(u32 v)
{
    static const char hex[] = "0123456789abcdef";
    for (int i = 28; i >= 0; i -= 4)
        con_putc(hex[(v >> i) & 0xF]);
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

// tiny printf: %s %x %d %u %c
static void con_printf(const char *fmt, ...)
{
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    for (; *fmt; fmt++) {
        if (*fmt != '%') {
            con_putc(*fmt);
            continue;
        }
        switch (*++fmt) {
        case 's': con_puts(__builtin_va_arg(ap, const char *)); break;
        case 'x': con_puthex(__builtin_va_arg(ap, u32)); break;
        case 'd': {
            int v = __builtin_va_arg(ap, int);
            if (v < 0) { con_putc('-'); v = -v; }
            con_putdec((u32)v);
            break;
        }
        case 'u': con_putdec(__builtin_va_arg(ap, u32)); break;
        case 'c': con_putc((char)__builtin_va_arg(ap, int)); break;
        default:  con_putc('%'); con_putc(*fmt); break;
        }
    }
    __builtin_va_end(ap);
}

// ------------------------------------------------------------------
// spike log ring (in BRAM, drained by the PS)
// ------------------------------------------------------------------
static volatile u32 *const sph = (volatile u32 *)SPIKE_LOG_BASE;

static void spike_log_init(void)
{
    sph[0] = 0;                      // head
    sph[1] = 0;                      // tail
    sph[2] = 0;                      // dropped
    sph[3] = SPIKE_LOG_VERSION;
}

static void spike_log_push(u32 out_id)
{
    u32 t = sph[1], h = sph[0];
    if (h - t >= SPIKE_LOG_NWORDS) {
        sph[2] = sph[2] + 1;         // ring full: count a drop
        return;
    }
    u32 ts = timer_now() & 0xFFFFFFu;
    ((volatile u32 *)SPIKE_LOG_DATA)[h & SPIKE_LOG_MASK] =
        (ts << 8) | (out_id & 0xFFu);
    sph[0] = h + 1;
}

// ------------------------------------------------------------------
// the demo neuron (LIF, integer arithmetic)
// ------------------------------------------------------------------
static s32 weight[NCH];      // per-channel input weight
static s32 v = 0;            // membrane potential
static s32 leak = 1;         // leak applied once per millisecond
static s32 threshold = 20;   // fire threshold
static u32 n_spikes = 0;
static u32 err_count = 0;

// ------------------------------------------------------------------
// mailbox (commands from the PS)
// ------------------------------------------------------------------
static mailbox_t *const mb = (mailbox_t *)MAILBOX_BASE;

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

    u32 c = mb->cmd[0], a0 = mb->cmd[1], a1 = mb->cmd[2], a2 = mb->cmd[3];

    switch (c) {
    case MB_CMD_NOP:
        mb_reply(0, 0, 0, 0);
        break;
    case MB_CMD_ECHO:
        mb_reply(a0, a1, a2, 0);
        break;
    case MB_CMD_STATUS:
        mb_reply(timer_now(), n_spikes, (u32)v, err_count);
        break;
    case MB_CMD_SET_RATE:
        if (a0 < NCH) {
            wr32(POIS_RATE(a0), a1);
            mb_reply(a0, a1, 0, 0);
        } else {
            mb_reply(0xFFFFFFFFu, 0, 0, 0);
        }
        break;
    case MB_CMD_SET_WEIGHT:
        if (a0 < NCH) {
            weight[a0] = (s32)a1;
            mb_reply(a0, (u32)weight[a0], 0, 0);
        } else {
            mb_reply(0xFFFFFFFFu, 0, 0, 0);
        }
        break;
    case MB_CMD_SET_THRESHOLD:
        threshold = (s32)a0;
        mb_reply((u32)threshold, 0, 0, 0);
        break;
    case MB_CMD_SET_LEAK:
        leak = (s32)a0;
        mb_reply((u32)leak, 0, 0, 0);
        break;
    case MB_CMD_RESET_V:
        v = 0;
        mb_reply(0, 0, 0, 0);
        break;
    default:
        mb_reply(0xDEADBEEFu, 0, 0, 0);
        break;
    }
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------
void main(void)
{
    // rings: the PS zeroed the region before releasing reset; re-init the
    // parts only the CPU owns (never seq_in — a pre-queued command survives)
    con->head = 0;
    con->tail = 0;
    spike_log_init();
    mb->seq_out = 0;

    // default weights + Poisson rates (Hz) for the 8 input channels
    for (int i = 0; i < NCH; i++)
        weight[i] = 3;
    wr32(POIS_RATE(0), 20);
    wr32(POIS_RATE(1), 30);
    wr32(POIS_RATE(2), 40);
    wr32(POIS_RATE(3), 50);
    wr32(POIS_RATE(4), 60);
    wr32(POIS_RATE(5), 70);
    wr32(POIS_RATE(6), 80);
    wr32(POIS_RATE(7), 90);
    wr32(POIS_CH_EN, 0xFFu);
    wr32(POIS_CTRL, 1u);

    led_write(0x001u);   // solid "alive" LED

    con_printf("SKEL rv firmware v1.0 (picorv32 rv32im @ %x Hz)\n", CLK_HZ);
    con_printf("single LIF neuron, %d Poisson input channels\n", NCH);

    u32 leak_next = timer_now() + (CLK_HZ / 1000u);   // 1 kHz leak tick
    u32 led_off_at = 0;
    int led_on = 0;

    for (;;) {
        mb_service();

        // --- input spikes from the hardware Poisson generator ---
        u32 pend = rd32(POIS_PENDING) & ((1u << NCH) - 1u);
        if (pend) {
            wr32(POIS_ACK, pend);            // clear the handled channels
            for (int c = 0; c < NCH; c++)
                if (pend & (1u << c))
                    v += weight[c];
        }

        // --- leak once per millisecond ---
        if ((s32)(timer_now() - leak_next) >= 0) {
            leak_next += (CLK_HZ / 1000u);
            v -= leak;
            if (v < 0)
                v = 0;
        }

        // --- fire when the membrane crosses threshold ---
        if (v >= threshold) {
            v = 0;
            n_spikes++;
            spike_log_push(0);              // output id 0
            led_write(0x3FFu);              // flash all LEDs
            led_off_at = timer_now() + (CLK_HZ / 10u);   // for 100 ms
            led_on = 1;
        }

        // --- LED off after the flash ---
        if (led_on && (s32)(timer_now() - led_off_at) >= 0) {
            led_write(0x001u);
            led_on = 0;
        }
    }
}
