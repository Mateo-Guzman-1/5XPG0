// main.c — keyword spotting ("sheila") with a 2-layer SNN on PicoRV32.
//
// Data path:
//   PC mic -> 16x16 spectrogram -> ZeroMQ -> board (host/keyword_bridge.py)
//   -> frame buffer in BRAM (FRAME_BASE) -> this firmware -> LEDs
//
// The firmware:
//   1. polls the frame buffer for a new spectrogram frame,
//   2. runs the integer SNN forward pass (snn.c, weights in weights.h),
//   3. writes the output spike counts back for the PS,
//   4. when the keyword wins KW_CONSECUTIVE frames in a row, lights all
//      LEDs for 1 s, logs an output spike (id 1) and prints to the console.

#include "board.h"
#include "hal.h"
#include "snn.h"

// frames in a row that must say "keyword" before the LED fires. The PC
// sends a 1 s window every 0.25 s, so one spoken word spans ~3 frames;
// requiring 2 suppresses most single-frame false alarms.
#define KW_CONSECUTIVE 2

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
// keyword-spotter state
// ------------------------------------------------------------------
static frame_buf_t *const fb = (frame_buf_t *)FRAME_BASE;

// Measured on the board: while the PS is writing frame data into the BRAM,
// CPU reads of seq_in can come back as 0 (even twice in a row), which made
// the CPU classify frames twice (12 passes for 8 frames) and fire the LED
// one frame early. Two defenses: read seq_in until two reads agree, and
// only accept seq_done + 1 as a new frame (the PS always bumps by exactly
// one), so a bogus value is simply ignored. The frame data itself is safe:
// the PS writes it before seq_in and never while the CPU is classifying.
static u32 stable_rd(volatile u32 *p)
{
    u32 a, b;
    do {
        a = *p;
        b = *p;
    } while (a != b);
    return a;
}

static u32 n_frames = 0;       // frames classified
static u32 n_detect = 0;       // LED triggers
static u32 last_cycles = 0;    // forward-pass time of the last frame

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
    (void)a0; (void)a1; (void)a2;

    switch (c) {
    case MB_CMD_NOP:
        mb_reply(0, 0, 0, 0);
        break;
    case MB_CMD_ECHO:
        mb_reply(a0, a1, a2, 0);
        break;
    case MB_CMD_STATUS:
        mb_reply(timer_now(), n_frames, n_detect, last_cycles);
        break;
    default:            // the Poisson demo-neuron commands are gone
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
    fb->seq_done = stable_rd(&fb->seq_in);   // ignore a stale frame

    led_write(0x001u);   // solid "alive" LED

    con_printf("SKEL rv firmware v2.0 (picorv32 rv32im @ %x Hz)\n", CLK_HZ);
    con_printf("keyword SNN: 256-64-2 LIF, waiting for frames\n");

    static u8 x[FRAME_BYTES];
    u32 led_off_at = 0;
    int led_on = 0;
    int streak = 0;

    for (;;) {
        mb_service();

        // --- new spectrogram frame from the PS ---
        u32 seq = stable_rd(&fb->seq_in);
        if (seq == fb->seq_done + 1u) {     // the PS always bumps by exactly 1
            for (int i = 0; i < FRAME_BYTES; i++)
                x[i] = fb->data[i];

            u32 counts[2];
            u32 t0 = timer_now();
            snn_run(x, counts);
            last_cycles = timer_now() - t0;
            n_frames++;

            int kw = counts[1] > counts[0];
            streak = kw ? streak + 1 : 0;
            int trigger = (streak == KW_CONSECUTIVE);   // once per utterance

            if (trigger) {
                n_detect++;
                spike_log_push(1);                      // output id 1 = keyword
                led_write(0x3FFu);                      // all LEDs on ...
                led_off_at = timer_now() + CLK_HZ;      // ... for 1 s
                led_on = 1;
                con_printf("sheila! (%u vs %u spikes, %u cycles)\n",
                           counts[1], counts[0], last_cycles);
            }

            fb->result = (counts[0] & 0xFFu) | ((counts[1] & 0xFFu) << 8) |
                         ((u32)kw << 16) | ((u32)trigger << 17);
            fb->cycles = last_cycles;
            fb->seq_done = seq;
        }

        // --- LED off after 1 s ---
        if (led_on && (s32)(timer_now() - led_off_at) >= 0) {
            led_write(0x001u);
            led_on = 0;
        }
    }
}
