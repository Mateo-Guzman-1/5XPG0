/* Cascade firmware (research R3, research/PLAN_R3_GATING.md): window SNN stage 1 (inference.c)
 * + GRU keyword verifier stage 2 (verifier.c, from the streaming design, unchanged).
 * Build: research/tools/r3_build_cascade.py (-DUSE_KSDOT -DUSE_LIF_TABLE -DUSE_KDOT).
 *
 * MB (words at 0x10400):
 *   0 seq_in (the host writes it last)   1 seq_out   2 opcode   3 length (bytes)
 *   6 status   7..8 stage-1 scores   9 cycles   10 spikes   11 detected
 *  12 magic "KWC1"   13 T1 (CASCADE_T1)   14 window bytes   15 frame bytes   16 verifier frames
 *  17 max frames per request   18 frames since reset   20 stage-1 cycles   21 verifier cycles
 *  22 frame-push cycles   25 T2 (VERIFIER_THRESHOLD_A, on the keyword head)
 *  26..29 path scores (score_a, end_a, score_b, end_b)   30 logits hash (command 6)   31 1 (cascade build)
 *  32..33 keyword head, its step   34..35 POWER counters (clocks asleep, sleeps), only with ABI bit 2
 * Commands (status 0 = OK, 1 = bad command or length):
 *   1 = one window (length MODEL_INPUT), as main.c: MB[7..11], detected = margin >= MODEL_DECISION_THRESHOLD;
 *   2 = status;
 *   5 = reset: frame buffer zeroed, frames = 0, no previous reach, no previous detection;
 *   6 = verifier only: length VERIFIER_FRAMES x 24 verifies the frames in the input buffer, length 0
 *       the frame buffer; MB[21] (and MB[9]) its cycles, MB[26..33] its outputs, MB[30] = sum of
 *       logit x (index + 1) over all steps and classes (32-bit wrap-around);
 *   7 = cascade request: payload = [window (MODEL_INPUT bytes)] + k frames of 24 bytes, 1 <= k <= 25
 *       (length 24 k: no window, stage 1 not run). Steps: push the frames into the buffer; stage 1 on the
 *       window, reach = margin >= T1; if reach in this request or the previous one, the verifier scores the
 *       last VERIFIER_FRAMES frames; detection = head >= T2 at least HOLDOFF frames after the previous one.
 *       MB[11]: bit0 detection (LED0 for 1 s), bit1 reach, bit2 the verifier ran. MB[26..29], MB[32..33]:
 *       the verifier outputs (VERIFIER_NEG / -1 when it did not run). MB[9] = MB[20] + MB[21] + MB[22] + timer reads.
 * Between requests the core sleeps: a store of the wake mask to POWER (0x10005000) stalls until the PS
 * writes MB[0] (doorbell); on a fabric without ABI bit 2 the store completes at once (busy-polling).
 * The host owns the request words and the input until seq_out acknowledges the request.
 */
#include <stdint.h>
#include "inference.h"
#include "model_config.h"
#include "verifier.h"

#define MMIO(a) (*(volatile uint32_t *)(a))
#define VERSION MMIO(0x10000004u)
#define TIMER MMIO(0x10001000u)
#define LED MMIO(0x10002000u)
#define LED_DURATION MMIO(0x10002004u)
#define POWER_SLEEP MMIO(0x10005000u)
#define POWER_ASLEEP MMIO(0x10005004u)
#define POWER_SLEEPS MMIO(0x10005008u)
#define ABI_SLEEP (1u << 2)
#define WAKE_MAILBOX 1u
#define CLK_HZ 100000000u
#define MB ((volatile uint32_t *)0x10400u)
#define INPUT ((const uint8_t *)0x10800u)
#define MAGIC 0x4b574331u
#define FRAME_BYTES 24u
#define FRAME_WORDS (FRAME_BYTES / 4)
#define MAX_FRAMES 25u
#define HOLDOFF 100u

#ifndef CASCADE_T1
#error "needs CASCADE_T1 (verifier_export.export(cascade_t1=...))"
#endif
#ifndef VERIFIER_HEAD
#error "the cascade decides on the keyword head (VERIFIER_HEAD)"
#endif
_Static_assert(VERIFIER_IN == FRAME_BYTES * VERIFIER_STACK, "frame size");
_Static_assert(MAX_FRAMES * FRAME_BYTES < MODEL_INPUT, "a request with a window is longer than one without");

#define VSTEPS (VERIFIER_FRAMES / VERIFIER_STACK)
/* Each frame is stored twice (pos and pos + VERIFIER_FRAMES), so the last frames are always contiguous. */
static uint32_t vhist[2 * VERIFIER_FRAMES * FRAME_WORDS];
static uint32_t vpos;
static int32_t vlogits[VSTEPS * VERIFIER_CLASSES];

static void vhist_reset(void)
{
    for (unsigned i = 0; i < 2 * VERIFIER_FRAMES * FRAME_WORDS; ++i) vhist[i] = 0;
    vpos = 0;
}

static void vhist_push(const uint32_t *frame)
{
    uint32_t *a = vhist + vpos * FRAME_WORDS, *b = a + VERIFIER_FRAMES * FRAME_WORDS;
    for (unsigned i = 0; i < FRAME_WORDS; ++i) a[i] = b[i] = frame[i];
    if (++vpos == VERIFIER_FRAMES) vpos = 0;
}

static const uint8_t *vhist_last(void)
{
    return (const uint8_t *)(vhist + vpos * FRAME_WORDS);
}

/* Frames in a command-7 request (0 if the length is invalid); *window = 1 if a window comes first. */
static uint32_t cascade_frames(uint32_t length, uint32_t *window)
{
    *window = length > MAX_FRAMES * FRAME_BYTES;
    uint32_t bytes = *window ? length - MODEL_INPUT : length;
    if (*window && length < MODEL_INPUT) return 0;
    uint32_t k = bytes / FRAME_BYTES;
    return (bytes == k * FRAME_BYTES && k >= 1 && k <= MAX_FRAMES) ? k : 0;
}

void main(void)
{
    const uint32_t abi = VERSION;
    uint32_t frames = 0, last_event = 0, have_event = 0, prev_reach = 0;
    MB[13] = (uint32_t)CASCADE_T1;
    MB[14] = MODEL_INPUT;
    MB[15] = FRAME_BYTES;
    MB[16] = VERIFIER_FRAMES;
    MB[17] = MAX_FRAMES;
    MB[25] = (uint32_t)VERIFIER_THRESHOLD_A;
    MB[31] = 1;
    __asm__ volatile ("fence rw,rw" ::: "memory");
    MB[12] = MAGIC;   /* last: a host that sees the magic can read the words above */
    LED = 0;
    for (;;) {
        uint32_t seq = MB[0];
        if (seq == MB[1]) {
            POWER_SLEEP = WAKE_MAILBOX;   /* stalls until the PS writes MB[0]; then MB[0] is read again */
            continue;
        }
        uint32_t opcode = MB[2], length = MB[3], window, k;
        if (opcode == 7 && (k = cascade_frames(length, &window)) != 0) {
            int32_t scores[2] = {0, 0};
            int32_t v[6] = {VERIFIER_NEG, -1, VERIFIER_NEG, -1, VERIFIER_NEG, -1};
            uint32_t spikes = 0, reach = 0, detected, t_ver = 0;
            const uint32_t *fr = (const uint32_t *)(INPUT + (window ? MODEL_INPUT : 0));
            uint32_t t0 = TIMER;
            for (uint32_t f = 0; f < k; ++f) vhist_push(fr + f * FRAME_WORDS);
            frames += k;
            uint32_t t1 = TIMER;
            if (window) {
                keyword_infer(INPUT, scores, &spikes);
                reach = scores[1] - scores[0] >= CASCADE_T1;
            }
            uint32_t t2 = TIMER;
            detected = reach << 1;
            if (reach || prev_reach) {
                verifier_result_t r;
                verifier_run(vhist_last(), VERIFIER_FRAMES, &r, 0);
                t_ver = TIMER - t2;
                v[0] = r.score_a; v[1] = r.end_a; v[2] = r.score_b; v[3] = r.end_b; v[4] = r.head; v[5] = r.end_head;
                detected |= 4;
                if (r.head >= VERIFIER_THRESHOLD_A && (!have_event || frames - last_event >= HOLDOFF)) {
                    have_event = 1; last_event = frames;
                    detected |= 1;
                }
            }
            uint32_t t3 = TIMER;
            prev_reach = reach;
            MB[6] = 0; MB[7] = (uint32_t)scores[0]; MB[8] = (uint32_t)scores[1]; MB[9] = t3 - t0;
            MB[10] = spikes; MB[11] = detected; MB[18] = frames;
            MB[20] = t2 - t1; MB[21] = t_ver; MB[22] = t1 - t0;
            for (unsigned i = 0; i < 4; ++i) MB[26 + i] = (uint32_t)v[i];
            MB[30] = 0; MB[32] = (uint32_t)v[4]; MB[33] = (uint32_t)v[5];
            if (detected & 1) LED_DURATION = CLK_HZ;
        } else if (opcode == 1 && length == MODEL_INPUT) {
            int32_t scores[2];
            uint32_t spikes;
            uint32_t start = TIMER;
            keyword_infer(INPUT, scores, &spikes);
            uint32_t elapsed = TIMER - start;
            uint32_t detected = scores[1] - scores[0] >= MODEL_DECISION_THRESHOLD;
            MB[6] = 0;
            MB[7] = (uint32_t)scores[0]; MB[8] = (uint32_t)scores[1];
            MB[9] = elapsed; MB[10] = spikes; MB[11] = detected;
        } else if (opcode == 5) {
            vhist_reset();
            frames = 0; last_event = 0; have_event = 0; prev_reach = 0;
            MB[6] = 0; MB[18] = 0;
        } else if (opcode == 6 && (length == 0 || length == VERIFIER_FRAMES * FRAME_BYTES)) {
            verifier_result_t r;
            const uint8_t *win = length ? INPUT : vhist_last();
            uint32_t start = TIMER;
            verifier_run(win, VERIFIER_FRAMES, &r, vlogits);
            uint32_t elapsed = TIMER - start;
            uint32_t hash = 0;
            for (unsigned i = 0; i < VSTEPS * VERIFIER_CLASSES; ++i)
                hash += (uint32_t)vlogits[i] * (i + 1);
            MB[6] = 0; MB[9] = elapsed; MB[21] = elapsed;
            MB[26] = (uint32_t)r.score_a; MB[27] = (uint32_t)r.end_a;
            MB[28] = (uint32_t)r.score_b; MB[29] = (uint32_t)r.end_b; MB[30] = hash;
            MB[32] = (uint32_t)r.head; MB[33] = (uint32_t)r.end_head;
        } else if (opcode == 2) {
            MB[6] = 0;
        } else {
            MB[6] = 1;
            MB[7] = 0; MB[8] = 0; MB[9] = 0; MB[10] = 0; MB[11] = 0;
        }
        if (abi & ABI_SLEEP) { MB[34] = POWER_ASLEEP; MB[35] = POWER_SLEEPS; }
        __asm__ volatile ("fence rw,rw" ::: "memory");
        MB[1] = seq;
    }
}
