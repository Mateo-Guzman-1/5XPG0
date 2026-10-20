/* Cascade firmware (research R3, research/PLAN_R3_GATING.md): window SNN stage 1 (inference.c)
 * + GRU keyword verifier stage 2 (verifier.c, from the streaming design, unchanged).
 * Build: research/tools/r3_build_cascade.py (-DUSE_KSDOT -DUSE_LIF_TABLE -DUSE_KDOT).
 *
 * MB (words at 0x10400):
 *   0 seq_in (the host writes it last)   1 seq_out   2 opcode   3 length (bytes)
 *   6 status   7..8 stage-1 scores   9 cycles   10 spikes   11 detected
 *  12 magic "KWC1"   13 T1 (CASCADE_T1)   14 window bytes   15 frame bytes   16 verifier frames
 *  21 verifier cycles   25 T2 (VERIFIER_THRESHOLD_A, on the keyword head)
 *  26..29 path scores (score_a, end_a, score_b, end_b)   30 logits hash   31 1 (cascade build)
 *  32..33 keyword head, its step
 * Commands (status 0 = OK, 1 = bad command or length):
 *   1 = one window (length MODEL_INPUT), as main.c: MB[7..11], detected = margin >= MODEL_DECISION_THRESHOLD;
 *   2 = status;
 *   6 = verifier only: length VERIFIER_FRAMES x 24 verifies the frames in the input buffer, length 0
 *       the frame buffer (zeros after start-up); MB[21] (and MB[9]) its cycles, MB[26..33] its outputs,
 *       MB[30] = sum of logit x (index + 1) over all steps and classes (32-bit wrap-around).
 * The host owns the request words and the input until seq_out acknowledges the request.
 */
#include <stdint.h>
#include "inference.h"
#include "model_config.h"
#include "verifier.h"

#define MMIO(a) (*(volatile uint32_t *)(a))
#define TIMER MMIO(0x10001000u)
#define LED MMIO(0x10002000u)
#define LED_DURATION MMIO(0x10002004u)
#define CLK_HZ 100000000u
#define MB ((volatile uint32_t *)0x10400u)
#define INPUT ((const uint8_t *)0x10800u)
#define MAGIC 0x4b574331u
#define FRAME_BYTES 24u

#ifndef CASCADE_T1
#error "needs CASCADE_T1 (verifier_export.export(cascade_t1=...))"
#endif
#ifndef VERIFIER_HEAD
#error "the cascade decides on the keyword head (VERIFIER_HEAD)"
#endif
_Static_assert(VERIFIER_IN == FRAME_BYTES * VERIFIER_STACK, "frame size");

#define VSTEPS (VERIFIER_FRAMES / VERIFIER_STACK)
/* Each frame is stored twice (pos and pos + VERIFIER_FRAMES), so the last frames are always contiguous. */
static uint8_t vhist[2 * VERIFIER_FRAMES * FRAME_BYTES] __attribute__((aligned(4)));
static uint32_t vpos;
static int32_t vlogits[VSTEPS * VERIFIER_CLASSES];

void main(void)
{
    MB[13] = (uint32_t)CASCADE_T1;
    MB[14] = MODEL_INPUT;
    MB[15] = FRAME_BYTES;
    MB[16] = VERIFIER_FRAMES;
    MB[25] = (uint32_t)VERIFIER_THRESHOLD_A;
    MB[31] = 1;
    __asm__ volatile ("fence rw,rw" ::: "memory");
    MB[12] = MAGIC;   /* last: a host that sees the magic can read the words above */
    LED = 0;
    for (;;) {
        uint32_t seq = MB[0];
        if (seq == MB[1]) continue;
        uint32_t opcode = MB[2], length = MB[3];
        if (opcode == 1 && length == MODEL_INPUT) {
            int32_t scores[2];
            uint32_t spikes;
            uint32_t start = TIMER;
            keyword_infer(INPUT, scores, &spikes);
            uint32_t elapsed = TIMER - start;
            uint32_t detected = scores[1] - scores[0] >= MODEL_DECISION_THRESHOLD;
            MB[6] = 0;
            MB[7] = (uint32_t)scores[0]; MB[8] = (uint32_t)scores[1];
            MB[9] = elapsed; MB[10] = spikes; MB[11] = detected;
        } else if (opcode == 6 && (length == 0 || length == VERIFIER_FRAMES * FRAME_BYTES)) {
            verifier_result_t r;
            const uint8_t *win = length ? INPUT : vhist + vpos * FRAME_BYTES;
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
        __asm__ volatile ("fence rw,rw" ::: "memory");
        MB[1] = seq;
    }
}
