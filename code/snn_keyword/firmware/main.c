#include <stdint.h>
#include "inference.h"
#include "model_config.h"

#define MMIO(a) (*(volatile uint32_t *)(a))
#define TIMER MMIO(0x10001000u)
#define LED MMIO(0x10002000u)
#define LED_DURATION MMIO(0x10002004u)
#define CLK_HZ 100000000u
#define MB ((volatile uint32_t *)0x10400u)
#define INPUT ((const uint8_t *)0x10800u)
#define MAGIC 0x4b575331u
/* A stream window confirms only if one of the two previous stream windows
 * was also positive and arrived within this time (hop 250 ms + jitter). */
#define CONFIRM_CYCLES (CLK_HZ / 4 * 3)

/* MB: seq_in, seq_out, opcode, length, reserved[2],
 *     status, score0, score1, cycles, spikes, detected, magic, threshold,
 *     input_bytes, stream_threshold.
 * Command 1 = single inference, 2 = status, 3 = stream-window inference.
 * Status 0=OK, 1=bad command/length. For command 1, detected is the window
 * decision. For command 3, windows use the stream threshold; bit0 = confirmed
 * (this window and one of the two previous stream windows positive,
 * "2 of 3"), bit1 = this window alone; only
 * a confirmed detection lights LED0.
 * Host owns request words and input until seq_out acknowledges the request.
 */
void main(void)
{
    uint32_t hit_time[2] = {0, 0}, hit[2] = {0, 0};   /* previous two stream windows */
    MB[12] = MAGIC;
    MB[13] = (uint32_t)MODEL_DECISION_THRESHOLD;
    MB[14] = MODEL_INPUT;
    MB[15] = (uint32_t)MODEL_STREAM_THRESHOLD;
    LED = 0;
    for (;;) {
        uint32_t seq = MB[0];
        if (seq == MB[1]) continue;
        uint32_t opcode = MB[2], length = MB[3];
        if ((opcode == 1 || opcode == 3) && length == MODEL_INPUT) {
            int32_t scores[2];
            uint32_t spikes;
            uint32_t start = TIMER;
            keyword_infer(INPUT, scores, &spikes);
            uint32_t elapsed = TIMER - start;
            uint32_t detected = scores[1] - scores[0] >= MODEL_DECISION_THRESHOLD;
            if (opcode == 3) {
                uint32_t now = TIMER, window = scores[1] - scores[0] >= MODEL_STREAM_THRESHOLD;
                /* Unsigned subtraction handles timer wrap-around. */
                detected = window && ((hit[0] && now - hit_time[0] < CONFIRM_CYCLES) ||
                                      (hit[1] && now - hit_time[1] < CONFIRM_CYCLES));
                hit[1] = hit[0]; hit_time[1] = hit_time[0];
                hit[0] = window; hit_time[0] = now;
                detected |= window << 1;
            }
            MB[6] = 0;
            MB[7] = (uint32_t)scores[0]; MB[8] = (uint32_t)scores[1];
            MB[9] = elapsed; MB[10] = spikes; MB[11] = detected;
            if (detected & 1) {
                LED_DURATION = CLK_HZ;
            }
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
