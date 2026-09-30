/* Streaming keyword firmware, ABI v3 (IMPLEMENTATION_PLAN.md, Phase 5).
 *
 * MB (words at 0x10400):
 *   0 seq_in   1 seq_out   2 opcode   3 length (bytes)   4-5 reserved
 *   6 status   7 max score in the hop   8 score of the last frame   9 cycles
 *  10 spikes in the hop (layer 1 + layer 2)   11 detected
 *  12 magic "KWS3"   13 threshold   14 frame bytes   15 max frames per request
 *  16 synaptic events in the hop   17 frame of the detection within the hop (or ~0)
 *  18 frames since reset   19 decision window (frames)
 * Commands: 4 = stream frames (length = k * frame bytes, 1 <= k <= max frames;
 *           the network state persists between requests), 5 = reset state,
 *           2 = status. Status 0 = OK, 1 = bad command or length.
 * detected bit0: a detection in this hop (a frame with score >= threshold at
 * least HOLDOFF frames after the previous detection; lights LED0); bit1: some
 * frame of the hop reached the threshold. The hold-off counts frames, not
 * timer ticks, so the decision equals robust_eval.py's (1 s = 100 frames).
 * The detection score is the int64 sum of the last STREAM_WINDOW frame scores
 * (model.decision_scores; a partial sum after a reset); best and last in
 * MB[7..8] stay the raw frame scores.
 * A v2 client reads magic 0x4b575331 ("KWS1") and the window model instead.
 *
 * USE_VERIFIER builds (verifier track): the last VERIFIER_FRAMES frames are kept
 * (zeros after a reset) and command 6 runs the second-stage phoneme verifier
 * (firmware/verifier.c) on them: MB[26..29] = score_a, end_a, score_b, end_b,
 * MB[30] = a hash of the logits (sum of logit * (index + 1)), MB[9] = cycles.
 * Command 6 with length 0 verifies the history; with length = VERIFIER_FRAMES
 * frames it verifies the frames in the input buffer (test vectors).
 *
 * CASCADE builds (verifier_cascade.py, mode 'cascade'): after each command 4,
 * if the decision score reached CASCADE_T1 in this request or the previous one,
 * the verifier scores the last VERIFIER_FRAMES frames. A detection is
 * score_a >= VERIFIER_THRESHOLD_A at least HOLDOFF frames (counted at request
 * ends) after the previous one; it lights LED0. MB[11] bit0 detection, bit1
 * the decision score reached CASCADE_T1 in this request, bit2 the verifier ran;
 * MB[26..29] its scores (VERIFIER_NEG, -1 when it did not run); MB[17] the
 * last frame of the request on a detection; MB[9] cycles of stage 1 and the
 * verifier; MB[13] = CASCADE_T1, MB[25] = VERIFIER_THRESHOLD_A, MB[31] = 1.
 * With a keyword head (VERIFIER_HEAD) the decision is the head score, MB[32]
 * (MB[33] its step); MB[26..29] stay the phoneme path scores. Command 6 also
 * writes MB[32..33].
 */
#include <stdint.h>
#include "stream_infer.h"
#ifdef USE_VERIFIER
#include "verifier.h"
/* Each frame is written twice (pos and pos + N), so the last N frames are always contiguous. */
static uint8_t vhist[2 * VERIFIER_FRAMES * STREAM_BANDS] __attribute__((aligned(4)));
static uint32_t vpos;
static int32_t vlogits[VERIFIER_FRAMES / VERIFIER_STACK * VERIFIER_CLASSES];

static void vhist_reset(void)
{
    for (unsigned i = 0; i < sizeof vhist; ++i) vhist[i] = 0;
    vpos = 0;
}

static void vhist_push(const uint8_t *frame)
{
    for (unsigned i = 0; i < STREAM_BANDS; ++i)
        vhist[vpos * STREAM_BANDS + i] = vhist[(vpos + VERIFIER_FRAMES) * STREAM_BANDS + i] = frame[i];
    if (++vpos == VERIFIER_FRAMES) vpos = 0;
}
#endif
#if defined(CASCADE) && !(defined(USE_VERIFIER) && defined(CASCADE_T1))
#error "CASCADE needs USE_VERIFIER and CASCADE_T1 (verifier_export.export(cascade_t1=...))"
#endif

#define MMIO(a) (*(volatile uint32_t *)(a))
#define TIMER MMIO(0x10001000u)
#define LED MMIO(0x10002000u)
#define LED_DURATION MMIO(0x10002004u)
#define CLK_HZ 100000000u
#define MB ((volatile uint32_t *)0x10400u)
#define INPUT ((const uint8_t *)0x10800u)
#define MAGIC3 0x4b575333u
#define MAX_FRAMES 100u
#define HOLDOFF 100u

#ifndef STREAM_WINDOW
#define STREAM_WINDOW 1
#endif

static stream_state_t state;
static int32_t hist[STREAM_WINDOW];
static uint32_t hpos;
static int64_t dsum;

static void decision_reset(void)
{
    for (unsigned i = 0; i < STREAM_WINDOW; ++i) hist[i] = 0;
    hpos = 0; dsum = 0;
}
#ifdef PROFILE
extern uint32_t stream_prof[5];
#endif

void main(void)
{
    uint32_t frames = 0, last_event = 0, have_event = 0;
#ifdef CASCADE
    uint32_t prev_reach = 0;
#endif
    stream_init();
    stream_reset(&state);
    decision_reset();
#ifdef USE_VERIFIER
    vhist_reset();
#endif
#ifdef CASCADE
    MB[13] = (uint32_t)CASCADE_T1;
    MB[25] = (uint32_t)VERIFIER_THRESHOLD_A;
    MB[31] = 1;
#else
    MB[13] = (uint32_t)STREAM_THRESHOLD;
#endif
    MB[19] = STREAM_WINDOW;
    MB[14] = STREAM_BANDS;
    MB[15] = MAX_FRAMES;
    __asm__ volatile ("fence rw,rw" ::: "memory");
    MB[12] = MAGIC3;   /* last: a host that sees the magic can read the words above */
    LED = 0;
    for (;;) {
        uint32_t seq = MB[0];
        if (seq == MB[1]) continue;
        uint32_t opcode = MB[2], length = MB[3];
        uint32_t k = length / STREAM_BANDS;
        if (opcode == 4 && length == k * STREAM_BANDS && k >= 1 && k <= MAX_FRAMES) {
            int32_t best = INT32_MIN, score = 0;
            uint32_t spikes = 0, events = 0, detected = 0, at = ~0u;
            uint32_t start = TIMER;
#ifdef CASCADE
            int64_t hop_best = INT64_MIN;
#endif
#ifdef PROFILE
            for (unsigned i = 0; i < 5; ++i) stream_prof[i] = 0;
#endif
            for (uint32_t f = 0; f < k; ++f, ++frames) {
                uint32_t sp[2], ev;
                score = stream_step(&state, INPUT + f * STREAM_BANDS, sp, &ev);
#ifdef USE_VERIFIER
                vhist_push(INPUT + f * STREAM_BANDS);
#endif
                spikes += sp[0] + sp[1]; events += ev;
                if (score > best) best = score;
                dsum += (int64_t)score - hist[hpos];
                hist[hpos] = score;
                if (++hpos == STREAM_WINDOW) hpos = 0;
#ifdef CASCADE
                if (dsum > hop_best) hop_best = dsum;
#else
                if (dsum >= STREAM_THRESHOLD) {
                    detected |= 2;
                    if (!have_event || frames - last_event >= HOLDOFF) {
                        have_event = 1; last_event = frames;
                        if (!(detected & 1)) at = f;
                        detected |= 1;
                    }
                }
#endif
            }
#ifdef CASCADE
            {
                const uint32_t reach = hop_best >= (int64_t)CASCADE_T1;
                int32_t v[6] = {VERIFIER_NEG, -1, VERIFIER_NEG, -1, VERIFIER_NEG, -1};
                if (reach) detected |= 2;
                if (reach || prev_reach) {
                    verifier_result_t r;
                    verifier_run(vhist + vpos * STREAM_BANDS, VERIFIER_FRAMES, &r, 0);
                    v[0] = r.score_a; v[1] = r.end_a; v[2] = r.score_b; v[3] = r.end_b;
                    v[4] = r.head; v[5] = r.end_head;
                    detected |= 4;
#ifdef VERIFIER_HEAD
                    const int32_t decision = r.head;
#else
                    const int32_t decision = r.score_a;
#endif
                    if (decision >= VERIFIER_THRESHOLD_A && (!have_event || frames - last_event >= HOLDOFF)) {
                        have_event = 1; last_event = frames;
                        detected |= 1; at = k - 1;
                    }
                }
                prev_reach = reach;
                for (unsigned i = 0; i < 4; ++i) MB[26 + i] = (uint32_t)v[i];
                MB[32] = (uint32_t)v[4]; MB[33] = (uint32_t)v[5];
            }
#endif
            uint32_t elapsed = TIMER - start;
            MB[6] = 0; MB[7] = (uint32_t)best; MB[8] = (uint32_t)score; MB[9] = elapsed;
            MB[10] = spikes; MB[11] = detected; MB[16] = events; MB[17] = at; MB[18] = frames;
#ifdef PROFILE
            for (unsigned i = 0; i < 5; ++i) MB[20 + i] = stream_prof[i];
#endif
            if (detected & 1) LED_DURATION = CLK_HZ;
        } else if (opcode == 5) {
            stream_reset(&state);
            decision_reset();
#ifdef USE_VERIFIER
            vhist_reset();
#endif
            frames = 0; have_event = 0;
#ifdef CASCADE
            prev_reach = 0;
#endif
            MB[6] = 0; MB[18] = 0;
#ifdef USE_VERIFIER
        } else if (opcode == 6 && (length == 0 || length == VERIFIER_FRAMES * STREAM_BANDS)) {
            verifier_result_t r;
            const uint8_t *win = length ? INPUT : vhist + vpos * STREAM_BANDS;
            uint32_t start = TIMER;
            verifier_run(win, VERIFIER_FRAMES, &r, vlogits);
            MB[9] = TIMER - start;
            uint32_t hash = 0;
            for (unsigned i = 0; i < VERIFIER_FRAMES / VERIFIER_STACK * VERIFIER_CLASSES; ++i)
                hash += (uint32_t)vlogits[i] * (i + 1);
            MB[6] = 0; MB[26] = (uint32_t)r.score_a; MB[27] = (uint32_t)r.end_a;
            MB[28] = (uint32_t)r.score_b; MB[29] = (uint32_t)r.end_b; MB[30] = hash;
            MB[32] = (uint32_t)r.head; MB[33] = (uint32_t)r.end_head;
#endif
        } else if (opcode == 2) {
            MB[6] = 0;
        } else {
            MB[6] = 1;
            MB[7] = 0; MB[8] = 0; MB[9] = 0; MB[10] = 0; MB[11] = 0; MB[16] = 0; MB[17] = ~0u;
        }
        __asm__ volatile ("fence rw,rw" ::: "memory");
        MB[1] = seq;
    }
}
