/* Native harness for firmware/verifier.c (verifier_export.py).
 * stdin:  records of uint32 n_frames, then n_frames x 24 uint8.
 * stdout: per record the int32 logits (steps x classes), then score_a, end_a, score_b, end_b, head, end_head. */
#include <stdio.h>
#include <stdint.h>
#include "verifier.h"

static uint8_t frames[4096 * 24] __attribute__((aligned(4)));
static int32_t logits[2048 * VERIFIER_CLASSES];

int main(void) {
    uint32_t n;
    while (fread(&n, sizeof n, 1, stdin) == 1) {
        if (n > 4096 || fread(frames, 24, n, stdin) != n) return 2;
        verifier_result_t r;
        verifier_run(frames, n, &r, logits);
        const unsigned steps = n / VERIFIER_STACK;
        if (fwrite(logits, sizeof(int32_t), steps * VERIFIER_CLASSES, stdout) != steps * VERIFIER_CLASSES) return 3;
        int32_t rec[6] = {r.score_a, r.end_a, r.score_b, r.end_b, r.head, r.end_head};
        if (fwrite(rec, sizeof rec, 1, stdout) != 1) return 3;
    }
    return ferror(stdin) ? 4 : 0;
}
