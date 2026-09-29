#ifndef VERIFIER_H
#define VERIFIER_H
/* Second-stage keyword verifier: causal CTC phoneme GRU + keyword score (verifier_model.py).
 * Bit-exact with verifier_model.integer_forward + keyword_score. */
#include <stdint.h>
#include "verifier_config.h"

typedef struct {
    int32_t score_a;     /* policy (a): best Y+ b* EH+ b* S+ path, <= 0 (VERIFIER_NEG if none) */
    int32_t score_b;     /* policy (b): the same path followed by VERIFIER_BOUNDARY steps without a new phoneme */
    int32_t end_a;       /* step (20 ms) where the best (a) path ends, -1 if none */
    int32_t end_b;
} verifier_result_t;

/* frames: n x 24 uint8, oldest first, contiguous; word aligned. n is a multiple of VERIFIER_STACK.
 * logits (optional, may be 0): n / VERIFIER_STACK x VERIFIER_CLASSES int32, for the bit-exact tests. */
void verifier_run(const uint8_t *frames, unsigned n, verifier_result_t *res, int32_t *logits);
#endif
