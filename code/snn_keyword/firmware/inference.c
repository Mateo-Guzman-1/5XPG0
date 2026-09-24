#include "inference.h"
#include "model_data.h"

void keyword_infer(const uint8_t *input, int32_t *scores, uint32_t *spikes)
{
    int32_t drive[MODEL_HIDDEN], membrane[MODEL_HIDDEN] = {0};
    uint16_t counts[MODEL_HIDDEN] = {0};
#if MODEL_ENCODING
    uint16_t phase[MODEL_INPUT] = {0};
    uint8_t events[MODEL_INPUT];
#else
    for (unsigned h = 0; h < MODEL_HIDDEN; ++h) {
        int32_t sum = 0;
        for (unsigned i = 0; i < MODEL_INPUT; ++i)
            sum += (int32_t)model_w1[h * MODEL_INPUT + i] * input[i];
        drive[h] = sum / 255 + model_b1[h];
    }
#endif
    *spikes = 0;
    for (unsigned t = 0; t < MODEL_STEPS; ++t) {
#if MODEL_ENCODING
        for (unsigned i = 0; i < MODEL_INPUT; ++i) {
            phase[i] += input[i];
            events[i] = phase[i] >= 255;
            if (events[i]) phase[i] -= 255;
        }
        for (unsigned h = 0; h < MODEL_HIDDEN; ++h) {
            int32_t sum = model_b1[h];
            for (unsigned i = 0; i < MODEL_INPUT; ++i)
                if (events[i]) sum += model_w1[h * MODEL_INPUT + i];
            drive[h] = sum;
        }
#endif
        for (unsigned h = 0; h < MODEL_HIDDEN; ++h) {
            membrane[h] = (membrane[h] * 7) / 8 + drive[h];
            if (membrane[h] >= 1024) {
                membrane[h] -= 1024;
                ++counts[h];
                ++*spikes;
            }
        }
    }
    for (unsigned c = 0; c < 2; ++c) {
        scores[c] = model_b2[c] * MODEL_STEPS;
        for (unsigned h = 0; h < MODEL_HIDDEN; ++h)
            scores[c] += model_w2[c * MODEL_HIDDEN + h] * counts[h];
    }
}
