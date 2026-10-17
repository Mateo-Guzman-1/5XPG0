#include "inference.h"
#include "model_data.h"

void keyword_infer(const uint8_t *input, int32_t *scores, uint32_t *spikes)
{
    int32_t drive[MODEL_HIDDEN], membrane[MODEL_HIDDEN] = {0};
    uint16_t counts[MODEL_HIDDEN] = {0};
#if MODEL_ENCODING
    uint16_t phase[MODEL_INPUT] = {0};
    uint8_t events[MODEL_INPUT];
#elif defined(USE_PACKED)
    /* Research S2 (synapse precision): MODEL_W1_LANES input weights of MODEL_W1_BITS bits in each
     * 32-bit word (lane 0 in the low bits), inputs read four per word. The model stores
     * w1 = c * 2^shift, so sum(c * x) * 2^shift equals the int16 loop's sum exactly (quantize()
     * bounds 255 * sum|w1| below 2^31). */
    typedef uint32_t __attribute__((may_alias)) word_t;
    _Static_assert(MODEL_INPUT % 4 == 0 && MODEL_INPUT % MODEL_W1_LANES == 0, "whole words per neuron");
    enum { GROUP = MODEL_W1_LANES > 4 ? MODEL_W1_LANES : 4 };
    const uint32_t *words = model_w1p;
    const word_t *x4 = (const word_t *)input;
    for (unsigned h = 0; h < MODEL_HIDDEN; ++h, words += MODEL_INPUT / MODEL_W1_LANES) {
        int32_t sum = 0;
        for (unsigned g = 0; g < MODEL_INPUT; g += GROUP)
            for (unsigned i = g; i < g + GROUP; ++i) {
                uint32_t w = words[i / MODEL_W1_LANES], x = x4[i / 4];
                int32_t c = (int32_t)(w << (32 - MODEL_W1_BITS * (i % MODEL_W1_LANES + 1))) >> (32 - MODEL_W1_BITS);
                sum += c * (int32_t)((x >> (8 * (i % 4))) & 0xff);
            }
        drive[h] = sum * (1 << MODEL_W1_SHIFT) / 255 + model_b1[h];
    }
#elif defined(USE_SPARSE)
    /* Research S3 (synapse count): MODEL_W1_FANIN kept synapses per neuron, one word each:
     * (int16 Q10 weight << 16) | input index. */
    const uint32_t *syn = model_w1s;
    for (unsigned h = 0; h < MODEL_HIDDEN; ++h) {
        int32_t sum = 0;
        for (unsigned j = 0; j < MODEL_W1_FANIN; ++j, ++syn)
            sum += ((int32_t)*syn >> 16) * input[*syn & 0x3ff];
        drive[h] = sum / 255 + model_b1[h];
    }
#elif defined(USE_KDOT)
    /* Custom-0 PCPI instructions (rtl/kdot_pcpi.v); the core stalls while the
     * coprocessor streams weights and inputs from BRAM. Bit-exact with the loop below. */
    _Static_assert(MODEL_INPUT % 4 == 0, "kdot processes 4 elements per group");
    __asm__ volatile (".insn r 0x0b, 1, 0, x0, %0, x0" :: "r"(MODEL_INPUT));
    for (unsigned h = 0; h < MODEL_HIDDEN; ++h) {
        int32_t sum;
        __asm__ volatile (".insn r 0x0b, 0, 0, %0, %1, %2"
                          : "=r"(sum) : "r"(&model_w1[h * MODEL_INPUT]), "r"(input) : "memory");
        drive[h] = sum / 255 + model_b1[h];
    }
#else
    for (unsigned h = 0; h < MODEL_HIDDEN; ++h) {
        int32_t sum = 0;
        for (unsigned i = 0; i < MODEL_INPUT; ++i)
            sum += (int32_t)model_w1[h * MODEL_INPUT + i] * input[i];
        drive[h] = sum / 255 + model_b1[h];
    }
#endif
#if defined(USE_LIF_TABLE)
    /* Research S4: with a constant drive the spike count of the loop below is a monotone staircase
     * of the drive (model.lif_thresholds); counting the thresholds it reaches is bit-exact. */
    _Static_assert(!MODEL_ENCODING, "the threshold table needs current encoding");
    (void)membrane;
    *spikes = 0;
    for (unsigned h = 0; h < MODEL_HIDDEN; ++h) {
        unsigned n = 0;
        while (n < MODEL_STEPS && drive[h] >= model_lif_threshold[n])
            ++n;
        counts[h] = n;
        *spikes += n;
    }
#else
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
#endif
    for (unsigned c = 0; c < 2; ++c) {
        scores[c] = model_b2[c] * MODEL_STEPS;
        for (unsigned h = 0; h < MODEL_HIDDEN; ++h)
            scores[c] += model_w2[c * MODEL_HIDDEN + h] * counts[h];
    }
}
