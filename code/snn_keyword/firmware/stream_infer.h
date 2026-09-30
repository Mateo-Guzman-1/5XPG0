#ifndef STREAM_INFERENCE_H
#define STREAM_INFERENCE_H
#include <stdint.h>
#include "model_stream_config.h"

/* Streaming SNN state: persists between frames and between requests. */
typedef struct {
    int32_t u1[STREAM_N1], a1[STREAM_N1];
    uint8_t s1[STREAM_N1];
    uint8_t spk[32][STREAM_N1];    /* delay ring: indices of layer-1 neurons that spiked, per frame */
    uint8_t spk_n[32];             /* number of entries in each frame's list */
    uint32_t pos;                  /* ring slot of the current frame */
    int32_t u2[STREAM_N2], a2[STREAM_N2];
    uint8_t s2[STREAM_N2];
    int32_t o[STREAM_CLASSES];
} stream_state_t;

/* Once at boot: loads the model into the neuron engine (USE_ENGINE builds; else nothing). */
void stream_init(void);
void stream_reset(stream_state_t *st);
/* One 10 ms frame of STREAM_BANDS uint8 features; returns the decision score
 * o[keyword] - max(o[other]) (detect when >= STREAM_THRESHOLD). spikes[0..1]:
 * layer-1 and layer-2 spikes of this frame; events: synaptic additions done. */
int32_t stream_step(stream_state_t *st, const uint8_t *frame, uint32_t spikes[2], uint32_t *events);
#endif
