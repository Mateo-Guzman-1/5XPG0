/* Binary harness for the production streaming kernel (firmware/stream_infer.c).
 * stdin:  records of uint32 n_frames, then n_frames x STREAM_BANDS uint8; state resets per record.
 * stdout: per frame int32 score, uint32 layer-1 spikes, layer-2 spikes, synaptic events. */
#include <stdio.h>
#include <stdint.h>
#include "stream_infer.h"

static stream_state_t state;

int main(void) {
    uint32_t n;
    uint8_t frame[STREAM_BANDS];
    while (fread(&n, sizeof n, 1, stdin) == 1) {
        stream_reset(&state);
        for (uint32_t t = 0; t < n; ++t) {
            if (fread(frame, 1, sizeof frame, stdin) != sizeof frame) return 2;
            uint32_t spikes[2], events;
            int32_t rec[4];
            rec[0] = stream_step(&state, frame, spikes, &events);
            rec[1] = (int32_t)spikes[0]; rec[2] = (int32_t)spikes[1]; rec[3] = (int32_t)events;
            if (fwrite(rec, sizeof rec, 1, stdout) != 1) return 3;
        }
    }
    return ferror(stdin) ? 4 : 0;
}
