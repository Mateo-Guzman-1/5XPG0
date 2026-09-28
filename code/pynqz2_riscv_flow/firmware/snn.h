// snn.h — integer forward pass of the keyword SNN (see snn.c).

#ifndef SNN_H
#define SNN_H

#include <stdint.h>

// x: SNN_N_IN uint8 inputs (spectrogram, row-major [mel][time], x/256).
// counts: SNN_N_OUT output spike counts over SNN_T steps.
void snn_run(const uint8_t *x, uint32_t *counts);

#endif // SNN_H
