// firmware_host_check.c -- run the firmware's classify() natively on the laptop.
//
// Compiles keyword_main.c itself (same KW_VARIANT / KW_MODEL_HEADER defines as
// the board build), reads 128-byte frames from stdin and prints the two output
// spike counts per frame. research/check_firmware_host.py compares them with
// the expected counts in board_vectors.npz before anything goes to the board.
#include <stdio.h>

#define KW_LAYER1_NOW() 0u   // no TIMER peripheral on the laptop
#define main firmware_main
#include "keyword_main.c"
#undef main

int main(void)
{
    u8 frame[KW_INPUTS];
    s32 score[KW_OUTPUTS];
    u32 layer1_cycles;
    while (fread(frame, 1, sizeof frame, stdin) == sizeof frame) {
        classify(frame, score, &layer1_cycles);
        printf("%ld %ld\n", (long)score[0], (long)score[1]);
    }
    return 0;
}
