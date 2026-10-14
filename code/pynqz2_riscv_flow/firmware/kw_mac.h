// kw_mac.h — C wrappers for the MAC co-processor instructions (rtl/kw_mac_pcpi.v).
//
// The compiler does not know these instructions, so each wrapper emits one
// with the assembler's .insn directive:
//   .insn r opcode, funct3, funct7, rd, rs1, rs2
// opcode 0x0B is RISC-V custom-0; funct7 selects the operation (must match
// the F_* localparams in kw_mac_pcpi.v):
//
//   funct7  wrapper            rs1             rs2            returns
//   0       kw_ldx(pos, x4)    position 0-124  4 input bytes  -
//   1       kw_setacc(bias)    bias            -              -
//   2       kw_mac4(w)         4 int8 weights  -              new sum
//   3       kw_mac8(w)         8 int4 weights  -              new sum
//
// The unit keeps state between instructions (input buffer, sum, position),
// so every wrapper is asm volatile: the compiler may not drop or reorder them.
//
// Built for the laptop (research/firmware_host_check.c) there is no MAC unit;
// the same operations are then done in plain C, which also documents exactly
// what the hardware computes.

#ifndef KW_MAC_H
#define KW_MAC_H

#include "board.h"

// SYSCTRL VERSION of the first bitstream with the MAC unit (rtl/spike_soc.v).
#define KW_MAC_MIN_VERSION 0x00010100u
// resp[1] of a refused MB_CMD_CLASSIFY on an older bitstream ("KMAC").
#define KW_NO_MAC_UNIT 0x4B4D4143u

#if defined(__riscv)

static inline void kw_ldx(u32 pos, u32 x4)
{
    asm volatile (".insn r 0x0B, 0, 0, x0, %0, %1" : : "r"(pos), "r"(x4));
}

static inline void kw_setacc(s32 bias)
{
    asm volatile (".insn r 0x0B, 0, 1, x0, %0, x0" : : "r"(bias));
}

static inline s32 kw_mac4(u32 w)
{
    s32 sum;
    asm volatile (".insn r 0x0B, 0, 2, %0, %1, x0" : "=r"(sum) : "r"(w));
    return sum;
}

static inline s32 kw_mac8(u32 w)
{
    s32 sum;
    asm volatile (".insn r 0x0B, 0, 3, %0, %1, x0" : "=r"(sum) : "r"(w));
    return sum;
}

#else   // laptop model of the unit

static u8  kw_mac_buffer[128];
static s32 kw_mac_sum;
static u32 kw_mac_position;

static inline void kw_ldx(u32 pos, u32 x4)
{
    for (int j = 0; j < 4; j++)
        kw_mac_buffer[(pos + j) & 127] = (u8)(x4 >> (8 * j));
}

static inline void kw_setacc(s32 bias)
{
    kw_mac_sum = bias;
    kw_mac_position = 0;
}

static inline s32 kw_mac4(u32 w)
{
    for (int j = 0; j < 4; j++) {
        s32 weight = (s32)(w << (24 - 8 * j)) >> 24;     // signed lane j
        kw_mac_sum += weight * kw_mac_buffer[(kw_mac_position + j) & 127];
    }
    kw_mac_position = (kw_mac_position + 4) & 127;
    return kw_mac_sum;
}

static inline s32 kw_mac8(u32 w)
{
    for (int j = 0; j < 8; j++) {
        s32 weight = (s32)(w << (28 - 4 * j)) >> 28;     // signed lane j
        kw_mac_sum += weight * kw_mac_buffer[(kw_mac_position + j) & 127];
    }
    kw_mac_position = (kw_mac_position + 8) & 127;
    return kw_mac_sum;
}

#endif

#endif // KW_MAC_H
