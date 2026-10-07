// board.h — shared constants for the minimal RISC-V / SNN firmware.
//
// The peripheral bit layouts MUST match the RTL:
//   rtl/spike_soc.v  (address decode, LED width)
//   rtl/poisson.v    (Poisson register map, Q32 rates)
//
// You are expected to edit this file as your design evolves (add neurons,
// layers, encodings, ...). Keep the addresses in sync with the RTL.

#ifndef BOARD_H
#define BOARD_H

#include <stdint.h>

typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t  u8;
typedef int32_t  s32;

// ------------------------------------------------------------------
// Build constants
// ------------------------------------------------------------------
#define CLK_HZ        100000000u    // FCLK0 (also in ps_if.v / poisson.v)

// ------------------------------------------------------------------
// Peripheral registers (RISC-V memory map) — see rtl/spike_soc.v
// ------------------------------------------------------------------
#define SYS_BASE     0x10000000u
#define SYS_MAGIC    (SYS_BASE + 0x00)   // RO  "SKEL"
#define SYS_VERSION  (SYS_BASE + 0x04)   // RO
#define SYS_SCRATCH  (SYS_BASE + 0x08)   // RW
#define SYS_TRAP     (SYS_BASE + 0x0C)   // RO

#define TIMER_BASE   0x10001000u
#define TIMER_TIME   (TIMER_BASE + 0x00) // RO  32-bit free-running @ CLK_HZ

#define LED_BASE     0x10002000u
#define LED_OUT      (LED_BASE + 0x00)   // RW  [9:0] board LEDs, active-high

// Hardware Poisson spike generator (rtl/poisson.v)
#define POIS_BASE    0x10003000u
#define POIS_CTRL    (POIS_BASE + 0x00)  // RW  bit0 = enable
#define POIS_CH_EN   (POIS_BASE + 0x04)  // RW  [7:0] channel enable
#define POIS_PENDING (POIS_BASE + 0x08)  // RO  [7:0] spike pending
#define POIS_ACK     (POIS_BASE + 0x0C)  // WO  write 1 to clear pending bit
#define POIS_RATE(c) (POIS_BASE + 0x10 + 4*(c)) // RW  rate in Hz
#define POIS_TOTAL   (POIS_BASE + 0x30)  // RO  total spikes generated

#define NCH          8                   // input channels (match poisson.v NCH)

// ------------------------------------------------------------------
// BRAM region layout (MIRRORED in host/spike_pynq.py — keep in sync!)
//
//   0x00000 .. 0x0FFFF  code + data + bss        (linker.ld, 64 KB max)
//   0x10000             console ring             (1 KB)
//   0x10400             mailbox                  (1 KB)
//   0x11000 ..          spike log ring
//   0x22000             keyword input frame      (256 bytes)
//   0x3D000 .. 0x3FFFF  stack
// ------------------------------------------------------------------
#define CONSOLE_BASE  0x10000u
#define MAILBOX_BASE  0x10400u
#define SPIKE_LOG_BASE 0x11000u
#define INPUT_FRAME_BASE 0x22000u
#define INPUT_FRAME_BYTES 256u
#define STACK_TOP     0x40000u

// console ring: head/tail free-running, 512-byte buffer.
#define CONSOLE_BUF_MASK 0x1FFu

// spike log ring header (words at SPIKE_LOG_BASE):
//   [0] head  [1] tail  [2] dropped  [3] format version
// then output-spike words, 32-bit each:
//   [7:0]   output id (neuron index; 0 for the single demo neuron)
//   [31:8]  timestamp (TIMER ticks)
#define SPIKE_LOG_DATA   (SPIKE_LOG_BASE + 16u)
#define SPIKE_LOG_NWORDS 16384u          // 64 KB of events, mask 0x3FFF
#define SPIKE_LOG_MASK   (SPIKE_LOG_NWORDS - 1u)
#define SPIKE_LOG_VERSION 0x5A110001u

// mailbox: PS writes cmd[4], bumps seq_in. CPU handles, writes resp[4],
// bumps seq_out to match seq_in.
#define MB_CMD_NOP          0
#define MB_CMD_ECHO         1   // a0,a1 -> resp0,resp1
#define MB_CMD_STATUS       2   // -> resp0=time, resp1=n_spikes, resp2=v, resp3=err
#define MB_CMD_SET_RATE     3   // a0=channel a1=rate_hz
#define MB_CMD_SET_WEIGHT   4   // a0=channel a1=weight
#define MB_CMD_SET_THRESHOLD 5  // a0=threshold
#define MB_CMD_SET_LEAK     6   // a0=leak
#define MB_CMD_RESET_V      7   // reset membrane potential to 0
#define MB_CMD_CLASSIFY     8   // a0=frame address a1=bytes a2=sequence

typedef volatile struct {
    u32 seq_in;      // PS bumps after writing cmd
    u32 seq_out;     // CPU bumps after writing resp
    u32 cmd[4];
    u32 resp[4];
} mailbox_t;

typedef volatile struct {
    u32 head;
    u32 tail;
    char buf[512];
} console_ring_t;

#endif // BOARD_H
