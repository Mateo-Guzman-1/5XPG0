// hal.h — tiny I/O helpers: register access, timing, GPIO-like bit ops.

#ifndef HAL_H
#define HAL_H

#include "board.h"

static inline void wr32(u32 addr, u32 v) { *(volatile u32 *)addr = v; }
static inline u32  rd32(u32 addr)        { return *(volatile u32 *)addr; }

static inline u32 timer_now(void) { return rd32(TIMER_TIME); }

// busy-wait for ~n microseconds (may overshoot, never undershoot by much)
static inline void delay_us(u32 us)
{
    u32 t0 = timer_now();
    u32 ticks = us * (CLK_HZ / 1000000u);
    while ((timer_now() - t0) < ticks)
        ;
}

// busy-wait for n timer ticks
static inline void delay_ticks(u32 n)
{
    u32 t0 = timer_now();
    while ((timer_now() - t0) < n)
        ;
}

// board LED helpers (LED_OUT is 10 bits)
static inline void led_write(u32 v) { wr32(LED_OUT, v & 0x3FFu); }

#endif // HAL_H
