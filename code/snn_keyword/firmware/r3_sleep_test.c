/* R3 sleep self-test firmware (research/PLAN_R3_GATING.md 2C, sleep checks); built by r3_build_cascade.py --sleep-test.
 * At start: a SLEEP store with wake mask 0 must complete at once (MB[40] = its cycles including two timer reads,
 * MB[41] = POWER sleep count after it, MB[42] = VERSION). Then the cascade firmware's idle loop: sleep with the
 * mailbox mask until the PS writes MB[0], answer every request with status 0 and the POWER counters in MB[34..35].
 */
#include <stdint.h>

#define MMIO(a) (*(volatile uint32_t *)(a))
#define VERSION MMIO(0x10000004u)
#define TIMER MMIO(0x10001000u)
#define POWER_SLEEP MMIO(0x10005000u)
#define POWER_ASLEEP MMIO(0x10005004u)
#define POWER_SLEEPS MMIO(0x10005008u)
#define MB ((volatile uint32_t *)0x10400u)
#define MAGIC 0x4b575354u   /* "KWST" */

void main(void)
{
    uint32_t t0 = TIMER;
    POWER_SLEEP = 0;
    uint32_t t1 = TIMER;
    MB[40] = t1 - t0;
    MB[41] = POWER_SLEEPS;
    MB[42] = VERSION;
    __asm__ volatile ("fence rw,rw" ::: "memory");
    MB[12] = MAGIC;
    for (;;) {
        uint32_t seq = MB[0];
        if (seq == MB[1]) {
            POWER_SLEEP = 1;
            continue;
        }
        MB[6] = 0;
        MB[34] = POWER_ASLEEP; MB[35] = POWER_SLEEPS;
        __asm__ volatile ("fence rw,rw" ::: "memory");
        MB[1] = seq;
    }
}
