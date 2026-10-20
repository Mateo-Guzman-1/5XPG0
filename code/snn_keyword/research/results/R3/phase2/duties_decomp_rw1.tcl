# R3 power states (r3_verify.py duties): name cpu_clock_duty pb_en pb_we pa_en pa_we (per-block probabilities)
# decomp_rw1: ram_decomp = "power": 64 address-decoded 1K x 36 blocks, EN = 6-bit address decode: one block per port per cycle (also asleep, at the frozen address)
set states {
  busy 1 0.015625 7.69947834e-05 0.015625 0
  idle 1e-09 0.015625 0 0.015625 0
  day_B2 0.0576237854 0.015625 4.66377784e-06 0.015625 2.15625e-07
  day_B1 0.0913629956 0.015625 7.2408737e-06 0.015625 2.15625e-07
}
