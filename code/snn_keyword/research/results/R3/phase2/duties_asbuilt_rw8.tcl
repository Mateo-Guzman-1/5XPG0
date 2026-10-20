# R3 power states (r3_verify.py duties): name cpu_clock_duty pb_en pb_we pa_en pa_we (per-block probabilities)
# asbuilt_rw8: width-sliced 32K x 1 cascade pairs: synthesis ties EN to 1 (byte writes), opt_design gates the lower blocks by the address MSB, upper blocks always on; asleep the frozen addresses keep all on
set states {
  busy 1 1 0.00248902658 1 0
  idle 1e-09 1 0 1 0
  day_B2 0.114097105 1 0.000298478086 1 1.38e-05
  day_B1 0.180888233 1 0.000463405519 1 1.38e-05
}
