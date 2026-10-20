# R3 power states (r3_verify.py duties): name cpu_clock_duty pb_en pb_we pa_en pa_we
# READ_WAIT 1, POWER_OPT 1; measured in xsim (results/R3/phase2/activity.json)
set states {
  busy 1 1 0.0225476267 1 0
  idle 1e-09 0 0 0 0
  idle_gate_only 1e-09 1 0 1 0
  stage1 1 0.320638787 0.0225476267 0 0
  verifier 1 0.430240549 0.00488850019 0 0
  day_B2 0.0576237854 0.0246878946 0.000298481782 1.56162379e-05 1.38e-05
  day_B1 0.0913629956 0.0392038709 0.000463415917 1.595363e-05 1.38e-05
}
