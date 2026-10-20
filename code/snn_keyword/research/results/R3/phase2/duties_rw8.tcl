# R3 power states (r3_verify.py duties): name cpu_clock_duty pb_en pb_we pa_en pa_we
# READ_WAIT 8, POWER_OPT 1; measured in xsim (results/R3/phase2/activity.json)
set states {
  busy 1 1 0.0112540674 1 0
  idle 1e-09 0 0 0 0
  idle_gate_only 1e-09 1 0 1 0
  stage1 1 0.160038595 0.0112540674 0 0
  verifier 1 0.217325069 0.00246930152 0 0
  day_B2 0.114097105 0.0246870113 0.000298478086 1.61809711e-05 1.38e-05
  day_B1 0.180888233 0.0392023977 0.000463405519 1.68488823e-05 1.38e-05
}
