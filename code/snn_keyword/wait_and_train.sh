#!/usr/bin/env bash
# Waits until a main run has finished, then starts verifier training; retries while
# train_verifier.py exits 3 (not enough free RAM/GPU at start, or memory guard).
M=/c/Users/matut/FULL_AI/5XPG0/code/snn_keyword/runs_stream
PY=/c/Users/matut/FULL_AI/5XPG0/.venv/Scripts/python.exe
until [ -f $M/s2_wide_mine3/training.json ] || [ -f $M/s1_logw/training.json ]; do sleep 60; done
echo "main run finished: $(date)"
for i in $(seq 1 60); do
  $PY -u train_verifier.py "$@"; rc=$?
  echo "train_verifier exit $rc at $(date)"
  [ $rc -ne 3 ] && exit $rc
  sleep 300
done
