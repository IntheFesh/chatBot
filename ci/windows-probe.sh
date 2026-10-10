#!/usr/bin/env bash
# Temporary probe (D-623): does life_sleep still fail after life_one_day on Windows, now that a
# cancelled poll still announces its committed batch (D-625)?  Real window, real toasts, as in CI.
set -u
L=tests/integration/life
OPTS=(-q -p no:cacheprovider -m "not live" -rf --tb=short)
for i in 1 2 3 4 5 6; do
  echo "-- job $PROBE_INDEX run $i"
  uv run python -m pytest "${OPTS[@]}" $L/test_life_one_day.py $L/test_life_sleep.py > "out-$i.txt" 2>&1
  tail -n 1 "out-$i.txt"
  grep -a "^FAILED" "out-$i.txt" | head -3
done
