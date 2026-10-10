#!/usr/bin/env bash
# Temporary probe (D-623): how often does life_sleep fail after life_one_day on Windows, and why?
set -u
export PYTHONPATH="$PWD/ci"
L=tests/integration/life
OPTS=(-q -p no:cacheprovider -m "not live" -rf --log-level=INFO)
case "$PROBE_INDEX" in
  0|1)  echo "== $PROBE_INDEX: life_one_day + life_sleep, real hidden window, 4 times"
        for i in 1 2 3 4; do
          echo "-- run $i"
          uv run python -m pytest "${OPTS[@]}" $L/test_life_one_day.py $L/test_life_sleep.py 2>&1 | tail -n 400 | grep -a -v "^$" | tail -n 120
        done ;;
  2|3)  echo "== $PROBE_INDEX: life_one_day + life_sleep, no hidden window, 4 times"
        for i in 1 2 3 4; do
          echo "-- run $i"
          uv run python -m pytest "${OPTS[@]}" -p probe_nowindow $L/test_life_one_day.py $L/test_life_sleep.py 2>&1 | tail -n 400 | grep -a -v "^$" | tail -n 120
        done ;;
esac
