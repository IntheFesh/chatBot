#!/usr/bin/env bash
# Temporary probe (D-623): which earlier tests make the hidden power window / the life sleep test fail.
set -u
L=tests/integration/life
OPTS=(-q -p no:cacheprovider -m "not live" --log-cli-level=WARNING -rf)
case "$PROBE_INDEX" in
  0)  echo "== 0: life_sleep alone"
      uv run python -m pytest "${OPTS[@]}" $L/test_life_sleep.py ;;
  1)  echo "== 1: life_faults, then life_sleep"
      uv run python -m pytest "${OPTS[@]}" $L/test_life_faults.py $L/test_life_sleep.py ;;
  2)  echo "== 2: life_one_day, then life_sleep"
      uv run python -m pytest "${OPTS[@]}" $L/test_life_one_day.py $L/test_life_sleep.py ;;
  3)  echo "== 3: life_output_rules, then life_sleep"
      uv run python -m pytest "${OPTS[@]}" $L/test_life_output_rules.py $L/test_life_sleep.py ;;
  4)  echo "== 4: life_restart (kills an application in-process), then life_sleep"
      uv run python -m pytest "${OPTS[@]}" $L/test_life_restart.py $L/test_life_sleep.py ;;
  5)  echo "== 5: life_restart, then the real hidden window test"
      uv run python -m pytest "${OPTS[@]}" $L/test_life_restart.py tests/unit/test_power_events.py ;;
esac
