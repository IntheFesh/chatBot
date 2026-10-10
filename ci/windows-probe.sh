#!/usr/bin/env bash
# Temporary probe (D-623): why life_sleep fails after life_one_day on Windows.
set -u
export PYTHONPATH="$PWD/ci"
L=tests/integration/life
OPTS=(-q -p no:cacheprovider -m "not live" -rf -o log_cli=true --log-cli-level=INFO)
EVENING="test_a_wake_up_in_the_evening_voids_what_was_planned_and_answers_with_a_delay"
case "$PROBE_INDEX" in
  0)  echo "== 0: life_one_day, then the evening wake-up (INFO log)"
      uv run python -m pytest "${OPTS[@]}" $L/test_life_one_day.py "$L/test_life_sleep.py::$EVENING" ;;
  1)  echo "== 1: the evening wake-up alone (INFO log)"
      uv run python -m pytest "${OPTS[@]}" "$L/test_life_sleep.py::$EVENING" ;;
  2)  echo "== 2: life_one_day, then the evening wake-up, the power monitor without its hidden window"
      uv run python -m pytest "${OPTS[@]}" -p probe_nowindow $L/test_life_one_day.py "$L/test_life_sleep.py::$EVENING" ;;
esac
